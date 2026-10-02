from pathlib import Path

import numpy as np
import rasterio


class CorridorPredictorNormalization:
    """
    Normalize corridor predictors to ecological suitability scores [0, 1].

    Continuous predictors:
        - elevation: inverse min-max
        - slope: inverse min-max
        - NDVI: direct min-max
        - distance to water: inverse min-max

    Categorical predictor:
        - WorldCover: explicit reclassification table
    """

    NDVI_LOWER_PERCENTILE = 5
    NDVI_UPPER_PERCENTILE = 95

    LANDCOVER_SUITABILITY = {
        10: 0.90,  # Tree cover
        20: 1.00,  # Shrubland
        30: 0.70,  # Grassland
        40: 0.40,  # Cropland
        50: 0.00,  # Built-up
        60: 0.20,  # Bare / sparse vegetation
        70: 0.10,  # Snow / ice
        80: 0.00,  # Permanent water
        90: 0.70,  # Herbaceous wetland
        95: 0.70,  # Mangroves
        100: 0.30,  # Moss / lichen
    }

    def __init__(self, root=None):
        self.root = root or Path(__file__).resolve().parents[2]

        self.input_dir = self.root / "data" / "output_rasters" / "corridor" / "stack"

        self.output_dir = (
            self.root / "data" / "output_rasters" / "corridor" / "normalized"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.files = {
            "elevation": self.input_dir / "elevation_30m.tif",
            "slope": self.input_dir / "slope_30m.tif",
            "landcover": self.input_dir / "landcover_30m.tif",
            "ndvi": self.input_dir / "ndvi_30m.tif",
            "distance_to_water": (self.input_dir / "distance_to_water_30m.tif"),
        }

    def read(self, name):
        path = self.files[name]

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)

            profile = src.profile.copy()

        return data, profile

    @staticmethod
    def minmax(data):
        """
        Direct min-max normalization.

        Lowest value  -> 0
        Highest value -> 1
        """

        values = data.compressed()

        if len(values) == 0:
            raise ValueError("Raster contains no valid pixels.")

        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(minimum, maximum):
            raise ValueError(
                "Raster has no variation; min-max normalization is undefined."
            )

        result = (data.astype("float32") - minimum) / (maximum - minimum)

        return result, minimum, maximum

    @staticmethod
    def inverse_minmax(data):
        """
        Inverse min-max normalization.

        Lowest value  -> 1
        Highest value -> 0
        """

        result, minimum, maximum = CorridorPredictorNormalization.minmax(data)

        return (
            1.0 - result,
            minimum,
            maximum,
        )

    def normalize_percentile(
        self,
        name,
        lower_percentile=5,
        upper_percentile=95,
    ):
        input_path = self.input_dir / f"{name}_30m.tif"
        output_path = self.output_dir / f"{name}_suitability.tif"

        with rasterio.open(input_path) as src:
            data = src.read(1).astype("float32")
            profile = src.profile.copy()
            nodata = src.nodata

            valid = np.isfinite(data)

            if nodata is not None:
                valid &= data != nodata

            values = data[valid]

            if values.size == 0:
                raise ValueError(f"No valid pixels found in {input_path}")

            lower = np.percentile(
                values,
                lower_percentile,
            )

            upper = np.percentile(
                values,
                upper_percentile,
            )

            if upper <= lower:
                raise ValueError(
                    f"Invalid percentile range for {name}: {lower} >= {upper}"
                )

            result = (data - lower) / (upper - lower)

            result = np.clip(
                result,
                0.0,
                1.0,
            )

            if nodata is not None:
                result[~valid] = nodata
            else:
                result[~valid] = np.nan

            profile.update(
                dtype="float32",
                nodata=nodata,
            )

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as dst:
                dst.write(
                    result.astype("float32"),
                    1,
                )

        print(f"\n{name}")
        print(f"  Input min : {values.min():.4f}")
        print(f"  Input max : {values.max():.4f}")
        print(f"  Method    : percentile P{lower_percentile}–P{upper_percentile}")
        print(f"  P{lower_percentile:<2}       : {lower:.4f}")
        print(f"  P{upper_percentile:<2}       : {upper:.4f}")
        print(f"  Output    : {output_path}")

    def normalize_continuous(
        self,
        name,
        inverse=False,
    ):
        data, profile = self.read(name)

        if inverse:
            normalized, minimum, maximum = self.inverse_minmax(data)
        else:
            normalized, minimum, maximum = self.minmax(data)

        print(f"\n{name}")
        print(f"  Input min : {minimum:.4f}")
        print(f"  Input max : {maximum:.4f}")
        print(f"  Direction : {'inverse' if inverse else 'direct'}")

        output_path = self.output_dir / f"{name}_suitability.tif"

        profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
        )

        output = normalized.filled(-9999.0)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output.astype("float32"), 1)

        print(f"  Output    : {output_path}")

        return output_path

    def normalize_landcover(self):
        data, profile = self.read("landcover")

        output = np.full(
            data.shape,
            -9999.0,
            dtype="float32",
        )

        valid = ~data.mask

        classes = np.unique(data.compressed())

        print("\nlandcover")
        print(f"  Classes found : {classes.tolist()}")

        unmapped = []

        for class_value in classes:
            class_int = int(class_value)

            if class_int not in self.LANDCOVER_SUITABILITY:
                unmapped.append(class_int)
                continue

            output[valid & (data.data == class_int)] = self.LANDCOVER_SUITABILITY[
                class_int
            ]

        if unmapped:
            raise ValueError(f"Unmapped WorldCover classes: {unmapped}")

        profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
        )

        output_path = self.output_dir / "landcover_suitability.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output, 1)

        print(f"  Output    : {output_path}")

        return output_path

    def run(self):
        print("=" * 60)
        print("P7 — CORRIDOR PREDICTOR NORMALIZATION")
        print("=" * 60)

        print("\n[1] Elevation")
        self.normalize_continuous(
            "elevation",
            inverse=True,
        )

        print("\n[2] Slope")
        self.normalize_continuous(
            "slope",
            inverse=True,
        )

        print("\n[3] Land cover")
        self.normalize_landcover()

        print("\n[4] NDVI")
        self.normalize_percentile(
            "ndvi",
            lower_percentile=5,
            upper_percentile=95,
        )

        print("\n[5] Distance to water")
        self.normalize_continuous(
            "distance_to_water",
            inverse=True,
        )

        print("\n" + "=" * 60)
        print("P7 normalization completed.")
        print("=" * 60)
