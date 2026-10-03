import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape


class CorridorPredictorNormalization:
    """
    Normalize corridor predictors to ecological suitability scores [0, 1].

    All calculations are restricted to the watershed ROI:

        data/output_vectors/tangse_meureudu_roi.geojson

    Pixels outside the watershed are written as NoData (-9999).

    Continuous predictors:
        - elevation: inverse min-max
        - slope: inverse min-max
        - NDVI: percentile P5-P95, direct
        - distance to water: inverse min-max

    Categorical predictor:
        - WorldCover: explicit reclassification table
    """

    NDVI_LOWER_PERCENTILE = 5
    NDVI_UPPER_PERCENTILE = 95

    NODATA = -9999.0

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

        self.input_dir = self.root / "data" / "output_rasters" / "corridor"

        self.output_dir = (
            self.root / "data" / "output_rasters" / "corridor" / "normalized"
        )

        self.roi_path = (
            self.root / "data" / "output_vectors" / "tangse_meureudu_roi.geojson"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.files = {
            "elevation": (self.input_dir / "corridor_elevation.tif"),
            "slope": (self.input_dir / "corridor_slope.tif"),
            "landcover": (self.input_dir / "corridor_landcover_worldcover_2020.tif"),
            "ndvi": (self.input_dir / "corridor_ndvi.tif"),
            "distance_to_water": (self.input_dir / "corridor_distance_to_water.tif"),
        }

        self._roi_geometries = None

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def load_roi(self):
        """Load watershed ROI geometry from GeoJSON."""

        if not self.roi_path.exists():
            raise FileNotFoundError(f"ROI not found: {self.roi_path}")

        with open(self.roi_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        geometries = [shape(feature["geometry"]) for feature in geojson["features"]]

        if not geometries:
            raise ValueError(f"No geometries found in {self.roi_path}")

        self._roi_geometries = geometries

        print("\nROI")
        print(f"  File       : {self.roi_path}")
        print(f"  Geometries : {len(geometries)}")

    def get_roi_mask(self, src):
        """
        Create a boolean mask matching the raster grid.

        True  = inside watershed ROI
        False = outside watershed ROI
        """

        if self._roi_geometries is None:
            self.load_roi()

        return geometry_mask(
            self._roi_geometries,
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width),
            all_touched=False,
        )

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def read(self, name):
        path = self.files[name]

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()

            roi_mask = self.get_roi_mask(src)

        return data, profile, roi_mask

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def valid_mask(data, roi_mask, nodata=None):
        """
        Valid pixels are:

            1. inside ROI
            2. finite
            3. not equal to raster NoData
        """

        valid = roi_mask & np.isfinite(data)

        if nodata is not None:
            valid &= data != nodata

        return valid

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def minmax(data, valid):
        """
        Direct min-max normalization.

        Lowest value  -> 0
        Highest value -> 1
        """

        values = data[valid]

        if values.size == 0:
            raise ValueError("No valid pixels inside watershed ROI.")

        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(minimum, maximum):
            raise ValueError("Raster has no variation inside watershed ROI.")

        result = (data.astype("float32") - minimum) / (maximum - minimum)

        return result, minimum, maximum

    @staticmethod
    def inverse_minmax(data, valid):
        """
        Inverse min-max normalization.

        Lowest value  -> 1
        Highest value -> 0
        """

        result, minimum, maximum = CorridorPredictorNormalization.minmax(
            data,
            valid,
        )

        return (
            1.0 - result,
            minimum,
            maximum,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_output(
        self,
        output_path,
        output,
        profile,
    ):
        """
        Write normalized raster.

        Everything outside the watershed is NoData.
        """

        profile.update(
            dtype="float32",
            count=1,
            nodata=self.NODATA,
            compress="deflate",
            predictor=2,
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:
            dst.write(
                output.astype("float32"),
                1,
            )

    # ------------------------------------------------------------------
    # Continuous predictors
    # ------------------------------------------------------------------

    def normalize_continuous(
        self,
        name,
        inverse=False,
    ):
        data, profile, roi_mask = self.read(name)

        nodata = profile.get("nodata")

        valid = self.valid_mask(
            data,
            roi_mask,
            nodata,
        )

        if inverse:
            normalized, minimum, maximum = self.inverse_minmax(
                data,
                valid,
            )
        else:
            normalized, minimum, maximum = self.minmax(
                data,
                valid,
            )

        output = np.full(
            data.shape,
            self.NODATA,
            dtype="float32",
        )

        output[valid] = normalized[valid]

        output_path = self.output_dir / f"{name}_suitability.tif"

        self.write_output(
            output_path,
            output,
            profile,
        )

        print(f"\n{name}")
        print(f"  ROI pixels : {roi_mask.sum():,}")
        print(f"  Valid      : {valid.sum():,}")
        print(f"  Coverage   : {valid.sum() / roi_mask.sum() * 100:.2f}%")
        print(f"  Input min  : {minimum:.4f}")
        print(f"  Input max  : {maximum:.4f}")
        print(f"  Direction  : {'inverse' if inverse else 'direct'}")
        print(f"  Output     : {output_path}")

        return output_path

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    def normalize_ndvi(self):
        name = "ndvi"

        data, profile, roi_mask = self.read(name)

        nodata = profile.get("nodata")

        valid = self.valid_mask(
            data,
            roi_mask,
            nodata,
        )

        values = data[valid]

        if values.size == 0:
            raise ValueError("No valid NDVI pixels inside watershed ROI.")

        lower = np.percentile(
            values,
            self.NDVI_LOWER_PERCENTILE,
        )

        upper = np.percentile(
            values,
            self.NDVI_UPPER_PERCENTILE,
        )

        if upper <= lower:
            raise ValueError(f"Invalid NDVI percentile range: {lower} >= {upper}")

        normalized = (data.astype("float32") - lower) / (upper - lower)

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        output = np.full(
            data.shape,
            self.NODATA,
            dtype="float32",
        )

        output[valid] = normalized[valid]

        output_path = self.output_dir / "ndvi_suitability.tif"

        self.write_output(
            output_path,
            output,
            profile,
        )

        print("\nndvi")
        print(f"  ROI pixels : {roi_mask.sum():,}")
        print(f"  Valid      : {valid.sum():,}")
        print(f"  Coverage   : {valid.sum() / roi_mask.sum() * 100:.2f}%")
        print(f"  Input min  : {values.min():.4f}")
        print(f"  Input max  : {values.max():.4f}")
        print(
            f"  Method     : "
            f"P{self.NDVI_LOWER_PERCENTILE}–"
            f"P{self.NDVI_UPPER_PERCENTILE}"
        )
        print(f"  P5         : {lower:.4f}")
        print(f"  P95        : {upper:.4f}")
        print(f"  Output     : {output_path}")

        return output_path

    # ------------------------------------------------------------------
    # Land cover
    # ------------------------------------------------------------------

    def normalize_landcover(self):
        data, profile, roi_mask = self.read("landcover")

        nodata = profile.get("nodata")

        valid = self.valid_mask(
            data,
            roi_mask,
            nodata,
        )

        output = np.full(
            data.shape,
            self.NODATA,
            dtype="float32",
        )

        classes = np.unique(data[valid])

        print("\nlandcover")
        print(f"  ROI pixels : {roi_mask.sum():,}")
        print(f"  Valid      : {valid.sum():,}")
        print(f"  Coverage   : {valid.sum() / roi_mask.sum() * 100:.2f}%")
        print(f"  Classes    : {classes.tolist()}")

        unmapped = []

        for class_value in classes:
            class_int = int(class_value)

            if class_int not in self.LANDCOVER_SUITABILITY:
                unmapped.append(class_int)
                continue

            class_mask = valid & (data == class_int)

            output[class_mask] = self.LANDCOVER_SUITABILITY[class_int]

        if unmapped:
            raise ValueError(
                f"Unmapped WorldCover classes inside watershed ROI: {unmapped}"
            )

        output_path = self.output_dir / "landcover_suitability.tif"

        self.write_output(
            output_path,
            output,
            profile,
        )

        print(f"  Output     : {output_path}")

        return output_path

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        print("=" * 60)
        print("P7 — CORRIDOR PREDICTOR NORMALIZATION")
        print("=" * 60)

        self.load_roi()

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
        self.normalize_ndvi()

        print("\n[5] Distance to water")
        self.normalize_continuous(
            "distance_to_water",
            inverse=True,
        )

        print("\n" + "=" * 60)
        print("P7 normalization completed.")
        print("=" * 60)


if __name__ == "__main__":
    normalization = CorridorPredictorNormalization()
    normalization.run()
