from pathlib import Path

import numpy as np
import rasterio


class CompositeResistance:
    RESISTANCE_FILES = {
        "elevation": "elevation_resistance.tif",
        "slope": "slope_resistance.tif",
        "landcover": "landcover_resistance.tif",
        "ndvi": "ndvi_resistance.tif",
        "distance_to_water": "distance_to_water_resistance.tif",
    }

    # DEFAULT_WEIGHTS = {
    #     "elevation": 0.11,
    #     "slope": 0.17,
    #     "landcover": 0.43,
    #     "ndvi": 0.29,
    # }

    DEFAULT_WEIGHTS = {
        "elevation": 0.08,
        "slope": 0.12,
        "landcover": 0.30,
        "ndvi": 0.20,
        "distance_to_water": 0.30,
    }

    def __init__(
        self,
        resistance_dir: Path,
        output_dir: Path,
        weights: dict | None = None,
    ):
        self.resistance_dir = resistance_dir
        self.output_dir = output_dir

        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._validate_weights()

    def _validate_weights(self):
        expected = set(self.RESISTANCE_FILES)
        actual = set(self.weights)

        if actual != expected:
            raise ValueError("Weight keys must exactly match resistance predictors.")

        total = sum(self.weights.values())

        if not np.isclose(total, 1.0):
            raise ValueError(f"Resistance weights must sum to 1.0, got {total:.6f}")

        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Negative weight for {name}: {weight}")

    def run(self):
        print("=" * 70)
        print("P8.2 — COMPOSITE RESISTANCE")
        print("=" * 70)

        arrays = {}
        reference_profile = None
        common_valid = None

        for name, filename in self.RESISTANCE_FILES.items():
            path = self.resistance_dir / filename

            if not path.exists():
                raise FileNotFoundError(f"Resistance raster not found: {path}")

            with rasterio.open(path) as src:
                data = src.read(1).astype("float32")

                valid = np.isfinite(data)

                if src.nodata is not None:
                    valid &= data != src.nodata

                if reference_profile is None:
                    reference_profile = src.profile.copy()
                    common_valid = valid.copy()
                else:
                    if src.crs != reference_profile["crs"]:
                        raise ValueError(f"{name}: CRS mismatch.")

                    if (
                        src.width != reference_profile["width"]
                        or src.height != reference_profile["height"]
                    ):
                        raise ValueError(f"{name}: dimension mismatch.")

                    if not np.allclose(
                        src.transform,
                        reference_profile["transform"],
                    ):
                        raise ValueError(f"{name}: transform mismatch.")

                    # Composite hanya boleh dihitung pada piksel
                    # yang valid pada SEMUA predictor.
                    common_valid &= valid

                arrays[name] = data

                print(f"  {name:<20} valid={valid.sum():,}")

        common_count = int(common_valid.sum())

        if common_count == 0:
            raise ValueError("No common valid pixels across all resistance rasters.")

        print(f"\nCommon valid pixels : {common_count:,}")

        total_pixels = common_valid.size
        coverage = common_count / total_pixels * 100

        print(f"Common coverage     : {coverage:.2f}%")

        print("\nWeights:")

        for name, weight in self.weights.items():
            print(f"  {name:<20} {weight:.2f}")

        composite = np.full(
            arrays["elevation"].shape,
            np.nan,
            dtype="float32",
        )

        composite_values = np.zeros(
            common_count,
            dtype="float32",
        )

        for name, weight in self.weights.items():
            composite_values += arrays[name][common_valid] * weight

        composite[common_valid] = composite_values

        output_path = self.output_dir / "composite_resistance_literature_weighted.tif"

        profile = reference_profile.copy()

        profile.update(
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
        )

        output = np.full(
            composite.shape,
            -9999.0,
            dtype="float32",
        )

        output[common_valid] = composite[common_valid]

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:
            dst.write(output, 1)

        values = composite[common_valid]

        print("\nComposite statistics:")
        print(f"  Min    : {values.min():.6f}")
        print(f"  P05    : {np.percentile(values, 5):.6f}")
        print(f"  P25    : {np.percentile(values, 25):.6f}")
        print(f"  Median : {np.percentile(values, 50):.6f}")
        print(f"  P75    : {np.percentile(values, 75):.6f}")
        print(f"  P95    : {np.percentile(values, 95):.6f}")
        print(f"  Max    : {values.max():.6f}")
        print(f"  Mean   : {values.mean():.6f}")

        print(f"\nOutput: {output_path}")

        print("\n" + "=" * 70)
        print("P8.2 composite resistance completed.")
        print("=" * 70)

        return output_path
