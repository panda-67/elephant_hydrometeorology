from pathlib import Path

import numpy as np
import rasterio


class CorridorNormalizationQA:
    def __init__(self, root=None):
        self.root = root or Path(__file__).resolve().parents[2]

        self.input_dir = (
            self.root / "data" / "output_rasters" / "corridor" / "normalized"
        )

        self.files = {
            "elevation": self.input_dir / "elevation_suitability.tif",
            "slope": self.input_dir / "slope_suitability.tif",
            "landcover": self.input_dir / "landcover_suitability.tif",
            "ndvi": self.input_dir / "ndvi_suitability.tif",
            "distance_to_water": (self.input_dir / "distance_to_water_suitability.tif"),
        }

        self.reference = None

    def read(self, name):
        path = self.files[name]

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)

            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "res": src.res,
                "bounds": src.bounds,
            }

        return data, metadata

    def check_grid(self):
        print("\n[1] GRID CONSISTENCY")

        reference_data, reference = self.read("elevation")
        self.reference = reference

        print(f"    CRS        : {reference['crs']}")
        print(f"    Resolution : {reference['res']}")
        print(f"    Dimensions : {reference['width']} × {reference['height']}")

        for name in self.files:
            _, metadata = self.read(name)

            crs_ok = metadata["crs"] == reference["crs"]
            transform_ok = metadata["transform"] == reference["transform"]
            dimensions_ok = (
                metadata["width"] == reference["width"]
                and metadata["height"] == reference["height"]
            )

            passed = crs_ok and transform_ok and dimensions_ok

            print(f"    {name:<20} {'PASS' if passed else 'FAIL'}")

    def check_range(self):
        print("\n[2] SUITABILITY RANGE")

        for name in self.files:
            data, _ = self.read(name)

            if data.count() == 0:
                print(f"    {name:<20} FAIL — no valid pixels")
                continue

            values = data.compressed()

            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())

            anomalous = int(
                np.count_nonzero((values < 0) | (values > 1) | ~np.isfinite(values))
            )

            passed = anomalous == 0

            print(f"\n    {name}")
            print(f"      Min      : {minimum:.6f}")
            print(f"      Max      : {maximum:.6f}")
            print(f"      Mean     : {mean:.6f}")
            print(f"      Anomaly  : {anomalous:,}")
            print(f"      Range    : {'PASS' if passed else 'FAIL'}")

    def check_coverage(self):
        print("\n[3] COVERAGE")

        for name in self.files:
            data, _ = self.read(name)

            valid = int(data.count())
            total = data.size
            nodata = total - valid
            coverage = valid / total * 100

            print(
                f"    {name:<20} "
                f"valid={valid:,} "
                f"nodata={nodata:,} "
                f"coverage={coverage:.2f}%"
            )

    def check_landcover(self):
        print("\n[4] LAND COVER SUITABILITY")

        data, _ = self.read("landcover")

        values = data.compressed()

        if len(values) == 0:
            print("    FAIL — no valid pixels")
            return

        unique, counts = np.unique(
            values,
            return_counts=True,
        )

        for value, count in zip(unique, counts):
            percentage = count / len(values) * 100

            print(f"    Suitability {value:.2f} : {count:,} px ({percentage:.2f}%)")

    def run(self):
        print("=" * 60)
        print("P7 — CORRIDOR NORMALIZATION QA/QC")
        print("=" * 60)

        self.check_grid()
        self.check_range()
        self.check_coverage()
        self.check_landcover()

        print("\n" + "=" * 60)
        print("Normalization QA/QC completed.")
        print("=" * 60)
