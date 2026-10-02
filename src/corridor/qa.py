from pathlib import Path

import numpy as np
import rasterio


class CorridorPredictorQA:
    def __init__(self, root=None):
        self.root = root or Path(__file__).resolve().parents[2]

        self.stack_dir = self.root / "data" / "output_rasters" / "corridor" / "stack"

        self.files = {
            "elevation": self.stack_dir / "elevation_30m.tif",
            "slope": self.stack_dir / "slope_30m.tif",
            "landcover": self.stack_dir / "landcover_30m.tif",
            "ndvi": self.stack_dir / "ndvi_30m.tif",
            "distance_to_water": self.stack_dir / "distance_to_water_30m.tif",
        }

    def read(self, name):
        path = self.files[name]

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)

            return {
                "data": data,
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "res": src.res,
            }

    def check_grid(self):
        print("\n[1] GRID CONSISTENCY")

        reference = self.read("elevation")

        print(f"    CRS        : {reference['crs']}")
        print(f"    Resolution : {reference['res']}")
        print(f"    Dimensions : {reference['width']} × {reference['height']}")

        for name in self.files:
            raster = self.read(name)

            checks = {
                "CRS": raster["crs"] == reference["crs"],
                "transform": raster["transform"] == reference["transform"],
                "dimensions": (
                    raster["width"] == reference["width"]
                    and raster["height"] == reference["height"]
                ),
            }

            status = all(checks.values())

            print(f"    {name:<20} {'PASS' if status else 'FAIL'}")

            if not status:
                print(f"      CRS       : {checks['CRS']}")
                print(f"      Transform : {checks['transform']}")
                print(f"      Dimensions: {checks['dimensions']}")

    def check_ranges(self):
        print("\n[2] VALUE RANGE")

        expected = {
            "elevation": (0, 3000),
            "slope": (0, 90),
            "landcover": (10, 100),
            "ndvi": (-1, 1),
            "distance_to_water": (0, 2500 * np.sqrt(2)),
        }

        for name, (lower, upper) in expected.items():
            raster = self.read(name)
            data = raster["data"]

            if data.count() == 0:
                print(f"    {name:<20} FAIL — no valid pixels")
                continue

            values = data.compressed()

            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())

            valid = int(data.count())
            total = data.size
            coverage = valid / total * 100

            tolerance = 1e-3

            valid_range = minimum >= lower - tolerance and maximum <= upper + tolerance

            print(f"\n    {name}")
            print(f"      Min      : {minimum:.4f}")
            print(f"      Max      : {maximum:.4f}")
            print(f"      Mean     : {mean:.4f}")
            print(f"      Valid    : {valid:,}")
            print(f"      Coverage : {coverage:.2f}%")
            print(f"      Range    : {'PASS' if valid_range else 'FAIL'}")

            if not valid_range:
                print(f"      Expected : {lower} → {upper}")

    def check_anomalies(self):
        print("\n[3] ANOMALY CHECK")

        tests = {
            "elevation": lambda x: x < 0,
            "slope": lambda x: (x < 0) | (x > 90),
            "ndvi": lambda x: (x < -1) | (x > 1),
            "distance_to_water": lambda x: x < 0,
        }

        for name, test in tests.items():
            raster = self.read(name)
            data = raster["data"].compressed()

            if len(data) == 0:
                print(f"    {name:<20} FAIL — no valid pixels")
                continue

            count = int(np.count_nonzero(test(data)))

            print(
                f"    {name:<20} "
                f"{'PASS' if count == 0 else 'FAIL'}"
                f" — anomalous pixels: {count:,}"
            )

    def check_distance(self):
        print("\n[4] DISTANCE-TO-WATER QA")

        raster = self.read("distance_to_water")
        data = raster["data"]

        if data.count() == 0:
            print("    FAIL — no valid distance pixels")
            return

        values = data.compressed()

        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())

        zero_pixels = int(np.count_nonzero(values == 0))
        beyond_2500 = int(np.count_nonzero(values > 2500))

        print(f"    Min              : {minimum:.4f} m")
        print(f"    Max              : {maximum:.4f} m")
        print(f"    Mean             : {mean:.4f} m")
        print(f"    Zero-distance px : {zero_pixels:,}")
        print(f"    > 2500 m         : {beyond_2500:,}")

        if beyond_2500 > 0:
            print("    WARNING: values exceed the configured 2500 m search radius.")
        else:
            print("    Radius check     : PASS")

    def check_nodata(self):
        print("\n[5] NODATA / COVERAGE")

        for name in self.files:
            raster = self.read(name)
            data = raster["data"]

            valid = int(data.count())
            nodata = data.size - valid
            coverage = valid / data.size * 100

            print(
                f"    {name:<20} "
                f"valid={valid:,} "
                f"nodata={nodata:,} "
                f"coverage={coverage:.2f}%"
            )

    def run(self):
        print("=" * 60)
        print("P6 — CORRIDOR PREDICTOR QA/QC")
        print("=" * 60)

        self.check_grid()
        self.check_ranges()
        self.check_anomalies()
        self.check_distance()
        self.check_nodata()

        print("\n" + "=" * 60)
        print("QA/QC completed.")
        print("=" * 60)
