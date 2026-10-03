from pathlib import Path

import numpy as np
import rasterio


class CorridorPredictorQA:
    def __init__(self, root=None):
        self.root = root or Path(__file__).resolve().parents[2]

        self.corridor_dir = self.root / "data" / "output_rasters" / "corridor"

        self.files = {
            "elevation": self.corridor_dir / "corridor_elevation.tif",
            "slope": self.corridor_dir / "corridor_slope.tif",
            "landcover": (self.corridor_dir / "corridor_landcover_worldcover_2020.tif"),
            "ndvi": self.corridor_dir / "corridor_ndvi.tif",
            "distance_to_water": (self.corridor_dir / "corridor_distance_to_water.tif"),
        }

        # ESA WorldCover 2020 valid class codes.
        self.landcover_classes = {
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            95,
            100,
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

    def finite_values(self, data):
        """
        Return finite, unmasked raster values.

        Masked values and NaN/Inf are treated as invalid.
        """
        values = data.compressed()

        if values.size == 0:
            return np.array([], dtype=np.float64)

        values = np.asarray(values, dtype=np.float64)

        return values[np.isfinite(values)]

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
            "ndvi": (-1, 1),
            "distance_to_water": (0, None),
        }

        for name in [
            "elevation",
            "slope",
            "ndvi",
            "distance_to_water",
        ]:
            lower, upper = expected[name]

            raster = self.read(name)
            data = raster["data"]

            values = self.finite_values(data)

            if values.size == 0:
                print(f"    {name:<20} FAIL — no finite pixels")
                continue

            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())

            valid = int(values.size)
            total = data.size
            coverage = valid / total * 100

            tolerance = 1e-3

            lower_ok = minimum >= lower - tolerance

            if upper is None:
                upper_ok = True
            else:
                upper_ok = maximum <= upper + tolerance

            valid_range = lower_ok and upper_ok

            print(f"\n    {name}")
            print(f"      Min      : {minimum:.4f}")
            print(f"      Max      : {maximum:.4f}")
            print(f"      Mean     : {mean:.4f}")
            print(f"      Valid    : {valid:,}")
            print(f"      Coverage : {coverage:.2f}%")
            print(f"      Range    : {'PASS' if valid_range else 'FAIL'}")

            if not valid_range:
                if upper is None:
                    print(f"      Expected : >= {lower}")
                else:
                    print(f"      Expected : {lower} → {upper}")

        self.check_landcover_range()

    def check_landcover_range(self):
        print("\n    landcover")

        raster = self.read("landcover")
        data = raster["data"]

        values = self.finite_values(data)

        if values.size == 0:
            print("      FAIL — no finite pixels")
            return

        unique, counts = np.unique(
            values.astype(np.int16),
            return_counts=True,
        )

        invalid_mask = ~np.isin(
            unique,
            list(self.landcover_classes),
        )

        invalid_classes = unique[invalid_mask]
        invalid_count = int(counts[invalid_mask].sum())

        valid_count = values.size - invalid_count
        total = data.size

        coverage = valid_count / total * 100

        if invalid_count == 0:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"      Valid classes : {valid_count:,}")
        print(f"      Invalid px    : {invalid_count:,}")
        print(f"      Coverage      : {coverage:.2f}%")
        print(f"      Classes       : {sorted(unique.tolist())}")
        print(f"      Class check   : {status}")

        if invalid_classes.size > 0:
            print(f"      Invalid class codes: {invalid_classes.tolist()}")

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
            data = raster["data"]

            values = self.finite_values(data)

            if values.size == 0:
                print(f"    {name:<20} FAIL — no finite pixels")
                continue

            count = int(np.count_nonzero(test(values)))

            print(
                f"    {name:<20} "
                f"{'PASS' if count == 0 else 'FAIL'}"
                f" — anomalous pixels: {count:,}"
            )

    def check_distance(self):
        print("\n[4] DISTANCE-TO-WATER QA")

        raster = self.read("distance_to_water")
        data = raster["data"]

        values = self.finite_values(data)

        if values.size == 0:
            print("    FAIL — no finite distance pixels")
            return

        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())

        zero_pixels = int(np.count_nonzero(values == 0))

        beyond_2500 = int(np.count_nonzero(values > 2500))

        negative = int(np.count_nonzero(values < 0))

        print(f"    Min              : {minimum:.4f} m")
        print(f"    Max              : {maximum:.4f} m")
        print(f"    Mean             : {mean:.4f} m")
        print(f"    Zero-distance px : {zero_pixels:,}")
        print(f"    > 2500 m         : {beyond_2500:,}")
        print(f"    Negative px      : {negative:,}")

        if negative > 0:
            print("    Radius check     : FAIL — negative distance values detected.")
        else:
            print("    Radius check     : PASS — no negative distances.")

        if beyond_2500 > 0:
            print(
                "    Note             : "
                "values >2500 m detected; "
                "this is not treated as an error."
            )

    def check_nodata(self):
        print("\n[5] NODATA / COVERAGE")

        for name in self.files:
            raster = self.read(name)
            data = raster["data"]

            masked_count = int(np.ma.count_masked(data))

            finite_count = int(np.count_nonzero(np.isfinite(data.compressed())))

            nan_count = int(
                np.count_nonzero(
                    np.isnan(
                        np.asarray(
                            data.data,
                            dtype=np.float64,
                        )
                    )
                    & ~np.ma.getmaskarray(data)
                )
            )

            inf_count = int(
                np.count_nonzero(
                    np.isinf(
                        np.asarray(
                            data.data,
                            dtype=np.float64,
                        )
                    )
                    & ~np.ma.getmaskarray(data)
                )
            )

            total = data.size

            invalid_count = masked_count + nan_count + inf_count

            coverage = finite_count / total * 100

            print(
                f"    {name:<20} "
                f"finite={finite_count:,} "
                f"masked={masked_count:,} "
                f"nan={nan_count:,} "
                f"inf={inf_count:,} "
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
