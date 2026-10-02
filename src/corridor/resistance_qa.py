from pathlib import Path

import numpy as np
import rasterio


class ResistanceQA:
    RESISTANCE_FILES = {
        "elevation": "elevation_resistance.tif",
        "slope": "slope_resistance.tif",
        "landcover": "landcover_resistance.tif",
        "ndvi": "ndvi_resistance.tif",
        "distance_to_water": "distance_to_water_resistance.tif",
    }

    def __init__(self, resistance_dir: Path):
        self.resistance_dir = resistance_dir

    def _read(self, path: Path):
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")

            valid = np.isfinite(data)

            if src.nodata is not None:
                valid &= data != src.nodata

            profile = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "bounds": src.bounds,
            }

        return data, valid, profile

    def run(self):
        print("=" * 70)
        print("P8 — RESISTANCE SURFACE QA/QC")
        print("=" * 70)

        results = {}

        reference = None

        for name, filename in self.RESISTANCE_FILES.items():
            path = self.resistance_dir / filename

            if not path.exists():
                raise FileNotFoundError(f"Resistance raster not found: {path}")

            data, valid, profile = self._read(path)

            if not valid.any():
                raise ValueError(f"{name}: no valid pixels found.")

            values = data[valid]

            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())

            p05, p25, median, p75, p95 = np.percentile(
                values,
                [5, 25, 50, 75, 95],
            )

            valid_count = int(valid.sum())
            nodata_count = int((~valid).sum())
            total = data.size
            coverage = valid_count / total * 100

            range_pass = minimum >= 0.0 and maximum <= 1.0

            if reference is None:
                reference = profile

                grid_pass = True
                crs_pass = True
                resolution_pass = True
                dimension_pass = True
                extent_pass = True
            else:
                crs_pass = profile["crs"] == reference["crs"]

                resolution_pass = np.isclose(
                    profile["transform"].a,
                    reference["transform"].a,
                ) and np.isclose(
                    profile["transform"].e,
                    reference["transform"].e,
                )

                dimension_pass = (
                    profile["width"] == reference["width"]
                    and profile["height"] == reference["height"]
                )

                extent_pass = all(
                    np.isclose(a, b)
                    for a, b in zip(
                        profile["bounds"],
                        reference["bounds"],
                    )
                )

                grid_pass = (
                    crs_pass and resolution_pass and dimension_pass and extent_pass
                )

            results[name] = {
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "p05": float(p05),
                "p25": float(p25),
                "median": float(median),
                "p75": float(p75),
                "p95": float(p95),
                "valid": valid_count,
                "nodata": nodata_count,
                "coverage": coverage,
                "range_pass": range_pass,
                "grid_pass": grid_pass,
            }

            print(f"\n{name}")
            print(f"  File       : {path}")
            print(f"  CRS        : {profile['crs']}")
            print(f"  Dimensions : {profile['width']} × {profile['height']}")
            print(f"  Resolution : {profile['transform'].a:.15f}")
            print(f"  Min        : {minimum:.6f}")
            print(f"  P05        : {p05:.6f}")
            print(f"  P25        : {p25:.6f}")
            print(f"  Median     : {median:.6f}")
            print(f"  P75        : {p75:.6f}")
            print(f"  P95        : {p95:.6f}")
            print(f"  Max        : {maximum:.6f}")
            print(f"  Mean       : {mean:.6f}")
            print(f"  Valid      : {valid_count:,}")
            print(f"  NoData     : {nodata_count:,}")
            print(f"  Coverage   : {coverage:.2f}%")
            print(f"  Range [0,1]: {'PASS' if range_pass else 'FAIL'}")
            print(f"  Grid       : {'PASS' if grid_pass else 'FAIL'}")

        print("\n" + "=" * 70)
        print("GRID CONSISTENCY")
        print("=" * 70)

        grid_pass = all(result["grid_pass"] for result in results.values())

        print(f"Overall grid consistency: {'PASS' if grid_pass else 'FAIL'}")

        print("\n" + "=" * 70)
        print("RESISTANCE RANGE")
        print("=" * 70)

        range_pass = all(result["range_pass"] for result in results.values())

        print(f"All resistance values within [0,1]: {'PASS' if range_pass else 'FAIL'}")

        overall_pass = grid_pass and range_pass

        print("\n" + "=" * 70)
        print(
            "P8 QA/QC:",
            "PASS" if overall_pass else "FAIL",
        )
        print("=" * 70)

        if not overall_pass:
            raise ValueError("P8 resistance QA/QC failed.")

        return results
