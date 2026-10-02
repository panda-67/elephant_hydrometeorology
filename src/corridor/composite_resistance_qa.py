from pathlib import Path

import numpy as np
import rasterio


COMPOSITES = {
    "equal_weight": "composite_resistance_equal_weight.tif",
    "literature_weighted": "composite_resistance_literature_weighted.tif",
}


class CompositeResistanceQA:
    def __init__(self, resistance_dir: Path):
        self.resistance_dir = resistance_dir

    def read_raster(self, path: Path):
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
        print("P8.3 — COMPOSITE RESISTANCE QA/QC")
        print("=" * 70)

        results = {}

        reference_data = None
        reference_valid = None
        reference_profile = None

        for name, filename in COMPOSITES.items():
            path = self.resistance_dir / filename

            if not path.exists():
                raise FileNotFoundError(f"Composite raster not found: {path}")

            data, valid, profile = self.read_raster(path)

            if not valid.any():
                raise ValueError(f"{name}: no valid pixels.")

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

            if reference_profile is None:
                reference_data = data
                reference_valid = valid
                reference_profile = profile

                grid_pass = True
                mask_pass = True
            else:
                grid_pass = (
                    profile["crs"] == reference_profile["crs"]
                    and profile["width"] == reference_profile["width"]
                    and profile["height"] == reference_profile["height"]
                    and np.allclose(
                        profile["transform"],
                        reference_profile["transform"],
                    )
                    and all(
                        np.isclose(a, b)
                        for a, b in zip(
                            profile["bounds"],
                            reference_profile["bounds"],
                        )
                    )
                )

                mask_pass = np.array_equal(
                    valid,
                    reference_valid,
                )

            results[name] = {
                "data": data,
                "valid": valid,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "p05": float(p05),
                "p25": float(p25),
                "median": float(median),
                "p75": float(p75),
                "p95": float(p95),
                "valid_count": valid_count,
                "nodata_count": nodata_count,
                "coverage": coverage,
                "range_pass": range_pass,
                "grid_pass": grid_pass,
                "mask_pass": mask_pass,
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
            print(f"  Valid mask : {'PASS' if mask_pass else 'FAIL'}")

        # ----------------------------------------------------------
        # Compare the two composite surfaces
        # ----------------------------------------------------------

        equal = results["equal_weight"]
        literature = results["literature_weighted"]

        common_valid = equal["valid"] & literature["valid"]

        if not common_valid.any():
            raise ValueError("No common valid pixels between composite surfaces.")

        difference = literature["data"][common_valid] - equal["data"][common_valid]

        mae = float(np.mean(np.abs(difference)))
        mean_difference = float(np.mean(difference))
        p05_diff, p50_diff, p95_diff = np.percentile(
            difference,
            [5, 50, 95],
        )

        max_abs_difference = float(np.max(np.abs(difference)))

        print("\n" + "=" * 70)
        print("COMPOSITE SCENARIO COMPARISON")
        print("=" * 70)

        print(f"Common valid pixels : {common_valid.sum():,}")

        print(f"Mean difference     : {mean_difference:.6f}")

        print(f"MAE                 : {mae:.6f}")

        print(f"P05 difference      : {p05_diff:.6f}")

        print(f"Median difference   : {p50_diff:.6f}")

        print(f"P95 difference      : {p95_diff:.6f}")

        print(f"Max absolute diff.  : {max_abs_difference:.6f}")

        # ----------------------------------------------------------
        # Overall QA/QC
        # ----------------------------------------------------------

        range_pass = all(result["range_pass"] for result in results.values())

        grid_pass = all(result["grid_pass"] for result in results.values())

        mask_pass = all(result["mask_pass"] for result in results.values())

        overall_pass = range_pass and grid_pass and mask_pass

        print("\n" + "=" * 70)
        print("P8.3 QA/QC SUMMARY")
        print("=" * 70)

        print(f"Range [0,1] : {'PASS' if range_pass else 'FAIL'}")

        print(f"Grid         : {'PASS' if grid_pass else 'FAIL'}")

        print(f"Valid mask   : {'PASS' if mask_pass else 'FAIL'}")

        print(f"Overall       : {'PASS' if overall_pass else 'FAIL'}")

        if not overall_pass:
            raise ValueError("P8.3 composite resistance QA/QC failed.")

        return results
