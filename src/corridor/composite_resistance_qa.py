from pathlib import Path

import numpy as np
import rasterio


COMPOSITE = "composite_resistance_literature_weighted.tif"


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

        path = self.resistance_dir / COMPOSITE

        if not path.exists():
            raise FileNotFoundError(f"Composite raster not found: {path}")

        data, valid, profile = self.read_raster(path)

        if not valid.any():
            raise ValueError("Composite raster contains no valid pixels.")

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

        # ----------------------------------------------------------
        # Value range
        # ----------------------------------------------------------

        range_pass = minimum >= 0.0 and maximum <= 1.0

        # ----------------------------------------------------------
        # Grid consistency
        #
        # Composite should use the same grid as the resistance
        # predictors.
        # ----------------------------------------------------------

        reference_path = self.resistance_dir / "elevation_resistance.tif"

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference resistance raster not found: {reference_path}"
            )

        _, _, reference_profile = self.read_raster(reference_path)

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

        # ----------------------------------------------------------
        # Report
        # ----------------------------------------------------------

        print(f"\nComposite")
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

        # ----------------------------------------------------------
        # Overall QA/QC
        # ----------------------------------------------------------

        overall_pass = range_pass and grid_pass

        print("\n" + "=" * 70)
        print("P8.3 QA/QC SUMMARY")
        print("=" * 70)

        print(f"Range [0,1] : {'PASS' if range_pass else 'FAIL'}")

        print(f"Grid         : {'PASS' if grid_pass else 'FAIL'}")

        print(f"Overall      : {'PASS' if overall_pass else 'FAIL'}")

        if not overall_pass:
            raise ValueError("P8.3 composite resistance QA/QC failed.")

        return {
            "file": str(path),
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
            "overall_pass": overall_pass,
        }
