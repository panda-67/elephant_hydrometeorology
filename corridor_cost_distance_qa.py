from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parent

CONNECTIVITY_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity"

SOURCE_MASK = CONNECTIVITY_DIR / "source" / "khl_source_mask.tif"

SCENARIOS = {
    "equal_weight": CONNECTIVITY_DIR / "equal_weight" / "cost_distance.tif",
    "literature_weighted": (
        CONNECTIVITY_DIR / "literature_weighted" / "cost_distance.tif"
    ),
}


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
        nodata = src.nodata

    return data, profile, transform, crs, shape, nodata


def percentile_stats(values):
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main():
    print("=" * 70)
    print("P9.2 — COST DISTANCE QA/QC")
    print("=" * 70)

    source, _, source_transform, source_crs, source_shape, source_nodata = read_raster(
        SOURCE_MASK
    )

    source_mask = source == 1

    print("\nSource mask")
    print(f"  CRS              : {source_crs}")
    print(f"  Dimensions       : {source_shape[1]} × {source_shape[0]}")
    print(f"  Source pixels    : {int(source_mask.sum()):,}")

    results = {}

    for scenario, path in SCENARIOS.items():
        print("\n" + "-" * 70)
        print(f"SCENARIO: {scenario}")
        print("-" * 70)

        if not path.exists():
            raise FileNotFoundError(f"Cost-distance raster not found: {path}")

        data, profile, transform, crs, shape, nodata = read_raster(path)

        # --------------------------------------------------------------
        # Grid checks
        # --------------------------------------------------------------

        grid_pass = (
            shape == source_shape
            and transform == source_transform
            and crs == source_crs
        )

        print("\nGrid")
        print(f"  CRS              : {crs}")
        print(f"  Dimensions       : {shape[1]} × {shape[0]}")
        print(f"  Grid             : {'PASS' if grid_pass else 'FAIL'}")

        if not grid_pass:
            raise ValueError(f"{scenario}: grid does not match source mask.")

        # --------------------------------------------------------------
        # Valid mask
        # --------------------------------------------------------------

        valid = np.isfinite(data)

        if nodata is not None:
            valid &= data != nodata

        non_source = valid & ~source_mask

        valid_count = int(valid.sum())
        source_count = int(source_mask.sum())
        non_source_count = int(non_source.sum())

        print("\nPixel accounting")
        print(f"  Valid pixels     : {valid_count:,}")
        print(f"  Source pixels    : {source_count:,}")
        print(f"  Non-source       : {non_source_count:,}")

        # --------------------------------------------------------------
        # Source cost
        # --------------------------------------------------------------

        source_values = data[source_mask]

        source_zero = np.allclose(
            source_values,
            0.0,
            atol=1e-7,
        )

        print("\nSource cost")
        print(f"  Min              : {np.min(source_values):.9f}")
        print(f"  Max              : {np.max(source_values):.9f}")
        print(f"  Source = 0       : {'PASS' if source_zero else 'FAIL'}")

        if not source_zero:
            raise ValueError(f"{scenario}: source pixels are not all zero.")

        # --------------------------------------------------------------
        # Reachability
        # --------------------------------------------------------------

        unreachable = valid & ~np.isfinite(data)

        reachable_pass = not unreachable.any()

        print("\nReachability")
        print(f"  Unreachable      : {int(unreachable.sum()):,}")
        print(f"  All reachable    : {'PASS' if reachable_pass else 'FAIL'}")

        if not reachable_pass:
            raise ValueError(f"{scenario}: valid pixels contain unreachable cells.")

        # --------------------------------------------------------------
        # Non-source statistics
        # --------------------------------------------------------------

        if non_source_count == 0:
            raise ValueError(f"{scenario}: no non-source valid pixels found.")

        non_source_values = data[non_source]

        stats = percentile_stats(non_source_values)

        print("\nNon-source cost statistics")
        print(f"  Min              : {stats['min']:.9f}")
        print(f"  P05              : {stats['p05']:.9f}")
        print(f"  P25              : {stats['p25']:.9f}")
        print(f"  Median           : {stats['median']:.9f}")
        print(f"  P75              : {stats['p75']:.9f}")
        print(f"  P95              : {stats['p95']:.9f}")
        print(f"  Max              : {stats['max']:.9f}")
        print(f"  Mean             : {stats['mean']:.9f}")

        positive_pass = np.all(non_source_values > 0)

        print("\nCost behaviour")
        print(f"  Non-source > 0   : {'PASS' if positive_pass else 'FAIL'}")

        if not positive_pass:
            raise ValueError(
                f"{scenario}: non-source pixels contain zero/negative cost."
            )

        results[scenario] = {
            "data": data,
            "valid": valid,
            "non_source": non_source,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------

    equal = results["equal_weight"]
    weighted = results["literature_weighted"]

    comparison_mask = equal["valid"] & weighted["valid"]

    equal_values = equal["data"][comparison_mask]
    weighted_values = weighted["data"][comparison_mask]

    difference = weighted_values - equal_values
    absolute_difference = np.abs(difference)

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    print(f"\nCommon valid pixels : {len(equal_values):,}")

    print("\nDifference")
    print(f"  Mean difference   : {np.mean(difference):.9f}")
    print(f"  MAE               : {np.mean(absolute_difference):.9f}")
    print(f"  P05               : {np.percentile(difference, 5):.9f}")
    print(f"  Median            : {np.median(difference):.9f}")
    print(f"  P95               : {np.percentile(difference, 95):.9f}")
    print(f"  Max abs difference: {np.max(absolute_difference):.9f}")

    scenarios_differ = not np.allclose(
        equal_values,
        weighted_values,
        atol=1e-7,
    )

    print("\nScenario differentiation")
    print(f"  Equal ≠ weighted : {'PASS' if scenarios_differ else 'FAIL'}")

    # ------------------------------------------------------------------
    # Final QA/QC
    # ------------------------------------------------------------------

    checks = [
        grid_pass,
        source_zero,
        reachable_pass,
        positive_pass,
        scenarios_differ,
    ]

    overall_pass = all(checks)

    print("\n" + "=" * 70)
    print(f"P9.2 QA/QC RESULT: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
