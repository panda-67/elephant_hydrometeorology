from pathlib import Path

import numpy as np
import rasterio

BASELINE = Path(
    "data/output_rasters/corridor/connectivity/"
    "literature_weighted/corridor_potential.tif"
)

FLOOD = Path(
    "data/output_rasters/corridor/connectivity/flood/flood_exposed_connectivity.tif"
)

NODATA = -9999.0


def main():

    with rasterio.open(BASELINE) as src:
        baseline = src.read(1).astype(np.float32)

    with rasterio.open(FLOOD) as src:
        flood = src.read(1).astype(np.float32)

    valid = (
        np.isfinite(baseline)
        & np.isfinite(flood)
        & (baseline != NODATA)
        & (flood != NODATA)
    )

    b = baseline[valid]
    f = flood[valid]

    difference = f - b

    increased = difference > 1e-6
    decreased = difference < -1e-6
    unchanged = ~increased & ~decreased

    print()
    print("=" * 60)
    print("CORRIDOR CONNECTIVITY QA")
    print("=" * 60)

    print()
    print("VALID PIXELS")
    print("-" * 60)
    print(f"Baseline valid : {len(b):,}")
    print(f"Flood valid    : {len(f):,}")

    print()
    print("CONNECTIVITY")
    print("-" * 60)
    print(f"{'Metric':<25} {'Baseline':>15} {'Flood':>15}")
    print("-" * 60)
    print(f"{'Minimum':<25} {b.min():>15.6f} {f.min():>15.6f}")
    print(f"{'Maximum':<25} {b.max():>15.6f} {f.max():>15.6f}")
    print(f"{'Mean':<25} {b.mean():>15.6f} {f.mean():>15.6f}")
    print(f"{'Median':<25} {np.median(b):>15.6f} {np.median(f):>15.6f}")

    print()
    print("CHANGE: FLOOD - BASELINE")
    print("-" * 60)
    print(f"{'Minimum change':<25} {difference.min():>15.6f}")
    print(f"{'Maximum change':<25} {difference.max():>15.6f}")
    print(f"{'Mean change':<25} {difference.mean():>15.6f}")
    print(f"{'Median change':<25} {np.median(difference):>15.6f}")

    print()
    print("PIXEL DIRECTION")
    print("-" * 60)
    print(f"{'Increased':<25} {increased.sum():>15,}")
    print(f"{'Decreased':<25} {decreased.sum():>15,}")
    print(f"{'Unchanged':<25} {unchanged.sum():>15,}")

    print()
    print("PERCENTAGE")
    print("-" * 60)
    total = len(difference)

    print(f"{'Increased':<25} {increased.sum() / total * 100:>14.4f}%")

    print(f"{'Decreased':<25} {decreased.sum() / total * 100:>14.4f}%")

    print(f"{'Unchanged':<25} {unchanged.sum() / total * 100:>14.4f}%")

    print()
    print("THEORETICAL EXPECTATION")
    print("-" * 60)

    if difference.max() <= 1e-6:
        print("PASS")
        print("Flood connectivity tidak meningkat.")
        print("Sesuai teori: adjusted resistance ↑ → cost ↑ → connectivity ↓.")
    else:
        print("WARNING")
        print("Terdapat pixel dengan flood connectivity > baseline.")
        print(f"Maximum increase: {difference.max():.10f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
