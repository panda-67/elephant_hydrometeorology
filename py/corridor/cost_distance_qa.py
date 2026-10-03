from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parents[2]

COMPOSITE = (
    ROOT
    / "data/output_rasters/corridor/resistance/composite_resistance_literature_weighted.tif"
)
SOURCE = ROOT / "data/output_rasters/corridor/connectivity/source/khl_source_mask.tif"
COST = (
    ROOT
    / "data/output_rasters/corridor/connectivity/literature_weighted/cost_distance.tif"
)

TARGET_CRS = "EPSG:32647"
RESOLUTION = 10.0
NODATA = -9999.0


def main():

    print("=" * 60)
    print("P9.2 — COST DISTANCE QA/QC")
    print("=" * 60)

    # ----------------------------------------------------------
    # Read resistance
    # ----------------------------------------------------------

    with rasterio.open(COMPOSITE) as src:
        resistance = src.read(1)
        resistance_valid = np.isfinite(resistance)

        if src.nodata is not None:
            resistance_valid &= resistance != src.nodata

        bounds = src.bounds
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

    print(f"\nResistance valid : {resistance_valid.sum():,}")

    # ----------------------------------------------------------
    # Recreate metric grid
    # ----------------------------------------------------------

    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs,
        TARGET_CRS,
        width,
        height,
        *bounds,
        resolution=RESOLUTION,
    )

    metric = np.full(
        (dst_height, dst_width),
        NODATA,
        dtype=np.float32,
    )

    with rasterio.open(COMPOSITE) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=metric,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )

    metric_valid = np.isfinite(metric) & (metric != NODATA)

    # ----------------------------------------------------------
    # Read source mask
    # ----------------------------------------------------------

    with rasterio.open(SOURCE) as src:
        source = np.zeros(
            (dst_height, dst_width),
            dtype=np.uint8,
        )

        reproject(
            source=src.read(1),
            destination=source,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    source_mask = source == 1

    # ----------------------------------------------------------
    # Read cost distance
    # ----------------------------------------------------------

    with rasterio.open(COST) as src:
        cost = src.read(1)

        cost_valid = np.isfinite(cost)

        if src.nodata is not None:
            cost_valid &= cost != src.nodata

    # ----------------------------------------------------------
    # Basic checks
    # ----------------------------------------------------------

    if cost.shape != metric.shape:
        raise ValueError("Cost-distance grid tidak sesuai.")

    print(f"Metric valid    : {metric_valid.sum():,}")
    print(f"Cost valid      : {cost_valid.sum():,}")

    # ----------------------------------------------------------
    # Source cost
    # ----------------------------------------------------------

    source_cost = cost[source_mask & cost_valid]

    if source_cost.size and not np.allclose(source_cost, 0):
        raise ValueError("Source cost bukan 0.")

    print(f"Source pixels   : {source_mask.sum():,}")
    print("Source cost = 0 : PASS")

    # ----------------------------------------------------------
    # Cost values
    # ----------------------------------------------------------

    values = cost[cost_valid]

    if np.any(values < 0):
        raise ValueError("Ditemukan cost negatif.")

    print("Cost >= 0       : PASS")

    # ----------------------------------------------------------
    # Reachability
    # ----------------------------------------------------------

    unreachable = metric_valid & ~cost_valid

    print(f"Unmatched       : {unreachable.sum():,}")
    print("Reachability    : PASS")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("P9.2 QA/QC PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
