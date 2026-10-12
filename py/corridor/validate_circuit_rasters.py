from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/output_rasters/corridor/connectivity")
RESISTANCE = Path(
    "data/output_rasters/corridor/resistance/"
    "composite_resistance_literature_weighted.tif"
)
SOURCE = Path("data/output_rasters/corridor/connectivity/source/khl_source_mask.tif")

RASTERS = {
    "connectivity": ROOT / "circuit/connectivity.tif",
    "current_density": ROOT / "circuit/current_density.tif",
}


def inspect_raster(path):
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True)
        values = data.compressed().astype(np.float64)

        finite = np.isfinite(values)

        print(f"\n{'=' * 65}")
        print(f"Raster: {path}")
        print(f"Dimensions : {ds.width} x {ds.height}")
        print(f"CRS        : {ds.crs}")
        print(f"Transform  : {ds.transform}")
        print(f"NoData     : {ds.nodata}")
        print(f"Valid cells: {values.size:,}")

        if values.size == 0:
            print("ERROR: Raster has no valid cells.")
            return

        print(f"Finite     : {finite.sum():,}")
        print(f"Non-finite : {(~finite).sum():,}")

        values = values[finite]

        if values.size == 0:
            print("ERROR: No finite values.")
            return

        print(f"Minimum    : {values.min():.10g}")
        print(f"Maximum    : {values.max():.10g}")
        print(f"Mean       : {values.mean():.10g}")
        print(f"Median     : {np.median(values):.10g}")

        for q in (1, 5, 25, 75, 95, 99):
            print(f"P{q:02d}        : {np.percentile(values, q):.10g}")

        print(f"Zero cells : {(values == 0).sum():,}")
        print(f"Negative   : {(values < 0).sum():,}")

        if path == RASTERS["connectivity"]:
            print(f"Outside [0,1]: {((values < 0) | (values > 1)).sum():,}")


def check_grid(reference_path, path):
    with rasterio.open(reference_path) as ref, rasterio.open(path) as ds:
        checks = {
            "dimensions": (ref.width, ref.height) == (ds.width, ds.height),
            "CRS": ref.crs == ds.crs,
            "transform": ref.transform == ds.transform,
        }

        print(f"\nGrid comparison: {path.name}")
        for name, passed in checks.items():
            print(f"  {name:12s}: {'PASS' if passed else 'FAIL'}")


def main():
    for path in RASTERS.values():
        if not path.exists():
            raise FileNotFoundError(path)

        inspect_raster(path)
        check_grid(RESISTANCE, path)

    check_grid(RESISTANCE, SOURCE)


if __name__ == "__main__":
    main()
