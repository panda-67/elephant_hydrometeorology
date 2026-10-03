from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "data" / "output_rasters" / "corridor" / "normalized"

PREDICTORS = {
    "elevation": "elevation_suitability.tif",
    "slope": "slope_suitability.tif",
    "landcover": "landcover_suitability.tif",
    "ndvi": "ndvi_suitability.tif",
    "distance_to_water": "distance_to_water_suitability.tif",
}

BINS = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
)


def load_valid_values(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata

        if nodata is not None:
            data = data[data != nodata]

        data = data[np.isfinite(data)]
        data = data[(data >= 0) & (data <= 1)]

    return data


def summarize(name, values):
    print(f"\n{name}")
    print("-" * 60)

    print(f"  Valid pixels : {len(values):,}")
    print(f"  Min          : {values.min():.6f}")
    print(f"  P05          : {np.percentile(values, 5):.6f}")
    print(f"  P25          : {np.percentile(values, 25):.6f}")
    print(f"  Median       : {np.percentile(values, 50):.6f}")
    print(f"  P75          : {np.percentile(values, 75):.6f}")
    print(f"  P95          : {np.percentile(values, 95):.6f}")
    print(f"  Max          : {values.max():.6f}")
    print(f"  Mean         : {values.mean():.6f}")
    print(f"  Std          : {values.std():.6f}")


def distribution(values):
    counts, edges = np.histogram(values, bins=BINS)

    total = len(values)

    print("\n  Distribution:")
    for i, count in enumerate(counts):
        low = edges[i]
        high = edges[i + 1]

        percentage = count / total * 100

        print(f"    {low:.1f}–{high:.1f} : {count:>9,} px ({percentage:>6.2f}%)")


def main():
    print("=" * 60)
    print("P7 — CORRIDOR NORMALIZATION FINAL DIAGNOSTIC")
    print("=" * 60)

    results = {}

    for name, filename in PREDICTORS.items():
        path = NORMALIZED_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing raster: {path}")

        values = load_valid_values(path)

        if len(values) == 0:
            raise ValueError(f"No valid pixels found: {path}")

        results[name] = values

        summarize(name, values)
        distribution(values)

    print("\n" + "=" * 60)
    print("P7 diagnostic completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
