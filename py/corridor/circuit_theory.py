import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio

from src.corridor.circuit_theory import CircuitTheory

ROOT = Path(__file__).resolve().parents[2]

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"

SOURCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity" / "source"

OUTPUT_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity" / "circuit"


def main():
    start_time = datetime.now().astimezone()
    workflow_start = perf_counter()

    print("=" * 65)
    print("CIRCUIT THEORY ANALYSIS")
    print(f"Started at : {start_time.isoformat(timespec='seconds')}")
    print("=" * 65)

    circuit = CircuitTheory(
        resistance_path=(
            RESISTANCE_DIR / "composite_resistance_literature_weighted.tif"
        ),
        source_path=SOURCE_DIR / "khl_source_mask.tif",
        current_output=OUTPUT_DIR / "current_density.tif",
        connectivity_output=OUTPUT_DIR / "connectivity.tif",
    )

    # Build dan perhitungan Circuit Theory.
    circuit_start = perf_counter()
    circuit.run()
    circuit_elapsed = perf_counter() - circuit_start

    print(
        f"\n[TIME] Circuit Theory execution: "
        f"{circuit_elapsed:.2f} seconds "
        f"({circuit_elapsed / 60:.2f} minutes)"
    )

    # Statistik hasil raster.
    analysis_start = perf_counter()

    analyze_outputs(
        current_output=OUTPUT_DIR / "current_density.tif",
        connectivity_output=OUTPUT_DIR / "connectivity.tif",
        output_dir=OUTPUT_DIR,
    )

    analysis_elapsed = perf_counter() - analysis_start
    workflow_elapsed = perf_counter() - workflow_start

    print(f"\n[TIME] Raster analysis: {analysis_elapsed:.2f} seconds")
    print(
        f"[TIME] Total workflow: {workflow_elapsed:.2f} seconds "
        f"({workflow_elapsed / 60:.2f} minutes)"
    )


def analyze_raster(path):
    """Calculate descriptive statistics for valid raster cells."""

    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(np.float64)
        values = data.compressed()

        if values.size == 0:
            raise ValueError(f"No valid raster cells found: {path}")

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            raise ValueError(f"No finite raster values found: {path}")

        stats = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "valid_cells": int(finite_values.size),
            "nodata_cells": int(data.size - finite_values.size),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
            "median": float(np.median(finite_values)),
            "std": float(np.std(finite_values)),
            "zero_cells": int(np.count_nonzero(finite_values == 0)),
            "negative_cells": int(np.count_nonzero(finite_values < 0)),
            "percentiles": {
                f"p{percentile:02d}": float(np.percentile(finite_values, percentile))
                for percentile in (1, 5, 25, 75, 95, 99)
            },
        }

        return stats


def analyze_outputs(current_output, connectivity_output, output_dir):
    """Analyze and save statistics for Circuit Theory outputs."""

    print("\n[ANALYSIS] Analyzing Circuit Theory outputs...")

    results = {
        "current_density": analyze_raster(current_output),
        "connectivity": analyze_raster(connectivity_output),
    }

    report_path = output_dir / "circuit_analysis_summary.json"

    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    for name, stats in results.items():
        print(f"\n[{name.upper()}]")
        print(f"  Valid cells : {stats['valid_cells']:,}")
        print(f"  Min         : {stats['min']:.8f}")
        print(f"  Max         : {stats['max']:.8f}")
        print(f"  Mean        : {stats['mean']:.8f}")
        print(f"  Median      : {stats['median']:.8f}")
        print(f"  Std         : {stats['std']:.8f}")
        print(f"  Zero cells  : {stats['zero_cells']:,}")

    print(f"\n[ANALYSIS] Summary saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
