import time
from pathlib import Path

from src.corridor.cost_distance import SourceBasedCostDistance

ROOT = Path(__file__).resolve().parents[2]

RESISTANCE = (
    ROOT
    / "data"
    / "output_rasters"
    / "corridor"
    / "resistance"
    / "composite_resistance_literature_weighted.tif"
)

SOURCE_MASK = (
    ROOT
    / "data"
    / "output_rasters"
    / "corridor"
    / "connectivity"
    / "source"
    / "khl_source_mask.tif"
)

OUTPUT = (
    ROOT
    / "data"
    / "output_rasters"
    / "corridor"
    / "connectivity"
    / "literature_weighted"
    / "cost_distance.tif"
)


def main():
    start_time = time.perf_counter()

    print("\n")
    print("#" * 70)
    print("# P9.2 — SOURCE-BASED COST DISTANCE")
    print("# SCENARIO: literature_weighted")
    print("#" * 70)

    if not RESISTANCE.exists():
        raise FileNotFoundError(f"Composite resistance tidak ditemukan:\n{RESISTANCE}")

    if not SOURCE_MASK.exists():
        raise FileNotFoundError(f"Source mask tidak ditemukan:\n{SOURCE_MASK}")

    print(f"\nResistance : {RESISTANCE}")
    print(f"Source     : {SOURCE_MASK}")
    print(f"Output     : {OUTPUT}")

    model = SourceBasedCostDistance(
        resistance_raster=RESISTANCE,
        source_mask=SOURCE_MASK,
        output_path=OUTPUT,
    )

    model.run()

    elapsed = time.perf_counter() - start_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "#" * 70)
    print("# P9.2 COMPLETED")
    print("#" * 70)
    print(f"# Runtime : {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("#" * 70)


if __name__ == "__main__":
    main()
