from datetime import datetime
from pathlib import Path
from time import perf_counter

from src.corridor.connectivity_synthesis import ConnectivitySynthesis

ROOT = Path(__file__).resolve().parents[2]

CONNECTIVITY_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity"

RESISTANCE = (
    ROOT
    / "data"
    / "output_rasters"
    / "corridor"
    / "resistance"
    / "composite_resistance_literature_weighted.tif"
)

SOURCE = CONNECTIVITY_DIR / "source" / "khl_source_mask.tif"

COST_DISTANCE = CONNECTIVITY_DIR / "literature_weighted" / "cost_distance.tif"

CORRIDOR_POTENTIAL = CONNECTIVITY_DIR / "literature_weighted" / "corridor_potential.tif"

CURRENT_DENSITY = CONNECTIVITY_DIR / "circuit" / "current_density.tif"

CIRCUIT_CONNECTIVITY = CONNECTIVITY_DIR / "circuit" / "connectivity.tif"

FLOOD_CONNECTIVITY = CONNECTIVITY_DIR / "flood" / "flood_exposed_connectivity.tif"

FLOOD_CONNECTIVITY_LOSS = (
    CONNECTIVITY_DIR / "flood" / "flood_induced_connectivity_loss.tif"
)

OUTPUT_DIR = CONNECTIVITY_DIR / "synthesis"


def main() -> None:
    started = perf_counter()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    print("=" * 70)
    print("CONNECTIVITY SYNTHESIS")
    print(f"Started at: {timestamp}")
    print("=" * 70)

    synthesis = ConnectivitySynthesis(
        resistance_path=RESISTANCE,
        source_path=SOURCE,
        cost_distance_path=COST_DISTANCE,
        corridor_potential_path=CORRIDOR_POTENTIAL,
        current_density_path=CURRENT_DENSITY,
        circuit_connectivity_path=CIRCUIT_CONNECTIVITY,
        flood_connectivity_path=FLOOD_CONNECTIVITY,
        flood_connectivity_loss_path=FLOOD_CONNECTIVITY_LOSS,
        output_dir=OUTPUT_DIR,
    )

    summary_path = synthesis.run()

    print(f"\n[OUTPUT] {summary_path}")
    print(f"[TIME] Total workflow: {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
