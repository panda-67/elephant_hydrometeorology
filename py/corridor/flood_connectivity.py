from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.corridor.flood_connectivity import FloodConnectivity
from config import config


def main() -> None:

    base_dir = ROOT_DIR / "data" / "output_rasters"

    baseline_resistance = (
        base_dir
        / "corridor"
        / "resistance"
        / "composite_resistance_literature_weighted.tif"
    )

    causal_matrix = (
        base_dir / "forensic" / "p4_spatial_causal_matrix_20260908_220928.tif"
    )

    causal_tier_band = 7

    connectivity_dir = base_dir / "corridor" / "connectivity" / "literature_weighted"

    baseline_cost_distance = connectivity_dir / "cost_distance.tif"

    baseline_connectivity = connectivity_dir / "corridor_potential.tif"

    source_mask = (
        base_dir / "corridor" / "connectivity" / "source" / "khl_source_mask.tif"
    )

    output_dir = base_dir / "corridor" / "connectivity" / "flood"

    analysis = FloodConnectivity(
        baseline_resistance=baseline_resistance,
        causal_matrix=causal_matrix,
        causal_tier_band=causal_tier_band,
        baseline_cost_distance=baseline_cost_distance,
        baseline_connectivity=baseline_connectivity,
        source_mask=source_mask,
        output_dir=output_dir,
        causal_resistance_factors=(config.CAUSAL_RESISTANCE_FACTORS),
    )

    outputs = analysis.run()

    print()
    print("Outputs:")

    for name, path in outputs.items():
        print(f"  {name:<35} {path}")


if __name__ == "__main__":
    main()
