from pathlib import Path

from src.corridor.corridor_potential import CorridorPotential


ROOT = Path(__file__).resolve().parent

CONNECTIVITY_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity"

SOURCE_MASK = CONNECTIVITY_DIR / "source" / "khl_source_mask.tif"


SCENARIOS = {
    "equal_weight": (CONNECTIVITY_DIR / "equal_weight" / "cost_distance.tif"),
    "literature_weighted": (
        CONNECTIVITY_DIR / "literature_weighted" / "cost_distance.tif"
    ),
}


def main():
    for scenario, cost_distance in SCENARIOS.items():
        print("\n")
        print("#" * 70)
        print(f"# SCENARIO: {scenario}")
        print("#" * 70)

        output_path = CONNECTIVITY_DIR / scenario / "corridor_potential.tif"

        model = CorridorPotential(
            cost_distance_raster=cost_distance,
            source_mask=SOURCE_MASK,
            output_path=output_path,
        )

        model.run()


if __name__ == "__main__":
    main()
