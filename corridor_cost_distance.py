from pathlib import Path

from src.corridor.cost_distance import SourceBasedCostDistance


ROOT = Path(__file__).resolve().parent

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"

SOURCE_MASK = (
    ROOT
    / "data"
    / "output_rasters"
    / "corridor"
    / "connectivity"
    / "source"
    / "khl_source_mask.tif"
)

CONNECTIVITY_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity"


SCENARIOS = {
    "equal_weight": {
        "resistance": "composite_resistance_equal_weight.tif",
        "output": "cost_distance.tif",
    },
    "literature_weighted": {
        "resistance": "composite_resistance_literature_weighted.tif",
        "output": "cost_distance.tif",
    },
}


def main():
    for scenario, config in SCENARIOS.items():
        print("\n")
        print("#" * 70)
        print(f"# SCENARIO: {scenario}")
        print("#" * 70)

        resistance = RESISTANCE_DIR / config["resistance"]

        output_dir = CONNECTIVITY_DIR / scenario
        output_path = output_dir / config["output"]

        model = SourceBasedCostDistance(
            resistance_raster=resistance,
            source_mask=SOURCE_MASK,
            output_path=output_path,
        )

        model.run()


if __name__ == "__main__":
    main()
