from pathlib import Path

from src.corridor.corridor_potential import CorridorPotential


ROOT = Path(__file__).resolve().parents[2]

CONNECTIVITY_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity"

SOURCE_MASK = CONNECTIVITY_DIR / "source" / "khl_source_mask.tif"

COST_DISTANCE = CONNECTIVITY_DIR / "literature_weighted" / "cost_distance.tif"

OUTPUT_PATH = CONNECTIVITY_DIR / "literature_weighted" / "corridor_potential.tif"


def main():

    print("\n" + "#" * 70)
    print("# P9.3 — POTENTIAL CONNECTIVITY")
    print("#" * 70)

    print(f"\nSource mask       : {SOURCE_MASK}")
    print(f"Cost distance     : {COST_DISTANCE}")
    print(f"Output            : {OUTPUT_PATH}")

    model = CorridorPotential(
        cost_distance_raster=COST_DISTANCE,
        source_mask=SOURCE_MASK,
        output_path=OUTPUT_PATH,
    )

    model.run()


if __name__ == "__main__":
    main()
