from pathlib import Path

from src.corridor.composite_resistance import CompositeResistance


ROOT = Path(__file__).resolve().parent

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"

OUTPUT_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"


def main():
    composite = CompositeResistance(
        resistance_dir=RESISTANCE_DIR,
        output_dir=OUTPUT_DIR,
    )

    composite.run()


if __name__ == "__main__":
    main()
