from pathlib import Path

from src.corridor.resistance import CorridorResistance


ROOT = Path(__file__).resolve().parent

NORMALIZED_DIR = ROOT / "data" / "output_rasters" / "corridor" / "normalized"

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"


def main():
    resistance = CorridorResistance(
        normalized_dir=NORMALIZED_DIR,
        output_dir=RESISTANCE_DIR,
    )

    resistance.run()


if __name__ == "__main__":
    main()
