from pathlib import Path

from src.corridor.composite_resistance_qa import (
    CompositeResistanceQA,
)


ROOT = Path(__file__).resolve().parents[2]

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"


def main():
    qa = CompositeResistanceQA(
        resistance_dir=RESISTANCE_DIR,
    )

    qa.run()


if __name__ == "__main__":
    main()
