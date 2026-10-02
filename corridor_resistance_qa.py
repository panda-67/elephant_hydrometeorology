from pathlib import Path

from src.corridor.resistance_qa import ResistanceQA


ROOT = Path(__file__).resolve().parent

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"


def main():
    qa = ResistanceQA(
        resistance_dir=RESISTANCE_DIR,
    )

    qa.run()


if __name__ == "__main__":
    main()
