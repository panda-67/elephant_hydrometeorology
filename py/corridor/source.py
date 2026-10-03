from pathlib import Path

from src.corridor.source import CorridorSource


ROOT = Path(__file__).resolve().parents[2]

SOURCE_VECTOR = ROOT / "data" / "output_vectors" / "KHL_PP_tangse_meureudu.geojson"

RESISTANCE_DIR = ROOT / "data" / "output_rasters" / "corridor" / "resistance"

OUTPUT_DIR = ROOT / "data" / "output_rasters" / "corridor" / "connectivity" / "source"


def main():
    source = CorridorSource(
        source_vector=SOURCE_VECTOR,
        resistance_dir=RESISTANCE_DIR,
        output_dir=OUTPUT_DIR,
    )

    source.create_source_mask()


if __name__ == "__main__":
    main()
