from pathlib import Path

from src.corridor.spatial_analysis import CorridorSpatialAnalysis


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    evidence_root = (
        root / "data" / "output_rasters" / "corridor" / "connectivity" / "evidence"
    )

    current_density = (
        root
        / "data"
        / "output_rasters"
        / "corridor"
        / "connectivity"
        / "circuit"
        / "current_density.tif"
    )

    output_dir = (
        root
        / "data"
        / "output_rasters"
        / "corridor"
        / "connectivity"
        / "synthesis"
        / "spatial"
    )

    analysis = CorridorSpatialAnalysis(
        evidence_root=evidence_root,
        current_density_path=current_density,
        output_dir=output_dir,
        percentiles=(65, 75, 90),
        minimum_component_cells=5,
    )

    analysis.run()


if __name__ == "__main__":
    main()
