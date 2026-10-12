from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.corridor.connectivity_evidence import ConnectivityEvidence

ROOT = Path(__file__).resolve().parents[2]

CONNECTIVITY = ROOT / "data/output_rasters/corridor/connectivity"
EVIDENCE_DIR = CONNECTIVITY / "evidence"

SCENARIOS = (65.0, 75.0, 90.0)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    scenario_results = {}

    for percentile in SCENARIOS:
        scenario_name = f"p{int(percentile)}"
        output_dir = EVIDENCE_DIR / scenario_name

        print("\n" + "=" * 70)
        print(f"RUNNING EVIDENCE SCENARIO: {scenario_name}")
        print("=" * 70)

        analyzer = ConnectivityEvidence(
            corridor_potential_path=(
                CONNECTIVITY / "literature_weighted/corridor_potential.tif"
            ),
            circuit_connectivity_path=(CONNECTIVITY / "circuit/connectivity.tif"),
            current_density_path=(CONNECTIVITY / "circuit/current_density.tif"),
            source_mask_path=(CONNECTIVITY / "source/khl_source_mask.tif"),
            output_dir=output_dir,
            percentile=percentile,
        )

        summary_path = analyzer.run()

        with summary_path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        scenario_results[scenario_name] = {
            "percentile": percentile,
            "thresholds": result["thresholds"],
            "counts": result["counts"],
            "overlap_fraction_of_analysis_domain": (
                result["overlap_fraction_of_analysis_domain"]
            ),
            "correlation": result["correlation"],
            "outputs": result["outputs"],
            "summary": str(summary_path),
        }

    sensitivity_summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenarios": list(SCENARIOS),
        "method": (
            "Sensitivity analysis using percentile thresholds "
            "65, 75, and 90. Input rasters, source mask, alignment "
            "method, and analysis grid are held constant."
        ),
        "results": scenario_results,
    }

    master_summary_path = EVIDENCE_DIR / "threshold_sensitivity.json"

    with master_summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            sensitivity_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY ANALYSIS COMPLETED")
    print("=" * 70)

    for name, result in scenario_results.items():
        counts = result["counts"]

        print(f"\n[{name}]")
        print(f"  Analysis cells : {counts['analysis_cells']:,}")
        print(f"  Both high      : {counts['both_high_cells']:,}")
        print(f"  Class 0        : {counts['neither_high_cells']:,}")
        print(
            "  Class 0 (%)    : "
            f"{100 * counts['neither_high_cells'] / counts['analysis_cells']:.2f}%"
        )

    print(f"\n[OUTPUT] {master_summary_path}")


if __name__ == "__main__":
    main()
