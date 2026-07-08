"""
╔════════════════════════════════════════════════════════════════════════╗
║                  GEO-FORENSIC WATERSHED ANALYSIS PIPELINE              ║
║         Attribution Analysis: Deforestation Impact on Flood Event      ║
║                        Tangse-Meureudu DAS, Aceh                       ║
╚════════════════════════════════════════════════════════════════════════╝

Pipeline Stages:
  1. ROI Delineation    → Automatic multi-watershed extraction via HydroSHEDS
  2. Satellite Analysis → Pre/Post-flood vegetation degradation (P1)
  3. Hydrology Model    → SCS-CN runoff response simulation (P2)
  4. Upstream Terrain   → Critical slope deforestation mapping (P3)
  5. Causal Integration → Multi-source evidence synthesis (P4)
  6. Statistical Summary→ Forensic attribution metrics & reporting

Execution Time: ~15-30 min (depending on ROI size & GEE server load)
Output: JSON metrics + interactive map + geojson ROI
"""

import sys
import traceback
from datetime import datetime

from src.services.forensic_service import ForensicAnalysisService


def main():
    """
    Main orchestrator untuk seluruh analisis forensik spasial watershed.
    Menjalankan 6 stage pipeline dengan error handling & progress tracking.
    """

    print("\n" + "=" * 70)
    print("    INITIALIZING GEO-FORENSIC WATERSHED ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\n[•] Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ====================================================================
        # STAGE 0: Service Initialization & ROI Delineation
        # ====================================================================
        print("\n[STAGE 0] Initializing ForensicAnalysisService...")
        print("          └─ Delineating Multi-Watershed ROI (HydroSHEDS v1.12)")
        forensic_service = ForensicAnalysisService()
        print("          └─ ✓ Service initialized, ROI loaded")

        # ====================================================================
        # STAGE 1: Export ROI to GeoJSON for external tools (QGIS, ArcGIS)
        # ====================================================================
        print("\n[STAGE 1] Exporting ROI geometry to GeoJSON...")
        forensic_service.export_roi(filename="tangse_meureudu_roi.geojson")
        print("          └─ ✓ ROI exported (dapat dibuka di QGIS/ArcGIS)")

        # ====================================================================
        # STAGE 2: Run Analysis Pipelines (P1, P2, P3, P4)
        # ====================================================================
        print("\n[STAGE 2] Running geospatial analysis pipelines...")
        print("          ├─ P1: Satellite vegetation degradation analysis")
        print("          ├─ P2: Hydrological runoff simulation (SCS-CN)")
        print("          ├─ P3: Upstream critical slope deforestation")
        print("          └─ P4: Spatial causal integration")

        pipelines = forensic_service.run_analysis_pipelines()
        print("          └─ ✓ All pipelines executed successfully")

        # ====================================================================
        # STAGE 3: Execute Server-Side Geospatial Reduction
        # ====================================================================
        print("\n[STAGE 3] Executing server-side GEE reduction...")
        print("          └─ Batching: mean, median, max, sum reducers...")

        raw_stats = forensic_service.execute_geospatial_reduction(pipelines)
        print("          └─ ✓ Raw statistics retrieved from GEE")

        # ====================================================================
        # STAGE 4: Parse & Validate Metrics
        # ====================================================================
        print("\n[STAGE 4] Parsing and validating forensic metrics...")
        print("          ├─ Topography: elevation, slope")
        print("          ├─ Land cover: forest loss, degradation")
        print("          ├─ Vegetation: NDVI/NDMI dynamics")
        print("          └─ Hydrology: runoff simulation & attribution")

        metrics = forensic_service.parse_and_validate_metrics(raw_stats)
        print("          └─ ✓ All metrics validated")

        # ====================================================================
        # STAGE 5: Generate & Display Report
        # ====================================================================
        print("\n[STAGE 5] Generating forensic attribution report...")
        forensic_service.display_report(metrics, raw_stats)
        print("          └─ ✓ Report displayed to console")

        # ====================================================================
        # STAGE 6: Save Metrics Payload to JSON
        # ====================================================================
        print("\n[STAGE 6] Saving metrics payload to JSON...")
        forensic_service.save_metrics_payload(metrics, raw_stats)
        print("          └─ ✓ Metrics saved (check data/output_metrics/)")

        # ====================================================================
        # COMPLETION
        # ====================================================================
        print("\n" + "=" * 70)
        print("    ✓ GEO-FORENSIC ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(
            f"\n[•] Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("\n[OUTPUT FILES]")
        print("  • data/output_metrics/tangse_meureudu_roi.geojson")
        print("  • data/output_metrics/forensic_map.html")
        print("  • data/output_metrics/tangse_meureudu_metrics_*.json")
        print("\n[NEXT STEPS]")
        print("  1. Open GeoJSON in QGIS to inspect ROI boundary")
        print("  2. Review HTML interactive map for visual validation")
        print("  3. Parse JSON metrics for forensic report generation")
        print("=" * 70 + "\n")

        return 0  # Success exit code

    except ValueError as e:
        print(f"\n[ERROR] Validation Error: {str(e)}")
        print("        └─ Check if all required GEE bands are present")
        traceback.print_exc()
        return 1

    except RuntimeError as e:
        print(f"\n[ERROR] Runtime Error: {str(e)}")
        print("        └─ Check GEE initialization & network connection")
        traceback.print_exc()
        return 1

    except KeyError as e:
        print(f"\n[ERROR] Key Error (missing metric): {str(e)}")
        print("        └─ Check raw_stats structure from GEE reduction")
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected Error: {str(e)}")
        print("        └─ Full traceback below:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
