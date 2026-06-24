# export_assets.py
import sys
import traceback
from datetime import datetime
import ee

from src.services.export_service import ExportAssetsService
from src.services.forensic_service import ForensicAnalysisService


def main():
    """
    Dedicated entry point untuk trigger asynchronous raster export ke Google Drive.
    Memisahkan komputasi raster yang berat dari script metrics main.py.
    """
    print("\n" + "=" * 70)
    # Menandakan inisialisasi pipeline ekspor aset spasial
    print("      INITIALIZING GEO-FORENSIC RASTER ASSET EXPORT PIPELINE")
    print("=" * 70)
    print(
        f"\n[•] Export pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        # Panggil service utama (akan menjalankan inisialisasi GEE di __init__)
        print("\n[STAGE 0] Reconstituting ForensicAnalysisService...")
        forensic_service = ForensicAnalysisService()
        asset_service = ExportAssetsService(forensic_service.roi)

        # Bangun pipeline GEE (mengembalikan dict berisi objek ee.Image P1, P2, P3, P4)
        print("\n[STAGE 1] Resolving server-side graph for analysis pipelines...")
        pipelines = forensic_service.run_analysis_pipelines()

        # Trigger pengiriman task ekspor
        print("\n[STAGE 2] Submitting raster layers...")
        task_ids = asset_service.export_forensic_rasters(
            pipelines, scale_s2=10, scale_terrain=30
        )

        # === TRACKING JIKA ADA TASK DRIVE YANG AKTIF ===
        if task_ids:
            import time

            print(
                "\n[STAGE 3] Holding terminal for active Google Drive tasks (Ctrl+C to skip)..."
            )
            while True:
                statuses = [ee.data.getTaskStatus(t_id)[0] for t_id in task_ids]
                states = [status["state"] for status in statuses]
                done_count = sum(1 for s in states if s in ["COMPLETED", "FAILED"])

                print(
                    f"[Polling] Progress: {done_count}/{len(task_ids)} tasks finished. Current states: {states}",
                    end="\r",
                )

                if done_count == len(task_ids):
                    print("\n\n[✓] GEE Server finished processing fallback tasks!")
                    for status in statuses:
                        if status["state"] == "FAILED":
                            print(
                                f"  ❌ {status['description']} GAGAL: {status.get('error_message')}"
                            )
                        else:
                            print(
                                f"  ✅ {status['description']} SUKSES ditulis ke Google Drive."
                            )
                    break
                time.sleep(20)
        else:
            print(
                "\n[STAGE 3] Skip Polling: Semua asset langsung diunduh ke folder lokal lokal 'data/output_raster/'."
            )

        print("\n" + "=" * 70)
        print("  ✓ RASTER ASSET EXPORT PROCESS COMPLETED")
        print("=" * 70)

        print("\n[PRO-TIPS FOR QGIS INTERACTION]")
        print(
            "  1. Buka Google Drive Anda dan tunggu beberapa menit hingga proses render selesai."
        )
        print("  2. Download berkas GeoTIFF multi-band yang dihasilkan.")
        print("  3. Drag-and-drop file .tif ke QGIS bersamaan dengan file")
        print("     'data/output_metrics/tangse_meureudu_roi.geojson' hasil main.py.")
        print("  4. Gunakan Symbology -> 'Singleband pseudocolor' di QGIS untuk render")
        print(
            "     layer 'runoff_net_increase' atau 'critical_upstream_deforestation'."
        )
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Export pipeline failed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
