from datetime import datetime
import os

from src.core.engine import GEEEngine
from src.services.transect_service import TransectGenerator


def main():
    print("[STAGE 0] Reconstituting GEEEngine() untuk P5...")

    roi = GEEEngine().get_hydro_roi()

    print("[✓] GEE Terhubung & ROI Berhasil Dimuat.")

    print("[STAGE 1] Inisialisasi Transek Service menggunakan ROI dari Service Utama")

    generator = TransectGenerator(roi)

    transects = generator.run(
        min_elevation=100,
        max_elevation=300,
        max_slope=20,
        min_area_ha=14,
        transect_length_m=8000,
    )

    print("Transects:", transects.size().getInfo())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = "_"
    filename = f"transect_{timestamp}.geojson"

    print(f"[STAGE 2] Export {filename} untuk dapat diperguankan di QGIS ...")
    output_dir = os.path.join("data", "output_vectors")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, filename)
    generator.export_geojson(transects, output_file)


if __name__ == "__main__":
    main()
