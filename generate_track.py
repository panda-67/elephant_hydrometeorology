from datetime import datetime
import os

from src.core.engine import GEEEngine
from src.services.transect_service import TransectGenerator


def main():
    print("[STAGE 0] Reconstituting GEEEngine() untuk P5...")

    roi = GEEEngine().get_hydro_roi()

    least_cost = True

    print("[✓] GEE Terhubung & ROI Berhasil Dimuat.")

    print("[STAGE 1] Inisialisasi Transek Service menggunakan ROI dari Service Utama")

    generator = TransectGenerator(roi)

    if least_cost:
        cost_surface = generator.least_cost_surface(
            optimum_elevation=250.0,
            maximum_slope=25.0,
            aspect_ranges=[
                (45, 135),  # Timur
                (135, 225),  # Selatan
            ],
            elevation_weight=0.4,
            slope_weight=0.4,
            aspect_weight=0.2,
        )

        os.makedirs("data/output_rasters", exist_ok=True)

        generator.export_cost_surface(
            cost_surface,
            "data/output_rasters/cost_surface.tif",
        )

    transects = generator.run(
        least_cost=least_cost,
        cost_raster="data/output_rasters/cost_surface.tif",
        min_elevation=100,
        max_elevation=300,
        min_slope=5,
        max_slope=20,
        min_area_ha=40,
        transect_length_m=8000,
        aspect_ranges=[
            (45, 135),  # Timur
            (135, 225),  # Selatan
        ],
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
