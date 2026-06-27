from datetime import datetime

from src.core.engine import GEEEngine
from src.services.topography_service import TopographyService


def main():

    print("[STAGE 0] Loading ROI...")

    roi = GEEEngine().get_hydro_roi()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = f"data/output_topography/{timestamp}"

    topo = TopographyService(roi=roi, output_dir=output_dir)

    outputs = topo.run(contour_interval=25)

    print("\nGenerated Files:")

    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
