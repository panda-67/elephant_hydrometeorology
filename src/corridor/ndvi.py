import ee

from config import config
from src.core.vegetation import VegetationAnalyzer


class CorridorNDVI:
    """Generator predictor NDVI untuk analisis koridor gajah."""

    def __init__(
        self,
        roi: ee.Geometry,
        start_date: str = "2025-01-01",
        end_date: str = "2026-01-01",
    ):
        self.roi = roi
        self.start_date = start_date
        self.end_date = end_date

        self.analyzer = VegetationAnalyzer(
            roi=self.roi,
            mode=config.SATELLITE_MODE,
        )

    def get_ndvi(self) -> ee.Image:
        """Menghasilkan median NDVI pada periode analisis."""

        print("\n[7] Generating NDVI...")
        print(f"    Satellite : {config.SATELLITE_MODE}")
        print(f"    Period    : {self.start_date} → {self.end_date}")

        collection = self.analyzer.get_collection(
            self.start_date,
            self.end_date,
        )

        collection = collection.map(self.analyzer.calculate_indices)

        ndvi = (
            collection.select("NDVI").median().rename("ndvi").toFloat().clip(self.roi)
        )

        return ndvi
