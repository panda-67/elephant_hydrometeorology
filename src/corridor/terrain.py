import ee

from src.core.terrain import TerrainAnalyzer


class CorridorTerrain:
    """Generator predictor terrain untuk analisis koridor gajah."""

    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.analyzer = TerrainAnalyzer(roi)

    def get_elevation(self) -> ee.Image:
        """Elevation dalam meter."""

        print("\n[4] Generating elevation...")

        return self.analyzer.get_dem().rename("elevation").toFloat().clip(self.roi)

    def get_slope(self) -> ee.Image:
        """Slope dalam derajat."""

        print("\n[5] Generating slope...")

        return self.analyzer.get_slope().rename("slope").toFloat().clip(self.roi)
