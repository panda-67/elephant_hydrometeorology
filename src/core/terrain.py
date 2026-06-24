import ee
from config import config


class TerrainAnalyzer:
    """Analisis topografi berbasis DEM."""

    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.use_demnas = getattr(config, "USE_DEMNAS", False)

        if self.use_demnas:
            self.dem = (
                ee.Image("users/nandadata02/DEMNAS-ACEH")
                .clip(self.roi)
                .select(["b1"], ["elevation"])
            )
        else:
            col = (
                ee.ImageCollection("COPERNICUS/DEM/GLO30")
                .filterBounds(self.roi)
                .select("DEM")
            )

            reference_projection = ee.Image(col.first()).projection()

            self.dem = (
                col.mosaic()
                .setDefaultProjection(reference_projection)
                .rename("elevation")
            )

    def get_dem(self) -> ee.Image:
        """Elevation (meter)."""
        return self.dem.clip(self.roi)

    def get_slope(self) -> ee.Image:
        """Slope (derajat)."""
        return ee.Terrain.slope(self.dem).rename("Slope").clip(self.roi)

    def get_aspect(self) -> ee.Image:
        """Aspect (0–360°)."""
        return ee.Terrain.aspect(self.dem).rename("Aspect").clip(self.roi)

    def get_hillshade(self) -> ee.Image:
        """Hillshade."""
        return ee.Terrain.hillshade(self.dem).rename("Hillshade").clip(self.roi)

    def analyze_morfometry(self) -> ee.Image:
        """Gabungan elevasi dan slope."""
        return ee.Image.cat(
            [
                self.get_dem(),
                self.get_slope(),
            ]
        )
