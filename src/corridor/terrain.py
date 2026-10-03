import ee


class CorridorTerrain:
    """Generator predictor terrain untuk analisis koridor gajah."""

    DEM_ASSET = "users/nandadata02/DEMNAS-ACEH"

    def __init__(self, roi: ee.Geometry):
        self.roi = roi

        self.dem = (
            ee.Image(self.DEM_ASSET)
            .select("b1")
            .rename("elevation")
            .toFloat()
            .clip(self.roi)
        )

    def get_elevation(self) -> ee.Image:
        """Elevation dari DEMNAS."""

        print("\n[4] Generating elevation...")
        print(f"    Source : {self.DEM_ASSET}")
        print("    Native scale : ~8.35 m")
        print("    Output scale : 10 m")

        return self.dem

    def get_slope(self) -> ee.Image:
        """Slope dalam derajat dari DEMNAS."""

        print("\n[5] Generating slope...")
        print(f"    Source : {self.DEM_ASSET}")
        print("    Output scale : 10 m")

        slope = ee.Terrain.slope(self.dem)

        return slope.rename("slope").toFloat().clip(self.roi)
