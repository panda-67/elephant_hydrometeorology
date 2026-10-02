import ee

from src.core.landcover import LandCoverAnalyzer


class CorridorLandCover:
    """Generator predictor land cover untuk analisis koridor gajah."""

    def __init__(
        self,
        roi: ee.Geometry,
    ):
        self.roi = roi
        self.analyzer = LandCoverAnalyzer(roi)

    def get_worldcover(self) -> ee.Image:
        """ESA WorldCover 2020."""

        print("\n[6] Generating land cover...")

        landcover = (
            self.analyzer.get_worldcover_2020()
            .rename("landcover")
            .toInt16()
            .clip(self.roi)
        )

        return landcover
