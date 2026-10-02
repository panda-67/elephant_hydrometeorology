import ee


class CorridorWater:
    """
    Water-proximity predictor for elephant corridor analysis.

    Water source:
        WWF HydroSHEDS Free Flowing Rivers Network v1
    """

    DATASET = "WWF/HydroSHEDS/v1/FreeFlowingRivers"

    def __init__(self, roi):
        self.roi = roi

    def get_river_network(self):
        rivers = ee.FeatureCollection(self.DATASET).filterBounds(self.roi)

        return rivers

    def get_water_mask(self):
        rivers = self.get_river_network()

        river_count = rivers.size().getInfo()
        print(f"    HydroSHEDS river features : {river_count}")

        if river_count == 0:
            raise ValueError(
                "No HydroSHEDS river features found inside the corridor AOI."
            )

        river_mask = (
            ee.Image()
            .byte()
            .paint(rivers, 1)
            .rename("water_mask")
            .unmask(0)
            .clip(self.roi)
        )

        return river_mask

    def get_distance_to_water(self):
        print("\n[8] Generating distance-to-water...")

        water_mask = self.get_water_mask()

        # 2.5 km radius:
        # at 10 m scale this remains below Earth Engine's
        # maximum kernel dimension.
        distance = (
            water_mask.distance(kernel=ee.Kernel.euclidean(radius=2500, units="meters"))
            .rename("distance_to_water")
            .toFloat()
            .clip(self.roi)
        )

        return distance
