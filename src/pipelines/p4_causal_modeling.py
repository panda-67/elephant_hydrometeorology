import ee


class SpatialCausalPipeline:
    def __init__(self, satellite_img: ee.Image, hydrology_img: ee.Image):
        self.satellite = satellite_img
        self.hydrology = hydrology_img

    def execute(self) -> ee.Image:
        """Mengintegrasikan Akumulasi Penyebab (Pre) dengan Dampak Fisik (Flood & Post)."""
        causal_matrix = ee.Image.cat(
            [
                self.satellite.select(
                    "d_NDVI_degradation"
                ),  # CAUSE: Akumulasi degradasi lahan hulu
                self.hydrology.select(
                    "runoff_net_increase"
                ),  # EFFECT 1: Lonjakan air permukaan badai
                self.satellite.select(
                    "d_NDVI_destruction"
                ),  # EFFECT 2: Kerusakan vegetasi hilir pasca banjir
            ]
        )
        return causal_matrix.rename(
            ["cause_degradation", "effect_runoff_spike", "effect_post_destruction"]
        )
