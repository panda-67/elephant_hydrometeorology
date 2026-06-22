import ee
from src.core.terrain import TerrainAnalyzer
from src.core.landcover import LandCoverAnalyzer
from config import config


class MeureuduUpstreamPipeline:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.ta = TerrainAnalyzer(roi)
        self.lca = LandCoverAnalyzer(roi)

    def execute(self) -> ee.Image:
        slope = self.ta.get_slope()

        lc_2020_raw = self.lca.get_worldcover_2020()
        lc_2025_raw = self.lca.get_dynamic_world(
            config.F_PRE_EVENT_START, config.F_PRE_EVENT_END
        )  # Rentang waktu Pre-Event

        forest_2020 = self.lca.get_forest_mask(lc_2020_raw, source="worldcover")
        forest_2025 = self.lca.get_forest_mask(lc_2025_raw, source="dynamic_world")

        # TRANSFORMASI 1: Hitung Luas Forest 2020 dalam Hektar (ha)
        forest_cover_2020_ha = (
            ee.Image.pixelArea()
            .multiply(0.0001)
            .updateMask(forest_2020)
            .rename("forest_cover_2020")
        )

        # Logic: Ada di rona awal (2020) DAN tidak ada di masa pra-bencana (2025)
        loss_binary_mask = self.lca.get_forest_loss_mask(forest_2020, forest_2025)

        # TRANSFORMASI 2: Ubah Masker Kehilangan Hutan Menjadi Satuan Hektar (ha)
        loss_preevent_ha = (
            ee.Image.pixelArea()
            .multiply(0.0001)
            .updateMask(loss_binary_mask)
            .rename("forest_loss_preevent")
        )

        critical_clipping = loss_preevent_ha.updateMask(slope.gt(15)).rename(
            "critical_upstream_deforestation"
        )

        return ee.Image.cat([forest_cover_2020_ha, loss_preevent_ha, critical_clipping])
