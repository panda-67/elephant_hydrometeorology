import ee


class SpatialCausalPipeline:
    def __init__(
        self,
        satellite_img: ee.Image,
        hydrology_img: ee.Image,
        roi: ee.Geometry,
    ):
        self.satellite = satellite_img
        self.hydrology = hydrology_img
        self.roi = roi

    def execute(self) -> ee.Image:
        # ============================================================
        # 1. BAND DASAR DARI P1 (tidak berubah)
        # ============================================================
        d_degradation = self.satellite.select("d_NDVI_degradation")
        d_destruction = self.satellite.select("d_NDVI_destruction")

        ndvi_net_loss = d_degradation.add(d_destruction).rename("ndvi_net_loss")
        disturbance_shift = d_degradation.subtract(d_destruction).rename(
            "disturbance_shift"
        )

        degradation_loss = d_degradation.lt(-0.1)
        destruction_loss = d_destruction.lt(-0.1)

        significant_change_mask = degradation_loss.Or(destruction_loss).rename(
            "significant_change_mask"
        )
        compound_hotspot = (d_degradation.lt(0).And(d_destruction.lt(0))).rename(
            "compound_hotspot"
        )

        # ============================================================
        # 2. BUKTI HIDROLOGIS DARI P2 — resample dulu ke resolusi P1
        #    (hydrology_img biasanya di grid 30m, satellite di 10m —
        #    tanpa reproject, band akan mismatch pixel grid saat di-cat)
        # ============================================================
        TARGET_CRS = "EPSG:32646"  # UTM 46N, cocok untuk Aceh — sesuaikan kalau ROI Anda beda zona
        TARGET_SCALE = 10
        runoff_increase = (
            self.hydrology.select("runoff_net_increase")
            .resample("bilinear")
            .reproject(crs=TARGET_CRS, scale=TARGET_SCALE)
            .rename("runoff_increase_resampled")
        )

        # Threshold DINAMIS: percentile ke-75 dari kenaikan limpasan di ROI,
        # bukan angka mm hardcode — konsisten dengan pendekatan P1
        runoff_stats = runoff_increase.reduceRegion(
            reducer=ee.Reducer.percentile([75]),
            geometry=self.roi if hasattr(self, "roi") else d_degradation.geometry(),
            scale=30,
            maxPixels=1e13,
        )

        runoff_p75 = ee.Number(runoff_stats.values().get(0))
        significant_runoff_spike = runoff_increase.gt(runoff_p75)

        # ============================================================
        # 3. SKOR ATRIBUSI KAUSAL BERTINGKAT (0–4)
        #    Semakin tinggi nilai, semakin kuat bukti gabungan
        #    vegetasi + hidrologi di piksel tsb.
        # ============================================================
        causal_tier = (
            ee.Image(0)
            .where(degradation_loss, 1)  # degradasi vegetasi saja (dugaan antropogenik)
            .where(destruction_loss, 2)  # destruksi fisik saat bencana saja
            .where(compound_hotspot, 3)  # degradasi + destruksi (zona krisis)
            .where(
                compound_hotspot.And(significant_runoff_spike), 4
            )  # tier tertinggi: NDVI + hidrologi SAMA-SAMA mengkonfirmasi
            .rename("causal_evidence_tier")
        )

        # Flag khusus buat laporan: piksel di zona kritis yang
        # dikonfirmasi independen oleh limpasan (bukti paling kuat
        # untuk klaim kausal deforestasi → banjir)
        hydrologically_confirmed_hotspot = (
            compound_hotspot.And(significant_runoff_spike)
        ).rename("hydro_confirmed_hotspot")

        # ============================================================
        # 4. GABUNGKAN SEMUA
        # ============================================================
        extended_matrix = ee.Image.cat(
            [
                d_degradation,
                d_destruction,
                ndvi_net_loss,
                disturbance_shift,
                significant_change_mask,
                compound_hotspot,
                runoff_increase,
                causal_tier,
                hydrologically_confirmed_hotspot,
            ]
        )
        return extended_matrix
