import ee


class SpatialCausalPipeline:
    def __init__(self, satellite_img: ee.Image, hydrology_img: ee.Image):
        self.satellite = satellite_img
        self.hydrology = hydrology_img

    def execute(self) -> ee.Image:
        # 1. Ambil band eksis dari P1
        d_degradation = self.satellite.select("d_NDVI_degradation")
        d_destruction = self.satellite.select("d_NDVI_destruction")

        # 2. Tambahkan kalkulasi perubahan bersih absolut (Net Loss)
        ndvi_net_loss = d_degradation.add(d_destruction).rename("ndvi_net_loss")

        # 3. Tambahkan kalkulasi pergeseran tipe gangguan (Kritis untuk klasifikasi penyebab)
        # Degradasi dikurang destruksi
        disturbance_shift = d_degradation.subtract(d_destruction).rename(
            "disturbance_shift"
        )

        # 4. Buat Maker Dinamis: Hanya pixel yang memiliki perubahan signifikan di salah satu fase
        # 4.1. Zona Degradasi Hulu (Khusus penurunan vegetasi pra-bencana)
        degradation_loss = d_degradation.lt(-0.1)

        # 4.2. Zona Destruksi Bencana (Khusus hantaman fisik saat bencana)
        destruction_loss = d_destruction.lt(-0.1)

        # 4.3. Masker Perubahan Signifikan Gabungan (Hanya area yang BENAR-BENAR RUSAK)
        significant_change_mask = degradation_loss.Or(destruction_loss).rename(
            "significant_change_mask"
        )

        # BONUS FORENSIK: Zona Krisis Utama (Sudah didegradasi manusia, dihantam bencana pula)
        compound_hotspot = degradation_loss.And(destruction_loss).rename(
            "compound_hotspot"
        )

        # Katastrofe Terkonsentrasi (Mengisolasi wilayah yang terdegradasi SEKALIGUS hancur saat bencana)
        compound_hotspot = (d_degradation.lt(0).And(d_destruction.lt(0))).rename(
            "compound_hotspot"
        )

        # 5. Gabungkan ke dalam matriks kausalitas baru
        extended_matrix = ee.Image.cat(
            [
                d_degradation,  # CAUSE A: Akumulasi tekanan hulu
                d_destruction,  # EFFECT B: Hantaman fisik bencana
                ndvi_net_loss,  # TOTAL IMPACT
                disturbance_shift,  # PATTERN IDENTIFIER (Manusia vs Alam)
                significant_change_mask,  # RENDER FILTER FOR QGIS
                compound_hotspot,  # CORE FORENSIC ZONE
            ]
        )

        return extended_matrix

    # def execute(self) -> ee.Image:
    #     """Mengintegrasikan Akumulasi Penyebab (Pre) dengan Dampak Fisik (Flood & Post)."""
    #     causal_matrix = ee.Image.cat(
    #         [
    #             self.satellite.select(
    #                 "d_NDVI_degradation"
    #             ),  # CAUSE: Akumulasi degradasi lahan hulu
    #             self.hydrology.select(
    #                 "runoff_net_increase"
    #             ),  # EFFECT 1: Lonjakan air permukaan badai
    #             self.satellite.select(
    #                 "d_NDVI_destruction"
    #             ),  # EFFECT 2: Kerusakan vegetasi hilir pasca banjir
    #         ]
    #     )
    #     return causal_matrix.rename(
    #         ["cause_degradation", "effect_runoff_spike", "effect_post_destruction"]
    #     )
