import ee
from config import config


class VegetationAnalyzer:
    def __init__(self, roi: ee.Geometry, mode="sentinel2"):
        self.roi = roi
        self.mode = mode.lower()  # Opsi: "sentinel2", "sentinel1", atau "landsat"

    def get_collection(self, start_date, end_date):
        cloud_threshold = config.CLOUD_PROB_THRESHOLD

        # =====================================================================
        # OPTION 1: SENTINEL-2
        # =====================================================================
        if self.mode == "sentinel2":

            def advanced_mask(img):
                qa = img.select("QA60")
                cloud_bit_mask = 1 << 10
                cirrus_bit_mask = 1 << 11
                mask = (
                    qa.bitwiseAnd(cloud_bit_mask)
                    .eq(0)
                    .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                )
                spectral_bands = img.select(["B2", "B3", "B4", "B8", "B11", "B12"])
                scaled_bands = spectral_bands.divide(10000)
                return (
                    img.addBands(scaled_bands, overwrite=True)
                    .updateMask(mask)
                    .copyProperties(img, ["system:time_start"])
                )

            return (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(self.roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
                # .map(advanced_mask)  # Hilangkan komen jika ingin mengaktifkan masker
            )

        # =====================================================================
        # OPTION 2: LANDSAT (Landsat 8 & 9 Combined Collection 2 Level-2)
        # =====================================================================
        elif self.mode == "landsat":

            def mask_landsat_sr(img):
                # Bit 3 = Cloud, Bit 4 = Cloud Shadow bawaan QA_PIXEL
                qa = img.select("QA_PIXEL")
                cloud_shadow_bit_mask = 1 << 3
                cloud_bit_mask = 1 << 4
                mask = (
                    qa.bitwiseAnd(cloud_shadow_bit_mask)
                    .eq(0)
                    .And(qa.bitwiseAnd(cloud_bit_mask).eq(0))
                )

                # Landsat 8/9 SR perlu dikalikan sff (0.0000275) + offset (-0.2)
                # Namun untuk normalizedDifference (NDVI/NDMI), scaling ini opsional
                return img.updateMask(mask).copyProperties(img, ["system:time_start"])

            # Menggabungkan koleksi Landsat 8 dan 9 untuk kontinuitas data yang rapat
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(self.roi)
            l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(self.roi)
            merged_landsat = l8.merge(l9)

            return (
                merged_landsat.filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
                .map(mask_landsat_sr)
            )

        # =====================================================================
        # OPTION 3: SENTINEL-1 (RADAR)
        # =====================================================================
        else:
            return (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(self.roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            )

    def calculate_indices(self, img: ee.Image) -> ee.Image:
        # --- LOGIKA INDEKS SENTINEL-2 ---
        if self.mode == "sentinel2":
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
            return img.addBands([ndvi, ndmi])

        # --- LOGIKA INDEKS LANDSAT (Landsat 8 & 9) ---
        elif self.mode == "landsat":
            # Nir = SR_B5, Red = SR_B4, Swir1 = SR_B6
            ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
            ndmi = img.normalizedDifference(["SR_B5", "SR_B6"]).rename("NDMI")
            return img.addBands([ndvi, ndmi])

        # --- LOGIKA PROKSI INDEKS SENTINEL-1 (RADAR) ---
        else:
            vv_linear = ee.Image(10).pow(img.select("VV").divide(10))
            vh_linear = ee.Image(10).pow(img.select("VH").divide(10))
            vv_vh_ratio = vv_linear.divide(vh_linear)

            # Normalisasi Nilai VH (-23 dB s/d -10 dB) menjadi skala Proksi NDVI (0 s/d 1)
            radar_canopy_proxy = img.select("VH").clamp(-23.0, -10.0)
            radar_canopy_proxy = (
                radar_canopy_proxy.subtract(-23.0).divide(13.0).rename("NDVI")
            )

            # Normalisasi Rasio VV/VH (1.0 s/d 8.0) menjadi skala Proksi NDMI (0 s/d 1)
            radar_moisture_proxy = vv_vh_ratio.clamp(1.0, 8.0)
            radar_moisture_proxy = (
                ee.Image(1.0)
                .subtract(radar_moisture_proxy.subtract(1.0).divide(7.0))
                .rename("NDMI")
            )

            return img.addBands([radar_canopy_proxy, radar_moisture_proxy])

    @staticmethod
    def area_hectares(image_mask: ee.Image, roi: ee.Geometry) -> ee.Number:
        """Menghitung luasan area bermasker ke dalam satuan Hektar (ha)."""
        area_img = image_mask.multiply(ee.Image.pixelArea())
        total_area = area_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=30, maxPixels=1e13
        )
        return ee.Number(total_area.values().get(0)).divide(10000)
