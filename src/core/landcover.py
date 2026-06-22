import ee


class LandCoverAnalyzer:
    """
    Mengevaluasi, mengekstraksi, dan menyiapkan data klasifikasi tutupan lahan
    makro regional menggunakan ESA WorldCover dan Google Dynamic World.
    """

    def __init__(self, roi: ee.Geometry):
        self.roi = roi

    def get_worldcover_2020(self) -> ee.Image:
        """ESA WorldCover 2020 baseline data provider."""
        return ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(self.roi)

    def get_dynamic_world(self, start_date: str, end_date: str) -> ee.Image:
        """Dynamic World mode composite provider untuk kondisi pre-flood atau lini masa kustom."""
        collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
        )
        return (
            collection.select("label")
            .reduce(ee.Reducer.mode())
            .rename("Map")
            .clip(self.roi)
        )

    def get_forest_mask(self, image: ee.Image, source: str) -> ee.Image:
        """
        Menghasilkan mask area hutan biner (1 = Hutan, 0 = Bukan Hutan).
        Menangani perbedaan kode kelas antar-sensor secara internal.
        """
        if source == "worldcover":
            # ESA WorldCover: Code 10 melambangkan Tree Cover
            return image.eq(10).rename("forest_mask")

        if source == "dynamic_world":
            # Dynamic World: Code 1 melambangkan Trees
            return image.eq(1).rename("forest_mask")

        raise ValueError(
            f"Unknown data source: {source}. Gunakan 'worldcover' atau 'dynamic_world'."
        )

    def get_nonforest_mask(self, image: ee.Image, source: str) -> ee.Image:
        """Menghasilkan mask area bukan hutan biner."""
        return self.get_forest_mask(image, source).Not().rename("non_forest_mask")

    def get_forest_loss_mask(
        self, forest_mask_before: ee.Image, forest_mask_after: ee.Image
    ) -> ee.Image:
        """
        Mask biner kehilangan hutan (1 = ada hutan di citra 'before' DAN
        sudah tidak ada hutan lagi di citra 'after').
        """
        return forest_mask_before.And(forest_mask_after.Not()).rename(
            "forest_loss_mask"
        )

    def area_image(self, mask: ee.Image) -> ee.Image:
        """
        Helper spasial untuk mengubah mask biner (0 atau 1) menjadi
        citra bobot luas dalam satuan Hektar (ha) per piksel.
        """
        return mask.selfMask().multiply(ee.Image.pixelArea()).divide(10000)
