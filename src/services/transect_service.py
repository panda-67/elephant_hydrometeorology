import requests
from skimage.graph import route_through_array
import rasterio
from shapely.geometry import shape, LineString
import ee


class TransectGenerator:
    """
    Generate candidate field transects based on
    elevation and slope constraints.
    """

    def __init__(
        self,
        roi: ee.Geometry,
        dem: ee.Image = None,
    ):
        self.roi = roi

        self.dem = (dem if dem else ee.Image("USGS/SRTMGL1_003")).clip(roi)

        self.slope = ee.Terrain.slope(self.dem)

        self.aspect = ee.Terrain.aspect(self.dem)

    def suitable_area(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        min_slope: float = 0,
        max_slope: float = 20,
        aspect_ranges: list[tuple[int, int]] | None = None,
    ) -> ee.Image:
        """
        Create suitability mask.
        """

        elevation_mask = self.dem.gte(min_elevation).And(self.dem.lte(max_elevation))

        slope_mask = self.slope.gte(min_slope).And(self.slope.lte(max_slope))

        mask = elevation_mask.And(slope_mask)

        if aspect_ranges:
            aspect_mask = ee.Image.constant(0)

            for amin, amax in aspect_ranges:
                aspect_mask = aspect_mask.Or(
                    self.aspect.gte(amin).And(self.aspect.lte(amax))
                )

            mask = mask.And(aspect_mask)

        return mask.selfMask()

    def candidate_polygons(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        min_slope: float = 0,
        max_slope: float = 20,
        min_area_ha: float = 10,
        aspect_ranges: list[tuple[int, int]] | None = None,
        scale: int = 30,
    ) -> ee.FeatureCollection:
        """
        Convert suitable area to polygons.
        """

        mask = self.suitable_area(
            min_elevation,
            max_elevation,
            min_slope,
            max_slope,
            aspect_ranges,
        )

        polygons = mask.reduceToVectors(
            geometry=self.roi,
            scale=scale,
            geometryType="polygon",
            reducer=ee.Reducer.countEvery(),
            maxPixels=1e13,
        )

        polygons = polygons.map(
            lambda f: f.set(
                "area_ha", ee.Number(f.geometry().area(maxError=10)).divide(10000)
            )
        )

        return polygons.filter(ee.Filter.gte("area_ha", min_area_ha))

    def generate_transects(
        self,
        polygons: ee.FeatureCollection,
        length_m: float = 1000,
    ) -> ee.FeatureCollection:
        """
        Create E-W transects from polygon centroids.
        """

        half_length_deg = length_m / 111320 / 2

        def build(feature):

            geom = feature.geometry()

            centroid = geom.centroid(maxError=10)

            coords = centroid.coordinates()

            lon = ee.Number(coords.get(0))
            lat = ee.Number(coords.get(1))

            transect = ee.Geometry.LineString(
                [
                    [lon.subtract(half_length_deg), lat],
                    [lon.add(half_length_deg), lat],
                ]
            )

            return ee.Feature(transect, {"length_m": length_m})

        return ee.FeatureCollection(polygons.map(build))

    def least_cost_surface(
        self,
        optimum_elevation: float = 300.0,
        maximum_slope: float = 30.0,
        preferred_aspect: float = 180.0,
        aspect_ranges: list[tuple[int, int]] | None = None,
        elevation_weight: float = 0.4,
        slope_weight: float = 0.4,
        aspect_weight: float = 0.2,
    ) -> ee.Image:
        """
        Menghitung permukaan biaya (Cost Surface) berbasis kesesuaian lahan multi-kriteria.
        Skala output berkisar antara 0 (Sangat Sesuai/Biaya Rendah) hingga 1 (Sangat Buruk/Biaya Tinggi).
        """

        # 1. Elevasi: Semakin dekat dengan optimum_elevation, cost semakin mendekati 0
        # Menggunakan nilai maksimum teoritis elevasi untuk normalisasi (misal 4000m, atau dinamis)
        # Di sini kita gunakan nilai konstan 4000 sebagai batas atas normalisasi aman
        max_elevation_ref = 4000.0
        elev_score = (
            self.dem.subtract(ee.Image.constant(optimum_elevation))
            .abs()
            .divide(max_elevation_ref)
            .clamp(0, 1)  # Memastikan nilai tetap di rentang 0-1
        )

        # 2. Kemiringan Lereng (Slope): 0° -> biaya rendah, makin terjal -> biaya tinggi
        slope_score = self.slope.divide(maximum_slope).clamp(0, 1)

        # 3. Orientasi Lereng (Aspect): Jarak angular dari preferred_aspect
        # Menggunakan rumus absolut selisih arah (0 - 180 derajat)
        # preferred = ee.Image.constant(preferred_aspect)
        # aspect_score = self.aspect.subtract(preferred).abs().divide(180.0).clamp(0, 1)

        aspect_score = ee.Image.constant(1.0)

        if aspect_ranges is not None:
            for amin, amax in aspect_ranges:
                mask = self.aspect.gte(amin).And(self.aspect.lte(amax))
                aspect_score = aspect_score.where(mask, 0.0)

        # 4. Pembobotan Total Cost
        cost = (
            elev_score.multiply(elevation_weight)
            .add(slope_score.multiply(slope_weight))
            .add(aspect_score.multiply(aspect_weight))
        )

        # Mengembalikan citra cost yang digunting (clip) sesuai wilayah studi (ROI)
        return cost.clip(self.roi).rename("cost_surface")

    @staticmethod
    def least_cost_path(
        cost_raster: str,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
    ) -> LineString:
        """
        Compute true Least Cost Path from cost raster.
        """

        with rasterio.open(cost_raster) as src:
            cost = src.read(1)

            start = src.index(*start_xy)
            end = src.index(*end_xy)

            pixels, _ = route_through_array(
                cost,
                start,
                end,
                fully_connected=True,
                geometric=True,
            )

            coords = [src.xy(r, c) for r, c in pixels]

        return LineString(coords)

    def generate_least_cost_transects(
        self,
        polygons,
        cost_raster: str,
    ):
        features = []

        for feature in polygons.getInfo()["features"]:
            geom = shape(feature["geometry"])

            minx, miny, maxx, maxy = geom.bounds

            start = (minx, (miny + maxy) / 2)
            end = (maxx, (miny + maxy) / 2)

            line = self.least_cost_path(
                cost_raster,
                start,
                end,
            )

            features.append(ee.Feature(ee.Geometry.LineString(list(line.coords))))

        return ee.FeatureCollection(features)

    # def run(
    #     self,
    #     min_elevation: float = 200,
    #     max_elevation: float = 800,
    #     min_slope: float = 0,
    #     max_slope: float = 20,
    #     min_area_ha: float = 30,
    #     transect_length_m: float = 1000,
    #     aspect_ranges=[
    #         (45, 135),  # Timur
    #         (135, 225),  # Selatan
    #     ],
    # ) -> ee.FeatureCollection:
    #
    #     polygons = self.candidate_polygons(
    #         min_elevation=min_elevation,
    #         max_elevation=max_elevation,
    #         min_slope=min_slope,
    #         max_slope=max_slope,
    #         min_area_ha=min_area_ha,
    #         aspect_ranges=aspect_ranges,
    #     )
    #
    #     return self.generate_transects(
    #         polygons,
    #         length_m=transect_length_m,
    #     )

    def run(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        min_slope: float = 0,
        max_slope: float = 20,
        aspect_ranges: list[tuple[int, int]] | None = None,
        min_area_ha: float = 30,
        transect_length_m: float = 1000,
        least_cost: bool = False,
        cost_raster: str | None = None,
    ) -> ee.FeatureCollection:
        """
        Main pipeline.

        Binary mode:
            Suitable Area -> Polygon -> Transect

        Least Cost mode:
            Suitable Area -> Polygon -> Least Cost Path
            (menggunakan cost raster lokal yang telah diekspor sebelumnya)
        """

        polygons = self.candidate_polygons(
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            min_slope=min_slope,
            max_slope=max_slope,
            min_area_ha=min_area_ha,
            aspect_ranges=aspect_ranges,
        )

        if not least_cost:
            return self.generate_transects(
                polygons,
                length_m=transect_length_m,
            )

        if cost_raster is None:
            raise ValueError("cost_raster wajib diberikan jika least_cost=True.")

        return self.generate_least_cost_transects(
            polygons=polygons,
            cost_raster=cost_raster,
        )

    @staticmethod
    def export_geojson(fc: ee.FeatureCollection, output_file: str):

        url = fc.getDownloadURL(filetype="GeoJSON", filename="transects")

        response = requests.get(url)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"Saved: {output_file}")

    @staticmethod
    def export_cost_surface(
        cost_surface: ee.Image,
        output_file: str,
        scale: int = 30,
        crs: str = "EPSG:4326",
    ):
        """
        Export cost surface sebagai GeoTIFF.
        """

        url = cost_surface.getDownloadURL(
            {
                "format": "GEO_TIFF",
                "scale": scale,
                "crs": crs,
            }
        )

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print(f"Saved: {output_file}")
