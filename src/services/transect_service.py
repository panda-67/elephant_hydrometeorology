import requests

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

    def suitable_area(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        max_slope: float = 20,
    ) -> ee.Image:
        """
        Create suitability mask.
        """

        elevation_mask = self.dem.gte(min_elevation).And(self.dem.lte(max_elevation))

        slope_mask = self.slope.lte(max_slope)

        return elevation_mask.And(slope_mask).selfMask()

    def candidate_polygons(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        max_slope: float = 20,
        min_area_ha: float = 10,
        scale: int = 30,
    ) -> ee.FeatureCollection:
        """
        Convert suitable area to polygons.
        """

        mask = self.suitable_area(
            min_elevation,
            max_elevation,
            max_slope,
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

    def run(
        self,
        min_elevation: float = 200,
        max_elevation: float = 800,
        max_slope: float = 20,
        min_area_ha: float = 10,
        transect_length_m: float = 1000,
    ) -> ee.FeatureCollection:

        polygons = self.candidate_polygons(
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            max_slope=max_slope,
            min_area_ha=min_area_ha,
        )

        return self.generate_transects(
            polygons,
            length_m=transect_length_m,
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
