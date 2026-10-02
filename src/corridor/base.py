from pathlib import Path

import geopandas as gpd


class CorridorBase:
    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else Path(__file__).resolve().parents[2]

    def load_vector(self, path: Path) -> gpd.GeoDataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Vector not found: {path}")

        gdf = gpd.read_file(path)

        if gdf.empty:
            raise ValueError(f"Vector contains no features: {path}")

        if gdf.crs is None:
            raise ValueError(f"Vector has no CRS: {path}")

        invalid = ~gdf.geometry.is_valid

        if invalid.any():
            print(f"  Repairing {invalid.sum()} invalid geometries...")

            gdf = gdf.copy()
            gdf.geometry = gdf.geometry.make_valid()

        gdf = gdf.loc[~gdf.geometry.is_empty].copy()

        return gdf

    def metric_crs(self, gdf: gpd.GeoDataFrame):
        crs = gdf.estimate_utm_crs()

        if crs is None:
            raise ValueError("Could not determine suitable metric CRS.")

        return crs

    def area(self, gdf: gpd.GeoDataFrame) -> dict:
        crs = self.metric_crs(gdf)
        metric = gdf.to_crs(crs)

        area_m2 = metric.geometry.area.sum()

        return {
            "crs": crs,
            "area_m2": area_m2,
            "area_ha": area_m2 / 10_000,
            "area_km2": area_m2 / 1_000_000,
        }

    def dissolve(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:

        return gdf[["geometry"]].dissolve()

    def intersection(
        self,
        source: gpd.GeoDataFrame,
        boundary: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:

        if source.crs != boundary.crs:
            boundary = boundary.to_crs(source.crs)

        result = gpd.overlay(
            source,
            boundary,
            how="intersection",
        )

        if result.empty:
            raise ValueError("Spatial intersection returned no features.")

        return result
