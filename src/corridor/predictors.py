import io
import zipfile
from pathlib import Path

import ee
import geopandas as gpd
import requests

from py.utils.gee_drive import (
    download_earth_engine_export,
    download_existing_drive_export,
)
from src.core.engine import GEEEngine
from src.corridor.landcover import CorridorLandCover
from src.corridor.ndvi import CorridorNDVI
from src.corridor.terrain import CorridorTerrain
from src.corridor.water import CorridorWater


class CorridorPredictors:
    """Generator predictor lingkungan untuk analisis koridor gajah."""

    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else Path(__file__).resolve().parents[2]

        # Analysis domain:
        # watershed ROI, bukan KHL source polygon.
        self.aoi_path = self.root / "data/output_vectors/tangse_meureudu_roi.geojson"

        self.output_dir = self.root / "data/output_rasters/corridor"

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.engine = None
        self.aoi = None
        self.terrain = None
        self.landcover = None
        self.ndvi = None
        self.water = None

    def initialize(self):
        """Inisialisasi koneksi Google Earth Engine."""
        print("\n[1] Initializing Google Earth Engine...")
        self.engine = GEEEngine()

    def load_aoi(self):
        """Memuat watershed ROI sebagai analysis domain."""

        print("\n[2] Loading watershed analysis domain...")

        if not self.aoi_path.exists():
            raise FileNotFoundError(f"Watershed ROI tidak ditemukan: {self.aoi_path}")

        gdf = gpd.read_file(self.aoi_path)

        if gdf.empty:
            raise ValueError("Watershed ROI tidak memiliki feature.")

        if gdf.crs is None:
            raise ValueError("Watershed ROI tidak memiliki CRS.")

        print(f"    Features : {len(gdf):,}")
        print(f"    CRS      : {gdf.crs}")
        print(f"    Source   : {self.aoi_path}")

        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        geometry = gdf.geometry.union_all()

        self.aoi = ee.Geometry(geometry.__geo_interface__)

        print("    Watershed AOI siap digunakan oleh GEE.")

    def initialize_terrain(self):
        """Menginisialisasi terrain predictor."""

        print("\n[3] Initializing terrain predictor...")

        self.terrain = CorridorTerrain(
            roi=self.aoi,
        )

    def initialize_landcover(self):
        """Menginisialisasi land-cover analyzer."""

        self.landcover = CorridorLandCover(
            roi=self.aoi,
        )

    def initialize_ndvi(self):
        """Menginisialisasi NDVI predictor."""

        self.ndvi = CorridorNDVI(
            roi=self.aoi,
        )

    def initialize_water(self):
        """Menginisialisasi water distance."""

        self.water = CorridorWater(roi=self.aoi)

    def export_local(
        self,
        image: ee.Image,
        filename: str,
        scale: int = 10,
        crs: str = "EPSG:4326",
    ):
        """
        Download raster dari Google Earth Engine langsung ke lokal.

        Parameter:
        - scale : resolusi output dalam meter
        - crs   : sistem koordinat output
        - region: AOI watershed
        - format: GeoTIFF

        Jika download langsung gagal, otomatis fallback
        ke Google Drive dengan parameter export yang sama.
        """

        output_path = self.output_dir / filename

        print(f"    [~] Downloading {filename}...")
        print(f"        Scale : {scale} m")
        print(f"        CRS   : {crs}")

        try:
            download_url = image.getDownloadURL(
                {
                    "scale": scale,
                    "crs": crs,
                    "region": self.aoi,
                    "format": "GEO_TIFF",
                }
            )

            response = requests.get(
                download_url,
                timeout=120,
            )

            if response.status_code != 200:
                try:
                    error_msg = response.json()["error"]["message"]
                except Exception:
                    error_msg = response.text[:300]

                raise RuntimeError(f"GEE Server Error: {error_msg}")

            content = response.content

            # --------------------------------------------------
            # RAW GEOTIFF
            # --------------------------------------------------
            if content[:4] in {
                b"II*\x00",
                b"MM\x00*",
            }:
                with open(output_path, "wb") as f:
                    f.write(content)

                print(f"    [✓] GeoTIFF saved: {output_path}")

                return str(output_path)

            # --------------------------------------------------
            # ZIP CONTAINING GEOTIFF
            # --------------------------------------------------
            if zipfile.is_zipfile(io.BytesIO(content)):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    tif_files = [
                        info
                        for info in z.infolist()
                        if info.filename.lower().endswith((".tif", ".tiff"))
                    ]

                    if not tif_files:
                        raise RuntimeError("ZIP GEE tidak mengandung GeoTIFF.")

                    tif_info = tif_files[0]

                    with (
                        z.open(tif_info) as src,
                        open(output_path, "wb") as dst,
                    ):
                        dst.write(src.read())

                print(f"    [✓] GeoTIFF extracted: {output_path}")

                return str(output_path)

            raise RuntimeError("Respons GEE bukan GeoTIFF atau ZIP.")

        except Exception as exc:
            print(f"    [!] Local download failed: {exc}")

            print("    [~] Checking existing Google Drive export...")

            filename_prefix = Path(filename).stem

            # --------------------------------------------------
            # Check existing Drive file first
            # --------------------------------------------------

            existing_file = download_existing_drive_export(
                filename_prefix=filename_prefix,
                output_path=output_path,
                folder_name="GeoForensic_Tangse_Meureudu",
            )

            if existing_file is not None:
                print("    [✓] Using existing Drive export.")

                return existing_file

            # --------------------------------------------------
            # No existing file → create new EE task
            # --------------------------------------------------

            print("    [~] No existing Drive export found.")

            print("    [~] Creating new Earth Engine task...")

            task = ee.batch.Export.image.toDrive(
                image=image,
                description=filename_prefix,
                folder="GeoForensic_Tangse_Meureudu",
                fileNamePrefix=filename_prefix,
                scale=scale,
                crs=crs,
                region=self.aoi,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )

            task.start()

            print(f"    [✓] Drive task started: {task.id}")

            return download_earth_engine_export(
                task=task,
                filename_prefix=filename_prefix,
                output_path=output_path,
                folder_name="GeoForensic_Tangse_Meureudu",
            )

    def export_elevation(self):
        """Menghasilkan dan mengekspor raster elevation."""

        if self.terrain is None:
            raise RuntimeError("Terrain predictor belum diinisialisasi.")

        elevation = self.terrain.get_elevation()

        return self.export_local(
            elevation,
            "corridor_elevation.tif",
            scale=10,
            crs="EPSG:4326",
        )

    def export_slope(self):
        """Menghasilkan dan mengekspor raster slope."""

        if self.terrain is None:
            raise RuntimeError("Terrain predictor belum diinisialisasi.")

        slope = self.terrain.get_slope()

        return self.export_local(
            slope,
            "corridor_slope.tif",
            scale=10,
            crs="EPSG:4326",
        )

    def export_landcover(self):
        """Menghasilkan dan mengekspor raster land cover."""

        if self.landcover is None:
            raise RuntimeError("Land-cover analyzer belum diinisialisasi.")

        landcover = self.landcover.get_worldcover()

        return self.export_local(
            landcover,
            "corridor_landcover_worldcover_2020.tif",
            scale=10,
            crs="EPSG:4326",
        )

    def export_ndvi(self):
        """Menghasilkan dan mengekspor raster NDVI."""

        if self.ndvi is None:
            raise RuntimeError("NDVI predictor belum diinisialisasi.")

        ndvi = self.ndvi.get_ndvi()

        return self.export_local(
            ndvi,
            "corridor_ndvi.tif",
            scale=10,
            crs="EPSG:4326",
        )

    def export_distance_to_water(self):
        if self.water is None:
            raise RuntimeError("Distance-to-water predictor belum diinisialisasi.")

        distance = self.water.get_distance_to_water()

        return self.export_local(
            distance,
            "corridor_distance_to_water.tif",
            scale=10,
            crs="EPSG:4326",
        )
