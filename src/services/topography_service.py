from pathlib import Path
import subprocess
import requests
import ee


class TopographyService:
    """
    Topographic analysis service.

    Features:
    - Download DEM from GEE
    - Generate contours
    - Generate slope
    - Generate hillshade
    """

    def __init__(
        self,
        roi: ee.Geometry,
        output_dir: str,
        dem_dataset: str = "USGS/SRTMGL1_003",
    ):
        self.roi = roi
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dem = ee.Image(dem_dataset).clip(roi)

    # --------------------------------------------------
    # DOWNLOAD DEM
    # --------------------------------------------------

    def download_dem(
        self,
        filename: str = "dem.tif",
        scale: int = 30,
    ) -> Path:

        output_file = self.output_dir / filename

        url = self.dem.getDownloadURL(
            {
                "region": self.roi,
                "scale": scale,
                "format": "GEO_TIFF",
            }
        )

        print(f"Downloading DEM:\n{url}")

        response = requests.get(url, stream=True, timeout=300)

        response.raise_for_status()

        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"[✓] DEM saved: {output_file}")

        return output_file

    # --------------------------------------------------
    # CONTOUR
    # --------------------------------------------------

    def generate_contours(
        self,
        dem_path: Path,
        interval_m: int = 25,
        output_name: str = "contours.geojson",
    ) -> Path:

        output_file = self.output_dir / output_name

        cmd = [
            "gdal_contour",
            "-a",
            "elevation",
            "-i",
            str(interval_m),
            str(dem_path),
            str(output_file),
        ]

        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        print(f"[✓] Contours saved: {output_file}")

        return output_file

    # --------------------------------------------------
    # SLOPE
    # --------------------------------------------------

    def generate_slope(
        self,
        dem_path: Path,
        output_name: str = "slope.tif",
    ) -> Path:

        output_file = self.output_dir / output_name

        cmd = [
            "gdaldem",
            "slope",
            str(dem_path),
            str(output_file),
        ]

        subprocess.run(cmd, check=True)

        print(f"[✓] Slope saved: {output_file}")

        return output_file

    # --------------------------------------------------
    # HILLSHADE
    # --------------------------------------------------

    def generate_hillshade(
        self,
        dem_path: Path,
        output_name: str = "hillshade.tif",
    ) -> Path:

        output_file = self.output_dir / output_name

        cmd = [
            "gdaldem",
            "hillshade",
            str(dem_path),
            str(output_file),
        ]

        subprocess.run(cmd, check=True)

        print(f"[✓] Hillshade saved: {output_file}")

        return output_file

    # --------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------

    def run(
        self,
        contour_interval: int = 25,
    ):

        dem_file = self.download_dem()

        contour_file = self.generate_contours(dem_file, interval_m=contour_interval)

        slope_file = self.generate_slope(dem_file)

        hillshade_file = self.generate_hillshade(dem_file)

        return {
            "dem": dem_file,
            "contours": contour_file,
            "slope": slope_file,
            "hillshade": hillshade_file,
        }
