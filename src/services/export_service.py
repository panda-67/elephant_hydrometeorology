import os
import zipfile
import io
import requests
import ee
from datetime import datetime
from config import config


class ExportAssetsService:
    def __init__(self, roi: ee.Geometry):
        self.roi = roi
        self.use_demnas = getattr(config, "USE_DEMNAS", False)
        self.mode = getattr(config, "SATELLITE_MODE", "sentinel2")
        self.output_dir = os.path.join("data", "output_rasters", self.mode.lower())
        os.makedirs(self.output_dir, exist_ok=True)

    def _smart_export_image(
        self,
        image: ee.Image,
        description: str,
        filename_prefix: str,
        scale: int,
        folder_dest: str,
    ) -> str:
        """
        Helper untuk mencoba download langsung ke lokal.
        Mengecek kevalidan file ZIP sebelum diekstrak.
        """
        processed_img = image.toFloat().clip(self.roi)
        local_tif_path = os.path.join(
            self.output_dir,
            f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
        )

        try:
            print(f"  [~] Mencoba download langsung ke lokal untuk: {description}...")

            # Mendapatkan URL download langsung dari GEE
            download_url = processed_img.getDownloadURL(
                {
                    "scale": scale,
                    "crs": "EPSG:4326",
                    "region": self.roi,
                    "format": "GEO_TIFF",
                }
            )

            response = requests.get(download_url, timeout=60)

            # JIKA GEE MENGEMBALIKAN ERROR (Biasanya dalam bentuk JSON/Text)
            if response.status_code != 200 or b"error" in response.content[:100]:
                # Coba decode pesan error dari GEE server
                try:
                    error_msg = response.json()["error"]["message"]
                except Exception:
                    error_msg = response.text[
                        :200
                    ]  # Ambil potongan teks jika bukan JSON
                raise Exception(f"GEE Server Error: {error_msg}")

            # KONDISI 1: Respons berupa raw GeoTIFF langsung (Mengecek Magic Bytes TIFF)
            # 'II*\x00' (Little Endian) atau 'MM\x00*' (Big Endian)
            if response.content[:4] in [
                b"II*\x00",
                b"MM\x00*",
                b"II\x2a\x00",
                b"MM\x00\x2a",
            ]:
                with open(local_tif_path, "wb") as f:
                    f.write(response.content)
                print(
                    f"    ✅ [LOKAL SUKSES] File GeoTIFF langsung disimpan di: {local_tif_path}"
                )
                return "LOCAL_SUCCESS"

            # KONDISI 2: Respons berupa pembungkus ZIP standar GEE
            elif zipfile.is_zipfile(io.BytesIO(response.content)):
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for file_info in z.infolist():
                        if file_info.filename.endswith(".tif"):
                            file_info.filename = f"{filename_prefix}.tif"
                            z.extract(file_info, self.output_dir)
                print(f"    ✅ [LOKAL SUKSES] File ZIP diekstrak ke: {local_tif_path}")

                return "LOCAL_SUCCESS"

            else:
                raise Exception(
                    "Format respons tidak dikenali (bukan RAW TIF ataupun ZIP)."
                )

        except Exception as e:
            # Sekarang Anda akan melihat alasan REAL mengapa dia dialihkan ke Drive
            print(f"    ⚠️ [DOWNLOAD LOKAL GAGAL] Alasan: {str(e)}")
            print(
                f"    🔄 [FALLBACK] Mengalihkan ekspor {description} ke Google Drive Task..."
            )

            filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = ee.batch.Export.image.toDrive(
                image=processed_img,
                description=description,
                folder=folder_dest,
                fileNamePrefix=filename,
                scale=scale,
                region=self.roi,
                maxPixels=1e13,
            )
            task.start()
            return task.id

    def export_forensic_rasters(
        self, pipelines: tuple, scale_s2: int = 10, scale_terrain: int = 30
    ) -> list:
        """
        Orchestrator utama untuk mengekspor seluruh hasil pipeline analitik.
        """
        task_ids = []
        folder_dest = "GeoForensic_Tangse_Meureudu"

        print("\n" + "=" * 60)
        print("[~] RUNNING SMART EXPORT SERVICE (LOCAL FIRST -> DRIVE FALLBACK)")
        print("=" * 60)

        p1_img, p2_img, p3_img, p4_img = pipelines

        # ------------------------------------------------------------------------
        # 1. Pipeline 1: Satellites Metrics
        # ------------------------------------------------------------------------
        if p1_img:
            p1_export = p1_img.select(
                [
                    "d_NDVI_degradation",
                    "d_NDVI_destruction",
                    "NDVI_Degradation_Mask",
                    "ndvi_degradation_masif",
                ]
            )
            actual_scale = (
                scale_s2
                if p1_img.get("sensor_mode").getInfo() == "sentinel2"
                else scale_terrain
            )
            res = self._smart_export_image(
                p1_export,
                "P1 Satellite Vegetation Metrics",
                "p1_satellite_vegetation_metrics",
                actual_scale,
                folder_dest,
            )
            if res != "LOCAL_SUCCESS":
                task_ids.append(res)

        # ------------------------------------------------------------------------
        # 2. Pipeline 2: Hydrology Metrics
        # ------------------------------------------------------------------------
        if p2_img:
            p2_export = p2_img.select(
                [
                    "dynamic_rainfall_peak",
                    "Q_simulated_baseline",
                    "Q_actual_floodevent",
                    "runoff_net_increase",
                ]
            )
            res = self._smart_export_image(
                p2_export,
                "P2 Hydrology Runoff Simulation",
                "p2_hydrology_runoff_simulation",
                scale_s2 if self.use_demnas else scale_terrain,
                folder_dest,
            )
            if res != "LOCAL_SUCCESS":
                task_ids.append(res)

        # ------------------------------------------------------------------------
        # 3. Pipeline 3: Terrain Deforestation Metrics
        # ------------------------------------------------------------------------
        if p3_img:
            p3_export = p3_img.select(
                [
                    "forest_cover_2020",
                    "forest_loss_preevent",
                    "critical_upstream_deforestation",
                ]
            )
            res = self._smart_export_image(
                p3_export,
                "P3 Upstream Critical Deforestation",
                "p3_upstream_critical_deforestation",
                scale_s2 if self.use_demnas else scale_terrain,
                folder_dest,
            )
            if res != "LOCAL_SUCCESS":
                task_ids.append(res)

        # ------------------------------------------------------------------------
        # 4. Pipeline 4: Spatial Causal Matrix
        # ------------------------------------------------------------------------
        if p4_img:
            res = self._smart_export_image(
                p4_img,
                "P4 Spatial Causal Matrix",
                "p4_spatial_causal_matrix",
                scale_s2 if self.use_demnas else scale_terrain,
                folder_dest,
            )
            if res != "LOCAL_SUCCESS":
                task_ids.append(res)

        print("\n" + "=" * 60)
        print("[✓] Pemrosesan inisiasi ekspor selesai.")
        if task_ids:
            print(
                f"[INFO] Terdapat {len(task_ids)} task aktif yang dialihkan ke GEE Server / Google Drive."
            )
        else:
            print(
                "[INFO] Semua layer berhasil didownload langsung ke lokal! Tidak ada task di Google Drive."
            )
        print("=" * 60)

        return task_ids
