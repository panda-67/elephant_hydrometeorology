import csv
import json
import os
from datetime import datetime
from typing import Optional

import ee
import geemap

from config import config


class GEEEngine:
    """Kelas dasar untuk inisialisasi aman koneksi Google Earth Engine."""

    def __init__(self):
        try:
            ee.Initialize(project=config.PROJECT_ID)
            print(f"[✓] GEE Terhubung Menggunakan Project ID: {config.PROJECT_ID}")
        except Exception as e:
            raise RuntimeError(f"[X] Gagal menginisialisasi GEE: {str(e)}")

    def get_hydro_roi(self) -> ee.Geometry:
        """Delineasi Multi-DAS otomatis berbasis HydroSHEDS Level 12."""
        outlets = ee.Geometry.MultiPoint(
            [list(coord) for coord in config.OUTLET_COORDINATES]
        )
        hydrosheds = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
        multi_watersheds = hydrosheds.filterBounds(outlets)
        return multi_watersheds.union(maxError=1).geometry()

    def export_roi_to_geojson(
        self, roi: ee.Geometry, filename: str = "watershed_roi.geojson"
    ) -> str:
        """
        Mengekspor objek ee.Geometry dari server GEE menjadi berkas GeoJSON lokal
        agar dapat langsung dimuat ke dalam QGIS / ArcGIS.
        """
        print("[~] Fetching ROI geometry coordinates from Earth Engine server...")

        # Mengambil informasi spasial geometri dari server GEE ke lokal Python
        roi_info = roi.getInfo()

        # Menentukan direktori penyimpanan
        output_dir = os.path.join("data", "output_metrics")
        os.makedirs(output_dir, exist_ok=True)
        geojson_path = os.path.join(output_dir, filename)

        # Menyusun struktur GeoJSON standar yang dikenali QGIS menggunakan datetime lokal Python
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": roi_info,
                    "properties": {
                        "name": "Watershed ROI",
                        "project": "Geo-Forensic Study",
                        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
            ],
        }

        # Menulis data ke dalam berkas .geojson
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=4)

        print(f"[✓] Geometri ROI berhasil diekspor untuk QGIS di: {geojson_path}")
        return geojson_path

    def visualize_on_map(self, roi, p1, p2, p3, p4):
        print("[~] Generating Interactive Map with geemap...")

        # Inisialisasi peta interaktif di tengah ROI
        Map = geemap.Map()
        Map.centerObject(roi, 12)

        # 1. Visualisasi Degradasi Hutan (P1 - d_NDVI_degradation)
        # Nilai negatif berarti vegetasi memburuk (degradasi)
        ndvi_vis = {"min": -0.5, "max": 0, "palette": ["red", "yellow", "green"]}
        Map.addLayer(
            p1.select("d_NDVI_degradation"),
            ndvi_vis,
            "P1: Pre-Event NDVI Degradation",
        )

        # 2. Visualisasi Lonjakan Runoff (P2 - runoff_net_increase)
        runoff_vis = {"min": 0, "max": 100, "palette": ["white", "blue", "darkblue"]}
        Map.addLayer(
            p2.select("runoff_net_increase"), runoff_vis, "P2: Runoff Net Increase"
        )

        # 3. Visualisasi Zona Kritis Hulu (P3 - critical_upstream_deforestation)
        # Karena biner, kita beri warna merah solid untuk area yang aktif
        Map.addLayer(
            p3.select("critical_upstream_deforestation"),
            {"palette": ["purple"]},
            "P3: Critical Deforestation (>15 Deg)",
        )

        # 4. Visualisasi Model Kausal (P4 - gabungan multi-band)
        # Menampilkan visualisasi RGB False Color dari hubungan sebab-akibat
        # Red = Degradasi hulu, Green = Lonjakan runoff, Blue = Kerusakan hilir
        causal_vis = {
            "bands": [
                "cause_degradation",
                "effect_runoff_spike",
                "effect_post_destruction",
            ],
            "min": -0.5,
            "max": 0.5,
        }
        Map.addLayer(p4, causal_vis, "P4: Spatial Causal Composite (RGB)")

        # Tambahkan batas ROI sebagai garis merah
        Map.addLayer(
            roi, {"color": "red", "fillColor": "00000000"}, "Watershed Boundary (ROI)"
        )

        # Tampilkan peta (jika di Jupyter) atau simpan sebagai HTML interaktif
        Map.save("data/output_metrics/forensic_map.html")
        print("[✓] Interactive map saved to data/output_metrics/forensic_map.html")

    def safe_extract_metric(self, stats_dict: dict, key: str) -> Optional[float]:
        """
        Mengekstrak nilai GEE reducer secara ketat berdasarkan pencarian kata kunci band.
        Mengembalikan None jika data tidak ditemukan agar tidak memalsukan laporan forensik.

        PERBAIKAN: pencocokan sebelumnya menggunakan `key in k` (substring match
        di mana saja dalam string), yang berisiko false-positive bila satu nama
        band kebetulan menjadi substring dari nama band lain (mis. "NDVI_mean"
        vs "NDVI_preevent_mean"). Sekarang dicocokkan pada akhiran kunci dengan
        pembatas underscore eksplisit ("_" + key atau persis sama dengan key),
        sesuai pola kunci hasil reduceRegion yaitu "{band}_{reducer}".
        """
        if not stats_dict:
            return None
        for k, v in stats_dict.items():
            if k == key or k.endswith("_" + key):
                return v
        return None

    def tier_report(
        ee_result: dict,
        output_dir: str,
        title: str = "Distribusi Luas per Tier Bukti Kausal",
    ) -> str:
        """
        Menampilkan distribusi luas per tier dan menyimpannya sebagai CSV.

        Parameters
        ----------
        ee_result : dict
            Hasil .getInfo() dari reduceRegion(
            Reducer.sum().group()
            ).
            Format:
            {
                "groups": [
                    {"tier": int, "sum": float},
                    ...
                ]
            }

        output_dir : str
            Direktori penyimpanan CSV.

        title : str
            Judul laporan yang ditampilkan di terminal.

        Returns
        -------
        str
            Path file CSV yang dihasilkan.
        """

        TIER_LABELS = {
            0: "Tidak ada perubahan signifikan",
            1: "Degradasi vegetasi (pra-bencana)",
            2: "Destruksi fisik (saat bencana)",
            3: "Area kritis (compound hotspot)",
            4: "Terkonfirmasi hidrologis (hotspot utama)",
        }

        # Urutkan berdasarkan tier
        groups = sorted(
            ee_result.get("groups", []),
            key=lambda g: g["tier"],
        )

        # Total luas
        total = sum(float(g["sum"]) for g in groups)

        # ============================================================
        # 1. PRINT REPORT
        # ============================================================

        col_tier = 6
        col_label = 42
        col_luas = 16
        col_pct = 10
        width = col_tier + col_label + col_luas + col_pct

        print()
        print(title)
        print("=" * width)

        print(
            f"{'Tier':<{col_tier}}"
            f"{'Kategori':<{col_label}}"
            f"{'Luas (ha)':>{col_luas}}"
            f"{'Persen':>{col_pct}}"
        )

        print("-" * width)

        for g in groups:
            tier = int(g["tier"])
            luas = float(g["sum"])
            pct = luas / total * 100 if total else 0

            label = TIER_LABELS.get(
                tier,
                f"Tier {tier} (tidak dikenal)",
            )

            print(
                f"{tier:<{col_tier}}"
                f"{label:<{col_label}}"
                f"{luas:>{col_luas},.2f}"
                f"{pct:>{col_pct - 1}.2f}%"
            )

        print("-" * width)

        print(
            f"{'':<{col_tier}}"
            f"{'TOTAL ROI':<{col_label}}"
            f"{total:>{col_luas},.2f}"
            f"{100:>{col_pct - 1}.2f}%"
        )

        print("=" * width)
        print()

        # ============================================================
        # 2. EXPORT CSV
        # ============================================================

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        output_file = os.path.join(
            output_dir,
            "tier_area.csv",
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "tier",
                    "kategori",
                    "luas_ha",
                    "persen",
                ]
            )

            for g in groups:
                tier = int(g["tier"])
                luas = float(g["sum"])
                persen = luas / total * 100 if total else 0

                writer.writerow(
                    [
                        tier,
                        TIER_LABELS.get(
                            tier,
                            f"Tier {tier} (tidak dikenal)",
                        ),
                        luas,
                        persen,
                    ]
                )

        print(f"CSV saved to: {output_file}")

        return output_file
