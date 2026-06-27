# 🌍 Geo-Forensic Spatial Corridor: Tangse - Meureudu Watershed Study

[![OS - Arch Linux / CachyOS](https://img.shields.io/badge/OS-Arch%20%2FCachyOS-blueviolet?style=flat&logo=arch-linux)](https://archlinux.org)
[![Engine - Google Earth Engine](https://img.shields.io/badge/Engine-Google%20Earth%20Engine-green?style=flat&logo=google-earth)](https://earthengine.google.com)
[![GIS - QGIS Compatible](https://img.shields.io/badge/GIS-QGIS%20Compatible-darkgreen?style=flat&logo=qgis)](https://qgis.org)

Repositori ini berisi infrastruktur analisis spasial berbasis **Hidrologi
Forensik** untuk mendeteksi hubungan sebab-akibat (_causal-relationship_)
antara deforestasi antropogenik di wilayah hulu dengan bencana banjir
bandang di koridor DAS Tangse - Meureudu, Aceh.

Aplikasi mengekstrak anomali vegetasi multitemporal dan melakukan simulasi
limpasan permukaan menggunakan integrasi data satelit **ESA WorldCover**,
**Dynamic World**, **Copernicus DEM**, dan **CHIRPS Daily Rainfall** secara
dinamis melalui server-side Google Earth Engine (GEE).

---

## 🛰️ Alur Kronologis Forensik Spasial

Pemrosesan data di dalam pipeline ini dirancang ketat tanpa menggunakan data
_fallback_ (100% data empiris server-side) dengan pembagian lini masa ilmiah:

1. **Fase Akumulasi Pra-Bencana (2020 - Sesaat Sebelum Banjir 2025):** Melacak
   laju deforestasi tahunan makro dan mengunci jangkar kondisi rimbun kanopi asli.
2. **Fase Pemicu (Hidrologi):** Menjatuhkan hujan badai dinamis dari CHIRPS
   pada kondisi penggunaan lahan lintas tahun menggunakan model hidrologi SCS Curve
   Number (TR-55).
3. **Fase Dampak Instan (Post-Event):** Mengisolasi delta kerusakan fisik klorofil
   vegetasi akibat sapuan bencana/longsor.

---

## 📂 Struktur Repositori

```text
geo_forensic_corridor/
├── data/
│   └── output_metrics/   # Output otomatis laporan JSON & berkas spasial QGIS
├── src/
│   ├── core/
│   │   ├── engine.py     # Manajemen otentikasi GEE, ROI, & utilitas ekstraksi
│   │   ├── hydrology.py  # Pemodelan hidrologi SCS-CN dengan input CHIRPS dinamis
│   │   ├── landcover.py
│   │   ├── terrain.py
│   │   └── vegatation.py
│   └── pipelines/
│       ├── p1_gajah_satellite.py
│       ├── p2_gajah_hydrology.py
│       ├── p3_meureudu_upstream.py
│       └── p4_causal_modeling.py
├── config.py             # Konfigurasi terpusat parameter tanggal & batas watershed
├── environment.yml
├── export_assets.py
├── generate_countour.py
├── generate_track.py
├── main.py               # Script eksekusi utama, data parsing, & guardrail logika
└── README.md
```

## 🛠️ Langkah Instalasasi & Penggunaan

### Kebutuhan Lingkungan Sistem

Aplikasi ini dikembangkan dan dioptimalkan di atas sistem operasi Arch Linux /
CachyOS menggunakan package manager lingkungan Mamba / Conda. 2. Setup Environment

Kloning repositori ini dan pasang dependensi lingkungan geospatial yang dibutuhkan:

```{bash}

git clone https://github.com/panda-67/elephant_hydrometeorology
cd elephant_hydrometeorology

# Membuat environment via micromamba
micromamba create -f environment.yml
micromamba activate geo-forensic-env

```

### Otentikasi Google Earth Engine

Sebelum menjalankan skrip untuk pertama kali, pastikan Anda telah mengaktifkan
akses API kunci Google Earth Engine ke sistem lokal Anda:

```{bash}
earthengine authenticate
```

### Eksekusi Program

Jalankan pipa pemrosesan utama untuk memicu kalkulasi server GEE, pembuatan
laporan JSON, dan ekspor geometri spasial:

```{bash}
python main.py
```

## 🗺️ Integrasi Native dengan QGIS

Setiap kali skrip main.py selesai dieksekusi secara sukses, engine secara
otomatis akan mengekspor geometri batas wilayah studi riil dari server ke
direktori lokal:

- Lokasi Berkas: data/output_metrics/tangse_meureudu_roi.geojson

Cara Membuka di QGIS:

1. Buka Aplikasi QGIS Anda.

2. Lakukan operasi drag and drop berkas tangse_meureudu_roi.geojson langsung
   ke dalam Map Canvas QGIS.

3. Batas spasial makro wilayah hulu DAS akan terplot secara presisi di atas
   layer peta kerja Anda untuk kebutuhan kartografi lanjutan.

## ⚖️ Lisensi & Integritas Data

Seluruh metode ekstraksi metrik di dalam proyek ini dilengkapi dengan sistem
pengaman ketat (Strict Guardrails). Jika terjadi kegagalan transmisi data,
interupsi koneksi, atau ketidakcocokan nama band di sisi server GEE, aplikasi
akan melempar ValueError secara jujur untuk menjaga integritas keaslian data
forensik dari manipulasi angka gaib (magic numbers).
