# Corridor Analysis Pipeline

Dokumentasi pipeline analisis koridor gajah pada repository `elephant_hydrometeorologi`.

Pipeline dirancang untuk menghasilkan ecological resistance surface dan source-based potential connectivity dari kawasan KHL, kemudian menghubungkannya dengan analisis hydrological/flood hazard.

Seluruh runner dijalankan dari root repository menggunakan Python module execution:

```bash
python -m py.corridor.<module>
```

Pipeline menggunakan predictor raster pada grid analisis 10 m. Predictor awal berada pada `EPSG:4326`, sedangkan tahap connectivity menggunakan metric grid `EPSG:32647` dengan resolusi 10 m.

## 1. Pipeline Overview

Urutan eksekusi:

```text
P5  Boundary
 ↓
P6  Predictor Generation
 ↓
P6-QA/QC
 ↓
P7  Normalization
 ↓
P7-QA/QC
 ↓
P7  Diagnostic
 ↓
P8.1 Individual Resistance
 ↓
P8-QA/QC
 ↓
P8.2 Composite Resistance
 ↓
P8.2-QA/QC
 ↓
P9.1 KHL Source Mask
 ↓
P9.2 Source-Based Cost Distance
 ↓
P9.2-QA/QC
 ↓
P9.3 Potential Connectivity
```

Tahap flood hazard dan corridor–flood interaction berada di luar pipeline ecological resistance/connectivity ini dan diproses pada tahap berikutnya.

---

## 2. Execution Order

### P5 — Corridor Boundary

File:

```text
py/corridor/boundary.py
```

Run:

```bash
python -m py.corridor.boundary
```

Fungsi:

- menyiapkan boundary analisis;
- menggunakan watershed Tangse–Meureudu sebagai analysis domain;
- melakukan spatial intersection antara KHL dan watershed;
- menghasilkan boundary yang digunakan oleh predictor generation dan connectivity analysis.

Output utama:

```text
data/output_vectors/tangse_meureudu_roi.geojson
data/output_vectors/KHL_PP_tangse_meureudu.geojson
```

---

## 3. P6 — Predictor Generation

File:

```text
py/corridor/predictors.py
```

Run:

```bash
python -m py.corridor.predictors
```

Tahap ini menghasilkan ecological predictors dari Google Earth Engine.

Analysis domain:

```text
tangse_meureudu_roi.geojson
```

Target raster:

```text
Resolution : 10 m
CRS        : EPSG:4326
```

Predictors yang digunakan:

1. Elevation
2. Slope
3. Land cover
4. NDVI

`distance_to_water` tidak digunakan dalam model corridor.

### Elevation

Sumber DEM:

```text
COPERNICUS/DEM/GLO30_2024_1
```

Elevation diekspor pada target analysis grid 10 m.

### Slope

Slope dihitung dari DEM menggunakan terrain derivative Google Earth Engine.

### Land Cover

Sumber:

```text
ESA WorldCover 2020
```

### NDVI

NDVI dihitung dari Sentinel-2 untuk periode yang ditentukan pada module predictor.

Output:

```text
data/output_rasters/corridor/
├── corridor_elevation.tif
├── corridor_slope.tif
├── corridor_landcover_worldcover_2020.tif
└── corridor_ndvi.tif
```

---

## 4. P6-QA/QC — Predictor Validation

File:

```text
py/corridor/predictors_qa.py
```

Run:

```bash
python -m py.corridor.predictors_qa
```

Tahap ini melakukan validation terhadap predictor P6.

Parameter yang diperiksa:

```text
CRS
resolution
transform
width
height
bounds
dtype
NoData
valid pixel count
```

Tujuan utama adalah memastikan predictor kompatibel sebelum masuk ke normalization.

---

## 5. P7 — Predictor Normalization

File:

```text
py/corridor/normalization.py
```

Run:

```bash
python -m py.corridor.normalization
```

P7 membaca predictor langsung dari:

```text
data/output_rasters/corridor/
```

Predictor dinormalisasi menjadi ecological suitability score:

```text
0 ≤ suitability ≤ 1
```

### Elevation

Metode:

```text
inverse min-max
```

Interpretasi:

```text
lower elevation → higher suitability
higher elevation → lower suitability
```

### Slope

Metode:

```text
inverse min-max
```

Interpretasi:

```text
lower slope → higher suitability
higher slope → lower suitability
```

### Land Cover

WorldCover direklasifikasi menggunakan suitability lookup table.

### NDVI

NDVI menggunakan percentile normalization:

```text
P5 → 0
P95 → 1
```

Nilai di bawah P5 atau di atas P95 di-clamp ke `[0,1]`.

Output:

```text
data/output_rasters/corridor/normalized/
├── elevation_suitability.tif
├── slope_suitability.tif
├── landcover_suitability.tif
└── ndvi_suitability.tif
```

---

## 6. P7-QA/QC — Normalization Validation

File:

```text
py/corridor/normalization_qa.py
```

Run:

```bash
python -m py.corridor.normalization_qa
```

Validation meliputi:

```text
CRS
grid
dimensions
transform
NoData
valid pixels
minimum
maximum
common valid mask
```

Khusus suitability raster, nilai valid harus berada pada:

```text
0–1
```

---

## 7. P7 Diagnostic

File:

```text
py/corridor/normalization_diagnostic.py
```

Run:

```bash
python -m py.corridor.normalization_diagnostic
```

Diagnostic digunakan untuk memeriksa distribusi dan karakteristik predictor setelah normalization.

Tahap ini bersifat diagnostic dan tidak mengubah predictor utama.

Tujuannya memastikan transformasi:

```text
raw predictor
      ↓
suitability
```

tidak menghasilkan distribusi yang tidak wajar sebelum resistance surface dibangun.

---

## 8. P8.1 — Individual Resistance

File:

```text
py/corridor/resistance.py
```

Run:

```bash
python -m py.corridor.resistance
```

Resistance dihitung menggunakan:

```text
resistance = 1 - suitability
```

Predictor:

```text
elevation
slope
landcover
NDVI
```

Tidak ada `distance_to_water`.

Output:

```text
data/output_rasters/corridor/resistance/
├── elevation_resistance.tif
├── slope_resistance.tif
├── landcover_resistance.tif
└── ndvi_resistance.tif
```

Interpretasi:

```text
0 → low movement resistance
1 → high movement resistance
```

---

## 9. P8-QA/QC — Individual Resistance

File:

```text
py/corridor/resistance_qa.py
```

Run:

```bash
python -m py.corridor.resistance_qa
```

Validation mencakup:

```text
CRS
grid
dimensions
transform
NoData
valid pixel count
range
common valid mask
```

Tujuan utamanya memastikan keempat resistance raster dapat digunakan secara konsisten untuk composite resistance.

---

## 10. P8.2 — Composite Resistance

File:

```text
py/corridor/composite_resistance.py
```

Run:

```bash
python -m py.corridor.composite_resistance
```

Composite resistance menggunakan satu skenario:

```text
literature-informed weighted
```

Bobot:

```text
elevation   = 0.11
slope       = 0.17
landcover   = 0.43
NDVI        = 0.29
────────────────────
total       = 1.00
```

Formula:

```text
R =

    0.11 × elevation resistance
  + 0.17 × slope resistance
  + 0.43 × landcover resistance
  + 0.29 × NDVI resistance
```

Output:

```text
data/output_rasters/corridor/resistance/
└── composite_resistance_literature_weighted.tif
```

Tidak ada equal-weight scenario dalam pipeline utama.

---

## 11. P8.2-QA/QC — Composite Resistance

File:

```text
py/corridor/composite_resistance_qa.py
```

Run:

```bash
python -m py.corridor.composite_resistance_qa
```

Validation:

```text
grid consistency
common valid mask
NoData
minimum
maximum
percentiles
mean
valid pixel count
coverage
```

Tahap ini memastikan composite resistance valid sebelum digunakan sebagai cost surface.

---

## 12. P9.1 — KHL Source Mask

File:

```text
py/corridor/source.py
```

Run:

```bash
python -m py.corridor.source
```

Source ecological area:

```text
KHL_PP_tangse_meureudu.geojson
```

Source mask dibatasi oleh valid domain dari resistance surface.

Output:

```text
data/output_rasters/corridor/connectivity/source/
└── khl_source_mask.tif
```

Source mask digunakan sebagai titik awal/source dalam cost-distance calculation.

---

## 13. P9.2 — Source-Based Cost Distance

File:

```text
py/corridor/cost_distance.py
```

Run:

```bash
python -m py.corridor.cost_distance
```

Input:

```text
composite_resistance_literature_weighted.tif
khl_source_mask.tif
```

Cost-distance dihitung pada metric grid:

```text
CRS        : EPSG:32647
Resolution : 10 m
```

Algorithm:

```text
Dijkstra
```

Raster connectivity:

```text
8-neighbor
```

Step distance:

```text
cardinal  = 1
diagonal  = √2
```

Edge cost:

```text
mean resistance × step distance
```

Output:

```text
data/output_rasters/corridor/connectivity/
└── literature_weighted/
    └── cost_distance.tif
```

Istilah yang digunakan:

```text
source-based cost distance
```

Bukan:

```text
least-cost path
```

karena pipeline tidak memiliki destination dataset.

---

## 14. P9.2-QA/QC — Cost Distance

File:

```text
py/corridor/cost_distance_qa.py
```

Run:

```bash
python -m py.corridor.cost_distance_qa
```

Validation mencakup:

```text
CRS
grid
NoData
source pixels
non-source pixels
cost statistics
source cost
cost range
```

Kondisi utama:

```text
source cells → cost = 0
non-source cells → positive cumulative cost
```

Cost-distance menggunakan metric grid `EPSG:32647` untuk memastikan step distance dihitung dalam satuan meter.

Perbedaan kecil pada boundary antara domain resistance dan cost-distance dapat terjadi akibat reprojection dari `EPSG:4326` ke `EPSG:32647`.

---

## 15. P9.3 — Potential Connectivity

File:

```text
py/corridor/potential.py
```

Run:

```bash
python -m py.corridor.potential
```

Input:

```text
cost_distance.tif
khl_source_mask.tif
```

Source mask direproject ke grid cost-distance `EPSG:32647` sebelum digunakan.

Konsep:

```text
source-based cost distance
            ↓
cost normalization
            ↓
potential connectivity
```

Formula:

```text
potential = 1 - (cost / maximum non-source cost)
```

Source pixels secara eksplisit diberikan:

```text
potential = 1.0
```

Interpretasi:

```text
1 → highest relative potential connectivity
0 → lowest relative potential connectivity
```

Potential connectivity merupakan relative index yang diturunkan dari cumulative movement cost.

Output:

```text
data/output_rasters/corridor/connectivity/
└── literature_weighted/
    ├── cost_distance.tif
    └── corridor_potential.tif
```

Potential connectivity bukan:

```text
probability of elephant movement
probability of habitat use
habitat suitability
least-cost path
corridor classification
```

---

## 16. Complete Execution Sequence

Untuk menjalankan seluruh pipeline corridor:

```bash
python -m py.corridor.boundary

python -m py.corridor.predictors
python -m py.corridor.predictors_qa

python -m py.corridor.normalization
python -m py.corridor.normalization_qa
python -m py.corridor.normalization_diagnostic

python -m py.corridor.resistance
python -m py.corridor.resistance_qa

python -m py.corridor.composite_resistance
python -m py.corridor.composite_resistance_qa

python -m py.corridor.source

python -m py.corridor.cost_distance
python -m py.corridor.cost_distance_qa

python -m py.corridor.potential
```

Tidak terdapat tahap stack/resampling terpisah dalam pipeline.

---

## 17. Data Flow

Struktur data utama:

```text
data/
├── input_vectors/
│   └── KHL_PP.geojson
│
├── output_vectors/
│   ├── KHL_PP_tangse_meureudu.geojson
│   └── tangse_meureudu_roi.geojson
│
└── output_rasters/
    └── corridor/
        │
        ├── corridor_elevation.tif
        ├── corridor_slope.tif
        ├── corridor_landcover_worldcover_2020.tif
        ├── corridor_ndvi.tif
        │
        ├── normalized/
        │   ├── elevation_suitability.tif
        │   ├── slope_suitability.tif
        │   ├── landcover_suitability.tif
        │   └── ndvi_suitability.tif
        │
        └── resistance/
            ├── elevation_resistance.tif
            ├── slope_resistance.tif
            ├── landcover_resistance.tif
            ├── ndvi_resistance.tif
            └── composite_resistance_literature_weighted.tif
                │
                └── connectivity/
                    ├── source/
                    │   └── khl_source_mask.tif
                    │
                    └── literature_weighted/
                        ├── cost_distance.tif
                        └── corridor_potential.tif
```

---

## 18. Current Methodological Configuration

| Component               | Configuration                         |
| ----------------------- | ------------------------------------- |
| Analysis domain         | Tangse–Meureudu watershed             |
| KHL source              | KHL ∩ Tangse–Meureudu                 |
| Predictor CRS           | EPSG:4326                             |
| Predictor resolution    | 10 m                                  |
| Connectivity CRS        | EPSG:32647                            |
| Connectivity resolution | 10 m                                  |
| Elevation               | Copernicus GLO-30                     |
| Slope                   | DEM-derived                           |
| Land cover              | ESA WorldCover 2020                   |
| Vegetation              | Sentinel-2 NDVI                       |
| Distance to water       | Not used                              |
| Normalization           | Inverse min-max / P5–P95 / lookup     |
| Resistance              | `1 - suitability`                     |
| Composite predictors    | 4                                     |
| Composite scenario      | Literature-informed weighted          |
| Connectivity source     | KHL                                   |
| Destination             | None                                  |
| Cost algorithm          | Dijkstra                              |
| Neighborhood            | 8-neighbor                            |
| Cost distance           | Source-based cumulative movement cost |
| Potential output        | Relative potential connectivity       |

---

## 19. Execution Rule

Setiap tahap harus selesai dan lolos QA/QC sebelum tahap berikutnya dijalankan.

```text
Generate
   ↓
QA/QC
   ↓
Transform
   ↓
QA/QC
   ↓
Next stage
```

Pipeline menggunakan dua konteks grid:

```text
Predictor / resistance
        EPSG:4326
        10 m
           ↓
    metric reconstruction
           ↓
Connectivity analysis
        EPSG:32647
        10 m
```

Pemisahan ini penting karena cumulative movement cost menggunakan jarak antar-pixel dalam satuan meter.

Dengan demikian pipeline dapat ditelusuri, setiap tahap dapat diverifikasi secara independen, dan perubahan pada input atau parameter dapat diperiksa sebelum memengaruhi tahap berikutnya.
