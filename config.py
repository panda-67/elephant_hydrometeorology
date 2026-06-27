from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple


class GEEConfig(BaseSettings):
    # Meta Project (Nama variabel harus cocok dengan yang ada di kelas atau diizinkan lewat config)
    PROJECT_ID: str = "default-project"
    OUTPUT_DIR: str = "./data/output_metrics"

    # Atribut penampung tambahan agar pydantic mengenali variabel dari .env Anda
    gee_project_id: str = "default-project"
    data_output_dir: str = "./data/output_metrics"
    log_dir: str = "./logs"
    debug_mode: str = "True"

    # TIMELINE FORENSIK MULTI-FASE (Sesuai Cetak Biru Matriks Analisis)
    # 1. Baseline Fase
    F_BASELINE_START: str = "2020-01-01"
    F_BASELINE_END: str = "2020-12-31"

    # 2. Pre-Event Fase (Akumulasi Degradasi Lahan)
    F_PRE_EVENT_START: str = "2025-07-01"
    F_PRE_EVENT_END: str = "2025-11-20"

    # 3. Flood Event Fase (Puncak Hujan & Simulasi Limpasan)
    F_FLOOD_EVENT_START: str = "2025-11-21"
    F_FLOOD_EVENT_END: str = "2025-12-15"

    # 4. Post-Event Fase (Genangan Hilir & Sedimen)
    F_POST_EVENT_START: str = "2025-12-16"
    F_POST_EVENT_END: str = "2026-01-15"

    # Parameter Hidrologi (Skenario Curah Hujan Ekstrem Batas Atas)
    PEAK_RAINFALL_MM_DAY: float = 122.00

    # Ambang Batas Saintifik (Thresholds)
    CLOUD_PROB_THRESHOLD: int = 35
    NDVI_DEGRADATION_THRESHOLD: float = -0.1
    SATELLITE_MODE: str = "sentinel1"  # sentinel1, sentinel2, landsat
    USE_DEMNAS: bool = False

    das_pidie_plus: List[Tuple[float, float]] = [
        # (95.8514831, 5.1869573),  # Lhok Keutapang, Tangse
        (96.0849351, 5.2044313),  # Sarah Panyang
        (95.9369891, 5.1533933),  # Tiro, Pidie
        (95.9794531, 5.2757963),  # Beureunuen
        (96.1381043, 5.2735685),  # Pante Raja
        (96.1813460, 5.2591846),  # Trienggadeng
        (96.2216408, 5.2411777),  # Kuta Trieng
    ]

    das_meureudu: List[Tuple[float, float]] = [
        # (96.0638381, 5.2023043),  # Meunasah Jijiem, Bandar Baru
        (96.2547393, 5.2314908),  # Meureudu
        (96.2025621, 5.0877253),  # Hutan Meureudu
        (96.2359233, 4.9971973),  # Huta Meureudu Atas
    ]

    # Koordinat yang akan digunakan untuk pengenalan hydrosheds dari database
    # WWF/HydroSHEDS/v1/Basins/hybas_12
    OUTLET_COORDINATES: List[Tuple[float, float]] = das_meureudu + das_pidie_plus

    # Menggunakan SettingsConfigDict bawaan Pydantic v2 untuk melonggarkan pembacaan .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # <--- MENGIZINKAN INPUT EXTRA DARI .ENV AGAR TIDAK ERROR
        case_sensitive=False,  # <--- Mengabaikan perbedaan huruf besar/kecil antara .env dan python
    )

    # Helper method untuk mengalihkan sinkronisasi parameter dinamis
    def __init__(self, **values):
        super().__init__(**values)
        # Jika pydantic membaca 'gee_project_id' dari .env, timpa nilai PROJECT_ID utama
        if self.gee_project_id and self.gee_project_id != "default-project":
            self.PROJECT_ID = self.gee_project_id
        if self.data_output_dir:
            self.OUTPUT_DIR = self.data_output_dir


config = GEEConfig()
