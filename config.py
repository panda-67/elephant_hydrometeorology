from typing import Dict, List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class GEEConfig(BaseSettings):
    # Meta Project
    PROJECT_ID: str = "default-project"
    OUTPUT_DIR: str = "./data/output_metrics"

    # Atribut penampung tambahan
    gee_project_id: str = "default-project"
    data_output_dir: str = "./data/output_metrics"
    log_dir: str = "./logs"
    debug_mode: str = "True"

    # ============================================================
    # TIMELINE FORENSIK MULTI-FASE
    # ============================================================

    # 1. Baseline Fase
    F_BASELINE_START: str = "2020-01-01"
    F_BASELINE_END: str = "2020-12-31"

    # 2. Pre-Event Fase
    F_PRE_EVENT_START: str = "2025-07-01"
    F_PRE_EVENT_END: str = "2025-10-31"

    # 3. Flood Event Fase
    F_FLOOD_EVENT_START: str = "2025-11-01"
    F_FLOOD_EVENT_END: str = "2025-11-30"

    # 4. Post-Event Fase
    F_POST_EVENT_START: str = "2025-12-01"
    F_POST_EVENT_END: str = "2026-01-15"

    # ============================================================
    # PARAMETER HIDROLOGI
    # ============================================================

    PEAK_RAINFALL_MM_DAY: float = 122.00

    # ============================================================
    # AMBANG BATAS SAINTIFIK
    # ============================================================

    CLOUD_PROB_THRESHOLD: int = 35
    NDVI_DEGRADATION_THRESHOLD: float = -0.1

    SATELLITE_MODE: str = "sentinel2"  # sentinel1, sentinel2, landsat
    USE_DEMNAS: bool = False

    # ============================================================
    # CAUSAL EVIDENCE → CORRIDOR RESISTANCE
    # ============================================================
    #
    # causal_evidence_tier berasal dari SpatialCausalPipeline:
    #
    # Tier 0 = tidak ada qualifying disturbance evidence
    # Tier 1 = vegetation degradation
    # Tier 2 = post-event destruction
    # Tier 3 = compound disturbance
    # Tier 4 = hydrologically confirmed compound disturbance
    #
    # Factor digunakan untuk meningkatkan literature-weighted
    # baseline resistance:
    #
    # adjusted_resistance =
    #     baseline_resistance * causal_resistance_factor[tier]
    #
    # Tier 0 harus selalu 1.0 karena tidak boleh mengubah
    # baseline resistance.
    #
    # Nilai ini merupakan evidence-weighted adjustment,
    # bukan estimasi langsung perubahan perilaku gajah.
    #

    CAUSAL_RESISTANCE_FACTORS: Dict[int, float] = {
        0: 1.00,
        1: 1.10,
        2: 1.20,
        3: 1.40,
        4: 1.60,
    }

    CAUSAL_TIER_NAMES: Dict[int, str] = {
        0: "no_disturbance_evidence",
        1: "vegetation_degradation",
        2: "post_event_destruction",
        3: "compound_disturbance",
        4: "hydrologically_confirmed_compound_disturbance",
    }

    # ============================================================
    # SPATIAL INPUT / DAS
    # ============================================================

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

    # ============================================================
    # PYDANTIC SETTINGS
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    def __init__(self, **values):
        super().__init__(**values)

        if self.gee_project_id and self.gee_project_id != "default-project":
            self.PROJECT_ID = self.gee_project_id

        if self.data_output_dir:
            self.OUTPUT_DIR = self.data_output_dir


config = GEEConfig()
