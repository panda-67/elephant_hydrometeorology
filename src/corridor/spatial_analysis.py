from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from scipy.ndimage import label


class CorridorSpatialAnalysis:
    """
    Spatial synthesis of least-cost and circuit-theory connectivity.

    The analysis integrates:
      - least-cost corridor potential
      - circuit-theory connectivity
      - circuit-theory current density
      - P65/P75/P90 connectivity agreement classes

    Class 3 from ConnectivityEvidence represents cells where both
    corridor potential and circuit connectivity are above their
    respective percentile thresholds.

    A bottleneck candidate is conservatively defined as a cell that:
      1. belongs to Class 3, and
      2. has high current density for the corresponding threshold.

    These are connectivity concentration candidates, not confirmed
    biological bottlenecks.
    """

    def __init__(
        self,
        evidence_root: Path,
        current_density_path: Path,
        output_dir: Path,
        percentiles: tuple[int, ...] = (65, 75, 90),
        minimum_component_cells: int = 5,
    ) -> None:
        self.root = Path(__file__).resolve().parents[2]

        self.evidence_root = Path(evidence_root).resolve()
        self.current_density_path = Path(current_density_path).resolve()
        self.output_dir = Path(output_dir).resolve()

        self.percentiles = tuple(percentiles)
        self.minimum_component_cells = minimum_component_cells

        self.synthesis_path = self.output_dir / "corridor_spatial_synthesis.tif"
        self.bottleneck_synthesis_path = self.output_dir / "bottleneck_synthesis.tif"

        self.corridor_statistics_path = (
            self.output_dir / "corridor_spatial_statistics.json"
        )
        self.bottleneck_statistics_path = self.output_dir / "bottleneck_statistics.json"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        if not self.current_density_path.is_file():
            raise FileNotFoundError(
                f"Current density raster not found: {self.current_density_path}"
            )

        missing = []

        for percentile in self.percentiles:
            directory = self.evidence_root / f"p{percentile}"

            agreement = directory / "connectivity_agreement.tif"
            current_high = directory / "current_density_high.tif"

            if not agreement.is_file():
                missing.append(str(agreement))

            if not current_high.is_file():
                missing.append(str(current_high))

        if missing:
            raise FileNotFoundError(
                "Required evidence outputs are missing:\n" + "\n".join(missing)
            )

    @staticmethod
    def _same_grid(ds_a, ds_b) -> bool:
        return (
            ds_a.width == ds_b.width
            and ds_a.height == ds_b.height
            and ds_a.crs == ds_b.crs
            and ds_a.transform == ds_b.transform
        )

    # ------------------------------------------------------------------
    # Raster I/O
    # ------------------------------------------------------------------

    def _read_current_density(self):
        with rasterio.open(self.current_density_path) as ds:
            current = ds.read(1, masked=True).filled(np.nan)
            profile = ds.profile.copy()

            metadata = {
                "width": ds.width,
                "height": ds.height,
                "crs": str(ds.crs),
                "transform": tuple(ds.transform),
                "nodata": ds.nodata,
            }

        return current, profile, metadata

    def _read_evidence(
        self,
        percentile: int,
        reference_ds,
    ):
        directory = self.evidence_root / f"p{percentile}"

        agreement_path = directory / "connectivity_agreement.tif"
        current_high_path = directory / "current_density_high.tif"

        with (
            rasterio.open(agreement_path) as agreement_ds,
            rasterio.open(current_high_path) as current_high_ds,
        ):
            if not self._same_grid(agreement_ds, reference_ds):
                raise ValueError(
                    f"Grid mismatch in P{percentile} connectivity agreement."
                )

            if not self._same_grid(current_high_ds, reference_ds):
                raise ValueError(
                    f"Grid mismatch in P{percentile} current density classification."
                )

            agreement = agreement_ds.read(1)
            current_high = current_high_ds.read(1)

        return agreement, current_high

    # ------------------------------------------------------------------
    # Spatial analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _count_classes(
        raster: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict:
        values = raster[valid_mask]

        return {
            str(value): int(np.count_nonzero(values == value)) for value in range(4)
        }

    @staticmethod
    def _area_from_cells(
        cells: int,
        transform,
        crs,
    ) -> float:
        """
        Calculate area in hectares.

        For the current circuit grid in EPSG:4326, pixel area is
        approximated geodetically using rasterio's transform bounds
        only when a projected CRS is unavailable.

        The function intentionally reports approximate area for
        geographic grids and exact pixel area for projected grids.
        """

        if crs is not None and crs.is_projected:
            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)

            area_m2 = cells * pixel_width * pixel_height
            return area_m2 / 10_000.0

        # Geographic grid:
        # use a conservative nominal pixel area at the raster center.
        # This is only used for summary reporting; cell counts remain
        # the primary statistic.
        from math import cos, radians

        latitude = transform.f + transform.e * 0.5

        lat_height_deg = abs(transform.e)
        lon_width_deg = abs(transform.a)

        meters_per_degree_lat = 111_132.0
        meters_per_degree_lon = 111_320.0 * cos(radians(latitude))

        area_m2 = (
            cells
            * lat_height_deg
            * meters_per_degree_lat
            * lon_width_deg
            * meters_per_degree_lon
        )

        return area_m2 / 10_000.0

    @staticmethod
    def _connected_components(
        mask: np.ndarray,
        minimum_cells: int,
    ):
        """
        Identify spatially contiguous candidate components.

        Eight-neighbor connectivity is used to match the connectivity
        topology used by the circuit and cost-distance analyzes.
        """

        structure = np.ones((3, 3), dtype=np.uint8)

        labels, component_count = label(
            mask.astype(np.uint8),
            structure=structure,
        )

        if component_count == 0:
            return np.zeros(mask.shape, dtype=np.uint8), []

        component_sizes = np.bincount(
            labels.ravel(),
            minlength=component_count + 1,
        )

        retained = np.where(component_sizes >= minimum_cells)[0]

        retained = retained[retained != 0]

        retained_mask = np.isin(labels, retained)

        component_output = np.zeros(
            mask.shape,
            dtype=np.uint8,
        )

        component_output[retained_mask] = 1

        statistics = []

        for component_id in retained:
            size = int(component_sizes[component_id])

            statistics.append(
                {
                    "component_id": int(component_id),
                    "cells": size,
                }
            )

        return component_output, statistics

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_raster(
        self,
        path: Path,
        data: np.ndarray,
        profile: dict,
    ) -> None:
        output_profile = profile.copy()

        output_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(path, "w", **output_profile) as dst:
            dst.write(data.astype(np.uint8), 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def run(self) -> Path:
        started = perf_counter()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        print("=" * 70)
        print("CORRIDOR SPATIAL AND BOTTLENECK ANALYSIS")
        print(f"Started: {timestamp}")
        print(f"Thresholds: {self.percentiles}")
        print("=" * 70)

        self._validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.current_density_path) as reference_ds:
            reference_profile = reference_ds.profile.copy()
            reference_crs = reference_ds.crs
            reference_transform = reference_ds.transform

        current_density, _, _ = self._read_current_density()

        current_valid = np.isfinite(current_density)

        corridor_by_threshold = {}
        bottleneck_by_threshold = {}

        corridor_statistics = {}
        bottleneck_statistics = {}

        for percentile in self.percentiles:
            print(f"\nProcessing P{percentile}...")

            with rasterio.open(
                self.evidence_root / f"p{percentile}" / "connectivity_agreement.tif"
            ) as agreement_ds:
                agreement = agreement_ds.read(1)

            with rasterio.open(
                self.evidence_root / f"p{percentile}" / "current_density_high.tif"
            ) as current_high_ds:
                current_high = current_high_ds.read(1)

            valid = (agreement != 255) & current_valid & (current_high != 255)

            class3 = valid & (agreement == 3)

            # Candidate corridor:
            # Class 3 indicates simultaneous high least-cost potential
            # and high circuit connectivity.
            corridor = np.full(
                class3.shape,
                255,
                dtype=np.uint8,
            )
            corridor[valid] = 0
            corridor[class3] = 1

            # Current density high is calculated independently by the
            # ConnectivityEvidence percentile analysis.
            current_high_mask = valid & (current_high == 1)

            # Candidate bottleneck:
            # Class 3 + high current density.
            bottleneck_candidate = class3 & current_high_mask

            bottleneck = np.full(
                class3.shape,
                255,
                dtype=np.uint8,
            )
            bottleneck[valid] = 0

            retained_components, component_stats = self._connected_components(
                bottleneck_candidate,
                self.minimum_component_cells,
            )

            bottleneck[retained_components == 1] = 1

            corridor_path = self.output_dir / f"corridor_priority_p{percentile}.tif"

            bottleneck_path = (
                self.output_dir / f"bottleneck_candidate_p{percentile}.tif"
            )

            self._write_raster(
                corridor_path,
                corridor,
                reference_profile,
            )

            self._write_raster(
                bottleneck_path,
                bottleneck,
                reference_profile,
            )

            corridor_by_threshold[percentile] = class3
            bottleneck_by_threshold[percentile] = bottleneck == 1

            valid_cells = int(np.count_nonzero(valid))
            class3_cells = int(np.count_nonzero(class3))
            current_high_cells = int(np.count_nonzero(current_high_mask))
            candidate_cells = int(np.count_nonzero(bottleneck_candidate))
            retained_cells = int(np.count_nonzero(retained_components))

            corridor_statistics[str(percentile)] = {
                "percentile": percentile,
                "valid_cells": valid_cells,
                "class_3_cells": class3_cells,
                "class_3_percent": (
                    100.0 * class3_cells / valid_cells if valid_cells else 0.0
                ),
                "current_density_high_cells": current_high_cells,
                "candidate_bottleneck_cells": candidate_cells,
                "retained_bottleneck_cells": retained_cells,
                "corridor_area_ha": self._area_from_cells(
                    class3_cells,
                    reference_transform,
                    reference_crs,
                ),
                "bottleneck_candidate_area_ha": (
                    self._area_from_cells(
                        retained_cells,
                        reference_transform,
                        reference_crs,
                    )
                ),
                "outputs": {
                    "corridor_priority": str(corridor_path),
                    "bottleneck_candidate": str(bottleneck_path),
                },
            }

            bottleneck_statistics[str(percentile)] = {
                "percentile": percentile,
                "candidate_cells": candidate_cells,
                "retained_cells": retained_cells,
                "component_count": len(component_stats),
                "minimum_component_cells": (self.minimum_component_cells),
                "components": component_stats,
            }

        # --------------------------------------------------------------
        # Cross-threshold synthesis
        # --------------------------------------------------------------

        threshold_stack = np.stack(
            [corridor_by_threshold[p] for p in self.percentiles],
            axis=0,
        )

        bottleneck_stack = np.stack(
            [bottleneck_by_threshold[p] for p in self.percentiles],
            axis=0,
        )

        corridor_valid = np.any(
            threshold_stack,
            axis=0,
        )

        corridor_count = np.sum(
            threshold_stack,
            axis=0,
        ).astype(np.uint8)

        corridor_synthesis = np.full(
            corridor_count.shape,
            255,
            dtype=np.uint8,
        )

        corridor_synthesis[corridor_valid] = corridor_count[corridor_valid]

        bottleneck_valid = np.any(
            bottleneck_stack,
            axis=0,
        )

        bottleneck_count = np.sum(
            bottleneck_stack,
            axis=0,
        ).astype(np.uint8)

        bottleneck_synthesis = np.full(
            bottleneck_count.shape,
            255,
            dtype=np.uint8,
        )

        bottleneck_synthesis[bottleneck_valid] = bottleneck_count[bottleneck_valid]

        self._write_raster(
            self.synthesis_path,
            corridor_synthesis,
            reference_profile,
        )

        self._write_raster(
            self.bottleneck_synthesis_path,
            bottleneck_synthesis,
            reference_profile,
        )

        # --------------------------------------------------------------
        # Cross-threshold statistics
        # --------------------------------------------------------------

        synthesis_valid = corridor_synthesis != 255

        corridor_synthesis_statistics = {
            str(value): int(np.count_nonzero(corridor_synthesis == value))
            for value in range(len(self.percentiles) + 1)
        }

        bottleneck_synthesis_statistics = {
            str(value): int(np.count_nonzero(bottleneck_synthesis == value))
            for value in range(len(self.percentiles) + 1)
        }

        summary = {
            "analysis": "baseline corridor spatial and bottleneck analysis",
            "timestamp": timestamp,
            "percentiles": list(self.percentiles),
            "minimum_component_cells": (self.minimum_component_cells),
            "definitions": {
                "class_3": (
                    "Least-cost corridor potential and "
                    "circuit connectivity are both above "
                    "their respective percentile thresholds."
                ),
                "candidate_bottleneck": (
                    "Class 3 cell with high current density "
                    "for the corresponding percentile."
                ),
                "synthesis_value": (
                    "Number of retained thresholds in which "
                    "the cell satisfies the condition."
                ),
            },
            "corridor_by_threshold": corridor_statistics,
            "bottleneck_by_threshold": bottleneck_statistics,
            "cross_threshold_synthesis": {
                "corridor_cells_by_threshold_count": (corridor_synthesis_statistics),
                "bottleneck_cells_by_threshold_count": (
                    bottleneck_synthesis_statistics
                ),
            },
            "outputs": {
                "corridor_synthesis": str(self.synthesis_path),
                "bottleneck_synthesis": str(self.bottleneck_synthesis_path),
            },
            "runtime_seconds": (perf_counter() - started),
        }

        with self.corridor_statistics_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        with self.bottleneck_statistics_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "analysis": ("candidate bottleneck spatial statistics"),
                    "timestamp": timestamp,
                    "percentiles": list(self.percentiles),
                    "minimum_component_cells": (self.minimum_component_cells),
                    "by_threshold": bottleneck_statistics,
                    "cross_threshold": (bottleneck_synthesis_statistics),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print(f"Corridor synthesis: {self.synthesis_path}")
        print(f"Bottleneck synthesis: {self.bottleneck_synthesis_path}")
        print(f"Runtime: {perf_counter() - started:.2f} seconds")
        print("=" * 70)

        return self.synthesis_path
