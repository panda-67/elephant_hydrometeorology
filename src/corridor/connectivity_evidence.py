from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.stats import pearsonr, spearmanr


class ConnectivityEvidence:
    """Compare least-cost and circuit-theory connectivity evidence."""

    def __init__(
        self,
        corridor_potential_path: Path,
        circuit_connectivity_path: Path,
        current_density_path: Path,
        source_mask_path: Path,
        output_dir: Path,
        percentile: float = 90.0,
    ) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100.")

        self.root = Path(__file__).resolve().parents[2]

        self.corridor_potential_path = Path(corridor_potential_path).resolve()
        self.circuit_connectivity_path = Path(circuit_connectivity_path).resolve()
        self.current_density_path = Path(current_density_path).resolve()
        self.source_mask_path = Path(source_mask_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.percentile = percentile

        self.aligned_potential_path = self.output_dir / "corridor_potential_aligned.tif"
        self.agreement_path = self.output_dir / "connectivity_agreement.tif"
        self.current_density_high_path = self.output_dir / "current_density_high.tif"
        self.summary_path = self.output_dir / "connectivity_evidence_summary.json"

        self.inputs = {
            "corridor_potential": self.corridor_potential_path,
            "circuit_connectivity": self.circuit_connectivity_path,
            "current_density": self.current_density_path,
            "source_mask": self.source_mask_path,
        }

    def _validate_inputs(self) -> None:
        missing = [str(path) for path in self.inputs.values() if not path.is_file()]

        if missing:
            raise FileNotFoundError(
                "Required inputs are missing:\n" + "\n".join(missing)
            )

    @staticmethod
    def _same_grid(ds, reference) -> bool:
        return (
            ds.width == reference.width
            and ds.height == reference.height
            and ds.crs == reference.crs
            and ds.transform == reference.transform
        )

    def _align_corridor_potential(self) -> dict:
        """Reproject the continuous least-cost index to the circuit grid."""

        with (
            rasterio.open(self.corridor_potential_path) as src,
            rasterio.open(self.circuit_connectivity_path) as reference,
        ):
            if src.crs is None or reference.crs is None:
                raise ValueError("All rasters must have a defined CRS.")

            destination = np.full(
                (reference.height, reference.width),
                np.nan,
                dtype=np.float32,
            )

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )

            profile = reference.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=-9999.0,
                compress="deflate",
                predictor=3,
                BIGTIFF="IF_SAFER",
            )

            destination[~np.isfinite(destination)] = -9999.0

            with rasterio.open(self.aligned_potential_path, "w", **profile) as dst:
                dst.write(destination, 1)

            return {
                "source_crs": str(src.crs),
                "target_crs": str(reference.crs),
                "target_width": reference.width,
                "target_height": reference.height,
                "resampling": "bilinear",
                "output": str(self.aligned_potential_path),
            }

    def _read_inputs(self):
        with (
            rasterio.open(self.circuit_connectivity_path) as circuit_ds,
            rasterio.open(self.current_density_path) as current_ds,
            rasterio.open(self.source_mask_path) as source_ds,
            rasterio.open(self.aligned_potential_path) as potential_ds,
        ):
            for name, ds in (
                ("current_density", current_ds),
                ("source_mask", source_ds),
                ("aligned_corridor_potential", potential_ds),
            ):
                if not self._same_grid(ds, circuit_ds):
                    raise ValueError(
                        f"Grid mismatch: {name} does not match circuit connectivity."
                    )

            circuit = circuit_ds.read(1, masked=True).filled(np.nan)
            current = current_ds.read(1, masked=True).filled(np.nan)
            potential = potential_ds.read(1, masked=True).filled(np.nan)
            source = source_ds.read(1, masked=True).filled(0)

            profile = circuit_ds.profile.copy()

        valid = (
            np.isfinite(potential)
            & np.isfinite(circuit)
            & np.isfinite(current)
            & np.isfinite(source)
        )

        # Exclude source habitat: source cells are boundary conditions,
        # not independent evidence of a connecting corridor.
        non_source = source < 1
        analysis_mask = valid & non_source

        if not np.any(analysis_mask):
            raise ValueError("No common valid non-source cells.")

        return potential, circuit, current, analysis_mask, profile

    @staticmethod
    def _correlation(x: np.ndarray, y: np.ndarray) -> dict:
        """Compute correlations on a deterministic sample when needed."""

        sample_limit = 1_000_000

        if x.size > sample_limit:
            indices = np.linspace(0, x.size - 1, sample_limit, dtype=np.int64)
            x_sample = x[indices]
            y_sample = y[indices]
        else:
            x_sample = x
            y_sample = y

        if x_sample.size < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_cells": int(x_sample.size),
            }

        pearson = pearsonr(x_sample, y_sample)
        spearman = spearmanr(x_sample, y_sample)

        return {
            "status": "PASS",
            "sample_cells": int(x_sample.size),
            "sampling": (
                "deterministic evenly spaced sample"
                if x.size > sample_limit
                else "all common valid non-source cells"
            ),
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
        }

    def run(self) -> Path:
        started = perf_counter()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        print("=" * 70)
        print("CONNECTIVITY EVIDENCE ANALYSIS")
        print(f"Started: {timestamp}")
        print(f"High-value percentile: {self.percentile}")
        print("=" * 70)

        self._validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        alignment = self._align_corridor_potential()

        (
            potential,
            circuit,
            current,
            mask,
            profile,
        ) = self._read_inputs()

        # Thresholds are calculated separately for each indicator.
        potential_values = potential[mask]
        circuit_values = circuit[mask]
        current_values = current[mask]

        potential_threshold = float(np.percentile(potential_values, self.percentile))
        circuit_threshold = float(np.percentile(circuit_values, self.percentile))
        current_threshold = float(np.percentile(current_values, self.percentile))

        potential_high = mask & (potential >= potential_threshold)
        circuit_high = mask & (circuit >= circuit_threshold)
        current_high = mask & (current >= current_threshold)

        # 0 = neither high
        # 1 = least-cost potential only
        # 2 = circuit connectivity only
        # 3 = both indicators high
        # 255 = NoData / outside the analysis domain
        agreement = np.full(mask.shape, 255, dtype=np.uint8)
        agreement[mask] = 0
        agreement[potential_high & ~circuit_high] = 1
        agreement[circuit_high & ~potential_high] = 2
        agreement[potential_high & circuit_high] = 3

        current_density_high = np.full(mask.shape, 255, dtype=np.uint8)
        current_density_high[mask] = 0
        current_density_high[current_high] = 1

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

        with rasterio.open(self.agreement_path, "w", **output_profile) as dst:
            dst.write(agreement, 1)

        with rasterio.open(
            self.current_density_high_path, "w", **output_profile
        ) as dst:
            dst.write(current_density_high, 1)

        potential_high_count = int(np.count_nonzero(potential_high))
        circuit_high_count = int(np.count_nonzero(circuit_high))
        overlap_count = int(np.count_nonzero(potential_high & circuit_high))

        x = potential_values.astype(np.float64, copy=False)
        y = circuit_values.astype(np.float64, copy=False)

        correlation = self._correlation(x, y)

        counts = {
            "analysis_cells": int(np.count_nonzero(mask)),
            "least_cost_high_cells": potential_high_count,
            "circuit_connectivity_high_cells": circuit_high_count,
            "both_high_cells": overlap_count,
            "least_cost_only_cells": int(
                np.count_nonzero(potential_high & ~circuit_high)
            ),
            "circuit_only_cells": int(np.count_nonzero(circuit_high & ~potential_high)),
            "neither_high_cells": int(
                np.count_nonzero(mask & ~potential_high & ~circuit_high)
            ),
            "high_current_density_cells": int(np.count_nonzero(current_high)),
        }

        summary = {
            "started_at": timestamp,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "analysis_method": (
                "Separate percentile thresholds for least-cost potential, "
                "circuit connectivity, and current density. The agreement "
                "map compares least-cost potential with circuit connectivity. "
                "Current density remains a separate circuit-model diagnostic."
            ),
            "threshold_percentile": self.percentile,
            "thresholds": {
                "corridor_potential": potential_threshold,
                "circuit_connectivity": circuit_threshold,
                "current_density": current_threshold,
            },
            "counts": counts,
            "overlap_fraction_of_analysis_domain": (
                overlap_count / counts["analysis_cells"]
            ),
            "correlation": correlation,
            "alignment": alignment,
            "outputs": {
                "aligned_corridor_potential": str(self.aligned_potential_path),
                "connectivity_agreement": str(self.agreement_path),
                "current_density_high": str(self.current_density_high_path),
            },
            "methodological_notes": [
                "Corridor potential is bilinearly reprojected to the "
                "circuit-connectivity grid.",
                "Source habitat is excluded from threshold estimation "
                "and agreement classification.",
                "Cost distance is not counted as an independent indicator "
                "because corridor potential is derived from cost distance.",
                "Current density is not counted as an independent model "
                "because it comes from the same Circuit Theory model.",
                "Percentile thresholds are operational classifications, "
                "not empirically validated ecological thresholds.",
                "The agreement map represents model-indicator agreement, "
                "not confirmed elephant movement or occurrence probability.",
            ],
        }

        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print(f"\n[RESULT] Analysis cells: {counts['analysis_cells']:,}")
        print(f"[RESULT] Both indicators high: {counts['both_high_cells']:,}")
        print(f"[RESULT] Least-cost only: {counts['least_cost_only_cells']:,}")
        print(f"[RESULT] Circuit only: {counts['circuit_only_cells']:,}")
        print(f"[OUTPUT] Agreement: {self.agreement_path}")
        print(f"[OUTPUT] Summary: {self.summary_path}")
        print(f"[TIME] {perf_counter() - started:.2f} seconds")

        return self.summary_path
