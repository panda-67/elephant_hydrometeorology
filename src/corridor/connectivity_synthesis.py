from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio


class ConnectivitySynthesis:
    """Validate and summarize complementary corridor-connectivity outputs."""

    def __init__(
        self,
        resistance_path: Path,
        source_path: Path,
        cost_distance_path: Path,
        corridor_potential_path: Path,
        current_density_path: Path,
        circuit_connectivity_path: Path,
        flood_connectivity_path: Path,
        flood_connectivity_loss_path: Path,
        output_dir: Path,
    ) -> None:
        self.root = Path(__file__).resolve().parents[2]

        self.resistance_path = Path(resistance_path).resolve()
        self.source_path = Path(source_path).resolve()
        self.cost_distance_path = Path(cost_distance_path).resolve()
        self.corridor_potential_path = Path(corridor_potential_path).resolve()
        self.current_density_path = Path(current_density_path).resolve()
        self.circuit_connectivity_path = Path(circuit_connectivity_path).resolve()
        self.flood_connectivity_path = Path(flood_connectivity_path).resolve()
        self.flood_connectivity_loss_path = Path(flood_connectivity_loss_path).resolve()
        self.output_dir = Path(output_dir).resolve()

        self.summary_path = self.output_dir / "connectivity_synthesis_summary.json"

        self.inputs = {
            "resistance": self.resistance_path,
            "source_mask": self.source_path,
            "cost_distance": self.cost_distance_path,
            "corridor_potential": self.corridor_potential_path,
            "current_density": self.current_density_path,
            "circuit_connectivity": self.circuit_connectivity_path,
            "flood_exposed_connectivity": self.flood_connectivity_path,
            "flood_induced_connectivity_loss": (self.flood_connectivity_loss_path),
        }

    @staticmethod
    def _stats(path: Path) -> dict:
        with rasterio.open(path) as ds:
            values = ds.read(1, masked=True).compressed().astype(np.float64)

            finite = np.isfinite(values)
            values = values[finite]

            if values.size == 0:
                raise ValueError(f"Raster has no finite valid values: {path}")

            return {
                "path": str(path),
                "width": ds.width,
                "height": ds.height,
                "crs": str(ds.crs),
                "transform": list(ds.transform)[:6],
                "nodata": (
                    float(ds.nodata)
                    if ds.nodata is not None and np.isfinite(ds.nodata)
                    else None
                ),
                "valid_finite_cells": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "zero_cells": int(np.count_nonzero(values == 0)),
                "negative_cells": int(np.count_nonzero(values < 0)),
            }

    def _validate_inputs(self) -> None:
        missing = [str(path) for path in self.inputs.values() if not path.is_file()]

        if missing:
            raise FileNotFoundError(
                "Required input rasters are missing:\n" + "\n".join(missing)
            )

    def _validate_grids(self) -> dict:
        """Validate rasters within their native spatial-grid groups."""

        grid_groups = {
            "geographic_epsg4326": {
                "reference": "resistance",
                "members": [
                    "resistance",
                    "source_mask",
                    "current_density",
                    "circuit_connectivity",
                ],
                "expected_crs": "EPSG:4326",
            },
            "metric_epsg32647": {
                "reference": "cost_distance",
                "members": [
                    "cost_distance",
                    "corridor_potential",
                    "flood_exposed_connectivity",
                    "flood_induced_connectivity_loss",
                ],
                "expected_crs": "EPSG:32647",
            },
        }

        checks = {}

        for group_name, group in grid_groups.items():
            reference_name = group["reference"]
            reference_path = self.inputs[reference_name]

            with rasterio.open(reference_path) as reference:
                reference_grid = {
                    "width": reference.width,
                    "height": reference.height,
                    "crs": reference.crs,
                    "transform": reference.transform,
                }

                if str(reference.crs) != group["expected_crs"]:
                    raise ValueError(
                        f"Unexpected CRS for grid group '{group_name}': "
                        f"expected {group['expected_crs']}, "
                        f"got {reference.crs} in {reference_path}"
                    )

                group_checks = {}

                for name in group["members"]:
                    path = self.inputs[name]

                    with rasterio.open(path) as ds:
                        result = {
                            "dimensions": (
                                ds.width == reference_grid["width"]
                                and ds.height == reference_grid["height"]
                            ),
                            "crs": ds.crs == reference_grid["crs"],
                            "transform": (ds.transform == reference_grid["transform"]),
                        }

                        result["valid"] = all(result.values())
                        result["path"] = str(path)

                        group_checks[name] = result

                failures = {
                    name: result
                    for name, result in group_checks.items()
                    if not result["valid"]
                }

                if failures:
                    raise ValueError(
                        f"Grid mismatch in group '{group_name}':\n"
                        + json.dumps(failures, indent=2)
                    )

                checks[group_name] = {
                    "status": "PASS",
                    "reference": reference_name,
                    "crs": str(reference_grid["crs"]),
                    "width": reference_grid["width"],
                    "height": reference_grid["height"],
                    "transform": list(reference_grid["transform"])[:6],
                    "members": group_checks,
                }

        return checks

    def _compare_baseline_flood(self, tolerance: float = 1e-5) -> dict:
        """Compare baseline and flood indices on their common valid grid."""

        delta_path = self.output_dir / "baseline_flood_delta.tif"
        mask_path = self.output_dir / "synthesis_valid_mask.tif"
        report_path = self.output_dir / "baseline_flood_comparison.json"

        def read_valid(ds):
            masked = ds.read(1, masked=True)
            values = np.asarray(
                masked.astype(np.float64).filled(np.nan),
                dtype=np.float64,
            )
            valid = ~np.ma.getmaskarray(masked)
            valid &= np.isfinite(values)
            return values, valid

        with (
            rasterio.open(self.corridor_potential_path) as baseline_ds,
            rasterio.open(self.flood_connectivity_path) as flood_ds,
            rasterio.open(self.flood_connectivity_loss_path) as loss_ds,
        ):
            # Enforce identical grids; do not silently resample.
            for label, ds in (
                ("flood connectivity", flood_ds),
                ("flood connectivity loss", loss_ds),
            ):
                if (
                    ds.crs != baseline_ds.crs
                    or ds.width != baseline_ds.width
                    or ds.height != baseline_ds.height
                    or ds.transform != baseline_ds.transform
                ):
                    raise ValueError(
                        f"Grid mismatch for {label}. "
                        "Baseline, flood, and loss rasters must have "
                        "identical CRS, dimensions, and transform."
                    )

            baseline, baseline_valid = read_valid(baseline_ds)
            flood, flood_valid = read_valid(flood_ds)
            recorded_loss, loss_valid = read_valid(loss_ds)

            # Restrict comparisons to cells valid in all three rasters.
            common_valid = baseline_valid & flood_valid & loss_valid

            if not np.any(common_valid):
                raise ValueError(
                    "No common valid cells across baseline, flood, "
                    "and connectivity-loss rasters."
                )

            baseline_common = baseline[common_valid]
            flood_common = flood[common_valid]
            recorded_loss_common = recorded_loss[common_valid]

            # Signed delta: positive means a lower flood connectivity index.
            delta = baseline - flood
            delta_common = delta[common_valid]

            expected_loss = np.clip(delta_common, 0.0, 1.0)
            loss_error = recorded_loss_common - expected_loss

            unexpected_increases = delta_common < -tolerance
            small_negative_deltas = (delta_common < 0.0) & (delta_common >= -tolerance)
            loss_mismatches = np.abs(loss_error) > tolerance

            def describe(values: np.ndarray) -> dict:
                return {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                }

            delta_profile = baseline_ds.profile.copy()
            delta_profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=-9999.0,
                compress="deflate",
                predictor=3,
            )

            delta_output = np.full(
                delta.shape,
                -9999.0,
                dtype=np.float32,
            )
            delta_output[common_valid] = delta_common.astype(np.float32)

            with rasterio.open(delta_path, "w", **delta_profile) as dst:
                dst.write(delta_output, 1)
                dst.set_band_description(1, "baseline_minus_flood_connectivity_index")

            mask_profile = baseline_ds.profile.copy()
            mask_profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=None,
                compress="deflate",
            )

            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(common_valid.astype(np.uint8), 1)
                dst.set_band_description(1, "common_valid_mask_baseline_flood_loss")

            report = {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "method": {
                    "baseline": str(self.corridor_potential_path),
                    "flood": str(self.flood_connectivity_path),
                    "recorded_loss": str(self.flood_connectivity_loss_path),
                    "delta_formula": "baseline_corridor_potential - flood_exposed_connectivity",
                    "expected_loss_formula": "clip(delta, 0, 1)",
                    "valid_domain": (
                        "Cells valid in baseline, flood, and loss rasters"
                    ),
                    "tolerance": tolerance,
                    "interpretation": (
                        "Relative connectivity-index difference, not a probability."
                    ),
                },
                "grid": {
                    "crs": str(baseline_ds.crs),
                    "width": baseline_ds.width,
                    "height": baseline_ds.height,
                    "transform": list(baseline_ds.transform)[:6],
                },
                "validity": {
                    "baseline_valid_cells": int(baseline_valid.sum()),
                    "flood_valid_cells": int(flood_valid.sum()),
                    "loss_valid_cells": int(loss_valid.sum()),
                    "common_valid_cells": int(common_valid.sum()),
                },
                "baseline": describe(baseline_common),
                "flood": describe(flood_common),
                "delta": describe(delta_common),
                "delta_diagnostics": {
                    "decreased_connectivity_cells": int(
                        np.count_nonzero(delta_common > tolerance)
                    ),
                    "unchanged_within_tolerance_cells": int(
                        np.count_nonzero(np.abs(delta_common) <= tolerance)
                    ),
                    "unexpected_increase_cells": int(unexpected_increases.sum()),
                    "small_negative_delta_cells": int(small_negative_deltas.sum()),
                    "maximum_unexpected_increase": float(
                        max(0.0, -float(delta_common.min()))
                    ),
                },
                "loss_consistency": {
                    "cells_checked": int(common_valid.sum()),
                    "mismatch_cells": int(loss_mismatches.sum()),
                    "maximum_absolute_difference": float(np.max(np.abs(loss_error))),
                    "status": ("PASS" if not np.any(loss_mismatches) else "REVIEW"),
                },
                "outputs": {
                    "delta_raster": str(delta_path),
                    "valid_mask_raster": str(mask_path),
                    "comparison_report": str(report_path),
                },
            }

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        print("\n[COMPARISON] Baseline vs flood")
        print(f"  Common valid cells : {report['validity']['common_valid_cells']:,}")
        print(
            "  Unexpected increases : "
            f"{report['delta_diagnostics']['unexpected_increase_cells']:,}"
        )
        print(f"  Loss consistency : {report['loss_consistency']['status']}")
        print(f"  Delta raster : {delta_path}")
        print(f"  Valid mask   : {mask_path}")
        print(f"  Report       : {report_path}")

        return report

    def run(self) -> Path:
        started = perf_counter()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        print("=" * 70)
        print("CONNECTIVITY SYNTHESIS — INPUT VALIDATION")
        print(f"Started at: {timestamp}")
        print("=" * 70)

        self._validate_inputs()
        grid_checks = self._validate_grids()

        raster_stats = {}

        for name, path in self.inputs.items():
            print(f"\n[ANALYSIS] {name}: {path.name}")
            raster_stats[name] = self._stats(path)

            stats = raster_stats[name]
            print(f"  Valid cells : {stats['valid_finite_cells']:,}")
            print(f"  Min         : {stats['min']:.8g}")
            print(f"  Max         : {stats['max']:.8g}")
            print(f"  Mean        : {stats['mean']:.8g}")
            print(f"  Median      : {stats['median']:.8g}")
            print(f"  Zero cells  : {stats['zero_cells']:,}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        comparison_report = self._compare_baseline_flood()

        summary = {
            "started_at": timestamp,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "reference_grid": str(self.resistance_path),
            "grid_validation": grid_checks,
            "raster_statistics": raster_stats,
            "baseline_flood_comparison": comparison_report,
            "synthesis_status": (
                "COMPARISON_REVIEW_REQUIRED"
                if comparison_report["loss_consistency"]["status"] == "REVIEW"
                or comparison_report["delta_diagnostics"]["unexpected_increase_cells"]
                > 0
                else "SYNTHESIS_COMPLETE"
            ),
            "methodological_note": (
                "Connectivity outputs are validated within two native "
                "spatial-grid groups: geographic EPSG:4326 and metric "
                "EPSG:32647. Baseline-flood comparison uses the common "
                "valid domain of corridor potential, flood-exposed "
                "connectivity, and connectivity loss. No resampling or "
                "composite connectivity score is used. Cost distance, "
                "corridor potential, current density, circuit connectivity, "
                "and flood-derived metrics retain their distinct "
                "mathematical interpretations. Connectivity indices "
                "are not probabilities."
            ),
        }

        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print(f"\n[OUTPUT] Summary: {self.summary_path}")
        print(f"[TIME] Total analysis: {perf_counter() - started:.2f} seconds")
        print("\nCONNECTIVITY INPUT VALIDATION COMPLETED")

        return self.summary_path
