from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


class FloodConnectivity:
    """
    Flood-adjusted corridor connectivity analysis.

    Conceptual framework:

        causal_evidence_tier
                ↓
        resistance multiplier
                ↓
        adjusted resistance
                ↓
        cost distance
                ↓
        flood-exposed connectivity
                ↓
        connectivity loss

    The baseline resistance surface remains the literature-weighted
    resistance surface. Causal evidence only modifies resistance
    through the configured tier multipliers.
    """

    TARGET_CRS = "EPSG:32647"
    TARGET_RESOLUTION = 10.0
    NODATA = -9999.0

    def __init__(
        self,
        baseline_resistance: Path,
        causal_matrix: Path,
        causal_tier_band: int,
        baseline_cost_distance: Path,
        baseline_connectivity: Path,
        source_mask: Path,
        output_dir: Path,
        causal_resistance_factors: Dict[int, float],
    ):
        self.baseline_resistance = Path(baseline_resistance)
        self.causal_matrix = Path(causal_matrix)
        self.causal_tier_band = causal_tier_band
        self.baseline_cost_distance = Path(baseline_cost_distance)
        self.baseline_connectivity = Path(baseline_connectivity)
        self.source_mask = Path(source_mask)
        self.output_dir = Path(output_dir)

        self.causal_resistance_factors = causal_resistance_factors

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_configuration()

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _validate_configuration(self) -> None:
        expected_tiers = {0, 1, 2, 3, 4}

        if set(self.causal_resistance_factors.keys()) != expected_tiers:
            raise ValueError(
                "CAUSAL_RESISTANCE_FACTORS harus memiliki tier 0, 1, 2, 3, dan 4."
            )

        if self.causal_resistance_factors[0] != 1.0:
            raise ValueError("Resistance factor untuk Tier 0 harus 1.0.")

        factors = [self.causal_resistance_factors[tier] for tier in range(5)]

        if factors != sorted(factors):
            raise ValueError(
                "CAUSAL_RESISTANCE_FACTORS harus monoton "
                "meningkat dari Tier 0 sampai Tier 4."
            )

        if any(factor < 1.0 for factor in factors):
            raise ValueError("Resistance factor tidak boleh < 1.0.")

    # ============================================================
    # RASTER HELPERS
    # ============================================================

    @staticmethod
    def _read_raster(path: Path):
        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()

        return data, profile

    @staticmethod
    def _is_nodata(value: float, nodata: float | None) -> bool:
        if nodata is None:
            return False

        return np.isclose(value, nodata)

    def _reproject_to_reference(
        self,
        source_path: Path,
        reference_profile: dict,
        resampling: Resampling,
        fill_value: float,
        dtype,
        band: int = 1,
    ) -> np.ndarray:
        """
        Reproject a raster onto the exact reference grid.

        This is particularly important for causal_evidence_tier
        because the tier raster is categorical and therefore must
        use nearest-neighbour resampling.
        """

        with rasterio.open(source_path) as src:
            destination = np.full(
                (
                    reference_profile["height"],
                    reference_profile["width"],
                ),
                fill_value,
                dtype=dtype,
            )

            reproject(
                source=rasterio.band(src, band),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                dst_nodata=fill_value,
                resampling=resampling,
            )

        return destination

    def _reproject_causal_tier(
        self,
        reference_profile: dict,
    ) -> np.ndarray:

        with rasterio.open(self.causal_matrix) as src:
            source = src.read(
                self.causal_tier_band,
                masked=False,
            ).astype(np.float32)

            source[~np.isfinite(source)] = np.nan

            destination = np.full(
                (
                    reference_profile["height"],
                    reference_profile["width"],
                ),
                np.nan,
                dtype=np.float32,
            )

            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )

        return destination

    @staticmethod
    def _write_raster(
        path: Path,
        data: np.ndarray,
        reference_profile: dict,
        dtype: str = "float32",
    ) -> None:

        profile = reference_profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "dtype": dtype,
                "count": 1,
                "nodata": -9999.0,
                "compress": "deflate",
                "predictor": 2,
            }
        )

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype(dtype), 1)

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_inputs(self) -> None:

        required = [
            self.baseline_resistance,
            self.causal_matrix,
            self.baseline_cost_distance,
            self.baseline_connectivity,
            self.source_mask,
        ]

        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Input raster tidak ditemukan: {path}")

        with rasterio.open(self.baseline_resistance) as src:
            if src.count != 1:
                raise ValueError("Baseline resistance harus memiliki satu band.")

            if src.crs is None:
                raise ValueError("Baseline resistance tidak memiliki CRS.")

        with rasterio.open(self.causal_matrix) as src:
            if src.count != 8:
                raise ValueError(
                    "Spatial causal matrix harus memiliki 8 band. "
                    f"Ditemukan {src.count} band."
                )

            if not 1 <= self.causal_tier_band <= src.count:
                raise ValueError(
                    f"Causal tier band {self.causal_tier_band} "
                    f"tidak valid untuk raster dengan "
                    f"{src.count} band."
                )

        with rasterio.open(self.baseline_cost_distance) as src:
            if src.crs is None:
                raise ValueError("Baseline cost distance tidak memiliki CRS.")

            if src.crs.to_string() != self.TARGET_CRS:
                raise ValueError(
                    f"Baseline cost distance harus berada pada {self.TARGET_CRS}."
                )

            if not np.isclose(src.res[0], self.TARGET_RESOLUTION):
                raise ValueError(
                    "Baseline cost distance harus memiliki "
                    f"resolusi {self.TARGET_RESOLUTION} m."
                )

            if not np.isclose(src.res[1], self.TARGET_RESOLUTION):
                raise ValueError(
                    "Baseline cost distance harus memiliki "
                    f"resolusi {self.TARGET_RESOLUTION} m."
                )

    # ============================================================
    # BASELINE GRID
    # ============================================================

    def _load_baseline_grid(self):
        """
        Cost-distance grid is the authoritative metric grid.

        All flood-adjusted calculations are performed on this grid.
        """

        with rasterio.open(self.baseline_cost_distance) as src:
            profile = src.profile.copy()
            baseline_cost = src.read(1)

            # Rasterio profile tidak selalu menyediakan `res`.
            # Simpan resolusi secara eksplisit.
            profile["res"] = src.res

        return baseline_cost, profile

    # ============================================================
    # RESISTANCE ADJUSTMENT
    # ============================================================

    def _build_adjusted_resistance(
        self,
        baseline_profile: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reproject baseline resistance and causal tier to the
        authoritative metric cost-distance grid.

        Returns
        -------
        adjusted_resistance
            Literature-weighted resistance multiplied by the
            causal evidence factor.

        tier
            Integer causal evidence tier on the same grid.
        """

        print("  Reprojecting baseline resistance...")

        baseline_resistance = self._reproject_to_reference(
            source_path=self.baseline_resistance,
            reference_profile=baseline_profile,
            resampling=Resampling.bilinear,
            fill_value=self.NODATA,
            dtype="float32",
        )

        # --------------------------------------------------------
        # Causal evidence tier
        # --------------------------------------------------------

        print("  Reprojecting causal evidence tier...")

        tier = self._reproject_causal_tier(baseline_profile)

        # Valid hanya untuk tier 0-4.
        # NaN dianggap NoData.
        valid_tier = np.isfinite(tier) & np.isin(
            tier,
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )

        print("  Causal tier distribution:")

        for tier_value in range(5):
            count = int(np.sum(tier == float(tier_value)))
            print(f"    Tier {tier_value}: {count:,} cells")

        print(f"    NoData/NaN: {int(np.sum(~np.isfinite(tier))):,} cells")

        # --------------------------------------------------------
        # Build factor raster
        # --------------------------------------------------------

        factors = np.ones(
            tier.shape,
            dtype=np.float32,
        )

        for tier_value, factor in self.causal_resistance_factors.items():
            mask = tier == float(tier_value)
            factors[mask] = factor

        # --------------------------------------------------------
        # Apply multiplier
        # --------------------------------------------------------

        valid = (
            np.isfinite(baseline_resistance)
            & (baseline_resistance != self.NODATA)
            & valid_tier
            & (baseline_resistance >= 0)
        )

        adjusted_resistance = np.full(
            baseline_resistance.shape,
            self.NODATA,
            dtype=np.float32,
        )

        adjusted_resistance[valid] = baseline_resistance[valid] * factors[valid]

        # --------------------------------------------------------
        # QA
        # --------------------------------------------------------

        if np.any(adjusted_resistance[valid] < baseline_resistance[valid]):
            raise RuntimeError(
                "Adjusted resistance lebih kecil daripada "
                "baseline resistance. Ini tidak diizinkan."
            )

        return adjusted_resistance, tier

    # ============================================================
    # COST DISTANCE
    # ============================================================

    def _calculate_flood_cost_distance(
        self,
        adjusted_resistance: np.ndarray,
        baseline_profile: dict,
    ) -> Path:
        """
        Run the existing SourceBasedCostDistance engine using
        the adjusted resistance surface.
        """

        from src.corridor.cost_distance import SourceBasedCostDistance

        temp_resistance = self.output_dir / "_flood_adjusted_resistance_for_cost.tif"

        self._write_raster(
            temp_resistance,
            adjusted_resistance,
            baseline_profile,
            dtype="float32",
        )

        output_cost = self.output_dir / "flood_adjusted_movement_cost.tif"

        print("  Running flood-adjusted cost distance...")

        engine = SourceBasedCostDistance(
            resistance_raster=temp_resistance,
            source_mask=self.source_mask,
            output_path=output_cost,
        )

        engine.run()

        if temp_resistance.exists():
            temp_resistance.unlink()

        return output_cost

    # ============================================================
    # FLOOD-EXPOSED CONNECTIVITY
    # ============================================================

    def _calculate_flood_connectivity(
        self,
        flood_cost_path: Path,
        baseline_cost: np.ndarray,
        baseline_profile: dict,
    ) -> Path:
        """
        Convert flood-adjusted cost distance into connectivity
        using the SAME maximum cost reference as the baseline
        corridor potential.

            connectivity =
                1 - flood_cost / baseline_max_cost

        This preserves comparability with the baseline corridor
        potential.
        """

        with rasterio.open(flood_cost_path) as src:
            flood_cost = src.read(1)

        with rasterio.open(self.baseline_connectivity) as src:
            baseline_connectivity = src.read(1)

        valid_baseline = (
            np.isfinite(baseline_cost)
            & (baseline_cost != self.NODATA)
            & (baseline_cost >= 0)
        )

        if not np.any(valid_baseline):
            raise RuntimeError("Tidak ada nilai valid pada baseline cost distance.")

        baseline_max_cost = float(np.max(baseline_cost[valid_baseline]))

        if baseline_max_cost <= 0:
            raise RuntimeError("Baseline maximum cost harus > 0.")

        print(f"  Baseline maximum cost : {baseline_max_cost:.6f}")

        valid = (
            np.isfinite(flood_cost) & (flood_cost != self.NODATA) & (flood_cost >= 0)
        )

        flood_connectivity = np.full(
            flood_cost.shape,
            self.NODATA,
            dtype=np.float32,
        )

        flood_connectivity[valid] = np.clip(
            1.0 - (flood_cost[valid] / baseline_max_cost),
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Source cells should remain fully connected.
        # --------------------------------------------------------

        source_mask = self._reproject_to_reference(
            source_path=self.source_mask,
            reference_profile=baseline_profile,
            resampling=Resampling.nearest,
            fill_value=0,
            dtype="uint8",
        )

        source_cells = (source_mask > 0) & valid

        flood_connectivity[source_cells] = 1.0

        output = self.output_dir / "flood_exposed_connectivity.tif"

        self._write_raster(
            output,
            flood_connectivity,
            baseline_profile,
            dtype="float32",
        )

        return output

    # ============================================================
    # CONNECTIVITY LOSS
    # ============================================================

    def _calculate_connectivity_loss(
        self,
        flood_connectivity_path: Path,
        baseline_profile: dict,
    ) -> Path:
        """
        Calculate scenario-based connectivity loss:

            loss =
                baseline_connectivity
                -
                flood_exposed_connectivity

        The result is clipped to [0, 1].
        """

        with rasterio.open(self.baseline_connectivity) as src:
            baseline_connectivity = src.read(1)

        with rasterio.open(flood_connectivity_path) as src:
            flood_connectivity = src.read(1)

        valid = (
            (baseline_connectivity != self.NODATA)
            & (flood_connectivity != self.NODATA)
            & np.isfinite(baseline_connectivity)
            & np.isfinite(flood_connectivity)
        )

        loss = np.full(
            baseline_connectivity.shape,
            self.NODATA,
            dtype=np.float32,
        )

        loss[valid] = np.clip(
            baseline_connectivity[valid] - flood_connectivity[valid],
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Source cells should have zero connectivity loss.
        # --------------------------------------------------------

        source_mask = self._reproject_to_reference(
            source_path=self.source_mask,
            reference_profile=baseline_profile,
            resampling=Resampling.nearest,
            fill_value=0,
            dtype="uint8",
        )

        source_cells = (source_mask > 0) & valid

        loss[source_cells] = 0.0

        output = self.output_dir / "flood_induced_connectivity_loss.tif"

        self._write_raster(
            output,
            loss,
            baseline_profile,
            dtype="float32",
        )

        return output

    # ============================================================
    # COST QA
    # ============================================================

    def _validate_cost_increase(
        self,
        flood_cost_path: Path,
    ) -> None:
        """
        Validate that flood-adjusted cost distance does not
        materially decrease relative to the baseline.
        """

        with rasterio.open(self.baseline_cost_distance) as src:
            baseline_cost = src.read(1).astype(np.float32)

        with rasterio.open(flood_cost_path) as src:
            flood_cost = src.read(1).astype(np.float32)

        valid = (
            np.isfinite(baseline_cost)
            & np.isfinite(flood_cost)
            & (baseline_cost >= 0)
            & (flood_cost >= 0)
        )

        if not np.any(valid):
            print("  Cost increase QA : SKIP (no valid cells)")
            return

        difference = flood_cost[valid] - baseline_cost[valid]

        min_difference = float(np.min(difference))
        max_difference = float(np.max(difference))

        # Small numerical differences are tolerated.
        tolerance = 0.1

        negative = difference < -tolerance

        print("  Cost increase QA:")
        print(f"    Minimum difference : {min_difference:.6f}")
        print(f"    Maximum difference : {max_difference:.6f}")
        print(f"    Tolerance          : {tolerance:.6f}")

        if np.any(negative):
            count = int(np.sum(negative))

            raise RuntimeError(
                "Flood-adjusted cost distance benar-benar lebih kecil "
                f"daripada baseline pada {count:,} cell "
                f"(minimum difference = {min_difference:.6f})."
            )

        print("    Status             : PASS")

    # ============================================================
    # MAIN EXECUTION
    # ============================================================

    def run(self) -> dict:
        """
        Execute the complete flood-connectivity workflow.
        """

        print()
        print("=" * 70)
        print("FLOOD CONNECTIVITY ANALYSIS")
        print("=" * 70)

        self._validate_inputs()

        print()
        print("Step 1/5 - Loading baseline metric grid")

        baseline_cost, baseline_profile = self._load_baseline_grid()

        print(f"  CRS         : {baseline_profile['crs']}")

        print(
            f"  Dimensions  : "
            f"{baseline_profile['width']} × "
            f"{baseline_profile['height']}"
        )

        print(
            f"  Resolution  : "
            f"{baseline_profile['res'][0]:.2f} × "
            f"{baseline_profile['res'][1]:.2f} m"
        )

        print()
        print("Step 2/5 - Applying causal resistance factors")

        adjusted_resistance, tier = self._build_adjusted_resistance(baseline_profile)

        resistance_output = self.output_dir / "flood_adjusted_resistance.tif"

        self._write_raster(
            resistance_output,
            adjusted_resistance,
            baseline_profile,
            dtype="float32",
        )

        print(f"  Output      : {resistance_output}")

        print()
        print("  Causal resistance factors:")

        for tier_value in range(5):
            factor = self.causal_resistance_factors[tier_value]

            count = int(np.sum(tier == tier_value))

            print(f"    Tier {tier_value}: {factor:.2f} × ({count:,} cells)")

        print()
        print("Step 3/5 - Calculating adjusted movement cost")

        flood_cost = self._calculate_flood_cost_distance(
            adjusted_resistance=adjusted_resistance,
            baseline_profile=baseline_profile,
        )

        self._validate_cost_increase(flood_cost)

        print()
        print("Step 4/5 - Calculating flood-exposed connectivity")

        flood_connectivity = self._calculate_flood_connectivity(
            flood_cost_path=flood_cost,
            baseline_cost=baseline_cost,
            baseline_profile=baseline_profile,
        )

        print(f"  Output      : {flood_connectivity}")

        print()
        print("Step 5/5 - Calculating connectivity loss")

        connectivity_loss = self._calculate_connectivity_loss(
            flood_connectivity_path=flood_connectivity,
            baseline_profile=baseline_profile,
        )

        print(f"  Output      : {connectivity_loss}")

        print()
        print("=" * 70)
        print("FLOOD CONNECTIVITY ANALYSIS COMPLETE")
        print("=" * 70)

        return {
            "adjusted_resistance": resistance_output,
            "adjusted_movement_cost": flood_cost,
            "flood_exposed_connectivity": flood_connectivity,
            "flood_induced_connectivity_loss": connectivity_loss,
        }
