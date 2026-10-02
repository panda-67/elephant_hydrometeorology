from pathlib import Path

import numpy as np
import rasterio


class CorridorPotential:
    """
    Convert source-based cumulative cost distance into
    relative corridor potential.

    Transformation:

        potential = 1 - (cost / max_non_source_cost)

    Source pixels are explicitly assigned potential = 1.0.

    Interpretation:
        1 = highest relative connectivity potential
        0 = lowest relative connectivity potential within the analyzed domain

    This is NOT a probability of elephant movement or habitat use.
    """

    def __init__(
        self,
        cost_distance_raster: Path,
        source_mask: Path,
        output_path: Path,
    ):
        self.cost_distance_raster = cost_distance_raster
        self.source_mask = source_mask
        self.output_path = output_path

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read_inputs(self):
        with rasterio.open(self.cost_distance_raster) as cost_src:
            cost = cost_src.read(1).astype("float32")
            profile = cost_src.profile.copy()
            transform = cost_src.transform
            crs = cost_src.crs
            shape = (cost_src.height, cost_src.width)
            nodata = cost_src.nodata

        with rasterio.open(self.source_mask) as source_src:
            source = source_src.read(1)
            source_transform = source_src.transform
            source_crs = source_src.crs
            source_shape = (
                source_src.height,
                source_src.width,
            )

        if shape != source_shape:
            raise ValueError(
                "Cost-distance raster and source mask have different dimensions."
            )

        if transform != source_transform:
            raise ValueError(
                "Cost-distance raster and source mask have different transforms."
            )

        if crs != source_crs:
            raise ValueError("Cost-distance raster and source mask have different CRS.")

        valid = np.isfinite(cost)

        if nodata is not None:
            valid &= cost != nodata

        valid &= cost >= 0

        source_mask = source == 1

        if np.any(source_mask & ~valid):
            raise ValueError(
                "Source contains pixels outside the valid cost-distance domain."
            )

        if not source_mask.any():
            raise ValueError("Source mask contains no source pixels.")

        return (
            cost,
            valid,
            source_mask,
            profile,
            transform,
            crs,
        )

    def run(self):
        print("=" * 70)
        print("P9.3 — RELATIVE CORRIDOR POTENTIAL")
        print("=" * 70)

        print(f"Cost distance : {self.cost_distance_raster}")
        print(f"Source mask   : {self.source_mask}")
        print(f"Output        : {self.output_path}")

        (
            cost,
            valid,
            source_mask,
            profile,
            transform,
            crs,
        ) = self._read_inputs()

        non_source = valid & ~source_mask

        if not non_source.any():
            raise ValueError("No valid non-source pixels found.")

        non_source_cost = cost[non_source]

        max_cost = float(np.max(non_source_cost))
        min_cost = float(np.min(non_source_cost))

        if max_cost <= 0:
            raise ValueError("Maximum non-source cost must be greater than zero.")

        print("\nInput statistics")
        print(f"  CRS                 : {crs}")
        print(f"  Dimensions          : {cost.shape[1]} × {cost.shape[0]}")
        print(f"  Valid pixels        : {int(valid.sum()):,}")
        print(f"  Source pixels       : {int(source_mask.sum()):,}")
        print(f"  Non-source pixels   : {int(non_source.sum()):,}")
        print(f"  Minimum non-source cost : {min_cost:.9f}")
        print(f"  Maximum non-source cost : {max_cost:.9f}")

        # --------------------------------------------------------------
        # Relative potential
        # --------------------------------------------------------------

        potential = np.full(
            cost.shape,
            np.nan,
            dtype="float32",
        )

        potential[valid] = (1.0 - (cost[valid] / max_cost)).astype("float32")

        # Source is explicitly assigned maximum potential.
        potential[source_mask] = 1.0

        # Numerical protection.
        potential[valid] = np.clip(
            potential[valid],
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        valid_potential = potential[valid]
        source_potential = potential[source_mask]

        if not np.all(np.isfinite(valid_potential)):
            raise ValueError(
                "Potential contains non-finite values inside valid domain."
            )

        if np.min(valid_potential) < 0:
            raise ValueError("Potential contains values below 0.")

        if np.max(valid_potential) > 1:
            raise ValueError("Potential contains values above 1.")

        if not np.allclose(
            source_potential,
            1.0,
            atol=1e-7,
        ):
            raise ValueError("Source pixels do not have potential = 1.")

        # Check monotonic relationship:
        # higher cost must not produce higher potential.
        valid_cost = cost[valid]
        valid_potential_check = potential[valid]

        order = np.argsort(valid_cost)

        sorted_cost = valid_cost[order]
        sorted_potential = valid_potential_check[order]

        cost_increase = np.diff(sorted_cost) > 0
        potential_increase = np.diff(sorted_potential) > 1e-6

        if np.any(cost_increase & potential_increase):
            raise ValueError("Potential is not monotonically decreasing with cost.")

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        output = potential.copy()
        output[~valid] = np.float32(-9999.0)

        output_profile = profile.copy()
        output_profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
        )

        with rasterio.open(
            self.output_path,
            "w",
            **output_profile,
        ) as dst:
            dst.write(output, 1)

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        values = potential[valid]

        print("\nPotential statistics")
        print(f"  Minimum            : {np.min(values):.9f}")
        print(f"  P05                : {np.percentile(values, 5):.9f}")
        print(f"  P25                : {np.percentile(values, 25):.9f}")
        print(f"  Median             : {np.median(values):.9f}")
        print(f"  P75                : {np.percentile(values, 75):.9f}")
        print(f"  P95                : {np.percentile(values, 95):.9f}")
        print(f"  Maximum            : {np.max(values):.9f}")
        print(f"  Mean               : {np.mean(values):.9f}")

        print("\nValidation")
        print("  Range [0,1]        : PASS")
        print("  Source = 1         : PASS")
        print("  NoData preserved   : PASS")
        print("  Grid preserved     : PASS")
        print("  Cost → potential   : PASS")

        print(f"\nOutput : {self.output_path}")

        print("\n" + "=" * 70)
        print("P9.3 RELATIVE CORRIDOR POTENTIAL COMPLETED")
        print("=" * 70)

        return self.output_path
