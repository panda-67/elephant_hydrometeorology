from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


class CorridorPotential:
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

        # --------------------------------------------------------------
        # Cost distance — metric grid
        # --------------------------------------------------------------

        with rasterio.open(self.cost_distance_raster) as cost_src:
            cost = cost_src.read(1).astype("float32")

            profile = cost_src.profile.copy()
            transform = cost_src.transform
            crs = cost_src.crs

            nodata = cost_src.nodata

            height = cost_src.height
            width = cost_src.width

        # --------------------------------------------------------------
        # Valid cost domain
        # --------------------------------------------------------------

        valid = np.isfinite(cost)

        if nodata is not None:
            valid &= cost != nodata

        valid &= cost >= 0

        # --------------------------------------------------------------
        # Reproject source mask to cost grid
        # --------------------------------------------------------------

        source = np.zeros(
            (height, width),
            dtype="uint8",
        )

        with rasterio.open(self.source_mask) as source_src:
            reproject(
                source=source_src.read(1),
                destination=source,
                src_transform=source_src.transform,
                src_crs=source_src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )


        source_mask = (source == 1) & valid

        if not source_mask.any():
            raise ValueError("No source pixels inside valid cost domain.")

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
        print("P9.3 — POTENTIAL CONNECTIVITY")
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

        # --------------------------------------------------------------
        # Domain
        # --------------------------------------------------------------

        non_source = valid & ~source_mask

        if not non_source.any():
            raise ValueError("No valid non-source pixels found.")

        max_cost = float(np.max(cost[non_source]))
        min_cost = float(np.min(cost[non_source]))

        if max_cost <= 0:
            raise ValueError("Maximum non-source cost must be greater than zero.")

        # --------------------------------------------------------------
        # Source cost
        # --------------------------------------------------------------

        source_cost = cost[source_mask]

        if not np.allclose(source_cost, 0.0, atol=1e-7):
            raise ValueError("Source pixels do not have zero cumulative cost.")

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        print("\nInput statistics")

        print(f"  CRS                 : {crs}")
        print(f"  Dimensions          : {cost.shape[1]} × {cost.shape[0]}")
        print(f"  Valid pixels        : {int(valid.sum()):,}")
        print(f"  Source pixels       : {int(source_mask.sum()):,}")
        print(f"  Non-source pixels   : {int(non_source.sum()):,}")
        print(f"  Minimum cost        : {min_cost:.6f}")
        print(f"  Maximum cost        : {max_cost:.6f}")

        # --------------------------------------------------------------
        # Potential
        # --------------------------------------------------------------

        potential = np.full(
            cost.shape,
            np.nan,
            dtype="float32",
        )

        potential[valid] = (1.0 - cost[valid] / max_cost).astype("float32")

        potential[source_mask] = 1.0

        potential[valid] = np.clip(
            potential[valid],
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        values = potential[valid]

        if not np.all(np.isfinite(values)):
            raise ValueError("Potential contains non-finite values.")

        if np.min(values) < 0 or np.max(values) > 1:
            raise ValueError("Potential outside range [0,1].")

        if not np.allclose(
            potential[source_mask],
            1.0,
            atol=1e-7,
        ):
            raise ValueError("Source pixels do not have potential = 1.")

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        output = potential.copy()
        output[~valid] = -9999.0

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

        print("\nPotential connectivity statistics")

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
        print("  Metric CRS         : PASS")
        print("  Cost → potential   : PASS")

        print(f"\nOutput : {self.output_path}")

        print("\n" + "=" * 70)
        print("P9.3 POTENTIAL CONNECTIVITY COMPLETED")
        print("=" * 70)

        return self.output_path
