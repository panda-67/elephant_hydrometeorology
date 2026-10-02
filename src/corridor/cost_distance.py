from pathlib import Path
import heapq

import numpy as np
import rasterio


class SourceBasedCostDistance:
    """
    Calculate source-based cumulative movement cost using
    an 8-neighbor raster graph and Dijkstra's algorithm.

    Edge cost:
        mean(resistance_current, resistance_neighbor) * step_distance

    where:
        step_distance = 1.0      for cardinal neighbors
        step_distance = sqrt(2)  for diagonal neighbors

    Source cells have cumulative cost = 0.
    Invalid / NoData cells are excluded from the graph.
    """

    NEIGHBORS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    def __init__(
        self,
        resistance_raster: Path,
        source_mask: Path,
        output_path: Path,
    ):
        self.resistance_raster = resistance_raster
        self.source_mask = source_mask
        self.output_path = output_path

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read_inputs(self):
        with rasterio.open(self.resistance_raster) as resistance_src:
            resistance = resistance_src.read(1).astype("float32")
            resistance_profile = resistance_src.profile.copy()
            resistance_transform = resistance_src.transform
            resistance_crs = resistance_src.crs
            resistance_shape = (
                resistance_src.height,
                resistance_src.width,
            )
            resistance_nodata = resistance_src.nodata

        with rasterio.open(self.source_mask) as source_src:
            source = source_src.read(1)
            source_transform = source_src.transform
            source_crs = source_src.crs
            source_shape = (
                source_src.height,
                source_src.width,
            )

        if source_shape != resistance_shape:
            raise ValueError(
                "Source mask and resistance raster have different dimensions."
            )

        if source_transform != resistance_transform:
            raise ValueError(
                "Source mask and resistance raster have different transforms."
            )

        if source_crs != resistance_crs:
            raise ValueError("Source mask and resistance raster have different CRS.")

        valid = np.isfinite(resistance)

        if resistance_nodata is not None:
            valid &= resistance != resistance_nodata

        valid &= resistance >= 0
        valid &= resistance <= 1

        source_mask = source == 1

        # Source must be completely inside the valid resistance domain.
        invalid_source = source_mask & ~valid

        if invalid_source.any():
            raise ValueError(
                "Source mask contains pixels outside the valid resistance domain."
            )

        source_count = int(source_mask.sum())
        valid_count = int(valid.sum())

        if source_count == 0:
            raise ValueError("Source mask contains no source pixels.")

        if valid_count == 0:
            raise ValueError("Resistance raster contains no valid pixels.")

        return (
            resistance,
            valid,
            source_mask,
            resistance_profile,
            resistance_transform,
            resistance_crs,
            valid_count,
            source_count,
        )

    def _dijkstra(
        self,
        resistance,
        valid,
        source_mask,
    ):
        height, width = resistance.shape

        # Cumulative cost.
        distance = np.full(
            (height, width),
            np.inf,
            dtype=np.float64,
        )

        # Priority queue entries:
        # (cumulative_cost, row, col)
        heap = []

        source_rows, source_cols = np.where(source_mask)

        distance[source_rows, source_cols] = 0.0

        for row, col in zip(source_rows, source_cols):
            heapq.heappush(
                heap,
                (0.0, int(row), int(col)),
            )

        processed = 0

        while heap:
            current_cost, row, col = heapq.heappop(heap)

            # Ignore stale queue entries.
            if current_cost != distance[row, col]:
                continue

            processed += 1

            for drow, dcol, step_distance in self.NEIGHBORS:
                nrow = row + drow
                ncol = col + dcol

                if nrow < 0 or nrow >= height or ncol < 0 or ncol >= width:
                    continue

                if not valid[nrow, ncol]:
                    continue

                neighbor_resistance = resistance[nrow, ncol]
                current_resistance = resistance[row, col]

                edge_cost = (
                    (current_resistance + neighbor_resistance) / 2.0 * step_distance
                )

                new_cost = current_cost + edge_cost

                if new_cost < distance[nrow, ncol]:
                    distance[nrow, ncol] = new_cost

                    heapq.heappush(
                        heap,
                        (new_cost, nrow, ncol),
                    )

        print(f"  Processed pixels : {processed:,}")

        return distance

    def run(self):
        print("=" * 70)
        print("P9.2 — SOURCE-BASED COST DISTANCE")
        print("=" * 70)

        print(f"Resistance : {self.resistance_raster}")
        print(f"Source     : {self.source_mask}")
        print(f"Output     : {self.output_path}")

        (
            resistance,
            valid,
            source_mask,
            profile,
            transform,
            crs,
            valid_count,
            source_count,
        ) = self._read_inputs()

        print("\nInput statistics")
        print(f"  CRS                 : {crs}")
        print(f"  Dimensions          : {resistance.shape[1]} × {resistance.shape[0]}")
        print(f"  Resolution          : {transform.a:.15f}")
        print(f"  Valid pixels        : {valid_count:,}")
        print(f"  Source pixels       : {source_count:,}")
        print(f"  Source coverage     : {source_count / valid_count * 100:.3f}%")

        print("\nGraph configuration")
        print("  Connectivity        : 8-neighbor")
        print("  Algorithm           : Dijkstra")
        print("  Cardinal distance   : 1.0")
        print("  Diagonal distance   : sqrt(2)")
        print("  Edge cost           : mean resistance × step distance")

        print("\nCalculating cumulative cost...")

        distance = self._dijkstra(
            resistance=resistance,
            valid=valid,
            source_mask=source_mask,
        )

        valid_distance = distance[valid]

        if not np.isfinite(valid_distance).all():
            unreachable = int((~np.isfinite(distance) & valid).sum())
            raise ValueError(
                f"{unreachable:,} valid pixels are unreachable from the source."
            )

        source_distance = distance[source_mask]

        if not np.allclose(source_distance, 0.0):
            raise ValueError("Source pixels do not have zero cumulative cost.")

        output = distance.astype("float32")

        # NoData outside the valid ecological domain.
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

        finite_values = output[valid]

        print("\nOutput statistics")
        print(f"  Minimum cost       : {np.min(finite_values):.6f}")
        print(f"  P05                : {np.percentile(finite_values, 5):.6f}")
        print(f"  P25                : {np.percentile(finite_values, 25):.6f}")
        print(f"  Median             : {np.median(finite_values):.6f}")
        print(f"  P75                : {np.percentile(finite_values, 75):.6f}")
        print(f"  P95                : {np.percentile(finite_values, 95):.6f}")
        print(f"  Maximum cost       : {np.max(finite_values):.6f}")
        print(
            f"  Source min/max     : "
            f"{np.min(source_distance):.6f} / "
            f"{np.max(source_distance):.6f}"
        )

        print(f"\nOutput : {self.output_path}")

        print("\n" + "=" * 70)
        print("P9.2 SOURCE-BASED COST DISTANCE COMPLETED")
        print("=" * 70)

        return self.output_path
