from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

RasterKind = Literal["continuous", "categorical"]


@dataclass
class AlignmentReport:
    source: str
    reference: str
    output: str
    kind: str
    resampling: str
    source_crs: str
    target_crs: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    valid_pixels: int
    nodata_pixels: int
    valid_fraction: float
    minimum: float
    maximum: float
    mean: float
    grid_aligned: bool


class RasterGridAligner:
    """
    Reproject and align a raster to the exact grid of a reference raster.

    The reference supplies the target CRS, affine transform, width, and
    height. Its pixel values and NoData value are not copied.

    Continuous data:
        nearest, bilinear, cubic, average

    Categorical data:
        nearest, mode
    """

    RESAMPLING_METHODS = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }

    ALLOWED_METHODS = {
        "continuous": {"nearest", "bilinear", "cubic", "average"},
        "categorical": {"nearest", "mode"},
    }

    def __init__(
        self,
        source_path: str | Path,
        reference_path: str | Path,
        output_path: str | Path,
        kind: RasterKind = "continuous",
        resampling: str | None = None,
        output_nodata: float = -9999.0,
        band: int = 1,
        num_threads: int = 2,
    ) -> None:
        self.source_path = Path(source_path)
        self.reference_path = Path(reference_path)
        self.output_path = Path(output_path)

        if kind not in self.ALLOWED_METHODS:
            raise ValueError(f"Unsupported raster kind: {kind}")

        if resampling is None:
            resampling = "bilinear" if kind == "continuous" else "nearest"

        if resampling not in self.ALLOWED_METHODS[kind]:
            allowed = sorted(self.ALLOWED_METHODS[kind])
            raise ValueError(
                f"Resampling '{resampling}' is invalid for "
                f"'{kind}' data. Allowed: {allowed}"
            )

        if output_nodata != output_nodata or not np.isfinite(output_nodata):
            raise ValueError("output_nodata must be a finite number.")

        if num_threads < 1:
            raise ValueError("num_threads must be at least 1.")

        if self.output_path.resolve() in {
            self.source_path.resolve(),
            self.reference_path.resolve(),
        }:
            raise ValueError(
                "Output must not overwrite the source or reference raster."
            )

        self.kind = kind
        self.resampling_name = resampling
        self.resampling_method = self.RESAMPLING_METHODS[resampling]
        self.output_nodata = output_nodata
        self.band = band
        self.num_threads = num_threads

    @staticmethod
    def _grid_matches(
        a: rasterio.io.DatasetReader,
        b: rasterio.io.DatasetReader,
    ) -> bool:
        return (
            a.crs == b.crs
            and a.width == b.width
            and a.height == b.height
            and a.transform == b.transform
        )

    def run(self) -> AlignmentReport:
        for path in (self.source_path, self.reference_path):
            if not path.is_file():
                raise FileNotFoundError(f"Raster not found: {path}")

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            rasterio.open(self.source_path) as src,
            rasterio.open(self.reference_path) as ref,
        ):
            if src.crs is None or ref.crs is None:
                raise ValueError("Source and reference must have a CRS.")

            if not 1 <= self.band <= src.count:
                raise ValueError(
                    f"Band {self.band} does not exist in source "
                    f"(band count: {src.count})."
                )

            if src.width < 1 or src.height < 1:
                raise ValueError("Source raster has invalid dimensions.")

            if ref.width < 1 or ref.height < 1:
                raise ValueError("Reference raster has invalid dimensions.")

            logger.info("Source raster: %s", self.source_path)
            logger.info("Reference grid: %s", self.reference_path)
            logger.info(
                "Source grid: %s, %s x %s",
                src.crs,
                src.width,
                src.height,
            )
            logger.info(
                "Target grid: %s, %s x %s",
                ref.crs,
                ref.width,
                ref.height,
            )

            destination = np.full(
                (ref.height, ref.width),
                self.output_nodata,
                dtype=np.float32,
            )

            reproject(
                source=rasterio.band(src, self.band),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodatavals[self.band - 1],
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=self.output_nodata,
                resampling=self.resampling_method,
                init_dest_nodata=True,
                num_threads=self.num_threads,
            )

            valid = np.isfinite(destination) & (destination != self.output_nodata)

            if not np.any(valid):
                raise RuntimeError(
                    "Alignment produced no valid pixels. "
                    "Check source coverage, CRS, and NoData settings."
                )

            values = destination[valid]

            # Create output using source metadata but the exact
            # geospatial grid of the reference.
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                crs=ref.crs,
                transform=ref.transform,
                width=ref.width,
                height=ref.height,
                count=1,
                dtype="float32",
                nodata=self.output_nodata,
                compress="deflate",
                predictor=3,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(
                self.output_path,
                "w",
                **profile,
            ) as dst:
                dst.write(destination, 1)
                dst.set_band_description(
                    1,
                    f"Aligned {self.kind} raster",
                )
                dst.update_tags(
                    alignment_method="rasterio.warp.reproject",
                    resampling=self.resampling_name,
                    reference_raster=str(self.reference_path),
                    source_raster=str(self.source_path),
                    raster_kind=self.kind,
                )

            # Verify the written raster, not just the in-memory array.
            with rasterio.open(self.output_path) as out:
                aligned = self._grid_matches(out, ref)
                if not aligned:
                    raise RuntimeError("Output grid verification failed.")

            total_pixels = ref.width * ref.height

            report = AlignmentReport(
                source=str(self.source_path),
                reference=str(self.reference_path),
                output=str(self.output_path),
                kind=self.kind,
                resampling=self.resampling_name,
                source_crs=src.crs.to_string(),
                target_crs=ref.crs.to_string(),
                source_width=src.width,
                source_height=src.height,
                target_width=ref.width,
                target_height=ref.height,
                valid_pixels=int(valid.sum()),
                nodata_pixels=int(total_pixels - valid.sum()),
                valid_fraction=float(valid.sum() / total_pixels),
                minimum=float(values.min()),
                maximum=float(values.max()),
                mean=float(values.mean()),
                grid_aligned=aligned,
            )

        report_path = self.output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(asdict(report), indent=2),
            encoding="utf-8",
        )

        logger.info("Aligned raster: %s", self.output_path)
        logger.info("Alignment report: %s", report_path)
        logger.info(
            "Valid pixels: %s / %s (%.2f%%)",
            report.valid_pixels,
            report.valid_pixels + report.nodata_pixels,
            report.valid_fraction * 100,
        )
        logger.info(
            "Value range: %.6f – %.6f; mean: %.6f",
            report.minimum,
            report.maximum,
            report.mean,
        )
        logger.info("Grid verification: %s", report.grid_aligned)

        return report
