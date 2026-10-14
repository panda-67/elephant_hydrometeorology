from __future__ import annotations

import argparse
import logging
import sys

from src.corridor.raster_grid_aligner import RasterGridAligner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject and align a raster to the exact grid of a reference raster."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Input raster to align.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Raster supplying the target grid.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoTIFF path.",
    )
    parser.add_argument(
        "--kind",
        choices=("continuous", "categorical"),
        default="continuous",
        help="Raster data type; default: continuous.",
    )
    parser.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "cubic", "average", "mode"),
        default=None,
        help="Optional resampling method; otherwise chosen by kind.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Output NoData value; default: -9999.",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Source band to align; default: 1.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="GDAL reprojection threads; default: 2.",
    )

    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = build_parser().parse_args()

    try:
        aligner = RasterGridAligner(
            source_path=args.source,
            reference_path=args.reference,
            output_path=args.output,
            kind=args.kind,
            resampling=args.resampling,
            output_nodata=args.nodata,
            band=args.band,
            num_threads=args.threads,
        )

        report = aligner.run()

        if not report.grid_aligned:
            return 1

        print("\nAlignment completed successfully.")
        print(f"Output : {report.output}")
        print(f"Valid  : {report.valid_pixels:,} pixels")
        print(f"NoData : {report.nodata_pixels:,} pixels")
        print(f"Range  : {report.minimum:.6f}–{report.maximum:.6f}")
        print(f"Mean   : {report.mean:.6f}")
        print(f"Coverage: {report.valid_fraction * 100:.2f}%")

        return 0

    except Exception:
        logging.exception("Raster alignment failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
