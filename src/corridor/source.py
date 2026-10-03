from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


class CorridorSource:
    """
    Create a KHL source mask on the exact grid and valid domain
    of the composite resistance surface.
    """

    COMPOSITE_FILE = "composite_resistance_literature_weighted.tif"

    def __init__(
        self,
        source_vector: Path,
        resistance_dir: Path,
        output_dir: Path,
    ):
        self.source_vector = source_vector
        self.resistance_dir = resistance_dir
        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read_composite_grid(self):
        """
        Use the validated composite resistance raster as the
        reference grid and valid analysis domain.
        """
        path = self.resistance_dir / self.COMPOSITE_FILE

        if not path.exists():
            raise FileNotFoundError(f"Composite resistance raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")

            valid = np.isfinite(data)

            if src.nodata is not None:
                valid &= data != src.nodata

            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            shape = (src.height, src.width)

        return (
            profile,
            transform,
            crs,
            shape,
            valid,
        )

    def create_source_mask(self):
        print("=" * 70)
        print("P9.1 — KHL SOURCE MASK")
        print("=" * 70)

        (
            profile,
            transform,
            crs,
            shape,
            composite_valid,
        ) = self._read_composite_grid()

        print(f"Reference grid : {self.COMPOSITE_FILE}")
        print(f"CRS            : {crs}")
        print(f"Dimensions     : {shape[1]} × {shape[0]}")
        print(f"Resolution     : {transform.a:.15f}")

        # --------------------------------------------------------------
        # Load KHL source polygon
        # --------------------------------------------------------------

        if not self.source_vector.exists():
            raise FileNotFoundError(f"Source vector not found: {self.source_vector}")

        source = gpd.read_file(self.source_vector)

        if source.empty:
            raise ValueError(
                f"Source vector contains no features: {self.source_vector}"
            )

        if source.crs is None:
            raise ValueError(f"Source vector has no CRS: {self.source_vector}")

        print(f"Source vector  : {self.source_vector}")
        print(f"Source features: {len(source)}")
        print(f"Source CRS     : {source.crs}")

        # --------------------------------------------------------------
        # Reproject source to composite CRS
        # --------------------------------------------------------------

        if source.crs != crs:
            source = source.to_crs(crs)

            print(f"Reprojected to : {crs}")

        # --------------------------------------------------------------
        # Repair invalid geometries
        # --------------------------------------------------------------

        invalid = ~source.geometry.is_valid

        if invalid.any():
            print(f"Repairing {invalid.sum()} invalid source geometries...")

            source = source.copy()
            source.geometry = source.geometry.make_valid()

        source = source.loc[~source.geometry.is_empty].copy()

        if source.empty:
            raise ValueError("Source vector contains no valid geometries.")

        # --------------------------------------------------------------
        # Rasterize KHL source on EXACT composite grid
        # --------------------------------------------------------------

        geometries = [
            geometry
            for geometry in source.geometry
            if geometry is not None and not geometry.is_empty
        ]

        source_mask = geometry_mask(
            geometries,
            out_shape=shape,
            transform=transform,
            invert=True,
        )

        # --------------------------------------------------------------
        # Restrict source to the valid composite domain
        # --------------------------------------------------------------

        source_mask &= composite_valid

        source_pixels = int(source_mask.sum())
        valid_pixels = int(composite_valid.sum())

        source_coverage = source_pixels / valid_pixels * 100 if valid_pixels > 0 else 0

        print("\nMask statistics")
        print(f"  Composite valid pixels : {valid_pixels:,}")
        print(f"  KHL source pixels      : {source_pixels:,}")
        print(f"  Source coverage        : {source_coverage:.3f}%")

        if source_pixels == 0:
            raise ValueError(
                "No KHL source pixels remain inside the "
                "valid composite resistance domain."
            )

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        output_path = self.output_dir / "khl_source_mask.tif"

        output_profile = profile.copy()

        output_profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
        )

        with rasterio.open(
            output_path,
            "w",
            **output_profile,
        ) as dst:
            dst.write(
                source_mask.astype("uint8"),
                1,
            )

        print(f"\nOutput : {output_path}")
        print("  Value 0 : outside KHL source / invalid composite domain")
        print("  Value 1 : KHL source within valid composite domain")

        print("\n" + "=" * 70)
        print("P9.1 KHL SOURCE MASK COMPLETED")
        print("=" * 70)

        return output_path
