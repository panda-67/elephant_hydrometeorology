from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


class CorridorSource:
    """
    Create a source mask from the KHL landscape using the common-valid
    domain of the five individual resistance surfaces.
    """

    RESISTANCE_FILES = [
        "elevation_resistance.tif",
        "slope_resistance.tif",
        "landcover_resistance.tif",
        "ndvi_resistance.tif",
        "distance_to_water_resistance.tif",
    ]

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

    def _read_reference_grid(self):
        """
        Use the first resistance raster as the reference grid.
        All P8 resistance rasters have already passed grid QA/QC.
        """
        path = self.resistance_dir / self.RESISTANCE_FILES[0]

        if not path.exists():
            raise FileNotFoundError(f"Reference resistance raster not found: {path}")

        with rasterio.open(path) as src:
            profile = src.profile.copy()
            reference_data = src.read(1)
            transform = src.transform
            crs = src.crs
            shape = (src.height, src.width)
            nodata = src.nodata

        return (
            profile,
            reference_data,
            transform,
            crs,
            shape,
            nodata,
        )

    def _common_valid_mask(self, shape):
        """
        Build the common-valid mask across all five individual
        resistance surfaces.
        """
        common_valid = np.ones(shape, dtype=bool)

        for filename in self.RESISTANCE_FILES:
            path = self.resistance_dir / filename

            if not path.exists():
                raise FileNotFoundError(f"Resistance raster not found: {path}")

            with rasterio.open(path) as src:
                data = src.read(1).astype("float32")

                valid = np.isfinite(data)

                if src.nodata is not None:
                    valid &= data != src.nodata

                common_valid &= valid

        return common_valid

    def create_source_mask(self):
        print("=" * 70)
        print("P9.1 — KHL SOURCE MASK")
        print("=" * 70)

        (
            profile,
            _,
            transform,
            crs,
            shape,
            _,
        ) = self._read_reference_grid()

        print(f"Reference grid : {self.RESISTANCE_FILES[0]}")
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

        # Reproject source to the resistance grid CRS.
        if source.crs != crs:
            source = source.to_crs(crs)

        # Repair invalid geometries if necessary.
        invalid = ~source.geometry.is_valid

        if invalid.any():
            print(f"Repairing {invalid.sum()} invalid source geometries...")
            source = source.copy()
            source.geometry = source.geometry.make_valid()

        source = source.loc[~source.geometry.is_empty].copy()

        if source.empty:
            raise ValueError("Source vector contains no valid geometries.")

        # --------------------------------------------------------------
        # Rasterize KHL polygon on EXACT P8 grid
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

        source_mask = source_mask.astype("uint8")

        # --------------------------------------------------------------
        # Apply common-valid P8 domain
        # --------------------------------------------------------------
        common_valid = self._common_valid_mask(shape)

        source_mask &= common_valid.astype("uint8")

        source_pixels = int(source_mask.sum())
        common_valid_pixels = int(common_valid.sum())

        print("\nMask statistics")
        print(f"  Common valid pixels : {common_valid_pixels:,}")
        print(f"  Source pixels       : {source_pixels:,}")

        if source_pixels == 0:
            raise ValueError(
                "No KHL source pixels remain inside the common-valid P8 domain."
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
            dst.write(source_mask, 1)

        print(f"\nOutput : {output_path}")
        print("  Value 0 : outside source / invalid P8 domain")
        print("  Value 1 : KHL source within common-valid domain")

        print("\n" + "=" * 70)
        print("P9.1 KHL SOURCE MASK COMPLETED")
        print("=" * 70)

        return output_path


def main():
    root = Path(__file__).resolve().parents[2]

    source_vector = root / "data" / "output_vectors" / "KHL_PP_tangse_meureudu.geojson"

    resistance_dir = root / "data" / "output_rasters" / "corridor" / "resistance"

    output_dir = (
        root / "data" / "output_rasters" / "corridor" / "connectivity" / "source"
    )

    source = CorridorSource(
        source_vector=source_vector,
        resistance_dir=resistance_dir,
        output_dir=output_dir,
    )

    source.create_source_mask()


if __name__ == "__main__":
    main()
