from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


class CorridorPredictorStack:
    """
    Menyamakan seluruh predictor corridor ke master grid 30 m.

    Master grid ditentukan oleh elevation.
    """

    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else Path(__file__).resolve().parents[2]

        self.input_dir = self.root / "data/output_rasters/corridor"
        self.output_dir = self.input_dir / "stack"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.files = {
            "elevation": self.input_dir / "corridor_elevation.tif",
            "slope": self.input_dir / "corridor_slope.tif",
            "landcover": (self.input_dir / "corridor_landcover_worldcover_2020.tif"),
            "ndvi": self.input_dir / "corridor_ndvi.tif",
            "distance_to_water": (self.input_dir / "corridor_distance_to_water.tif"),
        }

        self.master = None

    # =========================================================================
    # MASTER GRID
    # =========================================================================

    def load_master_grid(self):
        """Menggunakan elevation sebagai master grid."""

        path = self.files["elevation"]

        if not path.exists():
            raise FileNotFoundError(f"Master raster tidak ditemukan: {path}")

        with rasterio.open(path) as src:
            self.master = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
                "res": src.res,
            }

        print("\n[1] Master grid")
        print(f"    Raster     : elevation")
        print(f"    CRS        : {self.master['crs']}")
        print(
            f"    Resolution : "
            f"{self.master['res'][0]:.6f} × "
            f"{self.master['res'][1]:.6f}"
        )
        print(f"    Dimensions : {self.master['width']} × {self.master['height']}")
        print(f"    Bounds     : {self.master['bounds']}")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_inputs(self):
        """Memastikan seluruh input tersedia."""

        print("\n[2] Checking input rasters...")

        for name, path in self.files.items():
            if not path.exists():
                raise FileNotFoundError(f"Predictor '{name}' tidak ditemukan: {path}")

            with rasterio.open(path) as src:
                print(f"    {name:<20} {src.width} × {src.height} | {src.res}")

    # =========================================================================
    # RESAMPLING
    # =========================================================================

    def _resampling_method(self, name: str) -> Resampling:
        """
        Menentukan metode resampling berdasarkan tipe predictor.
        """

        methods = {
            "landcover": Resampling.mode,
            "ndvi": Resampling.average,
            "distance_to_water": Resampling.average,
        }

        return methods.get(name, Resampling.bilinear)

    def _dtype(self, name: str):
        """Menentukan tipe data output."""

        if name == "landcover":
            return "int16"

        return "float32"

    def _nodata(self, name: str):
        """Menentukan nilai NoData output."""

        if name == "landcover":
            return -32768

        return -9999.0

    # =========================================================================
    # REPROJECT
    # =========================================================================

    def resample_predictor(self, name: str):
        """Resample satu predictor ke master grid."""

        if self.master is None:
            raise RuntimeError("Master grid belum diinisialisasi.")

        input_path = self.files[name]
        output_path = self.output_dir / f"{name}_30m.tif"

        print(f"\n    → {name}")

        with rasterio.open(input_path) as src:
            data = src.read(1)

            # -------------------------------------------------------------
            # Source NoData
            # -------------------------------------------------------------
            source_nodata = src.nodata

            if np.issubdtype(data.dtype, np.floating):
                # Raster kontinu:
                # -inf / inf dianggap sebagai NoData.
                invalid = ~np.isfinite(data)

                source_nodata = -9999.0

                data = data.astype("float32")
                data[invalid] = source_nodata

            else:
                # Raster kategorikal seperti WorldCover:
                # gunakan NoData asli, bukan NaN.
                if source_nodata is None:
                    source_nodata = -32768

                data = data.astype("int16")

            # -------------------------------------------------------------
            # Destination
            # -------------------------------------------------------------
            dtype = self._dtype(name)
            destination_nodata = self._nodata(name)

            destination = np.full(
                (
                    self.master["height"],
                    self.master["width"],
                ),
                destination_nodata,
                dtype=dtype,
            )

            # -------------------------------------------------------------
            # Reprojection / resampling
            # -------------------------------------------------------------
            reproject(
                source=data,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=source_nodata,
                dst_transform=self.master["transform"],
                dst_crs=self.master["crs"],
                dst_nodata=destination_nodata,
                resampling=self._resampling_method(name),
            )

            # -------------------------------------------------------------
            # Output profile
            # -------------------------------------------------------------
            profile = {
                "driver": "GTiff",
                "height": self.master["height"],
                "width": self.master["width"],
                "count": 1,
                "dtype": dtype,
                "crs": self.master["crs"],
                "transform": self.master["transform"],
                "nodata": destination_nodata,
                "compress": "deflate",
                "predictor": (2 if dtype == "float32" else 1),
            }

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as dst:
                dst.write(destination, 1)

        print(f"       Output : {output_path}")

    # =========================================================================
    # BUILD
    # =========================================================================

    def build(self):
        """Membangun seluruh predictor stack."""

        print("\n[3] Building predictor stack...")

        for name in self.files:
            self.resample_predictor(name)

    # =========================================================================
    # QA/QC
    # =========================================================================

    def validate_stack(self):
        """Memvalidasi keseragaman seluruh output stack."""

        print("\n[4] Validating stack...")

        stack_files = [self.output_dir / f"{name}_30m.tif" for name in self.files]

        reference = None

        for path in stack_files:
            with rasterio.open(path) as src:
                current = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "res": src.res,
                    "bounds": src.bounds,
                }

                if reference is None:
                    reference = current
                    status = "MASTER"
                else:
                    status = "OK"

                    if current["crs"] != reference["crs"]:
                        status = "CRS MISMATCH"

                    elif current["transform"] != reference["transform"]:
                        status = "GRID MISMATCH"

                    elif (
                        current["width"] != reference["width"]
                        or current["height"] != reference["height"]
                    ):
                        status = "DIMENSION MISMATCH"

                print(f"    {path.name:<32} {status}")

        print("\n    Stack validation completed.")

    # =========================================================================
    # REPORT
    # =========================================================================

    def report(self):
        """Menampilkan statistik sederhana seluruh stack."""

        print("\n" + "=" * 60)
        print("P6.6 — CORRIDOR PREDICTOR STACK")
        print("=" * 60)

        for name in self.files:
            path = self.output_dir / f"{name}_30m.tif"

            with rasterio.open(path) as src:
                data = src.read(1, masked=True)

                valid = data.compressed()

                if valid.size == 0:
                    print(f"\n{name}: NO VALID DATA")
                    continue

                print(f"\n{name}")
                print(f"  Min      : {valid.min():.4f}")
                print(f"  Max      : {valid.max():.4f}")
                print(f"  Mean     : {valid.mean():.4f}")
                print(f"  Valid px : {valid.size:,}")
                print(f"  NoData   : {data.mask.sum():,}")

        print("\n" + "=" * 60)
