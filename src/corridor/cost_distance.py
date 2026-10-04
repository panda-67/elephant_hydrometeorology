import os
import shutil
import subprocess
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


class SourceBasedCostDistance:
    """
    P9.2 — Source-Based Cost Distance.

    GIS preparation:
        EPSG:4326 → EPSG:32647
        10 m × 10 m common grid

    Numerical solver:
        C++ multi-source Dijkstra
    """

    TARGET_CRS = "EPSG:32647"
    TARGET_RESOLUTION = 10.0

    def __init__(
        self,
        resistance_raster: Path,
        source_mask: Path,
        output_path: Path,
    ):
        self.resistance_raster = Path(resistance_raster)
        self.source_mask = Path(source_mask)
        self.output_path = Path(output_path)

        self.root = Path(__file__).resolve().parents[2]

        self.tmp_dir = self.output_path.parent / "_tmp"

        self.tmp_resistance = self.tmp_dir / "resistance.tif"

        self.tmp_source = self.tmp_dir / "source.tif"

        self.tmp_output = self.tmp_dir / "cost_distance.tif"

        self.cpp_source = self.root / "src/cpp/corridor/cost_distance.cpp"

        self.cpp_executable = self.root / "build" / "cost_distance"

    # ============================================================
    # Public API
    # ============================================================

    def run(self):
        self._validate_inputs()

        print("\nPreparing P9.2 metric grid...")

        self._prepare_metric_grid()

        self._validate_common_grid()

        self._run_cpp()

        self._copy_final_output()

        self._validate_output()

        print("\nP9.2 completed.")
        print(f"Output: {self.output_path}")

    # ============================================================
    # Compiling
    # ============================================================

    def _compile_cpp(self):
        source = self.cpp_source
        executable = self.cpp_executable

        if not source.exists():
            raise FileNotFoundError(f"C++ source tidak ditemukan:\n{source}")

        executable.parent.mkdir(parents=True, exist_ok=True)

        gxx = shutil.which("g++")
        gdal_config = shutil.which("gdal-config")

        if gxx is None:
            raise RuntimeError("g++ tidak ditemukan di PATH.")

        if gdal_config is None:
            raise RuntimeError("gdal-config tidak ditemukan di PATH.")

        cflags = subprocess.check_output(
            [gdal_config, "--cflags"],
            text=True,
        ).split()

        libs = subprocess.check_output(
            [gdal_config, "--libs"],
            text=True,
        ).split()

        conda_prefix = os.environ.get("CONDA_PREFIX")

        if not conda_prefix:
            raise RuntimeError("CONDA_PREFIX tidak ditemukan.")

        command = [
            gxx,
            "-std=c++17",
            "-O3",
            "-march=native",
            str(source),
            *cflags,
            *libs,
            f"-Wl,-rpath,{conda_prefix}/lib",
            "-o",
            str(executable),
        ]

        print("Compile command:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        if not executable.exists():
            raise RuntimeError(
                f"Compilation selesai tetapi executable tidak ditemukan:\n{executable}"
            )

        executable.chmod(executable.stat().st_mode | 0o111)

        print(f"C++ executable berhasil dibuat:\n{executable}")

    # ============================================================
    # Validation
    # ============================================================

    def _validate_inputs(self):
        if not self.resistance_raster.exists():
            raise FileNotFoundError(
                f"Resistance raster tidak ditemukan:\n{self.resistance_raster}"
            )

        if not self.source_mask.exists():
            raise FileNotFoundError(f"Source mask tidak ditemukan:\n{self.source_mask}")

        if not self.cpp_executable.exists():
            print("C++ executable tidak ditemukan.")
            print("Compiling cost-distance engine...")
            self._compile_cpp()

        elif self.cpp_source.stat().st_mtime > self.cpp_executable.stat().st_mtime:
            print("C++ source lebih baru daripada executable.")
            print("Recompiling cost-distance engine...")
            self._compile_cpp()

        else:
            print("C++ executable tersedia dan masih up-to-date.")
            print(f"Using: {self.cpp_executable}")

        # ============================================================

    # Metric grid preparation
    # ============================================================

    def _prepare_metric_grid(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Resistance defines the target grid
        # --------------------------------------------------------

        with rasterio.open(self.resistance_raster) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                self.TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
                resolution=self.TARGET_RESOLUTION,
            )

            profile = src.profile.copy()

            profile.update(
                {
                    "crs": self.TARGET_CRS,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "dtype": "float32",
                    "nodata": -9999.0,
                    "compress": "deflate",
                    "predictor": 2,
                    "tiled": True,
                }
            )

            with rasterio.open(self.tmp_resistance, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=self.TARGET_CRS,
                    dst_nodata=-9999.0,
                    resampling=Resampling.bilinear,
                )

        # --------------------------------------------------------
        # Source is warped onto EXACT resistance grid
        # --------------------------------------------------------

        with rasterio.open(self.source_mask) as src:
            profile = src.profile.copy()

            profile.update(
                {
                    "crs": self.TARGET_CRS,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "dtype": "uint8",
                    "nodata": 0,
                    "compress": "deflate",
                    "tiled": True,
                }
            )

            with rasterio.open(self.tmp_source, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=transform,
                    dst_crs=self.TARGET_CRS,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )

        print(f"Target CRS     : {self.TARGET_CRS}")

        print(f"Resolution     : {self.TARGET_RESOLUTION} m")

        print(f"Dimensions     : {width} × {height}")

        print(f"Resistance     : {self.tmp_resistance}")

        print(f"Source         : {self.tmp_source}")

    # ============================================================
    # Common grid validation
    # ============================================================

    def _validate_common_grid(self):
        with (
            rasterio.open(self.tmp_resistance) as resistance,
            rasterio.open(self.tmp_source) as source,
        ):
            if resistance.crs != source.crs:
                raise RuntimeError("Resistance/source CRS berbeda.")

            if resistance.width != source.width or resistance.height != source.height:
                raise RuntimeError("Resistance/source dimensions berbeda.")

            for a, b in zip(
                resistance.transform,
                source.transform,
            ):
                if abs(a - b) > 1e-9:
                    raise RuntimeError("Resistance/source transform tidak identik.")

            if (
                abs(resistance.res[0] - 10.0) > 1e-6
                or abs(resistance.res[1] - 10.0) > 1e-6
            ):
                raise RuntimeError("Resistance bukan grid 10 m.")

        print("Common grid validation: PASS")

    # ============================================================
    # Run C++ solver
    # ============================================================

    def _run_cpp(self):
        if self.tmp_output.exists():
            self.tmp_output.unlink()

        command = [
            str(self.cpp_executable),
            "--resistance",
            str(self.tmp_resistance),
            "--source",
            str(self.tmp_source),
            "--output",
            str(self.tmp_output),
        ]

        print("\nRunning C++ Dijkstra...\n")

        subprocess.run(
            command,
            check=True,
        )

    # ============================================================
    # Copy final output
    # ============================================================

    def _copy_final_output(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.exists():
            self.output_path.unlink()

        shutil.copy2(
            self.tmp_output,
            self.output_path,
        )

    # ============================================================
    # Output validation
    # ============================================================

    def _validate_output(self):
        with rasterio.open(self.output_path) as src:
            if src.crs.to_epsg() != 32647:
                raise RuntimeError(f"Output CRS salah: {src.crs}")

            if abs(src.res[0] - 10.0) > 1e-6 or abs(src.res[1] - 10.0) > 1e-6:
                raise RuntimeError(f"Output resolution salah: {src.res}")

            if src.nodata != -9999.0:
                raise RuntimeError(f"Output NoData salah: {src.nodata}")

            if src.count != 1:
                raise RuntimeError("Output harus memiliki 1 band.")

        print("Output validation: PASS")
