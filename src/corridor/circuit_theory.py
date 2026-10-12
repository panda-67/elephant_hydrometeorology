from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class CircuitTheory:
    """Build and run the C++ Circuit Theory corridor analysis."""

    def __init__(
        self,
        resistance_path: Path,
        source_path: Path,
        current_output: Path,
        connectivity_output: Path,
        tolerance: float = 1e-6,
        max_iterations: int = 4000,
    ) -> None:
        self.root = Path(__file__).resolve().parents[2]

        self.resistance_path = Path(resistance_path).resolve()
        self.source_path = Path(source_path).resolve()
        self.current_output = Path(current_output).resolve()
        self.connectivity_output = Path(connectivity_output).resolve()

        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.cpp_dir = self.root / "src" / "cpp" / "corridor"
        self.cpp_source = self.cpp_dir / "circuit_theory.cpp"

        self.build_dir = self.root / "build"
        self.executable = self.build_dir / "circuit_theory"
        self.manifest_path = self.build_dir / "circuit_theory.build.json"

        self.compiler = os.environ.get("CXX", "g++")
        self.compiler_flags = [
            "-std=c++17",
            "-O3",
            "-march=native",
        ]

    def _validate_inputs(self) -> None:
        """Validate source files and numerical parameters."""

        for path in (
            self.cpp_source,
            self.resistance_path,
            self.source_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Required file does not exist: {path}")

        if self.tolerance <= 0:
            raise ValueError("tolerance must be greater than zero.")

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        if self.resistance_path == self.current_output:
            raise ValueError(
                "Current-density output cannot overwrite resistance input."
            )

        if self.resistance_path == self.connectivity_output:
            raise ValueError("Connectivity output cannot overwrite resistance input.")

        if self.current_output == self.connectivity_output:
            raise ValueError("Current-density and connectivity outputs must differ.")

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calculate a file's SHA-256 hash."""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def _source_files(self) -> list[Path]:
        """Return the C++ source and local headers used for build tracking."""

        files = [self.cpp_source]

        if self.cpp_dir.is_dir():
            files.extend(
                path
                for path in self.cpp_dir.rglob("*")
                if path.is_file() and path.suffix in {".h", ".hpp", ".hh", ".hxx"}
            )

        return sorted(set(files))

    def _source_hashes(self) -> dict[str, str]:
        """Hash all tracked C++ source and header files."""

        return {
            str(path.relative_to(self.root)): self._sha256(path)
            for path in self._source_files()
        }

    def _compiler_command(self, output_path: Path) -> list[str]:
        """Construct the compiler command using GDAL configuration."""

        try:
            gdal_cflags = subprocess.run(
                ["gdal-config", "--cflags"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            gdal_libs = subprocess.run(
                ["gdal-config", "--libs"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        except FileNotFoundError as exc:
            raise RuntimeError(
                "gdal-config was not found. Activate the environment "
                "containing GDAL development files."
            ) from exc

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Unable to obtain compiler flags from gdal-config."
            ) from exc

        command = [
            self.compiler,
            *self.compiler_flags,
            *shlex.split(gdal_cflags),
            str(self.cpp_source),
            *shlex.split(gdal_libs),
        ]

        conda_prefix = os.environ.get("CONDA_PREFIX")

        if conda_prefix:
            conda_lib = Path(conda_prefix) / "lib"

            if conda_lib.is_dir():
                command.append(f"-Wl,-rpath,{conda_lib}")

        command.extend(["-o", str(output_path)])

        return command

    def _build_configuration(self) -> dict:
        """Return the configuration used to determine build freshness."""

        return {
            "compiler": self.compiler,
            "compiler_flags": self.compiler_flags,
            "source_hashes": self._source_hashes(),
        }

    def _is_build_current(self) -> bool:
        """Check whether the existing executable matches current sources."""

        if not self.executable.is_file():
            return False

        if not self.manifest_path.is_file():
            return False

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        return manifest == self._build_configuration()

    def build(self, force: bool = False) -> None:
        """Compile C++ only when necessary, unless forced."""

        self.build_dir.mkdir(parents=True, exist_ok=True)

        if not force and self._is_build_current():
            print("[BUILD] Source unchanged; skipping compilation.")
            return

        print("[BUILD] Compiling Circuit Theory C++ engine...")

        temporary_executable = self.build_dir / "circuit_theory.tmp"
        temporary_manifest = self.build_dir / "circuit_theory.build.tmp.json"

        temporary_executable.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

        command = self._compiler_command(temporary_executable)

        print("[BUILD] Command:")
        print(" ".join(shlex.quote(part) for part in command))

        try:
            subprocess.run(command, check=True)

            if not temporary_executable.is_file():
                raise RuntimeError("Compiler finished without producing an executable.")

            configuration = self._build_configuration()

            temporary_manifest.write_text(
                json.dumps(configuration, indent=2),
                encoding="utf-8",
            )

            # Replace the executable only after successful compilation.
            temporary_executable.replace(self.executable)
            temporary_manifest.replace(self.manifest_path)

        except (subprocess.CalledProcessError, OSError, RuntimeError):
            temporary_executable.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            raise

        print(f"[BUILD] Executable ready: {self.executable}")

    def _run_command(self) -> list[str]:
        """Construct the C++ engine's execution arguments."""

        return [
            str(self.executable),
            "--resistance",
            str(self.resistance_path),
            "--source",
            str(self.source_path),
            "--current-output",
            str(self.current_output),
            "--connectivity-output",
            str(self.connectivity_output),
            "--tolerance",
            str(self.tolerance),
            "--max-iterations",
            str(self.max_iterations),
        ]

    def run(self, force_build: bool | None = None) -> None:
        """Build if needed, then execute the corridor analysis.

        Passing force_build=True forces recompilation. When omitted,
        the command-line flag --force-build is also recognized.
        """

        self._validate_inputs()

        if force_build is None:
            force_build = "--force-build" in sys.argv[1:]

        self.current_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.connectivity_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.build(force=force_build)

        command = self._run_command()

        print("[RUN] Starting Circuit Theory analysis.")
        print(f"[RUN] Resistance: {self.resistance_path}")
        print(f"[RUN] Source mask: {self.source_path}")
        print(f"[RUN] Current density: {self.current_output}")
        print(f"[RUN] Connectivity: {self.connectivity_output}")

        subprocess.run(command, check=True)

        missing_outputs = [
            path
            for path in (
                self.current_output,
                self.connectivity_output,
            )
            if not path.is_file()
        ]

        if missing_outputs:
            missing = "\n".join(str(path) for path in missing_outputs)
            raise RuntimeError(
                f"C++ process finished, but expected outputs are missing:\n{missing}"
            )

        print("[RUN] Circuit Theory analysis completed successfully.")
