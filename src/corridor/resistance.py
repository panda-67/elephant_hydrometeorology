from pathlib import Path

import numpy as np
import rasterio


class CorridorResistance:
    PREDICTORS = {
        "elevation": "elevation_suitability.tif",
        "slope": "slope_suitability.tif",
        "landcover": "landcover_suitability.tif",
        "ndvi": "ndvi_suitability.tif",
        "distance_to_water": "distance_to_water_suitability.tif",
    }

    def __init__(self, normalized_dir: Path, output_dir: Path):
        self.normalized_dir = normalized_dir
        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def suitability_to_resistance(
        self,
        name: str,
        filename: str,
    ):
        input_path = self.normalized_dir / filename
        output_path = self.output_dir / f"{name}_resistance.tif"

        if not input_path.exists():
            raise FileNotFoundError(f"Suitability raster not found: {input_path}")

        with rasterio.open(input_path) as src:
            data = src.read(1).astype("float32")
            profile = src.profile.copy()
            nodata = src.nodata

            valid = np.isfinite(data)

            if nodata is not None:
                valid &= data != nodata

            if valid.any():
                minimum = data[valid].min()
                maximum = data[valid].max()

                if minimum < 0 or maximum > 1:
                    raise ValueError(
                        f"{name} suitability outside [0, 1]: {minimum}–{maximum}"
                    )

            resistance = 1.0 - data

            if nodata is not None:
                resistance[~valid] = nodata
            else:
                resistance[~valid] = np.nan

            profile.update(
                dtype="float32",
                nodata=nodata,
                compress="deflate",
                predictor=2,
            )

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as dst:
                dst.write(
                    resistance.astype("float32"),
                    1,
                )

        print(f"\n{name}")
        print(f"  Input    : {input_path}")
        print("  Formula  : resistance = 1 - suitability")
        print(f"  Output   : {output_path}")

        return output_path

    def run(self):
        print("=" * 60)
        print("P8 — INDIVIDUAL RESISTANCE SURFACES")
        print("=" * 60)

        outputs = {}

        for name, filename in self.PREDICTORS.items():
            outputs[name] = self.suitability_to_resistance(
                name,
                filename,
            )

        print("\n" + "=" * 60)
        print("P8 individual resistance completed.")
        print("=" * 60)

        return outputs
