from pathlib import Path

from src.corridor.base import CorridorBase


class CorridorBoundary(CorridorBase):
    def __init__(self, root: Path | None = None):
        super().__init__(root)

        self.khl_path = self.root / "data/input_vectors/KHL_PP.geojson"

        self.roi_path = self.root / "data/output_vectors/tangse_meureudu_roi.geojson"

        self.output_path = (
            self.root / "data/output_vectors/KHL_PP_tangse_meureudu.geojson"
        )

        self.khl = None
        self.roi = None
        self.result = None

    def load(self):

        print("\n[1] Loading KHL_PP...")

        self.khl = self.load_vector(self.khl_path)

        print(f"    Features : {len(self.khl):,}")
        print(f"    CRS      : {self.khl.crs}")

        print("\n[2] Loading study boundary...")

        self.roi = self.load_vector(self.roi_path)

        print(f"    Features : {len(self.roi):,}")
        print(f"    CRS      : {self.roi.crs}")

    def prepare(self):

        print("\n[3] Preparing study boundary...")

        self.roi = self.dissolve(self.roi)

        print(f"    Features : {len(self.roi):,}")

    def build(self):

        print("\n[4] Spatial intersection...")

        self.result = self.intersection(
            self.khl,
            self.roi,
        )

        print(f"    Features : {len(self.result):,}")

    def save(self):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.result.to_file(
            self.output_path,
            driver="GeoJSON",
        )

        print(f"\n[5] Output:\n    {self.output_path}")

    def report(self):

        khl_area = self.area(self.khl)
        roi_area = self.area(self.roi)
        result_area = self.area(self.result)

        retained = result_area["area_ha"] / khl_area["area_ha"] * 100

        print("\n" + "=" * 60)
        print("CORRIDOR BOUNDARY")
        print("=" * 60)

        print("\nArea:")

        print(f"  KHL_PP       : {khl_area['area_ha']:,.2f} ha")

        print(f"  Study area   : {roi_area['area_ha']:,.2f} ha")

        print(f"  Intersection : {result_area['area_ha']:,.2f} ha")

        print(f"               {result_area['area_km2']:,.2f} km²")

        print(f"  KHL retained : {retained:.2f}%")

        print("\nQA/QC:")

        print(f"  CRS          : {self.result.crs}")

        print(f"  Valid        : {self.result.geometry.is_valid.all()}")

        print(f"  Empty        : {self.result.geometry.is_empty.sum():,}")

        print(f"  Features     : {len(self.result):,}")
