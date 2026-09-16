import os
import sys

import ee

from src.core.engine import GEEEngine
from src.services.forensic_service import ForensicAnalysisService


def main():

    forensic_service = ForensicAnalysisService()
    pipelines = forensic_service.run_analysis_pipelines()

    p1, p2, p3, p4 = pipelines
    roi = forensic_service.roi

    pixel_area = ee.Image.pixelArea().divide(10000)  # hektar

    tier_area = pixel_area.addBands(p4.select("causal_evidence_tier")).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="tier"),
        geometry=roi,
        scale=10,
        maxPixels=1e13,
        bestEffort=True,
    )

    tier_info = tier_area.getInfo()

    output_dir = os.path.join(
        "data",
        "output_metrics",
    )

    GEEEngine.tier_report(
        tier_info,
        output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
