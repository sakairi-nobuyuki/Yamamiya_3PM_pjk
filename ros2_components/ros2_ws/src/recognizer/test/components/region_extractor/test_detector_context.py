# coding: utf-8

from recognizer.components.region_extractor import (
    DetectorContext,
    DetectorTemplate,
    ThresholdingDetectorHsv,
    ThresholdingDetectorSaturate,
)


class TestDetectorContext:
    def test_init(self):
        detector = DetectorContext(ThresholdingDetectorHsv())

        assert isinstance(detector, DetectorContext)
