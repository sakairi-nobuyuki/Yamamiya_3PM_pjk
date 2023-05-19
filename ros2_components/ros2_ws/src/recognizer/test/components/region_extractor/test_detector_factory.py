# coding: utf-8

from typing import Dict, Any
import pytest
import numpy as np

from recognizer.components.region_extractor import DetectorFactory, FactoryTemplate
from recognizer.components.region_extractor import ThresholdingDetectorHsv, ThresholdingDetectorSaturate, DetectorTemplate

@pytest.fixture
def mock_config_dict() -> Dict[str, Any]:
    return dict(type="hsv", threshold_value=20)

class TestDetectorFactory:
    def test_init(self, mock_config_dict: Dict[str, Any]) -> None:
        factory = DetectorFactory(mock_config_dict)
        assert isinstance(factory, DetectorFactory)
        assert isinstance(factory, FactoryTemplate)

    def test_create_hsv(self, mock_config_dict: Dict[str, Any]) -> None:
        factory = DetectorFactory(mock_config_dict)
        detector = factory.create()

        assert isinstance(detector, DetectorTemplate)
        assert isinstance(detector, ThresholdingDetectorHsv)
        assert detector.threshold_value == mock_config_dict["threshold_value"]
        assert isinstance(detector.lower_green, np.ndarray)

    def test_create_saturation(self, mock_config_dict: Dict[str, Any]) -> None:
        config = mock_config_dict
        config["type"] = "saturation"
        factory = DetectorFactory(config)
        detector = factory.create()

        assert isinstance(detector, DetectorTemplate)
        assert isinstance(detector, ThresholdingDetectorSaturate)