# coding: utf-8

from typing import Dict, Any

from . import ThresholdingDetectorHsv, ThresholdingDetectorSaturate, DetectorTemplate
from . import FactoryTemplate

class DetectorFactory(FactoryTemplate):
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.config_dict = config_dict
        
    def create(self) -> None:
        if self.config_dict["type"] == "hsv":
            config = {item_key: item_value for item_key, item_value in self.config_dict.items() if "type" not in item_key}
            return ThresholdingDetectorHsv(**config)
        elif self.config_dict["type"] == "saturation":
            config = {item_key: item_value for item_key, item_value in self.config_dict.items() if "type" not in item_key}
            return ThresholdingDetectorSaturate(**config)
        else:
            raise NotImplementedError(f"{self.config_dict['type']} is not implemented")
            
            
            