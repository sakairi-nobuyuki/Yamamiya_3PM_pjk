# coding: utf-8

from typing import List, Tuple, Union, Dict, Any
import numpy as np

from . import DetectorTemplate

class DetectorContext:
    def __init__(self, detector: DetectorTemplate) -> None:
        if not isinstance(detector, DetectorTemplate): 
            raise TypeError(f"Instance must be DetectorTemplate: {type(detector)}")
        self.__detector = detector

    @property
    def strategy(self) -> DetectorTemplate:
        return self.__detector
    
    @strategy.setter
    def strategy(self, detector: DetectorTemplate) -> None:
        self.__detector = detector

    def detect(self, input: np.ndarray) -> List[Tuple[int]]:

        return self.__detector.detect(input)