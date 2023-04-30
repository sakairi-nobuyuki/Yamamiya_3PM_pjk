# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import List

class ThresholdingTrainerTemplate(metaclass = ABCMeta):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def target(self, x: List[int]) -> float:
        pass