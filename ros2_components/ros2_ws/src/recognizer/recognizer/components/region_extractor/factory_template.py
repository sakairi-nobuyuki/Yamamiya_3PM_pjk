# coding: utf-8

from typing import Dict, Any
from abc import ABCMeta, abstractmethod

class FactoryTemplate(metaclass = ABCMeta):
    @abstractmethod
    def create(self) -> None:
        pass