# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Any, Dict


class FactoryTemplate(metaclass=ABCMeta):
    @abstractmethod
    def create(self) -> None:
        pass
