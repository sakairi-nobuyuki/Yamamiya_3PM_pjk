# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Any


class ModelFactoryTemplate(metaclass=ABCMeta):
    @abstractmethod
    def create_model(self) -> Any:
        pass

    @abstractmethod
    def create_forward(self) -> Any:
        pass
