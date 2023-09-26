# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Any


class TemplateFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> Any:
        pass
