# coding: utf-8

from typing import Any
from abc import ABCMeta, abstractmethod


class TemplateFactory(metaclass = ABCMeta):
    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> Any:
        pass