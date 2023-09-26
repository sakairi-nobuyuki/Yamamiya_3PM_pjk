# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Any


class TemplatePipeline(metaclass=ABCMeta):
    @abstractmethod
    def run(self, input: Any):
        pass
