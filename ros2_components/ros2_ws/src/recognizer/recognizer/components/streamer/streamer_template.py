# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Any


class StreamerTemplate(metaclass=ABCMeta):
    @abstractmethod
    def capture(self) -> Any:
        pass

    @abstractmethod
    def stop(self) -> Any:
        pass
