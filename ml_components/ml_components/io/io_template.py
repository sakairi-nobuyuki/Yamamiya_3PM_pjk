# coding: utf-8
from abc import ABCMeta, abstractmethod
from typing import Any

class IOTemplate(metaclass=ABCMeta):
    @abstractmethod
    def save(self, input: Any, key: str) -> Any:
        """Save method."""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load method."""
        pass

    @abstractmethod
    def get_blob(self) -> Any:
        """Blob method."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a file"""
        pass
