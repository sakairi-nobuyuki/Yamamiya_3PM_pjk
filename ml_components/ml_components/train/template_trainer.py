# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import Dict


class TemplateTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        """Run trainer"""
        pass

    @abstractmethod
    def get_label_map_dict(self) -> Dict[int, str]:
        """Get label map as a form of dict.
        Let's say there are a set of data and its label in a dataset like,
            file_1: label_1
            file_2: label_2
            file_3: label_1
            ...
        However the label itself is treated as integer, hence we should prepare a mapping from label in integer to
        label in string like this.

        {
            1: "label_1", 2: "label_2", ...
        }

        Returns:
            Dict[int, str]: A dict to map integer to string
        """
        pass
