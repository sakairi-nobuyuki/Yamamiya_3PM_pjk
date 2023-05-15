# coding: utf-8

from typing import List
from abc import ABCMeta, abstractmethod
import numpy as np

class DetectorTemplate(metaclass = ABCMeta):
    """An interface of rule baseddetectors

    Args:
        metaclass (_type_, optional): Inhering abstract class. Defaults to ABCMeta.
    """
    
    @abstractmethod
    def detect(self, input: np.ndarray) -> List[np.ndarray]:
        """Detect something and return a list of contours

        Args:
            input (np.ndarray): Input image of OpenCV

        Returns:
            List[np.ndarray]: A list of OpenCV contours of detected things. 
        """
        pass