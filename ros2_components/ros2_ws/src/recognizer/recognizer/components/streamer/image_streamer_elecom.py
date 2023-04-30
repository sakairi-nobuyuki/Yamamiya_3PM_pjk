# coding: utf-8

import cv2
from typing import Any
import numpy as np

from . import StreamerTemplate

class ImageStreamerElecom(StreamerTemplate):
    """Image streamer class

    Args:
        StreamerTemplate (_type_): Template class
    """
    window_size = (400, 300)
    
    def __init__(self, width: int = None, height: int = None) -> None:
        """Constructor

        Args:
            width (int, optional): Image width to output. Defaults to None.
            height (int, optional): Image height to output. Defaults to None.
        """
        if (width is not None and height is not None):
            self.window_size = (width, height)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_size[1])
        
    def capture(self) -> np.ndarray:
        ret, image = self.camera.read()
        print(">> Capture succeeded: ", ret)
        image = cv2.resize(image, self.window_size)

        return image

    def stop(self) -> Any:
        self.camera.release()