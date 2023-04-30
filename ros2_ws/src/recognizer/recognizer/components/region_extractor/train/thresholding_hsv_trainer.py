# coding: utf-8

from typing import List, Tuple
import numpy as np
np.int = int
import cv2
from skopt import gp_minimize
from skopt.space import Real, Integer
import typer

from recognizer.io import S3ImageIO
from recognizer.components.region_extractor import ThresholdingDetector
# from ....io import S3ImageIO
# from .. import ThresholdingDetector


app = typer.Typer()
class ThresholdingTrainer:
    def __init__(self, w: int = 100, h: int = 100, aspect_ratio: float = 1.0, w_aspect_ratio: float = 1.0) -> None:
        """Determine weights in the loss function and other configurations.

        train_config:
            w: int: average_width
            h: int: average_length
            weight:
                aspect_ratio: float: average_width / average_length
                w_aspect_ratio: flaot: weight of aspect ratio against bbox size
        """
        self.w = w
        self.h = h
        self.aspect_ratio = aspect_ratio
        self.w_aspect_ratio = w_aspect_ratio
        self.s3 = S3ImageIO(
            endpoint_url="http://192.168.1.194:9000", 
            access_key="sigma-chan", 
            secret_key="sigma-chan-dayo", 
            bucket_name="dataset")
        self.file_name_list = [item for item in self.s3.blob if "region_extractor/thresholding_train" in item]

    def target(self, x: List[int]) -> float:
        """
        input:
            x: List[int]: [thresholding_value, lower_green[0], lower_green[1], lower_green[2], 
                            upper_green[0], upper_green[1], upper_green[2]]
        parameters:
            lower_green:
            - type: List[int]
            - range: [0:255, 0:255, 0:255]
            upper_green:
            - type: List[int]
            - [0:255, 0:255, 0:255]
            thresholding_value:
            - int
            - [0:255]
        """
        detector = ThresholdingDetector(type="hsv")

        detector.threshold_value = x[0]
        detector.lower_green = np.array([x[1], x[2], x[3]])
        detector.upper_green = np.array([x[4], x[5], x[6]])

        train_loss = 0
        for file_name in self.file_name_list:
            img = self.s3.load(file_name)
            contours = detector.detect_green_lsv(img)
            train_loss += self.loss(img, contours)

        return train_loss

    def loss(self, img: np.ndarray, contours: List[Tuple[int]]) -> float:

        loss_value = 0
        n_contour = 0
        for i_contour, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"w: {w}, h: {h}")
            loss_value += self.w_aspect_ratio * abs(self.aspect_ratio - w/h) * (abs(self.w - w) + abs(self.h - h)) 
            n_contour = i_contour
        loss_value *= 1.0 / (abs(n_contour - 1) + 1)

        return loss_value

@app.command()
def main() -> None:
    trainer = ThresholdingTrainer()

    space = [
        Integer(10, 200),   # thresholding_value
        Integer(50, 100),   # lower_green h
        Integer(25, 75),   # lower_green s
        Integer(25, 75),   # lower_green v
        Integer(100, 120),   # upper_green h
        Integer(254, 255),   # upper_green s
        Integer(254, 255)    # upper_green v
    ]

    res = gp_minimize(trainer.target, space, n_calls=100)

    return res

if __name__ == "__main__":
    app()