# coding: utf-8

from typing import List, Tuple

import numpy as np
import scipy

np.int = int
import cv2
import typer
from recognizer.components.region_extractor import ThresholdingDetectorHsv
from recognizer.io import S3ImageIO
from skopt import gp_minimize
from skopt.space import Integer, Real

from . import ThresholdingTrainerTemplate

# from ....io import S3ImageIO
# from .. import ThresholdingDetector


app = typer.Typer()


class ThresholdingHsvTrainer(ThresholdingTrainerTemplate):
    def __init__(
        self,
        w: int = 100,
        h: int = 100,
        aspect_ratio: float = 1.0,
        w_aspect_ratio: float = 1.0,
        n_calls: int = 100,
        space: List[int] = None,
        s3: S3ImageIO = None,
    ) -> None:
        """Determine weights in the loss function and other configurations.

        train_config:
            w: int: average_width
            h: int: average_length
            weight:
                aspect_ratio: float: average_width / average_length
                w_aspect_ratio: flaot: weight of aspect ratio against bbox size
        """

        ### parameter range
        if space is None:
            self.space = [
                Integer(10, 200),  # thresholding_value
                Integer(50, 150),  # lower_green h
                Integer(25, 100),  # lower_green s
                Integer(25, 100),  # lower_green v
                Integer(50, 120),  # upper_green h
                Integer(200, 255),  # upper_green s
                Integer(200, 255),  # upper_green v
            ]
        else:
            self.space = space

        ### parameters for train
        self.w = w
        self.h = h
        self.aspect_ratio = aspect_ratio
        self.w_aspect_ratio = w_aspect_ratio
        self.n_calls = n_calls

        ### aux
        if s3 is None:
            self.s3 = S3ImageIO(
                endpoint_url="http://192.168.1.194:9000",
                access_key="sigma-chan",
                secret_key="sigma-chan-dayo",
                bucket_name="dataset",
            )
        else:
            self.s3 = s3

        ### target dataset
        self.file_name_list = [
            item for item in self.s3.blob if "region_extractor/thresholding_train" in item
        ]

    def run(self) -> scipy.optimize._optimize.OptimizeResult:
        res = gp_minimize(self.target, self.space, n_calls=self.n_calls, verbose=True)

        return res

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
        detector = ThresholdingDetectorHsv()

        detector.threshold_value = x[0]
        detector.lower_green = np.array([x[1], x[2], x[3]])
        detector.upper_green = np.array([x[4], x[5], x[6]])

        train_loss = 0
        for file_name in self.file_name_list:
            img = self.s3.load(file_name)
            contours = detector.detect(img)
            train_loss += self.loss(img, contours)

        return train_loss

    def loss(self, img: np.ndarray, contours: List[Tuple[int]]) -> float:
        return super().loss(img, contours)


@app.command()
def main() -> None:
    trainer = ThresholdingHsvTrainer()

    space = [
        Integer(10, 200),  # thresholding_value
        Integer(50, 150),  # lower_green h
        Integer(25, 100),  # lower_green s
        Integer(25, 100),  # lower_green v
        Integer(50, 120),  # upper_green h
        Integer(200, 255),  # upper_green s
        Integer(200, 255),  # upper_green v
    ]

    res = gp_minimize(trainer.target, space, n_calls=100)

    return res


if __name__ == "__main__":
    app()
