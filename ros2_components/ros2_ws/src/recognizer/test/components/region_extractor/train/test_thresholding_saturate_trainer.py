# coding: utf-8

from recognizer.components.region_extractor.train import ThresholdingSaturationTrainer, ThresholdingTrainerTemplate
from recognizer.components.region_extractor.train.thresholding_hsv_trainer import main

import scipy
import pytest

class TestThresholdingTrainer:
    def test_init(self):
        trainer = ThresholdingSaturationTrainer()

        assert isinstance(trainer, ThresholdingSaturationTrainer)

    def test_target(self):
        """
        self.lower_green = np.array([75, 50, 50])
        self.upper_green = np.array([110, 255, 255])

        """
        trainer = ThresholdingSaturationTrainer()

        x = [5, 75, 50, 50, 110, 255, 255]

        loss = trainer.target(x)

        print(loss)

        assert isinstance(loss, float)

    
    def test_run_hsv(self) -> None:
        trainer = ThresholdingSaturationTrainer()
        trainer = trainer
        trainer.n_calls = 10
        res = trainer.run()
        print(res, type(res), type(trainer)) 

        assert isinstance(res, scipy.optimize._optimize.OptimizeResult)
