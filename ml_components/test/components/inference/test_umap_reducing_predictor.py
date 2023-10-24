# coding: utf-8

import numpy as np

from ml_components.components.inference import UmapReducingPredictor


class TestUmapReducingPredictor:
    def test_init(self) -> None:
        reducer = UmapReducingPredictor()
        assert isinstance(reducer, UmapReducingPredictor)

    def test_reducing_something(self) -> None:
        reducer = UmapReducingPredictor()

        input = np.random.rand(64, 64)

        print(input)

        res = reducer.predict(input)

        assert isinstance(res, np.ndarray)

        print(res.shape)
        print(res)
