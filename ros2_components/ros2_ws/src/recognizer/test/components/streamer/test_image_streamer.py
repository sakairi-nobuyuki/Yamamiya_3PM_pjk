# coding: utf-8

import numpy as np
from recognizer.components.streamer import ImageStreamerElecom


class TestImageStreamerElecom:
    streamer = ImageStreamerElecom()

    def test_init(self):

        assert isinstance(self.streamer, ImageStreamerElecom)

    def test_capture(self):

        image = self.streamer.capture()

        assert image.shape == (self.streamer.window_size[1], self.streamer.window_size[0], 3)
        assert isinstance(image, np.ndarray)
        # print(image.shape, type(image))
