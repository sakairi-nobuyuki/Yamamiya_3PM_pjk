# coding: utf-8

import numpy as np
import torch
import pytest
import cv2

from recognizer.components.models import ConvolutionalRegionExtractNetwork
from recognizer.components.region_extractor import ConvolutionRegionExtractor
from recognizer.io.save_image_s3 import S3ImageIO

@pytest.mark.rule_base
class TestConvolutionRegionExtractor:
    
    def test_init(self) -> None:
        model = ConvolutionRegionExtractor(5, 3)
        assert isinstance(model, ConvolutionRegionExtractor)
        np.testing.assert_array_equal(np.array(model.kernel), np.array(np.ones((3,3))))

    @pytest.mark.parametrize("target_no", [0, 1, 2, 3])
    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7, 13])
    def test_diff_conv(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO, target_no: int, kernel_size: int) -> None:
        model = ConvolutionRegionExtractor(20, kernel_size)
        ref = mock_s3_dataset.load("region_extractor/base/ref.png")
        target = mock_s3_dataset.load(f"region_extractor/base/target{target_no}.png")
        
        img = model.get_conv_diff(ref, target)

        mock_s3.save(img, f"region_extractor/test/hoge{kernel_size}_{target_no}.png")

    @pytest.mark.parametrize("target_no", [0, 1, 2, 3])
    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7, 13])
    def test_conv_diff(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO, target_no: int, kernel_size: int) -> None:
        model = ConvolutionRegionExtractor(20, kernel_size)
        ref = mock_s3_dataset.load("region_extractor/base/ref.png")
        target = mock_s3_dataset.load(f"region_extractor/base/target{target_no}.png")
        
        img = model.get_diff_conv(ref, target)

        mock_s3.save(img, f"region_extractor/test/fuga{kernel_size}_{target_no}.png")

        
@pytest.mark.torch
class TestConvolutionalRegionExtractNetwork:
    def test_init(self):
        net = ConvolutionalRegionExtractNetwork(5, 3)

        assert isinstance(net, ConvolutionalRegionExtractNetwork)

        sample_array = np.array([[[111, 112, 113, 114, 115],
                        [121, 122, 123, 124, 125],
                        [131, 132, 133, 134, 135],
                        [141, 142, 143, 144, 145],
                        [151, 152, 153, 154, 155]],
                       [[211, 212, 213, 214, 215],
                        [221, 222, 223, 224, 225],
                        [231, 232, 233, 234, 235],
                        [241, 242, 243, 244, 245],
                        [251, 252, 253, 254, 255]]])
        inp_tensor = torch.Tensor(sample_array)

        assert isinstance(inp_tensor, torch.Tensor)
        assert inp_tensor.shape[0] == 2
        assert inp_tensor.shape[1] == 5
        assert inp_tensor.shape[2] == 5

        print(inp_tensor[1, ] - inp_tensor[0, ])
        print(net.conv(inp_tensor))