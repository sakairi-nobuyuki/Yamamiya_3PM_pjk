# coding: utf-8

import numpy as np
from recognizer.io import S3ImageIO, S3ConfigIO
import pytest

@pytest.fixture
def sample_array():
    return np.array([[[111, 112, 113, 114, 115],
                [121, 122, 123, 124, 125],
                [131, 132, 133, 134, 135],
                [141, 142, 143, 144, 145],
                [151, 152, 153, 154, 155]],
                [[211, 212, 213, 214, 215],
                [221, 222, 223, 224, 225],
                [231, 232, 233, 234, 235],
                [241, 242, 243, 244, 245],
                [251, 252, 253, 254, 255]]])
@pytest.fixture
def ref_file_name():
    return "20230422032404.png"

@pytest.fixture
def target_file_name():
    return "20230422032703.png"

@pytest.fixture
def mock_s3():
    return S3ImageIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="data")

@pytest.fixture
def mock_s3_dataset():
    return S3ImageIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="dataset")

@pytest.fixture
def mock_s3_config():
    return S3ConfigIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="config")

