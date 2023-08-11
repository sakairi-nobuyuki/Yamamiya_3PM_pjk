# coding: utf-8

from typing import Dict

from ml_components.components.factory import IoModuleFactory, TemplateFactory
from ml_components.io import OnnxS3, S3ConfigIO, S3ImageIO

class TestIoModuleFactory:

    def test_init(self, mock_io_module_config_dict: Dict[str, str]) -> None:
        factory = IoModuleFactory(**mock_io_module_config_dict)
        
        assert isinstance(factory, IoModuleFactory)
        assert isinstance(factory, TemplateFactory)
        assert factory.endpoint_url == mock_io_module_config_dict["endpoint_url"]
        assert factory.access_key == mock_io_module_config_dict["access_key"]
        assert factory.secret_key == mock_io_module_config_dict["secret_key"]

    def test_create(self, mock_io_module_config_dict: Dict[str, str]) -> None:
        factory = IoModuleFactory(**mock_io_module_config_dict)

        assert isinstance(factory.create(**dict(type="onnx", bucket_name="models")), OnnxS3)
        assert isinstance(factory.create(**dict(type="image", bucket_name="dataset")), S3ImageIO)
        assert isinstance(factory.create(**dict(type="config", bucket_name="config")), S3ConfigIO)