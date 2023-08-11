# coding: utf-8

from typing import List, Any, Dict

from ml_components.io import S3ConfigIO, S3ImageIO, OnnxS3, DataTransferS3, IOTemplate
from ml_components.components.factory import TemplateFactory

class IoModuleFactory(TemplateFactory):
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str):
        print("IO module factory:")
        print(">> endpoint: ", endpoint_url)
        print(">> access key: ", access_key)
        print(">> secret key: ", secret_key)
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key

    def create(self, *args: List[str], **kwargs: Dict[str, str]) -> IOTemplate:

        if kwargs["type"] == "onnx":
            return OnnxS3(self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"])
        elif kwargs["type"] == "image":
            return S3ImageIO(self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"])
        elif kwargs["type"] == "config":
            return S3ConfigIO(self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"])
        elif kwargs["type"] == "transfer":
            return DataTransferS3(self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"])
        else:
            raise NotImplementedError(f"{kwargs['type']} is not implemented")
        