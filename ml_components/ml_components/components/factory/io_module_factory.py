# coding: utf-8

import os
from typing import Any, Dict, List

from ml_components.components.factory import TemplateFactory
from ml_components.io import DataTransferS3, IOTemplate, OnnxS3, S3ConfigIO, S3ImageIO


class IoModuleFactory(TemplateFactory):
    def __init__(
        self, endpoint_url: str = None, access_key: str = None, secret_key: str = None
    ):
        print("IO module factory:")

        if endpoint_url is not None:
            self.endpoint_url = endpoint_url
        else:
            self.endpoint_url = os.getenv("MINIO_ENDPOINT_URL")
        if access_key is not None:
            self.access_key = access_key
        else:
            self.access_key = os.getenv("ACCESS_KEY")
        if secret_key is not None:
            self.secret_key = secret_key
        else:
            self.secret_key = os.getenv("SECRET_KEY")
        print(">> endpoint: ", endpoint_url)
        print(">> access key: ", access_key)
        print(">> secret key: ", secret_key)

    def create(self, *args: List[str], **kwargs: Dict[str, str]) -> IOTemplate:
        """Requires input dict as,
        Args:
            kwargs = {
                "type": str: "type of the storage IO",
                "bucket": str: "target bucket name",
            }

        Raises:
            NotImplementedError: _description_

        Returns:
            IOTemplate: _description_
        """
        if kwargs["type"] == "onnx":
            print(
                ">> create onnx s3: ", self.endpoint_url, self.access_key, self.secret_key
            )
            return OnnxS3(
                self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"]
            )
        elif kwargs["type"] == "image":
            print(
                ">> create image s3: ",
                self.endpoint_url,
                self.access_key,
                self.secret_key,
            )
            return S3ImageIO(
                self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"]
            )
        elif kwargs["type"] == "config":
            print(
                ">> create config s3: ",
                self.endpoint_url,
                self.access_key,
                self.secret_key,
            )
            return S3ConfigIO(
                self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"]
            )
        elif kwargs["type"] == "transfer":
            print(
                ">> create transfer s3: ",
                self.endpoint_url,
                self.access_key,
                self.secret_key,
            )
            return DataTransferS3(
                self.endpoint_url, self.access_key, self.secret_key, kwargs["bucket_name"]
            )
        else:
            raise NotImplementedError(f"{kwargs['type']} is not implemented")
