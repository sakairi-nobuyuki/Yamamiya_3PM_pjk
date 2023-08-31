# coding: utf-8

from unittest.mock import MagicMock

import boto3
from moto import mock_s3

from ml_components.io import OnnxS3
from ml_components.models.factory import VggLikeClassifierFactory


class TestOnnxS3:
    #    @mock_s3
    def test_save(sefl, mock_s3_onnx: OnnxS3):
        # Create an S3 bucket
        s3 = mock_s3_onnx
        print(s3.bucket_name)
        factory = VggLikeClassifierFactory()
        model = factory.create_model()
        print("model", model)

        # Mock the S3 client
        # mock_s3_client = MagicMock()
        # mock_s3_client.list_objects_v2.return_value = {'Contents': []}
        # boto3.client = MagicMock(return_value=mock_s3_client)

        # Call your function that uses boto3
        result = s3.save(model, "hoge")
        print("save result: ", result)
        # Make assertions about how boto3 was used
        # mock_s3_client.list_objects_v2.assert_called_once_with(Bucket=s3.bucket_name)
