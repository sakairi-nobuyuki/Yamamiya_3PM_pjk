# coding: utf-8

import torch
import torch.onnx
import torchvision.models as models
import boto3
from typing import List
import os

from ml_components.io import IOTemplate

class OnnxS3(IOTemplate):
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str) -> None:
        """
        Initializes S3Image class with access_key, secret_key and bucket_name.

        Parameters:
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.
        bucket_name (str): Name of the S3 bucket.

        Returns:
        None
        """
        print("Initializing storage")
        self.s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.bucket_name = bucket_name
        # self.client = boto3.client('s3',
        #     endpoint_url=endpoint_url,
        #    aws_access_key_id=access_key,
        #    aws_secret_access_key=secret_key,)
        self.bucket = self.s3.Bucket(self.bucket_name)
        self.blob = self.get_blob()
        print(">> endpoint: ", endpoint_url)
        print(">> bucket name: ", bucket_name)
        print(">> blob: ", self.blob)
        
    def load(self, key: str): 
        # Load the PyTorch model from a .pth file
        model = torch.load("pytorch_model.pth", map_location=torch.device('cpu'))

    def save(self, input: models.vgg.VGG, key: str) -> dict:
        print(">> save onnx: ", key)
        # Convert the PyTorch model to ONNX format using a dummy input
        dummy_input = torch.rand(1, 3, 224, 224) # change this according to your model input shape
        export_onnx_name = key
        torch.onnx.export(input, dummy_input, export_onnx_name)

        object = self.bucket.Object(export_onnx_name)
        response = object.put(Body=open(export_onnx_name, 'rb'))
        try:
            os.remove(export_onnx_name)
        except Exception as e:
            print(f"Failed to delete temporary ONNX file: {e}")


        return response

    def get_blob(self) -> List[str]:
        """
        Returns a list of all file names in the S3 bucket.

        Parameters:
        None

        Returns:
        list[str]: List of all file names in the S3 bucket.
        """
        file_names = []
        for obj in self.bucket.objects.all():
            if ".onnx" in str(obj.key):
                file_names.append(obj.key)
        return file_names
    
    def delete(key: str) -> None:
        pass