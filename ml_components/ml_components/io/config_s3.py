# coding: utf-8
from typing import Dict, List

import boto3
import cv2
import numpy as np
import yaml

from . import IOTemplate


class S3ConfigIO(IOTemplate):
    def __init__(
        self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str
    ) -> None:
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
        self.bucket = self.s3.Bucket(self.bucket_name)
        self.blob = self.get_blob()
        print(">> endpoint: ", endpoint_url)
        print(">> bucket name: ", bucket_name)
        print(">> blob: ", self.blob)
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
            file_names.append(obj.key)
        return file_names

    def save(self, input: Dict[str, str], key: str) -> None:
        """
        Saves a dict to S3 bucket.

        Parameters:
        input (Dict[str]): Dict to be saved.
        key (str): Key under which the dict will be saved.

        Returns:
        dict: Response from S3 bucket.
        """
        yaml_dict = yaml.dump(input)

        response = self.bucket.put_object(Key=key, Body=yaml_dict)

        return response

    def load(self, key: str) -> Dict[str, str]:
        """
        Loads an image from S3 bucket.

        Parameters:
        key (str): Key under which the image is saved.

        Returns:
        np.ndarray: Loaded image.
        """
        obj = self.bucket.Object(key=key)
        response = obj.get()
        loaded_yaml = yaml.safe_load(response["Body"])

        return loaded_yaml

    def delete(self, key: str) -> None:
        """
        Delete a file in S3

        Parameters:
        key (str): File to be deleted
        """
        self.bucket.objects.filter(Prefix="key").delete()
