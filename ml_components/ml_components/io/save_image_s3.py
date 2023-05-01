# coding: utf-8
import boto3
import cv2
import numpy as np
from . import IOTemplate
import os
import shutil

class S3ImageIO(IOTemplate):
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
        self.s3 = boto3.resource('s3', endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(self.bucket_name)
        self.blob = self.get_blob()
        print(">> endpoint: ", endpoint_url)
        print(">> bucket name: ", bucket_name)
        print(">> blob: ", self.blob)
    
    def get_blob(self) -> list[str]:
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

    def save(self, image: np.ndarray, key: str) -> dict:
        """
        Saves an image to S3 bucket.

        Parameters:
        image (np.ndarray): Image to be saved.
        key (str): Key under which the image will be saved.

        Returns:
        dict: Response from S3 bucket.
        """
        _, img_encoded = cv2.imencode('.png', image)
        response = self.bucket.put_object(Key=key, Body=img_encoded.tostring())
        return response

    def load(self, key: str) -> np.ndarray:
        """
        Loads an image from S3 bucket.

        Parameters:
        key (str): Key under which the image is saved.

        Returns:
        np.ndarray: Loaded image.
        """
        obj = self.bucket.Object(key=key)
        response = obj.get()
        file_stream = response['Body']
        file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    def delete(self, key: str) -> None:
        """
        Delete a file in S3

        Parameters:
        key (str): File to be deleted
        """
        self.bucket.objects.filter(Prefix="key").delete()

    def download_s3_folder(self, s3_folder: str, local_dir: str = None) -> None:
        if local_dir is None:
            local_dir = s3_folder

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_folder):
            if result.get('Contents'):
                for file in result.get('Contents'):
                    filename = file.get('Key')
                    self.s3.download_file(self.bucket_name, filename, f"{local_dir}/{filename.split('/')[-1]}")


    def delete_local(self, local_dir: str) -> None:
        shutil.rmtree(local_dir)