# coding: utf-8

from typing import List, Any
import os
import shutil
import boto3

from ml_components.io import IOTemplate

class DataTransferS3(IOTemplate):
    
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

    def save(self, file_name: str, key: str) -> dict:
        """
        Saves an image to S3 bucket.

        Parameters:
        image (np.ndarray): Image to be saved.
        key (str): Key under which the image will be saved.

        Returns:
        dict: Response from S3 bucket.
        """
        # self.s3.Bucket(bucket_name).upload_file(file_path, key_name)
        self.bucket.upload_file(file_name, key)

        return {"status": "200"}

    def load(self, key: str) -> Any:
        """
        Loads an image from S3 bucket.

        Parameters:
        key (str): Key under which the image is saved.

        Returns:
        np.ndarray: Loaded image.
        """
                
        self.bucket.download_file(key, os.path.basename(key))
        
        return {"status": "200"}

    def delete(self, key: str) -> None:
        """
        Delete a file in S3

        Parameters:
        key (str): File to be deleted
        """
        self.bucket.objects.filter(Prefix=key).delete()

    def download_s3_folder(self, s3_folder: str, local_dir: str = None) -> None:
        if local_dir is None:
            local_dir = s3_folder
            print(f"local_dir is None. Makedir {local_dir}")

        if not os.path.exists(local_dir):
            print("makedir: ", local_dir)
            os.makedirs(local_dir)

        for obj in self.bucket.objects.filter(Prefix=s3_folder):
            print("download: ", obj.key)
            if not os.path.exists(os.path.dirname(obj.key)):
                print("makedir: ", os.path.dirname(obj.key))
                os.makedirs(os.path.dirname(obj.key))
            self.s3.Object(self.bucket.name, obj.key).download_file(obj.key)

    def delete_local(self, local_dir: str) -> None:
        shutil.rmtree(local_dir)
