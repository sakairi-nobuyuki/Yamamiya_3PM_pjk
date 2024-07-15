# coding: utf-8
import os
import pickle
from typing import Any, List, Union

import boto3

from ml_components.io import IOTemplate


class PickleIO(IOTemplate):
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
        print(">> endpoint: ", endpoint_url)
        print(">> bucket name: ", bucket_name)
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(self.bucket_name)
        self.blob = self.get_blob()
        print(">> blob length: ", len(self.blob))

    def save(self, object_list: Union[List[Any], Any], key: str) -> dict:
        """
        Saves a list of objects to S3 bucket as a pickle.

        Parameters:
        object_list (List[bytes]): A list of pickle objects
        key (str): Key under which the image will be saved.

        Returns:
        dict: Response from S3 bucket.
        """
        pickle_list = pickle.dumps(object_list)
        response = self.s3.Object(self.bucket_name, key).put(Body=pickle_list)

        return response

    def load(self, key: str) -> Any:
        """
        Loads an image from S3 bucket.

        Parameters:
        key (str): Key under which the image is saved.

        Returns:
        np.ndarray: Loaded image.
        """
        # obj = self.bucket.Object(key=key)
        # response = obj.get()
        temp_file_name = "temp.pickle"
        self.bucket.download_file(key, temp_file_name)
        print("temp pickle file: ", os.path.exists(temp_file_name))
        #        with open(temp_file_name, 'wb') as f:
        # self.s3.bucket.download_fileobj(self.bucket_name, key, f)
        #            self.s3.download_fileobj(self.bucket_name, key, f)
        # self.bucket.download_file(self.bucket_name, key, f)
        # self.bucket.download_file(self.bucket_name, key, f)
        # self.bucket.download_file(key, file_name)

        with open(temp_file_name, "rb") as f:
            my_pickle = pickle.load(f)
        os.remove(temp_file_name)
        return my_pickle

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
            if ".pickle" in str(obj.key) or ".pkl" in str(obj.key):
                #            if ".onnx" in str(obj.key):
                file_names.append(obj.key)
        return file_names

    def delete(key: str) -> None:
        pass
