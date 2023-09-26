# coding: utf-8

import collections
import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List

import kaggle

from ...data_structures import DatasetLoaderParameters
from ...io import S3ImageIO
from .template_dataset_loader import TemplateDatasetLoader


class KaggleDatasetLoader(TemplateDatasetLoader):
    def __init__(self, parameters: DatasetLoaderParameters, s3_io: S3ImageIO) -> None:
        print("Kaggle dataset loader")
        if isinstance(parameters, DatasetLoaderParameters):
            self.parameters = parameters
        else:
            raise TypeError(
                f"Parameter is not that of KaggleDatasetLoaderParameters: {type(parameters)}"
            )
        if isinstance(s3_io, S3ImageIO):
            self.s3_io = s3_io
        else:
            raise TypeError(f"s3_io is not that of S3ImageIo: {type(s3_io)}")

    def load(self) -> List[str]:
        """Download dataset from kaggle to the local and upload to storage.
        Then return the label list.

        - Download dataset from kaggle to /home/${USER}/app/{self.parameters.local_dir}
          - Directory configuration of the files depends on the original dataset's configuration, however,
            it is supposed to be, /home/${USER}/app/{self.parameters.local_dir}/{something}/{label_name}/{file_name}.
        - Extract label list from the local path.
        - Upload these files to the storage.
          - Upload files
            from: /home/${USER}/app/{self.parameters.local_dir}/{something}/{label_name}/{file_name}
            to: /dataset/{self.parameters.s3_dir}/{label_name}/{phase}/{file_name}
          - These from and to shoule be correlated to a pair

        Returns:
            List[str]: label list
        """
        print(">> Checking dataset")

        ### check dataset
        if self.is_dataset_in_the_storage():
            print(">> Dataset seems to be Ok.")
            return self.get_label_list()

        print(">> Download Kaggle dataset")
        ### Download Kaggle dataset
        print(">> Download dataset to local")
        temp_dataset_path = os.path.join(
            str(Path(os.path.abspath(__file__)).parent.parent.parent.parent),
            self.parameters.local_dir,
        )
        print(">>   temporary dataset path: ", temp_dataset_path)

        # os.makedirs(temp_dataset_path, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            self.parameters.dataset_name, path=temp_dataset_path, unzip=True
        )

        # get label list
        print(">>   configuring dataset store")
        file_path_list = list(
            filter(
                lambda x: "jpg" in x.split("/")[-1] or "png" in x.split("/")[-1],
                glob.glob(f"{temp_dataset_path}/**/*.*", recursive=True),
            )
        )

        ### validate dataset data file path structure is Ok or not
        file_path_length_collection_dict = collections.Counter(
            list(map(lambda x: len(x.split("/")), file_path_list))
        )
        print(f">>   Dataset file path structure: {file_path_length_collection_dict}")
        if len(file_path_length_collection_dict.items()) > 1:
            raise ValueError(
                f">>  Dataset file path structure seems not good: {file_path_length_collection_dict}"
            )

        ### get something
        print(file_path_list)
        file_path_intersection = "/" + os.path.join(
            *[
                file_path_segment_summary[0]
                for file_path_segment_summary in [
                    list(set(file_path_segment))
                    for file_path_segment in zip(
                        *[file_path.split("/") for file_path in file_path_list]
                    )
                ]
                if len(file_path_segment_summary) == 1
            ]
        )
        print(f">>   file path intersection: {file_path_intersection}")

        label_list = sorted(
            list(set([file_path.split("/")[-2] for file_path in file_path_list]))
        )
        print(">>   label list: ", label_list)

        file_path_list_dict = {
            label: self.create_separated_file_path_list(file_path_list, label)
            for label in label_list
        }

        # upload files to S3 bucket
        print(">>   uploading dataset to storage")
        for label, file_path_list_phase_dict in file_path_list_dict.items():
            for phase, file_path_list in file_path_list_phase_dict.items():
                print(
                    f">>   configuring {label}, {phase} data and uploading. totally {len(file_path_list)}"
                )
                for file_path in file_path_list:
                    # print(phase, file_path)
                    # file_name = file_path.split("/")[-1]
                    ### TODO: {something} of this file path is needed
                    ### from: /home/${USER}/app/{self.parameters.local_dir}/{something}/{label_name}/{file_name}
                    file_name = file_path
                    file_path = os.path.join(file_path_intersection, label, file_path)
                    file_key = f"{self.parameters.s3_dir}/{phase}/{label}/{file_name}"

                    self.s3_io.s3.meta.client.upload_file(
                        file_path, self.s3_io.bucket_name, file_key
                    )
        print(">> finished dataset configuration")
        # remove temporary local files
        print(">> removing temporary files in local")
        shutil.rmtree(temp_dataset_path)

        return file_path_list

    def create_separated_file_path_list(
        self, file_list: List[str], class_name: str
    ) -> Dict[str, List[str]]:
        return super().create_separated_file_path_list(file_list, class_name)

    def is_dataset_in_the_storage(self) -> bool:
        ### If in the case of classifier

        ### if the subject dataset is in the storage
        print(f">>   If {self.parameters.s3_dir} is in the storage")
        if len(list(filter(lambda x: self.parameters.s3_dir in x, self.s3_io.blob))) == 0:
            print(f">>   {self.parameters.s3_dir} was not found in the storage")
            return False

        ### if train and validation is sufficient
        print(
            f">>   Train and val dataset size: {len(list(map(lambda x: 'train' in x or 'validation' in x, self.s3_io.blob)))}"
        )
        if (len(list(filter(lambda x: "train" in x, self.s3_io.blob))) < 1) or (
            len(list(filter(lambda x: "validation" in x, self.s3_io.blob))) < 1
        ):
            print(
                f">>   Too few dataset: {len(list(map(lambda x: 'train' in x or 'validation' in x, self.s3_io.blob)))}"
            )
            return False

        ### if the number of classes is sufficient
        if len(set(list(map(lambda x: x.split("/")[-2], self.s3_io.blob)))) < 1:
            print(
                f">>  the length of the classes is too small: {len(set(list(map(lambda x: x.split('/')[-2], self.s3_io.blob))))}"
            )
            return False

        return True

    def get_label_list(self) -> List[str]:
        """Get label list in the storage

        Returns:
            List[str]: Label list
        """

        file_path_list = filter(lambda x: self.parameters.s3_dir in x, self.s3_io.blob)
        print(file_path_list)
        label_list = sorted(
            list(set([file_path.split("/")[-2] for file_path in file_path_list]))
        )

        return label_list
