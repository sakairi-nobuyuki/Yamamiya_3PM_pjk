# coding: utf-8

from typing import Dict, List

from ...data_structures import DatasetLoaderParameters
from ...io import S3ImageIO
from .template_dataset_loader import TemplateDatasetLoader


class CustomDatasetLoader(TemplateDatasetLoader):
    def __init__(self, parameters: DatasetLoaderParameters, s3_io: S3ImageIO) -> None:
        print("custom dataset loader")
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

        self.is_dataset_in_the_storage()

    def load(self):
        ### Get data path list of the target directory
        file_path_list = [
            file_path
            for file_path in self.s3_io.blob
            if self.parameters.s3_dir in file_path
        ]

        ### get numer of classes
        label_list = list(set([file_path.split("/")[-2] for file_path in file_path_list]))
        print(">> label list: ", label_list)

        ### Create file path list of each classes
        label_file_path_dict_list = [
            {file_path: file_path.split("/")[-2]} for file_path in file_path_list
        ]

        return label_file_path_dict_list

    def create_separated_file_path_list(
        self, file_list: List[str], class_name: str
    ) -> Dict[str, List[str]]:
        return super().create_separated_file_path_list(file_list, class_name)

    def is_dataset_in_the_storage(self) -> bool:
        return super().is_dataset_in_the_storage()

    def get_label_list(self) -> List[str]:
        return super().get_label_list()
