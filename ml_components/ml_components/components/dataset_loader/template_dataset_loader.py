# coding: utf-8

import os
import random
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class TemplateDatasetLoader(metaclass=ABCMeta):
    """Template class of DatasetLoader

    Args:
        metaclass (optional): Inhering ABCMeta. Defaults to ABCMeta.
    """

    @abstractmethod
    def load(self, parameters: Any):
        pass

    @abstractmethod
    def create_separated_file_path_list(
        self, file_list: List[str], class_name: str
    ) -> Dict[str, List[str]]:
        """Create file lists from a list of whole file list of a class of a dataset into
        train, validation and test file lists.
        For example, if the dataset contains a classes of which names are "dog" and "cat", and,
        the list of the paths is ["dog/000.png", "dog/001.png", ..., "dog/099.png",
        "cat/000.png", ..., "cat/099.png"].
        Let the argument class_name is "dog", the lists returned are:
            train: ["000.png", ..., "0070.png"]
            val: ["071.png", ..., "090.png"]
            test: ["090.png", ..., "099.png"]
        For simplicity, these lists are not shuffled, but in actual the lists are shuffled.

        Args:
            file_list (List[str]): A list that contains whole file paths of a dataset
            class_name (str): class name to be separated and shuffled

        Returns:
            Dict[str, List[str]]: dict(train=train_file_list, val=val_file_list, test=test_file_list)
        """
        # file_list = glob.glob(f"{temp_dataset_path}/kagglecatsanddogs_3367a/PetImages/{class_name}/*.*")
        file_list = [
            # os.path.join(item.split("/")[-2], item.split("/")[-1]) for item in file_list
            item.split("/")[-1]
            for item in file_list
            if item.split("/")[-2] == class_name
        ]
        file_list_length = len(file_list)

        template_list = (
            int(file_list_length * self.parameters.train_data_rate) * ["train"]
            + int(file_list_length * self.parameters.val_data_rate) * ["validation"]
            + int(
                file_list_length
                * (1.0 - self.parameters.train_data_rate - self.parameters.val_data_rate)
            )
            * ["test"]
        )
        random.shuffle(template_list)

        train_file_list = [
            item for (flag, item) in zip(template_list, file_list) if flag == "train"
        ]
        val_file_list = [
            item for (flag, item) in zip(template_list, file_list) if flag == "validation"
        ]
        test_file_list = [
            item for (flag, item) in zip(template_list, file_list) if flag == "test"
        ]

        print(
            f">>   {class_name}: {len(file_list)}, {class_name} train: {len(train_file_list)}, {class_name} val: {len(val_file_list)}"
        )
        #        assert abs(len(file_list) - len(train_file_list) - len(val_file_list)) > 2
        #        assert (
        #            len(
        #                [
        #                    file_path
        #                    for file_path in train_file_list
        #                    if class_name not in file_path
        #                ]
        #            )
        #            == 0
        #        )
        #        assert (
        #            len([file_path for file_path in val_file_list if class_name not in file_path])
        #            == 0
        #        )

        return dict(train=train_file_list, val=val_file_list, test=test_file_list)
