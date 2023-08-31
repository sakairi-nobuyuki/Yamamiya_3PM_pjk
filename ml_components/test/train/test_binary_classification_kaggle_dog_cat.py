# coding: utf-8

import glob
import os
import random
from pathlib import Path
from typing import List

import kaggle

from ml_components.io import S3ImageIO

minio_endpoint_url = f"http://{os.environ['ENDPOINT_URL']}:9000"


s3_io = S3ImageIO(
    endpoint_url=minio_endpoint_url,
    access_key="sigma-chan",
    secret_key="sigma-chan-dayo",
    bucket_name="dataset",
)

### down loading data
s3_dataset_path = "classifier/kaggle/dog_cat"


# if not os.path.exists(dataset_path):
if not s3_dataset_path in s3_io.blob[0]:

    def create_partial_file_path_list(file_list: List[str], class_name: str) -> List[str]:
        # file_list = glob.glob(f"{temp_dataset_path}/kagglecatsanddogs_3367a/PetImages/{class_name}/*.*")
        file_list = [
            os.path.join(item.split("/")[-2], item.split("/")[-1]) for item in file_list
        ]
        file_list_length = len(file_list)
        template_list = int(file_list_length * 0.8) * ["train"] + int(
            file_list_length * 0.2
        ) * ["val"]
        random.shuffle(template_list)
        train_file_list = [
            item for (flag, item) in zip(template_list, file_list) if flag == "train"
        ]
        val_file_list = [
            item for (flag, item) in zip(template_list, file_list) if flag == "val"
        ]
        print(
            f"{class_name}: {len(file_list)}, {class_name} train: {len(train_file_list)}, {class_name} val: {len(val_file_list)}"
        )
        assert abs(len(file_list) - len(train_file_list) - len(val_file_list)) < 2
        assert (
            len(
                [
                    file_path
                    for file_path in train_file_list
                    if class_name not in file_path
                ]
            )
            == 0
        )
        assert (
            len([file_path for file_path in val_file_list if class_name not in file_path])
            == 0
        )

        return train_file_list, val_file_list

    print("Download Kaggle dataset")
    ### Download Kaggle dataset
    temp_dataset_path = os.path.join(
        str(Path(os.path.abspath(__file__)).parent.parent.parent), "classifier", "kaggle"
    )

    # os.makedirs(temp_dataset_path)
    kaggle.api.authenticate()
    # kaggle.api.dataset_download_files('karakaggle/kaggle-cat-vs-dog-dataset', path=dataset_path, unzip=True)

    # validation and train
    dog_file_list = glob.glob(
        f"{temp_dataset_path}/kagglecatsanddogs_3367a/PetImages/Dog/*.*"
    )
    dog_train_file_list, dog_val_file_list = create_partial_file_path_list(
        dog_file_list, "Dog"
    )
    cat_file_list = glob.glob(
        f"{temp_dataset_path}/kagglecatsanddogs_3367a/PetImages/Cat/*.*"
    )
    cat_train_file_list, cat_val_file_list = create_partial_file_path_list(
        cat_file_list, "Cat"
    )

    print(dog_file_list[0])
    prefix_dog = "kaggle/dog_cat/dog"
    # [s3_io.s3.meta.client.upload_file(file, s3_io.bucket_name, os.path.join(prefix_dog, "train", file.split("/")[-1])) for file in dog_train_file_list]
    for file in dog_train_file_list:
        print("transfer: ", file)
        s3_io.s3.meta.client.upload_file(
            file,
            s3_io.bucket_name,
            os.path.join(prefix_dog, "train", file.split("/")[-1]),
        )
