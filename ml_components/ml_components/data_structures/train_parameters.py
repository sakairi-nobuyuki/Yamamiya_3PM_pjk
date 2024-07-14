# coding: utf-8

from typing import Optional

from pydantic import BaseModel

from . import DatasetLoaderParameters


class TrainParameters(BaseModel):
    type: str = "classification"
    dataset: Optional[DatasetLoaderParameters]
    n_epoch: int = 100
    n_classes: int = 2
    base_model: str = "VGG19"
    model_dir_path: str = None
    local_dataset_path: str = "classifier/train"
