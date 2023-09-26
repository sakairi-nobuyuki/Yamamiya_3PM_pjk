# coding: utf-8

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class DatasetLoaderParameters(BaseModel):
    type: str
    train_data_rate: float
    val_data_rate: float
    dataset_name: str
    local_dir: Optional[str] = "tmp"
    s3_dir: str

    @model_validator(mode="after")
    def __validate_parameters_existence__(self) -> "DatasetLoaderParameters":
        total_rate = self.train_data_rate + self.val_data_rate
        if not self.is_prob_measure(total_rate):
            raise ValueError(
                f"train and validation dataset ratio is larger than 1: {total_rate}"
            )
        if not self.is_prob_measure(self.train_data_rate):
            raise ValueError(
                f"train dataset ratio is larger than 1: {self.train_data_rate}"
            )
        if not self.is_prob_measure(self.val_data_rate):
            raise ValueError(
                f"validation dataset ratio is larger than 1: {self.val_data_rate}"
            )

        return self

    def is_prob_measure(self, value: float) -> bool:
        return value * (value - 1.0) < 1.0
