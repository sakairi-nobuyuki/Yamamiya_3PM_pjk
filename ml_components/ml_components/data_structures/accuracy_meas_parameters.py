# coding: utf-8

from typing import Optional

from pydantic import BaseModel

from . import DatasetLoaderParameters


class AccuracyMeasurementParameters(BaseModel):
    type: str = "binary"
    base_model: str
    dataset: Optional[DatasetLoaderParameters]
    models_directory_path: str
