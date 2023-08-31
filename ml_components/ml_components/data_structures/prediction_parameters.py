# coding: utf-8

from typing import Optional
from pydantic import BaseModel

class PredictionParameters(BaseModel):
    """Prediction parametres.
    In the top level parameters data class, basic configuration of the inference is set.
    Detail configuration such as thresholding values, input image dimension or other items should be implemented sub-dataclass of 
    this dataclass.

    Attributes:
        model_path (str): DNN or other ML model path in a cloud storage.
        base_model (str): If the model using any kind of pretrained model such VGG, ResNet or Yolo, set the base model name.
            Options: VGG19, VGG16, ResNet
        category (str): The prediction category if the predictor applys DNN, DNN like model, or classical ML model.
            Options: dnn, umap, knn
        type (str): Type of the inference. It influences the return type of the inference.
            Options: binary, n-class, detection

    Args:
        BaseModel (_type_): Inhering pydantic.BaseModel
    """
    model_path: str
    base_model: Optional[str] = None
    category: str = "dnn"
    type: str = "binary"
    