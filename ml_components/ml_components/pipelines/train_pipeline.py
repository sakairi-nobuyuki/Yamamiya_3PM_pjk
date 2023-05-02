# coding: utf-8

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.io import S3ImageIO
from ml_components.models.factory import VggLikeClassifierFactory
from ml_components.train import VggLikeClassifierTrainer


class TrainPipeline:
    def __init__(self):
        image_s3 = S3ImageIO(
            endpoint_url="http://192.168.1.194:9000",
            access_key="sigma-chan",
            secret_key="sigma-chan-dayo",
            bucket_name="dataset",
        )

        self.vgg_like = VggLikeClassifierTrainer(
            "classifier/train",
            VggLikeClassifierFactory(),
            BinaryClassifierDataloaderFactory(image_s3),
            n_epoch=100,
        )

