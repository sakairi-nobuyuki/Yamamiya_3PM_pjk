# coding: utf-8

import json
import numpy as np
from ml_components.pipelines import TrainPipeline


class TestTrainPipeline:
    def test_vgg_like_umap_trainer(self) -> None:
        dataset_parameters_dict = dict(
            type="custom",
            train_data_rate=0.7,
            val_data_rate=0.2,
            dataset_name="yamamiya_pm",
            s3_dir="classifier/train",
        )
        train_parameters_dict = dict(
            type="umap_vgg_classification",
            dataset=dataset_parameters_dict,
            n_epoch=0,
            n_classes=2,
            base_model="VGG19",
        )
        parameters_str = json.dumps(train_parameters_dict)
        train_pipeline = TrainPipeline(parameters_str)
        reduced_feat = train_pipeline.run()

        assert isinstance(reduced_feat, np.ndarray)
