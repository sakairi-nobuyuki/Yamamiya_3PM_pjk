# coding: utf-8

import json

import numpy as np
import torchvision
import umap
from sklearn.linear_model import LogisticRegression

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
        label_map_dict = train_pipeline.trainer.get_label_map_dict()
        assert isinstance(label_map_dict, dict)
        assert len(label_map_dict) == 2

        model_dir_path = train_pipeline.run()

        assert isinstance(model_dir_path, str)

        assert isinstance(train_pipeline.trainer.reducer.reducer, umap.UMAP)
        assert isinstance(train_pipeline.trainer.reducer.regression, LogisticRegression)
        assert isinstance(train_pipeline.trainer.vgg.model, torchvision.models.vgg.VGG)

        s3_blob = train_pipeline.trainer.transfer_io.get_blob()

        model_cand_items = [item for item in s3_blob if model_dir_path]

        assert len(model_cand_items) >= 3

        model_cand_items_ext_list = [item.split(".")[-1] for item in model_cand_items]

        assert "pickle" in model_cand_items_ext_list
        assert "pth" in model_cand_items_ext_list
        assert "yaml" in model_cand_items_ext_list
