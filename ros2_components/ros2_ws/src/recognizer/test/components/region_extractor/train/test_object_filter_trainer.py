# coding: utf-8

from recognizer.components.region_extractor.train import ObjectFilterTrainer
from recognizer.io import S3ConfigIO


class TestObjectFilterTrainer:
    def test_run(self, mock_s3_config: S3ConfigIO):

        trainer = ObjectFilterTrainer()

        trainer.run()

        assert isinstance(trainer, ObjectFilterTrainer)

        loaded_config = mock_s3_config.load("detector/object_filter_config.yaml")

        print("loaded config: ", loaded_config)
