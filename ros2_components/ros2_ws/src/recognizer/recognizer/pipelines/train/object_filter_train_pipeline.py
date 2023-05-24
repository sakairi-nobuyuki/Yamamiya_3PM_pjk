# coding: utf-8

from ...components.region_extractor.train import ObjectFilterTrainer
from ...io import S3ConfigIO, S3ImageIO

class ObjectFilterTrainerPipeline:
    def __init__(self, dataset_s3: S3ConfigIO, config_s3: S3ConfigIO) -> None:
        """Initialize various training modules"""
        if not isinstance(dataset_s3, S3ImageIO):
            raise TypeError(f"dataset_s3 should be an instance of S3ImageIO: {type(dataset_s3)}")
        if not isinstance(config_s3, S3ConfigIO):
            raise TypeError(f"config_s3 should be an instance of S3ConfigIO: {type(config_s3)}")
        self.s3_config = config_s3
        self.s3_dataset = dataset_s3
        self.trained_parameter_file_name = "detector/object_filter.yaml"
        self.trainer = ObjectFilterTrainer(self.s3_dataset)
        
    def run(self) -> None:
        res_dict = self.trainer.run()

        self.s3_config.save(res_dict, self.trained_parameter_file_name)
