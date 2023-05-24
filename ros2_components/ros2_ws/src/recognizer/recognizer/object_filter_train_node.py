# coding: utf-8

import rclpy
from rclpy.node import Node

from .io import S3ConfigIO, S3ImageIO
from .pipelines.train.object_filter_train_pipeline import ObjectFilterTrainerPipeline


class ObjectFilterTrainNode(Node):
    def __init__(self) -> None:
        super().__init__("object_filter_trainer_node")
        self.node_name = self.get_name()
        self.get_logger().info(f"{self.node_name} Start train")

        config_s3 = S3ConfigIO(
            endpoint_url="http://192.168.1.194:9000",
            access_key="sigma-chan",
            secret_key="sigma-chan-dayo",
            bucket_name="config",
        )
        dataset_s3 = S3ImageIO(
            endpoint_url="http://192.168.1.194:9000",
            access_key="sigma-chan",
            secret_key="sigma-chan-dayo",
            bucket_name="dataset",
        )
        self.get_logger().info(f"{self.node_name} Initializing trainer")
        self.trainer = ObjectFilterTrainerPipeline(dataset_s3, config_s3)
        self.get_logger().info(f"{self.node_name} Initialized trainer")

    def spin_callback(self) -> None:
        self.get_logger().info(f"{self.node_name} Start train")

        self.trainer.run()


def main(args=None) -> None:

    rclpy.init(args=args)
    node = rclpy.create_node("object_filter_train_node")
    trainer = ObjectFilterTrainNode()
    trainer.spin_callback()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
