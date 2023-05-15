import os
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


def main(args: List[str] = None) -> None:
    rclpy.init(args=args)
    node = Node("node_name")
    parameter_file_path = os.path.join(os.path.dirname(__file__), "..", "config", "test.yaml")
    with open(parameter_file_path, "r") as f:
        parameters = Parameter.from_parameter_string(f.read())
        node.set_parameters(parameters)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
