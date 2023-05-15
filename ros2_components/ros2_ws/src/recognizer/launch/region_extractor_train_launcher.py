from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file", description="config/region_extractor/train_config.yaml"
            ),
            Node(
                package="recognizer",
                node_executable="region_extractor_train_node.py",
                node_name="region_extractor_train_node",
                parameters=[LaunchConfiguration("config_file")],
                remappings=[],
                output="screen",
            ),
        ]
    )
