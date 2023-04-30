# coding: utf-8

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from .pipelines.data_collection import DataCollector
from .components.streamer import ImageStreamerElecom
from .io import S3ImageIO

class ImageStreamingNode(Node):
    def __init__(self) -> None:
        super().__init__('image_publisher')

        ### loading parameters 
#        self.declare_parameter("interval")
#        self.declare_parameter("data_save_flag")        
        
#        timer_period = self.get_parameter('interval').get_parameter_value().integer_value
#        self.data_save_flag = False
#        if self.get_parameter('data_save_flat').get_parameter_value() is True:
#            self.data_save_flag = True

        timer_period = 5
        self.data_save_flag = True
        ### Initialize components
        self.streamer = ImageStreamerElecom()
        self.io = S3ImageIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="data")
        if self.data_save_flag is True:
            self.data_collector = DataCollector(self.streamer, self.io)

        ### create a publisher
        self.publisher_ = self.create_publisher(Image, 'image', 10)
        
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()

    def timer_callback(self) -> None:
        """Publishes an image periodically"""
        img = self.streamer.capture()
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.publisher_.publish(msg)
        if self.data_save_flag:
            self.data_collector.save(img)



def main(args=None) -> None:
    """Initializes ROS2 node and spins it."""
    rclpy.init(args=args)
    image_publisher = ImageStreamingNode()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()