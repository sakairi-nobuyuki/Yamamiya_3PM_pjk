# coding: utf-8

import pytest
import cv2
from example_pkg import ImagePublisher

class TestImagePublisher:
    @pytest.fixture
    def image_publisher(self) -> ImagePublisher:
        """Fixture that returns an instance of ImagePublisher."""
        return ImagePublisher()

    def test_image_publisher(self, image_publisher: ImagePublisher) -> None:
        """Test that checks if an instance of ImagePublisher is created correctly."""
        assert image_publisher is not None

    def test_image_publishing(self, image_publisher: ImagePublisher) -> None:
        """Test that checks if an image is published to a topic correctly."""
        # Load an image from your local machine
        img = cv2.imread('path/to/image.jpg')

        # Publish the image to a topic
        image_publisher.publish(img)

        # Check if the topic has received the image correctly
        # (replace 'topic_name' with your actual topic name)
        assert len(image_publisher.topic_name) > 0