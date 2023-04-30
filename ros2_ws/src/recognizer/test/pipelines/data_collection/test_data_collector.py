# coding: utf-8

from recognizer.pipelines.data_collection import DataCollector
from recognizer.components.streamer import ImageStreamerElecom
from recognizer.io.save_image_s3 import S3ImageIO

class TestDataCollector:
    data_collector = DataCollector(ImageStreamerElecom(), S3ImageIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="data"))
    def test_init(self):
        assert isinstance(self.data_collector, DataCollector)

    def test_run(self):

        file_name = self.data_collector.run()

        assert isinstance(file_name, str)
