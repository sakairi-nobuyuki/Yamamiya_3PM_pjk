# coding: utf-8


from recognizer.components.region_extractor import ThresholdingDetectorSaturate
from recognizer.io import S3ConfigIO, S3ImageIO


class TestThresholdingDetectorSaturate:
    def test_init(self) -> None:
        detector = ThresholdingDetectorSaturate()
        

        assert isinstance(detector, ThresholdingDetectorSaturate)

    def test_detect(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO) -> None:
        detector = ThresholdingDetectorSaturate()
        # detector = ThresholdingDetector(type="saturation", threshold_value=100)
        # detector = ThresholdingDetector(threshold_value=100)
        img_file_name_list = [
            item for item in mock_s3_dataset.blob if "thresholding_train" in item
        ]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding_black_back" in item]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding" in item]

        # print(img_file_name_list)

        for i_target, img_file_name in enumerate(img_file_name_list):
            target = mock_s3_dataset.load(img_file_name)
            # print(img_file_name,  type(target))
            cropped_img_list = detector.detect(target)
            # print("cropped img list: ", cropped_img_list)

            assert isinstance(cropped_img_list, list)
            # print(cropped_img_list)
        # print("threshold value: ", detector.threshold_value)
