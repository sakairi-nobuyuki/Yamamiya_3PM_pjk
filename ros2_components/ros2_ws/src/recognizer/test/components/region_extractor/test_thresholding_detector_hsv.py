# coding: utf-8


from recognizer.components.region_extractor import ThresholdingDetectorHsv
from recognizer.io import S3ConfigIO, S3ImageIO


class TestThresholdingDetectorHsv:
    def test_init(self) -> None:
        detector = ThresholdingDetectorHsv()

        assert isinstance(detector, ThresholdingDetectorHsv)


    def test_detect(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO) -> None:
        detector = ThresholdingDetectorHsv()

        img_file_name_list = [
            item for item in mock_s3_dataset.blob if "thresholding_train" in item
        ]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding_black_back" in item]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding" in item]

        print(img_file_name_list)

        for i_target, img_file_name in enumerate(img_file_name_list):
            target = mock_s3_dataset.load(img_file_name)
            print("target: ", type(target), detector.lower_green, detector.upper_green, detector.threshold_value)
            cropped_img_list = detector.detect(target)

            assert isinstance(cropped_img_list, list)
            print("cropped bboxes: ", cropped_img_list)

