# coding: utf-8


from recognizer.components.region_extractor import ThresholdingDetector
from recognizer.io import S3ConfigIO, S3ImageIO


class TestThresholdingDetector:
    def test_init(self) -> None:
        detector = ThresholdingDetector()

        assert isinstance(detector, ThresholdingDetector)

    def test_detect_filter(
        self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO, mock_s3_config: S3ConfigIO
    ) -> None:
        detector = ThresholdingDetector(
            type="saturation", threshold_value=20, config_s3=mock_s3_config
        )
        img_file_name_list = [
            item for item in mock_s3_dataset.blob if "thresholding_train" in item
        ]
        for i_target, img_file_name in enumerate(img_file_name_list):
            target = mock_s3_dataset.load(img_file_name)
            cropped_img_list = detector.detect(target)
            for i_img, cropped_img in enumerate(cropped_img_list):
                print(f">> {i_img}th image shape: ", cropped_img.shape)
                mock_s3.save(cropped_img, f"fuga_{i_target}_{i_img}.png")

    def test_detect(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO) -> None:
        detector = ThresholdingDetector(type="saturation", threshold_value=20)
        # detector = ThresholdingDetector(type="saturation", threshold_value=100)
        # detector = ThresholdingDetector(threshold_value=100)
        img_file_name_list = [
            item for item in mock_s3_dataset.blob if "thresholding_train" in item
        ]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding_black_back" in item]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "thresholding" in item]

        print(img_file_name_list)

        for i_target, img_file_name in enumerate(img_file_name_list):
            target = mock_s3_dataset.load(img_file_name)
            print(type(target))
            cropped_img_list = detector.detect(target)

            print(f"{len(cropped_img_list)} images were found")
            # mock_s3.save(detector.gray, f"hoge_{i_target}.png")
            # mock_s3.save(detector.thresh, f"piyo_{i_target}.png")
            for i_img, cropped_img in enumerate(cropped_img_list):
                print(f">> {i_img}th image shape: ", cropped_img.shape)
                # mock_s3.save(cropped_img, f"fuga_{i_target}_{i_img}.png")
