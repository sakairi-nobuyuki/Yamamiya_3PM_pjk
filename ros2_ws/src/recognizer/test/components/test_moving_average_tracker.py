# coding: utf-8

from recognizer.components.region_extractor import MovingAverageTracker
from recognizer.io.save_image_s3 import S3ImageIO

class TestMovingAverageTracker:
    def test_init(self) -> None:
        
        tracker = MovingAverageTracker()

        assert isinstance(tracker, MovingAverageTracker)

    def test_tracking(self, mock_s3_dataset: S3ImageIO, mock_s3: S3ImageIO) -> None:
        tracker = MovingAverageTracker()

        img_file_name_list = [item for item in mock_s3_dataset.blob if "threshold" in item]
        # img_file_name_list = [item for item in mock_s3_dataset.blob if "moving_average" in item]

        print(img_file_name_list)
        
        for i_target, img_file_name in enumerate(img_file_name_list):
            target = mock_s3_dataset.load(img_file_name)
            print(type(target))
            cropped_img_list = tracker.traking(target)

            print(f"{len(cropped_img_list)} images were found")
            mock_s3.save(tracker.old_img, f"piyo_{i_target}.png")
            mock_s3.save(tracker.thresh, f"fuga{i_target}.png")
            for i_img, cropped_img in enumerate(cropped_img_list):
                print(f">> {i_img}th image shape: ", cropped_img.shape)
                mock_s3.save(cropped_img, f"hoge_{i_target}_{i_img}.png")
                
