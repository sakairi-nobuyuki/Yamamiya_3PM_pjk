# coding: utf-8

import numpy as np

from recognizer.io.save_image_s3 import S3ImageIO

class TestS3ImageIO:

    s3 = S3ImageIO(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo", bucket_name="data")
    def test_init(self):
        
        img = np.zeros((64,64,3), np.uint8)

        save_state = self.s3.save(img, "hoge.png")

        assert "hoge.png" in self.s3.get_blob()
        assert save_state.key == "hoge.png"
        assert save_state.bucket_name == "data"
        
        load_img = self.s3.load("hoge.png")

        np.testing.assert_array_almost_equal(img, load_img)

        self.s3.delete("hoge.png")

        print(self.s3.get_blob())
#        assert "hoge.png" not in self.s3.get_blob()
