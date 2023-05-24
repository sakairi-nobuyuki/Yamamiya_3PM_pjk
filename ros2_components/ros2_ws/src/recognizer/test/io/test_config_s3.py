# coding: utf-8

import numpy as np
from recognizer.io import S3ConfigIO


class TestS3ConfigIO:

    s3 = S3ConfigIO(
        endpoint_url="http://192.168.1.194:9000",
        access_key="sigma-chan",
        secret_key="sigma-chan-dayo",
        bucket_name="config",
    )

    def test_init(self):

        test_dict = dict(hoge="piyo", fuga="hogera")

        save_state = self.s3.save(test_dict, "hoge.yaml")

        assert "hoge.yaml" in self.s3.get_blob()
        assert save_state.key == "hoge.yaml"
        assert save_state.bucket_name == "config"

        loaded_dict = self.s3.load("hoge.yaml")

        for item_key, item_value in loaded_dict.items():
            assert item_key in test_dict.keys()
            assert test_dict[item_key] == item_value
            print(item_key, item_value)

        self.s3.delete("hoge.png")

        print(self.s3.get_blob())


#        assert "hoge.png" not in self.s3.get_blob()
