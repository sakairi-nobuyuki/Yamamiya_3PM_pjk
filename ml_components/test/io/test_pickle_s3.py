# coding: utf-8

import pickle

from ml_components.io import IOTemplate, PickleIO


class TestPickleIO:
    def test_init(self):
        pickle_obj = pickle.dumps("hoge")

        assert isinstance(pickle_obj, object)

    def test_save_load(self, mock_s3_pickle: IOTemplate):
        file_content = "hoge"
        file_path = "test/hoge.pickle"

        mock_s3_pickle.save(file_content, file_path)

        loaded = mock_s3_pickle.load(file_path)

        assert loaded == file_content

    def test_save_load(self, mock_s3_pickle: IOTemplate):
        file_content_list = ["hoge", "piyo"]
        file_path = "test/hoge.pickle"

        mock_s3_pickle.save(file_content_list, file_path)

        loaded_list = mock_s3_pickle.load(file_path)

        for file_content, loaded in zip(file_content_list, loaded_list):
            assert loaded == file_content
