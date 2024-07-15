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

    def test_load_from_minio(self, mock_s3_pickle: IOTemplate):
        model_file_path = "classifier/vgg_umap/test_data/umap_model.pickle"
        mock_s3_pickle.get_blob()
        print(mock_s3_pickle.blob)
        assert (
            len([file_path for file_path in mock_s3_pickle.blob if model_file_path]) > 0
        )

        loaded_list = mock_s3_pickle.load(model_file_path)


#        assert isinstance(loaded_list, list)
