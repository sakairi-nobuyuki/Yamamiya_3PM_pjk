# coding: utf-8

from ml_components.data_structures import ClassifierDataloaderDataclass


class TestClassifierDataloaderDataclass:
    def test_init_blank(self, mock_dataloader):
        loader = ClassifierDataloaderDataclass(
            train_loader=None, validation_loader=None, test_loader=None
        )

        assert isinstance(loader, ClassifierDataloaderDataclass)

    def test_init_mock(self, mock_dataloader):
        loader = ClassifierDataloaderDataclass(
            train_loader=mock_dataloader,
            validation_loader=mock_dataloader,
            test_loader=mock_dataloader,
        )

        assert isinstance(loader, ClassifierDataloaderDataclass)
