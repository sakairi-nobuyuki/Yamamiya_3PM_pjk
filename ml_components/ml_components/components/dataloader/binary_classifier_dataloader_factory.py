# coding: utf-8

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ml_components.data_structures import ClassifierDataloaderDataclass
from ml_components.io import S3ImageIO


class BinaryClassifierDataloaderFactory:
    def __init__(self, s3: S3ImageIO):
        # Define transforms for data augmentation and normalization
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.s3 = s3

    def create(
        self,
        data_path: str,
        batch_size: int = 64,
        shuffle_train: bool = True,
        shuffle_val=False,
    ) -> ClassifierDataloaderDataclass:
        """Create Dataloader Dataclass from a dataset stored in a local storage.

        Args:
            data_path (str): Dataset path in a storage.

        Returns:
            ClassifierDataloaderDataclass: Dataloader dataclass.
        """
        # Downloading data
        # self.s3.download_s3_folder(data_path, local_dir=data_path)
        print(f"download files from {data_path}")
        self.s3.download_s3_folder(data_path)

        # Load images from directories using ImageFolder class
        train_dataset = datasets.ImageFolder(
            f"{data_path}/train", transform=self.train_transforms
        )
        val_dataset = datasets.ImageFolder(
            f"{data_path}/validation", transform=self.train_transforms
        )

        # Wrap datasets with DataLoader class
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle_val
        )

        # self.s3.delete_local(data_path)

        return ClassifierDataloaderDataclass(
            train_loader=train_loader, validation_loader=val_loader, test_loader=None
        )
