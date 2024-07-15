# coding: utf-8

import shutil
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.io import DataTransferS3, IOTemplate
from ml_components.models.factory import ModelFactoryTemplate
from ml_components.train import TemplateTrainer


class VggLikeClassifierTrainer(TemplateTrainer):
    """Train a binary classifier from VGG"""

    def __init__(
        self,
        data_path: str,
        factory: ModelFactoryTemplate,
        dataloader_factory: BinaryClassifierDataloaderFactory,
        io: DataTransferS3,
        n_epoch: int = 100,
    ) -> None:
        """Constructor
        - Initialize torch
        - Create model from VGG
        - Configure a sequential network
        - Configure a data loader

        Args:
            data_path (str): Train data path
            factory (ModelFactoryTemplate): An instance of model factory
            dataloader_factory (BinaryClassifierDataloaderFactory): An instance of data loader factory
            n_epoch (int, optional): Number of epoch. Defaults to 100.

        Raises:
            TypeError: Raise a type error for the model factory
        """
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")

        self.device = torch.device("cuda")
        self.io = io
        self.factory = factory
        self.model = self.factory.create_model()
        self.model.to(self.device)
        self.forward = self.factory.create_forward()
        self.n_epochs = n_epoch
        self.dataloader = dataloader_factory.create(data_path)
        # Train the model on your dataset using binary cross-entropy loss and SGD optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )

    def train(self):
        # Train the model for some number of epochs
        train_loss_list = []
        validation_loss_list = []
        summary_list = []
        for epoch in range(self.n_epochs):
            print(f"{epoch}th epoch:")
            train_loss = self.iterate_train(self.model, self.dataloader.train_loader)
            train_loss_list.append(train_loss)
            print(f">> train loss: {train_loss}")
            validation_loss = self.iterate_validation(
                self.model, self.dataloader.validation_loader
            )
            if epoch == 0:
                summary = self.save_model(
                    epoch, self.model, train_loss, validation_loss
                )
            elif validation_loss_list[-1] > validation_loss:
                summary = self.save_model(
                    epoch, self.model, train_loss, validation_loss
                )
            validation_loss_list.append(validation_loss)
            summary_list.append(summary)
            print(f">> validation loss: {validation_loss}")
        print("train loss history: ", train_loss_list)
        print("validation loss history: ", validation_loss_list)
        val_loss_list = [item["val_loss"] for item in summary_list]
        min_val_loss_index = validation_loss_list.index(min(val_loss_list))
        print(
            "minimum val loss: ", min_val_loss_index, val_loss_list[min_val_loss_index]
        )
        file_name = [item["file_name"] for item in summary_list][min_val_loss_index]
        print("File name to upload: ", file_name)
        self.io.save(file_name, f"binary_classifier/vgg_base/{file_name}")
        map(shutil.rmtree, [item for item in summary_list["file_name"]])

    def iterate_train(
        self, model: Any, dataloader: torch.utils.data.DataLoader
    ) -> float:
        model.train()
        train_loss = 0.0

        # Train the model on your dataset for one epoch
        with tqdm(dataloader, total=len(dataloader)) as train_progress:
            for inputs, labels in train_progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
        train_loss = train_loss / float(len(dataloader))
        return train_loss

    def iterate_validation(
        self, model: Any, dataloader: torch.utils.data.DataLoader
    ) -> float:
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()
        return validation_loss / float(len(dataloader))

    def save_model(
        self,
        epoch: int,
        model: Any,
        train_loss: float,
        val_loss: float,
        model_name: str = None,
    ) -> Dict[str, str]:
        # Save a checkpoint of the model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
        }
        current_time = datetime.now().strftime("%Y%m%d")
        if model_name is None:
            file_name = f"checkpoint_{current_time}_{epoch}epoch_train_loss{train_loss}_val_loss{val_loss}.pth"
        else:
            file_name = model_name
        torch.save(checkpoint, file_name)
        print(f"save model: {file_name}")

        return {"file_name": file_name, "train_loss": train_loss, "val_loss": val_loss}

    def get_label_map_dict(self) -> Dict[int, str]:
        pass


def main():
    """Production code should be implemented here for simplicity"""
    pass
