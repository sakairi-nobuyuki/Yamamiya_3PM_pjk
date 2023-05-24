# coding: utf-8

from typing import Any

import torch
import torch.nn as nn

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.models.factory import ModelFactoryTemplate
from ml_components.train import TemplateTrainer
from tqdm import tqdm

class VggLikeClassifierTrainer(TemplateTrainer):
    def __init__(
        self,
        data_path: str,
        factory: ModelFactoryTemplate,
        dataloader_factory: BinaryClassifierDataloaderFactory,
        n_epoch: int = 100,
    ) -> None:
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")
        self.device = torch.device('cuda')

        self.factory = factory
        self.model = self.factory.create_model()
        self.model.to(self.device)
        self.forward = self.factory.create_forward()
        self.n_epochs = n_epoch
        self.dataloader = dataloader_factory.create(data_path)
        # Train the model on your dataset using binary cross-entropy loss and SGD optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        # Train the model for some number of epochs
        train_loss_list = []
        validation_loss_list = []
        for epoch in range(self.n_epochs):
            print(f"{epoch}th epoch:")
            train_loss = self.iterate_train(self.model, self.dataloader.train_loader)
            train_loss_list.append(train_loss)
            print(f">> train loss: {train_loss}")
            validation_loss = self.iterate_validation(
                self.model, self.dataloader.validation_loader
            )
            if epoch == 0:
                self.save_model(epoch, self.model, train_loss)
            elif validation_loss_list[-1] > validation_loss:
                self.save_model(epoch, self.model, train_loss)
            validation_loss_list.append(validation_loss)
            print(f">> validation loss: {validation_loss}")
        print("train loss history: ", train_loss_list)
        print("validation loss history: ", validation_loss_list)

    def iterate_train(self, model: Any, dataloader: torch.utils.data.DataLoader) -> float:
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

    def iterate_validation(self, model: Any, dataloader: torch.utils.data.DataLoader) -> float:
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

    def save_model(self, epoch: int, model: Any, train_loss: float):
        print("save model")
        # Save a checkpoint of the model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
        }
        torch.save(checkpoint, f'checkpoint_{epoch}.pth')        
