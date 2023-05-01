# coding: utf-8

import torch
import torch.nn as nn

from ml_components.models.factory import ModelFactoryTemplate
from ml_components.train import TemplateTrainer


class VggLikeClassifierTrainer(TemplateTrainer):
    def __init__(self, factory: ModelFactoryTemplate, n_epoch: int = 100):
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")

        self.factory = factory
        self.model = self.factory.create_model()
        self.forward = self.factory.create_forward()
        self.n_epochs = n_epoch

    def train(self):
        # Train the model on your dataset using binary cross-entropy loss and SGD optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Train the model for some number of epochs
        for epoch in range(self.num_epochss):
            # Train the model on your dataset for one epoch
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
