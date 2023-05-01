# coding: utf-8

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ClassifierDataloaderDataclass:
    train_loader: Optional[torch.utils.data.DataLoader]
    validation_loader: Optional[torch.utils.data.DataLoader]
    test_loader: Optional[torch.utils.data.DataLoader]
