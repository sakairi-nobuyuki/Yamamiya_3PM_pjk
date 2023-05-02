# coding: utf-8

from abc import ABCMeta, abstractmethod


class TemplateTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass
