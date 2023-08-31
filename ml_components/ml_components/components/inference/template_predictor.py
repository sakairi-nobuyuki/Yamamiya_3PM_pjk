# coding: utf-8

from abc import ABCMeta, abstractmethod


class TemplatePredictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        pass
