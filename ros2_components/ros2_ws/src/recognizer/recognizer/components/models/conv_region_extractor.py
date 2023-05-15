# coding: utf-8

import torch


class ConvolutionalRegionExtractNetwork(torch.nn.Module):
    def __init__(self, input_size: int, n_conv_kernel: int) -> None:

        super().__init__()

        self.conv = torch.nn.Conv2d(2, input_size, 5)

    #        self.network = torch.nn.Conv2d(2, 128, 5)
    # self.network = torch.nn.RNN(1, hidden_layer_units, batch_first=True)
    # self.network = torch.nn.RNN(input_size, hidden_layer_units, batch_first=True)
    #        self.fc = torch.nn.Linear(hidden_layer_units, 1)

    def forward(self, x) -> torch.nn.Linear:

        x = self.conv(x)


#        y, h = self.network(x, None)
#        return self.fc(y[:, -1, :])
