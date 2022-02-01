# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 20:16:53 2021

@author: Christian Konstantinov
"""

import torch
from torch import nn

# TODO: Maxpooling
# TODO: 2 Models, or 1 end-to-end Model?
# TODO: maybe move train, test, and validate to this class?
# FIXME: Slow training time
class OnsetDetector(nn.Module):

    """Onset detection model for automatic rhythm game mapping."""

    def __init__(self, **kwargs):
        """Useful docstring goes here."""
        super(OnsetDetector, self).__init__()
        self.__dict__.update(**kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding='same')
        self.bidirectional = True
        self.D = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear((self.sequence_length * self.D * self.hidden_size) + 1, self.sequence_length)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0, x1 = x
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = self.relu(self.bn(self.conv(x0)))
        x0 = x0.squeeze(1)
        x0 = torch.transpose(x0, 1, 2)
        h0 = torch.zeros(self.num_layers * self.D, x0.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.D, x0.size(0), self.hidden_size).to(self.device)
        x0, _ = self.lstm(x0, (h0, c0))
        x0 = self.dropout(x0)
        x = torch.cat((x0.flatten(start_dim=1), x1), dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        return x