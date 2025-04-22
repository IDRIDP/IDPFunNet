import torch
from torch import nn
import numpy as np

class LinearwithShortcut(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, activate_class: nn.Module = nn.ReLU):
        super(LinearwithShortcut, self).__init__()
        self.activate_function = activate_class()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.linear3 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.linear4 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, out_features)
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = x + self.linear2(x)
        x = x + self.linear3(x)
        y = self.activate(self.linear4(x))
        return y


class Bi_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.Bi_LSTM = nn.LSTM(1024, 512,
                               batch_first=True,
                               bidirectional=True)
        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(1024, 512)
        self.PB_linear = LinearwithShortcut(1024, 512, 1)
        self.NB_linear = LinearwithShortcut(1024, 512, 1)
        self.LB_linear = LinearwithShortcut(1024, 512, 1)
        self.IB_linear = LinearwithShortcut(1024, 512, 1)
        self.SB_linear = LinearwithShortcut(1024, 512, 1)

    def forward(self, input_feature):
        input_feature = input_feature.to(torch.float32)  # [batch_size, L ,1024]
        prot5_feature = input_feature
        af2_feature = input_feature.permute(0, 2, 1)
        af2_feature = self.conv1(af2_feature)
        af2_feature = self.relu(af2_feature)
        af2_feature = self.conv2(af2_feature)
        af2_feature = af2_feature.permute(0, 2, 1)
        encoder_outputs, encoder_hiddens = self.Bi_LSTM(prot5_feature)  # layer_outputs: [batch, seq_len, 2*hidden_size]
        combined_feature = torch.cat([self.linear1(encoder_outputs), self.linear2(af2_feature)], dim=2)
        y_PB = self.PB_linear(combined_feature)
        y_NB = self.NB_linear(combined_feature)
        y_LB = self.LB_linear(combined_feature)
        y_IB = self.IB_linear(combined_feature)
        y_SB = self.SB_linear(combined_feature)
        return torch.cat((y_PB, y_NB, y_LB, y_IB, y_SB), dim=-1)


