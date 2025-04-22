import torch
from torch import nn


class LinearwithShortcut(nn.Module):
    def __init__(self,in_features:int,hidden_features:int,out_features:int,activate_class:nn.Module=nn.ReLU):
        super(LinearwithShortcut,self).__init__()
        self.activate_function = activate_class()
        self.linear1 = nn.Linear(in_features,hidden_features)
        self.linear2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features,hidden_features),
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

    def forward(self,x):
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
        
        self.DFL_linear = LinearwithShortcut(1024, 512, 1)


    def forward(self, input_feature):
        input_feature = input_feature.to(torch.float32)  # [batch_size, L ,1024]
        encoder_outputs, encoder_hiddens = self.Bi_LSTM(input_feature)  # layer_outputs: [batch, seq_len, 2*hidden_size]
        y_DFL = self.DFL_linear(encoder_outputs)
        return y_DFL
