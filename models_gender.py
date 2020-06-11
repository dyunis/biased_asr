import numpy as np
import torch


class LSTM_gender(torch.nn.Module):
#     51 magic number comes from 50 characters plus blank
    def __init__(self, input_dim=40, hidden_dim=512, out_dim=51, num_layers=1, 
                 bias=False, bidirectional=True):

        super(LSTM_gender, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                                  num_layers=num_layers, bias=bias, 
                                  bidirectional=bidirectional)

        self.num_layers = num_layers
        self.num_dirs = int(bidirectional) + 1
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(self.num_dirs * self.hidden_dim, 
                                      self.out_dim)
        self.linear_gender=torch.nn.Linear(self.num_dirs * self.hidden_dim, 
                                      2)
        self.classifier = torch.nn.LogSoftmax(dim=-1)

    def forward(self, X, classify=True):
        '''
        X: (batch, seq_len, F) 
        y: (batch, seq_len, V)
        gen_y:(batch, seq_len, V)
        '''
        # re-init hidden so that changing batch size isn't a problem
        h0 = self.init_hidden(len(X))
        embed, self.hidden = self.lstm(X.transpose(0, 1), h0)

        y = self.linear(embed).transpose(0, 1)
        y = self.classifier(y)
        gen_y=self.linear_gender(embed).transpose(0,1)
        gen_y=self.classifier(gen_y)

        return y,gen_y, embed.transpose(0, 1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers * self.num_dirs, batch_size,
                                 self.hidden_dim),
                weight.new_zeros(self.num_layers * self.num_dirs, batch_size,
                                 self.hidden_dim))
