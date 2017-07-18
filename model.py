from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

from config import Config

# class LSTMregrModel(nn.Module):
#     def __init__(self):
#         super(LSTMregrModel, self).__init__()
#         self.drop = nn.Dropout(Config.dropout)
#         self.lstm = nn.LSTM(
#             Config.ninput, Config.nhidd,
#             num_layers=Config.nlayers, batch_first=True, dropout=Config.dropout
#         )
#         self.fullc = nn.Linear(Config.nseries*Config.nhidd, Config.nout)
#         self.init_weight()
#         self.hidden = self.init_hidden()

#     def init_weight(self):
#         initrange = 0.01
#         self.fullc.bias.data.fill_(0)
#         self.fullc.weight.data.uniform_(-initrange, initrange)

#     def init_hidden(self):
#         h1 = torch.zeros(Config.nlayers, Config.batsize, Config.nhidd)
#         h2 = torch.zeros(Config.nlayers, Config.batsize, Config.nhidd)
#         if torch.cuda.is_available():
#             h1, h2 = h1.cuda(), h2.cuda()
#         return (Variable(h1), Variable(h2))

#     def forward(self, input):
#         lstm_out, _ = self.lstm(input, self.hidden)
#         lstm_out = self.drop(lstm_out)
#         output = self.fullc(lstm_out.view(-1, Config.nseries*Config.nhidd))
#         return output

class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()
        self.attnlinear = nn.Linear(Config.nhidd, Config.nhidd)
        self.tanh = nn.Tanh()
        self.attnvec = nn.Parameter(torch.Tensor(1, Config.nhidd))
        self.attnvec.data.uniform_(-0.1, 0.1)
        self.attnsoftmax = nn.Softmax()

    def forward(self, input):
        attn_scores = Variable(torch.zeros(Config.batsize, Config.nseries))
        if torch.cuda.is_available():
            attn_scores = attn_scores.cuda()
        for i in range(Config.batsize):
            for j in range(Config.nseries):
                attn_scores[i, j] = self.attnvec.dot(self.tanh(self.attnlinear(input[i, j].unsqueeze(0))))
        return self.attnsoftmax(attn_scores)

class LSTMregrModel(nn.Module):
    def __init__(self):
        super(LSTMregrModel, self).__init__()
        self.lstm = nn.LSTM(
            Config.ninput, Config.nhidd,
            num_layers=Config.nlayers, batch_first=True, dropout=Config.dropout
        )
        self.fullc = nn.Linear(Config.nhidd, Config.nout)
        self.init_weight()
        self.hidden = self.init_hidden()
        self.attn = Attn()

    def init_weight(self):
        initrange = 0.01
        self.fullc.bias.data.fill_(0)
        self.fullc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        h1 = torch.zeros(Config.nlayers, Config.batsize, Config.nhidd)
        h2 = torch.zeros(Config.nlayers, Config.batsize, Config.nhidd)
        if torch.cuda.is_available():
            h1, h2 = h1.cuda(), h2.cuda()
        return (Variable(h1), Variable(h2))

    def forward(self, input):
        lstm_out, _ = self.lstm(input, self.hidden)
        attn_weight = self.attn(lstm_out).view(-1, 1, Config.nseries)
        output = torch.bmm(attn_weight, lstm_out).view(-1, Config.nhidd)
        output = self.fullc(output)
        return output

def test():
    input = torch.rand(Config.batsize, Config.nseries, Config.ninput)
    model = LSTMregrModel()
    if torch.cuda.is_available():
        input = input.cuda()
        model.cuda()
    input = Variable(input)
    print(input)
    print(model.hidden)
    model.eval()
    out = model(input)
    print(out)

if __name__ == '__main__':
    test()