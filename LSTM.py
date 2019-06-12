import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, args, weight=None):
        super(LSTM, self).__init__()
        self.args = args
        word_embed_num = args.embed_num
        word_embed_dim = args.embed_dim
        class_num = args.class_num
        self.embed = nn.Embedding(word_embed_num, word_embed_dim)
        self.hidden_dim = args.hidden_dim
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(2 * self.hidden_dim , class_num)
        if weight is not None:
            self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.hidden = self.init_lstm_hidden()

    def init_lstm_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x)
        lstm_out, self.hidden = self.lstm(x, None)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        logit = self.fc(x)
        return logit