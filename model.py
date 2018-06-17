import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        in_channels = 1
        out_channels = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, embed_dim)
            )
                for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channels, class_num)

    def forward(self, x):
        #print(x)
        x = self.embed(x)  #(batch_size, text_max_len, embed_dim)

        if self.args.static:
            x = Variable(x)
        #text_max_len every batch
        x = x.unsqueeze(1) #(batch_size, in_channels=1, text_max_len, embed_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, in_channels=1, text_max_len), ...] * len(kernel_sizes)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, out_channels), ...]*len(kernel_sizes)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (batch_size, len(kernel_sizes)*out_channels)
        logit = self.fc(x)  # (batch_size, class_num)
        return logit