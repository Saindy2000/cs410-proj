from torch import nn
import torch
import numpy as np
import math

# from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.dropout(x + self.pos_table[:, :x.size(1)].clone().detach())

class Transformer(nn.Module):
    def __init__(self, ntoken, d_model, d_hidden, nhead=4, nlayer=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.d = d_model
        self.embedding = nn.Embedding(ntoken, d_model)
        self.posEncode = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hidden, dropout), nlayer)
        self.pool = nn.MaxPool1d(64)
        self.u_decode = nn.Linear(d_model, d_model)
        self.i_decode = nn.Linear(d_model, d_model)
        self.r_decode = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d)
        x = self.posEncode(x)
        o = self.transformer(x)
        o = self.pool(o.transpose(0, 2)).transpose(2, 0).squeeze(0)
        u_embs = self.u_decode(o)
        i_embs = self.i_decode(o)
        r = nn.functional.softmax(self.r_decode(o))[:, 0]
        return u_embs, i_embs, r
