import torch
import torch.nn as nn
import torch.nn.functional as F
from attVis.visualizer import get_local

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=2*d_model, dropout=dropout)

    def forward(self, x):
        att = self.attention(x, x, x)
        x = self.norm(x + att)
        ff = self.ff(x)
        x = self.norm(x + ff)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=2*d_model, dropout=dropout)

    def forward(self, x):
        att = self.attention(x, x, x)
        x = self.norm(x + att)
        ff = self.ff(x)
        x = self.norm(x + ff)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.l2(self.dropout(F.relu(self.l1(x))))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    @get_local('scores')
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.nhead, self.d_k).transpose(1, 2) # (bs, nhead, seq_len, d_k) -> (64, 8, 21, 128)
        k = self.k_linear(k).view(bs, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)
        return output
