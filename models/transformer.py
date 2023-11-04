import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.sublayers import EncoderLayer, DecoderLayer
import copy

class Transformer(nn.Module):
    def __init__(self, d_model, seq_len, n_layers ,nhead, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, seq_len=seq_len, n_layers=n_layers, nhead=nhead, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, seq_len=seq_len, n_layers=n_layers, nhead=nhead, dropout=dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        e_output = self.encoder(x)
        d_output = self.decoder(e_output)
        output = self.out(d_output)
        return output

class Encoder(nn.Module):
    def __init__(self, d_model, seq_len, n_layers, nhead, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, seq_len, n_layers, nhead, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(DecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return x