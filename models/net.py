import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import utils.pointnet2_utils as pointnet2_utils
from models.transformer import Transformer

class MuPCDFormer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.bs = opt.batch_size
        self.seq_len = opt.seq_len
        self.num_points = opt.num_points
        self.anchor_point_num = opt.anchor_point_num

        self.conv_pointnet2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=47, stride=15),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=1024, kernel_size=32),
            nn.ReLU()
        )

        self.biLSTM = nn.LSTM(input_size=self.anchor_point_num*3, hidden_size=self.anchor_point_num//2, num_layers=2, bidirectional=True, batch_first=True, dropout=opt.dropout)

        self.mlp_beforeTransformer = nn.Sequential(
            nn.Linear(opt.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, opt.d_model),
            nn.ReLU()
        )
        self.transformer = Transformer(opt.d_model, opt.seq_len, opt.n_layers, opt.n_heads, opt.dropout)
        self.mlp_afterTransformer = nn.Sequential(
            nn.Linear(opt.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, opt.feature_dim),
            nn.ReLU()
        )
        self.mlp_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(opt.feature_dim*opt.seq_len, opt.feature_dim),
            nn.ReLU(),
            nn.Linear(opt.feature_dim, int(math.sqrt(opt.feature_dim))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(opt.feature_dim)), opt.catagory_num),
            nn.Softmax(dim=1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):
        self.bs = x.shape[0]
        # 1. Furthest Point Sampling
        frames=[]
        for t in range(x.shape[1]):
            anchor_idx = pointnet2_utils.furthest_point_sample(x[:,t,:,:].contiguous(), self.anchor_point_num)
            anchor_point = pointnet2_utils.gather_operation(x[:,t,:,:].transpose(1, 2).contiguous(), anchor_idx)  # (bs, 3, anchor_point_num)
            frames.append(anchor_point)
        anchor_xyz = torch.stack(frames, dim=1)  # (bs, seq_len, 3, anchor_point_num)

        # 2. Bi-LSTM
        anchor_p = anchor_xyz.reshape(self.bs, self.seq_len, -1)  # (bs, seq_len, 3*anchor_point_num)
        anchor_p = self.biLSTM(anchor_p)[0].unsqueeze(2)  # (bs, seq_len, anchor_point_num)
        anchor_xyzp = torch.cat((anchor_xyz, anchor_p), dim=2)  # (bs, seq_len, 4, anchor_point_num)
        feature_frames = anchor_xyzp.reshape(-1, 4, self.anchor_point_num)  # (bs*seq_len, 4, anchor_point_num)

        # 3. Convolutions
        feature_frames = self.conv_pointnet2(feature_frames).reshape(self.bs, self.seq_len, -1) # (bs, seq_len, feature_dim)

        # 4. transformer
        transformer_out = self.mlp_afterTransformer(self.transformer(self.mlp_beforeTransformer(feature_frames)))
        out = self.mlp_out(feature_frames+transformer_out)
        return out

