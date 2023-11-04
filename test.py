import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.tensorboard.writer import SummaryWriter
import datetime

from utils.dataloader import Data
from models.net import MuPCDFormer

def num_to_class(num):
    if num==0:
        return "Approaching"
    elif num==1:
        return "Departing"
    elif num==2:
        return "Kicking"
    elif num==3:
        return "Pushing"
    elif num==4:
        return "Shaking Hands"
    elif num==5:
        return "Hugging"
    elif num==6:
        return "Exchanging"
    elif num==7:
        return "Punching"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--load_weights', required=True)
    parser.add_argument('--dataset', type=str, default='sbu')
    parser.add_argument('--dropout', type=float, default=0.3)
    # data
    parser.add_argument('--num_points', '-n', type=int, default=4096)
    parser.add_argument('--seq-len', '-l', type=int, default=21)
    parser.add_argument('--anchor-point-num', '-a', type=int, default=512)
    # feature
    parser.add_argument('--feature_dim', type=int, default=1024)
    # transformer
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=8)

    opt = parser.parse_args()

    print("Loading data...")
    test_dataset  = Data(dataset=opt.dataset, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    opt.catagory_num = test_dataset.catagory_num

    print("Creating model...")
    model = MuPCDFormer(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(opt.load_weights)['model']
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        correct_seq = 0
        all_seq = 0
        for i, batch_data in enumerate(test_dataloader):
            data, label, _ = batch_data
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            ans = output.argmax(dim=1)
            correct_seq += (ans==label).sum().item()
            all_seq += label.shape[0]
            if opt.batch_size==1:
                print(f"Data{i} Predict: {num_to_class(ans.item()):15s} GT: {num_to_class(label.item()):15s}, {'Correct' if ans==label else 'Wrong'}")
        print(f"Test acc: {correct_seq/all_seq}")



if __name__ == '__main__':
    main()