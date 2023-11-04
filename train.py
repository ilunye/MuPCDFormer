import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.tensorboard.writer import SummaryWriter
import datetime

from utils.dataloader import Data
from models.net import MuPCDFormer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

def train(model, batch_data, device, opt):
    data, label, _ = batch_data
    data = data.to(device)
    label = label.to(device)
    output = model(data)
    ans = output.argmax(dim=1)
    acc = (ans == label).sum().item() / label.shape[0]
    loss = F.cross_entropy(output, label)
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_step', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--load_weights')
    parser.add_argument('--dataset', type=str, default='sbu')
    parser.add_argument('--comment', type=str, default='')
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
    setup_seed(opt.seed)

    print("Loading data...")
    train_dataset = Data(dataset=opt.dataset, mode='train')
    test_dataset  = Data(dataset=opt.dataset, mode='test')
    opt.catagory_num = train_dataset.catagory_num

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    print("Creating model...")
    model = MuPCDFormer(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)

    best_acc = 0
    if opt.load_weights is not None:
        checkpoint = torch.load(opt.load_weights)['model']
        best_acc = torch.load(opt.load_weights)['best_acc']
        model.load_state_dict(checkpoint)
        print("best_acc:", best_acc)
    writer = SummaryWriter()
    
    print("Start training...")
    for epoch_i in range(1, opt.epochs+1):
        model.train()
        loss_list = []
        acc_list = []
        for _, batch_data in enumerate((train_dataloader)):
            loss, acc = train(model, batch_data, device, opt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            acc_list.append(acc)
        train_loss_cur = np.mean(loss_list)
        train_acc_cur = np.mean(acc_list)
        print(f'epoch: {epoch_i:5d}, train_loss: {train_loss_cur:.6f}, train_accuracy: {train_acc_cur:.6f}', end=' ')
        writer.add_scalar('train/loss', train_loss_cur, epoch_i)
        writer.add_scalar('train/accuracy', train_acc_cur, epoch_i)

        with torch.no_grad():
            correct_seq = 0
            all_seq = 0
            model.eval()
            for _, batch_data in enumerate(test_dataloader):
                data, label, _ = batch_data
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                ans = output.argmax(dim=1)
                correct_seq += (ans == label).sum().item()
                all_seq += label.shape[0]
            test_acc_cur = correct_seq / all_seq
            print(f'test_accuracy: {test_acc_cur:.6f}, lr={optimizer.param_groups[0]["lr"]:.6f}, best_acc={best_acc:.6f}')
            # writer.add_scalar('test/loss', test_loss_cur, epoch_i)
            writer.add_scalar('test/accuracy', test_acc_cur, epoch_i)

        scheduler.step()
        if test_acc_cur > best_acc:
            print("save best model!")
            best_acc = test_acc_cur
        #     torch.save({'model':model.state_dict(), 'best_acc':best_acc}, f'checkpoints/best{opt.comment}.pth')
        # torch.save({'model':model.state_dict(), 'best_acc':best_acc},     f'checkpoints/last{opt.comment}.pth')
        if epoch_i % 50 == 0:
            for g in optimizer.param_groups:
                g['lr'] = opt.lr


if __name__ == '__main__':
    main()