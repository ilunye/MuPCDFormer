import torch.utils.data as data
import torch
import numpy as np
import pickle

class Data(data.Dataset):
    def __init__(self, dataset, mode):
        assert(mode in ['train', 'test'])
        self.data = np.load(f'data/{dataset}_{mode}.npz')['sequences']  # (seq, frame, points, 3)
        self.label = np.load(f'data/{dataset}_{mode}.npz')['labels']    # (seq,)
        self.catagory_num = int(np.load(f'data/{dataset}_{mode}.npz')['num_class'])
        self.clip_video_map = np.load(f'data/{dataset}_{mode}.npz')['clip_video_map']
        self.len = len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), self.label[index], self.clip_video_map[index]

    def __len__(self):
        return self.len