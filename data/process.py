import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
# import scipy.io as sio
from tqdm import tqdm

'''
type 1: approaching
type 2: departing
type 3: kicking
type 4: pushing
type 5: shaking hands
type 6: hugging
type 7: exchanging
type 8: punching
'''

W = 640
H = 480
focal = 280

fig = plt.figure()
ax=fig.add_subplot(projection='3d')


videos=[]   # save after by npz
videos_train = []   # clips for training
videos_test = []    # clips for testing
labels_train = []     # labels for training
labels_test = []      # labels for testing
sequences_train = []  # a sequence is a short clip
sequences_test = []
txt_train = []
txt_test = []

dir_len = []
point_num = []

xx, yy = np.meshgrid(np.arange(W), np.arange(H))
cnt = 0
print("begin iterating")
num_class = 0
PATH_TO_DATA = '/data1/ilunye/datasets/SBU/data/'
for dir1 in tqdm(sorted(os.listdir(PATH_TO_DATA))):     # e.g. s01s03
    for dir2 in sorted(os.listdir(PATH_TO_DATA+dir1)):   # e.g. 04, means type4:pushing
        sub_cnt = 0
        for dir3 in sorted(os.listdir(PATH_TO_DATA+dir1+'/'+dir2)):  # e.g. 002
            dir_len.append(len(os.listdir(PATH_TO_DATA+dir1+'/'+dir2+'/'+dir3)))
            # print(dir1+'/'+dir2+'/'+dir3)
            video = []
            for depth_path in sorted(glob(PATH_TO_DATA+dir1+'/'+dir2+'/'+dir3+'/depth*')):
                img = plt.imread(depth_path)[:,:,0]
                x = xx[img>0]
                y = yy[img>0]
                z = img[img>0]
                x = (x-W/2)/focal*z
                y = (y-H/2)/focal*z
                point = np.stack([x,y,z], axis=-1)
                point = point[point[:,2]<0.75]
                point_num.append(point.shape[0])    # record the number of points in each frame
                video.append(point)                 # a video is a list of frames, each frame is a list of points
            videos.append((video,int(dir2)))        # type: list
            if(int(dir2) > num_class):
                num_class = int(dir2)

            if cnt < 5 and sub_cnt == 0:
                # videos_test.append((video, int(dir2)))
                txt_test.append(dir1+'/'+dir2+'/'+dir3+'\n')
                sequences_test.append(video)
                labels_test.append(int(dir2) - 1)

            else:
                # videos_train.append((video, int(dir2)))
                txt_train.append(dir1+'/'+dir2+'/'+dir3+'\n')
                sequences_train.append(video)
                labels_train.append(int(dir2) - 1)
            sub_cnt += 1
    cnt += 1

dir_len = np.array(dir_len)
print("max: ", np.max(dir_len))
print("min: ", np.min(dir_len))
print("mean: ", np.mean(dir_len))
print("median: ", np.median(dir_len))
point_num = np.array(point_num)
print("max: ", np.max(point_num))
print("min: ", np.min(point_num))
print("mean: ", np.mean(point_num))
print("median: ", np.median(point_num))

frame_per_clip = np.min(dir_len)
# num_points = np.median(point_num)
num_points = 4096
index = 0
new_label_test = []
new_label_train = []
new_sequence_test = []
new_sequence_train = []
clip_video_map = []
print('process test')
for i in tqdm(range(len(sequences_test))):
    label = labels_test[i]
    video = sequences_test[i]
    nframe = len(video)             # number of frames in a video
    for t in range(0, nframe - 1 * (frame_per_clip - 1)):           # frame interval is set as 1 by default
        clip = [video[t + i * 1] for i in range(frame_per_clip)]    # the clips of frame_per_clip frames
        for i, p in enumerate(clip):
            if p.shape[0] > num_points:                             # select some of the points randomly
                r = np.random.choice(p.shape[0], size = int(num_points), replace = False)
            else:                                                   # repeat the points if the number of points is not enough
                repeat, residue = num_points // p.shape[0], num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=int(residue), replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(int(repeat))] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        new_sequence_test.append(clip)
        clip_video_map.append(i)
        new_label_test.append(label)

print('process train')
for i in tqdm(range(len(sequences_train))):
    label = labels_train[i]
    video = sequences_train[i]
    nframe = len(video)             # number of frames in a video
    for t in range(0, nframe - 1 * (frame_per_clip - 1)):           # frame interval is set as 1 by default
        clip = [video[t + i * 1] for i in range(frame_per_clip)]    # the clips of frame_per_clip frames
        for i, p in enumerate(clip):
            if p.shape[0] > num_points:                             # select some of the points randomly
                r = np.random.choice(p.shape[0], size = int(num_points), replace = False)
            else:                                                   # repeat the points if the number of points is not enough
                repeat, residue = num_points // p.shape[0], num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=int(residue), replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(int(repeat))] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        new_sequence_train.append(clip)
        new_label_train.append(label)
        clip_video_map.append(i + len(sequences_test))

new_sequence_test = np.array(new_sequence_test)
new_sequence_train = np.array(new_sequence_train)
new_label_test = np.array(new_label_test)
new_label_train = np.array(new_label_train)

print('Start saving train')
np.savez('sbu_train.npz', sequences=new_sequence_train, labels=new_label_train, num_class=num_class, clip_video_map=clip_video_map)
print('Start saving test')
np.savez('sbu_test.npz', sequences=new_sequence_test, labels=new_label_test, num_class=num_class, clip_video_map=clip_video_map)