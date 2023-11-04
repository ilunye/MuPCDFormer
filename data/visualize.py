import os
import matplotlib.pyplot as plt
import numpy as np

videos = np.load('sbu_test.npz')['sequences']
labels = np.load('sbu_test.npz')['labels']

figure = plt.figure()
ax = figure.add_subplot(projection='3d')
for i in range(len(videos)):
    for j in range(len(videos[i])):
        point = videos[i][j]
        ax.clear()
        ax.plot(point[:,0],point[:,1],point[:,2], 'o', markersize=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(azim=-90, elev=-90)
        ax.set_xlim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_ylim([-1,1])
        ax.axis('off')
        plt.savefig(f'figs/{i}_{j}.png')
    # save to mp4
    os.system(f'ffmpeg -r 30 -pattern_type glob -i "figs/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p visual/{i}.mp4')
    os.system('rm -f figs/*')