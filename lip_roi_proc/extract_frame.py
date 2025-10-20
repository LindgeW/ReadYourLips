import os
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import time
import glob
import subprocess


class MyDataset(Dataset):
    def __init__(self):
        self.IN = 'video-high/'
        self.OUT = 'faces-small/'
        self.files = glob.glob(os.path.join(self.IN, 's*', '*.mpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]   # video-high/s1/bbcnzze.mpg
        _, ext = os.path.splitext(file)
        dst = file.replace(self.IN, self.OUT).replace(ext, '')
        if not os.path.exists(dst):
            os.makedirs(dst)

            #cmd = 'ffmpeg -i {} -qscale:v 2 -r 25 {}/%d.jpg'.format(file, dst)   # 720x576
            cmd = 'ffmpeg -i {} -qscale:v 2 -r 25 -s 360x288 {}/%d.jpg'.format(file, dst)
            os.system(cmd)
            # subprocess.run(['ffmpeg', '-i', file, '-qscale:v', 2, '-r', 25, '-s', '128x64', dst+'/%d.jpg'])
        return dst


if __name__ == '__main__':
    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=32, batch_size=128, shuffle=False, drop_last=False)
    tic = time.time()
    for (i, batch) in enumerate(loader):
        eta = (1.0 * time.time() - tic) / (i + 1) * (len(loader) - i)
        print('eta:{}'.format(eta / 3600.0))
