import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


def HorizontalFlip(video_imgs, p=0.5):
    # (T, C, H, W)
    if np.random.random() < p:
        video_imgs = np.flip(video_imgs, -1)
    return video_imgs


'''
class GRIDDataset(Dataset):
    def __init__(self, data, phase='train'):
        if isinstance(data, str):
            self.dataset = self.get_data_file(data)
        else:
            self.dataset = data
        print(len(self.dataset))
        self.phase = phase
        self.char_dict = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']    # 28
        self.max_vid_len = 75
        self.max_txt_len = 50

    # 得到所有speaker文件目录的list (一个包含多个不同speaker的文件夹)
    def get_data_file(self, root_path):
        # GRID\LIP_160x80\lip\s1
        dataset = []
        unseen_spk = ['s1', 's2', 's20', 's21', 's22']
        for spk in os.listdir(root_path):  # 根目录下的speaker目录
            if spk in unseen_spk:
                continue
            spk_path = os.path.join(root_path, spk)
            for fn in os.listdir(spk_path):  # 1000
                data_path = os.path.join(spk_path, fn)
                if len(os.listdir(data_path)) == 75:
                    dataset.append(data_path)
        return dataset

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        raw_txt = ' '.join(txt).upper()
        return np.asarray([self.char_dict.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        vid = self.load_video(item)
        if self.phase == 'train':
            vid = HorizontalFlip(vid, 0.5)
        txt_path = item.replace('lip', 'align_txt') + '.align'
        txt = self.load_txt(txt_path)
        vid_len = min(len(vid), self.max_vid_len)
        txt_len = min(len(txt), self.max_txt_len)
        vid = self.padding(vid, self.max_vid_len)
        txt = self.padding(txt, self.max_txt_len)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    txt=torch.LongTensor(txt),
                    vid_lens=torch.tensor(vid_len),
                    txt_lens=torch.tensor(txt_len))

    def __len__(self):
        return len(self.dataset)
'''


class Speaker(object):
    def __init__(self, data):
        # GRID\LIP_160x80\lip\s1\bbaf4p
        self.data = data
        '''
        self.data = []  # speaker目录下的视频目录
        for fn in os.listdir(dir_path):
            path = os.path.join(dir_path, fn.decode('utf-8'))
            if len(os.listdir(path)) > 0:
                self.data.append(path)
        '''
        self.char_dict = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']   # 28
        self.max_vid_len = 75
        self.max_txt_len = 50

    def sample_batch_data(self, bs):
        vids = []
        txts = []
        vid_lens = []
        txt_lens = []
        batch_paths = np.random.choice(self.data, size=bs, replace=False)  # 不重复地采样
        for path in batch_paths:
            vid = self.load_video(path)
            txt_path = path.replace('lip', 'align_txt') + '.align'
            txt = self.load_txt(txt_path)
            vid_lens.append(min(len(vid), self.max_vid_len))
            txt_lens.append(min(len(txt), self.max_txt_len))
            vids.append(self.padding(vid, self.max_vid_len))
            txts.append(self.padding(txt, self.max_txt_len))
        vids = np.stack(vids, axis=0)  # (B, T, C, H, W)
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (B, T, C, H, w)
                    txt=torch.LongTensor(txts),  # (B, L)
                    vid_lens=torch.tensor(vid_lens),  # (B, )
                    txt_lens=torch.tensor(txt_lens))  # (B, )

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        raw_txt = ' '.join(txt).upper()
        return np.asarray([self.char_dict.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])


class GRIDDataset(Dataset):
    def __init__(self, data_path, sample_size=2):
        self.sample_size = sample_size  # 每个speaker采样的样本数量
        self.file_list = self.get_file_list(data_path)

    # 得到所有speaker文件目录的list (一个包含多个不同speaker的文件夹)
    def get_file_list(self, data_path):
        # GRID\LIP_160x80\lip\s1
        unseen_spk = ['s1', 's2', 's20', 's21', 's22']
        return [os.path.join(data_path, fn) for fn in os.listdir(data_path) if fn not in unseen_spk]  # 根目录下的speaker目录

    # 返回一个speaker的数据
    def get_one_speaker_data(self, idx):  # one batch speaker data
        # GRID\LIP_160x80\lip\s1\bbaf4p
        speaker_data = []
        for fn in os.listdir(self.file_list[idx]):  # 1000
            data_path = os.path.join(self.file_list[idx], fn)
            if len(os.listdir(data_path)) == 75:
                speaker_data.append(data_path)
        speaker = Speaker(speaker_data)
        sampled_data = speaker.sample_batch_data(self.sample_size)
        return sampled_data

    def __getitem__(self, idx):
        return self.get_one_speaker_data(idx)

    def __len__(self):
        return len(self.file_list)
