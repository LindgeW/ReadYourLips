import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json


PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


def HorizontalFlip(video_imgs, p=0.5):
    # (T, C, H, W)
    if np.random.random() < p:
        video_imgs = np.flip(video_imgs, -1).copy()
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
        self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
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
        return np.asarray([self.vocab.index(c) for c in raw_txt])

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
        self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']   # 28
        self.max_vid_len = 75
        self.max_txt_len = 50

    def sample_batch_data(self, bs):
        vids = []
        txts = []
        vid_lens = []
        txt_lens = []
        batch_paths = np.random.choice(self.data, size=bs, replace=False)  # 不重复采样
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
        return np.asarray([self.vocab.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

'''
要求：
每次采样n个speaker，每个speaker采样2个不同样本，形成相同说话人不同内容的样本对
(注意不同说话人说相同内容实际并不常见，多见于实验室采集)
训练过程中，每个speaker的样本尽可能都能用到
1个batch中的数据来自不同说话人
'''
class GRIDDataset(Dataset):
    def __init__(self, root_path, data_path, sample_size=2, phase='train'):
        self.sample_size = sample_size  # 每个speaker采的样本数
        self.root_path = root_path
        self.phase = phase
        #self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # 28
        with open('word_vocab.txt', 'r', encoding='utf-8') as fin:
            vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab  # 52
        self.vocab = [PAD] + vocab + [EOS, BOS]  # 54

        self.max_vid_len = 75
        self.max_txt_len = 30

        with open(data_path, 'r', encoding='utf-8') as fr:
            self.spk_dict = json.load(fr)
            self.spks = list(self.spk_dict.keys())

        if self.phase == 'drl_train':
            self.data = self.spks
        else:
            self.data = []
            for spk_id in self.spks:
                self.data.extend([os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]])
        print(len(self.data), len(self.spks))

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]  # W, H
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.
        #return (array - 127.5) / 128


    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        #return np.asarray([self.vocab.index(c) for c in ' '.join(txt).upper()])
        #return np.asarray([self.vocab.index(w.upper()) for w in txt])
        return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w.upper()) for w in txt] + [self.vocab.index(EOS)])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    def fetch_data(self, vid_path, align_path):
        vid = self.load_video(vid_path)
        txt = self.load_txt(align_path)
        if self.phase != 'test':
            vid = HorizontalFlip(vid, 0.5)
        vid_len = min(len(vid), self.max_vid_len)
        txt_len = min(len(txt), self.max_txt_len) - 2  # excluding bos and eos
        vid = self.padding(vid, self.max_vid_len)
        txt = self.padding(txt, self.max_txt_len)
        return vid, txt, vid_len, txt_len

    def get_one_data(self, idx):
        vid_path = self.data[idx]
        spk_id = self.spks.index(vid_path.split(os.path.sep)[-2])
        txt_path = vid_path.replace('lip', 'align_txt') + '.align'
        vid, txt, vid_len, txt_len = self.fetch_data(vid_path, txt_path)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    txt=torch.LongTensor(txt),
                    spk_id=spk_id,
                    vid_lens=torch.tensor(vid_len),
                    txt_lens=torch.tensor(txt_len))

    # 返回一个speaker的数据
    def get_one_speaker(self, idx):  # one batch speaker data
        vids = []
        txts = []
        vid_lens = []
        txt_lens = []
        # GRID\LIP_160x80\lip\s1
        spk_id = self.data[idx]
        # GRID\LIP_160x80\lip\s1\bbaf4p
        spk_data = [os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]]
        batch_data = np.random.choice(spk_data, size=self.sample_size, replace=False)  # 不重复采样
        for vid_path in batch_data:
            txt_path = vid_path.replace('lip', 'align_txt') + '.align'
            vid, txt, vid_len, txt_len = self.fetch_data(vid_path, txt_path)
            vids.append(vid)
            txts.append(txt)
            vid_lens.append(vid_len)
            txt_lens.append(txt_len)
        vids = np.stack(vids, axis=0)  # (N, T, C, H, W)
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (N, T, C, H, w)
                    txt=torch.LongTensor(txts),  # (N, L)
                    vid_lens=torch.tensor(vid_lens),  # (N, )
                    txt_lens=torch.tensor(txt_lens))  # (N, )

    def __getitem__(self, idx):
        if self.phase == 'drl_train':
            return self.get_one_speaker(idx)
        else:
            return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)
