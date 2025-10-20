import os
import cv2
import numpy as np
import torch
import string
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import librosa
from data_augment import *
from torch.nn.utils.rnn import pad_sequence
from pypinyin import lazy_pinyin, Style

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
SIL = '<sil>'  # silent

# class BucketBatchSampler(Sampler):
#     '''
#         确保每个batch中的样本长度相似，减少padding数量，更高效地利用显存，提高训练效率
#     '''
#     def __init__(self, lens, batch_size, bucket_boundaries):
#         self.batch_size = batch_size
#         self.bucket_boundaries = bucket_boundaries
#         self.num_buckets = len(bucket_boundaries) + 1   # 桶数
#         self.bucket_sizes = [0] * self.num_buckets    # 每个桶的大小
#         self.bucket_indices = {i: [] for i in range(self.num_buckets)}
#
#         # 将数据分配到不同的桶中
#         for idx, length in enumerate(lens):
#             bucket_idx = self._get_bucket_index(length)
#             self.bucket_indices[bucket_idx].append(idx)
#             self.bucket_sizes[bucket_idx] += 1
#
#     def _get_bucket_index(self, length):
#         for i, boundary in enumerate(self.bucket_boundaries):
#             if length <= boundary:
#                 return i
#         return len(self.bucket_boundaries)
#
#     def __iter__(self):
#         # 打乱每个桶中的数据
#         for bucket_idx in range(self.num_buckets):
#             np.random.shuffle(self.bucket_indices[bucket_idx])
#         _batch = []
#         for bucket_idx in range(self.num_buckets):
#             for idx in self.bucket_indices[bucket_idx]:
#                 _batch.append(idx)
#                 if len(_batch) == self.batch_size:
#                     yield _batch
#                     _batch = []
#         if _batch:
#             yield _batch
#
#     def __len__(self):
#         return sum(self.bucket_sizes)


class BucketBatchSampler(Sampler):
    '''
        确保每个batch中的样本长度相似，减少padding数量，更高效地利用显存，提高训练效率
    '''
    def __init__(self, dataset, batch_size, bucket_boundaries):
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries
        self.num_buckets = len(bucket_boundaries) + 1   # 桶数
        self.bucket_sizes = [0] * self.num_buckets    # 每个桶的大小
        self.bucket_indices = {i: [] for i in range(self.num_buckets)}

        # 将数据分配到不同的桶中
        for idx, item in enumerate(dataset):
            length = len(item['vid'])   # 根据视频帧长进行分桶（这种取data长度的方式速度比较慢，最好直接传长度）
            bucket_idx = self._get_bucket_index(length)
            self.bucket_indices[bucket_idx].append(idx)
            self.bucket_sizes[bucket_idx] += 1

    def _get_bucket_index(self, length):
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)

    def __iter__(self):
        # 打乱每个桶中的数据
        for bucket_idx in range(self.num_buckets):
            np.random.shuffle(self.bucket_indices[bucket_idx])
        _batch = []
        for bucket_idx in range(self.num_buckets):
            for idx in self.bucket_indices[bucket_idx]:
                _batch.append(idx)
                if len(_batch) == self.batch_size:
                    yield _batch
                    _batch = []
        if _batch:
            yield _batch

    def __len__(self):
        return sum(self.bucket_sizes)


# class BucketBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, bucket_size, shuffle=True, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.bucket_size = bucket_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#
#     def create_batches(self):
#         indices = np.argsort([len(item['vid']) for item in self.dataset])   # 返回升序排序的索引
#         buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
#         if self.shuffle:
#             np.random.shuffle(buckets)
#         batches = []
#         for bucket in buckets:
#             if len(bucket) > self.batch_size:
#                 sub_batches = [bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
#                 if self.drop_last and len(sub_batches[-1]) < self.batch_size:
#                     sub_batches = sub_batches[:-1]
#                 batches.extend(sub_batches)
#             else:
#                 if not self.drop_last:
#                     batches.append(bucket)
#         if self.shuffle:
#             np.random.shuffle(batches)
#         return batches
#
#     def __iter__(self):
#         batches = self.create_batches()
#         for batch in batches:
#             yield batch
#
#     def __len__(self):
#         return len(self.create_batches())


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


class NoiseDataset(object):
    def __init__(self, noise_path):
        noise, sr = librosa.load(noise_path, sr=None)
        if sr != 16000:
            self.noise = librosa.resample(noise, orig_sr=sr, target_sr=16000)
        else:
            self.noise = noise
        self.snr_set = list(range(-10, 25, 5))
        print('Noise Data:', self.noise.shape, sr)

    # for testing
    def testing_noisy_signal(self, signal, snr_db=None):
        if snr_db is None:
            return normalize(signal)
        # snr_db = self.snr_set[snr_idx]
        # db to power
        SNR = 10. ** (snr_db / 10.)
        # power to ampl
        SNR = SNR ** 0.5

        # Matching Length
        if len(self.noise) < len(signal):
            noise = np.tile(self.noise, len(signal) // len(self.noise) + 1)[:len(signal)]
        else:
            noise = self.noise[:len(signal)]

        # Power
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        # Scale Noise
        target_noise = (1 / SNR) * noise * np.sqrt(signal_power / noise_power)
        # Add Noise
        corrupted_signal = signal + target_noise
        return normalize(corrupted_signal)

    # for training
    def training_noisy_signal(self, signal):
        snr_db = np.random.choice(range(-5, 25, 5))
        # snr_db = np.random.choice([-5, 0, 5, 10, 15, 20, None])
        # if snr_db is None:
        #     return normalize(signal)

        # db to power
        SNR = 10. ** (snr_db / 10.)
        # power to ampl
        SNR = SNR ** 0.5

        if np.random.rand() < 0.5:
            if len(self.noise) < len(signal):
                noise = np.tile(self.noise.copy(), len(signal) // len(self.noise) + 1)
            else:
                noise = self.noise.copy()
            # Randomly Select Noise Clip
            pos = np.random.randint(0, len(noise) - len(signal) + 1)
            noise = noise[pos: pos + len(signal)]
        else:
            noise = np.random.randn(len(signal))

        # Power
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        # Scale Noise
        target_noise = (1 / SNR) * noise * np.sqrt(signal_power / noise_power)
        # Add Noise
        corrupted_signal = signal + target_noise
        return normalize(corrupted_signal)


def normalize(x, norm='peak_norm'):
    if norm == 'z_score':
        mean, std = np.mean(x), np.std(x)
        if std == 0: std = 1.
        return (x - mean) / std
    elif norm == 'peak_norm':
        peak = np.max(np.abs(x))
        return x / peak if peak > 1. else x
    elif norm == 'max_min':
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif norm == 'log_norm':
        return np.log1p(x)
    elif norm is None:
        return x
    else:
        raise ValueError('Unknown Normalization!!')



'''
每次采样n个speaker，每个speaker采样2个不同样本，形成相同说话人不同内容的样本对
(注意不同说话人说相同内容实际并不常见，多见于实验室采集)
训练过程中，每个speaker的样本尽可能都能用到1个batch中的数据来自不同说话人
'''
class GRIDDataset(Dataset):
    def __init__(self, root_path, data_path, sample_size=2, phase='train', setting='unseen'):
        self.sample_size = sample_size  # 每个speaker采的样本数
        self.root_path = root_path
        self.phase = phase
        #self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # 28
        with open('word_vocab.txt', 'r', encoding='utf-8') as fin:
            vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab  # 52
        self.vocab = [PAD] + vocab + [EOS, BOS]  # 54

        self.max_vid_len = 75
        self.max_aud_len = 300
        self.max_txt_len = 30

        with open(data_path, 'r', encoding='utf-8') as fr:
            self.spk_dict = json.load(fr)
            # self.spks = list(self.spk_dict.keys())

        # totally 34 and #21 is missing
        self.spks = [f's{i}' for i in range(1, 35) if (setting == 'seen' and i != 21)
                     or (setting == 'unseen' and i not in [1, 2, 20, 21, 22])]

        if self.phase == 'drl_train':
            self.data = self.spks
        else:
            self.data = []
            for spk_id in self.spk_dict.keys():
                #self.data.extend([os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]])
                self.data.extend([(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]])
        print(len(self.data), len(self.spks))

        self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav')

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

    def load_audio(self, fn, sr=16000):
        # y, sr = librosa.load(fn, sr=sr)  # 16kHz
        y, sr0 = librosa.load(fn, sr=None)
        if sr0 != sr:
            y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

        # 以0.25的概率随机叠加噪声波形
        if self.phase == 'train' and np.random.rand() < 0.25:
            y = self.noise_generator.training_noisy_signal(y)
        else:
            y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB

        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  #转换到对数刻度
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=800, hop_length=160, n_mels=80)  #转换到对数刻度
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)  #转换到对数刻度
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        #return logmelspec.T  # (T, n_mels)
        log_mel = logmelspec.T   # (T, n_mels)
        norm_log_mel = np.nan_to_num((log_mel - log_mel.mean(0, keepdims=True)) / (log_mel.std(0, keepdims=True)+1e-8))  # z-norm
        return norm_log_mel

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        #return np.asarray([self.vocab.index(c) for c in ' '.join(txt).upper()])
        #return np.asarray([self.vocab.index(w.upper()) for w in txt])
        return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w.upper()) for w in txt] + [self.vocab.index(EOS)])

    def data_augment(self, vid, aud):
        if self.phase == 'train' and np.random.rand() < 0.5:
            vid = horizontal_flip(vid)
            #aud = spec_augment(aud, time_first=True)
            return vid, aud
        return vid, aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    def fetch_data(self, vid_path, aud_path, align_path):
        vid = self.load_video(vid_path)
        aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        # data augmentation
        vid, aud = self.data_augment(vid, aud)

        vid_len = min(len(vid), self.max_vid_len)
        aud_len = min(len(aud), self.max_aud_len)
        txt_len = min(len(txt), self.max_txt_len) - 1  # excluding bos 
        vid = self.padding(vid, self.max_vid_len)
        aud = self.padding(aud, self.max_aud_len)
        txt = self.padding(txt, self.max_txt_len)
        return vid, aud, txt, vid_len, aud_len, txt_len

    def get_one_data(self, idx):
        #vid_path = self.data[idx]
        #spk = vid_path.split(os.path.sep)[-2]
        root_path, spk, dir_path = self.data[idx]
        spk_id = self.spks.index(spk) if spk in self.spks else -1
        vid_path = os.path.join(root_path, spk, dir_path)
        txt_path = os.path.join(root_path.replace('lip', 'align_txt'), spk, dir_path+'.align')
        aud_path = os.path.join(root_path.replace('lip', 'audio'), spk, dir_path+'.wav')
        #txt_path = vid_path.replace('lip', 'align_txt') + '.align'
        #aud_path = vid_path.replace('lip', 'audio') + '.wav'
        vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    aud=torch.FloatTensor(aud),
                    txt=torch.LongTensor(txt),
                    spk_id=spk_id,
                    vid_lens=torch.tensor(vid_len),
                    aud_lens=torch.tensor(aud_len),
                    txt_lens=torch.tensor(txt_len))

    # 返回一个speaker的数据
    def get_one_speaker(self, idx):  # one batch speaker data
        vids = []
        auds = []
        txts = []
        vid_lens = []
        aud_lens = []
        txt_lens = []
        # GRID\LIP_160x80\lip\s1
        spk_id = self.data[idx]
        # GRID\LIP_160x80\lip\s1\bbaf4p
        spk_data = [os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]]
        batch_data = np.random.choice(spk_data, size=self.sample_size, replace=False)  # 不重复采样
        for vid_path in batch_data:
            txt_path = vid_path.replace('lip', 'align_txt') + '.align'
            aud_path = vid_path.replace('lip', 'audio') + '.wav'
            vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
            vids.append(vid)
            auds.append(aud)
            txts.append(txt)
            vid_lens.append(vid_len)
            aud_lens.append(aud_len)
            txt_lens.append(txt_len)
        vids = np.stack(vids, axis=0)  # (N, T, C, H, W)
        auds = np.stack(auds, axis=0)  
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (N, T, C, H, w)
                    aud=torch.FloatTensor(auds),
                    txt=torch.LongTensor(txts),  # (N, L)
                    vid_lens=torch.tensor(vid_lens),  # (N, )
                    aud_lens=torch.tensor(aud_lens),
                    txt_lens=torch.tensor(txt_lens))  # (N, )

    def __getitem__(self, idx):
        if self.phase == 'drl_train':
            return self.get_one_speaker(idx)
        else:
            return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    @classmethod
    def collate_pad(cls, batch):
        return torch.utils.data.dataloader.default_collate(batch)



class CMLRDataset(Dataset):
    # 类变量
    MAX_VID_LEN = 200
    MAX_AUD_LEN = 200
    #MAX_TXT_LEN = 35

    def __init__(self, root_path, file_list, phase='train', setting='unseen'):
        self.root_path = root_path
        self.phase = phase
        self.spks = [f's{i}' for i in range(1, 12) if setting == 'seen' or (setting == 'unseen' and i not in [2, 6])]
        self.data = []
        with open(file_list, 'r', encoding='utf-8') as f:
            for line in f:  # s5/20151009_section_3_030.36_032.65
                spk_id, section = line.strip().split('/')
                date, sec_id = section.split('_', 1)   # 20151009  section_3_030.36_032.65
                self.data.append((spk_id, date, sec_id))
        print(len(self.data), len(self.spks))

        #with open('zh_vocab.txt', 'r', encoding='utf-8') as fin:
        with open('pinyin.txt', 'r', encoding='utf-8') as fin:
            vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab
        self.vocab = [PAD] + vocab + [EOS, BOS]
        print('vocab size:', len(self.vocab))

        self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav')

    '''
    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        # array = [cv2.resize(img, (128, 64)) for img in array if img.shape[:2] != (64, 128) else img]  # W, H
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.
        #return (array - 127.5) / 128
    '''

    def load_video(self, fn):
        vid = np.load(fn)   # (T, H, W)
        array = vid[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_audio(self, fn, sr=16000):
        # y, sr = librosa.load(fn, sr=sr)  # 16kHz
        y, sr0 = librosa.load(fn, sr=None)
        if sr0 != sr: y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

        if self.phase == 'train' and np.random.rand() < 0.25:
            y = self.noise_generator.training_noisy_signal(y)
        else:
            y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB

        # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  # 25ms win / 10ms hop for grid
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)   # 128ms win / 32ms hop for cmlr
        # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)   # 64ms win / 16ms hop for cmlr
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=800, hop_length=160, n_mels=80)
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        #return logmelspec.T  # (T, n_mels)
        log_mel = logmelspec.T   # (T, n_mels)
        norm_log_mel = np.nan_to_num((log_mel - log_mel.mean(0, keepdims=True)) / (log_mel.std(0, keepdims=True)+1e-8))  # z-norm
        return norm_log_mel

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = f.readline().strip()   # 读第一行
        #return np.asarray([self.vocab.index(w) for w in txt])
        # return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w) for w in txt] + [self.vocab.index(EOS)])
        #return np.asarray(list(map(self.vocab.index, [BOS]+list(txt)+[EOS])))
        return np.asarray(list(map(self.vocab.index, [BOS]+lazy_pinyin(txt, style=Style.TONE3)+[EOS])))

    def data_augment(self, vid, aud):
        if self.phase == 'train':
            vid = vid_rand_crop(vid, 88, 88)
            if np.random.rand() < 0.5:
                vid = horizontal_flip(vid)
                #aud = spec_augment(aud, time_first=True)
        else:
            vid = vid_center_crop(vid, 88, 88)
        return vid, aud
        
        #if self.phase == 'train' and np.random.rand() < 0.5:
        #    vid = horizontal_flip(vid)
        #    #aud = spec_augment(aud, time_first=True)
        #    return vid, aud
        #return vid, aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    # def fetch_data(self, vid_path, aud_path, align_path):
    #     vid = self.load_video(vid_path)
    #     aud = self.load_audio(aud_path)
    #     txt = self.load_txt(align_path)
    #     # data augmentation
    #     vid, aud = self.data_augment(vid, aud)
    #     #print(vid.shape, aud.shape, len(txt), flush=True)
    #     vid_len = min(len(vid), self.MAX_VID_LEN)
    #     aud_len = min(len(aud), self.MAX_AUD_LEN)
    #     txt_len = min(len(txt), self.MAX_TXT_LEN) - 1  # excluding bos 
    #     vid = self.padding(vid, self.MAX_VID_LEN)
    #     aud = self.padding(aud, self.MAX_AUD_LEN)
    #     txt = self.padding(txt, self.MAX_TXT_LEN)
    #     return vid, aud, txt, vid_len, aud_len, txt_len

    def fetch_data(self, vid_path, aud_path, align_path):
        vid = self.load_video(vid_path)
        aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        # data augmentation
        vid, aud = self.data_augment(vid, aud)
        # print(vid.shape, aud.shape, len(txt), flush=True)
        return vid, aud, txt

    def get_one_data(self, idx):
        item = self.data[idx]
        data_path = os.path.join(*item)
        spk_id = self.spks.index(item[0]) if item[0] in self.spks else -1
        vid_path = os.path.join(self.root_path, 'video_96_96', data_path+'.npy')
        txt_path = os.path.join(self.root_path, 'text', data_path+'.txt')
        aud_path = os.path.join(self.root_path, 'audio', data_path+'.wav')
        # vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        vid, aud, txt = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    aud=torch.FloatTensor(aud),  # (T, D)
                    txt=torch.LongTensor(txt),   # (L, )
                    spk_id=spk_id,
                    # vid_lens=torch.tensor(vid_len),
                    # aud_lens=torch.tensor(aud_len),
                    # txt_lens=torch.tensor(txt_len)
                    )

    def __getitem__(self, idx):
        return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    # 按照batch中最长序列的长度进行padding，返回对齐后的序列和实际序列长度
    @classmethod
    def collate_pad(cls, batch):
        padded_batch = {}
        for data_type in batch[0].keys():
            if data_type == 'spk_id':
                padded_batch[data_type] = torch.tensor([s[data_type] for s in batch])
            else:
                if data_type == 'vid':
                    max_len = cls.MAX_VID_LEN
                elif data_type == 'aud':
                    max_len = cls.MAX_AUD_LEN
                elif data_type == 'txt':
                    max_len = None
                else:
                    max_len = None
                pad_vid, ret_lens = pad_seqs3([s[data_type] for s in batch if s[data_type] is not None], max_len)
                padded_batch[data_type] = pad_vid
                if data_type == 'txt':
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens) - 1  # excluding bos 
                else:
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens)
        return padded_batch

    # @classmethod
    # def collate_pad(cls, batch):
    #     return torch.utils.data.dataloader.default_collate(batch)



class VSRAppDataset(Dataset):
    # 类变量
    MAX_VID_LEN = 200
    MAX_AUD_LEN = 200
    #MAX_TXT_LEN = 35

    def __init__(self, root_path, file_list, phase='train', setting='unseen'):
        self.root_path = root_path
        self.phase = phase
        self.data = []
        self.spks = [f's{i}' for i in range(1, 12) if setting == 'seen' or (setting == 'unseen' and i not in [2, 6])]
        with open(file_list, 'r', encoding='utf-8') as f:
            for line in f:  # wyl/command_1.npy
                spk_id, cmd = line.strip().split('/')
                self.data.append((spk_id, cmd))
        print(len(self.data))

        with open('data/cmd.json', 'r') as fin:
        #with open('data/unseen_cmd.json', 'r') as fin:
            self.cmd_set = json.load(fin)
        self.cmd_set['command_100'] = SIL
        #print(self.cmd_set)

        #with open('zh_vocab.txt', 'r', encoding='utf-8') as fin:
        with open('pinyin.txt', 'r', encoding='utf-8') as fin:
            vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab
        self.vocab = [PAD] + vocab + [EOS, BOS]
        print('vocab size:', len(self.vocab))

        self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav')

    '''
    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        # array = [cv2.resize(img, (128, 64)) for img in array if img.shape[:2] != (64, 128) else img]  # W, H
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.
        #return (array - 127.5) / 128
    '''

    def load_video(self, fn):
        vid = np.load(fn)   # (T, H, W)
        array = vid[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_audio_0(self, fn, sr=16000):
        #y, sr0 = librosa.load(fn, sr=None)
        #if sr0 != sr: y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
        y, sr = librosa.load(fn, sr=sr)  # 16kHz

        #if self.phase == 'train' and np.random.rand() < 0.25:
        #    y = self.noise_generator.training_noisy_signal(y)
        #else:
        #    y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB

        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  # 25ms win / 10ms hop for grid
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)   # 128ms win / 32ms hop for cmlr
        # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)   # 64ms win / 16ms hop for cmlr
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=800, hop_length=160, n_mels=80)
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        #return logmelspec.T  # (T, n_mels)
        log_mel = logmelspec.T   # (T, n_mels)
        norm_log_mel = (log_mel - log_mel.mean(0, keepdims=True)) / (log_mel.std(0, keepdims=True) + 1e-8)  # z-norm
        return norm_log_mel
    
    def load_audio(self, fn, sr=16000):
        def stacker(feats, stack_order=4, trunct=False):
            """
            Concatenating consecutive audio frames
            Args:
                feats - numpy.ndarray of shape [T, F]
                stack_order - int (number of neighboring frames to concatenate
            Returns:
                feats - numpy.ndarray of shape [T', F']  = [T/stack_order, F*stack_order]
            """
            feat_dim = feats.shape[1]
            if trunct:
                return feats[:stack_order*(feats.shape[0]//stack_order)].reshape((-1, stack_order*feat_dim))
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        
        y, sr = librosa.load(fn, sr=sr)  # 16kHz

        #if self.phase == 'train' and np.random.rand() < 0.25:
        #    y = self.noise_generator.training_noisy_signal(y)
        #else:
        #    y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB
        y = normalize(y)

        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  # 25ms win / 10ms hop for lrs3
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=640, hop_length=160, n_mels=80)  # 40ms win / 10ms hop for lrs3
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=26)  # 25ms win / 10ms hop for avhubert
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=450, hop_length=266, n_mels=52)  # align video of 30fps 
        log_mel = librosa.power_to_db(melspec, ref=np.max)  # (F, T)
        #return log_mel.T  # (T, n_mels)
        log_mel = (log_mel - log_mel.mean(1, keepdims=True)) / (log_mel.std(1, keepdims=True) + 1e-8)  # z-norm along time
        #return stacker(log_mel.T, 4)  # only for avhubert: 26 * 4
        return stacker(log_mel.T, 2)  # only for avhubert: 52 * 2
        #return log_mel.T

    def load_txt(self, fn):
        #with open(fn, 'r', encoding='utf-8') as f:
        #    txt = f.readline().strip()   # 读第一行
        #return np.asarray([self.vocab.index(w) for w in txt])
        # return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w) for w in txt] + [self.vocab.index(EOS)])
        #return np.asarray(list(map(self.vocab.index, [BOS]+list(txt)+[EOS])))
        txt = self.cmd_set[fn].strip()
        if txt == SIL:
            txt = []
        else:
            txt = lazy_pinyin(txt, style=Style.TONE3)
            txt = ['men1' if x == 'men4' else x for x in txt]
        #print(txt, flush=True)
        return np.asarray(list(map(self.vocab.index, [BOS]+txt+[EOS])))

    def data_augment(self, vid, aud):
        if self.phase == 'train':
            #vid = vid_rand_crop(vid, 88, 88)
            if np.random.rand() < 0.5:
                vid = horizontal_flip(vid)
                #aud = spec_augment(aud, time_first=True)
        else:
            #vid = vid_center_crop(vid, 88, 88)
            pass
        return vid, aud
        
        #if self.phase == 'train' and np.random.rand() < 0.5:
        #    vid = horizontal_flip(vid)
        #    #aud = spec_augment(aud, time_first=True)
        #    return vid, aud
        #return vid, aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    # def fetch_data(self, vid_path, aud_path, align_path):
    #     vid = self.load_video(vid_path)
    #     aud = self.load_audio(aud_path)
    #     txt = self.load_txt(align_path)
    #     # data augmentation
    #     vid, aud = self.data_augment(vid, aud)
    #     #print(vid.shape, aud.shape, len(txt), flush=True)
    #     vid_len = min(len(vid), self.MAX_VID_LEN)
    #     aud_len = min(len(aud), self.MAX_AUD_LEN)
    #     txt_len = min(len(txt), self.MAX_TXT_LEN) - 1  # excluding bos 
    #     vid = self.padding(vid, self.MAX_VID_LEN)
    #     aud = self.padding(aud, self.MAX_AUD_LEN)
    #     txt = self.padding(txt, self.MAX_TXT_LEN)
    #     return vid, aud, txt, vid_len, aud_len, txt_len

    def fetch_data(self, vid_path, aud_path, align_path):
        vid = self.load_video(vid_path)
        aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        # data augmentation
        vid, aud = self.data_augment(vid, aud)
        # print(vid.shape, aud.shape, len(txt), flush=True)
        return vid, aud, txt

    def get_one_data(self, idx):
        item = self.data[idx]
        data_path = os.path.join(*item)
        vid_path = os.path.join(self.root_path, data_path+'.npy')
        aud_path = os.path.join(self.root_path, data_path+'.wav')
        txt_path = item[1]   # 'command_xxx'
        # vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        vid, aud, txt = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    aud=torch.FloatTensor(aud),  # (T, D)
                    txt=torch.LongTensor(txt),   # (L, )
                    # vid_lens=torch.tensor(vid_len),
                    # aud_lens=torch.tensor(aud_len),
                    # txt_lens=torch.tensor(txt_len)
                    )

    def __getitem__(self, idx):
        return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    # 按照batch中最长序列的长度进行padding，返回对齐后的序列和实际序列长度
    @classmethod
    def collate_pad(cls, batch):
        padded_batch = {}
        for data_type in batch[0].keys():
            if data_type == 'spk_id':
                padded_batch[data_type] = torch.tensor([s[data_type] for s in batch])
            else:
                if data_type == 'vid':
                    max_len = cls.MAX_VID_LEN
                elif data_type == 'aud':
                    max_len = cls.MAX_AUD_LEN
                elif data_type == 'txt':
                    max_len = None
                else:
                    max_len = None
                pad_vid, ret_lens = pad_seqs3([s[data_type] for s in batch if s[data_type] is not None], max_len)
                padded_batch[data_type] = pad_vid
                if data_type == 'txt':
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens) - 1  # excluding bos 
                else:
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens)
        return padded_batch

    # @classmethod
    # def collate_pad(cls, batch):
    #     return torch.utils.data.dataloader.default_collate(batch)



class LRS3Dataset(Dataset):   # 说话人数量多，但每个说话人的样本少
    # 类变量
    MAX_VID_LEN = 155
    #MAX_AUD_LEN = 620
    MAX_AUD_LEN = 155   # avhubert
    #MAX_TXT_LEN = 150

    def __init__(self, root_path, file_list, phase='train', setting='unseen'):
        self.root_path = root_path
        self.phase = phase
        self.data = []
        self.spks = list(range(4004))
        with open(file_list, 'r', encoding='utf-8') as f:
            for line in f:  # 1 test/stngBN4hp14/00001.npy
                spk_id, fn = line.strip().split(' ')
                self.data.append((int(spk_id), fn))
        self.vocab = [PAD] + list(" " + string.ascii_lowercase + string.digits + "'") + [EOS, BOS]
        print(len(self.data), len(self.spks), len(self.vocab))

        self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav')

    def load_video(self, fn):
        vid = np.load(fn)   # (T, H, W)
        array = vid[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_audio(self, fn, sr=16000):
        def stacker(feats, stack_order=4, trunct=False):
            """
            Concatenating consecutive audio frames
            Args:
                feats - numpy.ndarray of shape [T, F]
                stack_order - int (number of neighboring frames to concatenate
            Returns:
                feats - numpy.ndarray of shape [T', F']  = [T/stack_order, F*stack_order]
            """
            feat_dim = feats.shape[1]
            if trunct:
                return feats[:stack_order*(feats.shape[0]//stack_order)].reshape((-1, stack_order*feat_dim))
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        
        # y, sr = librosa.load(fn, sr=sr)  # 16kHz
        y, sr0 = librosa.load(fn, sr=None)
        if sr0 != sr: y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

        if self.phase == 'train' and np.random.rand() < 0.25:
            y = self.noise_generator.training_noisy_signal(y)
        else:
            y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB

        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=320, hop_length=160, n_mels=80)  # 20ms win / 10ms hop for lrs3
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  # 25ms win / 10ms hop for lrs3
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=640, hop_length=160, n_mels=80)  # 40ms win / 10ms hop for lrs3
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=26)  # 25ms win / 10ms hop for avhubert
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        #return logmelspec.T  # (T, n_mels)
        log_mel = logmelspec.T   # (T, n_mels)
        log_mel = stacker(log_mel, 4)  # only for avhubert: 26 * 4
        norm_log_mel = np.nan_to_num((log_mel - log_mel.mean(0, keepdims=True)) / (log_mel.std(0, keepdims=True)+1e-8))  # z-norm along time
        #norm_log_mel = np.nan_to_num((log_mel - log_mel.mean(1, keepdims=True)) / (log_mel.std(1, keepdims=True)+1e-8))  # z-norm along freq
        return norm_log_mel

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = f.readline().strip()   # 读第一行
        txt = txt.replace("Text:", "").replace("{LG}", "").replace("{NS}", "").strip().lower()
        #return np.asarray([self.vocab.index(w) for w in txt])
        # return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w) for w in txt] + [self.vocab.index(EOS)])
        return np.asarray(list(map(self.vocab.index, [BOS]+list(txt)+[EOS])))

    def data_augment(self, vid, aud):
        if self.phase == 'train':
            vid = vid_rand_crop(vid, 88, 88)
            if np.random.rand() < 0.5:
                vid = horizontal_flip(vid)
                #aud = spec_augment(aud, time_first=True)
        else:
            vid = vid_center_crop(vid, 88, 88)
        return vid, aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    # def fetch_data(self, vid_path, aud_path, align_path):
    #     vid = self.load_video(vid_path)
    #     aud = self.load_audio(aud_path)
    #     txt = self.load_txt(align_path)
    #     # data augmentation
    #     vid, aud = self.data_augment(vid, aud)
    #     #print(vid.shape, aud.shape, len(txt), flush=True)
    #     vid_len = min(len(vid), self.MAX_VID_LEN)
    #     aud_len = min(len(aud), self.MAX_AUD_LEN)
    #     txt_len = min(len(txt), self.MAX_TXT_LEN) - 1  # excluding bos 
    #     vid = self.padding(vid, self.MAX_VID_LEN)
    #     aud = self.padding(aud, self.MAX_AUD_LEN)
    #     txt = self.padding(txt, self.MAX_TXT_LEN)
    #     return vid, aud, txt, vid_len, aud_len, txt_len

    def fetch_data(self, vid_path, aud_path, align_path):
        vid = self.load_video(vid_path)
        aud = self.load_audio(aud_path)
        # 对齐帧长
        diff = len(aud) - len(vid)
        if diff < 0:
            aud = np.concatenate([aud, np.zeros((-diff, aud.shape[-1]), dtype=aud.dtype)])
        elif diff > 0:
            aud = aud[:-diff]

        txt = self.load_txt(align_path)
        # data augmentation
        vid, aud = self.data_augment(vid, aud)
        # print(vid.shape, aud.shape, len(txt), flush=True)
        return vid, aud, txt

    def get_one_data(self, idx):
        spk_id, data_path = self.data[idx]
        vid_path = os.path.join(self.root_path, data_path)
        aud_path = os.path.join(self.root_path, data_path.replace('.npy', '.wav'))
        txt_path = os.path.join(self.root_path, data_path.replace('.npy', '.txt'))
        # vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        vid, aud, txt = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    aud=torch.FloatTensor(aud),  # (T, D)
                    txt=torch.LongTensor(txt),   # (L, )
                    spk_id=spk_id,
                    # vid_lens=torch.tensor(vid_len),
                    # aud_lens=torch.tensor(aud_len),
                    # txt_lens=torch.tensor(txt_len)
                    )

    def __getitem__(self, idx):
        return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    # 按照batch中最长序列的长度进行padding，返回对齐后的序列和实际序列长度
    @classmethod
    def collate_pad(cls, batch):
        padded_batch = {}
        for data_type in batch[0].keys():
            if data_type == 'spk_id':
                padded_batch[data_type] = torch.tensor([s[data_type] for s in batch])
            else:
                if data_type == 'vid':
                    max_len = cls.MAX_VID_LEN
                elif data_type == 'aud':
                    max_len = cls.MAX_AUD_LEN
                elif data_type == 'txt':
                    max_len = None
                else:
                    max_len = None
                pad_vid, ret_lens = pad_seqs3([s[data_type] for s in batch if s[data_type] is not None], max_len)
                padded_batch[data_type] = pad_vid
                if data_type == 'txt':
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens) - 1  # excluding bos 
                else:
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens)
        return padded_batch

    # @classmethod
    # def collate_pad(cls, batch):
    #     return torch.utils.data.dataloader.default_collate(batch)
    

def pad_seqs(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = samples[0].new_full((len(samples), max_len, ) + samples[0].shape[1:], pad_val)
    for i, s in enumerate(samples):
        if len(s) < max_len:
            padded_batch[i][:len(s)] = s
        else:
            padded_batch[i] = s[:max_len]
    return padded_batch, lens


def pad_seqs2(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = []
    for seq in samples:
        if len(seq) < max_len:
            padding = seq.new_full((max_len-len(seq), ) + seq.shape[1:], pad_val)
            padded_seq = torch.cat((seq, padding), dim=0)
        else:
            padded_seq = seq[:max_len]
        padded_batch.append(padded_seq)
    return torch.stack(padded_batch), lens


def pad_seqs3(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = pad_sequence(samples, batch_first=True, padding_value=pad_val)  # (B, L_max, ...)
    return padded_batch[:, :min(max_len, padded_batch.shape[1])], lens
