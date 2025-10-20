import librosa
import soundfile as sf
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def resample(wav_path, save_path, sr=16000):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    y, sr = librosa.load(wav_path, sr=sr)
    sf.write(save_path, y, sr)


class MyDataset(Dataset):
    def __init__(self, root_dir):
        # D:\LipData\CMLR\audio\s1\20170121\section_6_001.69_005.57 + .wav
        # D:\LipData\CMLR\audio_sampled\s1\20170121\section_6_001.69_005.57 + .wav
        self.wave_paths = glob.glob(os.path.join(root_dir, 's*', '*', '*.wav'))
        self.save_paths = [p.replace('audio', 'audio_sampled') for p in self.wave_paths]
        print(len(self.wave_paths), len(self.save_paths), flush=True)

    def __getitem__(self, idx):
        resample(self.wave_paths[idx], self.save_paths[idx])
        return 0

    def __len__(self):
        return len(self.wave_paths)


def run():
    root_dir = r'D:\LipData\CMLR\audio'
    loader = DataLoader(MyDataset(root_dir),
                        batch_size=128,
                        num_workers=16,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=True
                    )
    for _ in tqdm(loader):
        pass
    print('Done!!!')


if __name__ == '__main__':
    print('running ...')
    run()

