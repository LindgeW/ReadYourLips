import numpy as np

def horizontal_flip(vid_imgs):
    # (T, C, H, W)
    return np.flip(vid_imgs, -1).copy()


def vid_rand_crop(vid, crop_height, crop_width):
    """
    对输入视频帧序列进行随机裁剪。
    参数:
        vid (numpy.ndarray): 形状为:
              (T, H, W) 或 (T, H, W, C)。
              (T, H, W) 或 (T, C, H, W)。
        crop_height (int): 裁剪的高度。
        crop_width (int): 裁剪的宽度。
    返回:
        numpy.ndarray: 随机裁剪后的序列。
    """
    #height, width = vid.shape[1], vid.shape[2]
    height, width = vid.shape[2], vid.shape[3]
    if height < crop_height or width < crop_width:
        raise ValueError("裁剪尺寸不能大于原图尺寸!")
    x = np.random.randint(0, width - crop_width)
    y = np.random.randint(0, height - crop_height)
    #return vid[:, y:y + crop_height, x:x + crop_width]
    return vid[..., y:y + crop_height, x:x + crop_width]


def vid_center_crop(vid, crop_height, crop_width):
    """
    对输入视频帧序列进行随机裁剪。
    参数:
        vid (numpy.ndarray): 形状为:
              (T, H, W) 或 (T, H, W, C)。
              (T, H, W) 或 (T, C, H, W)。
        crop_height (int): 裁剪的高度。
        crop_width (int): 裁剪的宽度。
    返回:
        numpy.ndarray: 随机裁剪后的序列。
    """
    #height, width = vid.shape[1], vid.shape[2]
    height, width = vid.shape[2], vid.shape[3]
    if height < crop_height or width < crop_width:
        raise ValueError("裁剪尺寸不能大于原图尺寸!")
    x = (width - crop_width) // 2
    y = (height - crop_height) // 2
    #return vid[:, y:y + crop_height, x:x + crop_width]
    return vid[..., y:y + crop_height, x:x + crop_width]


def spec_augment(mel_spec, freq_masking_para=5, time_masking_para=30, freq_mask_num=1, time_mask_num=1, time_first=False):
    """Spec augmentation Calculation Function.
    'specAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      freq_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      freq_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if time_first:
        mel_spec = mel_spec.transpose(1, 0)
    # (D, T)
    v = mel_spec.shape[0]
    tau = mel_spec.shape[1]
    # 如果logmel特征做了z-norm处理(均值为0，方差为1)，用0填充相当于均值填充
    repl_val = 0.  # or mel_spec.mean()
    # Step 1: Frequency masking
    for i in range(freq_mask_num):
        f = int(np.random.uniform(low=0.0, high=freq_masking_para))
        f0 = np.random.randint(0, v - f)
        mel_spec[f0:f0 + f, :] = repl_val  
    # Step 2: Time masking
    for i in range(time_mask_num):
        t = int(np.random.uniform(low=0.0, high=time_masking_para))
        t0 = np.random.randint(0, tau - t)
        # t0 = np.random.choice(range(0, tau - t))
        mel_spec[:, t0:t0 + t] = repl_val  
    if time_first:
        mel_spec = mel_spec.transpose(1, 0)
    return mel_spec



def batch_spec_augment(mel_spec, freq_masking_para=10, time_masking_para=30, freq_mask_num=1, time_mask_num=1, time_first=False):
    """Spec augmentation Calculation Function.
    'specAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      freq_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      freq_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if time_first:
        mel_spec = mel_spec.transpose(0, 2, 1)
    # (B, D, T)
    v = mel_spec.shape[1]
    tau = mel_spec.shape[2]
    repl_val = 0.  # or mel_spec.mean()
    # Step 1: Frequency masking
    for i in range(freq_mask_num):
        f = int(np.random.uniform(low=0.0, high=freq_masking_para))
        f0 = np.random.randint(0, v - f)
        mel_spec[:, f0:f0 + f, :] = repl_val  
    # Step 2: Time masking
    for i in range(time_mask_num):
        t = int(np.random.uniform(low=0.0, high=time_masking_para))
        t0 = np.random.randint(0, tau - t)
        # t0 = np.random.choice(range(0, tau - t))
        mel_spec[:, :, t0:t0 + t] = repl_val
    if time_first:
        mel_spec = mel_spec.transpose(0, 2, 1)
    return mel_spec

