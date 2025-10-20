
import hydra   # https://blog.csdn.net/qq_39537898/article/details/122162896
import omegaconf
import torch

from avhubert import AVHuBERT


def load_avhubert_from_torch_checkpoint(cfg: omegaconf.DictConfig) -> AVHuBERT:
    avhubert = AVHuBERT(cfg)
    ckpt_path = cfg.ckpt_path
    if cfg.load_pretrained_weight:
        print(f"Loading pretrained weight from {str(ckpt_path)}")
        pretrained_dict = torch.load(str(ckpt_path), weights_only=True)["avhubert"]
        avhubert.load_state_dict(pretrained_dict, strict=True)
    return avhubert


@hydra.main(config_path="./conf", config_name="base")  # conf/base.yaml
def main(cfg: omegaconf.DictConfig) -> None:
    # cfg包含了配置文件中的所有信息, 可通过cfg.<section>.<key>的方式来访问配置项
    print(cfg)
    avhubert = load_avhubert_from_torch_checkpoint(cfg)
    print(avhubert)
    '''
    图像帧采样频率为25fps（40ms取一帧），音频则是从原始波形以10ms步长提取26维对数滤波器组能量特征
    音频10ms提取一次特征，堆叠4个音频帧（4×10ms = 40ms）就能和一帧视频的时间跨度匹配，实现帧数同步
    '''
    vid = torch.randn(8, 1, 75, 88, 88)
    aud = torch.randn(8, 26*4, 75)
    feat = avhubert(vid, aud)
    print(feat.shape)

    print('Done')


if __name__ == "__main__":
    main()
