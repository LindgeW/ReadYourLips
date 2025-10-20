from pathlib import Path

import torch

from avhubert import AVHuBERT, Config

# https://github.com/kyushusouth/avhubert/tree/main

def load_avhubert_from_original_checkpoint(
    model_size: str, ckpt_path: Path
) -> AVHuBERT:
    if model_size not in ["base", "large"]:
        raise ValueError("model_size must be 'base' or 'large'")

    state = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    pretrained_dict = state["model"]
    
    cfg = Config(model_size)
    avhubert = AVHuBERT(cfg)
    avhubert_dict = avhubert.state_dict()
    match_dict = {k: v for k, v in pretrained_dict.items() if k in avhubert_dict}
    avhubert.load_state_dict(match_dict, strict=True)
    return avhubert


def main():
    model_size = "base"   # large

    # 下载ckpt的路径: https://facebookresearch.github.io/av_hubert
    # ckpt_path = "./ckpts/base_vox_iter5.pt"
    ckpt_path = "./base_lrs3_iter5.pt"
    #ckpt_path = "./ckpts/base_noise_vox_iter5.pt"
    # ckpt_path = "./ckpts/base_vox_433h.pt"
    # ckpt_path = "./ckpts/base_noise_pt_noise_ft_433h.pt"
    avhubert = load_avhubert_from_original_checkpoint(model_size, ckpt_path)

    # 重新保存为PyTorch ckpt
    # ckpt_path_new = "./base_vox_iter5_torch.ckpt"
    ckpt_path_new = "./base_lrs3_iter5_torch.ckpt"
    #ckpt_path_new = "./base_noise_vox_iter5_torch.ckpt"
    # ckpt_path_new = "./base_vox_433h_torch.ckpt"
    # ckpt_path_new = "./base_noise_pt_noise_ft_433h_torch.pt"
    
    torch.save(
        {
            "avhubert": avhubert.state_dict(),
        },
        ckpt_path_new,
    )

    print('Done')

    

if __name__ == "__main__":
    main()
