# ReadYourLips
a Chinese medical lip-reading app for assisting people who are unable to speak normally in clinical practice.

### Overall Framework

![app framework](imgs/app.jpg)


### Traing model on CMLR
- Download the original pre-trained (AV-HuBERT)[https://facebookresearch.github.io/av_hubert] weights.
- Convert the weights to pytorch version using [this](https://github.com/kyushusouth/avhubert).
- Translate Chinese characters into pinyin using [Hanzi2Pinyin](https://github.com/mozillazg/python-pinyin).
- [Train]: python avtrain.py 0 train

### Fine-tune model on healthcare data




#### Key Dependencies
- Hanzi2Pinyin:
- QWen3-Max
