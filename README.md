# ReadYourLips
a Chinese medical lip-reading app for assisting people who are unable to speak normally in clinical practice.

### Overall Framework

![app framework](imgs/app.jpg)


### Traing model on CMLR
- Use the original pre-trained (AV-HuBERT)[https://facebookresearch.github.io/av_hubert] encoder as the model backbone.
- Convert the weights to pytorch version using [this](https://github.com/kyushusouth/avhubert).
- Translate Chinese characters into pinyin using [Hanzi2Pinyin](https://github.com/mozillazg/python-pinyin).
- [Train]: `python avtrain.py 0 train`.
- Translate pinyin sequence into Chinese using [QWen3](https://github.com/QwenLM/Qwen3) LLM (online API).


### Fine-tune model on healthcare data
- Collect the healthcare data by mobile phone.
- [Adapt]:  `python avtrain.py 0 adapt` to continue fine-tuning the pre-trained model in the previous step.


### Start the service
- `python server.py` to start the host-side server (ip address: 0.0.0.0).
- The mobile phone connects to the host service by setting the right ip.
- Record the frontal talking face.
