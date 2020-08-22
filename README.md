<img src="./.github/logo.svg" width="480">

# **GPT2** for Multiple Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imcaspar/gpt2-ml/blob/master/pretrained_model_demo.ipynb)
[![GitHub](https://img.shields.io/github/license/imcaspar/gpt2-ml)](https://github.com/imcaspar/gpt2-ml)
[![GitHub All Releases](https://img.shields.io/github/downloads/imcaspar/gpt2-ml/total)](https://github.com/imcaspar/gpt2-ml/releases)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/imcaspar/gpt2-ml/issues)
[![GitHub stars](https://img.shields.io/github/stars/imcaspar/gpt2-ml?style=social)](https://github.com/imcaspar/gpt2-ml)

[**中文说明**](./README_CN.md) | [**English**](./README.md)

- [x] Simplifed GPT2 train scripts（based on Grover, supporting TPUs）
- [x] Ported bert tokenizer, multilingual corpus compatible
- [x] 1.5B GPT2 pretrained Chinese model ( ~15G corpus, 10w steps )
- [x] Batteries-included Colab demo [#](https://github.com/imcaspar/gpt2-ml#google-colab)
- [x] 1.5B GPT2 pretrained Chinese model ( ~30G corpus, 22w steps )


## Pretrained Model
| Size            | Language | Corpus | Vocab                 | Link                                                         | SHA256                                                       |
| --------------- | -------- | ------ | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.5B parameters | Chinese  | ~30G   | CLUE ( 8021 tokens )  | [**Google Drive**](https://drive.google.com/file/d/1mT_qCQg4AWnAXTwKfsyyRWCRpgPrBJS3) | e698cc97a7f5f706f84f58bb469d614e<br/>51d3c0ce5f9ab9bf77e01e3fcb41d482 |
| 1.5B parameters | Chinese  | ~15G   | Bert ( 21128 tokens ) | [**Google Drive**](https://drive.google.com/file/d/1IzWpQ6I2IgfV7CldZvFJnZ9byNDZdO4n) | 4a6e5124df8db7ac2bdd902e6191b807<br/>a6983a7f5d09fb10ce011f9a073b183e |

Corpus from [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) and [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

Using [Cloud TPU Pod v3-256](https://cloud.google.com/tpu/docs/types-zones#types) to train 22w steps

![loss](./.github/loss.png)


## Google Colab
With just 2 clicks (not including Colab auth process), the 1.5B pretrained Chinese model demo is ready to go:

[**[Colab Notebook]**](https://colab.research.google.com/github/imcaspar/gpt2-ml/blob/master/pretrained_model_demo.ipynb)

<img src="./.github/demo.png" width="640">

## Train
**The following commands will generate big costs on Google Cloud!**

### Initial setup

Set the zone to europe-west4-a (the only one with big TPUs)

`gcloud config set compute/zone europe-west4-a`

Create a Storage Bucket

`gsutil mb -p <PROJECT-ID> -c standard gs://<BUCKET-NAME>`

Upload the dataset to the Bucket. Run the following command on your local computer

`gsutil cp -r ./dataset gs://<BUCKET-NAME>/`

Create a TPU and VM machine. Here we are setting the small v2-8 TPU as a safeguard, you will need bigger TPUs to train the models.

`ctpu up --tf-version=2.1 --project=<PROJECT-ID> --tpu-size=v2-8 --preemptible`

When logged in on the new VM machine type the install Tensorflow 1.15.2

`pip3 install tensorflow==1.15.2`

Clone the gpt2-ml from Deep-ESP

`git clone https://github.com/DeepESP/gpt2-ml.git`

Edit the training script and setup the OUTPUT_DIR (`gs://<BUCKET-NAME>/output`) and input_file (`gs://<BUCKET-NAME>/dataset`)

`nano train_tpu_adafactor.sh`

### Training

Run the training script

`bash train_tpu_adafactor.sh`

### Monitoring Training

From your local computer

`tensorboard --logdir gs://<BUCKET-NAME>/output --reload_interval 60`

Note to self: loss, accuracy, perplexity. Accuracy shoulg get near 40%.

Trick for faster fine tuning: Keep the same vector for tokens in common between Spanish and English.

### Get checkpoint

To download the checkpoints run the following command on your local computer

`gsutil cp -r gs://<BUCKET-NAME>/output ./`

This will download *all* checkpoints, you might want to download only the last one.

## Disclaimer
The contents in this repository are for academic research purpose, and we do not provide any conclusive remarks.

## Citation

```
@misc{GPT2-ML,
  author = {Zhibo Zhang},
  title = {GPT2-ML: GPT-2 for Multiple Languages},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/imcaspar/gpt2-ml}},
}
```

## Reference
https://github.com/google-research/bert

https://github.com/rowanz/grover

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)
