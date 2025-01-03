# 🚀 Unsupervised Data Augmentation for Text Classification using BERT 🚀

*Authors:*
- Dhairya Lalwani  
- Gokul Ranga Naveen Chapala
---


## ✨ Features

- **Baseline & UDA Experiments**: Scripts to run both standard BERT fine-tuning and UDA-based fine-tuning.
- **Back Translation**: Automated data augmentation by translating to another language and back to English.
- **Flexible Hyperparameters**: Control augmentation diversity, unsupervised loss weight, batch size, etc.
- **GPU & TPU Support**: Easily switch between local GPU training or large-scale TPU v3-32 Pod training.
- **High Accuracy**: Achieve around 90% accuracy on baseline GPU and 95%+ with TPU + UDA.

---

## 🎉 Requirements

The code is tested on **Python 3.7** and **Tensorflow 1.13**.  

After installing Tensorflow, run the following commands to install dependencies:

```bash
pip install --user absl-py
pip install fire tqdm tensorboardX pandas numpy
pip install tensorflow  # or tensorflow-gpu if you have a compatible GPU
pip install torch torchvision  # This installs PyTorch
```
## ⚙️ Instructions

If you want to run **UDA with BERT base** on a GPU with **11 GB memory**, run the following commands:

```bash
# Set a larger max_seq_length if your GPU has more than 11GB memory
MAX_SEQ_LENGTH=128

# Download data and pretrained BERT checkpoints
bash scripts/download.sh

# Preprocessing
bash scripts/prepro.sh --max_seq_length=${MAX_SEQ_LENGTH}

# Baseline accuracy: around 68%
bash scripts/run_base.sh --max_seq_length=${MAX_SEQ_LENGTH}

# UDA accuracy: around 90%
# Set a larger train_batch_size to achieve better performance if your GPU has a larger memory.
bash scripts/run_base_uda.sh --train_batch_size=8 --max_seq_length=${MAX_SEQ_LENGTH}
```


## ☁️ Run on Cloud TPU v3-32 Pod

The best performance in this code is achieved by using a `max_seq_length` of **512** and initializing with **BERT large** finetuned on in-domain unsupervised data.

If you have access to **Google Cloud TPU v3-32 Pod**, try:

```bash
MAX_SEQ_LENGTH=512

# Download data and pretrained BERT checkpoints
bash scripts/download.sh

# Preprocessing
bash scripts/prepro.sh --max_seq_length=${MAX_SEQ_LENGTH}

# UDA accuracy: 95.3% - 95.9%
bash train_large_ft_uda_tpu.sh
```

## 🔀 Run Back Translation Data Augmentation on the Dataset

Install the following dependencies:

```bash
pip install --user nltk
python -c "import nltk; nltk.download('punkt')"
pip install --user tensor2tensor==1.13.4
```
How it works:

1. Splits paragraphs into sentences.
2. Translates English sentences to French.
3. Translates them back into English.
4. Composes the paraphrased sentences back into paragraphs.
5. Go to the back_translate directory and run:
```bash
bash download.sh
bash run.sh
```

## ⚙️ Guidelines for Hyperparameters

**sampling_temp**: Controls the diversity and quality of paraphrases. Increasing `sampling_temp` increases diversity but may reduce quality.  
- Try `sampling_temp = 0.7, 0.8, or 0.9`.  
- If your task is robust to noise, `sampling_temp = 0.9` or `0.8` should lead to improved performance.  

To perform back translation on a large file, adjust `replicas` and `worker_id` in `run.sh`. For example, `replicas=3` divides the data into three parts, and each `run.sh` processes one part based on the `worker_id`.

---

## 📌 General Guidelines for Setting Hyperparameters

- **unsup_coeff**: Set to **1** for good results.
- **Learning Rate**: Use a lower learning rate than pure supervised learning because there are two loss terms (labeled + unlabeled data).
- If you have a **small dataset**, tweak `uda_softmax_temp` and `uda_confidence_thresh`.
- **Effective augmentation** for supervised learning typically works well for UDA.

---

## 🙏 Acknowledgement

A good portion of the code is taken from **BERT** and **RandAugment**.  
**Thanks!**


## 📚 Citation

```bibtex
@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}
