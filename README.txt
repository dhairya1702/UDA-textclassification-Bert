***

Unsupervised Data Augmentation ​for Text Classification ​using Bert​

-Dhairya Lalwani  ​
-Gokul Ranga Naveen Chapala ​
-Nikhil Chikkam​
-Vikas Reddy Sangireddy

***

Requirements:

The code is tested on Python 3.7 and Tensorflow 1.13. After installing
Tensorflow, run the following command to install dependencies:


>>pip install --user absl-py
>>pip install fire tqdm tensorboardX pandas numpy
>>pip install tensorflow  # or tensorflow-gpu if you have a compatible GPU
>>pip install torch torchvision  # This installs PyTorch

		Instructions

If you want to run UDA with BERT base on a GPU with 11 GB memory, run the following commands:

# Set a larger max_seq_length if your GPU has a memory larger than 11GB
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


### Run on Cloud TPU v3-32 Pod

The best performance in this code is achieved by using a max_seq_length
of 512 and initializing with BERT large finetuned on in-domain
unsupervised data. If you have access to Google Cloud TPU v3-32 Pod,
try:

``` shell
MAX_SEQ_LENGTH=512

# Download data and pretrained BERT checkpoints
bash scripts/download.sh

# Preprocessing
bash scripts/prepro.sh --max_seq_length=${MAX_SEQ_LENGTH}

# UDA accuracy: 95.3% - 95.9%
bash train_large_ft_uda_tpu.sh
```

## Run back translation data augmentation on the dataset

Install the following dependencies:

>>pip install --user nltk
>>python -c "import nltk; nltk.download('punkt')"
>>pip install --user tensor2tensor==1.13.4

The following command translates the provided example file. It
automatically splits paragraphs into sentences, translates English
sentences to French and then translates them back into English. Finally,
it composes the paraphrased sentences into paragraphs. Go to the
*back_translate* directory and run:

>>bash download.sh
>>bash run.sh


		Guidelines for hyperparameters:

There is a variable *sampling_temp* in the bash file. It is used to
control the diversity and quality of the paraphrases. Increasing
sampling_temp will lead to increased diversity but worse quality.


We suggest trying to set sampling_temp to 0.7, 0.8 and 0.9. If your task
is very robust to noise, sampling_temp=0.9 or 0.8 should lead to
improved performance.

If you want to do back translation to a large file, you can change the
replicas and worker_id arguments in run.sh. For example, when
replicas=3, we divide the data into three parts, and each run.sh will
only process one part according to the worker_id.

		General guidelines for setting hyperparameters:

-   It works well to set the weight on unsupervised objective
    *'unsup_coeff'* to 1.
-   Use a lower learning rate than pure supervised learning because
    there are two loss terms computed on labeled data and unlabeled data
    respecitively.
-   If your have an extremely small amount of data, try to tweak
    'uda_softmax_temp' and 'uda_confidence_thresh' a bit. For more
    details about these two hyperparameters.
-   Effective augmentation for supervised learning usually works well
    for UDA.

			 Acknowledgement

A good portion of the code is taken from
[BERT](https://github.com/google-research/bert) and
[RandAugment](https://github.com/tensorflow/models/tree/master/research/autoaugment).
Thanks!

			   Citation

 @article{xie2019unsupervised, title={Unsupervised Data Augmentation for Consistency Training},
      author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
      journal={arXiv preprint arXiv:1904.12848},year={2019} }
