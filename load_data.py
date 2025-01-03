'''
Data Loader for Supervised and Unsupervised Text Classification Tasks

This module provides classes and functions to load data for supervised and unsupervised
text classification tasks. It handles data preprocessing, tokenization, and creating
PyTorch datasets and data loaders for training and evaluation.

Authors: Nikhil Chikkam, Vikas Sangireddy, Gokul Naveen Chapala, Dhairya Lalwani

'''
import ast  # Module for literal evaluation of strings to Python objects
import csv  # Module for reading and writing CSV files
import itertools  # Module for efficient looping and iteration
import pandas as pd  # Library for data manipulation and analysis
import torch  # PyTorch library for deep learning
from torch.utils.data import Dataset, DataLoader  # Classes and functions for creating custom datasets and data loaders
from tqdm import tqdm  # Library for adding progress bars to loops

from utils import tokenization  # Custom module for tokenization
from utils.utils import truncate_tokens_pair  # Function for truncating token pairs


class CsvDataset(Dataset):
    """
    Custom PyTorch dataset class for loading data from a CSV file.
    """
    labels = None  # Placeholder for class labels

    def __init__(self, file, need_prepro, pipeline, max_len, mode, d_type):
        """
        Initializes the CsvDataset.

        Args:
            file (str): Path to the CSV file.
            need_prepro (bool): Flag indicating whether preprocessing is needed.
            pipeline (list): List of preprocessing steps.
            max_len (int): Maximum sequence length.
            mode (str): Dataset mode ('train', 'train_eval', or 'eval').
            d_type (str): Dataset type ('sup' for supervised or 'unsup' for unsupervised).
        """
        Dataset.__init__(self)
        self.cnt = 0  # Counter for unsupervised dataset

        # Need preprocessing
        if need_prepro:
            with open(file, 'r', encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t', quotechar='"')

                # Supervised dataset
                if d_type == 'sup':
                    data = []
                    for instance in self.get_sup(lines):
                        for proc in pipeline:
                            instance = proc(instance, d_type)
                        data.append(instance)

                    self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

                # Unsupervised dataset
                elif d_type == 'unsup':
                    data = {'ori': [], 'aug': []}
                    for ori, aug in self.get_unsup(lines):
                        for proc in pipeline:
                            ori = proc(ori, d_type)
                            aug = proc(aug, d_type)
                        self.cnt += 1
                        data['ori'].append(ori)
                        data['aug'].append(aug)
                    ori_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['ori'])]
                    aug_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['aug'])]
                    self.tensors = ori_tensor + aug_tensor
        # Already preprocessed
        else:
            f = open(file, 'r', encoding='utf-8')
            data = pd.read_csv(f, sep='\t')

            if d_type == 'sup':
                input_columns = ['input_ids', 'input_type_ids', 'input_mask', 'label_ids']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long) \
                                for c in input_columns[:-1]]
                self.tensors.append(torch.tensor(data[input_columns[-1]], dtype=torch.long))

            elif d_type == 'unsup':
                input_columns = ['ori_input_ids', 'ori_input_type_ids', 'ori_input_mask',
                                 'aug_input_ids', 'aug_input_type_ids', 'aug_input_mask']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long) \
                                for c in input_columns]
            else:
                raise "d_type error. (d_type have to sup or unsup)"

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple of tensors representing the sample.
        """
        return tuple(tensor[index] for tensor in self.tensors)

    def get_sup(self, lines):
        """
        Generator function to yield supervised data instances.

        Args:
            lines (iterator): Iterator over the lines of the CSV file.

        Yields:
            tuple: Tuple representing a supervised data instance.
        """
        raise NotImplementedError

    def get_unsup(self, lines):
        """
        Generator function to yield unsupervised data instances.

        Args:
            lines (iterator): Iterator over the lines of the CSV file.

        Yields:
            tuple: Tuple representing an unsupervised data instance.
        """
        raise NotImplementedError


class Pipeline():
    """
    Base class for defining data preprocessing pipelines.
    """

    def __init__(self):
        pass

    def __call__(self, instance):
        """
        Applies the preprocessing pipeline to a data instance.

        Args:
            instance: Input data instance.

        Returns:
            object: Processed data instance.
        """
        raise NotImplementedError


class Tokenizing(Pipeline):
    """
    Preprocessing pipeline for tokenizing input text.
    """

    def __init__(self, preprocessor, tokenize):
        """
        Initializes the Tokenizing pipeline.

        Args:
            preprocessor (function): Preprocessing function for text.
            tokenize (function): Tokenization function.
        """
        super().__init__()
        self.preprocessor = preprocessor
        self.tokenize = tokenize

    def __call__(self, instance, d_type):
        """
        Applies tokenization to the input instance.

        Args:
            instance (tuple): Input instance consisting of (label, text_a, text_b).
            d_type (str): Dataset type ('sup' or 'unsup').

        Returns:
            tuple: Processed instance.
        """
        label, text_a, text_b = instance

        label = self.preprocessor(label) if label else None
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b else []

        return label, tokens_a, tokens_b


class AddSpecialTokensWithTruncation(Pipeline):
    """
    Preprocessing pipeline for adding special tokens and truncating sequences.
    """

    def __init__(self, max_len=512):
        """
        Initializes the AddSpecialTokensWithTruncation pipeline.

        Args:
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance, d_type):
        """
        Adds special tokens and truncates sequences in the input instance.

        Args:
            instance (tuple): Input instance consisting of (label, tokens_a, tokens_b).
            d_type (str): Dataset type ('sup' or 'unsup').

        Returns:
            tuple: Processed instance.
        """
        label, tokens_a, tokens_b = instance

        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return label, tokens_a, tokens_b


class TokenIndexing(Pipeline):
    """
    Preprocessing pipeline for converting tokens to indexes and creating input tensors.
    """

    def __init__(self, indexer, labels, max_len=512):
        """
        Initializes the TokenIndexing pipeline.

        Args:
            indexer (function): Function to convert tokens to indexes.
            labels (tuple): Tuple of class labels.
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.indexer = indexer
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance, d_type):
        """
        Converts tokens to indexes and creates input tensors.

        Args:
            instance (tuple): Input instance consisting of (label, tokens_a, tokens_b).
            d_type (str): Dataset type ('sup' or 'unsup').

        Returns:
            tuple: Processed instance.
        """
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)
        input_mask = [1] * (len(tokens_a) + len(tokens_b))
        label_id = self.label_map[label] if label else None

        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        if label_id is not None:
            return input_ids, segment_ids, input_mask, label_id
        else:
            return input_ids, segment_ids, input_mask


def dataset_class(task):
    """
    Function to get the dataset class based on the task.

    Args:
        task (str): Task name.

    Returns:
        class: Dataset class corresponding to the task.
    """
    table = {'imdb': IMDB}
    return table[task]


class IMDB(CsvDataset):
    """
    Dataset class for the IMDB dataset.
    """
    labels = ('0', '1')

    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup'):
        """
        Initializes the IMDB dataset.

        Args:
            file (str): Path to the dataset file.
            need_prepro (bool): Flag indicating whether preprocessing is needed.
            pipeline (list): List of preprocessing steps.
            max_len (int): Maximum sequence length.
            mode (str): Dataset mode ('train', 'train_eval', or 'eval').
            d_type (str): Dataset type ('sup' or 'unsup').
        """
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type)

    def get_sup(self, lines):
        """
        Generator function to yield supervised data instances.

        Args:
            lines (iterator): Iterator over the lines of the dataset file.

        Yields:
            tuple: Tuple representing a supervised data instance.
        """
        raise NotImplementedError

    def get_unsup(self, lines):
        """
        Generator function to yield unsupervised data instances.

        Args:
            lines (iterator): Iterator over the lines of the dataset file.

        Yields:
            tuple: Tuple representing an unsupervised data instance.
        """
        raise NotImplementedError


class load_data:
    """
    Class for loading and processing data for training or evaluation.
    """

    def __init__(self, cfg):
        """
        Initializes the load_data class.

        Args:
            cfg: Configuration object.
        """
        self.cfg = cfg
        self.TaskDataset = dataset_class(cfg.task)
        self.pipeline = None

        # Create preprocessing pipeline if needed
        if cfg.need_prepro:
            tokenizer = tokenization.FullTokenizer(vocab_file=cfg.vocab, do_lower_case=cfg.do_lower_case)
            self.pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                             AddSpecialTokensWithTruncation(cfg.max_seq_length),
                             TokenIndexing(tokenizer.convert_tokens_to_ids, self.TaskDataset.labels,
                                          cfg.max_seq_length)]

        # Set up data loading parameters based on mode
        if cfg.mode == 'train':
            self.sup_data_dir = cfg.sup_data_dir
            self.sup_batch_size = cfg.train_batch_size
            self.shuffle = True
        elif cfg.mode == 'train_eval':
            self.sup_data_dir = cfg.sup_data_dir
            self.eval_data_dir = cfg.eval_data_dir
            self.sup_batch_size = cfg.train_batch_size
            self.eval_batch_size = cfg.eval_batch_size
            self.shuffle = True
        elif cfg.mode == 'eval':
            self.sup_data_dir = cfg.eval_data_dir
            self.sup_batch_size = cfg.eval_batch_size
            self.shuffle = False  # Do not shuffle when in eval mode

        # Set up unsupervised data loading parameters if in uda_mode
        if cfg.uda_mode:
            self.unsup_data_dir = cfg.unsup_data_dir
            self.unsup_batch_size = cfg.train_batch_size * cfg.unsup_ratio

    def sup_data_iter(self):
        """
        Creates and returns a data loader for supervised data.

        Returns:
            DataLoader: DataLoader for supervised data.
        """
        sup_dataset = self.TaskDataset(self.sup_data_dir, self.cfg.need_prepro, self.pipeline,
                                       self.cfg.max_seq_length, self.cfg.mode, 'sup')
        sup_data_iter = DataLoader(sup_dataset, batch_size=self.sup_batch_size, shuffle=self.shuffle)
        return sup_data_iter

    def unsup_data_iter(self):
        """
        Creates and returns a data loader for unsupervised data.

        Returns:
            DataLoader: DataLoader for unsupervised data.
        """
        unsup_dataset = self.TaskDataset(self.unsup_data_dir, self.cfg.need_prepro, self.pipeline,
                                         self.cfg.max_seq_length, self.cfg.mode, 'unsup')
        unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)
        return unsup_data_iter

    def eval_data_iter(self):
        """
        Creates and returns a data loader for evaluation data.

        Returns:
            DataLoader: DataLoader for evaluation data.
        """
        eval_dataset = self.TaskDataset(self.eval_data_dir, self.cfg.need_prepro, self.pipeline,
                                        self.cfg.max_seq_length, 'eval', 'sup')
        eval_data_iter = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False)
        return eval_data_iter
