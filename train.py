"""
Training Helper Class

This script defines the Trainer class, which assists in training deep learning models, specifically Transformer-based models, with support for unsupervised data augmentation (UDA).

It also includes functionality for evaluation and model saving/loading.

"""

import os  # Operating system module
import json  # Module for working with JSON data
from copy import deepcopy  # Function for creating deep copies of objects
from typing import NamedTuple  # For defining named tuples
from tqdm import tqdm  # Progress bar library

import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module
from tensorboardX import SummaryWriter  # TensorBoardX for logging

from utils import checkpoint  # Custom utility functions for checkpointing
# from utils.logger import Logger  # Custom logger (not used in this script)
from utils.utils import output_logging  # Custom utility functions


class Trainer(object):
    """
    Training Helper class

    This class provides methods for training deep learning models, including functionality for unsupervised data augmentation (UDA),
    evaluation, and model saving/loading.
    """

    def __init__(self, cfg, model, data_iter, optimizer, device):
        """
        Initializes the Trainer object.

        Args:
            cfg: Configuration object containing training settings.
            model: PyTorch model to be trained.
            data_iter: Iterable containing training data.
            optimizer: PyTorch optimizer for training.
            device: Device (CPU or GPU) on which to perform training.
        """
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file):
        """
        Trains the model.

        Args:
            get_loss: Function to compute the loss.
            get_acc: Function to compute the accuracy.
            model_file: Path to the file containing the model weights.
            pretrain_file: Path to the file containing pre-trained model weights (if available).

        Returns:
            int: Total number of training steps.
        """

        # tensorboardX logging
        if self.cfg.results_dir:
            logger = SummaryWriter(log_dir=os.path.join(self.cfg.results_dir, 'logs'))

        self.model.train()
        self.load(model_file, pretrain_file)  # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        if self.cfg.data_parallel:  # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]  # acc, step

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
            else tqdm(self.sup_iter, total=self.cfg.total_steps)
        for i, batch in enumerate(iter_bar):

            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step)
            final_loss.backward()
            self.optimizer.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f' \
                                         % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging
            if self.cfg.uda_mode:
                logger.add_scalars('data/scalar_group',
                                   {'final_loss': final_loss.item(),
                                    'sup_loss': sup_loss.item(),
                                    'unsup_loss': unsup_loss.item(),
                                    'lr': self.optimizer.get_lr()[0]
                                    }, global_step)
            else:
                logger.add_scalars('data/scalar_group',
                                   {'sup_loss': final_loss.item()}, global_step)

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)

            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > 4999:
                results = self.eval(get_acc, None, model)
                total_accuracy = torch.cat(results).mean().item()
                logger.add_scalars('data/scalar_group', {'eval_acc': total_accuracy}, global_step)
                if max_acc[0] < total_accuracy:
                    self.save(global_step)
                    max_acc = total_accuracy, global_step
                print('Accuracy : %5.3f' % total_accuracy)
                print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' % (
                max_acc[0], max_acc[1], global_step), end='\n\n')

            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum / (i + 1)))
                if get_acc:
                    results = self.eval(get_acc, None, model)
                    total_accuracy = torch.cat(results).mean().item()
                    logger.add_scalars('data/scalar_group', {'eval_acc': total_accuracy}, global_step)
                    if max_acc[0] < total_accuracy:
                        max_acc = total_accuracy, global_step
                    print('Accuracy :', total_accuracy)
                    print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' % (
                    max_acc[0], max_acc[1], global_step), end='\n\n')
                self.save(global_step)
                return
        return global_step

    def eval(self, evaluate, model_file, model):
        """
        Evaluates the model.

        Args:
            evaluate: Function to evaluate the model.
            model_file: Path to the file containing the model weights.
            model: PyTorch model to evaluate.

        Returns:
            list: List of evaluation results.
        """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results

    def load(self, model_file, pretrain_file):
        """
        Loads model weights.

        Args:
            model_file: Path to the file containing the model weights.
            pretrain_file: Path to the file containing pre-trained model weights.
        """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                     for key, value in torch.load(pretrain_file).items()
                     if key.startswith('transformer')}
                )  # load only transformer parts

    def save(self, i):
        """
        Saves the model weights.

        Args:
            i: Index to differentiate saved model files.
        """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                   os.path.join(self.cfg.results_dir, 'save', 'model_steps_' + str(i) + '.pt'))

    def repeat_dataloader(self, iterable):
        """
        Repeats the dataloader.

        Args:
            iterable: Iterable containing data to repeat.

        Returns:
            generator: Generator object for repeated data.
        """
        while True:
            for x in iterable:
                yield x
