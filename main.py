"""
UDA (Unsupervised Data Augmentation) Framework
Authors: Nikhil Chikkam, Vikas Sangireddy, Gokul Naveen Chapala, Dhairya Lalwani

This script provides functionality for training and evaluating models using the UDA framework.
"""

import fire

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration


# TSA: Temporal Shift Adaptation
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    """
    Calculate the threshold for Temporal Shift Adaptation.

    Args:
        schedule (str): The schedule type ('linear_schedule', 'exp_schedule', or 'log_schedule').
        global_step (int): The current global step in training.
        num_train_steps (int): Total number of training steps.
        start (float): Start value for threshold.
        end (float): End value for threshold.

    Returns:
        torch.Tensor: Threshold value for TSA.

    """
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def main(cfg, model_cfg):
    """
    Main function to train or evaluate a model using the UDA (Unsupervised Data Augmentation) framework.

    Args:
        cfg (str): Path to the configuration file for training or evaluation.
        model_cfg (str): Path to the BERT model configuration file.

    """
    # Load Configuration
    cfg = configuration.params.from_json(cfg)  # Train or Eval cfg
    model_cfg = configuration.model.from_json(model_cfg)  # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode == 'train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        data_iter = [data.sup_data_iter()]
    sup_criterion = nn.CrossEntropyLoss(reduction='none')

    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))

    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device())

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):
        """
        Compute the loss for supervised and unsupervised batches.

        Args:
            model: The model being trained.
            sup_batch: Batch of supervised data.
            unsup_batch: Batch of unsupervised data.
            global_step (int): Current global step in training.

        Returns:
            torch.Tensor: Total loss.
            torch.Tensor: Supervised loss.
            torch.Tensor: Unsupervised loss.

        """
        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
                aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)

        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1. / logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (
                        1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                           torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob = F.softmax(ori_logits, dim=-1)  # KLdiv target

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())

            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            # KLdiv loss
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                     torch_device_one())
            final_loss = sup_loss + cfg.uda_coeff * unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        """
        Compute accuracy of the model.

        Args:
            model: The model being evaluated.
            batch: Batch of data.

        Returns:
            torch.Tensor: Accuracy.
            torch.Tensor: Binary results.

        """
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)

        result = (label_pred == label_id).float()
        accuracy = result.mean()

        return accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
        trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
    # main('config/uda.json', 'config/bert_base.json')
