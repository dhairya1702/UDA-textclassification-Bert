"""
Transformer Model Classes & Config Class
This script defines various classes and functions related to Transformer-based models, including configuration, layers, and modules for classification and opinion extraction tasks.
"""
import math  # Mathematical functions
import json  # Module for working with JSON data
from typing import NamedTuple  # For defining named tuples

import numpy as np  # Library for numerical computing
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Functional module of PyTorch

from utils.utils import split_last, merge_last  # Custom utility functions


class Config(NamedTuple):
    """
    Configuration class for the BERT model.
    """
    vocab_size: int = None  # Size of Vocabulary
    dim: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12  # Number of Hidden Layers
    n_heads: int = 12  # Number of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768 * 4  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0.1  # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1  # Probability of Dropout of Attention Layers
    max_len: int = 512  # Maximum Length for Positional Embeddings
    n_segments: int = 2  # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        """
        Class method to create Config object from a JSON file.

        Args:
            file (str): Path to the JSON file containing configuration.

        Returns:
            Config: Config object created from JSON.
        """
        return cls(**json.load(open(file, "r")))


def gelu(x):
    """
    Implementation of the gelu activation function by Hugging Face.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after applying gelu activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """
    A layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, cfg, variance_epsilon=1e-12):
        """
        Initializes the LayerNorm module.

        Args:
            cfg (Config): Configuration object.
            variance_epsilon (float): Epsilon value for numerical stability.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after normalization.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """
    The embedding module from word, position and token_type embeddings.
    """
    def __init__(self, cfg):
        """
        Initializes the Embeddings module.

        Args:
            cfg (Config): Configuration object.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)  # Token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)  # Position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)  # Segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        """
        Forward pass of the Embeddings module.

        Args:
            x (Tensor): Input tensor.
            seg (Tensor): Segment tensor.

        Returns:
            Tensor: Output tensor after applying embeddings.
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Dot Product Attention.
    """
    def __init__(self, cfg):
        """
        Initializes the MultiHeadedSelfAttention module.

        Args:
            cfg (Config): Configuration object.
        """
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # For visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        Forward pass of the MultiHeadedSelfAttention module.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])

        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))

        h = (scores @ v).transpose(1, 2).contiguous()
        return merge_last(h, 2)


class PositionWiseFeedForward(nn.Module):
    """
    FeedForward Neural Networks for each position.
    """
    def __init__(self, cfg):
        """
        Initializes the PositionWiseFeedForward module.

        Args:
            cfg (Config): Configuration object.
        """
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        """
        Forward pass of the PositionWiseFeedForward module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying feedforward operation.
        """
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """
    Transformer Block.
    """
    def __init__(self, cfg):
        """
        Initializes the Block module.

        Args:
            cfg (Config): Configuration object.
        """
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        """
        Forward pass of the Block module.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Output tensor after passing through the block.
        """
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """
    Transformer with Self-Attentive Blocks.
    """
    def __init__(self, cfg):
        """
        Initializes the Transformer module.

        Args:
            cfg (Config): Configuration object.
        """
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        """
        Forward pass of the Transformer module.

        Args:
            x (Tensor): Input tensor.
            seg (Tensor): Segment tensor.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Output tensor after passing through the transformer.
        """
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


class Classifier(nn.Module):
    """
    Classifier with Transformer.
    """
    def __init__(self, cfg, n_labels):
        """
        Initializes the Classifier module.

        Args:
            cfg (Config): Configuration object.
            n_labels (int): Number of output labels.
        """
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        """
        Forward pass of the Classifier module.

        Args:
            input_ids (Tensor): Input IDs tensor.
            segment_ids (Tensor): Segment IDs tensor.
            input_mask (Tensor): Input mask tensor.

        Returns:
            Tensor: Output logits tensor.
        """
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits


class Opinion_extract(nn.Module):
    """
    Opinion Extraction module.
    """
    def __init__(self, cfg, max_len, n_labels):
        """
        Initializes the Opinion_extract module.

        Args:
            cfg (Config): Configuration object.
            max_len (int): Maximum length.
            n_labels (int): Number of output labels.
        """
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.extract = nn.Linear(cfg.dim, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, segment_ids, input_mask):
        """
        Forward pass of the Opinion_extract module.

        Args:
            input_ids (Tensor): Input IDs tensor.
            segment_ids (Tensor): Segment IDs tensor.
            input_mask (Tensor): Input mask tensor.

        Returns:
            Tensor: Output tensor.
        """
        h = self.transformer(input_ids, segment_ids, input_mask)
        h = self.drop(self.activ(self.fc(h[:, 1:-1])))
        seq_h = self.extract(h)
        seq_h = seq_h.squeeze()
        return self.sigmoid(seq_h)
