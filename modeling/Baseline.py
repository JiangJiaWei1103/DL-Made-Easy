"""
Baseline model architectures.
Author: JiaWei Jiang

This file contains definition of simple baseline model architectures,
such as fully-connected architecture, simple 1D conv, etc., which could
give users a sense about how well DL can perform on the specific task.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .common import Mish, Swish


class MLP(nn.Module):
    """Naive multi-layer perceptrons.

    Parameters:
        n_layers_emb: number of fully-connected layers for embedding
        n_layers_feat: number of fully-connected layers for numeric
            feature matrix
        n_layers_cat: number of fully-connected layers for concatenated
            feature matrix
        h_dim_emb: hidden dimension for embedding
        input_dim: input dimension
        h_dim_feat: hidden dimension for numeric feature matrix
        act_fn: activation function
        dropout: dropout ratio
    """

    act: Module

    def __init__(
        self,
        n_layers_emb: int = 3,
        n_layers_feat: int = 3,
        n_layers_cat: int = 3,
        h_dim_emb: int = 32,
        input_dim: int = 300,
        h_dim_feat: int = 128,
        act_fn: str = "swish",
        dropout: float = 0,
    ):
        self.name = self.__class__.__name__
        super(MLP, self).__init__()

        # Network parameters
        self.n_layers_emb = n_layers_emb
        self.n_layers_feat = n_layers_feat
        self.n_layers_cat = n_layers_cat
        self.h_dim_emb = h_dim_emb
        self.input_dim = input_dim
        self.h_dim_feat = h_dim_feat
        self.act_fn = act_fn
        self.dropout = dropout

        # Model blocks
        # FC structure for inv embedding
        if n_layers_emb is not None:
            self.embed = nn.Embedding(3774, h_dim_emb)
            self.fc_emb = nn.ModuleList()
            self.fc_emb.append(nn.Linear(h_dim_emb, 32))
            for layer in range(n_layers_emb - 1):
                self.fc_emb.append(nn.Linear(32, 32))
        # FC structure for feature map
        self.fc_feat = nn.ModuleList()
        self.fc_feat.append(nn.Linear(input_dim, h_dim_feat))
        for layer in range(n_layers_feat - 1):
            self.fc_feat.append(nn.Linear(h_dim_feat, h_dim_feat))
        # FC structure for concatenated latent representation
        self.fc_cat = nn.ModuleList()
        if n_layers_emb is not None:
            h_dim_cat_prev = 32 + h_dim_feat
        else:
            h_dim_cat_prev = h_dim_feat
        h_dim_cat_cur = h_dim_feat
        for layer in range(n_layers_cat):
            self.fc_cat.append(nn.Linear(h_dim_cat_prev, h_dim_cat_cur))
            h_dim_cat_prev = h_dim_cat_cur
            h_dim_cat_cur = int(h_dim_feat / 2 ** (layer + 1))
        # Output layer
        self.l_out = nn.Linear(h_dim_cat_prev, 1)
        # Activation
        if act_fn == "swish":
            self.act = Swish()
        elif act_fn == "tanh":
            self.act = nn.ReLU()
        elif act_fn == "mish":
            self.act = Mish()

    def forward(self, x: Tensor, inv_id: Tensor) -> Tensor:
        # Embed investment identifiers
        if self.n_layers_emb is not None:
            inv_emb = self.embed(inv_id)
            for layer in range(self.n_layers_emb):
                inv_emb = self.act(self.fc_emb[layer](inv_emb))
            if self.dropout != 0:
                inv_emb = nn.Dropout(self.dropout * 2)(inv_emb)

        # Non-linearly transform feature vectors
        for layer in range(self.n_layers_feat):
            x = self.act(self.fc_feat[layer](x))
            if self.dropout != 0:
                x = nn.Dropout(self.dropout)(x)

        # Non-linearly transform concatenate feature vectors
        if self.n_layers_emb is not None:
            x = torch.cat((inv_emb, x), dim=-1)
        for layer in range(self.n_layers_cat):
            x = self.act(self.fc_cat[layer](x))
            if self.dropout != 0:
                x = nn.Dropout(self.dropout)(x)

        output = self.l_out(x)

        return output
