# This Source Code Form is subject to the terms of the
# BSD 2-Clause "Simplified" License. If a copy of the same
# was not distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/gnnNAS/blob/master/LICENSE.

import numpy as np
import torch
import torch_geometric as pyg


class MPNN(torch.nn.Module):
    """Creates an MPNN model in pytorch geometric"""

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        MPNN_inp: torch.nn.Module,
        MPNN_hidden: torch.nn.Module,
        n_conv_blocks: int,
        skip_connection: str = "plain",
    ) -> None:
        """
        Build the MPNN model
        Parameters
        ----------
        arg1 | n_node_features: int
            Number of features at node level
        arg2 | n_edge_features: int
            Number of features at edge level
        arg3 | n_hidden: int
            Number of hidden activations
        arg4 | n_output: int
            Number of output activations
        arg5 | n_conv_blocks: int
            Number of convolutional kernels
        Returns
        -------
        Nothing
            None
        """
        # super class the class structure
        super().__init__()

        # set the growth dimension
        self.growth_dimension = n_hidden

        # encode the node information
        self.node_encoder = MPNN_inp(n_node_features, n_hidden)

        # add the ability to add one or more conv layers
        conv_blocks = []

        # ability to add one or more conv blocks
        for block in range(n_conv_blocks):
            if skip_connection == "dense":
                self.growth_dimension = n_hidden + (n_hidden * block)
            conv = MPNN_hidden(self.growth_dimension, n_hidden)
            norm = torch.nn.LayerNorm(n_hidden, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)
            layer = pyg.nn.DeepGCNLayer(conv, norm, act, block=skip_connection)
            conv_blocks.append(layer)

        # group all the conv layers
        self.conv_layers = torch.nn.ModuleList(conv_blocks)

        # add the linear layers for flattening the output from MPNN
        self.flatten = torch.nn.Sequential(
            torch.nn.Linear(self.growth_dimension, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Process the MPNN model
        Parameters
        ----------
        arg1 | x: torch.Tensor
            Input features at node level
        arg2 | edge_index: torch.Tensor
            Index pairs of verticies
        arg3 | batch_idx: torch.Tensor
            Batch index
        Returns
        -------
        Tensor
            torch.Tensor
        """
        # obtaint the input
        if isinstance(self.node_encoder, pyg.nn.MessagePassing):
            x = self.node_encoder(x, edge_index)
        else:
            x = self.node_encoder(x)

        # pass the node information to the conv layer
        x = self.conv_layers[0].conv(x, edge_index)

        # process the layers
        for layer in range(len(self.conv_layers[1:])):
            x = self.conv_layers[layer](x, edge_index)

        # obtain the output from the MPNN final layer
        y = pyg.nn.global_add_pool(x, batch=batch_idx)

        # pass the output to the linear output layer
        out = self.flatten(y)

        # return the output
        return out
