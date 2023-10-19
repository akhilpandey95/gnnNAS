# This Source Code Form is subject to the terms of the
# BSD 2-Clause "Simplified" License. If a copy of the same
# was not distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/gnnNAS/blob/master/LICENSE.

import numpy as np
import torch
import torch_geometric as pyg


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_molnet_phys_chem_data(
        name: str, batch_size: int, training_split: float, seed: int = 2022
    ):
        """
        Load the specific Graph dataset from MoleculeNet
        Parameters
        ----------
        arg1 | name: str
            Name of the dataset to import from Pytorch Geometric MoleculeNet dataloader.
        arg2 | batch_size: int
            Batch size for creating the train/test dataloaders.
        arg3 | training_split: float
            Percentage of samples to be kept in training set.
        arg4 | seed: int
            Torch Random seed to ensure reproducibility. Default value is 2022
        Returns
        -------
        Pytorch Geometric Dataset(s)
            torch_geometric.datasets.molecule_net.MoleculeNet
        """
        # load the dataset
        dataset = pyg.datasets.MoleculeNet(root="/tmp/Molnet", name=name)

        # set the seed
        torch.manual_seed(seed)

        # shuffle the data
        dataset = dataset.shuffle()

        # set an stop index for gathering train data
        stop_index = int(np.floor(training_split * dataset.len()))

        # separate training data
        train_dataset = dataset[0:stop_index]

        # separate test data
        test_dataset = dataset[stop_index:]

        # create dataloaders for train and test samples
        train_loader = pyg.loader.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = pyg.loader.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return dataset, train_loader, test_loader
