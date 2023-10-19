# This Source Code Form is subject to the terms of the
# BSD 2-Clause "Simplified" License. If a copy of the same
# was not distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/gnnNAS/blob/master/LICENSE.

import logging
from collections import defaultdict
from typing import Callable

import torch
import torch_geometric as pyg
from sklearn.metrics import *

# check if CUDA exists
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {DEVICE} device")


class Util:
    """Utility methods"""

    # define the helper method to train
    def model_train(
        model: torch.nn.Module,
        dataloader: pyg.loader.DataLoader,
        learning_rate: float,
        ITERATIONS: int,
        optimizer: torch.optim.Optimizer,
        compute_loss: Callable,
        logging=False,
    ) -> torch.nn.Module:
        """
        Train the Pytorch Geometric model and return
        the model
        Parameters
        ----------
        arg1 | model: torch.nn.Module
            Trained Neural network model
        arg2 | dataloader: DataLoader
            Dataset as a DataLoader object
        arg3 | model: float
            Trained Neural network model
        arg4 | dataloader: int
            Dataset as a DataLoader object
        arg5 | model: torch.optim.Optimizer
            Trained Neural network model
        arg6 | dataloader: DataLoader
            Dataset as a DataLoader object
        Returns
        -------
        Pytorch model
            torch.nn.Module
        """
        for iteration in range(ITERATIONS):
            # set the model for training
            model.train()

            # iterate in batches over the training dataset
            for data in dataloader:
                # set the gradients to zero
                optimizer.zero_grad()

                # forward pass and compute the y hat values
                y_hat = model(
                    data.x.float().to(DEVICE),
                    data.edge_index.long().to(DEVICE),
                    data.batch.long().to(DEVICE),
                )

                # compute the mean squared error loss
                cost = compute_loss(y_hat, data.y.to(DEVICE))

                # compute mse loss again for the backward pass
                cost.backward()

                # update the weights
                optimizer.step()

            # display the stats
            if logging:
                print(f"Epoch: {iteration:03d}, Loss: {cost:.4f}")

        # return the tuple [Ground truth, Predictions]
        return model

    # define the helper method to evaluate
    def model_evaluate(
        model: torch.nn.Module,
        test_loader: pyg.loader.DataLoader,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Evaluate the Pytorch model and return
        ground truth along with predictions
        Parameters
        ----------
        arg1 | model: torch.nn.Module
            Trained Neural network model
        arg2 | test_loader: DataLoader
            Dataset as a DataLoader object
        """
        # init an empty list to capture y hats
        y_preds = []

        # init an empty list to capture ground truth
        y_true = []

        # set the model to evaluate
        model.eval()

        # Iterate in batches over the test dataset.
        for data in test_loader:
            # store the ground truth
            y_true.append(data.y)

            # gather the model prediction
            out = model(
                data.x.float().to(DEVICE),
                data.edge_index.long().to(DEVICE),
                data.batch.long().to(DEVICE),
            )

            # store the model predictions
            y_preds.append(torch.flatten(out, start_dim=1))

        # concat the predictions obtained in batches
        y_preds = torch.cat(y_preds)

        # concat the ground truth obtained in batches
        y_true = torch.cat(y_true)

        # return the tuple [Ground truth, Predictions]
        return (y_true, y_preds)

    # define the helper method to obtain evaluation metrics
    def regression_evaluation_metrics(
        y_true: torch.Tensor, y_preds: torch.Tensor, metric: str
    ) -> None:
        """
        Print the Pytorch model metrics based
        on the ground truth vs predictions
        Parameters
        ----------
        arg1 | y_true: torch.Tensor
            Ground truth values of the data
        arg2 | y_preds: torch.Tensor
            Model Predictions for the input data
        """
        # init an empty dict to store results
        results = defaultdict(dict)

        # store y_preds and y_true as numpy arrays
        y_true = y_true.detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        # MSE
        results["mse"] = mean_squared_error(y_true, y_preds)

        # MAE
        results["mae"] = mean_absolute_error(y_true, y_preds)

        # RMSE
        results["rmse"] = mean_squared_error(y_true, y_preds, squared=False)

        # R2
        results["r2"] = r2_score(y_true, y_preds)

        # return appropriate metric(s)
        if metric == "all":
            return results
        if metric in results.keys():
            return results[metric]
        else:
            return defaultdict(dict)
