"""Common torch training utilities"""

import os
import pathlib
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import torch
import logging
import argparse
import statistics
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from typing import Any
from dataclasses import dataclass, field, asdict

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from matplotlib import pyplot as plt
import seaborn as sb


class TorchResnetAction(argparse.Action):
    """Argument parser action to transform
    a string to a resnet class
    """

    supported_architextures = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    def __error_msg(self, value_str: str) -> str:
        return f"Supported values are {', '.join(self.supported_architextures)}, received {value_str}"

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            model = self.supported_architextures[values]
        except KeyError as exp:
            raise KeyError(self.__error_msg(values)) from exp
        setattr(namespace, self.dest, model)


class TorchDeviceAction(argparse.Action):
    """Argument parser action to transform
    a string to a torch device object
    """

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            device = torch.device(values)
        except RuntimeError as exp:
            raise ValueError("Invalid device passed") from exp
        setattr(
            namespace,
            self.dest,
            device,
        )


class CreateFolderAction(argparse.Action):
    """Argument parser action to create passed folder during parsing"""

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            os.makedirs(values, exist_ok=True)
        except OSError as exp:
            raise OSError(f"Attempt to create outfolder {values} failed") from exp
        setattr(namespace, self.dest, values)


def non_negative_int(value: str) -> int:
    """Confirm that passed integer string is positive

    Args:
        value (str): Argument string

    Raises:
        argparse.ArgumentTypeError: Negative integer

    Returns:
        int: Value as a integer
    """
    if (ret := int(value)) >= 0:
        return ret

    raise argparse.ArgumentTypeError("'%s' is not non-negative integer", value)


@dataclass
class DefaultArgumentParserArgs:
    """Commonly utilized machine learning parameters"""

    learning_rate: float
    batch_size: int
    max_epochs: int
    device: torch.device
    patience: int
    verbose: bool


def setup_default_argument_parser(
    prog: str,
    description: str,
    epilog: str,
) -> argparse.ArgumentParser:
    """Setup default argument parser with default arguments

    Args:
        prog (str): Program name
        description (str): Program description
        epilog (str): Program footer

    Returns:
        argparse.ArgumentParser: Argument parser with default arguments
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        required=True,
        help="Training learning rate",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=non_negative_int,
        required=True,
        help="Trainin batch size",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=non_negative_int,
        required=True,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        action=TorchDeviceAction,
        default="cpu",
        help="Utilized device, for example 'cpu' or 'cuda:0'",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=non_negative_int,
        required=True,
        help="Halt training if no improvement happens in patience epochs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream output to terminal",
    )

    return parser


def setup_logging(outfolder: pathlib.Path, verbose: bool) -> None:
    """Setup logging to outfolder with nice formatting defaults

    Args:
        outfolder (pathlib.Path): Training outfolder
        verbose (bool): Use verbose output
    """
    handlers = [logging.FileHandler(outfolder / "main.log", mode="w+")]
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
    )


@dataclass
class EpochValues:
    """Dataclass for epoch quanities"""

    training_acc: list[float] = field(default_factory=list)
    training_loss: list[float] = field(default_factory=list)
    validation_acc: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)


@dataclass
class TrainingMonitor:
    """Class to save all training quanities"""

    training_acc: list[float] = field(default_factory=list)
    training_loss: list[float] = field(default_factory=list)
    validation_acc: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    _epoch_values: EpochValues = field(default_factory=EpochValues)

    @property
    def epoch_values(self) -> EpochValues:
        """Read only property for `_epoch_values`

        Returns:
            EpochValues: Saved epoch values in dataclass
        """
        return self._epoch_values

    def save_epoch(self, learning_rate: float):
        """Saved

        Args:
            learning_rate (float): Epoch learning rate
        """
        self._epoch_values.training_acc.append(statistics.mean(self.training_acc))
        self._epoch_values.training_loss.append(statistics.mean(self.training_loss))
        self._epoch_values.validation_acc.append(statistics.mean(self.validation_acc))
        self._epoch_values.validation_loss.append(statistics.mean(self.validation_loss))
        self._epoch_values.learning_rate.append(learning_rate)

        # Reset values
        self.training_acc = []
        self.training_loss = []
        self.validation_acc = []
        self.validation_loss = []

    def get_epoch_string(self, idx: int = -1) -> str:
        """Get given epoch values as string

        Args:
            idx (int, optional): Epoch index. Defaults to -1 or the latest.

        Returns:
            str: String with formatted accuracies and losses
        """
        return " ".join(
            f"{key}={values[idx]:.2e}"
            for key, values in asdict(self._epoch_values).items()
        )

    def early_exit(self, patience: int) -> bool:
        """Check for improvemnt with range

        Args:
            patience (int): Range of required improvement

        Returns:
            bool: Boolean of check
        """
        if len(self.epoch_values.validation_loss) < patience:
            return False
        return (
            min(self.epoch_values.validation_loss)
            not in self.epoch_values.validation_loss[:-patience]
        )

    def latest_epoch_best(self) -> bool:
        """Check if last insert had lowest validation loss

        Returns:
            bool: Boolean of the check
        """
        return (
            min(self.epoch_values.validation_loss)
            == self.epoch_values.validation_loss[-1]
        )

    def visualize_epochs(self) -> tuple[Figure, Axis]:
        """Visualize monitor graphs

        Returns:
            tuple[Figure, Axis]: Matplotlib figure and axis
        """
        fig, axis = plt.subplots(nrows=3)
        sb.lineplot(
            data={
                "training": self.epoch_values.training_acc,
                "validation": self.epoch_values.validation_acc,
            },
            ax=axis[0],
        )

        sb.lineplot(
            data={
                "training": self.epoch_values.training_loss,
                "validation": self.epoch_values.validation_loss,
            },
            ax=axis[1],
        )
        sb.lineplot(
            data={"learning rate": self.epoch_values.learning_rate},
            ax=axis[2],
        )
        return fig, axis
