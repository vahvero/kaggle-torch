import argparse
import pathlib
from typing import Callable
from dataclasses import dataclass
from src.utils import DefaultArgumentParserArgs, TorchResnetAction
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

@dataclass
class BloodSmearTrainingKwargs(DefaultArgumentParserArgs):
    backbone: nn.Module
    dataset_root: pathlib.Path
    outfolder: pathlib.Path

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--backbone",
            type=str,
            action=TorchResnetAction,
            required=True,
            help="Set utilized resnet backbone, for example 'resnet18'",
        )

        parser.add_argument(
            "--dataset-root",
            type=pathlib.Path,
            required=True,
            help="Blood smear dataset root directory",
        )
        parser.add_argument(
            "--outfolder",
            type=pathlib.Path,
            required=True
            help="Result output folder, Attempts to create if does not exist",
        )


def setup_model(model_cls: Callable[[Any], nn.Module]) -> nn.Module:
    pass

def split_dataset(dataset: Dataset, batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    pass
    
def launch_training():pass

