import argparse
from src.utils import (
    CreateFolderAction,
    DefaultArgumentParserArgs,
    TorchResnetAction,
    non_negative_int,
)

from torch import nn
import pathlib
import logging

logger = logging.getLogger(__name__)


class BrainTumorDetectionTrainingKwargs(DefaultArgumentParserArgs):
    model: nn.Module
    dataset_root: pathlib.Path
    outfolder: pathlib.Path
    dataset_max_size: int

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "-b",
            "--backbone",
            type=str,
            action=TorchResnetAction,
            required=True,
            help="Utilized backbone resnet as a string, for example resnet18",
        )
        parser.add_argument(
            "--dataset-root",
            type=pathlib.Path,
            required=True,
            help="Brain tumor dataset root directory",
        )
        parser.add_argument(
            "--outfolder",
            type=pathlib.Path,
            required=True,
            action=CreateFolderAction,
            help="Folder to output results to. Attempts to create if does not exist.",
        )
        parser.add_argument(
            "--dataset-max-size",
            type=non_negative_int,
            default=0,
            help="Dataset size limit",
        )


def launch_training(kwargs: BrainTumorDetectionTrainingKwargs):

    logger.info("Setting up training with %s", kwargs)
    dataset = BrainTumorDetectionDataset(kwargs.dataset_root, kwargs.dataset_max_size)
    if kwargs.verbose:
        logger.info("Exporting dataset images")
