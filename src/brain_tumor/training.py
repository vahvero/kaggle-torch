"""Brain tumor binary training functions"""
import argparse
import logging
import os
import statistics
import pathlib
from typing import Any, Callable, TypeVar
import torch
import torchmetrics
from torch import nn, optim
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, Compose
from src.brain_tumor.data import BrainTumorDataset
from src.utils import (
    TrainingMonitor,
    DefaultArgumentParserArgs,
    TorchResnetAction,
    CreateFolderAction,
    non_negative_int,
)
from dataclasses import asdict, dataclass
from src.visualize import square_gallery
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def setup_model(model_cls: Callable[[Any], nn.Module]) -> nn.Module:
    """Create binary model with backbone of `model_cls`

    Args:
        model_cls (Callable[[Any], nn.Module]): Model factory function

    Returns:
        nn.Module: Resnet module with backbone resnet weigths and
        single sigmoid output
    """
    # Always use transfer learning from imagenet
    model = model_cls(weights="DEFAULT")

    in_dims = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_dims, in_dims),
        nn.ReLU(),
        nn.Linear(in_dims, 1),
        nn.Sigmoid(),
    )

    # Test
    with torch.no_grad():
        batch = torch.rand([2, 3, 512, 512], dtype=torch.float32)
        _ = model(batch)

    return model


A = TypeVar("A")
B = TypeVar("B")


def collate_fn(samples: list[tuple[A, B]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate image

    Args:
        samples (list[tuple[A,B]): List of image-target tuples

    Returns:
        tuple[Tensor, Tensor]: Tuple of concatenated images tensor and targets
    """
    imgs, targets = tuple(zip(*samples))
    return (torch.stack(imgs), torch.tensor(targets).unsqueeze(dim=1))


def split_dataset(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset to train, validation and test loaders

    Args:
        dataset (Dataset): Image dataset
        batch_size (int): Utilized batch size
        device (torch.device): Target device

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train,
    """
    train, val, test = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42),
    )

    num_workers = min(batch_size, cpu_count())
    train_loader = DataLoader(
        dataset=train,
        pin_memory="cuda" in device.type,
        num_workers=num_workers,
        persistent_workers=bool(num_workers),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        dataset=val,
        pin_memory="cuda" in device.type,
        num_workers=num_workers,
        persistent_workers=bool(num_workers),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        dataset=test,
        pin_memory="cuda" in device.type,
        num_workers=num_workers,
        persistent_workers=bool(num_workers),
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader


@dataclass
class BrainTumorTrainingKwargs(DefaultArgumentParserArgs):
    """Brain tumor training arguements"""

    model: nn.Module
    dataset_root: pathlib.Path
    outfolder: pathlib.Path
    dataset_max_size: int

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add class arguments to existing argument parsera

        Args:
            parser (argparse.ArgumentParser): Argumenparsing object

        Returns:
            argparse.ArgumentParser: Argument parser with dataclass objects as arguments
        """
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            action=TorchResnetAction,
            required=True,
            help="Utilized resnet as a string, for example resnet18",
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

    def __str__(self):
        return " ".join(f"{key}={value}" for key, value in asdict(self).items())


def launch_training(
    kwargs: BrainTumorTrainingKwargs,
) -> tuple[nn.Module, TrainingMonitor, float, torch.Tensor]:
    """Run brain tumor training

    Args:
        kwargs (BrainTumorTrainingKwargs): _description_

    Returns:
        tuple[nn.Module, TrainingMonitor, float, torch.Tensor]: Model with best weights,
        monitor, test loss and test confusion matrix 2d-tensor
    """
    logger.info("Setting up training with %s", kwargs)
    dataset = BrainTumorDataset(kwargs.dataset_root, kwargs.dataset_max_size)
    if kwargs.verbose:
        logger.info("Exporting example images")
        fig, _ = square_gallery(dataset, 2)
        fig.savefig(kwargs.outfolder / "example_images.png")

    model = setup_model(kwargs.model)
    model = model.to(kwargs.device)
    model_save_path = kwargs.outfolder / "best_model.pth"

    training_monitor = TrainingMonitor()

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=kwargs.learning_rate,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=kwargs.patience // 3,
    )
    # Add first lr to scheduler, patience implications do not matter
    lr_scheduler.step(float("-inf"))

    loss_fn = nn.BCELoss()

    train_loader, val_loader, test_loader = split_dataset(
        dataset,
        kwargs.batch_size,
        kwargs.device,
    )

    augments = Compose(
        [
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
        ]
    )
    logger.info("Starting training")
    for epoch in range(1, kwargs.max_epochs + 1):
        model.train()
        for imgs, targets in tqdm(
            train_loader,
            desc=f"Training {epoch}",
            disable=not kwargs.verbose,
        ):
            optimizer.zero_grad()
            imgs = imgs.to(kwargs.device)
            imgs = augments(imgs)
            targets = targets.to(kwargs.device)
            response = model(imgs)
            loss = loss_fn(response, targets)
            loss.backward()
            training_monitor.training_loss.append(loss.item())
            training_monitor.training_acc.append(
                torch.sum((response > 0.5) == targets).item() / len(response)
            )
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for imgs, targets in tqdm(
                val_loader,
                desc=f"Validation {epoch}",
                disable=not kwargs.verbose,
            ):
                imgs = imgs.to(kwargs.device)
                targets = targets.to(kwargs.device)

                response = model(imgs)
                loss = loss_fn(response, targets)
                training_monitor.validation_loss.append(loss.item())
                training_monitor.validation_acc.append(
                    torch.sum((response > 0.5) == targets).item() / len(response)
                )

        training_monitor.save_epoch(lr_scheduler._last_lr[0])
        lr_scheduler.step(training_monitor.epoch_values.validation_loss[-1])
        logger.info("%d. epoch %s", epoch, training_monitor.get_epoch_string())

        # Save model, if best
        if training_monitor.latest_epoch_best():
            logger.info(
                "New best model found with %.4f",
                training_monitor.epoch_values.validation_loss[-1],
            )
            torch.save(model.state_dict(), model_save_path)

        # Early exit
        if training_monitor.early_exit(kwargs.patience):
            logger.info(
                "No improvement in %d epochs, exiting training",
                kwargs.patience,
            )
            break

    logger.info("Running tests")
    model.load_state_dict(torch.load(model_save_path))

    test_image_folder = kwargs.outfolder / "test_images"
    os.makedirs(test_image_folder, exist_ok=True)
    conf_matr = torchmetrics.classification.BinaryConfusionMatrix()
    conf_matr = conf_matr.to(kwargs.device)
    test_loss: list[float] = []
    max_out = 25
    with torch.no_grad():
        model.eval()
        for imgs, targets in tqdm(
            test_loader,
            desc="Testing",
            disable=not kwargs.verbose,
        ):
            imgs = imgs.to(kwargs.device)
            targets = targets.to(kwargs.device)

            response = model(imgs)
            loss = loss_fn(response, targets)
            test_loss.append(loss.item())
            conf_matr.update(response, targets)

            if max_out > 0:
                for img, resp, target in zip(imgs, response, targets, strict=True):
                    fig, axis = plt.subplots()
                    axis.imshow(to_pil_image(img))
                    axis.set_title(f"{(resp > 0.5).item()}, gt={target.item()}")
                    fig.savefig(test_image_folder / f"img{max_out}.png")
                    plt.close(fig)
                    max_out -= 1
                    if max_out <= 0:
                        break

    mean_test_loss = statistics.mean(test_loss)
    conf_matr = conf_matr.compute()
    accuracy = torch.diag(conf_matr).sum() / torch.sum(conf_matr)
    logger.info("Test loss=%.4f acc=%.2f%%", mean_test_loss, 100 * accuracy.item())
    return model, training_monitor, mean_test_loss, conf_matr
