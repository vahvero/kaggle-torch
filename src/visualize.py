"""Visualization utility functions"""
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
import seaborn as sb


def square_gallery(dataset: Dataset, box_width: int) -> tuple[Figure, Axis]:
    """Create image gallery

    Args:
        dataset (Dataset): Image dataset
        box_width (int): Image square width

    Returns:
        tuple[Figure, Axis]: Images' figure and axis
    """
    fig, axis = plt.subplots(nrows=box_width, ncols=box_width)
    idx_last = box_width**2 - 1
    for idx, (img, label) in enumerate(dataset):
        ax = axis[divmod(idx, box_width)]
        ax.imshow(to_pil_image(img))
        ax.set_title(f"Label={label}")
        if idx >= idx_last:
            break
    return fig, axis


def confusion_matrix(data: torch.Tensor) -> tuple[Figure, Axis]:
    """Generate confusion matrix from data

    Args:
        data (torch.Tensor): 2d image tensor

    Returns:
        tuple[Figure, Axis]: Image's figure and axis

    References:
        https://stackoverflow.com/questions/74780953/add-row-wise-accuracy-to-a-seaborn-heatmap

    """
    assert (
        len(data.shape) == 2 and data.shape[0] == data.shape[1]
    ), f"Unexpected data shape {data.shape}"
    fig, axis = plt.subplots()
    sb.heatmap(
        data.cpu().numpy(),
        annot=True,
        cbar=False,
        fmt="3d",
        ax=axis,
    )
    row_accuracies = 100 * torch.diag(data) / data.sum(dim=1)
    axis.tick_params(
        axis="y",
        which="major",
        left=True,
        right=True,
        labelleft=True,
        labelright=False,
    )
    for i, acc in enumerate(row_accuracies):
        axis.text(axis.get_xlim()[1] * 1.05, axis.get_yticks()[i] * 1.01, f"{acc:.2f}%")

    return fig, axis
