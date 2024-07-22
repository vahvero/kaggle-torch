"""Visualization utility functions"""

from matplotlib.axis import Axis
from matplotlib.figure import Figure
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import seaborn as sb

from enum import StrEnum, auto


class DatasetTypeEnum(StrEnum):
    """Supported dataset types


    >>> DatasetTypeEnum.DETECTION == 'detection'
    True
    >>> DatasetTypeEnum.CLASSIFICATION == 'detection'
    False
    >>> DatasetTypeEnum.CLASSIFICATION == DatasetTypeEnum.CLASSIFICATION
    True
    """

    DETECTION = auto()
    CLASSIFICATION = auto()
    MASK = auto()


def square_gallery(
    dataset: Dataset,
    box_width: int,
    dataset_type: DatasetTypeEnum | str = DatasetTypeEnum.DETECTION,
) -> tuple[Figure, Axis]:
    """Create image gallery

    Args:
        dataset (Dataset): Image dataset
        box_width (int): Image square width
        dataset_type (str | DatasetTypeEnum): Dataset type, eg. 'classification' or DatasetTypeEnum.CLASSIFICATION

    Returns:
        tuple[Figure, Axis]: Images' figure and axis
    """
    # StrEnums can be compared to str directly
    match dataset_type:
        case DatasetTypeEnum.DETECTION:
            return _square_gallery_detection(dataset, box_width)
        case DatasetTypeEnum.MASK:
            return _square_gallery_mask(dataset, box_width)
        case DatasetTypeEnum.CLASSIFICATION:
            return _square_gallery_classification(dataset, box_width)
    raise NotImplementedError(f"Unrecogized dataset type '{dataset_type}'")


def _square_gallery_detection(dataset: Dataset, box_width: int):
    fig, axis = plt.subplots(nrows=box_width, ncols=box_width)
    idx_last = box_width**2 - 1
    for idx, (img, (boxes, labels)) in enumerate(dataset):
        ax = axis[divmod(idx, box_width)]

        img = draw_bounding_boxes(
            img,
            boxes,
            labels,
        )

        ax.imshow(to_pil_image(img))
        if idx >= idx_last:
            break
    return fig, axis


def _square_gallery_classification(
    dataset: Dataset,
    box_width: int,
):
    fig, axis = plt.subplots(nrows=box_width, ncols=box_width)
    idx_last = box_width**2 - 1
    for idx, (img, label) in enumerate(dataset):
        ax = axis[divmod(idx, box_width)]
        ax.imshow(to_pil_image(img))
        ax.set_title(f"Label={label}")
        if idx >= idx_last:
            break
    return fig, axis


def _square_gallery_mask():
    raise NotImplementedError()


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
