from typing import Literal
import torch
import pathlib
from torch.utils.data import Dataset
import os
import logging
from torch import Tensor

from src.io import read_image

logger = logging.getLogger(__name__)


class BrainTumorDetectionDataset(Dataset):
    """Dataset to read brain tumor detection datasets

    User must exctract folders correctly manually

    References:
       [dataset source](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)

    """

    valid_sets = frozenset("train", "valid", "test")

    def __init__(
        self,
        root: str | pathlib.Path,
        used_set: Literal["train", "valid", "test"],
        max_size: int = 8,
    ):
        if used_set not in self.valid_sets:
            logger.error("Invalid set '%s' given", used_set)
            raise ValueError(
                f"Invalid used set '{used_set}'. Valid are "
                + ", ".join(self.valid_sets)
                + "."
            )
        super().__init__()
        self.root = pathlib.Path(os.fspath(root))
        if not self.root.is_dir():
            raise FileNotFoundError(f"Root folder '{root}' does not exist")

        # There should be test, train and valid folders
        target_folder = self.root / used_set

        if not target_folder.is_dir():
            logger.error(f"'{target_folder}' folder does not exist in {root}")
            raise FileNotFoundError("Expected folder structure does not exist")

        if error := _assert_data_folder_existance(target_folder):
            raise error

        self.data = _read_data_folder(target_folder)
        self.keys = list(self.data.keys())
        if max_size:
            self.keys = self.keys[:max_size]
            self.data = {
                key: value for (key, value) in self.data.items() if key in self.keys
            }
        self.keys = tuple(self.keys)

    def __getitem__(self, index: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        img_path = self.keys[index]
        image = read_image(img_path)
        height, width = image.shape[:1]
        boxes = self.data[img_path]
        # TODO: Boxes have been marked to range [0,1]. Change to [0, w] and [0, h] respectively.
        labels = torch.ones([len(boxes)])
        return image, (boxes, labels)

    def __len__(self) -> int:
        return len(self.keys)


def _read_data_folder(folderpath) -> dict[pathlib.Path, Tensor]:
    out = {}
    image_folder = folderpath / "images"
    label_folder = folderpath / "labels"
    for image in image_folder.iterdir():
        if not image.is_file():
            logger.warning("'%s' is not a file", image)
            continue
        basename = os.basename(image)
        label_file = label_folder / f"{basename}.txt"
        if not label_file.is_file():
            logger.warning("'%s' does not corresponding label file", image)
            continue
        out[image] = _read_label_file(label_file)
    return out


def _read_label_file(filepath: os.PathLike) -> Tensor:
    with open(filepath) as fobj:
        lines = fobj.readlines()
        return torch.tensor(
            list(
                map(
                    lambda x: list(map(float, x.split()[1:])),
                    lines,
                )
            ),
            dtype=torch.float32,
        )


def _assert_data_folder_existance(folder: pathlib.Path) -> FileNotFoundError | None:
    if not (
        images_exist := (folder / "images").is_dir()
        and (labels_exist := (folder / "labels").is_dir())
    ):
        if not images_exist:
            logger.error("'images' folder does not exist in '%s'", folder)
        if not labels_exist:
            logger.error("'labels' folder does not exist in '%s'", folder)

        return FileNotFoundError(f"Expected '{folder}' folders do not exist")
    return None
