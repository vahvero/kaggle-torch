"""Brain-tumor dataset utilities"""

import torch
import pathlib
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import io
from torchvision.transforms import Resize, Compose
from torchvision.transforms.functional import convert_image_dtype


class BrainTumorDataset(Dataset):
    """Dataset to read brain-tumor dataset

    User must have extracted the zip manually

    References:
        [dataset source](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor)

    """

    def __init__(self, root: str | os.PathLike, max_size: int = 0):
        super().__init__()
        self.root = pathlib.Path(os.fspath(root))
        assert self.root.is_dir(), "Root folder does not exist"
        # At root there should be "Brain tumor.csv"
        csv_file = self.root / "Brain Tumor.csv"
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"'Brain Tumor.csv' does no exist in {self.root}")

        df = pd.read_csv(csv_file)

        # In brain tumor folder, there should be a subdirectort "Brain Tumor"
        self.data: list[tuple[pathlib.Path, int]] = [
            (
                self.root / "Brain Tumor" / (fname + ".jpg"),
                torch.tensor(label, dtype=torch.float32),
            )
            for _, (fname, label) in df[["Image", "Class"]].iterrows()
        ]

        if max_size:
            self.data = self.data[:max_size]

        does_not_exist = []
        for fname, _ in self.data:
            if not os.path.isfile(fname):
                does_not_exist.append(fname)

        if does_not_exist:
            raise FileNotFoundError(
                "Images do not exist but should\n\t" + "\n\t".join(does_not_exist)
            )

        self.preprocess = Compose(
            [
                Resize(
                    [512, 512],
                    antialias=False,
                ),
            ]
        )

    @staticmethod
    def read_image(fpath: pathlib.Path) -> torch.FloatTensor:
        """Read image to 0-1 float tensor

        Args:
            fpath (pathlib.Path): Path to an image

        Returns:
            torch.FloatTensor: Image as float tensor
        """
        return convert_image_dtype(
            image=io.read_image(
                os.fspath(fpath),
                mode=io.ImageReadMode.RGB,
            ),
            dtype=torch.float32,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get dataset items at index

        Args:
            idx (int): Dataset fetch index

        Returns:
            tuple[torch.Tensor, int]: Image float tensor and integer label in {0,1}
        """
        fpath, label = self.data[idx]
        return self.preprocess(self.read_image(fpath)), label

    def __len__(self) -> int:
        """Dataset length

        Returns:
            int: Dataset length
        """
        return len(self.data)
