"""Blood smear dataset utilities"""

import torch
import pathlib
from torch.utils.data import Dataset
import os
from torchvision import io
from torchvision.transforms.functional import convert_image_dtype


class BloodSmearDataset(Dataset):
    """Dataset to read brain-tumor dataset

    User must have extracted the zip manually

    References:
        [dataset source](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor)

    """

    def __init__(self, root: str | os.PathLike, max_size: int = 0):
        super().__init__()
        self.root = pathlib.Path(os.fspath(root))
        assert self.root.is_dir(), "Root folder does not exist"

        self.data = []

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

        self.preprocess = preprocesstarget

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get dataset items at index

        Args:
            idx (int): Dataset fetch index

        Returns:
            tuple[torch.Tensor, int]: Image float tensor and integer label in {0,1}
        """
        fpath, targets = self.data[idx]
        return (self.preprocess(self.read_image(fpath), targets),)

    def __len__(self) -> int:
        """Dataset length

        Returns:
            int: Dataset length
        """
        return len(self.data)
