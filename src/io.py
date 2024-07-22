from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype
from os import PathLike, fspath
import torch


def read_image(fpath: PathLike) -> torch.FloatTensor:
    """Read image to 0-1 float tensor

    Args:
        fpath (pathlib.Path): Path to an image

    Returns:
        torch.FloatTensor: Image as float tensor
    """
    return convert_image_dtype(
        image=read_image(
            fspath(fpath),
            mode=ImageReadMode.RGB,
        ),
        dtype=torch.float32,
    )
