# %% Example

import pathlib
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision import io
from torchvision.transforms import functional as f
from torchvision import utils
import os

# %%

model = fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
)

# %%
path = pathlib.Path("../assets/dog.jpeg")
assert path.is_file()
dog_int = io.read_image(os.fspath(path))
img = f.convert_image_dtype(dog_int)

# %%
with torch.no_grad():
    model.eval()
    response = model(img.unsqueeze(0))

# %%
with torch.no_grad():
    model.eval()
    imgs = [img.detach().clone() for _ in range(16)]
    response = model(imgs)

# %%
dog_image = utils.draw_bounding_boxes(
    dog_int, response[0]["boxes"][response[0]["scores"] > 0.3]
)
f.to_pil_image(dog_image)

# %%
