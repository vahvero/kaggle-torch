"""Run blood smear dataset

Download dataset from https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask

and extract to "assets" folder. You may use any other folder. Just chance
"--dataset-root" option to point to this folder.

Execute with example options

```bash
python -m scripts.blood_smear -lr 1e-4 \
    -bs 4 -e 1000 -d cuda:0 -p 30 \
    -m resnet34 --dataset-root assets/Brain\ Tumor \
    --outfolder runs/brain_tumor
```

from the project root.
"""

import matplotlib as mpl
import torch

from src.utils import setup_default_argument_parser, setup_logging
from src.visualize import confusion_matrix

# Non-interactive backend
mpl.use("agg")
# Bigger images
mpl.rcParams["figure.figsize"] = 9, 9

if __name__ == "__main__":
    parser = setup_default_argument_parser(
        description="""Program fits a mask rcnn model to the blood-smear dataset
        from https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask. User
        must download this dataset manually.""",
        epilog="Implemented by vahvero, <https://github.com/vahvero>",
    )
