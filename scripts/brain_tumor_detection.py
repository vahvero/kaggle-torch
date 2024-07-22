"""Run brain tumor detection dataset

Download dataset from https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection

and extract to "assets" folder. You may use any other folder. Just chance
"--dataset-root" option to point to this folder.

Execute with example options

```bash
python -m scripts.brain_tumor_detection -lr 1e-4 \
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
from src.brain_tumor_detection import launch_training, BrainTumorDetectionTrainingKwargs

# Non-interactive backend
mpl.use("agg")
# Bigger images
mpl.rcParams["figure.figsize"] = 9, 9

if __name__ == "__main__":
    parser = setup_default_argument_parser(
        description="""Program fits a mask frcnn model to the brain tumor dataset
        from https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection. User
        must download this dataset manually.""",
        epilog="Implemented by vahvero, <https://github.com/vahvero>",
    )

    kwargs = BrainTumorDetectionTrainingKwargs(**vars(parser.parse_args()))
    setup_logging(kwargs.outfolder, kwargs.verbose)
    model, monitor, test_loss = launch_training(kwargs)
    fig, axis = monitor.visualize_epochs()
    fig.suptitle("Brain tumor detection training graphs")
    fig.savefig(kwargs.outfolder / "monitor.png")
