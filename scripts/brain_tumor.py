"""Run brain tumor dataset

Download dataset from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor.

and extract to "assets" folder. You may use any other folder. Just chance
"--dataset-root" option to point to this folder.

Execute with example options

```bash
python -m scripts.brain_tumor -lr 1e-4 \
    -bs 4 -e 1000 -d cuda:0 -p 30 \
    -m resnet34 --dataset-root assets/Brain\ Tumor \
    --outfolder runs/brain_tumor
```

from the project root.

"""
import matplotlib as mpl
import torch

from src.brain_tumor.training import BrainTumorTrainingKwargs, launch_training
from src.utils import setup_default_argument_parser, setup_logging
from src.visualize import confusion_matrix

# Non-interactive backend
mpl.use("agg")
# Bigger images
mpl.rcParams["figure.figsize"] = 9, 9

if __name__ == "__main__":
    parser = setup_default_argument_parser(
        prog="Brain tumor trainer",
        description="""Program fits a resnet model to the brain-tumor dataset
        from https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor. User
        must download this dataset manually.""",
        epilog="Implemented by vahvero, <https://github.com/vahvero>",
    )
    BrainTumorTrainingKwargs.add_arguments(parser)

    kwargs = BrainTumorTrainingKwargs(**vars(parser.parse_args()))
    setup_logging(kwargs.outfolder, kwargs.verbose)
    model, monitor, test_loss, test_conf_matr = launch_training(kwargs)
    test_acc = torch.diag(test_conf_matr).sum() / torch.sum(test_conf_matr)
    fig, axis = monitor.visualize_epochs()
    fig.suptitle("Brain tumor training graphs")
    fig.savefig(kwargs.outfolder / "monitor.png")

    fig, axis = confusion_matrix(test_conf_matr)
    fig.suptitle(f"Brain tumor acc={100 * test_acc.item():.2f}%")
    fig.savefig(kwargs.outfolder / "testset_confusion_matrix.png")
