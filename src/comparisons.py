import fire
import torch

from utils import load_cifar10, load_fashion_mnist, load_mnist, load_j

from typing import Dict, Callable
from pathlib import Path

from torch_harness import TorchHarness
from dilation_erosion import DenMoNet

model_dir = Path(__file__).parent.parent / 'models'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_map: Dict[str, Callable] = {
    "cifar10": load_cifar10,
    "fashion_mnist": load_fashion_mnist,
    "mnist": load_mnist,
    "j": load_j
}

# Models

def run_and_save_model(model, train, test, epochs, name):
    harness = TorchHarness(model, model.name(name), train, test, epochs=epochs)
    harness.train_and_evaluate()
    model.store(name, model_dir)


# Runners

def run_denmo(dset_name: str, erosions: int = 5, dilations: int = 5, epochs: int = 2):
    """Run denmo on a dataset.

    Datasets:
        * mnist
        * fashion_mnist
        * cifar10
    """
    train, test, size = dataset_map[dset_name]()
    model = DenMoNet(size, dilations, erosions, 10)
    run_and_save_model(model, train, test, epochs, dset_name)


def run_denmo_se(dset_name: str, erosions: int = 5, dilations: int = 5, epochs: int = 2):
    train, test, size = dataset_map[dset_name]()
    pass

if __name__ == '__main__':
    fire.Fire()
