from torch.utils import data
from torchvision import datasets, transforms

from typing import Tuple

from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data'

# Datasets

def load_cifar10() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load CIFAR-10 train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 32 * 32 * 3


def load_fashion_mnist() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load fashion MNIST train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 28 * 28


def load_mnist() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load MNIST train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 28 * 28

def load_j() -> Tuple[data.Dataset, data.Dataset, int]: 
    return None, None, 99