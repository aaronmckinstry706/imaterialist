"""
This file defines the TestDataset class, which is used for reading in test samples (i.e., samples having no label)
from a directory.
"""


import logging
import pathlib
import sys
from typing import List, Tuple, Any, Callable


import torch
import torch.utils.data as data
import PIL.Image as Image


LOGGER = logging.getLogger(__name__)


class TestDataset(data.Dataset):

    def __init__(self, directory: pathlib.Path,
                 read_fn: Callable[[pathlib.Path], Any],
                 transform: Callable[[Any], Any]):
        super().__init__()
        self.files = [path for path in directory.iterdir() if path.is_file()]
        self.read_fn = read_fn
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item: int):
        return self.transform(self.read_fn(self.files[item])), self.files[item]


def test_test_dataset():
    test_dataset = TestDataset(pathlib.Path('data/testing'), Image.open, lambda image: image.resize((224, 224)))
    image, path = test_dataset[0]
    assert isinstance(image, Image.Image)
    assert image.size == (224, 224)
    assert isinstance(path, pathlib.Path)
    assert len(test_dataset) > 10000


def collate_samples(samples: List[Tuple[torch.Tensor, pathlib.Path]]) -> Tuple[torch.Tensor, List[pathlib.Path]]:
    tensors, paths = zip(*samples)
    return torch.stack(tensors, dim=0), paths


def test_collate_fn():
    samples = [(torch.randn(3, 224, 224), pathlib.Path(str(i))) for i in range(32)]
    batch, paths = collate_samples(samples)
    expected_paths = tuple([pathlib.Path(str(i)) for i in range(32)])
    assert batch.size() == (32, 3, 224, 224)
    assert paths == expected_paths


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout)
    test_test_dataset()
    test_collate_fn()
