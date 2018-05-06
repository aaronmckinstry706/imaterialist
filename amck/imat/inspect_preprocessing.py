import argparse
import pathlib
import random
import typing


import matplotlib.pyplot as pyplot
import PIL.Image as Image
import torch
import torchvision.utils as utils
import torchvision.transforms as transforms


def main():
    CLASS = str(random.randint(0, 128))
    NUM_IMAGES = 4

    class_path = pathlib.Path('data/training') / CLASS
    assert class_path.exists()
    assert class_path.is_dir()
    image_paths = [d for d in class_path.iterdir()]
    random.shuffle(image_paths)
    image_paths = image_paths[:NUM_IMAGES]

    regular_image = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    preprocessed_image = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, scale=(1.0, 1.3), resample=Image.BILINEAR, fillcolor=2**24 - 1),
        transforms.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    images: typing.List[torch.Tensor] = []

    for image_path in image_paths:
        pil_image = Image.open(str(image_path))
        images.append(regular_image(pil_image))
        images.append(preprocessed_image(pil_image))

    image_grid = utils.make_grid(images, nrow=2)
    pyplot.imshow(image_grid.permute(1, 2, 0).numpy())
    pyplot.show(block=True)


if __name__ == '__main__':
    main()