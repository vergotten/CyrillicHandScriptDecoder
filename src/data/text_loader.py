import torch
from torchvision import transforms
from utils.text_utils import text_to_labels, labels_to_text
from .augmentations import Vignetting, UniformNoise, LensDistortion
import Augmentor
import random
import numpy as np
# import argparse


class TextLoader(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class takes in a generator of image paths and corresponding labels, and applies transformations to the images.
    """
    def __init__(self, images_generator, labels, char2idx, idx2char, hp, eval=False):
        """
        Args:
            images_generator (generator): Generator of paths to images.
            labels (list): List of corresponding labels for images.
            char2idx (dict): Dictionary mapping characters to indices.
            idx2char (dict): Dictionary mapping indices to characters.
            eval (bool, optional): If True, applies different transformations to images. Defaults to False.
        """
        self.images_generator = images_generator
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval

        # Create an instance of each augmentation class
        self.vignet = Vignetting()
        self.un = UniformNoise()
        self.tt = transforms.ToTensor()
        self.p = Augmentor.Pipeline()
        self.ld = LensDistortion()
        self.p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
        self.p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            self.p.torch_transform(),  # random distortion and shear
            # transforms.Resize((int(hp.height *1.05), int(hp.width *1.05))),
            # transforms.RandomCrop((hp.height, hp.width)),
            transforms.ColorJitter(contrast=(0.5, 1), saturation=(0.5, 1)),
            transforms.RandomRotation(degrees=(-9, 9), fill=255),
            transforms.RandomAffine(10, None, [0.6, 1], 3),  # ,fillcolor=255
            transforms.ToTensor()
        ])

    def _transform(self, X, tt, ld, un, vignet):
        """
        Applies transformations to an image.

        Args:
            X (np.array): Input image.

        Returns:
            np.array: Transformed image.
        """
        j = np.random.randint(0, 3, 1)[0]
        if j == 0:
            return self.transform(X)
        if j == 1:
            return self.tt(self.ld(self.vignet(X)))
        if j == 2:
            return self.tt(self.ld(self.un(X)))

    def __getitem__(self, index):
        """
        Returns an image and its corresponding label at a given index.

        Args:
            index (int): Index.

        Returns:
            tuple: (image, label)
        """
        img = next(self.images_generator)
        img = img / img.max()
        img = (img ** (random.random() * 0.7 + 0.6) * 255).astype(np.uint8)
        img = np.transpose(img, (2, 0, 1))  # Always transpose the image
        img = (img / img.max() * 255).astype(np.uint8)

        print(f"self.labels[index]: {self.labels[index]}")
        # print(f"Shape of img: {img.shape}")
        label = text_to_labels(self.labels[index], self.char2idx); print(f"label: {label}")
        label2text = labels_to_text(label, self.idx2char); print(f"label2text: {label2text}")

        return torch.FloatTensor(img), torch.LongTensor(label)

    def __len__(self):
        """
        Returns the total number of images.

        Returns:
            int: Total number of images.
        """
        return len(self.labels)
