import torch
from torchvision import transforms
import Augmentor
import random
import numpy as np

from .text_utils import text_to_labels, labels_to_text
from .augmentations import Vignetting, UniformNoise, LensDistortion
from .data_processing import generate_data


class TransformedTextDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class takes in a generator of image paths and corresponding labels, and applies a series of transformations to the images. The transformations include random distortions, shearing, color jittering, rotation, and affine transformations. The class also supports different transformations for evaluation mode.

    Attributes:
        images_generator: A generator that yields images from the provided paths.
        labels: A list of labels corresponding to the images.
        char2idx: A dictionary mapping characters to indices.
        idx2char: A dictionary mapping indices to characters.
        train: A boolean indicating whether the dataset is in training mode.
    """
    def __init__(self, img_paths, labels, char2idx, idx2char, hp, train=True):
        """
        Args:
            img_paths: Paths to images.
            labels (list): List of corresponding labels for images.
            char2idx (dict): Dictionary mapping characters to indices.
            idx2char (dict): Dictionary mapping indices to characters.
            train (bool, optional): If True, applies different transformations to images. Defaults to True.
        """
        self.images_generator = generate_data(img_paths, hp)
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.train = train

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

    def _transform(self, X):
        """
        Applies transformations to an image.

        Args:
            X (np.array): Input image.

        Returns:
            np.array: Transformed image.
        """
        if self.train:
            j = np.random.randint(0, 3, 1)[0]
            if j == 0:
                return self.transform(X)
            if j == 1:
                return self.tt(self.ld(self.vignet(X)))
            if j == 2:
                return self.tt(self.ld(self.un(X)))
        else:
            # Apply fewer or no transformations during evaluation
            return self.tt(X)

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

        # print(f"self.labels[index]: {self.labels[index]}")
        # print(f"Shape of img: {img.shape}")
        label = text_to_labels(self.labels[index], self.char2idx) # ; print(f"label: {label}")
        label2text = labels_to_text(label, self.idx2char) # ; print(f"label2text: {label2text}")
        # print(f"torch.LongTensor(label): {torch.LongTensor(label)}")

        return torch.FloatTensor(img), torch.LongTensor(label)

    def __len__(self):
        """
        Returns the total number of images.

        Returns:
            int: Total number of images.
        """
        return len(self.labels)
