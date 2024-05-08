import torch
from torchvision import transforms
from utils.data_processing import text_to_labels
# from augmentations import Vignetting, UniformNoise, LensDistortion
# import Augmentor
import random
import numpy as np
import argparse
from src.config import Hparams


# # Create an instance of each augmentation class
# vignet = Vignetting()
# un = UniformNoise()
# tt = transforms.ToTensor()
# p = Augmentor.Pipeline()
# ld = LensDistortion()
# p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
# p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)


class TextLoader(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class takes in a list of image paths and corresponding labels, and applies transformations to the images.
    """
    def __init__(self, images_name, labels, char2idx, idx2char, eval=False):
        """
        Args:
            images_name (list): List of paths to images.
            labels (list): List of corresponding labels for images.
            char2idx (dict): Dictionary mapping characters to indices.
            idx2char (dict): Dictionary mapping indices to characters.
            eval (bool, optional): If True, applies different transformations to images. Defaults to False.
        """
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            p.torch_transform(),  # random distortion and shear
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
            return tt(ld(vignet(X)))
        if j == 2:
            return tt(ld(un(X)))

    def __getitem__(self, index):
        """
        Returns an image and its corresponding label at a given index.

        Args:
            index (int): Index.

        Returns:
            tuple: (image, label)
        """
        img = self.images_name[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = img / img.max()

        label = text_to_labels(self.labels[index], self.char2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        """
        Returns the total number of images.

        Returns:
            int: Total number of images.
        """
        return len(self.labels)


# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", help="Path to JSON configuration file")
    # args = parser.parse_args()
    #
    # hp = Hparams(args.config)
    #
    # char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    # idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}
    #
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Example usage:
    # images_name = ["./data/test/test0.png", "./data/test/test10.png"]
    # labels = ["ибо", "поле"]
    #
    # dataset = TextLoader(images_name, labels, char2idx, idx2char)
    #
    # for img, label in dataset:
    #     print("Image shape:", img.shape)
    #     print("Label:", label)
