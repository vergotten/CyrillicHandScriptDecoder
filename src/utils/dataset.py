import torch
from torchvision import transforms
import numpy as np
import Augmentor
import random
from PIL import Image

from .text_utils import text_to_labels
from .augmentations import Vignetting, UniformNoise, LensDistortion


class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, char2idx, idx2char, eval=False):
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval

        # Initialize the augmentation functions
        self.vignet = Vignetting()
        self.un = UniformNoise()
        self.tt = transforms.ToTensor()
        self.ld = LensDistortion()

        # Initialize Augmentor pipeline
        self.p = Augmentor.Pipeline()
        self.p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
        self.p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            self.p.torch_transform(),  # random distortion and shear
            transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
            transforms.RandomRotation(degrees=(-9, 9), fill=255),
            # transforms.RandomAffine(10 ,None ,[0.6 ,1] ,3 ), # ,fillcolor=255
            transforms.RandomAffine(10, None, [0.6, 1], 3, fillcolor=255),
            transforms.ToTensor()
        ])

    def _transform(self, X):
        j = np.random.randint(0, 3, 1)[0]
        if j == 0:
            # Convert X to a numpy array before transforming it
            X = np.array(X).astype(np.uint8)
            return self.transform(X)
        if j == 1:
            # Convert X to a numpy array before passing it to self.vignet
            X = np.array(X)
            return self.tt(self.ld(self.vignet(X)))
        if j == 2:
            # Convert X to a numpy array before passing it to self.un
            X = np.array(X)
            return self.tt(self.ld(self.un(X)))

    def __getitem__(self, index):
        # Use a context manager to open and automatically close the image
        with Image.open(self.images_name[index]) as img:
            if not self.eval:
                img = self._transform(img)
                img = img / img.max()
                img = img ** (random.random() * 0.7 + 0.6)
            else:
                img = np.transpose(img, (2, 0, 1))
                img = img / img.max()

            # Ensure img is a single precision tensor before converting it to a float tensor
            img = img.float()

        label = text_to_labels(self.labels[index], self.char2idx)
        return torch.FloatTensor(img), torch.LongTensor(label)

    def __len__(self):
        # Ensure the length of the dataset is the minimum of the number of images and labels
        return min(len(self.images_name), len(self.labels))
