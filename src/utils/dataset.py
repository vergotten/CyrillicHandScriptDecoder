import torch
import json
from torchvision import transforms
import numpy as np
import Augmentor
import random
from PIL import Image
import traceback

from .text_utils import text_to_labels, labels_to_text
from .data_processing import process_data, generate_data
from .augmentations import Vignetting, UniformNoise, LensDistortion


class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, char2idx, idx2char, eval=False):
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval
        # print(f"labels: {labels}")
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
            transforms.RandomAffine(10, None, [0.6, 1], 3), # , fillcolor=255
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
        # print(f"Index: {index}, Type of index: {type(index)}")
        # print(f"Keys in self.labels: {self.labels.keys()}")

        if isinstance(index, torch.Tensor):
            index = index.item()

        img = self.images_name[index]

        # Check if img is a file path (a string)
        if isinstance(img, str):
            # Load the image file
            img = Image.open(img)
            # Convert the image to a NumPy array
            img = np.array(img)

        # Convert the numpy array to a PIL Image
        img = Image.fromarray(img)

        if not self.eval:
            img = self._transform(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = img / img.max()

        # Convert the image to float type
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif torch.is_tensor(img):
            img = img.float()

        label = text_to_labels(self.labels[index], self.char2idx)

        return img, torch.LongTensor(label)

    def __len__(self):
        # Ensure the length of the dataset is the minimum of the number of images and labels
        return min(len(self.images_name), len(self.labels))


if __name__ == "__main__":
    try:
        # Load the configuration file
        with open("configs/config.json") as json_file:
            config = json.load(json_file)

        print(f"config: {config}")

        # Process and generate data
        img2label, _, _ = process_data(config['path_train_dir'], config['path_train_labels'])

        # Get a small subset of your data for testing
        img2label = {k: img2label[k] for k in list(img2label)[:10]}
        # print(f"img2label: {img2label}")

        images_name = list(img2label.keys())
        # print(f"images_name: {images_name}")
        labels = list(img2label.values())
        # print(f"labels: {labels}")

        cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
                    '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З',
                    'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э',
                    'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
                    'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        char2idx = {char: idx for idx, char in enumerate(cyrillic)}
        idx2char = {idx: char for idx, char in enumerate(cyrillic)}

        # Generate data
        images_name = generate_data(images_name, config)

        # Create an instance of the TextLoader
        dataset = TextLoader(images_name, labels, char2idx, idx2char, eval=True)

        # Create a DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # Get only the first batch from the validation loader
        images, labels = next(iter(loader))
        # print(f"Images shape: {images.shape}")
        # print(f"Labels shape: {labels.shape}")
        # print(f"Labels: {labels}")

        # Assert that the shapes are as expected
        assert images.shape[0] == 1, "Unexpected batch size for images"
        assert labels.shape[0] == 1, "Unexpected batch size for labels"

        print("Preprocessing of one batch completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback (most recent call last):")
        traceback.print_exc()
