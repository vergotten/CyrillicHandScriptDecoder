import unittest
import torch
import os
import json
import pandas as pd

from src.utils.dataset import TextLoader
from src.utils.text_utils import labels_to_text


class TestTextLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the configuration file
        with open('configs/config.json', 'r') as f:
            cls.config = json.load(f)

        # Define character to index and index to character mappings
        cls.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        cls.char2idx = {char: idx for idx, char in enumerate(cls.cyrillic)}
        cls.idx2char = {idx: char for idx, char in enumerate(cls.cyrillic)}

        # Read the train.tsv file
        df = pd.read_csv(cls.config['path_train_labels'], sep='\t', header=None)
        df.columns = ['image', 'label']

        # Convert the labels to a list
        cls.labels = df['label'].tolist()

        # Convert the image names to a list with full path
        cls.images_name = [os.path.join(cls.config['path_train_dir'], img) for img in df['image'].tolist()]

    def test_len(self):
        dataset = TextLoader(self.images_name, self.labels, self.char2idx, self.idx2char)
        self.assertEqual(len(dataset), len(self.images_name))

    def test_getitem(self):
        dataset = TextLoader(self.images_name, self.labels, self.char2idx, self.idx2char)
        for i in range(len(dataset))[:10]:
            img, label = dataset[i]
            self.assertIsInstance(img, torch.FloatTensor)
            self.assertIsInstance(label, torch.LongTensor)
            # print(f"img.shape: {img.shape}")
            # print(f"img.shape[0]: {img.shape[0]}")
            # Check the size of the image tensor
            self.assertEqual(len(img.shape), 3)  # Check if the image tensor is 3D
            self.assertEqual(img.shape[0],
                             4)  # Check if the number of channels is 4 (assuming you're working with RGBA images)

            # Check the size of the label tensor
            self.assertEqual(len(label.shape), 1)  # Check if the label tensor is 1D

    def test_label_conversion(self):
        dataset = TextLoader(self.images_name, self.labels, self.char2idx, self.idx2char)
        for i in range(len(dataset))[:10]:
            _, label = dataset[i] # ; print(f"label: {label}")
            text_label = labels_to_text(label, self.idx2char) # ; print(text_label)
            self.assertEqual(text_label, self.labels[i]) # ; print(f"\n\"{text_label}\" is equal to \"{self.labels[i]}\"")


if __name__ == '__main__':
    unittest.main()
