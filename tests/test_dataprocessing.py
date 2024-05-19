import unittest
import os
import numpy as np
import cv2
import json

from src.utils.data_processing import process_data, process_image, generate_data, train_valid_split, get_mixed_data, get_batch


class Hyperparameters:
    def __init__(self, height, width):
        self.height = height
        self.width = width


class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the configuration file
        with open('configs/config.json', 'r') as f:
            cls.config = json.load(f)

        # Create a Hyperparameters object
        cls.hp = Hyperparameters(cls.config['height'], cls.config['width'])

    def test_process_data(self):
        img2label, chars, all_labels = process_data(self.config['path_train_dir'], self.config['path_train_labels'])
        self.assertIsInstance(img2label, dict)
        self.assertIsInstance(chars, list)
        self.assertIsInstance(all_labels, list)

    def test_process_image(self):
        img_path = os.path.join(self.config['path_train_dir'], os.listdir(self.config['path_train_dir'])[0])  # get the first image
        img = cv2.imread(img_path)
        processed_img = process_image(img, self.hp)
        self.assertIsInstance(processed_img, np.ndarray)
        self.assertEqual(processed_img.shape, (self.config['height'], self.config['width'], 3))

    def test_generate_data(self):
        img_paths = [os.path.join(self.config['path_train_dir'], img) for img in os.listdir(self.config['path_train_dir'])[:10]]  # get the first 10 images
        data_images = generate_data(img_paths, self.hp)
        self.assertIsInstance(data_images, list)
        self.assertEqual(len(data_images), len(img_paths))

    def test_train_valid_split(self):
        img2label, _, _ = process_data(self.config['path_train_dir'], self.config['path_train_labels'])
        imgs_val, labels_val, imgs_train, labels_train = train_valid_split(img2label)
        self.assertIsInstance(imgs_val, list)
        self.assertIsInstance(labels_val, list)
        self.assertIsInstance(imgs_train, list)
        self.assertIsInstance(labels_train, list)

    def test_get_mixed_data(self):
        img2label = get_mixed_data(self.config['path_train_dir'], self.config['path_train_labels'], self.config['path_train_dir'], self.config['path_train_labels'])
        self.assertIsInstance(img2label, dict)

    def test_get_batch(self):
        img2label, _, _ = process_data(self.config['path_train_dir'], self.config['path_train_labels'])
        img_paths = list(img2label.keys())[:10]  # get the first 10 images
        labels = list(img2label.values())[:10]  # get the corresponding labels
        batch_images, batch_labels = get_batch(img_paths, labels, 5, self.hp)
        self.assertIsInstance(batch_images, list)
        self.assertIsInstance(batch_labels, list)
        self.assertEqual(len(batch_images), 5)
        self.assertEqual(len(batch_labels), 5)


if __name__ == '__main__':
    unittest.main()
