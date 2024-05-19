import unittest
import json
import os
from src.config import Hparams


class TestHparams(unittest.TestCase):

    def setUp(self):
        self.config_path = 'configs/config.json'
        self.hparams = Hparams(self.config_path)

    def test_hparams_initialization(self):
        # Check if the Hparams object was initialized correctly
        self.assertIsInstance(self.hparams, Hparams)

    def test_config_loading(self):
        # Check if the configuration file was loaded correctly
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        hparams_dict = vars(self.hparams).copy()
        hparams_dict.pop("cyrillic")  # Don't compare the 'cyrillic' attribute
        self.assertDictEqual(hparams_dict, config)

    def test_paths(self):
        # Check if the paths in the configuration file exist
        self.assertTrue(os.path.exists(self.hparams.path_test_dir))
        self.assertTrue(os.path.exists(self.hparams.path_test_labels))
        self.assertTrue(os.path.exists(self.hparams.path_train_dir))
        self.assertTrue(os.path.exists(self.hparams.path_train_labels))


if __name__ == '__main__':
    unittest.main()
