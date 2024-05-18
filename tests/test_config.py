import unittest
import json
import os


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Load the configuration file
        with open('configs/config.json', 'r') as f:
            self.config = json.load(f)

    def test_config_loading(self):
        # Check if the configuration file was loaded correctly
        self.assertIsInstance(self.config, dict)

    def test_paths(self):
        # Check if the paths in the configuration file exist
        self.assertTrue(os.path.exists(self.config['path_test_dir']))
        self.assertTrue(os.path.exists(self.config['path_test_labels']))
        self.assertTrue(os.path.exists(self.config['path_train_dir']))
        self.assertTrue(os.path.exists(self.config['path_train_labels']))


if __name__ == '__main__':
    unittest.main()
