import json
import argparse
import os


class Hparams():
    """
    Hyperparameters class.

    This class holds all the hyperparameters for the model. If a configuration file path is provided, it will load the configuration from the file. If the file doesn't exist, it will save the default configuration to the file.

    Attributes:
        cyrillic (list): List of Cyrillic characters.
        del_sym (list): List of characters to delete.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        hidden (int): Number of hidden units.
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        nhead (int): Number of heads in the multihead attention models.
        dropout (float): Dropout rate.
        width (int): Width of the image.
        height (int): Height of the image.
        path_test_dir (str): Path to the test utils directory.
        path_test_labels (str): Path to the test labels file.
        path_train_dir (str): Path to the train utils directory.
        path_train_labels (str): Path to the train labels file.
    """
    def __init__(self, config_path=None):
        """
        Initialize hyperparameters.

        Args:
            config_path (str, optional): Path to the JSON configuration file.
        """
        # Default values
        self.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        self.del_sym = []
        self.lr = 0.01
        self.batch_size = 1
        self.hidden = 512
        self.enc_layers = 2
        self.dec_layers = 2
        self.nhead = 4
        self.dropout = 0.0
        self.width = 256
        self.height = 64
        self.path_test_dir = "data/test/"
        self.path_test_labels = "data/test.tsv"
        self.path_train_dir = "data/train/"
        self.path_train_labels = "data/train.tsv"

        # Load values from JSON file if provided
        if config_path is not None:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                config = vars(self).copy()
                config.pop("cyrillic")  # Don't save the 'cyrillic' attribute
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)  # Pretty-print the JSON file
            else:
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    args = parser.parse_args()

    hp = Hparams(args.config)
    print(vars(hp))
