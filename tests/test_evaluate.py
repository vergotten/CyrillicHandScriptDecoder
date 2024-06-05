import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from src.model import TransformerModel
from src.utils.model_utils import evaluate
from src.utils.data_processing import process_data, train_valid_split, generate_data
from src.utils.collate import TextCollate
from src.utils.dataset import TextLoader
from src.config import Hparams


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Hparams("configs/config.json")

        # Define character to index and index to character mappings
        cls.config.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4',
                               '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е',
                               'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч',
                               'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
                               'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю',
                               'я', 'ё', 'EOS']
        cls.config.char2idx = {char: idx for idx, char in enumerate(cls.config.cyrillic)}
        cls.config.idx2char = {idx: char for idx, char in enumerate(cls.config.cyrillic)}

        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Define the device based on availability of CUDA
        cls.model = TransformerModel('resnet50', len(cls.config.cyrillic), hidden=cls.config.hidden,
                                     enc_layers=cls.config.enc_layers, dec_layers=cls.config.dec_layers,
                                     nhead=cls.config.nhead, dropout=cls.config.dropout, pretrained=False)
        cls.model.to(cls.device)  # Move the model to the GPU if available

        # Load the model weights
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    cls.config.weights_path)
        state_dict = torch.load(weights_path, map_location=cls.device)
        cls.model.load_state_dict(state_dict['model'])

        cls.criterion = nn.CrossEntropyLoss(ignore_index=cls.config.char2idx['PAD'])

        img2label, _, all_words = process_data(cls.config.path_train_dir, cls.config.path_train_labels)
        X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)
        X_train = generate_data(X_train, cls.config)
        X_val = generate_data(X_val, cls.config)

        val_dataset = TextLoader(X_val, y_val, cls.config.char2idx, cls.config.idx2char, cls.config)
        cls.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=cls.config.batch_size, pin_memory=False,
                                    drop_last=False, collate_fn=TextCollate())

    def test_evaluate(self):
        eval_loss = evaluate(self.model, self.criterion, self.val_loader)

        # Print the evaluation loss for debugging
        print(f"Evaluation loss: {eval_loss}")

        # Check that the output is a float
        self.assertIsInstance(eval_loss, float, "Evaluation loss is not a float.")

        # Check that the output is not negative
        self.assertGreaterEqual(eval_loss, 0, "Evaluation loss is negative.")

        # You can also add tests to check if the loss is within an expected range
        # For example, if you expect the loss to be between 0 and 10, you can do:
        self.assertLessEqual(eval_loss, 10, "Evaluation loss is greater than 10.")


if __name__ == '__main__':
    unittest.main()
