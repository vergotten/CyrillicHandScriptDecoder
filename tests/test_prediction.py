import unittest
import torch
import os

from src.model import TransformerModel
from src.utils.data_processing import process_data, train_valid_split, generate_data
from src.config import Hparams
from src.utils.model_utils import prediction


class TestPrediction(unittest.TestCase):
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

    def test_prediction(self):
        # Define a test directory with images
        test_dir = 'path_to_your_test_images_directory/'

        # Perform prediction
        preds = prediction(self.model, test_dir, self.config.char2idx, self.config.idx2char, self.device)

        # Check that the output is a dictionary
        self.assertIsInstance(preds, dict)

        # Check that the dictionary is not empty
        self.assertGreater(len(preds), 0)

        # Check that each entry in the dictionary has the correct format
        for filename, pred in preds.items():
            self.assertIsInstance(filename, str)
            self.assertIsInstance(pred, dict)
            self.assertIn('predicted_label', pred)
            self.assertIn('p_values', pred)
            self.assertIsInstance(pred['predicted_label'], str)
            self.assertIsInstance(pred['p_values'], float)


if __name__ == '__main__':
    unittest.main()
