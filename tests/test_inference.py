import unittest
import torch
from torch.utils.data import DataLoader
import cv2
import os

from src.inference import inference
from src.model import TransformerModel
from src.utils.model_utils import validate
from src.utils.data_processing import process_data, train_valid_split, generate_data
from src.utils.collate import TextCollate
from src.utils.dataset import TextLoader
from src.config import Hparams


class TestInferenceFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Hparams("configs/config.json")
        cls.config.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
                         '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
                         'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я',
                         'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
                         'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        cls.config.char2idx = {char: idx for idx, char in enumerate(cls.config.cyrillic)}
        cls.config.idx2char = {idx: char for idx, char in enumerate(cls.config.cyrillic)}

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = TransformerModel('resnet50', len(cls.config.cyrillic), hidden=cls.config.hidden,
                                      enc_layers=cls.config.enc_layers, dec_layers=cls.config.dec_layers,
                                      nhead=cls.config.nhead, dropout=cls.config.dropout, pretrained=False)
        cls.model.to(cls.device)

        # Load the model weights
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    cls.config.weights_path)
        state_dict = torch.load(weights_path, map_location=cls.device)
        cls.model.load_state_dict(state_dict['model'])

    def test_inference(self):
        # Load a test image
        image = cv2.imread('path_to_your_test_image.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform inference
        predicted_transcript = inference(self.model, image, self.config.char2idx, self.config.idx2char, self.config)

        # Check that the output is a string
        self.assertIsInstance(predicted_transcript, str)

        # Check that the output is not empty
        self.assertGreater(len(predicted_transcript), 0)


if __name__ == '__main__':
    unittest.main()
