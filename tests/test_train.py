import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import TransformerModel
from src.utils.model_utils import train, train_all
from src.utils.data_processing import process_data, train_valid_split, generate_data
from src.utils.collate import TextCollate
from src.utils.dataset import TextLoader
from src.config import Hparams


class TestTrainFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Hparams("configs/config.json")

        # Define character to index and index to character mappings
        cls.config.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
                         '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
                         'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я',
                         'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
                         'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        cls.config.char2idx = {char: idx for idx, char in enumerate(cls.config.cyrillic)}
        cls.config.idx2char = {idx: char for idx, char in enumerate(cls.config.cyrillic)}

        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Define the device based on availability of CUDA
        cls.model = TransformerModel('resnet50', len(cls.config.cyrillic), hidden=cls.config.hidden,
                                      enc_layers=cls.config.enc_layers, dec_layers=cls.config.dec_layers,
                                      nhead=cls.config.nhead, dropout=cls.config.dropout, pretrained=False)
        cls.model.to(cls.device)  # Move the model to the GPU if available
        cls.optimizer = SGD(cls.model.parameters(), lr=cls.config.lr)
        cls.criterion = nn.CrossEntropyLoss(ignore_index=cls.config.char2idx['PAD'])
        cls.scheduler = ReduceLROnPlateau(cls.optimizer, 'min')

        img2label, _, all_words = process_data(cls.config.path_train_dir, cls.config.path_train_labels)
        X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)
        X_train = generate_data(X_train, cls.config)
        X_val = generate_data(X_val, cls.config)

        train_dataset = TextLoader(X_train, y_train, cls.config.char2idx, cls.config.idx2char, cls.config)
        cls.train_loader = DataLoader(train_dataset, shuffle=False, batch_size=cls.config.batch_size, pin_memory=True, drop_last=True, collate_fn=TextCollate())
        val_dataset = TextLoader(X_val, y_val, cls.config.char2idx, cls.config.idx2char, cls.config)
        cls.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=cls.config.batch_size, pin_memory=False, drop_last=False, collate_fn=TextCollate())

    def test_train(self):
        loss = train(self.model, self.optimizer, self.criterion, self.train_loader)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_train_all(self):
        params_before = [p.clone() for p in self.model.parameters()]
        train_all(self.model, self.optimizer, self.config.idx2char, self.criterion, self.scheduler, self.train_loader, self.val_loader, epoch_limit=2)
        params_after = [p for p in self.model.parameters()]
        for p_before, p_after in zip(params_before, params_after):
            self.assertTrue(torch.any(p_before != p_after), "Model parameters did not update during training.")


if __name__ == '__main__':
    unittest.main()
