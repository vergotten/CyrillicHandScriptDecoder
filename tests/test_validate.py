import unittest
from torch.utils.data import DataLoader
import torch
import os

from src.utils.data_processing import process_data, train_valid_split, generate_data
from src.utils.collate import TextCollate
from src.utils.dataset import TextLoader
from src.config import Hparams
from src.utils.text_utils import labels_to_text, text_to_labels
from src.model import TransformerModel
from src.utils.model_utils import validate


class TestValidate(unittest.TestCase):
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

        img2label, _, all_words = process_data(cls.config.path_train_dir, cls.config.path_train_labels)
        X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)
        X_train = generate_data(X_train, cls.config)
        X_val = generate_data(X_val, cls.config)

        # Convert the labels to a list
        cls.labels = y_val

        # Convert the image names to a list with full path
        cls.images_name = X_val

        val_dataset = TextLoader(X_val, y_val, cls.config.char2idx, cls.config.idx2char, cls.config)
        cls.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=cls.config.batch_size, pin_memory=False,
                                    drop_last=False, collate_fn=TextCollate())

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device based on availability of CUDA
        cls.model = TransformerModel('resnet50', len(cls.config.cyrillic), hidden=cls.config.hidden,
                                     enc_layers=cls.config.enc_layers, dec_layers=cls.config.dec_layers,
                                     nhead=cls.config.nhead, dropout=cls.config.dropout, pretrained=False)
        cls.model.to(cls.device)  # Move the model to the GPU if available

        # Load the model weights
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    cls.config.weights_path)
        state_dict = torch.load(weights_path, map_location=cls.device)
        cls.model.load_state_dict(state_dict['model'])

    def test_label_conversion(self):
        # Create a TextLoader dataset
        dataset = TextLoader(self.images_name, self.labels, self.config.char2idx, self.config.idx2char)

        # Iterate over the items in the dataset
        for i in range(len(dataset)):
            # Get the data and target for this item
            data, target = dataset[i]

            # Convert the target to a string of numbers
            target_str = labels_to_text(target, self.config.idx2char)

            # Convert the target string to labels
            label_conversion = text_to_labels(target_str, self.config.char2idx)

            # Print the original text and its corresponding digit representation
            print(f"Original text: {target_str}, Digit representation: {label_conversion}")

    def test_validate(self):
        eval_cer, eval_wer = validate(self.model, self.val_loader, self.device)

        # Print the evaluation loss, CER, and WER for debugging
        # print(f"Evaluation loss: {eval_loss}")
        print(f"Evaluation CER: {eval_cer}")
        print(f"Evaluation WER: {eval_wer}")

        # Check that the output is a float
        self.assertIsInstance(eval_loss, float, "Evaluation loss is not a float.")
        self.assertIsInstance(eval_cer, float, "Evaluation CER is not a float.")
        self.assertIsInstance(eval_wer, float, "Evaluation WER is not a float.")

        # Check that the output is not negative
        self.assertGreaterEqual(eval_loss, 0, "Evaluation loss is negative.")
        self.assertGreaterEqual(eval_cer, 0, "Evaluation CER is negative.")
        self.assertGreaterEqual(eval_wer, 0, "Evaluation WER is negative.")

        # You can also add tests to check if the values are within an expected range
        # For example, if you expect the values to be between 0 and 1, you can do:
        self.assertLessEqual(eval_loss, 1, "Evaluation loss is greater than 1.")
        self.assertLessEqual(eval_cer, 1, "Evaluation CER is greater than 1.")
        self.assertLessEqual(eval_wer, 1, "Evaluation WER is greater than 1.")


if __name__ == '__main__':
    unittest.main()
