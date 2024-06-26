import unittest
import Levenshtein as lev

from src.utils.text_utils import labels_to_text, text_to_labels, char_error_rate, word_error_rate, levenshtein


class TestTextUtils(unittest.TestCase):

    def setUp(self):
        # Define character to index and index to character mappings
        self.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й',
                    'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а',
                    'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
                    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        self.char2idx = {char: idx for idx, char in enumerate(self.cyrillic)}
        self.idx2char = {idx: char for idx, char in enumerate(self.cyrillic)}

        # For the purpose of this example, let's use a subset of Cyrillic characters
        self.truth = 'АБВГДЕЖЗ'
        self.pred = 'ЗЖЕДВБА'

    def test_labels_to_text(self):
        indices = [1, 29, 30, 31, 32, 33, 34, 35, 36, 2]
        expected_output = 'АБВГДЕЖЗ'
        output = labels_to_text(indices, self.idx2char)
        self.assertEqual(output, expected_output)

    def test_text_to_labels(self):
        text = 'АБВГДЕЖЗ'
        expected_output = [1, 29, 30, 31, 32, 33, 34, 35, 36, 91]
        output = text_to_labels(text, self.char2idx)
        self.assertEqual(output, expected_output)

    def test_char_error_rate(self):
        # Use a known example where the CER is easy to calculate
        truth = 'АБВГДЕЖЗ'
        pred = 'АБВГДЕЖ'  # One character missing at the end
        expected_cer = 1 / len(truth)  # CER should be 1/length of truth
        cer = char_error_rate(truth, pred)
        self.assertEqual(cer, expected_cer)

    def test_word_error_rate_with_multiple_words(self):
        truth = 'АБ ВГ ДЕ ЖЗ'
        pred = 'ЗЖ ЕД ВБ А'
        truth_words = truth.split()
        pred_words = pred.split()
        expected_wer = levenshtein(truth_words, pred_words) / len(truth_words)
        wer = word_error_rate(truth, pred)
        self.assertEqual(wer, expected_wer, f"Expected {expected_wer}, but got {wer}")

    def test_labels_to_text_with_special_characters(self):
        indices = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2]  # Corrected indices
        expected_output = '!"%(),-.'
        output = labels_to_text(indices, self.idx2char)
        self.assertEqual(output, expected_output)

    def test_text_to_labels_with_special_characters(self):
        text = '!\"%()-.,'
        expected_output = [1, 3, 4, 5, 6, 7, 9, 10, 8, 91]
        output = text_to_labels(text, self.char2idx)
        self.assertEqual(output, expected_output)

    def test_char_error_rate_with_different_lengths(self):
        truth = 'АБВГДЕЖЗ'
        pred = 'ЗЖЕДВ'
        expected_cer = lev.distance(truth, pred) / len(truth)
        cer = char_error_rate(truth, pred)
        self.assertEqual(cer, expected_cer)


if __name__ == '__main__':
    unittest.main()
