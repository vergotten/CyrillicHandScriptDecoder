import torch
import Levenshtein as lev


def labels_to_text(s, idx2char):
    """
    Translate indices to text.

    Args:
        s (list): List of indices.
        idx2char (dict): Map from indices to chars.

    Returns:
        str: Translated string.
    """
    # Convert indices to characters, ignoring 'SOS', 'EOS', and 'PAD'
    S = "".join([idx2char[j.item() if isinstance(j, torch.Tensor) else j] if idx2char[j.item() if isinstance(j, torch.Tensor) else j] not in ['SOS', 'EOS', 'PAD'] else '' for i in s for j in (i.flatten() if isinstance(i, torch.Tensor) else [i])])

    return S.strip()


def text_to_labels(s, char2idx):
    """
    Convert text to array of indices.

    Args:
        s (str): Input string.
        char2idx (dict): Map from chars to indices.

    Returns:
        list: List of indices.
    """
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]


def levenshtein(seq1, seq2):
    """
    Compute the Levenshtein distance between two sequences.

    Args:
        seq1 : str or list of str
        seq2 : str or list of str

    Returns:
        int: The Levenshtein distance between seq1 and seq2.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    return matrix[size_x - 1][size_y - 1]


def char_error_rate(truth, pred):
    """
    Compute the Character Error Rate (CER).

    Args:
        truth (str): The ground truth string.
        pred (str): The predicted string.

    Returns:
        float: The CER.
    """
    # Compute the Levenshtein distance between the truth and prediction
    dist = levenshtein(truth, pred)
    # Compute the length of the truth string
    length = len(truth)
    # Compute the CER
    cer = dist / length
    return cer


def word_error_rate(truth, pred):
    """
    Compute the Word Error Rate (WER).

    Args:
        truth (str): The ground truth string.
        pred (str): The predicted string.

    Returns:
        float: The WER.
    """
    # Split the truth and prediction into words
    truth_words = truth.split()
    pred_words = pred.split()

    # Compute the Levenshtein distance between the truth and prediction
    dist = levenshtein(truth_words, pred_words)

    # Compute the length of the truth words
    length = len(truth_words)

    # Compute the WER
    wer = dist / length

    return wer


if __name__ == "__main__":
    # Define character to index and index to character mappings
    cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
                'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г',
                'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш',
                'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
    char2idx = {char: idx for idx, char in enumerate(cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(cyrillic)}

    # Test labels_to_text with special characters
    indices = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2]  # Corrected indices
    expected_output = '!"%(),-.'
    output = labels_to_text(indices, idx2char)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test text_to_labels with special characters
    text = '!\"%()-.,'
    expected_output = [1, 3, 4, 5, 6, 7, 9, 10, 8, 91]  # Corrected expected output
    output = text_to_labels(text, char2idx)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test WER with multiple words
    truth = 'АБ ВГ ДЕ ЖЗ'
    pred = 'ЗЖ ЕД ВБ А'
    truth_words = truth.split()
    pred_words = pred.split()
    expected_wer = levenshtein(truth.split(), pred.split()) / len(truth.split())
    wer = word_error_rate(truth, pred)
    assert wer == expected_wer, f"Expected {expected_wer}, but got {wer}"
