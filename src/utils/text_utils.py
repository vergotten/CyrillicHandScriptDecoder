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
    S = "".join([idx2char[i.item() if isinstance(i, torch.Tensor) else i] if idx2char[i.item() if isinstance(i, torch.Tensor) else i] not in ['SOS', 'EOS', 'PAD'] else '' for i in s])
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
    dist = lev.distance(truth, pred)
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
    dist = lev.distance(' '.join(truth_words), ' '.join(pred_words))

    # Compute the length of the truth words
    length = len(truth_words)

    # Compute the WER
    wer = dist / length

    return wer
