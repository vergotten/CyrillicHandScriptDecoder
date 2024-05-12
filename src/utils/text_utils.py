import editdistance
import torch

def labels_to_text(s, idx2char):
    """
    Translate indices to text.

    Args:
        s (list): List of indices.
        idx2char (dict): Map from indices to chars.

    Returns:
        str: Translated string.
    """
    S = "".join([idx2char[i.item()] if isinstance(i, torch.Tensor) else idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


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


def char_error_rate(p_seq1, p_seq2):
    """
    Compute character error rate.

    Args:
        p_seq1 (str): First string.
        p_seq2 (str): Second string.

    Returns:
        float: Character error rate.
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))
