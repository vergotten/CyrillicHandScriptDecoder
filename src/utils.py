import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm


def count_parameters(model):
    """
    Count the parameters in a model.

    Args:
        model (nn.Module): The model to count the parameters of.

    Returns:
        int: The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def text_to_labels(text, char2idx):
    """
    Convert text to labels.

    Args:
        text (str): The text to convert.
        char2idx (dict): Dictionary mapping characters to indices.

    Returns:
        list: The labels corresponding to the text.
    """
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]


def process_data(image_dir, labels_dir, ignore=[]):
    """
    Process data.

    Args:
        image_dir (str): Path to the directory with images.
        labels_dir (str): Path to the directory with labels.

    Returns:
        img2label (dict): Dictionary mapping image paths to labels.
        chars (list): List of characters.
        all_words (list): List of all words.
    """
    chars = []
    img2label = dict()

    raw = open(labels_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        try:
            x = t.split('\t')
            flag = False
            for item in ignore:
                if item in x[1]:
                    flag = True
            if flag == False:
                img2label[image_dir + x[0]] = x[1]
                for char in x[1]:
                    if char not in chars:
                        chars.append(char)
        except:
            print('ValueError:', x)
            pass

    all_labels = sorted(list(set(list(img2label.values()))))
    chars.sort()
    chars = ['PAD', 'SOS'] + chars + ['EOS']

    return img2label, chars, all_labels


class TextCollate():
    """
    Collate function for text data.

    This class should be used as the collate_fn argument in a DataLoader.
    """
    def __call__(self, batch):
        """
        Process a batch of data.

        Args:
            batch (list): A batch of data.

        Returns:
            tuple: A tuple of processed data and labels.
        """
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded


def labels_to_text(labels, idx2char):
    """
    Convert labels to text.

    Args:
        labels (list): The labels to convert.
        idx2char (dict): Dictionary mapping indices to characters.

    Returns:
        str: The text corresponding to the labels.
    """
    S = "".join([idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


def char_error_rate(preds, targets):
    """
    Calculate the character error rate.

    Args:
        preds (list): The predicted labels.
        targets (list): The target labels.

    Returns:
        float: The character error rate.
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))


def process_image(img, hp):
    """
    Process an image.

    Args:
        img (np.array): The image to process.

    Returns:
        np.array: The processed image.
    """
    w, h, _ = img.shape
    new_w = hp.height
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = hp.width
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


def generate_data(img_paths):
    """
    Generate data.

    Args:
        generator (callable): The generator function.
        num_samples (int): The number of samples to generate.

    Returns:
        list: The generated data.
    """
    data_images = []
    for path in tqdm(img_paths):
        # Ensure path is a string
        if not isinstance(path, str):
            path = str(path)
        img = cv2.imread(path)
        try:
            img = process_image(img)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue
        data_images.append(img)
    return data_images


def train(model, criterion, optimizer, iterator):
    """
    Train a model.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        iterator (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The average loss.
    """
    model.train()
    epoch_loss = 0
    counter = 0
    for src, trg in iterator:
        counter += 1
        if counter % 500 == 0:
            print('[', counter, '/', len(iterator), ']')
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_all(model, criterion, optimizer, train_iterator, valid_iterator, num_epochs):
    """
    Train a model for multiple epochs.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_iterator (torch.utils.data.DataLoader): The training data loader.
        valid_iterator (torch.utils.data.DataLoader): The validation data loader.
        num_epochs (int): The number of epochs to train for.

    Returns:
        None
    """
    pass


def validate(model, criterion, iterator):
    """
    Validate a model.

    Args:
        model (nn.Module): The model to validate.
        criterion (nn.Module): The loss function.
        iterator (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The average loss.
    """
    pass


def train_valid_split(img2label, val_part=0.3):
    """
    Split data into training and validation sets.

    Args:
        img2label (dict): Dictionary mapping image paths to labels.
        val_part (float): The fraction of data to use for validation.

    Returns:
        tuple: Four lists containing the validation images, validation labels, training images, and training labels.
    """
    pass


def evaluate(model, criterion, iterator):
    """
    Evaluate a model.

    Args:
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): The loss function.
        iterator (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The average loss.
    """
    pass


def get_mixed_data(pretrain_image_dir, pretrain_labels_dir, train_image_dir, train_labels_dir, pretrain_part=0.0):
    """
    Get mixed data for training.

    Args:
        pretrain_image_dir (str): Path to the directory with pretraining images.
        pretrain_labels_dir (str): Path to the directory with pretraining labels.
        train_image_dir (str): Path to the directory with training images.
        train_labels_dir (str): Path to the directory with training labels.
        pretrain_part (float): The fraction of data to use for pretraining.

    Returns:
        dict: Dictionary mapping image paths to labels.
    """
    pass


def prediction(model, test_dir, char2idx, idx2char):
    """
    Make predictions with a model.

    Args:
        model (nn.Module): The model to make predictions with.
        test_dir (str): Path to the directory with test images.
        char2idx (dict): Dictionary mapping characters to indices.
        idx2char (dict): Dictionary mapping indices to characters.

    Returns:
        dict: Dictionary with keys as image names and values as dictionaries with keys ['p_value', 'predicted_label'].
    """
    pass
