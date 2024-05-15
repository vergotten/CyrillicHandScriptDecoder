import cv2
import numpy as np
from tqdm import tqdm
import random
import os
import logging


logging.basicConfig(filename='image_processing.log', level=logging.ERROR)


def process_data(image_dir, labels_dir, ignore=[]):
    """
    Convert images and labels into defined data structures.

    Args:
        image_dir (str): Path to directory with images.
        labels_dir (str): Path to tsv file with labels.
        ignore (list, optional): List of items to ignore. Defaults to [].

    Returns:
        img2label (dict): Keys are names of images and values are correspondent labels.
        chars (list): All unique chars used in data.
        all_labels (list): List of all labels.
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
                img2label[os.path.join(image_dir, x[0])] = x[1]
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


def process_image(img, hp):
    """
    Resize and normalize image.

    Args:
        img (np.array): Input image.

    Returns:
        img (np.array): Processed image.
    """
    # Check if image is None
    if img is None:
        print("Warning: Image is None, skipping processing.")
        return None

    w, h, _ = img.shape

    # Check if width or height is zero
    if w == 0 or h == 0:
        print("Warning: Image has zero width or height, skipping processing.")
        return img

    new_w = hp.height # 64
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape
    img = img.astype('float32')
    new_h = hp.width # 256
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


def generate_data(img_paths, hp):
    """
    Generate images from folder.

    Args:
        img_paths (list of str): Paths to images.

    Yields:
        np.array: An image in np.array format.
    """
    for path in tqdm(img_paths):
        # Ensure path is a string
        if not isinstance(path, str):
            path = str(path)

        # Check if image file exists
        if not os.path.isfile(path):
            print(f"Image file does not exist: {path}")
            continue

        img = cv2.imread(path)
        try:
            img = process_image(img, hp)

            # Check if processed image is None
            if img is None:
                print(f"Processed image is None: {path}")
                continue

        except Exception as e:
            print(f"Error processing image {path}: {e}")
            logging.error(f"Error processing image {path}: {e}")
            continue

        yield img


def train_valid_split(img2label, train_part=0.1, val_part=0.3):
    """
    Split dataset into train and valid parts.

    Args:
        img2label (dict): Keys are paths to images, values are labels (transcripts of crops).
        train_part (float, optional): Training part. Defaults to 0.1.
        val_part (float, optional): Validation part. Defaults to 0.3.

    Returns:
        imgs_val (list of str): Paths for validation.
        labels_val (list of str): Labels for validation.
        imgs_train (list of str): Paths for training.
        labels_train (list of str): Labels for training.
    """
    # Check if train_part and val_part sum to more than 1.0
    if train_part + val_part > 1.0:
        print("Error: train_part and val_part sum to more than 1.0")
        return None, None, None, None

    imgs_val, labels_val = [], []
    imgs_train, labels_train = [], []

    N_train = int(len(img2label) * train_part)
    N_val = int(len(img2label) * val_part)
    items = list(img2label.items())
    random.shuffle(items)
    for i, item in enumerate(items):
        if i < N_train:
            imgs_train.append(item[0])
            labels_train.append(item[1])
        elif i < N_train + N_val:
            imgs_val.append(item[0])
            labels_val.append(item[1])
        else:
            break  # Ignore the rest of the data

    print('train part:{}'.format(len(imgs_train)))
    print('valid part:{}'.format(len(imgs_val)))
    return imgs_val, labels_val, imgs_train, labels_train


def get_mixed_data(pretrain_image_dir, pretrain_labels_dir, train_image_dir, train_labels_dir, pretrain_part=0.0):
    """
    Prepare dataset from training. It creates mixed dataset: the first part comes from real data and the second part comes from generator.

    Args:
        pretrain_image_dir (str): Path to pretrain image directory.
        pretrain_labels_dir (str): Path to pretrain labels directory.
        train_image_dir (str): Path to train image directory.
        train_labels_dir (str): Path to train labels directory.
        pretrain_part (float, optional): Pretrain part. Defaults to 0.0.

    Returns:
        img2label2 (dict): Keys are paths to images, values are labels (transcripts of crops).
    """
    img2label1, chars1, all_words1 = process_data(pretrain_image_dir, pretrain_labels_dir)  # PRETRAIN PART
    img2label2, chars2, all_words2 = process_data(train_image_dir, train_labels_dir)  # TRAIN PART
    img2label1_list = list(img2label1.items())
    N = len(img2label1_list)
    for i in range(N):
        j = np.random.randint(0, N)
        item = img2label1_list[j]
        img2label2[item[0]] = item[1]
    return img2label2


def get_batch(img_paths, labels, batch_size, hp):
    """
    Collect images into batches.

    Args:
        img_paths (list of str): Paths to images.
        labels (list of str): Corresponding labels of images.
        batch_size (int): Size of the batch.

    Returns:
        list: Batch of images.
        list: Batch of labels.
    """
    generator = generate_data(img_paths, hp)
    batch_images = [next(generator) for _ in range(batch_size)]
    batch_labels = labels[:batch_size]
    return batch_images, batch_labels
