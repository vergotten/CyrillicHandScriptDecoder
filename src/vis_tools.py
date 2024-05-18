import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from utils.data_processing import process_data, train_valid_split
from config import Hparams
from utils.collate import TextCollate
from utils.dataset import TextLoader
from utils.text_utils import labels_to_text


def visualize_batch(data_loader, idx2char, num_images=10):
    """
    Display a batch of images with their labels.

    Args:
        data_loader (DataLoader): PyTorch DataLoader.
        idx2char (dict): Mapping from indices to characters.
        num_images (int): Number of images to display.
    """

    first_batch_images, first_batch_labels = next(iter(data_loader))

    # Calculate the number of rows and columns for the subplot grid
    columns = int(np.sqrt(num_images)) + 1
    rows = int(np.ceil(num_images / columns)) + 1

    # Create a figure for the entire grid
    fig = plt.figure(figsize=(columns * 2, rows * 2))

    # Iterate over each label in the batch
    for i in range(first_batch_labels.size(1)):
        # Get the label for the i-th sample in the batch
        label = first_batch_labels[:, i]

        # Remove trailing zeros (padding)
        label = label[label != 0]

        # Convert the label to text and print it
        print(f"Label {i}: {labels_to_text(label.tolist(), idx2char)}")

        # Visualize the i-th image in the batch
        image = first_batch_images[i].numpy().transpose((1, 2, 0))  # Convert image from PyTorch tensor to numpy array
        # Check if the image is not already normalized
        if image.max() > 1.0:
            # Divide by 255 to get pixel values between 0 and 1
            image = image / 255.0

        # Add the image to the subplot
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.title(f"{labels_to_text(label.tolist(), idx2char)}")
        plt.axis('on')  # to hide the axis

    # Show the entire grid
    plt.tight_layout(pad=0.1)  # Adjust the layout to minimize overlap and reduce padding
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Training')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--train_data', type=str, default='data/train/', help='Path to training utils')
    parser.add_argument('--test_data', type=str, default='data/test/', help='Path to testing utils')
    parser.add_argument('--train_labels', type=str, default='data/train.tsv', help='Path to training labels')
    parser.add_argument('--test_labels', type=str, default='data/test.tsv', help='Path to testing labels')

    args = parser.parse_args()

    args.train_data = os.path.abspath(args.train_data)
    args.test_data = os.path.abspath(args.test_data)
    args.train_labels = os.path.abspath(args.train_labels)
    args.test_labels = os.path.abspath(args.test_labels)

    hp = Hparams(args.config)
    print(vars(hp))

    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    img2label, _, all_words = process_data(args.train_data, args.train_labels)

    X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)

    train_dataset = TextLoader(X_train, y_train, char2idx, idx2char, hp)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False,
                                               batch_size=hp.batch_size, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())
    val_dataset = TextLoader(X_val, y_val, char2idx, idx2char, hp)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                             batch_size=hp.batch_size, pin_memory=False,
                                             drop_last=False, collate_fn=TextCollate())

    visualize_batch(train_loader, idx2char, num_images=hp.batch_size)
