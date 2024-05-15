import matplotlib.pyplot as plt
import numpy as np


def display_images(images, labels, batch_size):
    """
    Display a batch of images with their labels.

    Args:
        images (list of np.array): Batch of images.
        labels (list of str): Corresponding labels of images.
        batch_size (int): Size of the batch.
    """
    # Calculate the number of rows and columns
    columns = int(np.sqrt(batch_size)) + 1
    rows = int(np.ceil(batch_size / columns)) + 1

    # Adjust the figure size based on the number of columns and rows
    fig = plt.figure(figsize=(columns * 2, rows * 2))

    for i in range(batch_size):
        ax = fig.add_subplot(rows, columns, i + 1)
        image = images[i]
        # Check if the image is not already normalized
        if image.max() > 1.0:
            # Divide by 255 to get pixel values between 0 and 1
            image = image / 255.0
        plt.imshow(image)
        # Display the label as the title of the subplot
        plt.title(f"{labels[i]}")
        plt.axis('off')  # to hide the axis
        # Reduce space between rows
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.5)  # Adjust the layout to minimize overlap and reduce padding
    plt.show()


def display_batch(data_loader, batch_size):
    """
    Display a batch of images with their labels.

    Args:
        data_loader (DataLoader): PyTorch DataLoader.
        batch_size (int): Size of the batch.
    """
    # Get a batch of data
    images, labels = next(iter(data_loader))

    # Calculate the number of rows and columns
    columns = int(np.sqrt(batch_size)) + 1
    rows = int(np.ceil(batch_size / columns)) + 1

    # Adjust the figure size based on the number of columns and rows
    fig = plt.figure(figsize=(columns * 2, rows * 2))

    for i in range(batch_size):
        ax = fig.add_subplot(rows, columns, i + 1)
        image = images[i].numpy().transpose((1, 2, 0))  # Convert the image from PyTorch tensor to numpy array
        # Check if the image is not already normalized
        if image.max() > 1.0:
            # Divide by 255 to get pixel values between 0 and 1
            image = image / 255.0
        plt.imshow(image)
        # Display the label as the title of the subplot
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')  # to hide the axis
        # Reduce space between rows
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.1)  # Adjust the layout to minimize overlap and reduce padding
    plt.show()


# Display a batch from the train_loader and val_loader
# display_batch(train_loader, batch_size=18)
# display_batch(val_loader, batch_size=18)
