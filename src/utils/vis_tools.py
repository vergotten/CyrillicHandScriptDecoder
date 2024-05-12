import matplotlib.pyplot as plt

from .text_utils import labels_to_text


def display_images(data_loader, hp):
    for idx, (img, label) in enumerate(data_loader):
        print(f"idx, (img, label): {idx, (img.shape, label.shape)}")
        batch_size = img.shape[0]; print(batch_size)  # Get the actual size of the current batch
        fig = plt.figure(figsize=(13, 13))
        rows = int(batch_size / 4) + 2
        columns = int(batch_size / 8) + 2
        # for j in range(batch_size):  # Loop over the images in the current batch
        #     print(f"img: {img[idx][j].shape} | label: {label[idx][j].shape}")
        #     fig.add_subplot(rows, columns, j + 1)
        #     image = img[j].permute(1, 2, 0)
        #     # Check if the image is not already normalized
        #     if image.max() > 1.0:
        #         # Divide by 255 to get pixel values between 0 and 1
        #         image = image / 255.0
        #     plt.imshow(image)
        #     # Decode the label and display it as the title of the subplot
        #     text_label = labels_to_text(labels[:, j].tolist(), data_loader.dataset.idx2char)
        #     plt.title(f"Label: {text_label}")
        #     plt.axis('on')  # to show the axis
        # plt.show()
        # Break after displaying hp.batch_size number of batches
        if idx + 1 == hp.batch_size:
            break
