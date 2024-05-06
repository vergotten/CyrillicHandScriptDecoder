import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from utils import process_image, labels_to_text  # Assuming these functions are in __utils.py
from models import TransformerModel  # Assuming your model is defined in models.py
from transforms import Vignetting, LensDistortion, UniformNoise  # Assuming these classes are in transforms.py
import Augmentor
import keras_ocr
import os
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def inference(model, image_input, char2idx, idx2char):
    """
    params
    ---
    model : pytorch model
    image_path : str
        path to the image
    char2idx : dict
    idx2char : dict
    returns
    ---
    predicted_transcript : str
    """
    # Load and preprocess the image

    # Check if image_input is a numpy array
    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        # Load and preprocess the image
        img = cv2.imread(image_input)

    img = process_image(img).astype('uint8')
    img = img / img.max()
    img = np.transpose(img, (2, 0, 1))
    src = torch.FloatTensor(img).unsqueeze(0).to(device)

    # Make prediction
    p_values = 1
    out_indexes = [char2idx['SOS'], ]

    for i in range(100):
        trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
        output = model(src,trg_tensor)
        out_token = output.argmax(2)[-1].item()
        out_indexes.append(out_token)
        if out_token == char2idx['EOS']:
            break

    # Convert prediction to transcript
    predicted_transcript = labels_to_text(out_indexes[1:], idx2char)

    return predicted_transcript


def draw_annotations(image, bboxes, labels):
    # Convert bounding boxes to the correct format
    bboxes = [np.array(bbox).astype('float32') for bbox in bboxes]

    # Create a list of predictions where each prediction is a tuple of a word and its box
    predictions = list(zip(labels, bboxes))

    # Use Keras OCR's drawAnnotations function to draw the predictions
    fig = keras_ocr.tools.drawAnnotations(image=image, predictions=predictions)

    return fig


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_bboxes(image, detected_bboxes, model, char2idx, idx2char):
    image_copy = image.copy()
    all_bboxes = []
    predicted_transcripts = []

    for detected_bbox in detected_bboxes:
        bbox, label, score = detected_bbox
        # Convert the bounding box to the format used by Keras OCR
        pts = [tuple(map(int, pt)) for pt in bbox]
        all_bboxes.append(pts)

        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image_copy, [pts], True, (0,255,0), 3)

        x_min, y_min = np.min(pts, axis=0)[0]
        x_max, y_max = np.max(pts, axis=0)[0]
        extracted_image = image[y_min:y_max, x_min:x_max]

        predicted_transcript = inference(model, extracted_image, char2idx, idx2char)
        predicted_transcripts.append(predicted_transcript)

        print(f"Predicted transcript: {predicted_transcript}")

    return image_copy, all_bboxes, predicted_transcripts


def display_image(image):
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Perform inference on an image.')
    parser.add_argument('--input', default='demo/input', help='Path to the input image.')
    parser.add_argument('--output', default='demo/output', help='Path to save the output image.')
    parser.add_argument('--vis_dump', default=True, type=bool, help='Whether to save visualizations.')
    args = parser.parse_args()

    # Load the image
    image_path = args.input
    image = load_image(image_path)

    # Initialize the reader and detect bounding boxes
    reader = Reader(['en'])  # Initialize with English language
    detected_bboxes = reader.readtext(image_path)

    # Load the model
    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers,
                             dec_layers=hp.dec_layers, nhead=hp.nhead, dropout=hp.dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Assuming 'bboxes' is your list of bounding boxes and 'labels' is your list of labels
    image_with_bboxes, all_bboxes, predicted_transcripts = draw_bboxes(image, detected_bboxes, model, char2idx,
                                                                       idx2char)

    # Display the annotated image
    display_image(image_with_bboxes)

    # Save the output figure to the current directory with the same name
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Get the base name of the image path and add '_output' before the extension
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_name = f"{name}_output{ext}"

    if args.vis_dump:
        plt.savefig(os.path.join(output_dir, output_name), bbox_inches='tight', pad_inches=0)
