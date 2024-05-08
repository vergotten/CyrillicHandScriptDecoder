import os
import cv2
import numpy as np
import torch
import argparse
from torchvision import transforms
from utils.data_processing import text_to_labels, process_image
from utils.text_utils import labels_to_text
from data.augmentations import Vignetting, UniformNoise, LensDistortion
from model import TransformerModel  # replace with your actual import statement
from config import Hparams
import traceback

# Add the new imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import keras_ocr


def draw_annotations(image, bboxes, labels):
    """
    Draw bounding boxes and labels on the image.

    Parameters:
    image (np.array): The image to draw annotations on.
    bboxes (list): List of bounding boxes.
    labels (list): List of labels corresponding to the bounding boxes.

    Returns:
    fig: A Matplotlib figure with the annotations drawn.
    """
    # Convert bounding boxes to the correct format
    bboxes = [np.array(bbox).astype('float32') for bbox in bboxes]

    # Create a list of predictions where each prediction is a tuple of a word and its box
    predictions = list(zip(labels, bboxes))

    # Use Keras OCR's drawAnnotations function to draw the predictions
    fig = keras_ocr.tools.drawAnnotations(image=image, predictions=predictions)

    return fig


def inference(model, image_input, char2idx, idx2char):
    """
    Perform inference on an image.

    Parameters:
    model (torch.nn.Module): The PyTorch model to use for inference.
    image_input (str): Path to the image file.
    char2idx (dict): Dictionary mapping characters to indices.
    idx2char (dict): Dictionary mapping indices to characters.

    Returns:
    predicted_transcript (str): The predicted transcript of the image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the image
    img = cv2.imread(image_input)
    img = process_image(img).astype('uint8')
    img = img / img.max()
    img = np.transpose(img, (2, 0, 1))
    src = torch.FloatTensor(img).unsqueeze(0).to(device)

    # Make prediction
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


def main():
    """
    Main function to perform OCR inference on images in a directory.
    """
    parser = argparse.ArgumentParser(description='OCR Inference')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--weights', type=str, default='ocr_transformer_rn50_64x256_53str_jit.pt', help='Path to the weights file')
    parser.add_argument('--input_dir', type=str, default='demo/input', help='Directory of input images')
    parser.add_argument('--output_dir', type=str, default='demo/output', help='File to output the results')

    args = parser.parse_args()

    # Convert relative paths to absolute paths
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.weights = os.path.abspath(args.weights)

    hp = Hparams(args.config)
    print(vars(hp))

    # Print the absolute path of the weights file
    print("Weights file path:", os.path.abspath(args.weights))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers,
                             nhead=hp.nhead, dropout=hp.dropout).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])

    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    # Perform inference on each image in the input directory
    for image_path in os.listdir(args.input_dir):
        # Check if the file is an image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_input = os.path.join(args.input_dir, image_path)
                predicted_transcript = inference(model, image_input, char2idx, idx2char)

                # Add the new code here
                # Load the image
                img = mpimg.imread(image_input)

                # Display the image
                plt.imshow(img)
                plt.show()

                detected_bboxes = reader.readtext(image_input)

                df = pd.DataFrame(detected_bboxes, columns=['bbox','text','conf'])
                df['bbox'] = df['bbox'].apply(lambda bbox: [[int(coordinate) for coordinate in point] for point in bbox])
                bboxes_series = df['bbox']
                bboxes_series

                image = load_image(image_input)
                detected_bboxes.sort(key=lambda bbox: (np.mean([pt[1] for pt in bbox[0]]), np.mean([pt[0] for pt in bbox[0]])))
                image_with_bboxes, all_bboxes, predicted_transcripts = draw_bboxes(image, detected_bboxes, model, char2idx, idx2char)
                display_image(image_with_bboxes)

                image_path = "/content/rukopi3.png"
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                fig = draw_annotations(image, all_bboxes, predicted_transcripts)

                # Save the output figure to the current directory with the same name
                output_dir = "./demo/output/"
                os.makedirs(output_dir, exist_ok=True)

                # Get the base name of the image path and add '_output' before the extension
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_name = f"{name}_output{ext}"

                plt.savefig(os.path.join(output_dir, output_name), bbox_inches='tight', pad_inches=0)

                # Display the annotated image
                plt.show()

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
