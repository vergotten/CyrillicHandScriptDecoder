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


def inference(model, image_input, char2idx, idx2char):
    """
        params
        ---
        model : pytorch model
        image_input : str
            path to the image
        char2idx : dict
        idx2char : dict
        returns
        ---
        predicted_transcript : str
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
    # print(args.weights)

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

                print(predicted_transcript)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
