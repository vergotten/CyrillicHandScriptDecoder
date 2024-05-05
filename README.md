# CyrillicHandScriptDecoder

CyrillicHandScriptDecoder is a sophisticated machine learning project that employs a combination of ResNet, Transformer models, and CRAFT architecture to recognize and interpret Cyrillic handwriting.

## Workflow

1. **Text Detection (CRAFT):** The first step involves detecting text in the input image. For this, we use the CRAFT (Character Region Awareness for Text Detection) architecture. CRAFT is a novel text detection method that effectively localizes text areas by exploring each character's region and affinity to other characters. This allows us to accurately detect text bounding boxes in the image.

2. **Feature Extraction (ResNet):** Once the text regions are identified, we use a ResNet (Residual Network) model to extract features from these regions. ResNet is a convolutional neural network that excels at learning from images. It captures the intricate patterns in the Cyrillic handwriting within the detected text boxes.

3. **Sequence Prediction (Transformer):** The extracted features are then fed into a Transformer model. The Transformer uses its self-attention mechanism to understand the context and sequence of the handwritten text. It predicts the sequence of characters, effectively translating the handwritten Cyrillic script into digital text.

4. **Evaluation Metrics (CER and WER):** The performance of `CyrillicHandScriptDecoder` is evaluated using Character Error Rate (CER) and Word Error Rate (WER). These metrics provide a quantitative measure of the model's accuracy in recognizing individual characters and words, respectively.

By integrating these technologies within a Flask web application, `CyrillicHandScriptDecoder` provides a user-friendly platform for Cyrillic handwriting recognition, contributing to the digitization and preservation of Cyrillic handwritten texts.

## Project Setup

This project uses several data files and a Python environment for its operation. Here's a brief description of the files we're downloading and how to set up the project:

### Data Files

1. `craft_mlt_25k.pth`: This is a pretrained model for the CRAFT text detector, which we use to detect text regions in images.

2. `ocr_transformer_rn50_64x256_53str_jit.pt`: This is a pretrained model for the OCR transformer, which we use to recognize text within the detected regions.

3. `Cyrillic_Handwriting_Dataset.zip`: This is a dataset of Cyrillic handwriting, which we use for training and testing our models.

### Project Setup Script

We provide a setup script that creates a Python environment, downloads the necessary data files, and installs the required Python packages. You can run this script with the following command:

```bash
bash scripts/setup.sh
```

This command assumes that you're in the project root directory and that the `setup.sh` script is located in the `scripts/` directory. Please adjust the command if your directory structure is different.

### Python Environment

The setup script creates a conda environment with Python 3.9 and installs the required Python packages from the `requirements.txt` file. To activate this environment, use the following command:

```bash
conda activate ocr
```