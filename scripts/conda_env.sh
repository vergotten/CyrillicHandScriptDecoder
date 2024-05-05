#!/bin/bash

# Create a conda environment with Python 3.9
conda create -n ocr python=3.9
conda activate ocr

# Download the necessary files
# craft_mlt_25k.pth: Pretrained model for CRAFT text detector
gdown --id 1SznyDq4Hddcy7PGoPQGsycSpNjXv8a_U -O /content/craft_mlt_25k.pth

# ocr_transformer_rn50_64x256_53str_jit.pt: Pretrained model for OCR transformer
gdown --id 1_NgWNm7R9xfkKqWYqo35j5rbUAURvm3P -O /content/ocr_transformer_rn50_64x256_53str_jit.pt

# Cyrillic Handwriting Dataset.zip: Dataset for Cyrillic handwriting
gdown --id 1fOCrZsjiXtX8N_Yeb_PK6lcyDGcTJQoB -O /content/Cyrillic_Handwriting_Dataset.zip

# Extract the zip file to the same name directory
unzip /content/Cyrillic_Handwriting_Dataset.zip -d /content/Cyrillic_Handwriting_Dataset

# Install the requirements
pip install -r requirements.txt
