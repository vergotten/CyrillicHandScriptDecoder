FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
# RUN apt-get update && apt-get install -y \
#     python3.9 \
#     python3-pip \
#     unzip \
#     gdown

# Create a conda environment with Python 3.9
RUN conda create -n ocr python=3.9

# Activate the conda environment
RUN echo "conda activate ocr" >> ~/.bashrc

# Install the requirements
RUN pip install -r requirements.txt
RUN pip install gdown

# Download the necessary files
# craft_mlt_25k.pth: Pretrained model for CRAFT text detector
# RUN gdown --id 1SznyDq4Hddcy7PGoPQGsycSpNjXv8a_U -O /content/craft_mlt_25k.pth

# ocr_transformer_rn50_64x256_53str_jit.pt: Pretrained model for OCR transformer
RUN gdown --id 1_NgWNm7R9xfkKqWYqo35j5rbUAURvm3P -O /app/ocr_transformer_rn50_64x256_53str_jit.pt

# Cyrillic Handwriting Dataset.zip: Dataset for Cyrillic handwriting
# RUN gdown --id 1fOCrZsjiXtX8N_Yeb_PK6lcyDGcTJQoB -O /content/Cyrillic_Handwriting_Dataset.zip

# Extract the zip file to the same name directory
# RUN unzip /content/Cyrillic_Handwriting_Dataset.zip -d /content/Cyrillic_Handwriting_Dataset

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
# CMD ["python", "app.py"]
