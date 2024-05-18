import os
import torch
import argparse
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import cv2
from torchvision import transforms

from utils.data_processing import process_data, train_valid_split
from model import TransformerModel
from config import Hparams
from utils.collate import TextCollate
from utils.dataset import TransformedTextDataset
from utils.data_processing import process_image
# from utils.text_utils import labels_to_text, char_error_rate
from utils.model_utils import train_all


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hp = Hparams(args.config)

    # CREATE MAPS FROM CHARACTERS TO INDICIES AND VISA VERSA
    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    print(vars(hp))
    print(f"device: {device}")

    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers,
                             dec_layers=hp.dec_layers,
                             nhead=hp.nhead, dropout=hp.dropout).to(device)

    optimizer = optim.SGD(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    img2label, _, all_words = process_data(args.train_data, args.train_labels)

    X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)

    train_dataset = TransformedTextDataset(X_train, y_train, char2idx, idx2char, hp)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False,
                                               batch_size=hp.batch_size, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())
    val_dataset = TransformedTextDataset(X_val, y_val, char2idx, idx2char, hp)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                             batch_size=hp.batch_size, pin_memory=False,
                                             drop_last=False, collate_fn=TextCollate())

    train_all(model, optimizer, criterion, scheduler, train_loader, val_loader, epoch_limit=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Training')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--train_data', type=str, default='data/train/', help='Path to training utils')
    parser.add_argument('--test_data', type=str, default='data/test/', help='Path to testing utils')
    parser.add_argument('--train_labels', type=str, default='data/train.tsv', help='Path to training labels')
    parser.add_argument('--test_labels', type=str, default='data/test.tsv', help='Path to testing labels')

    args = parser.parse_args()
    main(args)

