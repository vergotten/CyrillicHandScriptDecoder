import os
import torch
import argparse
from torch import optim
import torch.nn as nn

from utils.data_processing import process_data, train_valid_split, generate_data
from model import TransformerModel
from config import Hparams
from utils.model_utils import train_all
from utils.collate import TextCollate
from data.text_loader import TextLoader


def main():
    parser = argparse.ArgumentParser(description='OCR Training')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--train_data', type=str, default='data/train/', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='data/test/', help='Path to testing data')
    parser.add_argument('--train_labels', type=str, default='data/train.tsv', help='Path to training labels')
    parser.add_argument('--test_labels', type=str, default='data/test.tsv', help='Path to testing labels')

    args = parser.parse_args()

    args.train_data = os.path.abspath(args.train_data)
    args.test_data = os.path.abspath(args.test_data)
    args.train_labels = os.path.abspath(args.train_labels)
    args.test_labels = os.path.abspath(args.test_labels)

    hp = Hparams(args.config)
    print(vars(hp))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers,
                             dec_layers=hp.dec_layers,
                             nhead=hp.nhead, dropout=hp.dropout).to(device)

    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    optimizer = optim.SGD(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Initialize model, optimizer, criterion, scheduler, train_loader, and val_loader here
    img2label, _, all_words = process_data(args.train_data, args.train_labels); # print(f"img2label, _, all_words: {img2label, _, all_words}")

    X_val, y_val, X_train, y_train = train_valid_split(img2label, val_part=0.1)
    print(f"train_valid_split is completed...")

    print(f"generating X_train data...")
    X_train = generate_data(X_train, hp)

    print(f"generating X_val data...")
    X_val = generate_data(X_val, hp)

    train_dataset = TextLoader(X_train, y_train, char2idx, idx2char, hp, eval=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=hp.batch_size, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate()); print(f"train_loader: {train_loader}")
    val_dataset = TextLoader(X_val, y_val, char2idx, idx2char, hp, eval=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                             batch_size=hp.batch_size, pin_memory=False,
                                             drop_last=False, collate_fn=TextCollate()); print(f"val_loader: {val_loader}")

    # for batch_idx, (images, labels) in enumerate(val_loader):
    #     # Now you can work with `images` and `labels` directly
    #     # For example, you can print their shapes:
    #     print("Images shape:", images.shape)
    #     print("Labels shape:", labels.shape)

    # Train the model
    train_all(model, optimizer, criterion, scheduler, train_loader, val_loader, epoch_limit=10)


if __name__ == "__main__":
    main()
