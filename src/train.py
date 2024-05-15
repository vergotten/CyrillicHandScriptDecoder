import os
import torch
import argparse
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src.data.data_processing import process_data, train_valid_split
from model import TransformerModel
from config import Hparams
from data.collate import TextCollate
from data.dataset import TransformedTextDataset
from data.text_utils import labels_to_text, char_error_rate


def train(model, optimizer, criterion, iterator, device):
    model.train()
    epoch_loss = 0
    counter = 0
    for src, trg in iterator:
        counter += 1
        if counter % 500 == 0:
            print('[', counter, '/', len(iterator), ']')
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        # print("Output shape:", output.shape)
        # print("Target shape:", trg.shape)

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))

        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_all(model, optimizer, criterion, scheduler, train_loader, val_loader, epoch_limit, device):
    """
    General function for training and validation.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (nn.Object): The optimizer for training.
        criterion (nn.Object): The loss function.
        scheduler (nn.Object): The learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        epoch_limit (int): The maximum number of epochs for training.

    Returns:
        None
    """
    train_loss = 0
    confuse_dict = dict()
    for epoch in range(0, epoch_limit):
        print(f'Epoch: {epoch + 1:02}')
        print("-----------train------------")
        train_loss = train(model, optimizer, criterion, train_loader, device)
        print(f"Train loss for epoch {epoch + 1}: {train_loss}")

        # print("\n-----------valid------------")
        # valid_loss = evaluate(model, criterion, val_loader)
        # print(f"Validation loss for epoch {epoch + 1}: {valid_loss}")
        #
        # print("-----------eval------------")
        # eval_loss_cer, eval_accuracy = validate(model, val_loader, device)
        # print(f"Evaluation loss for epoch {epoch + 1}: {eval_loss_cer}")
        # print(f"Evaluation accuracy for epoch {epoch + 1}: {eval_accuracy}")

        # scheduler.step(eval_loss_cer)


def evaluate(model, criterion, iterator):
    """
    params
    ---
    model : nn.Module
    criterion : nn.Object
    iterator : torch.utils.data.DataLoader
    returns
    ---
    epoch_loss / len(iterator) : float
        overall loss
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (src, trg) in tqdm(iterator):
            src, trg = src.cuda(), trg.cuda()
            output = model(src, trg[:-1, :])
            loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def validate(model, dataloader, device):
    idx2char = dataloader.dataset.idx2char
    char2idx = dataloader.dataset.char2idx
    model.eval()
    wer_overall = 0
    cer_overall = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            print(f"Processing batch {i+1}...")
            img = np.moveaxis(src[0].numpy(), 0, 2)
            if torch.cuda.is_available():
                src = src.cuda()

            out_indexes = [char2idx['SOS'], ]
            print("Generating output sequence...")
            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model(src, trg_tensor)
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            out_char = labels_to_text(out_indexes[1:], idx2char)
            real_char = labels_to_text(trg[1:, 0].numpy(), idx2char)
            wer_overall += int(real_char != out_char)
            print(f"WER for batch {i+1}: {int(real_char != out_char)}")
            if out_char:
                cer = char_error_rate(real_char, out_char)
            else:
                cer = 1
            cer_overall += cer
            print(f"CER for batch {i+1}: {cer}")

    return cer_overall / len(dataloader) * 100, wer_overall / len(dataloader) * 100


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

    img2label, _, all_words = process_data(args.train_data, args.train_labels)

    X_val, y_val, X_train, y_train = train_valid_split(img2label, train_part=0.01, val_part=0.001)

    train_dataset = TransformedTextDataset(X_train, y_train, char2idx, idx2char, hp, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False,
                                               batch_size=hp.batch_size, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())
    val_dataset = TransformedTextDataset(X_val, y_val, char2idx, idx2char, hp, train=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                             batch_size=hp.batch_size, pin_memory=False,
                                             drop_last=False, collate_fn=TextCollate())

    # # train_all(model, optimizer, criterion, scheduler, train_loader, val_loader, epoch_limit=10, device=device)
    #
    # print(f"Displaying images...")
    #
    # from utils.text_utils import labels_to_text, text_to_labels
    #
    # # Get the first batch of data
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # #
    # # # Print the 0th image and label
    # print("Image: ", images[0])
    # print("Label: ", labels[0])
    # print("Label text: ", labels_to_text(labels[0], idx2char))
    #
    # # Call the function
    # # display_images(train_loader, hp)


if __name__ == "__main__":
    main()
