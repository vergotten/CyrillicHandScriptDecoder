import torch
from tqdm import tqdm
import numpy as np
from .text_utils import labels_to_text, char_error_rate
import os
import cv2
from .data_processing import process_image


def count_parameters(model):
    """
    Count the number of parameters in the model.

    Args:
        model (nn.Module): The model.

    Returns:
        int: The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, optimizer, criterion, iterator):
    """
    Train the model.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (nn.Object): The optimizer for training.
        criterion (nn.Object): The loss function.
        iterator (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The average loss per batch in the dataset.
    """
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

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_all(model,optimizer,criterion,scheduler, train_loader, val_loader,epoch_limit):
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
        train_loss = train(model, optimizer, criterion, train_loader)
        print("train loss :", train_loss)
        print("\n-----------valid------------")
        valid_loss = evaluate(model, criterion, val_loader)
        print("validation loss :", valid_loss)
        print("-----------eval------------")
        eval_loss_cer, eval_accuracy = validate(model, val_loader)
        scheduler.step(eval_loss_cer)


def evaluate(model, criterion, iterator):
    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to be evaluated.
        criterion (nn.Object): The loss function.
        iterator (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The average loss per batch in the dataset.
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
    """
    Validate the model.

    Args:
        model (nn.Module): The model to be validated.
        dataloader (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The character error rate.
        float: The word error rate.
    """
    idx2char = dataloader.dataset.idx2char
    char2idx = dataloader.dataset.char2idx
    model.eval()
    show_count = 0
    wer_overall = 0
    cer_overall = 0
    with torch.no_grad():
        for (src, trg) in dataloader:
            img = np.moveaxis(src[0].numpy(), 0, 2)
            if torch.cuda.is_available():
                src = src.cuda()

            out_indexes = [char2idx['SOS'], ]

            for i in range(100):

                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                # output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                output = model(src, trg_tensor)
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            out_char = labels_to_text(out_indexes[1:], idx2char)
            real_char = labels_to_text(trg[1:, 0].numpy(), idx2char)
            wer_overall += int(real_char != out_char)
            if out_char:
                cer = char_error_rate(real_char, out_char)
            else:
                cer = 1

            cer_overall += cer

    return cer_overall / len(dataloader) * 100, wer_overall / len(dataloader) * 100


def prediction(model, test_dir, char2idx, idx2char, device):
    """
    Make prediction.

    Args:
        model (nn.Module): The model to make prediction.
        test_dir (str): Path to directory with images.
        char2idx (dict): Map from chars to indices.
        idx2char (dict): Map from indices to chars.

    Returns:
        dict: Predictions. Key is the name of image in directory, value is a dict with keys ['p_value', 'predicted_label'].
    """
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = cv2.imread(test_dir + filename)
            img = process_image(img).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))
            src = torch.FloatTensor(img).unsqueeze(0).to(device)
            p_values = 1
            out_indexes = [char2idx['SOS'], ]

            for i in range(100):

                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                output = model(src, trg_tensor)
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            pred = labels_to_text(out_indexes[1:], idx2char)
            preds[filename] = {'predicted_label': pred, 'p_values': p_values}

    return preds
