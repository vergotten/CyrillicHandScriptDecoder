import torch
from tqdm import tqdm
import numpy as np
import os
import cv2
import datetime
import logging


from .data_processing import process_image
from .text_utils import labels_to_text, char_error_rate, word_error_rate


logger = logging.getLogger()


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
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    iterator : torch.utils.utils.DataLoader
    returns
    ---
    epoch_loss / len(iterator) : float
        overall loss
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


def train_all(model, optimizer, idx2char, criterion, scheduler, train_loader, val_loader, epoch_limit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss = 0
    best_loss = float('inf')
    confuse_dict = dict()

    # Initialize metrics
    best_eval_loss_cer = float('inf')
    valid_loss_all = []
    train_loss_all = []
    eval_loss_cer_all = []
    eval_accuracy_all = []

    start_epoch = 0

    for epoch in range(start_epoch, epoch_limit):
        start_time = datetime.datetime.now()
        message = f'Epoch: {epoch + 1:02}'
        print(message)
        logger.info(message)
        message = "-----------train------------"
        print(message)
        logger.info(message)
        train_loss = train(model, optimizer, criterion, train_loader)
        message = f"train loss : {train_loss}"
        print(message)
        logger.info(message)
        train_loss_all.append(train_loss)
        message = "\n-----------valid------------"
        print(message)
        logger.info(message)
        valid_loss = evaluate(model, criterion, val_loader)
        message = f"validation loss : {valid_loss}"
        print(message)
        logger.info(message)
        valid_loss_all.append(valid_loss)
        message = "-----------eval------------"
        print(message)
        logger.info(message)
        eval_loss_cer, eval_accuracy = validate(model, val_loader, device)
        message = f"eval_loss_cer: {eval_loss_cer} | eval_accuracy: {eval_accuracy}"
        print(message)
        logger.info(message)
        eval_loss_cer_all.append(eval_loss_cer)
        eval_accuracy_all.append(eval_accuracy)
        scheduler.step(eval_loss_cer)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        message = f"Time taken for Epoch {epoch + 1}: {elapsed_time}"
        print(message)
        logger.info(message)

        # Save the best checkpoint during training
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_eval_loss_cer = eval_loss_cer
            # Get current date and time
            now = datetime.datetime.now()
            # Format as a string
            now_str = now.strftime('%d%m%Y%H%M%S')
            # Append date and time to filename
            filename = 'ocr_transformer_rn50_64x256_53str_jit_{}.pt'.format(now_str)
            # Create checkpoints directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            # Save checkpoint
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_eval_loss_cer': best_eval_loss_cer,
                'valid_loss_all': valid_loss_all,
                'train_loss_all': train_loss_all,
                'eval_loss_cer_all': eval_loss_cer_all,
                'eval_accuracy_all': eval_accuracy_all
            }, os.path.join('checkpoints', filename))
            message = f'Saved best checkpoint to {filename}.'
            print(message)
            logger.info(message)


def validate(model, dataloader, device):
    """
    params
    ---
    model : nn.Module
    dataloader :
    returns
    ---
    cer_overall / len(dataloader) * 100 : float
    wer_overall / len(dataloader) * 100 : float
    """
    idx2char = dataloader.dataset.idx2char
    char2idx = dataloader.dataset.char2idx
    model.eval()
    show_count = 0
    wer_overall = 0
    cer_overall = 0
    with torch.no_grad():
        for (src, trg) in dataloader:
            print(f"src, trg: {src.shape, trg.shape}")
            img = np.moveaxis(src[0].numpy(), 0, 2)
            if torch.cuda.is_available():
              src = src.cuda()

            out_indexes = [char2idx['SOS'], ]

            for i in range(100):

                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                # output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                output = model(src,trg_tensor)
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

    return cer_overall / len(dataloader) * 100, wer_overall / len(dataloader) * 100


def evaluate(model, criterion, iterator):
    """
    params
    ---
    model : nn.Module
    criterion : nn.Object
    iterator : torch.utils.utils.DataLoader
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
