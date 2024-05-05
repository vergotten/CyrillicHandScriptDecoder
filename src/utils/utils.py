import os
import math
import string
import random
import time
import subprocess
import sys
# import wandb
import torch
# import torch.nn as nn
# from torch import optim
from tqdm import tqdm

try:
    import editdistance
except ImportError:
    print("editdistance module not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "editdistance"])
    import editdistance


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# text to array of indicies
def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]


# convert images and labels into defined data structures
def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str
      path to directory with images
    labels_dir : str
      path to tsv file with labels
    returns
    ---
    img2label : dict
      keys are names of images and values are correspondent labels
    chars : list
      all unique chars used in data
    all_labels : list
    """

    chars = []
    img2label = dict()

    raw = open(labels_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        try:
            x = t.split('\t')
            flag = False
            for item in ignore:
                if item in x[1]:
                    flag = True
            if flag == False:
                img2label[image_dir + x[0]] = x[1]
                for char in x[1]:
                    if char not in chars:
                        chars.append(char)
        except:
            print('ValueError:', x)
            pass

    all_labels = sorted(list(set(list(img2label.values()))))
    chars.sort()
    chars = ['PAD', 'SOS'] + chars + ['EOS']

    return img2label, chars, all_labels


# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded


# TRANSLATE INDICIES TO TEXT
def labels_to_text(s, idx2char):
    """
    params
    ---
    idx2char : dict
        keys : int
            indicies of characters
        values : str
            characters
    returns
    ---
    S : str
    """
    S = "".join([idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


# COMPUTE CHARACTER ERROR RATE
def char_error_rate(p_seq1, p_seq2):
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str
    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))


# RESIZE AND NORMALIZE IMAGE
def process_image(img):
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = hp.height
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = hp.width
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


# GENERATE IMAGES FROM FOLDER
def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images
    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        # Ensure path is a string
        if not isinstance(path, str):
            path = str(path)
        img = cv2.imread(path)
        try:
            img = process_image(img)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue
        data_images.append(img)
    return data_images


def train(model, optimizer, criterion, iterator):
    """
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    iterator : torch.utils.data.DataLoader
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


# GENERAL FUNCTION FROM TRAINING AND VALIDATION
def train_all(model,optimizer,criterion,scheduler, train_loader, val_loader,epoch_limit):
    train_loss = 0
    confuse_dict = dict()
    for epoch in range(0, epoch_limit):
        print(f'Epoch: {epoch + 1:02}')
        print("-----------train------------")
        train_loss = train(model, optimizer, criterion, train_loader)
        print("train loss :",train_loss)
        print("\n-----------valid------------")
        valid_loss = evaluate(model, criterion, val_loader)
        print("validation loss :",valid_loss)
        print("-----------eval------------")
        eval_loss_cer, eval_accuracy = validate(model, val_loader)
        scheduler.step(eval_loss_cer)

        if WANDB_LOG and epoch%4 == 0:
            wandb.log({'Train loss WER': train_loss, "Validation loss WER": valid_loss, 'Validation Word Accuracy': 100 - eval_accuracy,
                       'Validation loss CER': eval_loss_cer,'Learning Rate':scheduler._last_lr[0]})


def validate(model, dataloader):
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
            img = np.moveaxis(src[0].numpy(), 0, 2)
            if torch.cuda.is_available():
              src = src.cuda()

            out_indexes = [char2idx['SOS'], ]

            for i in range(100):

                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

                #output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
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

            cer_overall += cer
            if WANDB_LOG and out_char != real_char:
                wandb.log({'Validation Character Accuracy': (1-cer)*100})
                wandb.log({"Validation Examples": wandb.Image(img, caption="Pred: {} Truth: {}".format(out_char, real_char))})
                show_count += 1

    return cer_overall / len(dataloader) * 100, wer_overall / len(dataloader) * 100


# SPLIT DATASET INTO TRAIN AND VALID PARTS
def train_valid_split(img2label, val_part=0.3):
    """
    params
    ---
    img2label : dict
        keys are paths to images, values are labels (transcripts of crops)
    returns
    ---
    imgs_val : list of str
        paths
    labels_val : list of str
        labels
    imgs_train : list of str
        paths
    labels_train : list of str
        labels
    """

    imgs_val, labels_val = [], []
    imgs_train, labels_train = [], []

    N = int(len(img2label) * val_part)
    items = list(img2label.items())
    random.shuffle(items)
    for i, item in enumerate(items):
        if i < N:
            imgs_val.append(item[0])
            labels_val.append(item[1])
        else:
            imgs_train.append(item[0])
            labels_train.append(item[1])
    print('valid part:{}'.format(len(imgs_val)))
    print('train part:{}'.format(len(imgs_train)))
    return imgs_val, labels_val, imgs_train, labels_train


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

# PREPARE DATASET FROM TRAINING
# IT CREATES MIXED DATASET: THE FIRST PART COMES FROM REAL DATA AND THE SECOND PART COMES FORM GENERATOR
def get_mixed_data(pretrain_image_dir, pretrain_labels_dir, train_image_dir, train_labels_dir, pretrain_part=0.0):
    img2label1, chars1, all_words1 = process_data(pretrain_image_dir, pretrain_labels_dir)  # PRETRAIN PART
    img2label2, chars2, all_words2 = process_data(train_image_dir, train_labels_dir)  # TRAIN PART
    img2label1_list = list(img2label1.items())
    N = len(img2label1_list)
    for i in range(N):
        j = np.random.randint(0, N)
        item = img2label1_list[j]
        img2label2[item[0]] = item[1]
    return img2label2


# MAKE PREDICTION
def prediction(model, test_dir, char2idx, idx2char):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    char2idx : dict
        map from chars to indicies
    id2char : dict
        map from indicies to chars
    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
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

                output = model(src,trg_tensor)
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            pred = labels_to_text(out_indexes[1:], idx2char)
            preds[filename] = {'predicted_label': pred, 'p_values': p_values}

    return preds