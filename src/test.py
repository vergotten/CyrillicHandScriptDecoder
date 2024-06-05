import torch
import argparse
import logging
import os
import datetime
import string

from utils.data_processing import process_data
from model import TransformerModel
from config import Hparams
from utils.collate import TextCollate
from utils.dataset import TextLoader
from utils.model_utils import validate, prediction
from utils.data_processing import generate_data
from utils.text_utils import char_error_rate


# Set up logging
now = datetime.datetime.now()
now_str = now.strftime('%d%m%Y%H%M%S')
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'test_{}.log'.format(now_str)), level=logging.INFO)
logger = logging.getLogger()


def test(model, image_dir, label_dir, char2idx, idx2char, case=True, punct=False):
    """
    params
    ---
    model : pytorch model
    image_dir : str
        path to the folder with images
    label_dir : str
        path to the tsv file with labels
    char2idx : dict
    idx2char : dict
    case : bool
        if case is False then case of letter is ignored while comparing true and predicted transcript
    punct : bool
        if punct is False then punctution marks are ignored while comparing true and predicted transcript
    returns
    ---
    character_accuracy : float
    string_accuracy : float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img2label = dict()
    raw = open(label_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        x = t.split('\t')
        img2label[image_dir + x[0]] = x[1]
    preds = prediction(model, image_dir, char2idx, idx2char, device)
    N = len(preds)

    wer = 0
    cer = 0

    for item in preds.items():
        print(item)
        img_name = item[0]
        true_trans = img2label[image_dir + img_name]
        predicted_trans = item[1]

        if 'ё' in true_trans:
            true_trans = true_trans.replace('ё', 'е')
        if 'ё' in predicted_trans['predicted_label']:
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].replace('ё', 'е')

        if not case:
            true_trans = true_trans.lower()
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].lower()

        if not punct:
            true_trans = true_trans.translate(str.maketrans('', '', string.punctuation))
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].translate(str.maketrans('', '', string.punctuation))

        if true_trans != predicted_trans['predicted_label']:
            print('true:', true_trans)
            print('predicted:', predicted_trans)
            print('cer:', char_error_rate(predicted_trans['predicted_label'], true_trans))
            print('---')
            wer += 1
            cer += char_error_rate(predicted_trans['predicted_label'], true_trans)

    character_accuracy = 1 - cer / N
    string_accuracy = 1 - (wer / N)
    return character_accuracy, string_accuracy


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hp = Hparams(args.config)

    # CREATE MAPS FROM CHARACTERS TO INDICIES AND VISA VERSA
    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}; print(f"char2idx: {char2idx}")
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}; print(f"idx2char: {idx2char}")

    logger.info(vars(hp))
    logger.info(f"device: {device}")

    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers,
                             dec_layers=hp.dec_layers, nhead=hp.nhead, dropout=hp.dropout).to(device)

    # Load the model state from the checkpoint file
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])

    word_accur, char_accur = test(model, args.test_data, args.test_labels, char2idx, idx2char, case=False, punct=False)
    print(f"word_accur: {word_accur} | char_accur: {char_accur}")

    # img2label, _, all_words = process_data(args.test_data, args.test_labels)

    # Print the keys of img2label
    # print(f"img2label keys: {list(img2label.keys())}")

    # X_test = generate_data(img2label, hp)
    #
    # test_dataset = TextLoader(X_test, img2label, char2idx, idx2char, eval=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
    #                                           batch_size=hp.batch_size, pin_memory=False,
    #                                           drop_last=False, collate_fn=TextCollate())
    #
    # eval_loss_cer, eval_accuracy = validate(model, test_loader, device)
    # logger.info(f"eval_loss_cer: {eval_loss_cer} | eval_accuracy: {eval_accuracy}")
    #
    # # Call the test function
    # word_accur, char_accur = test(model, args.test_data, args.test_labels, char2idx, idx2char, case=False, punct=False)
    # logger.info(f"word_accuracy: {word_accur} | char_accuracy: {char_accur}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Testing')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--weights', type=str, default='ocr_transformer_rn50_64x256_53str_jit.pt',
                        help='Path to the weights file')
    parser.add_argument('--test_data', type=str, default='data/test/', help='Path to testing data')
    parser.add_argument('--test_labels', type=str, default='data/test.tsv', help='Path to testing labels')

    args = parser.parse_args()
    main(args)
