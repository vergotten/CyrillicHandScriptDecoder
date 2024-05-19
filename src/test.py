import torch
import argparse
import logging
import os
import datetime

from utils.data_processing import process_data
from model import TransformerModel
from config import Hparams
from utils.collate import TextCollate
from utils.dataset import TextLoader
from utils.model_utils import validate
from utils.data_processing import generate_data


# Set up logging
now = datetime.datetime.now()
now_str = now.strftime('%d%m%Y%H%M%S')
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'test_{}.log'.format(now_str)), level=logging.INFO)
logger = logging.getLogger()


def main(args):
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

    img2label, _, all_words = process_data(args.test_data, args.test_labels)

    X_test = generate_data(img2label, hp)

    test_dataset = TextLoader(X_test, img2label, char2idx, idx2char, eval=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                              batch_size=hp.batch_size, pin_memory=False,
                                              drop_last=False, collate_fn=TextCollate())

    eval_loss_cer, eval_accuracy = validate(model, test_loader, device)
    logger.info(f"eval_loss_cer: {eval_loss_cer} | eval_accuracy: {eval_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Testing')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--weights', type=str, default='ocr_transformer_rn50_64x256_53str_jit.pt',
                        help='Path to the weights file')
    parser.add_argument('--test_data', type=str, default='data/test/', help='Path to testing data')
    parser.add_argument('--test_labels', type=str, default='data/test.tsv', help='Path to testing labels')

    args = parser.parse_args()
    main(args)
