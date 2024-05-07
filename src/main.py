import torch
from config import Hparams
import argparse


def main(hp):
    # CREATE MAPS FROM CHARACTERS TO INDICIES AND VISA VERSA
    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    return char2idx, idx2char


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON configuration file")
    args = parser.parse_args()

    hp = Hparams(args.config)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(vars(hp))
    # print(main(hp))
