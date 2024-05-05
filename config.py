import argparse
import json
import os


def load_config():
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser()

    # SETS OF CHARACTERS
    parser.add_argument('--cyrillic', type=list, default=['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS'], help='Sets of characters')
    # CHARS TO REMOVE
    parser.add_argument('--del_sym', type=list, default=[], help='Characters to remove')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--hidden', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--enc_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in multihead attention models')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

    # IMAGE SIZE
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--height', type=int, default=64, help='Image height')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    with open('config/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    config = load_config()
    print(config)
