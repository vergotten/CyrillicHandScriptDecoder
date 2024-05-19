import os
import argparse
import torch
import matplotlib.pyplot as plt
from model import TransformerModel
from config import Hparams


def load_model_weights(filepath):
    if filepath is None:
        print("No weights loaded.")
        return None, None

    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Print all keys in the weights file
    print("Keys in weights file:", checkpoint.keys())
    print("Number of epochs:", checkpoint['epoch'])

    # Assuming 'train_loss_all' is a list of training losses for each epoch
    for i, train_loss in enumerate(checkpoint['train_loss_all']):
        print(f"Epoch {i + 1}: Train Loss = {train_loss}")

    # Load model state
    model_state = checkpoint['model']

    # Load training history
    train_loss_all = checkpoint.get('train_loss_all', [])
    valid_loss_all = checkpoint.get('valid_loss_all', [])
    eval_loss_cer_all = checkpoint.get('eval_loss_cer_all', [])
    eval_accuracy_all = checkpoint.get('eval_accuracy_all', [])

    return model_state, (train_loss_all, valid_loss_all, eval_loss_cer_all, eval_accuracy_all)



def main():
    parser = argparse.ArgumentParser(description='OCR Vis Tools')
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON configuration file")
    parser.add_argument('--weights', type=str, default='ocr_transformer_rn50_64x256_53str_jit.pt',
                        help='Path to the weights file')

    args = parser.parse_args()

    if args.weights:
        args.weights = os.path.abspath(args.weights)

    hp = Hparams(args.config)
    print(vars(hp))

    if args.weights:
        print("Weights file path:", os.path.abspath(args.weights))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel('resnet50', len(hp.cyrillic), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers,
                             nhead=hp.nhead, dropout=hp.dropout).to(device)
    model_state, history = load_model_weights(args.weights)
    if model_state is not None:
        model.load_state_dict(model_state)

    if history is not None:
        train_loss_all, valid_loss_all, eval_loss_cer_all, eval_accuracy_all = history

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_all, label='Train Loss')
        plt.plot(valid_loss_all, label='Validation Loss')
        plt.plot(eval_loss_cer_all, label='Evaluation Loss (CER)')
        plt.plot(eval_accuracy_all, label='Evaluation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
