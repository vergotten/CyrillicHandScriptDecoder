import torch
from config import Hparams
import argparse
import Augmentor
from utils.data_processing import text_to_labels
from data.augmentations import Vignetting, UniformNoise, LensDistortion
from torchvision import transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON configuration file")
    args = parser.parse_args()

    hp = Hparams(args.config)

    char2idx = {char: idx for idx, char in enumerate(hp.cyrillic)}
    idx2char = {idx: char for idx, char in enumerate(hp.cyrillic)}

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(vars(hp))
    # print(main(hp))

    # Create an instance of each augmentation class
    vignet = Vignetting()
    un = UniformNoise()
    tt = transforms.ToTensor()
    p = Augmentor.Pipeline()
    ld = LensDistortion()
    p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
    p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

