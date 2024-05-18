import torch
import torch.nn as nn
from torchvision import models
import math

from .utils.model_utils import count_parameters


class TransformerModel(nn.Module):
    """
    Transformer Model for Cyrillic Handwritten Script Decoding.
    """
    def __init__(self, bb_name, outtoken, hidden, enc_layers=1, dec_layers=1, nhead=1, dropout=0.1, pretrained=False):
        """
        Initialize the Transformer model.
        """
        super(TransformerModel, self).__init__()
        self.backbone = models.__getattribute__(bb_name)(pretrained=pretrained)
        self.backbone.fc = nn.Conv2d(2048, int(hidden/2), 1)

        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dim_feedforward=hidden * 4, dropout=dropout,
                                          activation='relu')

        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

        print('backbone: {}'.format(bb_name))
        print('layers: {}'.format(enc_layers))
        print('heads: {}'.format(nhead))
        print('dropout: {}'.format(dropout))
        print(f'{count_parameters(self):,} trainable parameters')

    def generate_square_subsequent_mask(self, sz):
        """
        Generate square subsequent mask.
        """
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        """
        Make length mask.
        """
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        """
        Forward pass of the Transformer model.
        """
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
        x = self.backbone.conv1(src)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # [64, 2048, 2, 8] : [B,C,H,W]

        x = self.backbone.fc(x) # [64, 256, 2, 8] : [B,C,H,W]
        x = x.permute(0, 3, 1, 2) # [64, 8, 256, 2] : [B,W,C,H]
        x = x.flatten(2) # [64, 8, 512] : [B,W,CH]
        x = x.permute(1, 0, 2) # [8, 64, 512] : [W,B,CH]

        src_pad_mask = self.make_len_mask(x[:, :, 0])
        src = self.pos_encoder(x) # [8, 64, 512]
        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask) # [13, 64, 512] : [L,B,CH]
        output = self.fc_out(output) # [13, 64, 92] : [L,B,H]

        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the Positional Encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the Positional Encoding.
        """
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Define the configuration
    config = {
        "del_sym": [],
        "lr": 0.01,
        "batch_size": 1,
        "hidden": 512,
        "enc_layers": 2,
        "dec_layers": 2,
        "nhead": 4,
        "dropout": 0.0,
        "width": 256,
        "height": 64,
        "path_test_dir": "utils/test/",
        "path_test_labels": "utils/test.tsv",
        "path_train_dir": "utils/train/",
        "path_train_labels": "utils/train.tsv"
    }

    # Define the model using the configuration
    model = TransformerModel(bb_name='resnet50', outtoken=92, hidden=config['hidden'], enc_layers=config['enc_layers'], dec_layers=config['dec_layers'], nhead=config['nhead'], dropout=config['dropout'], pretrained=False)

    # Create some tensors using the configuration
    src = torch.rand(config['batch_size'], 3, config['height'], config['width'])  # [B,C,H,W]
    trg = torch.randint(0, 92, (13, config['batch_size']))  # [L,B]

    # Make a forward pass
    output = model(src, trg)

    print(output)

