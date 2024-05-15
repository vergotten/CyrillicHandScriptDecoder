import unittest
import torch
import torch.nn as nn

from src.model import TransformerModel


class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.config = {
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
        self.model = TransformerModel(bb_name='resnet50', outtoken=92, hidden=self.config['hidden'], enc_layers=self.config['enc_layers'], dec_layers=self.config['dec_layers'], nhead=self.config['nhead'], dropout=self.config['dropout'], pretrained=False)

    def test_forward_pass(self):
        src = torch.rand(self.config['batch_size'], 3, self.config['height'], self.config['width'])  # [B,C,H,W]
        trg = torch.randint(0, 92, (13, self.config['batch_size']))  # [L,B]
        output = self.model(src, trg)
        self.assertEqual(output.shape, (13, self.config['batch_size'], 92))

    def test_model_initialization(self):
        try:
            model = TransformerModel(bb_name='resnet50', outtoken=92, hidden=self.config['hidden'], enc_layers=self.config['enc_layers'], dec_layers=self.config['dec_layers'], nhead=self.config['nhead'], dropout=self.config['dropout'], pretrained=False)
        except Exception as e:
            self.fail(f"Model initialization failed with error: {e}")

    def test_model_trainability(self):
        # Define a simple loss function
        loss_fn = nn.CrossEntropyLoss()

        # Create some tensors
        src = torch.rand(self.config['batch_size'], 3, self.config['height'], self.config['width'])  # [B,C,H,W]
        trg = torch.randint(0, 92, (13, self.config['batch_size']))  # [L,B]

        # Forward pass
        output = self.model(src, trg)

        # Compute loss
        loss = loss_fn(output.view(-1, output.shape[-1]), trg.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        optimizer.step()

        # Check if model parameters are updated
        params_before = [p.clone() for p in self.model.parameters()]
        optimizer.step()
        params_after = [p for p in self.model.parameters()]

        for p_before, p_after in zip(params_before, params_after):
            self.assertTrue(torch.any(p_before != p_after), "Model parameters did not update during training.")


if __name__ == "__main__":
    unittest.main()
