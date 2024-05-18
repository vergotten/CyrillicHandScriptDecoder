import unittest
import torch
from src.utils.collate import TextCollate


class TestTextCollate(unittest.TestCase):
    def setUp(self):
        self.collate_fn = TextCollate()

    def test_call(self):
        # Create a mock batch
        batch = [(torch.randn(3, 64, 256), torch.randint(0, 10, (5,))) for _ in range(10)]

        # Apply the collate function
        x_padded, y_padded = self.collate_fn(batch)

        # Check the output types
        self.assertIsInstance(x_padded, torch.Tensor)
        self.assertIsInstance(y_padded, torch.Tensor)

        # Check the output shapes
        self.assertEqual(x_padded.shape, (10, 3, 64, 256))
        self.assertEqual(y_padded.shape[1], 10)

        # Check that y_padded is correctly zero-padded
        max_y_len = max([i[1].size(0) for i in batch])
        self.assertEqual(y_padded.shape[0], max_y_len)
        for i in range(10):
            y_len = batch[i][1].size(0)
            if y_len < max_y_len:
                self.assertTrue(torch.all(y_padded[y_len:, i] == 0))


if __name__ == '__main__':
    unittest.main()
