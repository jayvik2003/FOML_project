import unittest
import torch
from torch.utils.data import DataLoader
from model_transformer import TransformerModel
from TPrime_transformer_train import train_epoch

class TestTrainEpoch(unittest.TestCase):
    def setUp(self):
        # Set up dummy data and model for testing
        self.batch_size = 32
        self.seq_len = 64
        self.slice_len = 128
        self.num_classes = 10
        self.device = torch.device("cuda")
        self.dummy_data_loader = self.generate_dummy_data_loader()

        # Create a dummy model
        self.model = TransformerModel(classes=self.num_classes, d_model=256, seq_len=self.seq_len, nlayers=3, use_pos=True)
        self.model.to(self.device)

    def generate_dummy_data_loader(self):
        # Dummy dataset generation
        dummy_dataset = [(torch.randn(self.batch_size, 2, self.slice_len), torch.randint(0, self.num_classes, (self.batch_size,))) for _ in range(10)]
        return DataLoader(dummy_dataset, batch_size=self.batch_size, shuffle=True)

    def test_train_epoch(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Run train_epoch
        loss, accuracy = train_epoch(self.dummy_data_loader, self.model, loss_fn, optimizer)

        # Check if loss and accuracy are within expected range
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= loss <= 10)  # assuming loss is within some reasonable range
        self.assertTrue(0 <= accuracy <= 1)

if __name__ == '__main__':
    unittest.main()
