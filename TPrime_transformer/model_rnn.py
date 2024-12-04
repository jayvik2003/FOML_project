import torch
import torch.nn as nn
import torch.optim as optim

class RNNMLPClassifier(nn.Module):
    def __init__(self, input_size: int = 64*2, hidden_size: int = 64, seq_len: int = 64,
                 num_layers: int = 2, dropout: float = 0.5, num_classes: int = 4):
        super(RNNMLPClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # Define the RNN (LSTM) layer for capturing dependencies
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Define the MLP layers for classification
        self.pre_classifier = nn.Linear(hidden_size * seq_len, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):

        # Initialize hidden and cell state
        h_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Flatten the LSTM output for the MLP
        lstm_out_flat = lstm_out.contiguous().view(x.size(0), -1)  # Shape: [batch_size, hidden_size * seq_len]
        
        # Pass through MLP layers
        pooler = self.pre_classifier(lstm_out_flat)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        
        # Output class probabilities
        output = self.logSoftmax(output)
        return output

# Example Usage:
# Set hyperparameters
input_size = 2*64          # Number of features per time step
hidden_size = 64        # Hidden size of LSTM
seq_len = 24             # Length of input sequences
num_layers = 2           # Number of LSTM layers
num_classes = 4          # Number of output classes
dropout = 0.1           # Dropout rate

# Initialize model
model = RNNMLPClassifier(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len,
                         num_layers=num_layers, dropout=dropout, num_classes=num_classes)

# Print model architecture
print(model)

# Example input (e.g., a batch of sequences of sine waves or other data)
# Assuming batch size of 32, sequence length of 64, and 1 feature per time step
x_example = torch.randn(32, seq_len, input_size)

# Forward pass
output = model(x_example)
print("Output shape:", output.shape)  # Should be [32, num_classes]

# Total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
