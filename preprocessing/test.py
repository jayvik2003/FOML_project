import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchviz import make_dot
from PIL import Image

class TransformerModel(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 32, nhead: int = 2, nlayers: int = 1,
                 dropout: float = 0.1, classes: int = 2, use_pos: bool = False):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(d_model)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model*seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """
        #src = src * math.sqrt(self.d_model)
        #src = self.norm(src)
        # if self.use_positional_enc:
        #     src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        # t_out = torch.flatten(t_out, start_dim=1)
        # pooler = self.pre_classifier(t_out)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.dropout(pooler)
        # output = self.classifier(pooler)
        # output = self.logSoftmax(output)
        return t_out
        # return src
        # return pooler
        # return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Define a sample input tensor
src = torch.randn(1, 32, 512)
model = TransformerModel()

# Forward pass to get the output tensor
output = model(src)

# Generate a visualization of the computation graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("transformer_model", format="png")



# # Visualize the positional encoding
# def plot_positional_encoding(positional_encoding):
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(positional_encoding.squeeze().detach().cuda().numpy(), cmap='viridis')
#     plt.title('Positional Encoding')
#     plt.xlabel('Position')
#     plt.ylabel('Dimension')
#     plt.show()

# Get positional encoding of the sample input
# pos_enc = model.pos_encoder.pe.detach().cuda()
# plot_positional_encoding(pos_enc)

# Visualize the output of the transformer encoder
def plot_transformer_output(transformer_output):
    plt.figure(figsize=(6, 3))
    sns.heatmap(transformer_output.squeeze().detach().cpu().numpy(), cmap='viridis')
    plt.title('Output of Transformer Encoder')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Sequence Length')
    # plt.show()


transformer_output_forward = model.transformer_encoder(src).detach().cpu()
plot_transformer_output(transformer_output_forward)
# Count the number of parameters in each block/sub-block
sum = 0;
for name, param in model.named_parameters():
    print(name, param.numel())
    sum =  sum+ param.numel()

print('herllo')
print(sum)