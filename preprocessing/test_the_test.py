import torch
from torch import nn
from torchviz import make_dot

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Define the linear layers for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Linearly project the input into queries, keys, and values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape Q, K, and V to split heads
        Q = Q.view(Q.size(0), -1, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(K.size(0), -1, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(V.size(0), -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) /  (self.d_k ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Reshape and concatenate attention heads
        context = context.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        
        return context

# # Instantiate the SelfAttention module
# attention = SelfAttention(d_model=512, n_head=2)

# # Define a sample input tensor
# src = torch.randn(1, 32, 512)

# # Forward pass to get the output tensor
# output = attention(src)

# # Generate a visualization of the computation graph
# dot = make_dot(output, params=dict(attention.named_parameters()))

# # Render the graph to a PNG file
# dot.render("self_attention", format="png")

# sum = 0;
# for name, param in attention.named_parameters():
#     print(name, param.numel())
#     sum =  sum+ param.numel()

# print('herllo')
# print(sum)



class PreClassifier(nn.Module):
    def __init__(self, d_model, n_head):
        super(PreClassifier, self).__init__()
        self.attention = SelfAttention(d_model, n_head)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply self-attention mechanism
        x = self.attention(x)
        
        # Average pooling over the sequence length
        x = x.mean(dim=1)
        
        # Linear layer and sigmoid activation
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

# Instantiate the PreClassifier module
pre_classifier = PreClassifier(d_model=512, n_head=2)
src = torch.randn(1, 32, 512)
# Forward pass to get the output tensor
output = pre_classifier(src)

# Generate a visualization of the computation graph
dot = make_dot(output, params=dict(pre_classifier.named_parameters()))

# Render the graph to a PNG file
dot.render("pre_classifier", format="png")

# Count the number of parameters in the pre-classifier module
sum = 0
for name, param in pre_classifier.named_parameters():
    print(name, param.numel())
    sum += param.numel()

print('Total Parameters:', sum)
