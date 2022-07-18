import torch, math
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncodinold(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        variable = self.pe[:x.size(0)]
        print(f"Input Embeddings (ver1): {x.shape}")
        print(f"Positional Encodings (ver1): {variable.shape}")
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe          = torch.zeros(max_len, d_model)
        position    = torch.arange(0, max_len).unsqueeze(1)
        div_term    = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe          = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Feed_Forward(nn.Module):
    def __init__(self, embed_dim=512, expand_ratio=4):
        super(Feed_Forward, self).__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim*expand_ratio)
        self.relu    = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim*expand_ratio, embed_dim)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x