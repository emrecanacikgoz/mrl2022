import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(q, k, v, d_k, mask, dropout):

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_k))

    if mask is not None:
        mask = mask.unsqueeze(1)
        att  = att.masked_fill(mask == 0, -1e9)

    normalized_attention = F.softmax(att, dim=-1)
    #att = dropout(normalized_attention)

    y = normalized_attention @ v
    return y    

class MultiHead_Attention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.d_k       = embed_dim//num_heads
        self.key       = nn.Linear(embed_dim, embed_dim)
        self.query     = nn.Linear(embed_dim, embed_dim)
        self.value     = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout_rate)
        self.fc        = nn.Linear(embed_dim, embed_dim)
        

    def forward(self, q, k, v, mask=None):
        Bq, Tq, Cq = q.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(q).view(Bq, -1, self.num_heads, Cq//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)
        k = self.key(k).view(  Bq, -1, self.num_heads, Cq//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)
        v = self.value(v).view(Bq, -1, self.num_heads, Cq//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)

        #y = attention @ v # (B, num_heads, T, T) x (B, num_heads, T, C//num_heads) ===> (B, num_heads, T, C//num_heads)
        y = attention(q, k, v, self.d_k, mask, self.dropout)
        y = y.transpose(1, 2).contiguous().view(Bq, -1, Cq) # re-assemble all head outputs side by side (concat)
        
        # output projection
        y = self.fc(y)

        return y
