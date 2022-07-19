import torch.nn as nn
from model.encoder import Encoder_Block
from model.sublayers import PositionalEncoding
from model.decoder import Decoder_Block

class LemmaEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(LemmaEncoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, source_mask=None):

        x    = self.input_embedding(input)
        x    = self.pos_encoding(x)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out  = self.encoder3(x, source_mask)

        return out

class TagEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(TagEncoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, source_mask=None):

        x    = self.input_embedding(input)
        x    = self.pos_encoding(x)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out  = self.encoder3(x, source_mask)

        return out

class SourceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(SourceEncoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, source_mask=None):

        x    = self.input_embedding(input)
        x    = self.pos_encoding(x)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out  = self.encoder3(x, source_mask)

        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.decoder1        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder2        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder3        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, word_encoder_outputs, target_mask=None, source_mask=None):

        x   = self.input_embedding(input)
        x   = self.pos_encoding(x)
        x   = self.decoder1(x, word_encoder_outputs, target_mask, source_mask)
        x   = self.decoder2(x, word_encoder_outputs, target_mask, source_mask)
        out = self.decoder3(x, word_encoder_outputs, target_mask, source_mask)

        return out

