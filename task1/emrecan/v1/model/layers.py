import torch.nn as nn
from model.encoder import Encoder_Block
from model.sublayers import PositionalEncoding, SinusoidalPositionalEmbedding
from model.decoder import Decoder_Block

# For task3-ver1
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.sin_encoding    = SinusoidalPositionalEmbedding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, source_mask=None):

        #print(f"Input (Encoder): {input.shape}")
        x1   = self.input_embedding(input)
        #print(f"Embedding (Encoder): {x1.shape}")
        #x    = self.pos_encoding(x)
        x2   = self.sin_encoding(input)
        #print(f"Positional Encoding Output (Encoder): {x2.shape}")
        x    = self.dropout(x1 + x2)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out  = self.encoder3(x, source_mask)

        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.sin_encoding    = SinusoidalPositionalEmbedding(embed_dim)
        self.decoder1        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder2        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder3        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.dropout         = nn.Dropout(dropout_rate)

    def forward(self, input, word_encoder_outputs, target_mask=None, source_mask=None):

        #print(f"Input (Decoder): {input.shape}")
        x1  = self.input_embedding(input)
        #print(f"Embedding (Decoder): {x1.shape}")
        #x    = self.pos_encoding(x)
        x2  = self.sin_encoding(input)
        #print(f"Positional Encoding Output (Decoder): {x2.shape}")
        x   = self.dropout(x1 + x2)
        x   = self.decoder1(x, word_encoder_outputs, target_mask, source_mask)
        x   = self.decoder2(x, word_encoder_outputs, target_mask, source_mask)
        out = self.decoder3(x, word_encoder_outputs, target_mask, source_mask)

        return out


# For task3-ver2
class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(WordEncoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.encoder1        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3        = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.adaptive_pool   = nn.AdaptiveAvgPool1d(1)

    def forward(self, input, we_reshape, source_mask=None):

        x    = self.input_embedding(input)
        x    = self.pos_encoding(x)
        x    = self.encoder1(x, source_mask)
        x    = self.encoder2(x, source_mask)
        out1 = self.encoder3(x, source_mask)

        x    = out1.permute(0,2,1)
        x    = self.adaptive_pool(x).squeeze()
        out2 = x.reshape(we_reshape.shape[0],  we_reshape.shape[1], x.shape[-1])
        return out1, out2

class ContextEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(ContextEncoder, self).__init__()

        self.encoder1 = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder2 = Encoder_Block(embed_dim, num_heads, dropout_rate)
        self.encoder3 = Encoder_Block(embed_dim, num_heads, dropout_rate)

    def forward(self, input, source_mask=None):
        
        x   = self.encoder1(input, source_mask)
        x   = self.encoder2(x, source_mask)
        out = self.encoder3(x, source_mask)

        return out

class WordDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, dropout_rate=0.1):
        super(WordDecoder, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)
        self.decoder1        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder2        = Decoder_Block(embed_dim, num_heads, dropout_rate)
        self.decoder3        = Decoder_Block(embed_dim, num_heads, dropout_rate)

    def forward(self, input, word_encoder_outputs, target_mask=None, source_mask=None):

        x   = self.input_embedding(input)
        x   = self.pos_encoding(x)
        x   = self.decoder1(x, word_encoder_outputs, target_mask, source_mask)
        x   = self.decoder2(x, word_encoder_outputs, target_mask, source_mask)
        out = self.decoder3(x, word_encoder_outputs, target_mask, source_mask)

        return out
