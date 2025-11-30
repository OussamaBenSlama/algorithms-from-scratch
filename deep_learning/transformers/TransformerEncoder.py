from torch import nn
from InputEmbedding import InputEmbedding
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int,max_seq_length :int, dropout: float = 0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x