import torch.nn as nn
import torch.nn.functional as F
from .InputEmbedding import InputEmbedding
from .PositionalEncoding import PositionalEncoding
from .DecoderLayer import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self,x,y, tgt_mask=None, cross_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:  
            x = layer(x,y, tgt_mask, cross_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)