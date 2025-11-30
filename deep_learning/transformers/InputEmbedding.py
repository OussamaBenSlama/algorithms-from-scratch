import torch 
import math

class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Scale the embeddings by sqrt(d_model)
    # without scaling embeddings might have smaller values compared to other components in the transformer, which can lead to issues in training stability.