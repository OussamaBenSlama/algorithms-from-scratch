import torch 
from torch.nn import functional as F
import math


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_linear = torch.nn.Linear(d_model, d_model,bias=False)
        self.key_linear = torch.nn.Linear(d_model, d_model,bias=False)
        self.value_linear = torch.nn.Linear(d_model, d_model,bias=False)
        self.out_linear = torch.nn.Linear(d_model, d_model,bias=False)
        
    def split_heads(self,x,batch_size):
        
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
    
    def compute_attention(self,query, key, value,mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)
        
    def combine_heads(self, x , batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)
        
        attention_output = self.compute_attention(query, key, value, mask)
        
        output = self.combine_heads(attention_output, batch_size)
        
        return self.out_linear(output)