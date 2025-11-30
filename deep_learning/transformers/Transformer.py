from torch.nn import nn
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,d_ff,num_layers,max_seq_length,dropout:float=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size,d_model,num_heads,d_ff,num_layers,max_seq_length,dropout)
        self.decoder = TransformerDecoder(vocab_size,d_model,num_layers,num_heads,d_ff,dropout,max_seq_length)
    def forward(self,x,src_mask=None,tgt_mask=None,cross_mask=None):
        encoder_output = self.encoder(x,src_mask)
        decoder_output = self.decoder(x,encoder_output,tgt_mask,cross_mask)
        return decoder_output
        