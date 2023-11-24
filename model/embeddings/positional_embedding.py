from torch import nn
from torch.autograd import Variable
import torch
import math

class PositionalEmbedding(nn.Module):
    """class for positional embedding based on
        "Attention is all you need paper"
    """
    def __init__(self, config):
        super().__init__()
        max_seq_len = config['max_seq_len']
        self.embed_dims = config['embed_dims']
        
        positional_embedding = torch.zeros(max_seq_len, self.embed_dims)
        
        for position in range(max_seq_len):
            for i in range(0, self.embed_dims, 2):
                even_angle = position / (10000 ** ((2 * i) / self.embed_dim))
                positional_embedding[position, i] = math.sin(even_angle)
                
                odd_angle = position / (10000 ** ((2 * (i+1)) / self.embed_dim))
                positional_embedding[position, i+1] = math.cos(odd_angle)
        
        positional_embedding = positional_embedding.unsqueeze(0)
        self.register_buffer(positional_embedding)
        
    def forward(self, x):
        x = x * math.sqrt(self.embed_dims)
        seq_len = x.size(1)
        
        var = Variable(self.positional_embedding[:,:seq_len], requires_grad=False)
        x = x + var
        return x
        