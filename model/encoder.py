from torch import nn, Tensor
from attentions import *


class EncoderLayer(nn.Module):
    """Single Encoder layer class for transformer
    """
    def __init__(self, config):
        super().__init__()
        
        self.embed_dims = config['embed_dims']
        self.heads = config['heads']
        self.dropout = config['dropout']
        self.device = config['device']
        ff_layer_sizes = config['ff_layer_sizes']
        self.device = config['device']
        
        self.mha_layer_1 = MultiHeadAttention(
            embed_dims=self.embed_dims, 
            heads=self.heads, 
            dropout=self.dropout
        )
        
        self.norm_layer_1 = nn.LayerNorm()
        self.norm_layer_2 = nn.LayerNorm()
        self.feed_forward = self.create_sequential_model(ff_layer_sizes)
        
    def forward(self, 
        key: Tensor,
        query: Tensor,
        value: Tensor,
        mask: bool=False
    ):
        """forward function for single encoder layer

        Args:
            key (Tensor): (batch, seq_len, embed_dims)
            query (Tensor): (batch, seq_len, embed_dims)
            value (Tensor): (batch, seq_len, embed_dims)
            mask (bool, optional): Defaults to False.
        """
        
        mha_output = self.mha_layer_1(key, query, value, mask)
        add_output_1 = mha_output + value
        norm_output_1 = self.norm_layer_1(add_output_1)
        ff_output = self.feed_forward(norm_output_1)
        add_output_2 = ff_output + norm_output_1
        norm_output_2 = self.norm_layer_2(add_output_2)
        
        return norm_output_2
        
    def create_sequential_model(sizes):
        """given list of sizes, returns nn.Sequential layer

        Args:
            sizes (list): list of sizes

        Returns:
            nn.Sequential: sequential layer made of nn.Linear layers
        """
        if len(sizes) < 2:
            raise ValueError("The list of sizes should have at least two elements.")

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        return model


class Encoder(nn.Module):
    ...
    
