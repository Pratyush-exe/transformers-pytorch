import math

import torch
import torch.nn as nn
from torch import Tensor
        
        
class SelfAttention:
    """Class to calculate self-attention
    """
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        
    def generate_mask(self, vector_shape):
        """Generates mask for a given vector_shape (batch, heads, seq_len, head_dims)

        Args:
            vector_shape (tuple): input vector shape

        Returns:
            mask: mask for masked-multi-head-attention
        """
        # (batch, heads, seq_len, head_dims)
        
        seq_length = vector_shape[-2]
        required_shape = (*vector_shape[:-2], seq_length, seq_length)
        
        temp_matrix = torch.full(required_shape, fill_value=-1e5)
        mask = torch.triu(temp_matrix, diagonal=1)
        mask = mask.type(torch.int)
        
        return mask
        
    def calculate(self, 
        key: Tensor, 
        query: Tensor,
        value: Tensor, 
        mask: bool=False
    ):
        """calculate function for SelfAttention class

        Args:
            key (Tensor): (batch, heads, seq_len, head_dims)
            query (Tensor): (batch, heads, seq_len, head_dims)
            value (Tensor): (batch, heads, seq_len, head_dims)
            mask (bool, optional): Defaults to False.
        """
        key_t = key.transpose(-2, -1)
        attention_scores = (query @ key_t) / math.sqrt(self.embed_dims)
        
        mask_matrix = torch.zeros_like(attention_scores) if not mask else self.generate_mask(tuple(key.shape))
        attention_scores = attention_scores + mask_matrix
        
        attention_scores = attention_scores.softmax(dim=-1)
        attention_values = attention_scores @ value
        
        return attention_scores, attention_values
        
    
class MultiHeadAttention(nn.Module):
    """Class to calculate multi-head-attentions
    """
    def __init__(
            self, 
            embed_dims: int = 256,
            heads: int = 8, 
            dropout: float = .0, 
            device: str = "cpu"
        ):
        super().__init__()
        self.embed_dims = embed_dims
        self.heads = heads
        
        assert embed_dims % heads == 0, "embed_dims sould be divisible by heads"
        self.head_dims = embed_dims // heads
        
        self.key_layer = nn.Linear(embed_dims, embed_dims, bias=False, device=device)
        self.query_layer = nn.Linear(embed_dims, embed_dims, bias=False, device=device)
        self.value_layer = nn.Linear(embed_dims, embed_dims, bias=False, device=device)
        self.output_layer = nn.Linear(embed_dims, embed_dims, bias=False, device=device)
        
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def split_heads(self, input_tensor):
        """
        Splits given input tensors of size 
        (batch, seq_len, embed_dims) to (batch, seq_len, heads, head_dims) 
        where embed_dims = heads * head_dims and then finally to size
        (batch, heads, seq_len, head_dims) for parallelizing calulations

        Args:
            input_tensor (torch.Tensor): input tensor
        """
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]
        
        # (batch, seq_len, embed_dims) -> (batch, seq_len, heads, head_dims)
        # -> (batch, heads, seq_len, head_dims)
        
        input_tensor = input_tensor.view(batch_size, seq_len, self.heads, self.head_dims)
        input_tensor = input_tensor.transpose(1, 2)
        
        return input_tensor
        
    def concatinate_heads(self, attentions):
        """
        concatinates given input tensors of size 
        (batch, heads, seq_len, head_dims) to (batch, seq_len, embed_dims)

        Args:
            attentions (torch.Tensor): attentions tensor
        """
        
        batch_size = attentions.shape[0]
        seq_len = attentions.shape[-2]
        
        # (batch, heads, seq_len, head_dims) -> (batch, seq_len, heads, head_dims)
        # -> (batch, seq_len, embed_dims) 
        
        concat_attentions = attentions.transpose(1, 2)
        concat_attentions = concat_attentions.contiguous()
        concat_attentions = concat_attentions.view(batch_size, seq_len, self.embed_dims)
        
        return concat_attentions
        
    def forward(self, 
        key: Tensor,
        query: Tensor,
        value: Tensor,
        mask: bool=False
    ):
        """forward function for MultiHeadAttention class

        Args:
            key (Tensor): (batch, seq_len, embed_dims)
            query (Tensor): (batch, seq_len, embed_dims)
            value (Tensor): (batch, seq_len, embed_dims)
            mask (bool, optional): Defaults to False.
        """
        
        key = self.key_layer(key)
        query = self.query_layer(query)
        value = self.value_layer(value)
        
        key = self.split_heads(key)
        query = self.split_heads(query)
        value = self.split_heads(value)
        
        # (batch, heads, seq_len, head_dims)
        sa = SelfAttention(self.embed_dims)
        attention_scores, attention_values = sa.calculate(key, query, value, mask)
        self.attention_scores = attention_scores
        
        # (batch, heads, seq_len, head_dims) -> (batch, seq_len, head_dims)
        output = self.concatinate_heads(attention_values)
        output = self.output_layer(output)
        
        return output
        
        
        
        
        