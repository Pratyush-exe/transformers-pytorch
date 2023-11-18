import os
import sys

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_folder)

import torch
from model.attentions import MultiHeadAttention

embed_dim = 4
heads = 2
dropout = 0.3

batch_size = 3
seq_len = 3

k = torch.rand((batch_size, seq_len, embed_dim))
q = torch.rand((batch_size, seq_len, embed_dim))
v = torch.rand((batch_size, seq_len, embed_dim))

mha = MultiHeadAttention(embed_dims=embed_dim, heads=heads, dropout=dropout)

print("\nUnmasked Multi-head Attention")
unmasked_mha_output = mha(k, q, v, mask=False)
assert unmasked_mha_output.shape == k.shape, "shapes not match"
print(unmasked_mha_output.shape)
print(mha.attention_scores[0][0])

print("\nMasked Multi-head Attention")
masked_mha_output = mha(k, q, v, mask=True)
assert masked_mha_output.shape == k.shape, "shapes not match"
print(masked_mha_output.shape)
print(mha.attention_scores[0][0])