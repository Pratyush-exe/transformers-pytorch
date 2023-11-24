import os
import sys

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_folder)

import torch
from model import EncoderLayer

embed_dims = 4
batch_size = 3
seq_len = 3

config = {
    'embed_dims': embed_dims,
    'heads': 2,
    'dropout': 0.2,
    'device': 'cpu',
    'ff_layer_sizes': [4, 16, 32, 4]
}

k = torch.rand((batch_size, seq_len, embed_dims))
q = torch.rand((batch_size, seq_len, embed_dims))
v = torch.rand((batch_size, seq_len, embed_dims))

encoder_layer_1 = EncoderLayer(config)

print("\nUnmasked Single Encoder Layer")
unmasked_mha_output = encoder_layer_1(k, q, v, mask=False)
assert unmasked_mha_output.shape == k.shape, "shapes not match"
print(unmasked_mha_output.shape)
print(encoder_layer_1.mha_layer_1.attention_scores[0][0])

print("\nMasked Single Encoder Layer")
masked_mha_output = encoder_layer_1(k, q, v, mask=True)
assert masked_mha_output.shape == k.shape, "shapes not match"
print(masked_mha_output.shape)
print(encoder_layer_1.mha_layer_1.attention_scores[0][0])

print(encoder_layer_1)