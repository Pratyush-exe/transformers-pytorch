import os
import sys

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_folder)

import torch
from model import Encoder

embed_dims = 4
batch_size = 3
seq_len = 3

config = {
    'embed_dims': embed_dims,
    'heads': 2,
    'dropout': 0.2,
    'device': 'cpu',
    'num_layers': 5,
    'vocab_size': 10,
    'max_seq_len': 10,
    'ff_layer_sizes': [4, 16, 32, 4]
}

x = torch.randint(1, 5, (seq_len, embed_dims))
encoder_1 = Encoder(config)

encoder_output = encoder_1(x)
print(encoder_output.shape)

print("\n", encoder_1)