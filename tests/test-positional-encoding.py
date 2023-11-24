import os
import sys

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_folder)

import torch
from model.embeddings import PositionalEmbedding

config = {
    'max_seq_len': 5,
    'embed_dims': 2
}

x = torch.rand((1, 4, 2))
pos_embed = PositionalEmbedding(config)
output = pos_embed(x)

print("input vec: ", x, "\n")
print("output vec: ", output)