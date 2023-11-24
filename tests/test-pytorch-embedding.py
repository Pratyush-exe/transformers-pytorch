import os
import sys

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(root_folder)

import torch
from model.embeddings import PytorchEmbedding

config = {
    'vocab_size': 10,
    'embed_dims': 4
}

x = torch.randint(1, 5, (1, 5))
pytorch_embedding = PytorchEmbedding(config)
output = pytorch_embedding(x)

print("input vec: ", x, "\n")
print("output vec: ", output)
