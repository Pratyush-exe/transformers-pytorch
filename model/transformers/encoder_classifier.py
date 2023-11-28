from torch import nn
from model.embeddings import Embedding
from model import Encoder

class EncoderClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_layer = Embedding(config)
        self.encoder = Encoder(config)
        embed_dims = config['embed_dims']
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(embed_dims, config['num_classes']),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        output = self.embed_layer(x)
        output = self.encoder(output)
        output = output.mean(dim=0)
        output = self.classifier_head(output)
        
        return output 