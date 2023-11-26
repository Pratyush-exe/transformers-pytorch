from torch import nn

class Embedding(nn.Module):
    """class for pytorch nn.Embedding
    """
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        embed_dims = config['embed_dims']
        
        self.embed_1 = nn.Embedding(vocab_size, embed_dims)
        self.embed_2 = nn.Embedding(vocab_size, embed_dims)
        
    def forward(self, x):
        """forward function for Embedding

        Args:
            x (Tensor): input vector

        Returns:
            output (Tensor): output vector
        """
        output = self.embed_1(x)
        output += self.embed_2(x)
        return output