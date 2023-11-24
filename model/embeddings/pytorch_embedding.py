from torch import nn

class PytorchEmbedding(nn.Module):
    """class for pytorch nn.Embedding
    """
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        embed_dims = config['embed_dims']
        
        self.embed = nn.Embedding(vocab_size, embed_dims)
        
    def forward(self, x):
        """forward function for PytorchEmbedding

        Args:
            x (Tensor): input vector

        Returns:
            output (Tensor): output vector
        """
        output = self.embed(x)
        return output