import torch
from torch import nn



class AutoFusion(nn.Module):
    def __init__(self, latent_dim, input_features):
        super(AutoFusion, self).__init__()
        self.input_features = input_features

        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features//2),
            #nn.Tanh(),
            #nn.GELU(),
            nn.ReLU(),
            nn.Linear(input_features//2, latent_dim),
        )
    
        self.fuse_out = nn.Sequential(
            nn.Linear(latent_dim, input_features//2),
            #nn.GELU(),
            nn.ReLU(),
            nn.Linear(input_features//2, input_features))
        
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output
