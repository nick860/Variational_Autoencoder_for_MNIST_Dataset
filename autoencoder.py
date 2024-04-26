import torch
import torch.nn.functional as F
from torch import nn

# Input img -> Hidden dim -> Latent dim (mean, std + Parametrization trick) -> Hidden dim -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        # one possible architecture for the VAE, you can modify it to CNN if you want but it's not necessary 
        # because the MNIST dataset is simple, you can use a simple feedforward neural network
        super(VariationalAutoEncoder, self).__init__()
        # encoder
        self.image_input = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.latent = nn.Linear(latent_dim, hidden_dim)
        self.image_output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
    
    def encode(self, x):
        """
        encode the input image
        param x: tensor
        return: tensor, tensor - mu, sigma
        """
        h = self.relu(self.image_input(x))
        mu, sigma = self.mu(h), self.sigma(h)
        return mu, sigma
    
    def decode(self, z):
        """
        decode the latent representation
        param z: tensor
        return: tensor
        """
        h = self.relu(self.latent(z))
        return torch.sigmoid(self.image_output(h))
    
    def forward(self, x):
        """
        forward pass with normal distribution and reparametrization trick
        param x: tensor
        return: tensor, tensor, tensor - x_reconstructed, mu, sigma
        """
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma) # sample from normal distribution
        z = mu + sigma * epsilon
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma
    
