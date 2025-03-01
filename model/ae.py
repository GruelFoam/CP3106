import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, drop_rate):
        super(AE, self).__init__()
        
        # Encoder network (module)
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),  # Layer Norm to stabilize training
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Norm to stabilize training
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(hidden_dim, latent_dim),  # Outputs mean and log-variance
            nn.Tanh()
        )

        # Decoder network (with Layer Normalization)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Norm to stabilize training
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1024, input_dim),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder_net(x)
        return h

    
    def decode(self, z):
        x_reconstructed = self.decoder_net(z)
        return x_reconstructed

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed