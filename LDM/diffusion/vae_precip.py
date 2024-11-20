from vae.VAE import Encoder, Decoder
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, num_variables, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(num_variables, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        self.decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, out_channels=num_variables)

    def forward(self, x):
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        x_hat = self.decoder(z)
        return x_hat

