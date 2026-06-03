import torch
import torch.nn as nn

class PEMP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, n_max=10, m_max=10, num_layers=3, batch_size=32):
        super(PEMP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_max = n_max
        self.m_max = m_max
        self.num_layers = num_layers
        self.batch_size = batch_size

        enc_layers = []
        for i in range(num_layers-1):
            if i == 0:
                enc_layers.append(nn.Linear(2 * (input_dim + output_dim), hidden_dim))  # Adjusted for relative positions
            else:
                enc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            enc_layers.append(nn.ReLU())
        enc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(num_layers-1):
            if i == 0:
                dec_layers.append(nn.Linear(2 * input_dim + hidden_dim, hidden_dim))  # Adjusted for relative positions
            else:
                dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
            dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(hidden_dim, 2 * output_dim))
        self.decoder = nn.Sequential(*dec_layers)


    def forward(self, obs, tar):
        # obs: (batch_size, n_o (<n_max_obs), 2*input_dim+output_dim)
        # tar: (batch_size, n_t (<n_max_tar), 2*input_dim)
        encoded_obs = self.encoder(obs)  # (batch_size, n_o (<n_max_obs), hidden_dim)
        encoded_rep = encoded_obs.mean(dim=1).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        repeated_encoded_rep = torch.repeat_interleave(encoded_rep, tar.shape[1], dim=1)  # each encoded_rep is repeated to match tar
        rep_tar = torch.cat([repeated_encoded_rep, tar], dim=-1)
        
        pred = self.decoder(rep_tar)  # (batch_size, n_t (<n_max_tar), 2*output_dim)        
        return pred, encoded_rep

    def loss(self, pred, real):
        # pred: (batch_size, n_t (<n_max_tar), 2*output_dim)
        # real: (batch_size, n_t (<n_max_tar), output_dim)
        pred_mean = pred[:, :, :self.output_dim]
        pred_std = torch.nn.functional.softplus(pred[:, :, self.output_dim:]) +1e-6 # predicted value is std. In comb. with softplus to ensure positivity

        pred_dist = torch.distributions.Normal(pred_mean, pred_std)

        return (-pred_dist.log_prob(real)).mean()