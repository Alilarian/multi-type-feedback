import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

class Net(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, encoding_dims=10, action_dims=5):
        super(Net, self).__init__()
        
        # Encoder (MLP for flattened 5x5 state)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # input_dim = 5 * 5 = 25
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, encoding_dims)
        self.fc_var = nn.Linear(hidden_dim, encoding_dims)  # Outputs log-variance
        
        # Decoder (reconstructs flattened 5x5 state)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output 25-dim state
        )

        # Reward predictor
        self.fc_reward = nn.Linear(encoding_dims, 1)

        # Temporal difference
        self.temporal_difference = nn.Linear(2 * encoding_dims, 1, bias=False)
        
        # Inverse dynamics (predicts 5 discrete actions)
        self.inverse_dynamics = nn.Linear(2 * encoding_dims, action_dims, bias=False)
        
        # Forward dynamics (takes 5-dim one-hot actions)
        self.forward_dynamics = nn.Linear(encoding_dims + action_dims, encoding_dims, bias=False)
        
        # Normal distribution for VAE reparameterization
        self.normal = tdist.Normal(0, 1)
        
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def reparameterize(self, mu, var):
        """Reparameterize for VAE: z = mu + std * eps"""
        if self.training:
            std = var.mul(0.5).exp()
            eps = self.normal.sample(mu.shape).to(self.device)
            return eps.mul(std).add(mu)
        else:
            return mu

    def encode(self, x):
        """Encode state to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        var = self.fc_var(h)  # Log-variance
        z = self.reparameterize(mu, var)
        return z, mu, var
    
    def decode(self, z):
        """Decode latent encoding to reconstructed state"""
        return self.decoder(z)
    
    def predict_reward(self, z):
        """Predict per-frame reward from latent encoding"""
        return self.fc_reward(z)
    
    def estimate_temporal_difference(self, z1, z2):
        """Predict time difference between two latent encodings"""
        x = torch.cat((z1, z2), dim=1)
        return self.temporal_difference(x)
 
    def estimate_inverse_dynamics(self, z1, z2):
        """Predict action (logits for 5 actions) from two consecutive latent encodings"""
        x = torch.cat((z1, z2), dim=1)
        return self.inverse_dynamics(x)

    def estimate_forward_dynamics(self, z1, action):
        """Predict next latent encoding from current encoding and one-hot action"""
        x = torch.cat((z1, action), dim=1)
        return self.forward_dynamics(x)

    def cum_return(self, traj):
        """Calculate cumulative return and related outputs for a trajectory"""
        # Reshape traj from (T, 1, 5, 5) to (T, 25)
        traj = traj.view(traj.size(0), -1)  # Shape: (T, 25)
        z, mu, var = self.encode(traj)  # Shape: (T, encoding_dims)
        rewards = self.predict_reward(z)  # Shape: (T, 1)
        sum_rewards = torch.sum(rewards)  # Scalar
        sum_abs_rewards = torch.sum(torch.abs(rewards))  # Scalar
        return sum_rewards, sum_abs_rewards, z, mu, var
    
    def forward(self, traj_i, traj_j):
        """Compute outputs for two trajectories (for T-REX ranking)"""
        # Compute cumulative returns and latent encodings
        cum_r_i, abs_r_i, z_i, mu_i, var_i = self.cum_return(traj_i)  # traj_i: (T_i, 1, 5, 5)
        cum_r_j, abs_r_j, z_j, mu_j, var_j = self.cum_return(traj_j)  # traj_j: (T_j, 1, 5, 5)
        
        # Reconstructed states
        recon_i = self.decode(z_i)  # Shape: (T_i, 25)
        recon_j = self.decode(z_j)  # Shape: (T_j, 25)
        
        # Return logits for ranking, absolute rewards, and latent encodings
        return (
            torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0),  # Logits for ranking
            abs_r_i + abs_r_j,  # Sum of absolute rewards
            z_i, z_j, mu_i, mu_j, var_i, var_j,  # Latent encodings
            recon_i, recon_j  # Reconstructed states
        )

# Function to encode preferences and compute Phi_tau_i, Phi_tau_j
def encode_preferences(net, preferences, segments, device):
    # Ensure net is frozen
    for param in net.parameters():
        param.requires_grad = False
    
    # Precompute Phi_tau (sum of embeddings over each trajectory using mean vector mu)
    Phi_tau_list = []
    for segment in segments:
        # Use all states, ignoring done
        seg_states = torch.tensor(np.array([state_t.flatten() for state_t, _, _, _ in segment]), dtype=torch.float32).to(device)  # (T, 25)
        if not seg_states.size(0):  # Check if segment is empty
            Phi_tau_list.append(None)
            continue
        with torch.no_grad():
            _, mu, _ = net.encode(seg_states)  # mu: (T, encoding_dims)
            Phi_tau = mu.sum(dim=0)  # Sum over trajectory: (encoding_dims,)
        Phi_tau_list.append(Phi_tau)
    
    # Precompute Phi_tau_i (better) and Phi_tau_j (worse) for valid preference pairs
    # Assuming preferences = [(better_idx, worse_idx, label)]
    valid_pairs = [(b, w) for b, w, _ in preferences if Phi_tau_list[b] is not None and Phi_tau_list[w] is not None]
    if not valid_pairs:
        raise ValueError("No valid preference pairs found after filtering None values.")
    
    Phi_tau_i = torch.stack([Phi_tau_list[b] for b, w in valid_pairs])  # better: (num_pairs, encoding_dims)
    Phi_tau_j = torch.stack([Phi_tau_list[w] for b, w in valid_pairs])  # worse: (num_pairs, encoding_dims)
    
    return Phi_tau_i, Phi_tau_j

# Function to encode corrections and compute Phi_tau_i, Phi_tau_j
def encode_corrections(net, corrections, device):
    # Ensure net is frozen
    for param in net.parameters():
        param.requires_grad = False
    
    # Precompute Phi_tau for better and worse segments in each correction pair
    Phi_tau_better_list = []
    Phi_tau_worse_list = []
    for worse_seg, better_seg in corrections:
        # Compute for better_seg
        #better_states = torch.tensor(np.array([state_t.flatten() for state_t, _, _, _, _ in better_seg]), dtype=torch.float32).to(device)  # (T_b, 25)
        # Compute for better_seg
        better_states = torch.tensor(np.array([item[0].flatten() for item in better_seg]), dtype=torch.float32).to(device)
        
        if not better_states.size(0):  # Check if better_seg is empty
            continue  # Skip this pair if better_seg is empty
        
        # Compute for worse_seg
        worse_states = torch.tensor(np.array([item[0].flatten() for item in worse_seg]), dtype=torch.float32).to(device)  # (T_w, 25)
        if not worse_states.size(0):  # Check if worse_seg is empty
            continue  # Skip this pair if worse_seg is empty
        
        with torch.no_grad():
            # Encode better_seg
            _, mu_better, _ = net.encode(better_states)  # mu_better: (T_b, encoding_dims)
            Phi_tau_better = mu_better.sum(dim=0)  # Sum over trajectory: (encoding_dims,)
            
            # Encode worse_seg
            _, mu_worse, _ = net.encode(worse_states)  # mu_worse: (T_w, encoding_dims)
            Phi_tau_worse = mu_worse.sum(dim=0)  # Sum over trajectory: (encoding_dims,)
        
        Phi_tau_better_list.append(Phi_tau_better)
        Phi_tau_worse_list.append(Phi_tau_worse)
    
    if not Phi_tau_better_list:
        raise ValueError("No valid correction pairs found after filtering empty segments.")
    
    Phi_tau_i = torch.stack(Phi_tau_better_list)  # better: (num_pairs, encoding_dims)
    Phi_tau_j = torch.stack(Phi_tau_worse_list)  # worse: (num_pairs, encoding_dims)
    
    return Phi_tau_i, Phi_tau_j
