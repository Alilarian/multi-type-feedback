import torch
import numpy as np
import wandb

from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.special import gammaln
from torch.amp import autocast
from scipy.special import logsumexp

def predict_reward_sequence(net, segment, log_to_wandb=False):
    """
    Predict per-state rewards for a segment from the Highway environment and extract true rewards.
    
    Args:
        net: Net model instance (input_dim=25, action_dims=5).
        segment: List of (state, action, reward, done) tuples, where state has shape (1, 5, 5).
        log_to_wandb: If True, log true vs. predicted rewards to W&B.
    
    Returns:
        rewards_from_obs: List of predicted per-state rewards (scalars).
        true_rewards: List of true per-state rewards (scalars).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Extract all states and true rewards (no filtering based on done)
    valid_states = [s for s, _, r, _ in segment]  # Shape: List of (1, 5, 5) arrays
    true_rewards = [r for _, _, r, _ in segment]  # Shape: List of scalars
    
    if not valid_states:
        return [], []
    
    # Convert states to tensor
    traj = torch.tensor(np.array(valid_states), dtype=torch.float32).to(device)  # Shape: (T, 1, 5, 5)
    
    with torch.no_grad():
        # Compute predicted rewards for all states in a single forward pass
        rewards, _, _, _, _ = net.cum_return(traj)  # rewards: (T, 1)
        rewards_from_obs = rewards.squeeze(-1)  # Shape: (T,) or scalar if T=1
        # Ensure rewards_from_obs is a list
        if rewards_from_obs.dim() == 0:  # Single state case
            rewards_from_obs = [rewards_from_obs.item()]
        else:
            rewards_from_obs = rewards_from_obs.cpu().numpy().tolist()  # Shape: (T,)
    
    # Log to W&B if requested
    if log_to_wandb:
        wandb.log({
            "per_state_true_rewards": true_rewards,
            "per_state_predicted_rewards": rewards_from_obs,
            "per_state_mae": np.mean(np.abs(np.array(true_rewards) - np.array(rewards_from_obs))) if true_rewards else 0.0
        })
    
    return rewards_from_obs, true_rewards

def predict_traj_return(net, segment, log_to_wandb=False):
    """
    Predict the cumulative return for a segment from the Highway environment and compute true return.
    
    Args:
        net: Net model instance (input_dim=25, action_dims=5).
        segment: List of (state, action, reward, done) tuples, where state has shape (1, 5, 5).
        log_to_wandb: If True, log true vs. predicted cumulative return to W&B.
    
    Returns:
        pred_cum_return: Scalar, the sum of predicted per-state rewards.
        true_cum_return: Scalar, the sum of true per-state rewards.
    """
    rewards, true_rewards = predict_reward_sequence(net, segment, log_to_wandb)
    pred_cum_return = sum(rewards) if rewards else 0.0
    true_cum_return = sum(true_rewards) if true_rewards else 0.0
    
    # Log to W&B if requested
    if log_to_wandb:
        wandb.log({
            "true_cumulative_return": true_cum_return,
            "predicted_cumulative_return": pred_cum_return,
            "cumulative_mae": abs(true_cum_return - pred_cum_return)
        })
    
    return pred_cum_return, true_cum_return

def predict_all_segments(net, segments, log_to_wandb=False):
    """
    Predict per-state and cumulative returns for all segments and compare with true rewards.
    
    Args:
        net: Net model instance (input_dim=25, action_dims=5).
        segments: List of segments, each a list of (state, action, reward, done) tuples.
        log_to_wandb: If True, log results to W&B.
    
    Returns:
        all_pred_rewards: List of lists, predicted per-state rewards for each segment.
        all_true_rewards: List of lists, true per-state rewards for each segment.
        all_pred_cum_returns: List of predicted cumulative returns for each segment.
        all_true_cum_returns: List of true cumulative returns for each segment.
    """
    all_pred_rewards = []
    all_true_rewards = []
    all_pred_cum_returns = []
    all_true_cum_returns = []
    
    # For W&B logging
    if log_to_wandb:
        table_data = []
    
    for idx, segment in enumerate(segments):
        # Predict rewards and cumulative return for the segment
        pred_rewards, true_rewards = predict_reward_sequence(net, segment, log_to_wandb=False)  # Avoid redundant per-segment logging
        pred_cum_return, true_cum_return = predict_traj_return(net, segment, log_to_wandb=False)
        
        # Store results
        all_pred_rewards.append(pred_rewards)
        all_true_rewards.append(true_rewards)
        all_pred_cum_returns.append(pred_cum_return)
        all_true_cum_returns.append(true_cum_return)
        
        # Prepare W&B table data
        if log_to_wandb:
            per_state_mae = np.mean(np.abs(np.array(true_rewards) - np.array(pred_rewards))) if true_rewards else 0.0
            cum_mae = abs(true_cum_return - pred_cum_return)
            table_data.append([idx, true_cum_return, pred_cum_return, cum_mae, per_state_mae])
    
    # Log aggregated results to W&B
    if log_to_wandb:
        # Create a W&B table for per-segment results
        table = wandb.Table(columns=["segment_idx", "true_cum_return", "pred_cum_return", "cumulative_mae", "per_state_mae"])
        for row in table_data:
            table.add_data(*row)
        
        # Log table and overall metrics
        wandb.log({
            "segment_results_table": table,
            "overall_per_state_mae": np.mean([np.mean(np.abs(np.array(true) - np.array(pred))) for true, pred in zip(all_true_rewards, all_pred_rewards) if true]),
            "overall_cumulative_mae": np.mean([abs(t - p) for t, p in zip(all_true_cum_returns, all_pred_cum_returns) if t != 0.0]),
            "true_cum_returns": all_true_cum_returns,
            "pred_cum_returns": all_pred_cum_returns
        })
    
    return all_pred_rewards, all_true_rewards, all_pred_cum_returns, all_true_cum_returns

def calc_accuracy(reward_network, training_inputs, training_outputs, log_to_wandb=False):

    """
    Calculate the accuracy of the reward network in predicting trajectory preferences.
    
    Args:
        reward_network: Net model instance (input_dim=25, action_dims=5).
        training_inputs: List of (traj_i, traj_j) tuples, each traj of shape (T', 1, 5, 5).
        training_outputs: List of binary labels (1 if traj_i better, 0 if traj_j better).
        log_to_wandb: If True, log accuracy to W&B.
    
    Returns:
        accuracy: Fraction of correctly predicted preferences (float).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_network.eval()  # Set model to evaluation mode
    
    num_correct = 0.0
    total_valid_pairs = 0
    
    with torch.no_grad():
        for i, (traj_pair, label) in enumerate(zip(training_inputs, training_outputs)):
            traj_i, traj_j = traj_pair
            if len(traj_i) < 2 or len(traj_j) < 2:  # Need at least 2 steps for dynamics
                continue
            
            total_valid_pairs += 1
            traj_i = torch.tensor(traj_i, dtype=torch.float32).to(device)  # (T_i', 1, 5, 5)
            traj_j = torch.tensor(traj_j, dtype=torch.float32).to(device)  # (T_j', 1, 5, 5)
            label = torch.tensor([label], dtype=torch.long).to(device)  # 1 if traj_i better
            
            # Forward pass to get cumulative rewards
            outputs, _, _, _, _, _, _, _, _, _ = reward_network.forward(traj_i, traj_j)  # outputs: (2,)
            #print(outputs)
         
            _, pred_label = torch.max(outputs, 0)  # Predict 0 (traj_i better) or 1 (traj_j better)
            
  

            # Compare predicted and true labels
            if pred_label == label.item():
                num_correct += 1.0
        
    # Compute accuracy
    accuracy = num_correct / total_valid_pairs if total_valid_pairs > 0 else 0.0
    
    # Log to W&B if requested
    if log_to_wandb:
        wandb.log({
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total_valid_pairs": total_valid_pairs
        })
    
    return accuracy


############################################################## This is with scipy
# def kozachenko_entropy(samples, epsilon=1e-10):

#     """
#     Compute entropy using the Kozachenko-Leonenko estimator.
    
#     Parameters:
#     - samples: numpy array of shape (m, d), where m is the number of samples and d is the dimension
#     - epsilon: small value to avoid log(0) issues (default: 1e-10)
    
#     Returns:
#     - entropy: estimated differential entropy
#     """
#     m, d = samples.shape
#     if m < 2:
#         raise ValueError("Need at least 2 samples for entropy estimation.")

#     # Compute pairwise Euclidean distances
#     distances = cdist(samples, samples, metric='euclidean')
#     # Set diagonal to infinity to exclude self-distances
#     np.fill_diagonal(distances, np.inf)
#     # Get nearest neighbor distances
#     rho = np.min(distances, axis=1)

#     # Replace zero distances with epsilon to avoid log(0)
#     if np.any(rho == 0):
#         print("Warning: Zero distances found; replacing with epsilon.")
#         rho = np.maximum(rho, epsilon)

#     # Constants for the KL estimator
#     euler_mascheroni = 0.5772156649  # Euler-Mascheroni constant
#     log_ball_volume = gammaln(d / 2 + 1) - (d / 2) * np.log(np.pi)  # Log of volume of d-dimensional unit ball

#     # Compute entropy
#     entropy = (d / m) * np.sum(np.log(rho)) + log_ball_volume + np.log(m - 1) + euler_mascheroni
#     return entropy


############################################################## This is with pytorch
# def kozachenko_entropy(samples, epsilon=1e-10):
#     """
#     Compute entropy using the Kozachenko-Leonenko estimator with PyTorch.
    
#     Parameters:
#     - samples: torch tensor of shape (m, d), where m is the number of samples and d is the dimension
#     - epsilon: small value to avoid log(0) issues (default: 1e-10)
    
#     Returns:
#     - entropy: estimated differential entropy (float)
#     """
#     if not isinstance(samples, torch.Tensor):
#         raise ValueError("Input samples must be a PyTorch tensor.")
    
#     m, d = samples.shape
#     if m < 2:
#         raise ValueError("Need at least 2 samples for entropy estimation.")

#     # Compute pairwise Euclidean distances
#     distances = torch.cdist(samples, samples, p=2)
    
#     # Set diagonal to infinity to exclude self-distances
#     distances.diagonal().fill_(float('inf'))
    
#     # Get nearest neighbor distances
#     rho = torch.min(distances, dim=1).values

#     # Replace zero distances with epsilon to avoid log(0)
#     if torch.any(rho == 0):
#         print("Warning: Zero distances found; replacing with epsilon.")
#         rho = torch.maximum(rho, torch.tensor(epsilon, dtype=rho.dtype, device=rho.device))

#     # Constants for the KL estimator
#     euler_mascheroni = 0.5772156649  # Euler-Mascheroni constant
#     # Log of volume of d-dimensional unit ball; using torch.lgamma for gammaln equivalent
#     log_ball_volume = torch.lgamma(torch.tensor(d / 2 + 1)) - (d / 2) * torch.log(torch.tensor(torch.pi))
    
#     # Compute entropy
#     entropy = (float(d) / m) * torch.sum(torch.log(rho)) + log_ball_volume + torch.log(torch.tensor(m - 1)) + euler_mascheroni
    
#     return entropy.item()  # Return as float for consistency with original

# import torch

############################################################## This is with pytorch and batching
def kozachenko_entropy(samples, epsilon=1e-10, batch_size=100):
    """
    Compute entropy using the Kozachenko-Leonenko estimator with PyTorch and batching to avoid OOM.
    
    Parameters:
    - samples: torch tensor of shape (m, d), where m is the number of samples and d is the dimension
    - epsilon: small value to avoid log(0) issues (default: 1e-10)
    - batch_size: int, size of query batches for memory efficiency (default: 1000; adjust based on GPU memory)
    
    Returns:
    - entropy: estimated differential entropy (float)
    """
    if not isinstance(samples, torch.Tensor):
        raise ValueError("Input samples must be a PyTorch tensor.")
    
    m, d = samples.shape
    if m < 2:
        raise ValueError("Need at least 2 samples for entropy estimation.")
    
    device = samples.device
    dtype = samples.dtype
    
    # Pre-allocate rho
    rho = torch.empty(m, dtype=dtype, device=device)
    
    # Batch over query points
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        query_batch = samples[start:end]
        
        # Compute distances: (batch_size, m)
        dists = torch.cdist(query_batch, samples, p=2)
        
        # Mask self-distances (diagonal elements in this sub-matrix)
        for j in range(end - start):
            dists[j, start + j] = float('inf')
        
        # Get min distances for this batch
        rho[start:end] = torch.min(dists, dim=1).values
    
    # Replace zero distances with epsilon
    if torch.any(rho == 0):
        print("Warning: Zero distances found; replacing with epsilon.")
        rho = torch.maximum(rho, torch.tensor(epsilon, dtype=dtype, device=device))
    
    # Constants for the KL estimator
    euler_mascheroni = 0.5772156649  # Euler-Mascheroni constant
    # Log of volume of d-dimensional unit ball
    log_ball_volume = torch.lgamma(torch.tensor(d / 2 + 1, device=device)) - (d / 2) * torch.log(torch.tensor(torch.pi, device=device))
    
    # Compute entropy
    entropy = (float(d) / m) * torch.sum(torch.log(rho)) + log_ball_volume + torch.log(torch.tensor(m - 1, device=device)) + euler_mascheroni
    
    return entropy.item()  # Return as float



def compute_entropy_importance_sampling(Phi_tau_i, Phi_tau_j, samples, beta, device='cuda:0', batch_size_samples=5000, batch_size_pairs=5000):
    num_mcmc_samples = len(samples)
    
    # Store samples on CPU to save GPU memory
    #samples_tensor = torch.tensor(samples, dtype=torch.float16, device='cuda:0')  # Shape: (num_samples, encoding_dims)
    samples_tensor = samples
    # Define batched log likelihood function
    def log_likelihood_vectorized(w_batch, Phi_tau_i, Phi_tau_j, beta):
        log_probs = []
        with autocast(device_type=device):
            for j in range(0, len(Phi_tau_i), batch_size_pairs):
                batch_i = Phi_tau_i[j:j+batch_size_pairs].to(device)  # Shape: (batch_size_pairs, encoding_dims)
                batch_j = Phi_tau_j[j:j+batch_size_pairs].to(device)
                R_i = torch.matmul(batch_i, w_batch.T)  # Shape: (batch_size_pairs, batch_size_samples)
                R_j = torch.matmul(batch_j, w_batch.T)
                beta_R_i = beta * R_i
                beta_R_j = beta * R_j
                max_val = torch.maximum(beta_R_i, beta_R_j)
                log_sum_exp = max_val + torch.log(
                    torch.exp(beta_R_i - max_val) + torch.exp(beta_R_j - max_val)
                )
                terms = beta * R_i - log_sum_exp
                log_probs.append(torch.sum(terms, dim=0))  # Shape: (batch_size_samples,)
                del batch_i, batch_j, R_i, R_j, beta_R_i, beta_R_j, max_val, log_sum_exp, terms
                torch.cuda.empty_cache()
        return torch.sum(torch.stack(log_probs), dim=0)  # Shape: (batch_size_samples,)
    
    # Compute log_probs in batches
    log_probs = []
    for i in range(0, num_mcmc_samples, batch_size_samples):
        w_batch = samples_tensor[i:i+batch_size_samples].to(device)  # Shape: (batch_size_samples, encoding_dims)
        batch_log_probs = log_likelihood_vectorized(w_batch, Phi_tau_i, Phi_tau_j, beta)
        print("inside the iterating over mcmc samples ")
        log_probs.append(batch_log_probs.cpu().numpy())
        del w_batch, batch_log_probs
        torch.cuda.empty_cache()
    
    log_probs = np.concatenate(log_probs)  # Shape: (num_mcmc_samples,)
    
    # Check for invalid values
    if np.any(np.isnan(log_probs)) or np.any(np.isinf(log_probs)):
        raise ValueError("Log probabilities contain NaN or Inf values, indicating numerical instability.")
    
    # Compute entropy terms
    first_term = -np.mean(log_probs)
    second_term = -logsumexp(-log_probs) + np.log(num_mcmc_samples)
    entropy = first_term + second_term
    
    return entropy

# Compute entropy using importance sampling with numerical stability

# def compute_entropy_importance_sampling(Phi_tau_i, Phi_tau_j, samples, beta, device):
#     num_mcmc_samples = len(samples)
    
#     # Convert samples to a tensor for vectorized computation
#     samples_tensor = torch.tensor(samples, dtype=torch.float16).to(device)  # Shape: (num_samples, encoding_dims)
    
#     # Define the log likelihood function with numerical stability
#     def log_likelihood_vectorized(w_batch, Phi_tau_i, Phi_tau_j, beta):
#         # w_batch = w_batch.to('cpu')  # Move to CPU
#         # Phi_tau_i = Phi_tau_i.to('cpu')
#         # Phi_tau_j = Phi_tau_j.to('cpu')
#         R_i = torch.matmul(Phi_tau_i, w_batch.T)  # Shape: (num_pairs, num_samples)
#         R_j = torch.matmul(Phi_tau_j, w_batch.T)  # Shape: (num_pairs, num_samples)
        
#         # Numerically stable computation of log(exp(beta * R_i) + exp(beta * R_j))
#         beta_R_i = beta * R_i
#         beta_R_j = beta * R_j
#         max_val = torch.maximum(beta_R_i, beta_R_j)
#         log_sum_exp = max_val + torch.log(
#             torch.exp(beta_R_i - max_val) + torch.exp(beta_R_j - max_val)
#         )
        
#         terms = beta * R_i - log_sum_exp  # Favor R_i > R_j (better > worse)
#         log_likelihoods = torch.sum(terms, dim=0)  # Shape: (num_samples,)
#         return log_likelihoods
    
#     # Compute log likelihoods for all samples in parallel
#     log_probs = log_likelihood_vectorized(samples_tensor, Phi_tau_i, Phi_tau_j, beta)
#     log_probs = log_probs.cpu().numpy()
    
#     # Check for invalid values
#     if np.any(np.isnan(log_probs)) or np.any(np.isinf(log_probs)):
#         raise ValueError("Log probabilities contain NaN or Inf values, indicating numerical instability.")
    
#     # First term: - (1 / m) * sum(log P(D, P | w^{(k)}))
#     first_term = -np.mean(log_probs)
    
#     # Second term: -log( (1 / m) * sum( exp(-log P(D, P | w^{(k)})) ) )
#     neg_log_probs = -log_probs
#     second_term = -logsumexp(neg_log_probs) + np.log(num_mcmc_samples)
    
#     # Entropy estimate
#     entropy = first_term + second_term
#     return entropy