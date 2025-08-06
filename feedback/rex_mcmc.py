import argparse
import os
import pickle
import torch
import numpy as np
from utils import kozachenko_entropy, compute_entropy_importance_sampling
from trex_model import Net, encode_preferences, encode_corrections
from data_prep import process_preferences

import os
import pickle
from torch.amp import autocast

# def bayesian_rex_mcmc(net, Phi_tau_i, Phi_tau_j, num_samples=10000, burn_in=1000, beta=1.0, proposal_std=0.005, update_model_with_map=False, device='cuda:0'):
#     # Get initial weights w from net (fc_reward linear layer)
#     w_current = net.fc_reward.weight.data.clone().squeeze().to(device)  # Shape: (encoding_dims,)
    
#     # Normalize initial weights to have L2 norm of 1
#     w_current = w_current / torch.norm(w_current, p=2)
    
#     # Prior: Assume a standard normal prior N(0, 1) for each weight
#     def log_prior(w):
#         return -0.5 * torch.sum(w ** 2)  # Log of N(0, 1)
    
#     # Parallelized Likelihood: Bradley-Terry model with numerical stability
#     def log_likelihood(w, Phi_tau_i, Phi_tau_j, beta):
#         R_i = torch.matmul(Phi_tau_i, w)  # Shape: (num_pairs,)
#         R_j = torch.matmul(Phi_tau_j, w)  # Shape: (num_pairs,)
        
#         beta_R_i = beta * R_i
#         beta_R_j = beta * R_j
#         max_val = torch.maximum(beta_R_i, beta_R_j)
#         log_sum_exp = max_val + torch.log(
#             torch.exp(beta_R_i - max_val) + torch.exp(beta_R_j - max_val)
#         )
        
#         terms = beta * R_i - log_sum_exp  # Favor R_i > R_j (better > worse)
#         return torch.sum(terms)
    
#     # Preallocate tensor for samples on CPU to save GPU memory
#     samples = torch.zeros((num_samples, w_current.shape[0]), dtype=torch.float32, device='cpu')
    
#     # MCMC sampling using Metropolis-Hastings with MAP tracking
#     accepted = 0
#     sample_idx = 0  # Index for filling samples tensor
#     w_map = w_current.clone()
#     log_posterior_map = log_prior(w_current) + log_likelihood(w_current, Phi_tau_i, Phi_tau_j, beta)
    
#     for step in range(num_samples + burn_in):
#         # Propose new weights
#         w_proposed = w_current + torch.normal(mean=0, std=proposal_std, size=w_current.shape).to(device)
        
#         # Normalize proposed weights to have L2 norm of 1
#         w_proposed = w_proposed / torch.norm(w_proposed, p=2)
        
#         log_posterior_current = log_prior(w_current) + log_likelihood(w_current, Phi_tau_i, Phi_tau_j, beta)
#         log_posterior_proposed = log_prior(w_proposed) + log_likelihood(w_proposed, Phi_tau_i, Phi_tau_j, beta)
        
#         if log_posterior_proposed > log_posterior_map:
#             w_map = w_proposed.clone()
#             log_posterior_map = log_posterior_proposed
        
#         log_alpha = log_posterior_proposed - log_posterior_current
#         alpha = torch.exp(log_alpha.clamp(max=0.0))  # Clamp for stability
        
#         if torch.rand(1, device=device) < alpha:
#             w_current = w_proposed
#             accepted += 1
#             if step >= burn_in:
#                 samples[sample_idx] = w_current.cpu()  # Store on CPU
#                 sample_idx += 1
        
        
#         if step % 1000 == 0:
#             acceptance_rate = accepted / (step + 1) if step > 0 else 0
#             print(f"MCMC Step {step}, Acceptance Rate: {acceptance_rate:.4f}")
    
#     print(f"Final Acceptance Rate: {accepted / (num_samples + burn_in):.4f}")
#     print(f"MAP Log Posterior: {log_posterior_map.item()}")
    
#     # Trim samples if not all were filled (e.g., due to low acceptance)
#     samples = samples[:sample_idx]
    
#     # Compute entropy
#     entropy = compute_entropy_importance_sampling(Phi_tau_i, Phi_tau_j, samples, beta, device=device)
#     sample_based_entropy = kozachenko_entropy(samples.numpy())  # Convert to NumPy only if required
    
#     w_map = w_map.cpu().detach().numpy()

#     if update_model_with_map:
#         # Update net with MAP weights
#         net.fc_reward.weight.data = torch.from_numpy(w_map).unsqueeze(0).to(device)  # (1, encoding_dims)

#     return samples, w_map, entropy, sample_based_entropy

def bayesian_rex_mcmc(net, 
                      Phi_tau_i, 
                      Phi_tau_j, 
                      num_samples=2000000, 
                      burn_in=1000, 
                      beta=1.0, 
                      proposal_std=0.005, 
                      update_model_with_map=False, 
                      device='cuda:0', 
                      batch_size=2000, 
                      thinning=2):
    # Get initial weights w from net (fc_reward linear layer)
    w_current = net.fc_reward.weight.data.clone().squeeze().to(device)  # Shape: (encoding_dims,)
    
    # Normalize initial weights to have L2 norm of 1
    w_current = w_current / torch.norm(w_current, p=2)
    
    # Prior: Assume a standard normal prior N(0, 1) for each weight
    def log_prior(w):
        return -0.5 * torch.sum(w ** 2)  # Log of N(0, 1)
    
    # Parallelized Likelihood with batching
    def log_likelihood(w, Phi_tau_i, Phi_tau_j, beta, batch_size=batch_size):
        terms = []
        with autocast(device_type='cuda'):
            for i in range(0, len(Phi_tau_i), batch_size):
                batch_i = Phi_tau_i[i:i+batch_size]
                batch_j = Phi_tau_j[i:i+batch_size]
                R_i = torch.matmul(batch_i, w)  # Shape: (batch_size,)
                R_j = torch.matmul(batch_j, w)  # Shape: (batch_size,)
                beta_R_i = beta * R_i
                beta_R_j = beta * R_j
                max_val = torch.maximum(beta_R_i, beta_R_j)
                log_sum_exp = max_val + torch.log(
                    torch.exp(beta_R_i - max_val) + torch.exp(beta_R_j - max_val)
                )
                terms.append(beta * R_i - log_sum_exp)
        return torch.sum(torch.cat(terms))
    
    # Preallocate tensor for samples on CPU
    samples = torch.zeros((num_samples // thinning, w_current.shape[0]), dtype=torch.float32, device='cuda')
    
    # MCMC sampling with Metropolis-Hastings, MAP tracking, and thinning
    accepted = 0
    sample_idx = 0
    w_map = w_current.clone()
    log_posterior_map = log_prior(w_current) + log_likelihood(w_current, Phi_tau_i, Phi_tau_j, beta)
    
    proposal_dist = torch.distributions.Normal(0, proposal_std)
    
    for step in range(num_samples + burn_in):
        # Propose new weights
        w_proposed = w_current + proposal_dist.sample(w_current.shape).to(device)
        w_proposed = w_proposed / torch.norm(w_proposed, p=2)
        
        log_posterior_current = log_prior(w_current) + log_likelihood(w_current, Phi_tau_i, Phi_tau_j, beta)
        log_posterior_proposed = log_prior(w_proposed) + log_likelihood(w_proposed, Phi_tau_i, Phi_tau_j, beta)
        
        if log_posterior_proposed > log_posterior_map:
            w_map = w_proposed.clone()
            log_posterior_map = log_posterior_proposed
        
        log_alpha = log_posterior_proposed - log_posterior_current
        alpha = torch.exp(log_alpha.clamp(max=0.0))
        
        if torch.rand(1, device=device) < alpha:
            w_current = w_proposed
            accepted += 1
            if step >= burn_in and (accepted % thinning == 0):  # Apply thinning
                samples[sample_idx] = w_current.cpu()
                sample_idx += 1
                if sample_idx >= len(samples):  # Early stopping
                    break
        
        #del w_proposed
        #torch.cuda.empty_cache()
        
        if step % 10000 == 0:  # Print more frequently for large num_samples
            acceptance_rate = accepted / (step + 1) if step > 0 else 0
            print(f"MCMC Step {step}, Acceptance Rate: {acceptance_rate:.4f}, Samples Collected: {sample_idx}")
    
    print(f"Final Acceptance Rate: {accepted / (step + 1):.4f}")
    print(f"MAP Log Posterior: {log_posterior_map.item()}")
    print(f"Samples Collected: {sample_idx}")
    
    # Trim samples
    samples = samples[:sample_idx]
    
    #entropy = compute_entropy_importance_sampling(Phi_tau_i, Phi_tau_j, samples, beta, device)
    print("computing entropy......")
    entropy =  compute_entropy_importance_sampling(Phi_tau_i, Phi_tau_j, samples, beta, device='cuda:0', batch_size_samples=5000, batch_size_pairs=2000)
    sample_based_entropy = kozachenko_entropy(samples)
    #sample_based_entropy = 0 
    w_map = w_map.cpu().detach().numpy()

    if update_model_with_map:
        net.fc_reward.weight.data = torch.from_numpy(w_map).unsqueeze(0).to(device)

    return samples, w_map, entropy, sample_based_entropy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian REX MCMC with specified feedback type")
    parser.add_argument("--feedback_type", choices=["preferences", "corrections"], required=True,
                        help="Type of feedback data: preferences or corrections")
    parser.add_argument("--feedback_path", type=str, default="ppo_merge-v0_1377.pkl",
                        help="Path to the feedback data file (.pkl)")
    parser.add_argument("--model_path", type=str, default="preference_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="mcmc_results",
                        help="Directory to save MCMC results (samples, MAP, entropies)")
    parser.add_argument("--num_samples", type=int, default=100000,
                        help="Number of MCMC samples to collect")
    parser.add_argument("--burn_in", type=int, default=5000,
                        help="Number of burn-in steps for MCMC")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Rationality parameter for Bradley-Terry likelihood")
    parser.add_argument("--proposal_std", type=float, default=0.001,
                        help="Standard deviation for MCMC proposal distribution")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run MCMC on (e.g., cuda:0, cpu)")
    parser.add_argument("--step_size", type=int, default=1000,
                        help="Step size for iterating over feedback sizes")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)

    # Load feedback data
    with open(args.feedback_path, 'rb') as file:
        feedback_data = pickle.load(file)

    segments = feedback_data.get('segments', [])
    preferences = feedback_data.get('preferences', [])
    preferences = process_preferences(preferences, segments)
    corrections = feedback_data.get('corrections', [])

    # Print feedback data summary
    print(f"Feedback Data Loaded from {args.feedback_path}:")
    print(f"  Number of segments: {len(segments)}")
    print(f"  Number of preferences: {len(preferences)}")
    print(f"  Number of corrections: {len(corrections)}")
    print(f"Selected feedback type for processing: {args.feedback_type}")

    # Select feedback data based on type
    feedback_data = preferences if args.feedback_type == "preferences" else corrections

    # Initialize model
    net = Net(input_dim=25, hidden_dim=64, encoding_dims=10, action_dims=5)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net = net.to(device)
    net.eval()  # Ensure deterministic embeddings

    print(f"Model loaded from {args.model_path}")

    # Iterate over feedback sizes
    print(f"Running Bayesian REX MCMC with increasing {args.feedback_type} sizes...")
    feedback_sizes = range(args.step_size, len(feedback_data) + args.step_size, args.step_size)
    entropies = []
    kozachenko_entropies = []
    all_posterior_samples = []
    all_map_solutions = []

    for feedback_size in feedback_sizes:
        # Take feedback subset
        feedback_subset = feedback_data[:feedback_size]
        print(f"\nProcessing {args.feedback_type} 0:{feedback_size} (Total: {len(feedback_subset)})")

        # # Encode feedback with batching to save memory
        # def encode_feedback_batched(net, feedback_subset, segments, device, batch_size=1000):
        #     Phi_tau_i_list, Phi_tau_j_list = [], []
        #     for i in range(0, len(feedback_subset), batch_size):
        #         batch = feedback_subset[i:i+batch_size]
        #         with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        #             if args.feedback_type == "preferences":
        #                 Phi_i, Phi_j = encode_preferences(net, batch, segments, device)
        #             else:
        #                 Phi_i, Phi_j = encode_corrections(net, batch, device)
        #         Phi_tau_i_list.append(Phi_i)
        #         Phi_tau_j_list.append(Phi_j)
        #     Phi_tau_i = torch.cat(Phi_tau_i_list, dim=0)
        #     Phi_tau_j = torch.cat(Phi_tau_j_list, dim=0)
        #     return Phi_tau_i, Phi_tau_j

        # Phi_tau_i, Phi_tau_j = encode_feedback_batched(net, feedback_subset, segments, device)
        # print(f"Processed {Phi_tau_i.shape[0]} {args.feedback_type} pairs")
            # Encode feedback all at once

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            if args.feedback_type == "preferences":
                Phi_tau_i, Phi_tau_j = encode_preferences(net, feedback_subset, segments, device)
            else:
                Phi_tau_i, Phi_tau_j = encode_corrections(net, feedback_subset, device)
        
        print(f"Processed {Phi_tau_i.shape[0]} {args.feedback_type} pairs")

        # Run MCMC with profiling
        #with torch.profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            posterior_samples, map_solution, entropy, sample_based_entropy = bayesian_rex_mcmc(
                net=net,
                Phi_tau_i=Phi_tau_i,
                Phi_tau_j=Phi_tau_j,
                num_samples=args.num_samples,
                burn_in=args.burn_in,
                beta=args.beta,
                proposal_std=args.proposal_std,
                update_model_with_map=False,
                device=device
            )
        #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print("after mcmc!")
        entropies.append(entropy)
        kozachenko_entropies.append(sample_based_entropy)
        all_posterior_samples.append(posterior_samples)
        all_map_solutions.append(map_solution)

        # Print entropies
        print(f"Importance Sampling Entropy with {feedback_size} {args.feedback_type}: {entropy:.4f}")
        print(f"Kozachenko-Leonenko Entropy with {feedback_size} {args.feedback_type}: {sample_based_entropy:.4f}")

        # Clear GPU memory
        #del Phi_tau_i, Phi_tau_j
        #torch.cuda.empty_cache()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    np.save(os.path.join(args.output_dir, f"entropies_{args.feedback_type}.npy"), np.array(entropies))
    np.save(os.path.join(args.output_dir, f"kozachenko_entropies_{args.feedback_type}.npy"), np.array(kozachenko_entropies))
    

    
    for i, feedback_size in enumerate(feedback_sizes):
        # Convert PyTorch tensor to NumPy only when saving
        np.save(os.path.join(args.output_dir, f"posterior_samples_{args.feedback_type}_{feedback_size}.npy"), 
                all_posterior_samples[i].cpu())
        np.save(os.path.join(args.output_dir, f"map_solution_{args.feedback_type}_{feedback_size}.npy"), 
                all_map_solutions[i])

    print(f"\nSummary of Entropies for {args.feedback_type}:")
    for feedback_size, entropy, koz_entropy in zip(feedback_sizes, entropies, kozachenko_entropies):
        print(f"{args.feedback_type.capitalize()} 0:{feedback_size}: Importance Sampling Entropy = {entropy:.4f}, "
            f"Kozachenko-Leonenko Entropy = {koz_entropy:.4f}")

    print(f"\nResults saved to {args.output_dir}:")
    print(f"  entropies_{args.feedback_type}.npy")
    print(f"  kozachenko_entropies_{args.feedback_type}.npy")
    print(f"  posterior_samples_{args.feedback_type}_*.npy")
    print(f"  map_solution_{args.feedback_type}_*.npy")
