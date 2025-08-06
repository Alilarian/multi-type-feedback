import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import pickle
import argparse
import torch.optim as optim

from data_prep import (create_training_data_for_preferences, 
                       create_training_data_for_corrections,
                       process_preferences)
from trex_model import Net
from utils import predict_all_segments, calc_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb


def reconstruction_loss(decoded, target, mu, logvar):
    """Compute reconstruction loss (MSE + KL-divergence) for VAE."""
    mse = F.mse_loss(decoded, target, reduction='sum') / target.numel()
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.numel()
    return mse + kld

def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_times, training_actions, num_iter, l1_reg, checkpoint_dir, loss_fn, wandb_project_name):
    """
    Train the reward network using T-REX and self-supervised losses, logging to W&B.
    
    Args:
        reward_network: Net model instance (input_dim=25, action_dims=5).
        optimizer: PyTorch optimizer (e.g., Adam).
        training_inputs: List of (traj_i, traj_j) tuples, each traj of shape (T', 1, 5, 5).
        training_outputs: List of binary labels (0 if traj_i better than traj_j).
        training_times: List of (time_i, time_j) tuples.
        training_actions: List of (actions_i, actions_j) tuples, each of shape (T', 5).
        num_iter: Number of epochs.
        l1_reg: L1 regularization weight (unused).
        checkpoint_dir: Path to save model checkpoints.
        loss_fn: Loss function type ('trex', 'ss', 'trex+ss').
        wandb_project_name: W&B project name for logging.
    
    Returns:
        reward_network: Trained Net model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wandb.init(project=wandb_project_name, config={
        "num_iter": num_iter,
        "loss_fn": loss_fn,
        "lr": optimizer.param_groups[0]['lr'],
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0.0)
    })
    
    loss_criterion = nn.CrossEntropyLoss()
    temporal_difference_loss = nn.MSELoss()
    inverse_dynamics_loss = nn.CrossEntropyLoss()
    forward_dynamics_loss = nn.MSELoss()
    
    training_data = list(zip(training_inputs, training_outputs, training_times, training_actions))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        cum_loss = 0.0
        cum_trex_loss = 0.0
        cum_recon_loss = 0.0
        cum_inv_loss = 0.0
        cum_fwd_loss = 0.0
        cum_dt_loss = 0.0
        skipped_dynamics_i = 0
        skipped_dynamics_j = 0
        
        for i, (obs, label, times, actions) in enumerate(training_data):
            traj_i, traj_j = torch.tensor(obs[0], dtype=torch.float32).to(device), torch.tensor(obs[1], dtype=torch.float32).to(device)  # (T_i', 1, 5, 5), (T_j', 1, 5, 5)
            label = torch.tensor([label], dtype=torch.long).to(device)  # 0 if traj_i (corrected) better
            actions_i, actions_j = torch.tensor(actions[0], dtype=torch.float32).to(device), torch.tensor(actions[1], dtype=torch.float32).to(device)  # (T_i', 5), (T_j', 5)
            times_i, times_j = times
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, abs_rewards, z1, z2, mu1, mu2, var1, var2, recon_i, recon_j = reward_network(traj_i, traj_j)
            
            # Reconstruction loss (works for T=1)
            recon_loss_i = reconstruction_loss(recon_i, traj_i.view(traj_i.size(0), -1), mu1, var1)
            recon_loss_j = reconstruction_loss(recon_j, traj_j.view(traj_j.size(0), -1), mu2, var2)
            recon_loss = 10 * (recon_loss_i + recon_loss_j)
            
            # Initialize dynamics losses
            inv_loss_i = torch.tensor(0.0, device=device, requires_grad=True)
            inv_loss_j = torch.tensor(0.0, device=device, requires_grad=True)
            fwd_loss_i = torch.tensor(0.0, device=device, requires_grad=True)
            fwd_loss_j = torch.tensor(0.0, device=device, requires_grad=True)
            dt_loss_i = torch.tensor(0.0, device=device, requires_grad=True)
            dt_loss_j = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Compute dynamics losses for traj_i (corrected_seg) if T>=2
            if len(traj_i) >= 2:
                # Inverse dynamics
                actions_1 = reward_network.estimate_inverse_dynamics(mu1[:-1], mu1[1:])
                target_actions_1 = torch.argmax(actions_i[1:], dim=1)
                inv_loss_i = inverse_dynamics_loss(actions_1, target_actions_1) / 1.9
                
                # Forward dynamics
                forward_dynamics_1 = reward_network.estimate_forward_dynamics(mu1[:-1], actions_i[:-1])
                fwd_loss_i = 100 * forward_dynamics_loss(forward_dynamics_1, mu1[1:])
                
                # Temporal difference
                t1_i, t2_i = np.random.randint(0, len(times_i)), np.random.randint(0, len(times_i))
                est_dt_i = reward_network.estimate_temporal_difference(mu1[t1_i].unsqueeze(0), mu1[t2_i].unsqueeze(0))
                real_dt_i = (times_i[t2_i] - times_i[t1_i]) / 100.0
                dt_loss_i = 4 * temporal_difference_loss(est_dt_i, torch.tensor([[real_dt_i]], dtype=torch.float32, device=device))
            else:
                skipped_dynamics_i += 1
            
            # Compute dynamics losses for traj_j (original_seg) if T>=2
            if len(traj_j) >= 2:
                # Inverse dynamics
                actions_2 = reward_network.estimate_inverse_dynamics(mu2[:-1], mu2[1:])
                target_actions_2 = torch.argmax(actions_j[1:], dim=1)
                inv_loss_j = inverse_dynamics_loss(actions_2, target_actions_2) / 1.9
                
                # Forward dynamics
                forward_dynamics_2 = reward_network.estimate_forward_dynamics(mu2[:-1], actions_j[:-1])
                fwd_loss_j = 100 * forward_dynamics_loss(forward_dynamics_2, mu2[1:])
                
                # Temporal difference
                t1_j, t2_j = np.random.randint(0, len(times_j)), np.random.randint(0, len(times_j))
                est_dt_j = reward_network.estimate_temporal_difference(mu2[t1_j].unsqueeze(0), mu2[t2_j].unsqueeze(0))
                real_dt_j = (times_j[t2_j] - times_j[t1_j]) / 100.0
                dt_loss_j = 4 * temporal_difference_loss(est_dt_j, torch.tensor([[real_dt_j]], dtype=torch.float32, device=device))
            else:
                skipped_dynamics_j += 1
            
            # Combine dynamics losses
            inv_loss = inv_loss_i + inv_loss_j
            fwd_loss = fwd_loss_i + fwd_loss_j
            dt_loss = dt_loss_i + dt_loss_j
            
            # T-REX loss (works for T=1)
            trex_loss = loss_criterion(outputs.unsqueeze(0), label)
            
            # Combine losses
            if loss_fn == "trex":
                loss = trex_loss
            elif loss_fn == "ss":
                loss = recon_loss + inv_loss + fwd_loss + dt_loss
            elif loss_fn == "trex+ss":
                loss = trex_loss + recon_loss + inv_loss + fwd_loss + dt_loss
            
            loss.backward()
            optimizer.step()
            
            # Log losses to W&B
            wandb.log({
                "epoch": epoch,
                "iteration": i,
                "total_loss": loss.item(),
                "trex_loss": trex_loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "inverse_dynamics_loss": inv_loss.item(),
                "forward_dynamics_loss": fwd_loss.item(),
                "temporal_difference_loss": dt_loss.item(),
                "skipped_dynamics_i": skipped_dynamics_i,
                "skipped_dynamics_j": skipped_dynamics_j
            })
            
            cum_loss += loss.item()
            cum_trex_loss += trex_loss.item()
            cum_recon_loss += recon_loss.item()
            cum_inv_loss += inv_loss.item()
            cum_fwd_loss += fwd_loss.item()
            cum_dt_loss += dt_loss.item()
            
            if i % 500 == 499:
                avg_loss = cum_loss / 500
                avg_trex_loss = cum_trex_loss / 500
                avg_recon_loss = cum_recon_loss / 500
                avg_inv_loss = cum_inv_loss / 500
                avg_fwd_loss = cum_fwd_loss / 500
                avg_dt_loss = cum_dt_loss / 500
                print(f"epoch {epoch}:{i} total_loss {avg_loss:.4f} "
                      f"trex {avg_trex_loss:.4f} recon {avg_recon_loss:.4f} "
                      f"inv {avg_inv_loss:.4f} fwd {avg_fwd_loss:.4f} dt {avg_dt_loss:.4f}")
                wandb.log({
                    "epoch": epoch,
                    "iteration": i,
                    "avg_total_loss": avg_loss,
                    "avg_trex_loss": avg_trex_loss,
                    "avg_reconstruction_loss": avg_recon_loss,
                    "avg_inverse_dynamics_loss": avg_inv_loss,
                    "avg_forward_dynamics_loss": avg_fwd_loss,
                    "avg_temporal_difference_loss": avg_dt_loss,
                    "skipped_dynamics_fraction_i": skipped_dynamics_i / (i + 1),
                    "skipped_dynamics_fraction_j": skipped_dynamics_j / (i + 1)
                })
                cum_loss = 0.0
                cum_trex_loss = 0.0
                cum_recon_loss = 0.0
                cum_inv_loss = 0.0
                cum_fwd_loss = 0.0
                cum_dt_loss = 0.0
                skipped_dynamics_i = 0
                skipped_dynamics_j = 0
                torch.save(reward_network.state_dict(), checkpoint_dir)
    
    print("finished training")
    return reward_network



# def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_times, training_actions, num_iter, l1_reg, checkpoint_dir, loss_fn, wandb_project_name):
#     """
#     Train the reward network using T-REX and self-supervised losses, logging to W&B.
    
#     Args:
#         reward_network: Net model instance (input_dim=25, action_dims=5).
#         optimizer: PyTorch optimizer (e.g., Adam).
#         training_inputs: List of (traj_i, traj_j) tuples, each traj of shape (T', 1, 5, 5).
#         training_outputs: List of binary labels (1 if traj_i better, 0 if traj_j better).
#         training_times: List of (time_i, time_j) tuples.
#         training_actions: List of (actions_i, actions_j) tuples, each of shape (T', 5).
#         num_iter: Number of epochs.
#         l1_reg: L1 regularization weight (unused).
#         checkpoint_dir: Path to save model checkpoints.
#         loss_fn: Loss function type ('trex', 'ss', 'trex+ss').
    
#     Returns:
#         reward_network: Trained Net model.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
    
#     # Initialize W&B
#     wandb.init(project=wandb_project_name, config={
#         "num_iter": num_iter,
#         "loss_fn": loss_fn,
#         "lr": optimizer.param_groups[0]['lr'],
#         "weight_decay": optimizer.param_groups[0].get('weight_decay', 0.0)
#     })
    
#     loss_criterion = nn.CrossEntropyLoss()
#     temporal_difference_loss = nn.MSELoss()
#     inverse_dynamics_loss = nn.CrossEntropyLoss()
#     forward_dynamics_loss = nn.MSELoss()
    
#     training_data = list(zip(training_inputs, training_outputs, training_times, training_actions))
#     for epoch in range(num_iter):
#         np.random.shuffle(training_data)
#         cum_loss = 0.0
#         cum_trex_loss = 0.0
#         cum_recon_loss = 0.0
#         cum_inv_loss = 0.0
#         cum_fwd_loss = 0.0
#         cum_dt_loss = 0.0
        
#         for i, (obs, label, times, actions) in enumerate(training_data):
#             traj_i, traj_j = torch.tensor(obs[0], dtype=torch.float32).to(device), torch.tensor(obs[1], dtype=torch.float32).to(device)  # (T_i', 1, 5, 5), (T_j', 1, 5, 5)
#             label = torch.tensor([label], dtype=torch.long).to(device)  # 1 if traj_i better
#             actions_i, actions_j = torch.tensor(actions[0], dtype=torch.float32).to(device), torch.tensor(actions[1], dtype=torch.float32).to(device)  # (T_i', 5), (T_j', 5)
#             times_i, times_j = times
            
#             #if len(traj_i) < 2 or len(traj_j) < 2:  # Need at least 2 steps for dynamics
#             #    continue
#             ## I should change 
            
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs, abs_rewards, z1, z2, mu1, mu2, var1, var2, recon_i, recon_j = reward_network(traj_i, traj_j)
            
#             # Reconstruction loss (MSE + KL-divergence)
#             recon_loss_i = reconstruction_loss(recon_i, traj_i.view(traj_i.size(0), -1), mu1, var1)
#             recon_loss_j = reconstruction_loss(recon_j, traj_j.view(traj_j.size(0), -1), mu2, var2)
#             recon_loss = 10 * (recon_loss_i + recon_loss_j)
            
#             # Inverse dynamics
#             actions_1 = reward_network.estimate_inverse_dynamics(mu1[:-1], mu1[1:])  # (T_i'-1, 5)
#             actions_2 = reward_network.estimate_inverse_dynamics(mu2[:-1], mu2[1:])  # (T_j'-1, 5)
#             target_actions_1 = torch.argmax(actions_i[1:], dim=1)  # Convert one-hot to indices
#             target_actions_2 = torch.argmax(actions_j[1:], dim=1)
#             inv_loss = (inverse_dynamics_loss(actions_1, target_actions_1) + inverse_dynamics_loss(actions_2, target_actions_2)) / 1.9
            
#             # Forward dynamics (single-step)
#             forward_dynamics_1 = reward_network.estimate_forward_dynamics(mu1[:-1], actions_i[:-1])  # (T_i'-1, encoding_dims)
#             forward_dynamics_2 = reward_network.estimate_forward_dynamics(mu2[:-1], actions_j[:-1])  # (T_j'-1, encoding_dims)
#             fwd_loss = 100 * (forward_dynamics_loss(forward_dynamics_1, mu1[1:]) + forward_dynamics_loss(forward_dynamics_2, mu2[1:]))
            
#             # Temporal difference
#             t1_i, t2_i = np.random.randint(0, len(times_i)), np.random.randint(0, len(times_i))
#             t1_j, t2_j = np.random.randint(0, len(times_j)), np.random.randint(0, len(times_j))
#             est_dt_i = reward_network.estimate_temporal_difference(mu1[t1_i].unsqueeze(0), mu1[t2_i].unsqueeze(0))
#             est_dt_j = reward_network.estimate_temporal_difference(mu2[t1_j].unsqueeze(0), mu2[t2_j].unsqueeze(0))
#             real_dt_i = (times_i[t2_i] - times_i[t1_i]) / 100.0
#             real_dt_j = (times_j[t2_j] - times_j[t1_j]) / 100.0
#             dt_loss = 4 * (temporal_difference_loss(est_dt_i, torch.tensor([[real_dt_i]], dtype=torch.float32, device=device)) +
#                            temporal_difference_loss(est_dt_j, torch.tensor([[real_dt_j]], dtype=torch.float32, device=device)))
            
#             # T-REX loss
#             trex_loss = loss_criterion(outputs.unsqueeze(0), label)
            
#             # Combine losses
#             if loss_fn == "trex":
#                 loss = trex_loss
#             elif loss_fn == "ss":
#                 loss = recon_loss + inv_loss + fwd_loss + dt_loss
#             elif loss_fn == "trex+ss":
#                 loss = trex_loss + recon_loss + inv_loss + fwd_loss + dt_loss
            
#             loss.backward()
#             optimizer.step()
            
#             # Log losses to W&B
#             wandb.log({
#                 "epoch": epoch,
#                 "iteration": i,
#                 "total_loss": loss.item(),
#                 "trex_loss": trex_loss.item(),
#                 "reconstruction_loss": recon_loss.item(),
#                 "inverse_dynamics_loss": inv_loss.item(),
#                 "forward_dynamics_loss": fwd_loss.item(),
#                 "temporal_difference_loss": dt_loss.item()
#             })
            
#             cum_loss += loss.item()
#             cum_trex_loss += trex_loss.item()
#             cum_recon_loss += recon_loss.item()
#             cum_inv_loss += inv_loss.item()
#             cum_fwd_loss += fwd_loss.item()
#             cum_dt_loss += dt_loss.item()
            
#             if i % 500 == 499:
#                 avg_loss = cum_loss / 500
#                 avg_trex_loss = cum_trex_loss / 500
#                 avg_recon_loss = cum_recon_loss / 500
#                 avg_inv_loss = cum_inv_loss / 500
#                 avg_fwd_loss = cum_fwd_loss / 500
#                 avg_dt_loss = cum_dt_loss / 500
#                 print(f"epoch {epoch}:{i} total_loss {avg_loss:.4f} "
#                       f"trex {avg_trex_loss:.4f} recon {avg_recon_loss:.4f} "
#                       f"inv {avg_inv_loss:.4f} fwd {avg_fwd_loss:.4f} dt {avg_dt_loss:.4f}")
#                 wandb.log({
#                     "epoch": epoch,
#                     "iteration": i,
#                     "avg_total_loss": avg_loss,
#                     "avg_trex_loss": avg_trex_loss,
#                     "avg_reconstruction_loss": avg_recon_loss,
#                     "avg_inverse_dynamics_loss": avg_inv_loss,
#                     "avg_forward_dynamics_loss": avg_fwd_loss,
#                     "avg_temporal_difference_loss": avg_dt_loss
#                 })
#                 cum_loss = 0.0
#                 cum_trex_loss = 0.0
#                 cum_recon_loss = 0.0
#                 cum_inv_loss = 0.0
#                 cum_fwd_loss = 0.0
#                 cum_dt_loss = 0.0
#                 torch.save(reward_network.state_dict(), checkpoint_dir)
    
#     print("finished training")
#     return reward_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, test, or predict with T-REX reward model")
    parser.add_argument("--mode", choices=["train", "test", "predict"], required=True,
                        help="Mode to run: train, test, or predict")
    parser.add_argument("--feedback_type", choices=["preferences", "corrections"], required=True,
                        help="Type of feedback data: preferences or corrections")
    parser.add_argument("--feedback_path", type=str, default='ppo_merge-v0_1377.pkl',
                        help="Path to the feedback data file (.pkl)")
    parser.add_argument("--model_path", type=str, default="preference_model.pth",
                        help="Path to save/load the model checkpoint")
    parser.add_argument("--wandb_project", type=str, default="preference_model",
                        help="W&B project name for logging")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs (for train mode)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--loss_fn", choices=["trex", "ss", "trex+ss"], default="trex+ss",
                        help="Loss function type: trex, ss, or trex+ss")
    
    args = parser.parse_args()
    
    # Load feedback data
    with open(args.feedback_path, 'rb') as file:
        feedback_data = pickle.load(file)
    
    segments = feedback_data['segments']
    demos = feedback_data.get('demos', [])
    preferences = feedback_data.get('preferences', [])
    corrections = feedback_data.get('corrections', [])
    
    # Print number and types of feedback data
    print(f"Feedback Data Loaded from {args.feedback_path}:")
    print(f"  Number of segments: {len(segments)}")
    print(f"  Number of demos: {len(demos)}")
    print(f"  Number of preferences: {len(preferences)}")
    print(f"  Number of corrections: {len(corrections)}")
    print(f"Selected feedback type for processing: {args.feedback_type}")

    ## For future I have to choose env and define the input_dim etc based on that

    # Initialize model and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(input_dim=25, hidden_dim=64, encoding_dims=10, action_dims=5)
    reward_net = reward_net.to(device)
    
    if args.mode == "train":
        # Prepare data based on feedback type
        if args.feedback_type == "preferences":
            preferences = process_preferences(preferences, segments)
            training_obs, training_labels, times, actions = create_training_data_for_preferences(segments, preferences)
        elif args.feedback_type == "corrections":
            training_obs, training_labels, times, actions = create_training_data_for_corrections(corrections)
        else:
            raise ValueError(f"Unknown feedback type: {args.feedback_type}")
        
        # Initialize optimizer
        optimizer = optim.Adam(reward_net.parameters(), lr=args.lr, weight_decay=0.001)
        
        # Train the model
        trained_model = learn_reward(
            reward_network=reward_net,
            optimizer=optimizer,
            training_inputs=training_obs,
            training_outputs=training_labels,
            training_times=times,
            training_actions=actions,
            num_iter=args.num_epochs,
            l1_reg=0.0,
            checkpoint_dir=args.model_path,
            loss_fn=args.loss_fn,
            wandb_project_name=args.wandb_project
        )
        
        print(f"Model saved to {args.model_path}")
    
    elif args.mode == "test":
        # Load trained model
        reward_net.load_state_dict(torch.load(args.model_path, map_location=device))
        reward_net.eval()
        
        # Prepare data for testing (use same feedback type as training)
        if args.feedback_type == "preferences":
            preferences = process_preferences(preferences, segments)
            training_obs, training_labels, times, actions = create_training_data_for_preferences(segments, preferences)
        elif args.feedback_type == "corrections":
            training_obs, training_labels, times, actions = create_training_data_for_corrections(corrections)
        
        # Compute accuracy
        wandb.init(project=args.wandb_project, name="test_accuracy")
        accuracy = calc_accuracy(reward_net, training_obs, training_labels, log_to_wandb=True)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    elif args.mode == "predict":
        # Load trained model
        reward_net.load_state_dict(torch.load(args.model_path, map_location=device))
        reward_net.eval()
        
        # Predict over all segments
        wandb.init(project=args.wandb_project, name="all_segments_prediction")
        all_pred_rewards, all_true_rewards, all_pred_cum_returns, all_true_cum_returns = predict_all_segments(
            reward_net, segments, log_to_wandb=True
        )
        
        # Print results for a few segments
        for idx, (pred_r, true_r, pred_cum, true_cum) in enumerate(zip(all_pred_rewards, all_true_rewards, all_pred_cum_returns, all_true_cum_returns)):
            if idx >= 5:  # Limit to first 5 for brevity
                break
            print(f"Segment {idx}:")
            print(f"  Predicted rewards: {pred_r}")
            print(f"  True rewards: {true_r}")
            print(f"  Predicted cumulative return: {pred_cum}")
            print(f"  True cumulative return: {true_cum}")
            print(f"  Per-state MAE: {np.mean(np.abs(np.array(true_r) - np.array(pred_r))) if true_r else 0.0}")
            print(f"  Cumulative MAE: {abs(true_cum - pred_cum)}")