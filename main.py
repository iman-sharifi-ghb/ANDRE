#!/usr/bin/env python3
"""
ANDRE Neural-Symbolic Learning Main Script
This script implements ANDRE (Adaptive Neural-symbolic Differentiable Rule Engine)
for learning logical rules from data using neural networks.
"""

import argparse
import math
import pickle
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
from ANDRE_Net2 import ANDRE
from collections import Counter
from utils import check_binary_dataset, create_V_matrix
from utils import plot_andre_weights
from utils import interpret_neuron_weights_entropy, interpret_neuron_weights
from utils import extract_rules
from utils import parse_rule_and_generate_data
from utils import compute_ranking_metrics, print_ranking_metrics
from utils import filter_valid_rules
from utils import extract_head_and_connected_vars
from utils import top_k_cosine_similar, mean_reciprocal_rank, hits_at_k, compute_mrr_hits
from utils import schedule_loss_weights
from utils import str2bool, Tee
from utils import get_predicate_name, evaluate_single_rule
import numpy as np
import pandas as pd
from datetime import datetime

class BalancedBatchSampler:
    """Sampler that yields balanced batches with equal positive/negative samples."""
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        
        # Get indices for positive and negative samples
        self.pos_indices = torch.where(labels == 1)[0]
        self.neg_indices = torch.where(labels == 0)[0]
        
        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        
        # Batch size per class
        self.batch_size_per_class = batch_size // 2
        
    def __iter__(self):
        # Shuffle indices
        pos_perm = torch.randperm(self.n_pos)
        neg_perm = torch.randperm(self.n_neg)
        
        pos_indices_shuffled = self.pos_indices[pos_perm]
        neg_indices_shuffled = self.neg_indices[neg_perm]
        
        # Create batches
        n_batches = min(self.n_pos, self.n_neg) // self.batch_size_per_class
        
        for i in range(n_batches):
            pos_batch = pos_indices_shuffled[i * self.batch_size_per_class:(i + 1) * self.batch_size_per_class]
            neg_batch = neg_indices_shuffled[i * self.batch_size_per_class:(i + 1) * self.batch_size_per_class]
            
            # Combine and shuffle batch indices
            batch_indices = torch.cat([pos_batch, neg_batch])
            batch_perm = torch.randperm(len(batch_indices))
            
            yield batch_indices[batch_perm]
    
    def __len__(self):
        return min(self.n_pos, self.n_neg) // self.batch_size_per_class

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ANDRE Neural-Symbolic Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument('--synthetic_data', type=str2bool, default=True,
                        help='Use synthetic data (True) or load from file (False)')
    parser.add_argument('--structured_data', type=str2bool, default=True,
                        help='Use structured data with predicates')
    parser.add_argument('--dataset_type', type=str, default='binary', choices=['binary', 'continuous'],
                        help='Type of dataset')
    parser.add_argument('--data_path', type=str, default='test_dir/',
                        help='Path to data directory (for non-synthetic data)')
    parser.add_argument('--head_predicate', type=str, default='grandparent(X1, X3)',
                        help='Head predicate for structured data (e.g., "grandparent(X1, X3)")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model architecture
    parser.add_argument('--num_rules', type=int, default=4,
                        help='Maximum number of rules to learn (used when rule_counts is not specified)')
    parser.add_argument('--rule_counts', type=str, default=None,
                        help='Comma-separated list of rule counts to train (e.g., "1,3,5,10"). If specified, overrides num_rules iteration.')
    parser.add_argument('--input_dim', type=int, default=6,
                        help='Input dimension (for unstructured data)')
    
    # Training parameters
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of training samples (for synthetic data)')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--num_restarts', type=int, default=3,
                        help='Number of random restarts')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Loss weights
    parser.add_argument('--loss_schedule', type=str2bool, default=True,
                        help='Use loss weight scheduling')
    parser.add_argument('--sim_weight', type=float, default=0.2,
                        help='Weight for similarity penalty')
    parser.add_argument('--ent_weight', type=float, default=0.1,
                        help='Weight for entropy loss')
    parser.add_argument('--rr_weight', type=float, default=1.0,
                        help='Weight for range restriction loss')
    parser.add_argument('--conn_weight', type=float, default=5.0,
                        help='Weight for connectivity loss')
    parser.add_argument('--digit_weight', type=float, default=0.01,
                        help='Weight for digitization loss')
    

    parser.add_argument('--pp', type=float, default=0.1,
                        help='Perturbation parameter for synthetic data preprocessing')
    parser.add_argument('--k', type=int, default=100,
                        help='Parameter k for evaluation metrics')
    parser.add_argument('--r', type=int, default=100,
                        help='Parameter r for evaluation metrics')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='Eta parameter for evaluation metrics')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for optimizer')
    parser.add_argument('--acc_threshold', type=float, default=0.98,
                        help='Accuracy threshold for early stopping')
    parser.add_argument('--best_model_criterion', type=str, default='val_loss',
                        choices=['val_loss', 'train_loss'],
                        help='Criterion to select best model: validation loss or training loss')
    
    # Output options
    parser.add_argument('--visualize_weights', type=str2bool, default=False,
                        help='Visualize learned weights')
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='Print verbose output')
    parser.add_argument('--save_model', type=str2bool, default=False,
                        help='Save the best model')
    
    # Device options
    parser.add_argument('--device', type=str, default='cpu', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training (auto will use CUDA if available)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    # Evaluation options
    parser.add_argument('--evaluate_individual_rules', type=str2bool, default=False,
                        help='Evaluate and sort individual rules by accuracy')
    parser.add_argument('--rule_confidence_metric', type=str, default='val_accuracy',
                        choices=['val_accuracy', 'val_f1', 'train_accuracy', 'val_precision', 'nr_nb_ratio'],
                        help='Metric to use as rule confidence for ranking (MRR/Hits@K). nr_nb_ratio uses N_r/N_b ratio from validation set')
    
    # Data preprocessing options
    parser.add_argument('--perturb', type=str2bool, default=True,
                        help='Perturb binary data by adding small noise for better training')
    parser.add_argument('--np', type=float, default=0.0,
                        help='Noise percentage for label flipping (0.0 to 1.0). Default is 0 (no noise)')
    
    return parser.parse_args()


def main():
    """Main training and evaluation function."""
    args = parse_args()
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_hour = datetime.now().strftime("%H-%M-%S")

    # Set up directories based on data source
    base_path = args.data_path.rstrip('/')
    results_dir = os.path.join(base_path, 'results')
    models_dir = os.path.join(base_path, 'saved_models')
    
    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Set up logging to save terminal output to file
    log_file = os.path.join(results_dir, f"training_log_{today_date}_{today_hour}.log")
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"Logging session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print("="*70)

    # Set device
    if args.force_cpu:
        device = torch.device("cpu")
    elif args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)

    # Initialize variables
    synthetic_data = args.synthetic_data
    structured_data = args.structured_data
    visualize_weights = args.visualize_weights
    verbose = args.verbose
    save_model = args.save_model
    dataset_type = args.dataset_type
    seed = args.seed
    
    # Determine which rule counts to train
    if args.rule_counts:
        # Parse comma-separated list of rule counts
        rule_counts_to_train = [int(x.strip()) for x in args.rule_counts.split(',')]
        print(f"Training ANDRE with specific rule counts: {rule_counts_to_train}")
    else:
        # Use range from 1 to num_rules (inclusive)
        rule_counts_to_train = list(range(1, args.num_rules + 1))
        print(f"Training ANDRE with rule counts from 1 to {args.num_rules}")
    
    num_epochs = args.num_epochs
    num_restarts = args.num_restarts
    batch_size = args.batch_size
    
    sim_weight = args.sim_weight
    ent_weight = args.ent_weight
    rr_weight = args.rr_weight
    conn_weight = args.conn_weight
    digit_weight = args.digit_weight

    if synthetic_data:
        num_samples = args.num_samples
        
        if not structured_data:
            input_dim = args.input_dim
            num_samples = 200 * input_dim
            body_predicates = [f"b{i+1}" for i in range(input_dim)]
            head_predicate = "h"
            variable_matrix, variables = None, None
            body_predicates = None
        else:
            body_predicates = ["father(X1, X2)", "father(X2, X3)", "father(X1, X3)", 
                               "mother(X1, X2)", "mother(X2, X3)", "mother(X1, X3)"]
            head_predicate = args.head_predicate
            variable_matrix, variables = create_V_matrix(body_predicates)
            head_vars, connected_vars = extract_head_and_connected_vars(head_predicate, variables)
            input_dim = len(body_predicates)
            body_predicates = body_predicates
            head_predicate = head_predicate
        
        # Generate Random Dataset
        torch.manual_seed(seed)
        if dataset_type == "binary":
            X_train = (torch.rand(num_samples, input_dim) > 0.5).float()
            acc_threshold = args.acc_threshold
        else:
            X_train = torch.rand(num_samples, input_dim)
            acc_threshold = args.acc_threshold

        if check_binary_dataset(X_train) and args.perturb:
            # perturb data by adding small noise to zero and reducing one to less than one
            X_train = X_train - args.pp * torch.rand_like(X_train) * (X_train - 0.5).sign()
            # clip to [0, 1]
            X_train = torch.clamp(X_train, 0.0, 1.0)

        y_train = (((X_train[:, 0] > 0.5) & (X_train[:, 1] > 0.5)) | 
                ((X_train[:, 0] > 0.5) & (X_train[:, 4] > 0.5)) | 
                ((X_train[:, 3] > 0.5) & (X_train[:, 4] > 0.5)) | 
                ((X_train[:, 3] > 0.5) & (X_train[:, 1] > 0.5))).float()

        # Move to device
        X_train = X_train.to(device)
        y_train = y_train.to(device)

    else:
        file_path = args.data_path

        # read the x_train.dt and y_train.dt files
        with open(file_path + 'x_train.dt', 'rb') as xf:
            x_train = pickle.load(xf)
        with open(file_path + 'y_train.dt', 'rb') as yf:
            y_train = pickle.load(yf)

        print("👉 The x_train data is:")
        print(len(x_train[0]))
        print("👉 The y_train data is:")
        print(len(y_train))

        x_train_np = np.concatenate(x_train, axis=1)
        X_train = torch.tensor(x_train_np[:, :-1], dtype=torch.float32)
        y_train = torch.tensor(x_train_np[:, -1], dtype=torch.float32)

        # Move to device
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        if check_binary_dataset(X_train):
            dataset_type = "binary"
            if args.perturb:
                print("X_train is a binary dataset. We are perturbing the data for a better training.")
                # perturb data by adding small noise to zero and reducing one to less than one
                X_train = X_train - args.pp * torch.rand_like(X_train) * (X_train - 0.5).sign()
                # clip to [0, 1]
                X_train = torch.clamp(X_train, 0.0, 1.0)
            else:
                print("X_train is a binary dataset. Perturbation is disabled.")

        # Load the predicate list
        with open(file_path + 'predicate.list', 'rb') as f:
            predicate_list = pickle.load(f)

        # Load relation variable permutations
        with open(file_path + 'relation_variable.dt', 'rb') as f:
            relation_variable_permutations = pickle.load(f)

        # Load relation names
        with open(file_path + 'all_relation_dic.dt', 'rb') as f:
            all_relations = pickle.load(f)

        print("Predicates:", len(predicate_list))
        print("Relations:", len(all_relations))
        print("Variable permutations:", len(relation_variable_permutations))

        # Example: Map X_train column 0
        body_predicates = []
        for idx in range(len(predicate_list)):
            predicate_name = get_predicate_name(idx, predicate_list, 
                                                relation_variable_permutations, 
                                                all_relations)
            body_predicates.append(predicate_name)

        # check if the length of body_predicates matches the number of columns in X_train
        assert len(body_predicates) == X_train.shape[1], "Mismatch between body predicates and X_train columns."
        _, variables = create_V_matrix(body_predicates)
        head_predicate = args.head_predicate
        head_vars, connected_vars = extract_head_and_connected_vars(head_predicate, variables)

        # head_predicate = "blockpositionindex(X1, X2)" #body_predicates[252]  # Example head predicate
        for idx in range(len(predicate_list)):
            if body_predicates[idx] == head_predicate:
                headpred_idx = idx
                print(f"Found head predicate at index {idx} in body predicates.")
                break

        # remove the head predicate from body predicates
        body_predicates.remove(head_predicate)
        print("Number of body predicates:", len(body_predicates))

        # create variable matrix without head predicate
        variable_matrix, _ = create_V_matrix(body_predicates)

        # remove the corresponding column from X_train
        X_train = torch.cat((X_train[:, :headpred_idx], X_train[:, headpred_idx+1:]), dim=1)

        input_dim = X_train.shape[1]
        num_samples = X_train.shape[0]
        acc_threshold = args.acc_threshold

    # Remove predicates containing 'X4' (only for non-synthetic structured data)
    if not synthetic_data and structured_data:
        indices_keep = [i for i, p in enumerate(body_predicates) if "X4" not in p]
        indices_remove = [i for i, p in enumerate(body_predicates) if "X4" in p]

        if not indices_remove:
            print("No predicates containing 'X4' were found. No changes made.")
        else:
            # Update body_predicates
            body_predicates = [body_predicates[i] for i in indices_keep]
            variable_matrix, variables = create_V_matrix(body_predicates)
            head_vars, connected_vars = extract_head_and_connected_vars(head_predicate, variables)
            # Update X_train (keep columns corresponding to indices_keep)
            X_train = X_train[:, indices_keep].clone()
            # Update input_dim
            input_dim = X_train.shape[1]
            # Summary
            print(f"Removed {len(indices_remove)} predicates containing 'X4'.")
            print(f"Old input_dim -> New input_dim: {input_dim}")

    # Remove predicates with identical variables (e.g., 'sameperson(X1,X1)', 'advisedby(X3,X3)')
    # Only for advisedby head predicate
    if not synthetic_data and structured_data and args.head_predicate == 'advisedby(X1, X2)':
        import re
        indices_keep = []
        indices_remove = []
        
        for i, p in enumerate(body_predicates):
            # Extract variables from predicate, e.g., 'sameperson(X1,X1)' -> ['X1', 'X1']
            match = re.search(r'\((.*?)\)', p)
            if match:
                vars_str = match.group(1)
                vars_list = [v.strip() for v in vars_str.split(',')]
                # Check if any variable appears more than once (identical variables)
                if len(vars_list) != len(set(vars_list)):
                    indices_remove.append(i)
                else:
                    indices_keep.append(i)
            else:
                # If no variables found, keep the predicate
                indices_keep.append(i)
        
        if not indices_remove:
            print("No predicates with identical variables were found. No changes made.")
        else:
            removed_predicates = [body_predicates[i] for i in indices_remove]
            print(f"\nRemoving predicates with identical variables:")
            for p in removed_predicates:
                print(f"  - {p}")
            
            # Update body_predicates
            body_predicates = [body_predicates[i] for i in indices_keep]
            variable_matrix, variables = create_V_matrix(body_predicates)
            head_vars, connected_vars = extract_head_and_connected_vars(head_predicate, variables)
            # Update X_train (keep columns corresponding to indices_keep)
            X_train = X_train[:, indices_keep].clone()
            # Update input_dim
            input_dim = X_train.shape[1]
            # Summary
            print(f"Removed {len(indices_remove)} predicates with identical variables.")
            print(f"New input_dim: {input_dim}\n")

    # Remove half of the data samples in X_train and y_train for faster experimentation
    # if args.num_samples < num_samples:
    #     num_samples = args.num_samples
    #     X_train = X_train[:num_samples].clone()
    #     y_train = y_train[:num_samples].clone()
    #     print(f"Reduced dataset size to {num_samples} samples for faster experimentation.")

    # Create a validation split
    n = X_train.size(0)
    val_size = max(1, int(args.val_split * n))

    # reproducible permutation if seed exists
    try:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g)
    except NameError:
        perm = torch.randperm(n)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    # create validation tensors and update training tensors (clone to avoid accidental views)
    X_val = X_train[val_idx].clone()
    y_val = y_train[val_idx].clone()

    X_train = X_train[train_idx].clone()
    y_train = y_train[train_idx].clone()

    # update num_samples to reflect new training set size
    num_samples = X_train.size(0)

    # Apply label noise to training set only (not validation set)
    if args.np > 0:
        num_noisy_labels = int(args.np * num_samples)
        if num_noisy_labels > 0:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            noise_indices = torch.randperm(num_samples)[:num_noisy_labels]
            y_train[noise_indices] = 1 - y_train[noise_indices]  # Flip labels (0->1, 1->0)
            print(f"Applied label noise: flipped {num_noisy_labels} labels ({args.np*100:.1f}% of training set)")
        else:
            print(f"Label noise percentage too small to flip any labels ({args.np*100:.1f}%)")
    else:
        print("No label noise applied (np=0)")

    # Move to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, num_samples (train) = {num_samples}")

    # Cosine similarity function
    cosine = nn.CosineSimilarity(dim=1)

    # Multi-Restart Training
    best_state_dict = None
    best_loss = float('inf')
    best_eval_metric = -float('inf')
    best_train_metric = -float('inf')
    best_overall_metric = -float('inf')
    best_losses_train = []
    best_losses_val = []
    
    # Store all discovered rules with their accuracies
    all_discovered_rules = []
    
    # Dictionary to store training times for each num_rules
    training_times = {}

    for num_rules in rule_counts_to_train:
        # Start timing for this num_rules
        start_time = time.time()
        for restart in range(num_restarts):
            print(f"\n=== Training ANDRE with {num_rules} rules, Restart {restart + 1}/{num_restarts} ===")
            model = ANDRE(num_rules, pot_body_pred_dim=input_dim, 
                         variable_matrix=variable_matrix if structured_data else None, 
                         lr=args.lr, weight_decay=args.weight_decay)
            model = model.to(device)

            losses_andre_train = []
            losses_andre_val = []

            for epoch in range(num_epochs):
                model.train()

                # Schedule loss weights
                if args.loss_schedule:
                    ent_weight_, rr_weight_, conn_weight_, digit_weight_, sim_weight_ = schedule_loss_weights(
                                epoch=epoch, total_epochs=num_epochs,
                                final_ent_weight=ent_weight, 
                                final_rr_weight=rr_weight, 
                                final_conn_weight=conn_weight,
                                final_digit_weight=digit_weight,
                                initial_sim_weight=2.0,
                                final_sim_weight=sim_weight
                                )
                else:
                    ent_weight_ = ent_weight
                    rr_weight_ = rr_weight
                    conn_weight_ = conn_weight
                    digit_weight_ = digit_weight
                    sim_weight_ = sim_weight

                epoch_loss = 0.0

                # Create balanced batch sampler
                batch_sampler = BalancedBatchSampler(y_train, batch_size)
                
                # Mini-batch training loop with balanced batches
                for batch_indices in batch_sampler:
                    xb = X_train[batch_indices]
                    yb = y_train[batch_indices]

                    model.optimizer.zero_grad()
                    outputs, andre_probs, weight_entropies, Ms = model(xb)

                    # Binary Cross-Entropy Loss (no weighting needed with balanced batches)
                    bce_loss = nn.BCELoss()(outputs, yb)

                    # Similarity Penalty
                    sim_loss = 0.0
                    num_sim_pairs = 0
                    for i in range(num_rules):
                        wi = andre_probs[i].view(input_dim, -1)
                        for j in range(i + 1, num_rules):
                            wj = andre_probs[j].view(input_dim, -1)
                            sim_loss += nn.CosineSimilarity(dim=1)(wi, wj).mean()
                            num_sim_pairs += 1
                    sim_loss = sim_loss / num_sim_pairs if num_sim_pairs > 0 else 0

                    if structured_data:
                        # Build tensors from M entries (handle lists or tensors)
                        M_head_list = [torch.as_tensor(Ms[i][head_vars], device=xb.device, dtype=xb.dtype) for i in range(num_rules)]
                        M_head = torch.stack(M_head_list)

                        # Range Restriction Loss with asymmetric penalty
                        eta = args.eta  # Small penalty for values above 1
                        mask_below = (M_head < 1.0).float()
                        mask_above = (M_head >= 1.0).float()
                        rr_loss = mask_below * (M_head - 1.0)**2 + mask_above * eta * (M_head - 1.0)
                        rr_loss = rr_loss.sum() / len(head_vars) / num_rules

                        # Connectivity Loss
                        M_connected_list = [torch.as_tensor(Ms[i][connected_vars], device=xb.device, dtype=xb.dtype) for i in range(num_rules)]
                        M_connected = torch.stack(M_connected_list)  # tensor form

                        # Connectivity Loss with conditional masking
                        mask_below_2 = (M_connected < 2.0).float()
                        mask_above_2 = (M_connected >= 2.0).float()
                        
                        # Smoothness / connectivity term computed elementwise and averaged
                        conn_loss_1 = torch.stack([torch.exp(-0.5 * ((M_conn - 1.0) / 0.2)**2) / (0.2 * math.sqrt(2 * math.pi)) for M_conn in M_connected]).sum()
                        conn_loss_1 = 0.5 * conn_loss_1
                        
                        # MSE against constant 2
                        conn_loss_2 = 1 / (input_dim**2) * (M_connected - 2.0)**2
                        
                        # Apply masking: use conn_loss_1 if below 2, conn_loss_2 if above or equal to 2
                        conn_loss = mask_below_2 * conn_loss_1 + mask_above_2 * conn_loss_2
                        # conn_loss = conn_loss_2
                        conn_loss = conn_loss.sum() / len(connected_vars) / num_rules

                        # Digitization Loss - encourage M elements to be integers
                        M_all_list = [torch.as_tensor(Ms[i], device=xb.device, dtype=xb.dtype) for i in range(num_rules)]
                        M_all = torch.stack(M_all_list)  # [num_rules, num_variables]
                        digit_loss = 0.5 * (1 - torch.sin(2 * math.pi * M_all + math.pi / 2)).sum()
                        digit_loss = digit_loss / M_all.numel()

                        # Entropy Loss
                        entropy_term = torch.stack(weight_entropies).sum()
                        entropy_term = entropy_term / num_rules /  input_dim

                        # Total Loss
                        total_loss = bce_loss + ent_weight_ * entropy_term + \
                                    rr_weight_ * rr_loss + conn_weight_ * conn_loss + \
                                    digit_weight_ * digit_loss + sim_weight_ * sim_loss
                        # if epoch % 10 == 0:
                        #     print(f"Epoch {epoch+1}, Batch {i//batch_size+1}: BCE={bce_loss.item():.4f}, RR={rr_weight_ * rr_loss.item():.4f}, Conn={conn_weight_ * conn_loss.item():.4f}, Digit={digit_weight_ * digit_loss.item():.4f}, Ent={ent_weight_ * entropy_term.item():.4f}, Sim={sim_weight_ * float(sim_loss):.4f}")
                    else: 
                        entropy_term = torch.stack(weight_entropies).sum()
                        entropy_term = entropy_term / num_rules /  input_dim
                        total_loss = bce_loss + ent_weight_ * entropy_term + sim_weight_ * sim_loss

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    model.optimizer.step()

                    epoch_loss += total_loss.item()

                model.scheduler.step()
                losses_andre_train.append(epoch_loss)
                
                # Compute validation loss
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for i in range(0, X_val.size(0), batch_size):
                        xb_val = X_val[i:i + batch_size]
                        yb_val = y_val[i:i + batch_size]
                        
                        outputs_val, andre_probs_val, weight_entropies_val, Ms_val = model(xb_val)
                        bce_loss_val = model.criterion(outputs_val, yb_val)
                        
                        # Compute other loss components for validation
                        if structured_data:
                            # Similarity Penalty
                            sim_loss_val = 0.0
                            num_sim_pairs = 0
                            for i_r in range(num_rules):
                                wi = andre_probs_val[i_r].view(input_dim, -1)
                                for j_r in range(i_r + 1, num_rules):
                                    wj = andre_probs_val[j_r].view(input_dim, -1)
                                    sim_loss_val += nn.CosineSimilarity(dim=1)(wi, wj).mean()
                                    num_sim_pairs += 1
                            sim_loss_val = sim_loss_val / num_sim_pairs if num_sim_pairs > 0 else 0
                            
                            # Range Restriction Loss
                            M_head_list_val = [torch.as_tensor(Ms_val[i_r][head_vars], device=xb_val.device, dtype=xb_val.dtype) for i_r in range(num_rules)]
                            M_head_val = torch.stack(M_head_list_val)
                            eta = args.eta  # Small penalty for values above 1
                            mask_below = (M_head_val < 1.0).float()
                            mask_above = (M_head_val >= 1.0).float()
                            rr_loss_val = mask_below * (M_head_val - 1.0)**2 + mask_above * eta * (M_head_val - 1.0)
                            rr_loss_val = rr_loss_val.sum() / len(head_vars) / num_rules
                            
                            # Connectivity Loss
                            M_connected_list_val = [torch.as_tensor(Ms_val[i_r][connected_vars], device=xb_val.device, dtype=xb_val.dtype) for i_r in range(num_rules)]
                            M_connected_val = torch.stack(M_connected_list_val)
                            mask_below_2 = (M_connected_val < 2.0).float()
                            mask_above_2 = (M_connected_val >= 2.0).float()
                            conn_loss_1_val = torch.stack([torch.exp(-0.5 * ((M_conn - 1.0) / 0.2)**2) / (0.2 * math.sqrt(2 * math.pi)) for M_conn in M_connected_val]).sum()
                            conn_loss_1_val = 0.5 * conn_loss_1_val
                            conn_loss_2_val = (M_connected_val - 2.0)**2
                            conn_loss_val = mask_below_2 * conn_loss_1_val + mask_above_2 * conn_loss_2_val
                            conn_loss_val = conn_loss_val.sum() / len(connected_vars) / num_rules
                            
                            # Digitization Loss
                            M_all_list_val = [torch.as_tensor(Ms_val[i_r], device=xb_val.device, dtype=xb_val.dtype) for i_r in range(num_rules)]
                            M_all_val = torch.stack(M_all_list_val)
                            digit_loss_val = 0.5 * (1 - torch.sin(2 * math.pi * M_all_val + math.pi / 2)).sum()
                            digit_loss_val = digit_loss_val / M_all_val.numel()
                            
                            # Entropy Loss
                            entropy_term_val = torch.stack(weight_entropies_val).sum()
                            entropy_term_val = entropy_term_val / num_rules / input_dim
                            
                            total_loss_val = bce_loss_val + ent_weight_ * entropy_term_val + \
                                           rr_weight_ * rr_loss_val + conn_weight_ * conn_loss_val + \
                                           digit_weight_ * digit_loss_val + sim_weight_ * sim_loss_val
                        else:
                            entropy_term_val = torch.stack(weight_entropies_val).sum()
                            entropy_term_val = entropy_term_val / num_rules / input_dim
                            
                            sim_loss_val = 0.0
                            num_sim_pairs = 0
                            for i_r in range(num_rules):
                                wi = andre_probs_val[i_r].view(input_dim, -1)
                                for j_r in range(i_r + 1, num_rules):
                                    wj = andre_probs_val[j_r].view(input_dim, -1)
                                    sim_loss_val += nn.CosineSimilarity(dim=1)(wi, wj).mean()
                                    num_sim_pairs += 1
                            sim_loss_val = sim_loss_val / num_sim_pairs if num_sim_pairs > 0 else 0
                            
                            total_loss_val = bce_loss_val + ent_weight_ * entropy_term_val + sim_weight_ * sim_loss_val
                        
                        val_loss += total_loss_val.item()
                
                losses_andre_val.append(val_loss)

            # Extract RAW weights from model parameters (not softmaxed)
            raw_weights = []
            for rule_block in model.rules:
                # Get raw_weights from the activation layer [D, 3]
                raw_w = rule_block.activation.raw_weights.detach().cpu().numpy()
                raw_weights.append(raw_w.tolist())

            # Extract softmaxed probabilities for rule interpretation
            # Do a forward pass to get the softmaxed probabilities
            with torch.no_grad():
                _, andre_probs_output, _, _ = model(X_train[:min(100, len(X_train))])
            
            andre_probs = []
            for rule_probs_tensor in andre_probs_output:
                rule_probs_tensor = rule_probs_tensor.detach().cpu().numpy()
                body_predicate_probs = rule_probs_tensor.reshape(-1, 3)
                andre_probs.append(body_predicate_probs.tolist())

            # Rule interpretation
            final_rule = extract_rules(andre_probs=andre_probs, X_train=X_train, 
                                    dataset_type=dataset_type, body_predicates=body_predicates,
                                    head_predicate=head_predicate)
            
            if verbose:
                print("\n======================================")
                print("Final Rule:")
                print(final_rule)
                print("======================================")

            training_metrics = model.evaluate(X_train, y_train)
            evaluation_metrics = model.evaluate(X_val, y_val)

            eval_metric = 0.5 * (evaluation_metrics['accuracy'] + evaluation_metrics['recall'])
            train_metric = 0.5 * (training_metrics['accuracy'] + training_metrics['recall'])
            overall_metric = 0.5 * (train_metric + eval_metric)

            # Store each individual subrule with its metrics
            for rule_idx, rule_probs in enumerate(andre_probs):
                # Extract individual rule text
                # if dataset_type == "binary":
                #     preds = interpret_neuron_weights(body_preds=body_predicates, andre_probs=rule_probs)
                # else:
                #     sample_pred_values = X_train[0].cpu().numpy()
                #     preds = interpret_neuron_weights_entropy(sample_pred_values, rule_probs)
                preds = interpret_neuron_weights(body_preds=body_predicates, andre_probs=rule_probs)
                
                refined_preds = [pred for pred in preds if pred != '1' and pred != '?']
                subrule_text = " and ".join(refined_preds) if refined_preds else "true"
                full_subrule = f"{head_predicate} :- {subrule_text}."
                
                # Evaluate this individual subrule using RAW weights
                subrule_metrics = evaluate_single_rule(
                    raw_weights[rule_idx],
                    X_train,
                    y_train,
                    variable_matrix if structured_data else None,
                    device
                )
                subrule_eval_metrics = evaluate_single_rule(
                    raw_weights[rule_idx],
                    X_val,
                    y_val,
                    variable_matrix if structured_data else None,
                    device
                )
                
                # Check if this rule already exists (avoid duplicates)
                rule_exists = any(r['rule'] == full_subrule for r in all_discovered_rules)
                
                if not rule_exists:
                    all_discovered_rules.append({
                        'rule': full_subrule,
                        'training_accuracy': subrule_metrics['accuracy'],
                        'eval_accuracy': subrule_eval_metrics['accuracy'],
                        'num_rules': num_rules,
                        'restart': restart + 1,
                        'subrule_idx': rule_idx + 1,
                        'training_precision': subrule_metrics['precision'],
                        'training_recall': subrule_metrics['recall'],
                        'training_f1': subrule_metrics['f1_score'],
                        'eval_precision': subrule_eval_metrics['precision'],
                        'eval_recall': subrule_eval_metrics['recall'],
                        'eval_f1': subrule_eval_metrics['f1_score'],
                        'raw_weights': raw_weights[rule_idx]  # Store RAW weights, not softmaxed
                    })

            print(f"Num Rules: {num_rules}, Restart {restart+1}/{num_restarts}, Train Metric: {train_metric:.4f}, Evaluation Metric: {eval_metric:.4f}.")
            
            # Print current discovered subrules count
            print(f"Total unique subrules discovered so far: {len(all_discovered_rules)}")
            
            # Select best model based on chosen criterion
            current_criterion_losses = losses_andre_val if args.best_model_criterion == 'val_loss' else losses_andre_train
            current_min_loss = min(current_criterion_losses)
            
            # if current_min_loss < best_loss:
            if eval_metric > best_eval_metric:
                print(f"**** New Best Model (based on {args.best_model_criterion}) ****")
                best_eval_metric = eval_metric
                best_train_metric = train_metric
                best_overall_metric = overall_metric
                best_state_dict = deepcopy(model.state_dict())
                best_losses_train = losses_andre_train
                best_losses_val = losses_andre_val
                best_num_rules = num_rules
                best_loss = current_min_loss

                if train_metric > acc_threshold and eval_metric > acc_threshold:
                    print(f"The accuracy threshold of {acc_threshold} has been reached. Stopping further restarts.")
                    break
        
        # End timing for this num_rules
        end_time = time.time()
        training_times[num_rules] = end_time - start_time
        print(f"\n*** Training time for {num_rules} rules: {training_times[num_rules]:.2f} seconds ({training_times[num_rules]/60:.2f} minutes) ***\n")

    # Print summary of all training times
    print("\n" + "="*70)
    print("TRAINING TIME SUMMARY")
    print("="*70)
    for num_rules in sorted(training_times.keys()):
        elapsed = training_times[num_rules]
        print(f"  {num_rules} rules: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("="*70 + "\n")

    # Load the best model
    best_model = ANDRE(best_num_rules, pot_body_pred_dim=input_dim, 
                      variable_matrix=variable_matrix if structured_data else None, 
                      lr=args.lr, weight_decay=args.weight_decay)
    best_model.load_state_dict(best_state_dict)
    best_model = best_model.to(device)

    # Find optimal decision threshold on validation set
    print("\n=== Finding Optimal Decision Threshold ===")
    best_model.eval()
    with torch.no_grad():
        val_probs = best_model(X_val)[0].cpu().numpy().flatten()
        y_val_np = y_val.cpu().numpy().flatten()
        
        # Try different thresholds
        thresholds = np.linspace(0.05, 0.95, 91)
        best_f1 = 0
        best_threshold = 0.5
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (val_probs > threshold).astype(float)
            tp = ((y_pred == 1) & (y_val_np == 1)).sum()
            fp = ((y_pred == 1) & (y_val_np == 0)).sum()
            fn = ((y_pred == 0) & (y_val_np == 1)).sum()
            tn = ((y_pred == 0) & (y_val_np == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal decision threshold: {best_threshold:.3f}")
        print(f"  F1-Score: {best_f1:.4f}")
        
        # Show metrics at default 0.5 threshold for comparison
        default_result = [r for r in threshold_results if abs(r['threshold'] - 0.5) < 0.01][0]
        print(f"\nComparison with default threshold (0.5):")
        print(f"  Default - F1: {default_result['f1']:.4f}, Precision: {default_result['precision']:.4f}, Recall: {default_result['recall']:.4f}")
        
        optimal_result = [r for r in threshold_results if abs(r['threshold'] - best_threshold) < 0.01][0]
        print(f"  Optimal - F1: {optimal_result['f1']:.4f}, Precision: {optimal_result['precision']:.4f}, Recall: {optimal_result['recall']:.4f}")
        
        # Store optimal threshold in model (as an attribute)
        best_model.decision_threshold = best_threshold
        
        # Save threshold results to file
        threshold_results_path = os.path.join(results_dir, f"threshold_analysis_{today_date}_{today_hour}.json")
        with open(threshold_results_path, 'w') as f:
            json.dump({
                'optimal_threshold': float(best_threshold),
                'optimal_f1': float(best_f1),
                'default_threshold_metrics': default_result,
                'optimal_threshold_metrics': optimal_result,
                'all_thresholds': threshold_results
            }, f, indent=2)
        print(f"Saved threshold analysis to: {threshold_results_path}")

    # Evaluate best model

    print("\n=== Best Model Training Evaluation (with default threshold 0.5) ===")

    best_model.print_evaluation(X_train, y_train, threshold=0.5)

    print("\n=== Best Model Evaluation on Validation Set (with default threshold 0.5) ===")
    best_model.print_evaluation(X_val, y_val, threshold=0.5)
    
    print("\n=== Best Model Evaluation on Validation Set (with optimal threshold) ===")
    best_model.print_evaluation(X_val, y_val, threshold=best_threshold)
    
    # Save best model metrics to JSON
    best_model.eval()
    with torch.no_grad():
        best_train_metrics = best_model.evaluate(X_train, y_train, threshold=best_threshold)
        best_val_metrics = best_model.evaluate(X_val, y_val, threshold=best_threshold)
        best_val_metrics_default = best_model.evaluate(X_val, y_val, threshold=0.5)
    
    best_model_metrics = {
        'num_rules': best_num_rules,
        'optimal_threshold': float(best_threshold),
        'training': {
            'accuracy': float(best_train_metrics['accuracy']),
            'precision': float(best_train_metrics['precision']),
            'recall': float(best_train_metrics['recall']),
            'f1_score': float(best_train_metrics['f1_score']),
            'roc_auc': float(best_train_metrics['roc_auc']),
            'specificity': float(best_train_metrics['specificity'])
        },
        'validation_optimal_threshold': {
            'accuracy': float(best_val_metrics['accuracy']),
            'precision': float(best_val_metrics['precision']),
            'recall': float(best_val_metrics['recall']),
            'f1_score': float(best_val_metrics['f1_score']),
            'roc_auc': float(best_val_metrics['roc_auc']),
            'specificity': float(best_val_metrics['specificity'])
        },
        'validation_default_threshold': {
            'accuracy': float(best_val_metrics_default['accuracy']),
            'precision': float(best_val_metrics_default['precision']),
            'recall': float(best_val_metrics_default['recall']),
            'f1_score': float(best_val_metrics_default['f1_score']),
            'roc_auc': float(best_val_metrics_default['roc_auc']),
            'specificity': float(best_val_metrics_default['specificity'])
        },
        'best_model_criterion': args.best_model_criterion,
        'final_train_loss': float(best_losses_train[-1]) if best_losses_train else None,
        'final_val_loss': float(best_losses_val[-1]) if best_losses_val else None,
        'min_train_loss': float(min(best_losses_train)) if best_losses_train else None,
        'min_val_loss': float(min(best_losses_val)) if best_losses_val else None
    }
    
    best_model_metrics_path = os.path.join(results_dir, f"best_model_metrics_{today_date}_{today_hour}.json")
    with open(best_model_metrics_path, 'w') as f:
        json.dump(best_model_metrics, f, indent=2)
    print(f"\nSaved best model metrics to: {best_model_metrics_path}")

    # Compute MRR and Hits@K using corrupted validation set
    print("\n=== Computing MRR and Hits@K on Validation Set for Best Model ===")
    # Split validation set into positive and negative samples
    X_val_pos = X_val[y_val == 1]
    X_val_neg = X_val[y_val == 0]
    print(f"Positive validation samples: {X_val_pos.shape[0]}")
    print(f"Negative validation samples: {X_val_neg.shape[0]}")
    mrr, hits = compute_mrr_hits(X_val, y_val, best_model, k=args.k, r=args.r)
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    for k in [1, 3, 10]:
        print(f"Hits@{k}: {hits[k]:.4f}")

    # Plot loss curves (both training and validation)
    if visualize_weights:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), best_losses_train, label="Training Loss", color="blue", linewidth=2)
        plt.plot(range(1, num_epochs + 1), best_losses_val, label="Validation Loss", color="orange", linewidth=2)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Training and Validation Loss - ({best_num_rules} Rules)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/loss_curve_{today_date}_{today_hour}.png", dpi=150, bbox_inches='tight')
        plt.show()

    _, andre_probs_output, _, _ = best_model(X_val)

    andre_probs = []
    for rule_probs in andre_probs_output:
        rule_probs = rule_probs.detach().cpu().numpy()
        body_predicate_probs = rule_probs.reshape(-1, 3)
        andre_probs.append(body_predicate_probs.tolist())

    if visualize_weights:
        plot_andre_weights(andre_probs, input_dim=input_dim, predicates=body_predicates, structured_data=structured_data)

    # Rule interpretation
    final_rule = extract_rules(andre_probs=andre_probs, X_train=X_train, 
                               dataset_type=dataset_type, body_predicates=body_predicates,
                               head_predicate=head_predicate)
    print("\n======================================")
    print("Final Rule:")
    print(final_rule)
    print("======================================")

    # Save the best model
    if save_model:
        best_model.save_model(save_dir=f"{models_dir}/andre_best_model_{today_date}", 
                             model_name=f"ANDRE_best_model_{today_hour}")

    # Evaluate individual rules and sort by accuracy
    if args.evaluate_individual_rules:
        print("\n" + "="*70)
        print("INDIVIDUAL RULE EVALUATION")
        print("="*70)

        rule_evaluations = []

        for rule_idx, rule_probs in enumerate(andre_probs):
            print(f"\nEvaluating Rule {rule_idx + 1}...")
            
            # Evaluate this rule
            metrics = evaluate_single_rule(
                rule_probs, 
                X_train, 
                y_train, 
                variable_matrix if structured_data else None,
                device
            )
            
            # Extract the rule text
            if dataset_type == "binary":
                preds = interpret_neuron_weights(body_preds=body_predicates, andre_probs=rule_probs)
            else:
                # For probabilistic, use a representative sample
                sample_pred_values = X_train[0].cpu().numpy()
                preds = interpret_neuron_weights_entropy(sample_pred_values, rule_probs)
            
            refined_preds = [pred for pred in preds if pred != '1' and pred != '?']
            rule_text = " and ".join(refined_preds) if refined_preds else "true"
            
            # Store evaluation results
            rule_evaluations.append({
                'rule_idx': rule_idx + 1,
                'rule_text': rule_text,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'andre_probs': rule_probs
            })
            
            print(f"Rule {rule_idx + 1}: {rule_text}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

        # Sort rules by accuracy (descending)
        sorted_rules = sorted(rule_evaluations, key=lambda x: x['accuracy'], reverse=True)

        # Display sorted results
        print("\n" + "="*70)
        print("RULES SORTED BY ACCURACY (Best to Worst)")
        print("="*70)

        for rank, rule_info in enumerate(sorted_rules, 1):
            print(f"\nRank {rank}: Rule {rule_info['rule_idx']}")
            print(f"  Formula: {head_predicate} :- {rule_info['rule_text']}")
            print(f"  Accuracy:  {rule_info['accuracy']:.4f}")
            print(f"  Precision: {rule_info['precision']:.4f}")
            print(f"  Recall:    {rule_info['recall']:.4f}")
            print(f"  F1-Score:  {rule_info['f1_score']:.4f}")
            print(f"  ROC-AUC:   {rule_info['roc_auc']:.4f}")

        print("\n" + "="*70)

        # Create final sorted rule set
        sorted_rule_formulas = [f"{rule_info['rule_text']}" for rule_info in sorted_rules]
        final_sorted_rule = f"{head_predicate} :- " + " OR ".join(sorted_rule_formulas) + "."

        print("\nFINAL RULE (sorted by individual accuracy):")
        print(final_sorted_rule)
        print("="*70)

    # Print summary of all discovered rules
    print("\n" + "="*80)
    print("ALL DISCOVERED SUBRULES (DEDUPLICATED)")
    print("="*80)
    
    # Sort by eval_accuracy
    sorted_all_rules = sorted(all_discovered_rules, key=lambda x: x['eval_accuracy'], reverse=True)
    
    print(f"\nTotal unique subrules discovered: {len(sorted_all_rules)}")
    
    # for idx, rule_info in enumerate(sorted_all_rules, 1):
    #     print(f"\n[{idx}] {rule_info['rule']}")
    #     print(f"    Training Accuracy: {rule_info['training_accuracy']:.4f} | Eval Accuracy: {rule_info['eval_accuracy']:.4f}")
    #     print(f"    Training F1: {rule_info['training_f1']:.4f} | Eval F1: {rule_info['eval_f1']:.4f}")
    #     print(f"    (Found in: Num_rules={rule_info['num_rules']}, Restart={rule_info['restart']}, Subrule #{rule_info['subrule_idx']})")
    
    # print("\n" + "="*80)
    
    # Save sorted rule weights if save_model is enabled
    if save_model and len(sorted_all_rules) > 0:
        # Filter rules based on format constraints (only for structured data)
        if structured_data and head_vars is not None and connected_vars is not None:
            valid_rules, invalid_rules = filter_valid_rules(
                sorted_all_rules, 
                head_vars, 
                connected_vars,
                verbose=args.verbose
            )
            
            # Use only valid rules for the unified model
            if len(valid_rules) == 0:
                print("\nWARNING: No valid rules found! Using all rules without filtering.")
                rules_to_use = sorted_all_rules
            else:
                print(f"\nUsing {len(valid_rules)} valid rules for unified model (filtered from {len(sorted_all_rules)} total rules)")
                rules_to_use = valid_rules
        else:
            print("\nSkipping rule validation (not structured data or head_vars/connected_vars not available)")
            rules_to_use = sorted_all_rules
        
        save_dir_rules = f"{results_dir}/sorted_rules_{today_date}_{today_hour}"
        os.makedirs(save_dir_rules, exist_ok=True)
        
        # Prepare rule weights for export (nested list format)
        sorted_rule_weights = [rule_info['raw_weights'] for rule_info in rules_to_use]
        
        # Build a unified model from sorted rules
        unified_model = ANDRE.build_from_rule_weights(
            sorted_rule_weights,
            pot_body_pred_dim=input_dim,
            variable_matrix=variable_matrix if structured_data else None
        )
        unified_model = unified_model.to(device)
        
        # Evaluate unified model performance
        print("\n" + "="*70)
        print("UNIFIED MODEL EVALUATION")
        print("="*70)
        
        unified_model.eval()
        with torch.no_grad():
            # Training set evaluation (using optimal threshold from best_model)
            train_metrics = unified_model.evaluate(X_train, y_train, threshold=best_threshold)
            print("\nUnified Model - Training Set Metrics (with optimal threshold):")
            print(f"  Accuracy:    {train_metrics['accuracy']:.4f}")
            print(f"  Precision:   {train_metrics['precision']:.4f}")
            print(f"  Recall:      {train_metrics['recall']:.4f}")
            print(f"  F1 Score:    {train_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:     {train_metrics['roc_auc']:.4f}")
            print(f"  Specificity: {train_metrics['specificity']:.4f}")
            
            # Validation/Test set evaluation (using optimal threshold from best_model)
            val_metrics = unified_model.evaluate(X_val, y_val, threshold=best_threshold)
            print("\nUnified Model - Validation/Test Set Metrics (with optimal threshold):")
            print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
            print(f"  Precision:   {val_metrics['precision']:.4f}")
            print(f"  Recall:      {val_metrics['recall']:.4f}")
            print(f"  F1 Score:    {val_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:     {val_metrics['roc_auc']:.4f}")
            print(f"  Specificity: {val_metrics['specificity']:.4f}")
            
            # Save unified model metrics
            unified_metrics = {
                'optimal_threshold': float(best_threshold),
                'training': {
                    'accuracy': float(train_metrics['accuracy']),
                    'precision': float(train_metrics['precision']),
                    'recall': float(train_metrics['recall']),
                    'f1_score': float(train_metrics['f1_score']),
                    'roc_auc': float(train_metrics['roc_auc']),
                    'specificity': float(train_metrics['specificity'])
                },
                'validation': {
                    'accuracy': float(val_metrics['accuracy']),
                    'precision': float(val_metrics['precision']),
                    'recall': float(val_metrics['recall']),
                    'f1_score': float(val_metrics['f1_score']),
                    'roc_auc': float(val_metrics['roc_auc']),
                    'specificity': float(val_metrics['specificity'])
                },
                'num_rules': len(rules_to_use)
            }
            
            unified_metrics_path = os.path.join(save_dir_rules, "unified_model_metrics.json")
            with open(unified_metrics_path, 'w') as f:
                json.dump(unified_metrics, f, indent=2)
            print(f"\nSaved unified model metrics to: {unified_metrics_path}")
        
        print("="*70)
        
        # Export the sorted rule weights
        pt_path, json_path = unified_model.export_rule_weights(save_dir_rules, file_prefix="sorted_rules")
        
        # Save metadata about valid rules used in unified model
        rules_metadata = []
        for rank, rule_info in enumerate(rules_to_use, 1):
            # Create serializable copy (exclude weights from JSON metadata)
            metadata_entry = {k: v for k, v in rule_info.items() if k != 'weights'}
            metadata_entry['rank'] = rank
            rules_metadata.append(metadata_entry)
        
        metadata_path = os.path.join(save_dir_rules, "sorted_rules_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(rules_metadata, f, indent=2)
        
        # Use individual rule evaluation metrics as scores instead of unified model predictions
        y_train_np = y_train.cpu().numpy()
        y_val_np = y_val.cpu().numpy()
        
        # Determine which metric to use for rule scores
        if args.rule_confidence_metric == 'nr_nb_ratio':
            # Will compute N_r/N_b ratio per rule during coverage statistics
            rule_score_metric = 'nr_nb_ratio'
        else:
            # Use validation accuracy from individual rule evaluation
            rule_score_metric = 'eval_accuracy'
        
        # Initialize score arrays with individual rule metrics
        scores_val_np = np.zeros((y_val_np.shape[0], len(rules_to_use)))
        scores_train_np = np.zeros((y_train_np.shape[0], len(rules_to_use)))
        
        print(f"\nUsing '{rule_score_metric}' as rule score metric for CSV export")
        
        # Print valid rules being used
        print(f"\nValid rules used for unified model ({len(rules_to_use)} rules):")
        
        # Compute coverage statistics for each rule and populate score arrays
        rule_coverage_stats_val = []
        rule_coverage_stats_train = []
        
        for idx, rule_info in enumerate(rules_to_use, 1):
            # Get individual rule predictions using evaluate_single_rule
            with torch.no_grad():
                # Evaluate on validation set
                single_rule_model_val = ANDRE.build_from_rule_weights(
                    [rule_info['raw_weights']],
                    pot_body_pred_dim=input_dim,
                    variable_matrix=variable_matrix if structured_data else None
                )
                single_rule_model_val = single_rule_model_val.to(device)
                single_rule_model_val.eval()
                val_outputs = single_rule_model_val(X_val)[0].cpu().numpy().flatten()
                rule_predictions_val = val_outputs > 0.5
                
                # Evaluate on training set
                train_outputs = single_rule_model_val(X_train)[0].cpu().numpy().flatten()
                rule_predictions_train = train_outputs > 0.5
            
            # Validation set statistics
            N_b_val = np.sum(rule_predictions_val)
            N_r_val = np.sum(rule_predictions_val & (y_val_np.flatten() == 1))
            ratio_val = N_r_val / N_b_val if N_b_val > 0 else 0.0
            rule_coverage_stats_val.append((int(N_b_val), int(N_r_val), float(ratio_val)))
            
            # Training set statistics
            N_b_train = np.sum(rule_predictions_train)
            N_r_train = np.sum(rule_predictions_train & (y_train_np.flatten() == 1))
            ratio_train = N_r_train / N_b_train if N_b_train > 0 else 0.0
            rule_coverage_stats_train.append((int(N_b_train), int(N_r_train), float(ratio_train)))
            
            # Set rule scores based on selected metric
            if rule_score_metric == 'nr_nb_ratio':
                # Use N_r/N_b ratio as score (broadcast to all samples)
                scores_val_np[:, idx-1] = ratio_val
                scores_train_np[:, idx-1] = ratio_train
            else:
                # Use validation accuracy as score (broadcast to all samples)
                scores_val_np[:, idx-1] = rule_info['eval_accuracy']
                scores_train_np[:, idx-1] = rule_info['training_accuracy']
            
            print(f"  [{idx}] {rule_info['rule']}")
            print(f"      Training Acc: {rule_info['training_accuracy']:.4f} | Eval Acc: {rule_info['eval_accuracy']:.4f}")
            print(f"      Val Coverage: (N_b={N_b_val}, N_r={N_r_val}, N_r/N_b={ratio_val:.4f})")
            print(f"      Train Coverage: (N_b={N_b_train}, N_r={N_r_train}, N_r/N_b={ratio_train:.4f})")
            # print(f"      Rule Score (val): {scores_val_np[0, idx-1]:.4f}")
            print("\n")
        
        # Save rule scores to CSV (using individual rule metrics, not adjusted correctness)
        df_train_initial = pd.DataFrame(scores_train_np,
                                        columns=[f"rule_rank_{i+1}_score" for i in range(len(rules_to_use))])
        df_train_initial['label'] = y_train_np
        df_train_initial.to_csv(os.path.join(save_dir_rules, "train_sorted_rule_scores.csv"), index=False)
        
        df_val_initial = pd.DataFrame(scores_val_np,
                                      columns=[f"rule_rank_{i+1}_score" for i in range(len(rules_to_use))])
        df_val_initial['label'] = y_val_np
        df_val_initial.to_csv(os.path.join(save_dir_rules, "val_sorted_rule_scores.csv"), index=False)
        
        print(f"\nSaved rule scores to CSV files in: {save_dir_rules}")
        
        # If using nr_nb_ratio metric, re-sort rules based on N_r/N_b ratio
        if args.rule_confidence_metric == 'nr_nb_ratio':
            print("\n" + "="*70)
            print("RE-SORTING RULES BY N_r/N_b RATIO (Validation Set)")
            print("="*70)
            
            # Create list of (rule_info, coverage_stats) tuples
            rules_with_stats = list(zip(rules_to_use, rule_coverage_stats_val, rule_coverage_stats_train))
            
            # Sort by N_r/N_b ratio (descending)
            rules_with_stats.sort(key=lambda x: x[1][2], reverse=True)
            
            # Unpack back to separate lists
            rules_to_use = [item[0] for item in rules_with_stats]
            rule_coverage_stats_val = [item[1] for item in rules_with_stats]
            rule_coverage_stats_train = [item[2] for item in rules_with_stats]
            
            # Rebuild the unified model with re-sorted rules
            sorted_rule_weights = [rule_info['raw_weights'] for rule_info in rules_to_use]
            unified_model = ANDRE.build_from_rule_weights(
                sorted_rule_weights,
                pot_body_pred_dim=input_dim,
                variable_matrix=variable_matrix if structured_data else None
            )
            unified_model = unified_model.to(device)
            
            # Re-evaluate unified model with new ordering (using optimal threshold)
            print("\nRe-evaluating unified model with N_r/N_b sorted rules...")
            unified_model.eval()
            with torch.no_grad():
                train_metrics = unified_model.evaluate(X_train, y_train, threshold=best_threshold)
                val_metrics = unified_model.evaluate(X_val, y_val, threshold=best_threshold)
                
                print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")

                mrr, hits = compute_mrr_hits(X_val, y_val, unified_model, k=args.k, r=args.r)
                print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
                for k in [1, 3, 10]:
                    print(f"Hits@{k}: {hits[k]:.4f}")
            
            # Re-compute scores with new rule ordering (using individual rule metrics)
            scores_train_np = np.zeros((y_train_np.shape[0], len(rules_to_use)))
            scores_val_np = np.zeros((y_val_np.shape[0], len(rules_to_use)))
            
            for idx, rule_info in enumerate(rules_to_use):
                if rule_score_metric == 'nr_nb_ratio':
                    # Use N_r/N_b ratio from re-sorted coverage stats
                    scores_val_np[:, idx] = rule_coverage_stats_val[idx][2]
                    scores_train_np[:, idx] = rule_coverage_stats_train[idx][2]
                else:
                    # Use validation accuracy
                    scores_val_np[:, idx] = rule_info['eval_accuracy']
                    scores_train_np[:, idx] = rule_info['training_accuracy']
            
            # Update metadata ranks
            rules_metadata = []
            for rank, rule_info in enumerate(rules_to_use, 1):
                metadata_entry = {k: v for k, v in rule_info.items() if k != 'raw_weights'}
                metadata_entry['rank'] = rank
                rules_metadata.append(metadata_entry)
            
            # Update saved files with re-sorted data
            with open(metadata_path, 'w') as f:
                json.dump(rules_metadata, f, indent=2)
            
            df_train = pd.DataFrame(scores_train_np,
                                   columns=[f"rule_rank_{i+1}_score" for i in range(len(rules_to_use))])
            df_train['label'] = y_train_np
            df_train.to_csv(os.path.join(save_dir_rules, "train_sorted_rule_scores.csv"), index=False)
            
            df_val = pd.DataFrame(scores_val_np,
                                 columns=[f"rule_rank_{i+1}_score" for i in range(len(rules_to_use))])
            df_val['label'] = y_val_np
            df_val.to_csv(os.path.join(save_dir_rules, "val_sorted_rule_scores.csv"), index=False)
            
            # Re-export weights
            pt_path, json_path = unified_model.export_rule_weights(save_dir_rules, file_prefix="sorted_rules")
            
            # Update unified model metrics
            unified_metrics = {
                'optimal_threshold': float(best_threshold),
                'training': {
                    'accuracy': float(train_metrics['accuracy']),
                    'precision': float(train_metrics['precision']),
                    'recall': float(train_metrics['recall']),
                    'f1_score': float(train_metrics['f1_score']),
                    'roc_auc': float(train_metrics['roc_auc']),
                    'specificity': float(train_metrics['specificity'])
                },
                'validation': {
                    'accuracy': float(val_metrics['accuracy']),
                    'precision': float(val_metrics['precision']),
                    'recall': float(val_metrics['recall']),
                    'f1_score': float(val_metrics['f1_score']),
                    'roc_auc': float(val_metrics['roc_auc']),
                    'specificity': float(val_metrics['specificity'])
                },
                'num_rules': len(rules_to_use),
                'sorting_metric': 'nr_nb_ratio'
            }
            with open(unified_metrics_path, 'w') as f:
                json.dump(unified_metrics, f, indent=2)
            
            print(f"\nRules re-sorted and files updated based on N_r/N_b ratio!")
            print("="*70)
        
        # Extract rule confidences based on selected metric
        if args.rule_confidence_metric == 'nr_nb_ratio':
            # Use N_r/N_b ratio from validation set coverage statistics
            rule_confidences = np.array([stats[2] for stats in rule_coverage_stats_val])
            print(f"\nUsing 'nr_nb_ratio' (N_r/N_b from validation set) as rule confidence metric")
        else:
            confidence_metric_map = {
                'val_accuracy': 'eval_accuracy',
                'val_f1': 'eval_f1',
                'train_accuracy': 'training_accuracy',
                'val_precision': 'eval_precision'
            }
            confidence_key = confidence_metric_map[args.rule_confidence_metric]
            rule_confidences = np.array([rule[confidence_key] for rule in rules_metadata])
            print(f"\nUsing '{args.rule_confidence_metric}' as rule confidence metric")
        
        print(f"Rule confidences: {rule_confidences}")
        
        # Save ranking metrics with coverage statistics to JSON
        ranking_metrics_path = os.path.join(save_dir_rules, "ranking_metrics.json")
        ranking_results = {
            'confidence_metric': args.rule_confidence_metric,
            'rule_confidences': rule_confidences.tolist(),
            'rule_coverage_stats_validation': rule_coverage_stats_val,
            'rule_coverage_stats_training': rule_coverage_stats_train
        }
        with open(ranking_metrics_path, 'w') as f:
            json.dump(ranking_results, f, indent=2)
        
        print(f"\nSaved sorted rule weights and scores to: {save_dir_rules}")
        print(f"  - {pt_path}")
        print(f"  - {json_path}")
        print(f"  - {metadata_path}")
        print(f"  - train_sorted_rule_scores.csv")
        print(f"  - val_sorted_rule_scores.csv")
        print(f"  - {ranking_metrics_path}")
        
        # Final Summary: Compare Best Model vs Unified Model
        print("\n" + "="*70)
        print("FINAL SUMMARY - MODEL COMPARISON ON TEST/VALIDATION SET")
        print("="*70)
        
        # Load best model metrics (already computed and saved earlier)
        with open(best_model_metrics_path, 'r') as f:
            best_model_metrics_loaded = json.load(f)
        
        print(f"\nBest Model (from training) - {best_num_rules} rules:")
        print(f"  Accuracy:    {best_model_metrics_loaded['validation_optimal_threshold']['accuracy']:.4f}")
        print(f"  Precision:   {best_model_metrics_loaded['validation_optimal_threshold']['precision']:.4f}")
        print(f"  Recall:      {best_model_metrics_loaded['validation_optimal_threshold']['recall']:.4f}")
        print(f"  F1 Score:    {best_model_metrics_loaded['validation_optimal_threshold']['f1_score']:.4f}")
        print(f"  ROC-AUC:     {best_model_metrics_loaded['validation_optimal_threshold']['roc_auc']:.4f}")
        
        # Get unified model metrics (already computed earlier)
        with open(unified_metrics_path, 'r') as f:
            unified_metrics_loaded = json.load(f)
        
        print(f"\nUnified Model (sorted valid rules) - {len(rules_to_use)} rules:")
        print(f"  Accuracy:    {unified_metrics_loaded['validation']['accuracy']:.4f}")
        print(f"  Precision:   {unified_metrics_loaded['validation']['precision']:.4f}")
        print(f"  Recall:      {unified_metrics_loaded['validation']['recall']:.4f}")
        print(f"  F1 Score:    {unified_metrics_loaded['validation']['f1_score']:.4f}")
        print(f"  ROC-AUC:     {unified_metrics_loaded['validation']['roc_auc']:.4f}")
        
        # Compute performance differences
        # acc_diff = unified_metrics_loaded['validation']['accuracy'] - best_model_metrics_loaded['validation']['accuracy']
        # f1_diff = unified_metrics_loaded['validation']['f1_score'] - best_model_metrics_loaded['validation']['f1_score']
        
        # print(f"\nPerformance Difference (Unified - Best):")
        # print(f"  Accuracy Δ:  {acc_diff:+.4f}")
        # print(f"  F1 Score Δ:  {f1_diff:+.4f}")
        
        print("="*70)
    
    # End logging session
    print("\n" + "="*70)
    print(f"Logging session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All output saved to: {log_file}")
    print("="*70)
    
    # Return the list for programmatic access
    return all_discovered_rules, results_dir


if __name__ == '__main__':
    try:
        all_rules, results_dir = main()
        
        # Optional: Save to file
        if all_rules:
            discovered_rules_path = os.path.join(results_dir, 'discovered_rules.json')
            with open(discovered_rules_path, 'w') as f:
                json.dump(all_rules, f, indent=2)
            print(f"\nSaved {len(all_rules)} discovered rules to '{discovered_rules_path}'")
    
    finally:
        # Restore stdout and stderr, close log file
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, 'terminal'):
            terminal = sys.stdout.terminal
            sys.stdout.close()
            sys.stdout = terminal
            sys.stderr = terminal
