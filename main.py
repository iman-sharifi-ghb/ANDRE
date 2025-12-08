#!/usr/bin/env python3
"""
ANDRE Neural-Symbolic Learning Main Script
This script implements ANDRE (Adaptive Neural-symbolic Differentiable Rule Engine)
for learning logical rules from data using neural networks.
"""

import argparse
import math
import pickle
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
from utils import compute_acc_ratio_all_rules_attention
from utils import extract_rules
from utils import parse_rule_and_generate_data
import numpy as np
from datetime import datetime


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model architecture
    parser.add_argument('--num_rules', type=int, default=4,
                        help='Maximum number of rules to learn')
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
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for optimizer')
    parser.add_argument('--acc_threshold', type=float, default=0.98,
                        help='Accuracy threshold for early stopping')
    
    # Output options
    parser.add_argument('--visualize_weights', type=str2bool, default=False,
                        help='Visualize learned weights')
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='Print verbose output')
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help='Save the best model')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='Directory to save models')
    
    # Device options
    parser.add_argument('--device', type=str, default='cpu', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training (auto will use CUDA if available)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    # Evaluation options
    parser.add_argument('--evaluate_individual_rules', type=str2bool, default=False,
                        help='Evaluate and sort individual rules by accuracy')
    
    return parser.parse_args()


def schedule_loss_weights(epoch, total_epochs, 
                        initial_ent_weight=0.0, final_ent_weight=0.1, 
                        initial_rr_weight=0.0, final_rr_weight=1.0, 
                        initial_conn_weight=0.0, final_conn_weight=3.0, 
                        initial_digit_weight=0.0, final_digit_weight=0.05, 
                        initial_sim_weight=1.0, final_sim_weight=0.1):
    
    """Schedule weights for range restriction, connectivity, and digitization losses."""
    # Linearly increase rr_weight and conn_weight from 0 to their initial values over the epochs
    ent_weight = initial_ent_weight + (epoch / total_epochs) * (final_ent_weight - initial_ent_weight)
    rr_weight = initial_rr_weight + (epoch / total_epochs)**2 * (final_rr_weight - initial_rr_weight)
    conn_weight = initial_conn_weight + (epoch / total_epochs)**2 * (final_conn_weight - initial_conn_weight)
    digit_weight = initial_digit_weight + (epoch / total_epochs)**2 * (final_digit_weight - initial_digit_weight)
    sim_weight = initial_sim_weight + (epoch / total_epochs)**2 * (final_sim_weight - initial_sim_weight)

    ent_weight = min(ent_weight, final_ent_weight)
    rr_weight = min(rr_weight, final_rr_weight)
    conn_weight = min(conn_weight, final_conn_weight)
    digit_weight = min(digit_weight, final_digit_weight)
    sim_weight = min(sim_weight, final_sim_weight)
    
    return ent_weight, rr_weight, conn_weight, digit_weight, sim_weight


def get_predicate_name(index, predicate_list, relation_variable_permutations, all_relations):
    """
    Convert a feature index in X_train to its symbolic predicate name.
    """
    # Get variable mapping (e.g., (0, 1) for X, Y)
    var_tuple = predicate_list[index]
    
    # Variable name mapping
    var_names = {0: 'X1', 1: 'X2', 2: 'X3', 3: 'X4', 4: 'X5', 5: 'X6'}
    
    # Find which relation this belongs to
    cumulative_index = 0
    for relation_name, permutations in relation_variable_permutations.items():
        if cumulative_index <= index < cumulative_index + len(permutations):
            # Found the relation
            local_index = index - cumulative_index
            var_pair = permutations[local_index]
            
            # Build symbolic predicate
            symbolic = f"{relation_name}({var_names[var_pair[0]]}, {var_names[var_pair[1]]})"
            return symbolic
        
        cumulative_index += len(permutations)
    
    return "Unknown"


def evaluate_single_rule(rule_weights, X_data, y_data, variable_matrix, device):
    """
    Create a single-rule ANDRE model and evaluate its accuracy.
    
    Args:
        rule_weights: List of weights for all neurons in the rule
        X_data: Input features
        y_data: True labels
        variable_matrix: Variable matrix for structured data
        device: Device to use for computation
        
    Returns:
        Dictionary with accuracy and other metrics
    """
    # Find x_data columns as input_dim
    input_dim = X_data.shape[1]

    # Create single-rule ANDRE model
    single_rule_model = ANDRE(num_rules=1, pot_body_pred_dim=input_dim, 
                              variable_matrix=variable_matrix, lr=0.005, weight_decay=1e-4)
    single_rule_model = single_rule_model.to(device)
    
    # Load the weights for this single rule
    # In ANDRE_Net2, the structure is: rules[0] is a RuleBlock, which contains activation (ProbSumActivation)
    # The raw_weights are in the activation layer
    with torch.no_grad():
        # Stack all neuron weights into a single tensor [input_dim, 3]
        weights_array = torch.tensor(rule_weights, dtype=torch.float32, device=device)  # [input_dim, 3]
        # Use log to approximate inverse softmax
        single_rule_model.rules[0].activation.raw_weights.data = torch.log(weights_array + 1e-10)
    
    # Evaluate the single rule
    single_rule_model.eval()
    metrics = single_rule_model.evaluate(X_data, y_data)
    
    return metrics


def main():
    """Main training and evaluation function."""
    args = parse_args()
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_hour = datetime.now().strftime("%H-%M-%S")

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
    
    Num_rules = args.num_rules
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
            head_predicate = None
        else:
            body_predicates = ["father(X1, X2)", "father(X2, X3)", "father(X1, X3)", 
                             "mother(X1, X2)", "mother(X2, X3)", "mother(X1, X3)"]
            head_predicate = "grandparent(X1, X3)"
            variable_matrix, variables = create_V_matrix(body_predicates)
            head_vars = [0, 2]
            connected_vars = [1]
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

        if check_binary_dataset(X_train):
            # perturb data by adding small noise to zero and reducing one to less than one
            X_train = X_train - 0.1 * torch.rand_like(X_train) * (X_train - 0.5).sign()
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
            print("X_train is a binary dataset. We are perturbing the data for a better training.")
            # perturb data by adding small noise to zero and reducing one to less than one
            X_train = X_train - 0.1 * torch.rand_like(X_train) * (X_train - 0.5).sign()
            # clip to [0, 1]
            X_train = torch.clamp(X_train, 0.0, 1.0)

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
        head_predicate = "blockpositionindex(X1, X2)" #body_predicates[252]  # Example head predicate
        for idx in range(len(predicate_list)):
            if body_predicates[idx] == head_predicate:
                headpred_idx = idx
                print(f"Found head predicate at index {idx} in body predicates.")
                break

        # remove the head predicate from body predicates
        body_predicates.remove(head_predicate)
        print("Number of body predicates:", len(body_predicates))

        # remove the corresponding column from X_train
        X_train = torch.cat((X_train[:, :headpred_idx], X_train[:, headpred_idx+1:]), dim=1)

        # check if the length of body_predicates matches the number of columns in X_train
        assert len(body_predicates) == X_train.shape[1], "Mismatch between body predicates and X_train columns."
        variable_matrix, variables = create_V_matrix(body_predicates)
        head_vars = [0, 1]
        connected_vars = [2, 3]

        input_dim = X_train.shape[1]
        num_samples = X_train.shape[0]
        acc_threshold = 0.95

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
            head_vars = [0, 1]
            connected_vars = [2]
            # Update X_train (keep columns corresponding to indices_keep)
            X_train = X_train[:, indices_keep].clone()
            # Update input_dim
            input_dim = X_train.shape[1]
            # Summary
            print(f"Removed {len(indices_remove)} predicates containing 'X4'.")
            print(f"Old input_dim -> New input_dim: {input_dim}")

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
    best_losses = []
    
    # Store all discovered rules with their accuracies
    all_discovered_rules = []

    for num_rules in range(Num_rules):
        for restart in range(num_restarts):
            print(f"\n=== Training ANDRE with {num_rules + 1} rules, Restart {restart + 1}/{num_restarts} ===")
            model = ANDRE(num_rules + 1, pot_body_pred_dim=input_dim, 
                         variable_matrix=variable_matrix if structured_data else None, 
                         lr=args.lr, weight_decay=args.weight_decay)
            model = model.to(device)

            losses_andre = []

            for epoch in range(num_epochs):
                model.train()

                # Schedule loss weights
                ent_weight_, rr_weight_, conn_weight_, digit_weight_, sim_weight_ = schedule_loss_weights(
                            epoch=epoch, total_epochs=num_epochs,
                            final_ent_weight=ent_weight, 
                            final_rr_weight=rr_weight, 
                            final_conn_weight=conn_weight,
                            final_digit_weight=digit_weight,
                            initial_sim_weight=2.0,
                            final_sim_weight=sim_weight
                            )

                # Shuffle dataset
                perm = torch.randperm(num_samples, device=device)
                X_train_shuffled = X_train[perm]
                y_train_shuffled = y_train[perm]

                epoch_loss = 0.0

                # Mini-batch training loop
                for i in range(0, num_samples, batch_size):
                    xb = X_train_shuffled[i:i + batch_size]
                    yb = y_train_shuffled[i:i + batch_size]

                    model.optimizer.zero_grad()
                    outputs, model_weights, weight_entropies, Ms = model(xb)

                    # Binary Cross-Entropy Loss
                    bce_loss = model.criterion(outputs, yb)

                    # Similarity Penalty
                    sim_loss = 0.0
                    num_sim_pairs = 0
                    for i in range(num_rules + 1):
                        wi = model_weights[i].view(input_dim, -1)
                        for j in range(i + 1, num_rules + 1):
                            wj = model_weights[j].view(input_dim, -1)
                            sim_loss += nn.CosineSimilarity(dim=1)(wi, wj).mean()
                            num_sim_pairs += 1
                    sim_loss = sim_loss / num_sim_pairs if num_sim_pairs > 0 else 0

                    if structured_data:
                        # Build tensors from M entries (handle lists or tensors)
                        M_head_list = [torch.as_tensor(Ms[i][head_vars], device=xb.device, dtype=xb.dtype) for i in range(num_rules + 1)]
                        M_head = torch.stack(M_head_list)

                        # Range Restriction Loss with asymmetric penalty
                        eta = 0.5  # Small penalty for values above 1
                        mask_below = (M_head < 1.0).float()
                        mask_above = (M_head >= 1.0).float()
                        rr_loss_1 = mask_below * (M_head - 1.0)**2 + mask_above * eta * (M_head - 1.0)
                        rr_loss = rr_loss_1
                        rr_loss = rr_loss.sum() / len(head_vars) / (num_rules + 1)

                        # Connectivity Loss
                        M_connected_list = [torch.as_tensor(Ms[i][connected_vars], device=xb.device, dtype=xb.dtype) for i in range(num_rules + 1)]
                        M_connected = torch.stack(M_connected_list)  # tensor form

                        # Connectivity Loss with conditional masking
                        mask_below_2 = (M_connected < 2.0).float()
                        mask_above_2 = (M_connected >= 2.0).float()
                        
                        # Smoothness / connectivity term computed elementwise and averaged
                        conn_loss_1 = torch.stack([torch.exp(-0.5 * ((M_conn - 1.0) / 0.2)**2) / (0.2 * math.sqrt(2 * math.pi)) for M_conn in M_connected]).sum()
                        conn_loss_1 = 0.5 * conn_loss_1
                        
                        # MSE against constant 2
                        conn_loss_2 = (M_connected - 2.0)**2
                        
                        # Apply masking: use conn_loss_1 if below 2, conn_loss_2 if above or equal to 2
                        conn_loss = mask_below_2 * conn_loss_1 + mask_above_2 * conn_loss_2
                        conn_loss = conn_loss.sum() / len(connected_vars) / (num_rules + 1)

                        # Digitization Loss - encourage M elements to be integers
                        M_all_list = [torch.as_tensor(Ms[i], device=xb.device, dtype=xb.dtype) for i in range(num_rules + 1)]
                        M_all = torch.stack(M_all_list)  # [num_rules+1, num_variables]
                        digit_loss = 0.5 * (1 - torch.sin(2 * math.pi * M_all + math.pi / 2)).sum()
                        digit_loss = digit_loss / M_all.numel()

                        # Entropy Loss
                        entropy_term = torch.stack(weight_entropies).sum()
                        entropy_term = entropy_term / (num_rules + 1) /  input_dim

                        # Total Loss
                        total_loss = bce_loss + ent_weight_ * entropy_term + \
                                    rr_weight_ * rr_loss + conn_weight_ * conn_loss + \
                                    digit_weight_ * digit_loss + sim_weight_ * sim_loss

                        # print(f"Epoch {epoch+1}, Batch {i//batch_size+1}: BCE={bce_loss.item():.4f}, \
                        #       RR={rr_weight_ * rr_loss.item():.4f}, Conn={conn_weight_ * conn_loss.item():.4f}, Digit={digit_weight_ * digit_loss.item():.4f}, \
                        #       Ent={ent_weight_ * entropy_term.item():.4f}, Sim={sim_weight_ * float(sim_loss):.4f}")

                    else: 
                        entropy_term = torch.stack(weight_entropies).sum()
                        entropy_term = entropy_term / (num_rules + 1) /  input_dim
                        total_loss = bce_loss + ent_weight_ * entropy_term + sim_weight_ * sim_loss

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    model.optimizer.step()

                    epoch_loss += total_loss.item()

                model.scheduler.step()
                losses_andre.append(epoch_loss)

            # Extract rules from this trained model
            Weights_andre = []
            for rule_weights in model_weights:
                rule_weights = rule_weights.detach().cpu().numpy()
                body_predicate_weights = rule_weights.reshape(-1, 3)
                Weights_andre.append(body_predicate_weights.tolist())

            # Rule interpretation
            final_rule = extract_rules(Weights_andre=Weights_andre, X_train=X_train, 
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
            for rule_idx, rule_weights in enumerate(Weights_andre):
                # Extract individual rule text
                if dataset_type == "binary":
                    preds = interpret_neuron_weights(body_preds=body_predicates, Weights=rule_weights)
                else:
                    sample_pred_values = X_train[0].cpu().numpy()
                    preds = interpret_neuron_weights_entropy(sample_pred_values, rule_weights)
                
                refined_preds = [pred for pred in preds if pred != '1' and pred != '?']
                subrule_text = " and ".join(refined_preds) if refined_preds else "true"
                full_subrule = f"{head_predicate} :- {subrule_text}."
                
                # Evaluate this individual subrule
                subrule_metrics = evaluate_single_rule(
                    rule_weights,
                    X_train,
                    y_train,
                    variable_matrix if structured_data else None,
                    device
                )
                subrule_eval_metrics = evaluate_single_rule(
                    rule_weights,
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
                        'num_rules': num_rules + 1,
                        'restart': restart + 1,
                        'subrule_idx': rule_idx + 1,
                        'training_precision': subrule_metrics['precision'],
                        'training_recall': subrule_metrics['recall'],
                        'training_f1': subrule_metrics['f1_score'],
                        'eval_precision': subrule_eval_metrics['precision'],
                        'eval_recall': subrule_eval_metrics['recall'],
                        'eval_f1': subrule_eval_metrics['f1_score']
                    })

            print(f"Num Rules: {num_rules + 1}, Restart {restart+1}/{num_restarts}, Train Metric: {train_metric:.4f}, Evaluation Metric: {eval_metric:.4f}.")
            
            # Print current discovered subrules count
            print(f"Total unique subrules discovered so far: {len(all_discovered_rules)}")
            
            if min(losses_andre) < best_loss:
                print(f"**** New Best Model ****")
                best_eval_metric = eval_metric
                best_train_metric = train_metric
                best_overall_metric = overall_metric
                best_state_dict = deepcopy(model.state_dict())
                best_losses = losses_andre
                best_num_rules = num_rules + 1
                best_loss = min(losses_andre)

                if train_metric > acc_threshold and eval_metric > acc_threshold:
                    print(f"The accuracy threshold of {acc_threshold} has been reached. Stopping further restarts.")
                    break

    # Load the best model
    best_model = ANDRE(best_num_rules, pot_body_pred_dim=input_dim, 
                      variable_matrix=variable_matrix if structured_data else None, 
                      lr=0.005, weight_decay=1e-4)
    best_model.load_state_dict(best_state_dict)
    best_model = best_model.to(device)

    # Evaluate best model
    print("\n=== Best Model Training Evaluation ===")
    best_model.print_evaluation(X_train, y_train)

    print("\n=== Best Model Evaluation on Validation Set ===")
    best_model.print_evaluation(X_val, y_val)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), best_losses, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - ({best_num_rules} Rules)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_dir}/loss_curve_{today_date}_{today_hour}.png")
    plt.show()

    _, andre_weights, _, _ = best_model(X_val)

    Weights_andre = []
    for rule_weights in andre_weights:
        rule_weights = rule_weights.detach().cpu().numpy()
        body_predicate_weights = rule_weights.reshape(-1, 3)
        Weights_andre.append(body_predicate_weights.tolist())

    if visualize_weights:
        plot_andre_weights(Weights_andre, input_dim=input_dim, predicates=body_predicates, structured_data=structured_data)

    # Rule interpretation
    final_rule = extract_rules(Weights_andre=Weights_andre, X_train=X_train, 
                               dataset_type=dataset_type, body_predicates=body_predicates,
                               head_predicate=head_predicate)
    print("\n======================================")
    print("Final Rule:")
    print(final_rule)
    print("======================================")

    # Save the best model
    if save_model:
        best_model.save_model(save_dir=f"{args.save_dir}/andre_best_model_{today_date}", 
                             model_name=f"ANDRE_best_model_{today_hour}")

    # Evaluate individual rules and sort by accuracy
    if args.evaluate_individual_rules:
        print("\n" + "="*70)
        print("INDIVIDUAL RULE EVALUATION")
        print("="*70)

        rule_evaluations = []

        for rule_idx, rule_weights in enumerate(Weights_andre):
            print(f"\nEvaluating Rule {rule_idx + 1}...")
            
            # Evaluate this rule
            metrics = evaluate_single_rule(
                rule_weights, 
                X_train, 
                y_train, 
                variable_matrix if structured_data else None,
                device
            )
            
            # Extract the rule text
            if dataset_type == "binary":
                preds = interpret_neuron_weights(body_preds=body_predicates, Weights=rule_weights)
            else:
                # For probabilistic, use a representative sample
                sample_pred_values = X_train[0].cpu().numpy()
                preds = interpret_neuron_weights_entropy(sample_pred_values, rule_weights)
            
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
                'weights': rule_weights
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
    
    for idx, rule_info in enumerate(sorted_all_rules, 1):
        print(f"\n[{idx}] {rule_info['rule']}")
        print(f"    Training Accuracy: {rule_info['training_accuracy']:.4f} | Eval Accuracy: {rule_info['eval_accuracy']:.4f}")
        print(f"    Training F1: {rule_info['training_f1']:.4f} | Eval F1: {rule_info['eval_f1']:.4f}")
        print(f"    (Found in: Num_rules={rule_info['num_rules']}, Restart={rule_info['restart']}, Subrule #{rule_info['subrule_idx']})")
    
    print("\n" + "="*80)
    
    # Return the list for programmatic access
    return all_discovered_rules


if __name__ == '__main__':
    all_rules = main()
    
    # Optional: Save to file
    if all_rules:
        import json
        with open('discovered_rules.json', 'w') as f:
            json.dump(all_rules, f, indent=2)
        print(f"\nSaved {len(all_rules)} discovered rules to 'discovered_rules.json'")
