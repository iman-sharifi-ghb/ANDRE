import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import matplotlib.pyplot as plt
from collections import Counter
from ANDRE_Net2 import ANDRE
import argparse
import os
import sys
import random

def parse_rule_and_generate_data(rule_str, num_samples=1000, sample_type="binary", seed=42):
    """
    Parse a logical rule and generate random training data X_train and y_train.
    
    Args:
        rule_str (str): Logical rule in the form "h :- b1 and not(b2) or not(b3) and b4."
        num_samples (int): Number of training samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        X_train (torch.Tensor): Training features, shape (num_samples, num_predicates)
        y_train (torch.Tensor): Training labels, shape (num_samples,)
        body_predicates (list): List of predicate names ["b1", "b2", ...]
        head_predicate (str): Head predicate name
    
    Example:
        rule = "h :- b1 and not(b2) or not(b3) and b4."
        X_train, y_train, body_preds, head = parse_rule_and_generate_data(rule, num_samples=1000)
    """
    torch.manual_seed(seed)
    
    # Parse the rule to extract head and body
    if ":-" not in rule_str:
        raise ValueError("Rule must contain ':-' separator")
    
    head_part, body_part = rule_str.split(":-")
    head_predicate = head_part.strip()
    body_part = body_part.strip().rstrip(".")
    
    # Extract all predicate names (handle both b1 and not(b1))
    all_predicates = set()
    # Find predicates with not()
    not_preds = re.findall(r'not\((\w+)\)', body_part)
    all_predicates.update(not_preds)
    # Find standalone predicates (not inside not())
    standalone_preds = re.findall(r'(?<!not\()\b(b\d+)\b', body_part)
    all_predicates.update(standalone_preds)
    
    # Sort predicates by number for consistent ordering
    body_predicates = sorted(all_predicates, key=lambda x: int(x[1:]))
    num_predicates = len(body_predicates)
    pred_to_idx = {pred: i for i, pred in enumerate(body_predicates)}
    
    # Generate random binary/continuous data
    if sample_type == "binary":
        X_train = (torch.rand(num_samples, num_predicates) > 0.5).float()
        X_train = X_train - 0.1 * torch.rand_like(X_train) * (X_train - 0.5).sign()
        X_train = X_train.clamp(0, 1)
    elif sample_type == "continuous":
        X_train = torch.rand(num_samples, num_predicates)
    else:
        raise ValueError("sample_type must be 'binary' or 'continuous'")
    
    # Parse the logical expression to create y_train
    # Split by 'or' to get disjunctive clauses
    or_clauses = [clause.strip() for clause in body_part.split(" or ")]
    
    # Build the label tensor
    y_train_parts = []
    
    for clause in or_clauses:
        # Split by 'and' to get conjunctive literals
        and_literals = [lit.strip() for lit in clause.split(" and ")]
        
        # Build conjunction for this clause
        clause_conditions = []
        for literal in and_literals:
            if literal.startswith("not(") and literal.endswith(")"):
                # Negated predicate
                pred_name = literal[4:-1]  # Extract "b1" from "not(b1)"
                idx = pred_to_idx[pred_name]
                clause_conditions.append((1 - X_train[:, idx]) > 0.5)
            else:
                # Positive predicate
                pred_name = literal
                idx = pred_to_idx[pred_name]
                clause_conditions.append(X_train[:, idx] > 0.5)
        
        # Combine with AND (all conditions must be true)
        if clause_conditions:
            clause_result = clause_conditions[0]
            for cond in clause_conditions[1:]:
                clause_result = clause_result & cond
            y_train_parts.append(clause_result)
    
    # Combine all clauses with OR (at least one clause must be true)
    if y_train_parts:
        y_train = y_train_parts[0]
        for part in y_train_parts[1:]:
            y_train = y_train | part
    else:
        y_train = torch.zeros(num_samples, dtype=torch.bool)
    
    y_train = y_train.float()
    
    return X_train, y_train, body_predicates, head_predicate


def interpret_neuron_weights_entropy_(body_preds, pred_values, andre_probs, entropy_threshold=0.8):
    """
    Interpret the softmaxed probabilities of neurons and return a list of 'b_{i}', 'not(b_{i})', or '1'.

    Args:
        body_preds (list): A list of predicate strings for each neuron, like ['b1', 'b2', ...] or ['b1(X1, X2)', 'b2(X3)', ...]
        pred_value: The predicted value to be used in interpretation. it is a scalar value between 0 and 1.
        andre_probs (numpy.ndarray): A 2D numpy array where each row corresponds to the softmaxed probabilities of a neuron.

    Returns:
        list: A list of strings ('b_{i}', 'not(b_{i})', or '1') for each neuron.
    """
    
    interpretations = []
    for i, weights in enumerate(andre_probs):
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        entropy = -torch.sum(weights_tensor * torch.log(weights_tensor + 1e-10)).item()
        # print(f"Neuron {i+1} weights: {weights}, Entropy: {entropy}")

        if entropy > entropy_threshold:
            interpretations.append('?')
            continue

        # Find predicate value of a given neuron
        pred_value = pred_values[i]
        body_pred = body_preds[i]

        # compute {pred_value*weights[0], (1-pred_value)*weights[1], weights[2]}
        new_weights = [pred_value * weights[0], (1 - pred_value) * weights[1], weights[2]]

        max_index = np.argmax(new_weights)
        if max_index == 0:
            interpretations.append(f'{body_pred}')
            continue
        elif max_index == 1:
            interpretations.append(f'not({body_pred})')
            continue
        else:
            interpretations.append('1')

    return interpretations

def interpret_neuron_weights_entropy(pred_values, andre_probs, entropy_threshold=0.8):
    """
    Interpret the softmaxed probabilities of neurons and return a list of 'b_{i}', 'not(b_{i})', or '1',
    only if entropy is below a threshold to ensure confidence in prediction.

    Args:
        pred_values (numpy.ndarray): A 1D array with predicate values (between 0 and 1).
        andre_probs (list of list): A 2D list where each row is the softmaxed probabilities of a neuron.
        entropy_threshold (float): The threshold of entropy above which output is considered uncertain.

    Returns:
        list: A list of strings ('b_{i}', 'not(b_{i})', or '1') for each neuron, or '?' for uncertain.
    """
    interpretations = []
    for i, weights in enumerate(andre_probs):
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        entropy = -torch.sum(weights_tensor * torch.log(weights_tensor + 1e-10)).item()
        # print(f"Neuron {i+1} weights: {weights}, Entropy: {entropy}")

        if entropy > entropy_threshold:
            interpretations.append('?')
            continue

        pred_value = pred_values[i]
        new_weights = [pred_value * weights[0], (1 - pred_value) * weights[1], weights[2]]
        max_index = np.argmax(new_weights)

        if max_index == 0:
            interpretations.append(f'b{i+1}')
        elif max_index == 1:
            interpretations.append(f'not(b{i+1})')
        else:
            interpretations.append('1')

    return interpretations

def interpret_neuron_weights_(pred_values, andre_probs):
    """
    Interpret the softmaxed probabilities of neurons and return a list of 'b_{i}', 'not(b_{i})', or '1'.

    Args:
        pred_value: The predicted value to be used in interpretation. it is a scalar value between 0 and 1.
        andre_probs (numpy.ndarray): A 2D numpy array where each row corresponds to the softmaxed probabilities of a neuron.

    Returns:
        list: A list of strings ('b_{i}', 'not(b_{i})', or '1') for each neuron.
    """
    

    interpretations = []
    for i, weights in enumerate(andre_probs):

        # Find predicate value of a given neuron
        pred_value = pred_values[i]

        # compute {pred_value*weights[0], (1-pred_value)*weights[1], weights[2]}
        new_weights = [pred_value * weights[0], (1 - pred_value) * weights[1], weights[2]]

        max_index = np.argmax(new_weights)
        if max_index == 0:
            interpretations.append(f'b{i+1}')
            continue
        elif max_index == 1:
            interpretations.append(f'not(b{i+1})')
            continue
        else:
            interpretations.append('1')

    return interpretations

def interpret_neuron_weights(body_preds=None, andre_probs=None):
    """
    Interpret the softmaxed probabilities of neurons and return a list of 'b_{i}', 'not(b_{i})', or '1'.

    Args:
        andre_probs (numpy.ndarray): A 2D numpy array where each row corresponds to the softmaxed probabilities of a neuron.

    Returns:
        list: A list of strings ('b_{i}', 'not(b_{i})', or '1') for each neuron.
    """
    
    if body_preds is None:
        interpretations = []
        for i, weights in enumerate(andre_probs):

            # compute {weights[0], weights[1], weights[2]}
            new_weights = [weights[0], weights[1], weights[2]]

            max_index = np.argmax(new_weights)
            if max_index == 0:
                interpretations.append(f'b{i+1}')
                continue
            elif max_index == 1:
                interpretations.append(f'not(b{i+1})')
                continue
            else:
                interpretations.append('1')
        return interpretations
    
    else:
        interpretations = []
        for i, weights in enumerate(andre_probs):

            body_pred = body_preds[i]
            # compute {weights[0], weights[1], weights[2]}
            new_weights = [weights[0], weights[1], weights[2]]

            max_index = np.argmax(new_weights)
            if max_index == 0:
                interpretations.append(f'{body_pred}')
                continue
            elif max_index == 1:
                interpretations.append(f'not({body_pred})')
                continue
            else:
                interpretations.append('1')
        return interpretations

def create_V_matrix(predicates):
    # Extract all variable names
    variable_set = set()
    predicate_vars = []
    for pred in predicates:
        # find if '(' exists in the predicate
        if '(' not in pred:
            vars_list = []
            predicate_vars.append(vars_list)
            continue
        # Extract variables inside parentheses
        vars_in_pred = re.findall(r'\((.*?)\)', pred)
        if vars_in_pred:
            vars_list = [v.strip() for v in vars_in_pred[0].split(',')]
        else:
            vars_list = []
        predicate_vars.append(vars_list)
        variable_set.update(vars_list)
    # Sort variables for consistent column order
    variables = sorted(variable_set, key=lambda x: (int(x[1:]) if x[1:].isdigit() else x))
    var_idx = {v: i for i, v in enumerate(variables)}
    # Build the matrix
    V = torch.zeros((len(predicates), len(variables)), dtype=torch.float32)
    for i, vars_list in enumerate(predicate_vars):
        for v in vars_list:
            V[i, var_idx[v]] = 1.0
    return V, variables

def compute_acc_ratio(model, x, y, rule_idx):
    with torch.no_grad():
        rule = model.rules[rule_idx]
        body_conj = sum(rule[i](x[:, [i]]) for i in range(model.num_neurons_per_rule))
        rule_weights = []
        for neuron in rule:
            weights = torch.softmax(neuron.raw_weights, dim=0).detach().numpy().tolist()
            rule_weights.append(weights)

        num_body_preds = x.shape[1] - sum([1 for w in rule_weights if np.round(w[2]) == 1.0])
        threshold = 0.5**(num_body_preds)
        body = (torch.exp(body_conj) >= threshold).float()

        # Match the shape of body with y
        if body.shape != y.shape:
            body = body.squeeze(-1)

        # find cosine similarity between body and y
        cos_sim = nn.CosineSimilarity(dim=0)(body, y)

        return cos_sim.item()

def compute_acc_ratio_all_rules(model, x, y):
    with torch.no_grad():
        for rule_idx, rule in enumerate(model.rules):
            pred_values = [rule[i](x[:, [i]]) for i in range(model.num_neurons_per_rule)]
            rule_weights = []
            for neuron in rule:
                weights = torch.softmax(neuron.raw_weights, dim=0).detach().numpy().tolist()
                rule_weights.append(weights)

            # This should be double checked because it may not be accurate
            num_body_preds = x.shape[1] - sum([1 for w in rule_weights if np.argmax(w) == 2 and np.round(w[2]) == 1.0]) # x.shape[1] is the number of potential body predicates
            threshold = 0.5**(num_body_preds)
            body = (torch.exp(body_conj) >= threshold).float()

            if rule_idx == 0:
                fuzzy_or = body
            else:
                fuzzy_or = fuzzy_or + body - fuzzy_or * body

        # Match the shape of body with y
        if fuzzy_or.shape != y.shape:
            fuzzy_or = fuzzy_or.squeeze(-1)

        # find cosine similarity between body and y
        cos_sim = nn.CosineSimilarity(dim=0)(fuzzy_or, y)

        return cos_sim.item()

def compute_acc_ratio_all_rules_attention(model, x, y, beta=20.0, lambda_ = 10.0):
    with torch.no_grad():
        rule_outputs = []
        for rule in model.rules:
            conjunctions = []
            for i, neuron in enumerate(rule):
                xi = x[:, [i % model.input_dim]]
                conjunctions.append(neuron(xi).squeeze())
            # Softmin attention over neurons (optional, or just use sum)
            conjunctions = torch.stack(conjunctions, dim=1)  # (batch, num_neurons)
            attention_weights = torch.softmax(-beta * conjunctions, dim=1)
            outputs = (attention_weights * conjunctions).sum(dim=1)  # (batch,)
            activated_outputs = torch.sigmoid(lambda_ * (outputs - 0.5))
            rule_outputs.append(activated_outputs)

        # Stack rule outputs: shape (num_rules, batch)
        rule_outputs_tensor = torch.stack(rule_outputs, dim=0)  # (num_rules, batch)
        # Softmax attention over rules for each sample (dim=0)
        attention_weights = torch.softmax(beta * rule_outputs_tensor, dim=0)
        # Weighted sum over rules for each sample
        final_output = (attention_weights * rule_outputs_tensor).sum(dim=0)  # (batch,)
        # Compute accuracy
        preds = (final_output > 0.5).float()
        acc = (preds == (y > 0.5)).float().mean().item()
        return acc

# compute_acc_ratio(best_model, X_train, y_train, 0)

def check_binary_dataset(x_train):
    for i in range(x_train.shape[1]):
        unique_values = torch.unique(x_train[:, i])
        if not torch.all((unique_values == 0) | (unique_values == 1)):
            return False
    return True

def plot_andre_weights(andre_probs, input_dim, predicates=None, structured_data=False):    

    # Grid of subplots: one subplot per (rule, body-predicate/neuron).
    # Each subplot shows the 3 probability components for that neuron as a small bar-chart.
    # (A true histogram isn't meaningful for just 3 values, so we use bars.)
    body_predicates = predicates if structured_data else [f"b{i+1}" for i in range(input_dim)]
    w_arr = np.array(andre_probs, dtype=float)
    if w_arr.ndim != 3 or w_arr.shape[2] != 3:
        raise ValueError(f"Expected Weights_nsre shape (R, N, 3), got {w_arr.shape}")

    R, N, C = w_arr.shape
    # prefer notebook variables if present
    R_use = Num_rules if 'Num_rules' in globals() else R
    N_use = input_dim if 'input_dim' in globals() else N
    if (R_use, N_use) != (R, N):
        # fall back to array shape but warn via title
        R_use, N_use = R, N

    fig, axes = plt.subplots(R_use, N_use, figsize=(N_use * 1.6, R_use * 1.6), constrained_layout=True)
    # Ensure axes is 2D
    if R_use == 1 and N_use == 1:
        axes = np.array([[axes]])
    elif R_use == 1:
        axes = axes[np.newaxis, :]
    elif N_use == 1:
        axes = axes[:, np.newaxis]

    for r in range(R_use):
        for n in range(N_use):
            ax = axes[r, n]
            vals = w_arr[r, n, :]
            ax.bar([0, 1, 2], vals, color=['green', 'red', 'silver'], alpha=0.8)
            ax.set_ylim(0.0, 1.2)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['w1', 'w2', 'w3'], fontsize=6)
            # keep titles small to avoid overlap
            # set title only for the first row
            if r == 0:
                ax.set_title(f"{body_predicates[n]}", fontsize=7)

            # Only the left-most subplot in each row shows the y-axis label and ticks
            if n == 0:
                ax.set_ylabel(f"R{r+1}", fontsize=8)
                # show y-ticks at 0.0, 0.5, 1.0
                ax.set_yticks([0.0, 0.5, 1.0])
                # ax.tick_params(axis='y', labelsize=6)
                
            else:
                ax.tick_params(labelleft=False)  # hide y labels for other columns
                # reduce clutter on very dense grids
                if N_use > 8:
                    ax.set_yticklabels([])

            if r < R_use - 1:
                # only bottom row shows x-labels clearly; remove x-labels for non-bottom rows
                ax.set_xticklabels([])

    suptitle = f"ANDRE Weights Visualization ({R_use} Rules x {N_use} Predicates)"
    fig.suptitle(suptitle, fontsize=10)
    plt.show()

def extract_rules(andre_probs, X_train, dataset_type, body_predicates, head_predicate):
    # Rule interpretation
    Predicate_values = X_train
    if dataset_type == "probabilistic":
        Final_rules = []
        for rule_num, weights in enumerate(andre_probs):
            Extracted_rules = []
            for i in range(len(Predicate_values)):
                pred_values = Predicate_values[i].numpy()
                preds = interpret_neuron_weights_entropy(pred_values, weights)
                refined_preds = [pred for pred in preds if pred != '1']
                rule = f"{head_predicate} :- " + " and ".join(refined_preds) + "."
                Extracted_rules.append(rule)

            rule_counts = Counter(Extracted_rules)
            most_common_rule, count = rule_counts.most_common(1)[0]
            print(f"The most repeated rule is: '{most_common_rule}'")
            print(f"Confidence over training data: {float(count/len(Extracted_rules))}.")

            Final_rules.append(most_common_rule)
    else:
        Final_rules = []
        for rule_num, weights in enumerate(andre_probs):
            preds = interpret_neuron_weights(body_preds=body_predicates, andre_probs=weights)
            refined_preds = [pred for pred in preds if pred != '1']
            rule = f"{head_predicate} :- " + " and ".join(refined_preds) + "."
            print(f"Extracted Rule: {rule}")
            Final_rules.append(rule)

    # remove the "h{i+1} :- " part from each rule
    Final_rules = [rule.replace(f"{head_predicate} :- ", "") for rule_num, rule in enumerate(Final_rules)]

    # remove the "."
    Final_rules = [rule.replace(".", "") for rule in Final_rules]

    # convert the rules to a single rule
    final_rule = f"{head_predicate} :- " + " OR ".join(Final_rules) + "."

    return final_rule


def compute_mrr(scores, labels, rule_confidences):
    """
    Compute Mean Reciprocal Rank (MRR) for rule ranking.
    
    Rules are ranked globally by their confidence scores (e.g., validation accuracy).
    For each sample, we find the rank of the first rule that correctly predicts it.
    
    Args:
        scores: numpy array of shape (num_samples, num_rules) with rule predictions
        labels: numpy array of shape (num_samples,) with binary labels
        rule_confidences: numpy array of shape (num_rules,) with rule confidence scores
        
    Returns:
        mrr: Mean Reciprocal Rank across all samples
        reciprocal_ranks: array of reciprocal ranks for each sample
    """
    num_samples, num_rules = scores.shape
    reciprocal_ranks = []
    
    # Rank rules globally by confidence (descending order)
    ranked_rule_indices = np.argsort(-rule_confidences)
    
    for i in range(num_samples):
        sample_scores = scores[i]
        label = labels[i]
        
        # Find the rank of the first rule that correctly predicts this sample
        rank = None
        for r, rule_idx in enumerate(ranked_rule_indices, start=1):
            # Check if this rule correctly predicts the sample
            prediction = sample_scores[rule_idx]
            # is_correct = (prediction > 0.5 and label == 1) or (prediction <= 0.5 and label == 0)
            is_correct = (prediction >= 0.5) == (label == 1)
            
            if is_correct:
                rank = r
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr, np.array(reciprocal_ranks)


def compute_hits_at_k(scores, labels, rule_confidences, k_values=[1, 3, 10]):
    """
    Compute Hits@K metric for rule ranking.
    
    Hits@K measures the proportion of samples where at least one correct rule
    appears in the top-K ranked rules (by confidence).
    
    Args:
        scores: numpy array of shape (num_samples, num_rules) with rule predictions
        labels: numpy array of shape (num_samples,) with binary labels
        rule_confidences: numpy array of shape (num_rules,) with rule confidence scores
        k_values: list of K values to compute Hits@K for
        
    Returns:
        hits_at_k: dictionary mapping K -> Hits@K score
    """
    num_samples, num_rules = scores.shape
    hits_at_k = {k: 0.0 for k in k_values}
    
    # Rank rules globally by confidence (descending order)
    ranked_rule_indices = np.argsort(-rule_confidences)
    
    for i in range(num_samples):
        sample_scores = scores[i]
        label = labels[i]
        
        for k in k_values:
            # Check if any of the top-K rules correctly predict this sample
            top_k_indices = ranked_rule_indices[:k]
            found_correct = False
            
            for rule_idx in top_k_indices:
                prediction = sample_scores[rule_idx]
                is_correct = (prediction > 0.5 and label == 1) or (prediction <= 0.5 and label == 0)
                
                if is_correct:
                    found_correct = True
                    break
            
            if found_correct:
                hits_at_k[k] += 1
    
    # Normalize by number of samples
    for k in k_values:
        hits_at_k[k] /= num_samples
    
    return hits_at_k


def compute_ranking_metrics(scores, labels, rule_confidences, k_values=[1, 3, 10]):
    """
    Compute both MRR and Hits@K metrics for rule ranking.
    
    Args:
        scores: numpy array of shape (num_samples, num_rules) with rule predictions
        labels: numpy array of shape (num_samples,) with binary labels
        rule_confidences: numpy array of shape (num_rules,) with rule confidence scores
        k_values: list of K values to compute Hits@K for
        
    Returns:
        metrics: dictionary containing 'mrr' and 'hits_at_k'
    """
    mrr, reciprocal_ranks = compute_mrr(scores, labels, rule_confidences)
    hits_at_k = compute_hits_at_k(scores, labels, rule_confidences, k_values)
    
    return {
        'mrr': mrr,
        'hits_at_k': hits_at_k,
        'reciprocal_ranks': reciprocal_ranks
    }


def print_ranking_metrics(metrics, dataset_name="Dataset"):
    """
    Pretty print ranking metrics.
    
    Args:
        metrics: dictionary from compute_ranking_metrics
        dataset_name: name of the dataset for display
    """
    print(f"\n{'='*60}")
    print(f"RANKING METRICS - {dataset_name}")
    print(f"{'='*60}")
    print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    print(f"{'-'*60}")
    print("Hits@K:")
    for k, score in sorted(metrics['hits_at_k'].items()):
        print(f"  Hits@{k:2d}: {score:.4f}")
    print(f"{'='*60}")


def extract_variables_from_predicate(predicate):
    """
    Extract variable indices from a predicate string.
    
    Args:
        predicate: string like "father(X1, X2)" or "grandparent(X1, X3)"
        
    Returns:
        list of variable indices (e.g., [0, 1] for X1, X2)
    """
    # Extract the part inside parentheses
    match = re.search(r'\((.*?)\)', predicate)
    if not match:
        return []
    
    args = match.group(1).split(',')
    variables = []
    for arg in args:
        arg = arg.strip()
        # Extract number from X1, X2, etc.
        var_match = re.search(r'X(\d+)', arg)
        if var_match:
            var_idx = int(var_match.group(1)) - 1  # Convert to 0-indexed
            variables.append(var_idx)
    
    return variables


def extract_head_and_connected_vars(head_predicate, all_variables):
    """
    Extract head_vars and connected_vars from head predicate and all variables.
    
    Args:
        head_predicate: string like "grandparent(X1, X3)"
        all_variables: list of all variable names like ['X1', 'X2', 'X3']
        
    Returns:
        head_vars: list of variable indices that appear in head predicate (e.g., [0, 2] for X1, X3)
        connected_vars: list of variable indices that don't appear in head (e.g., [1] for X2)
    """
    # Extract variables from head predicate
    head_vars = extract_variables_from_predicate(head_predicate)
    
    # All variable indices
    all_var_indices = list(range(len(all_variables)))
    
    # Connected vars are those not in head
    connected_vars = [i for i in all_var_indices if i not in head_vars]
    
    return head_vars, connected_vars


def count_variable_occurrences_in_rule_text(rule_text):
    """
    Count how many times each variable appears in a rule text.
    
    Args:
        rule_text: rule string like "grandparent(X1, X3) :- father(X1, X2) and mother(X2, X3)."
        
    Returns:
        dict mapping variable index to occurrence count
    """
    # Extract all variable references (X1, X2, etc.)
    var_matches = re.findall(r'X(\d+)', rule_text)
    var_counts = {}
    for var_str in var_matches:
        var_idx = int(var_str) - 1  # Convert to 0-indexed
        var_counts[var_idx] = var_counts.get(var_idx, 0) + 1
    
    return var_counts


def validate_rule_by_text(rule_text, head_vars, connected_vars):
    """
    Validate if a rule text satisfies the required format constraints:
    1. Head variables must appear at least once in body predicates (i.e., at least twice total)
    2. Connected (body) variables must appear at least twice in the rule
    
    Args:
        rule_text: rule string like "grandparent(X1, X3) :- father(X1, X2) and mother(X2, X3)."
        head_vars: list of head variable indices (e.g., [0, 2] for X1, X3 in grandparent(X1,X3))
        connected_vars: list of connected/body variable indices (e.g., [1] for X2)
        
    Returns:
        bool: True if rule is valid, False otherwise
        str: Reason for invalidity (or "Valid" if valid)
    """
    if not rule_text or ':-' not in rule_text:
        return False, "Invalid rule format (no ':-' separator)"
    
    # Split into head and body
    parts = rule_text.split(':-')
    if len(parts) != 2:
        return False, "Invalid rule format"
    
    head_part = parts[0].strip()
    body_part = parts[1].strip()
    
    # Count variables in the entire rule
    total_var_counts = count_variable_occurrences_in_rule_text(rule_text)
    
    # Count variables only in the body
    body_var_counts = count_variable_occurrences_in_rule_text(body_part)
    
    # Check constraint 1: Head variables must appear at least once in body
    # (which means at least twice total: once in head, once in body)
    for head_var in head_vars:
        body_count = body_var_counts.get(head_var, 0)
        if body_count < 1:
            return False, f"Head variable X{head_var+1} appears {body_count} times in body (need ≥1)"
    
    # Check constraint 2: Connected variables must appear at least twice
    for conn_var in connected_vars:
        total_count = total_var_counts.get(conn_var, 0)
        if total_count == 1:
            return False, f"Connected variable X{conn_var+1} appears {total_count} times (need != 1)"
    
    return True, "Valid"


def filter_valid_rules(all_rules, head_vars, connected_vars, verbose=True):
    """
    Filter rules to keep only those that satisfy format constraints based on rule text.
    
    Args:
        all_rules: list of rule dictionaries (each with 'rule' text and other fields)
        head_vars: list of head variable indices
        connected_vars: list of connected/body variable indices
        verbose: whether to print filtering statistics
        
    Returns:
        valid_rules: list of valid rule dictionaries
        invalid_rules: list of invalid rule dictionaries with 'invalid_reason' added
    """
    valid_rules = []
    invalid_rules = []
    
    for rule_info in all_rules:
        rule_text = rule_info.get('rule', '')
        
        # Validate the rule by text
        is_valid, reason = validate_rule_by_text(rule_text, head_vars, connected_vars)
        
        if is_valid:
            valid_rules.append(rule_info)
        else:
            rule_info_copy = rule_info.copy()
            rule_info_copy['invalid_reason'] = reason
            invalid_rules.append(rule_info_copy)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RULE VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Total rules: {len(all_rules)}")
        print(f"Valid rules: {len(valid_rules)}")
        print(f"Invalid rules: {len(invalid_rules)}")
        
        if invalid_rules:
            print(f"\n{'-'*70}")
            print("Invalid rules and reasons:")
            for i, rule_info in enumerate(invalid_rules, 1):
                print(f"\n{i}. {rule_info['rule']}")
                print(f"   Reason: {rule_info['invalid_reason']}")
        print(f"{'='*70}")
    
    return valid_rules, invalid_rules


def top_k_cosine_similar(X_pos, X_neg, k=100):
    """
    Compute top-k most similar negative samples for each positive sample using cosine similarity.
    
    Args:
        X_pos: Positive samples tensor [num_pos, dim]
        X_neg: Negative samples tensor [num_neg, dim]
        k: Number of top similar negatives to return
        
    Returns:
        topk_indices: Indices of top-k similar negatives for each positive [num_pos, k]
        topk_values: Cosine similarity values [num_pos, k]
    """
    dot_products = torch.mm(X_pos, X_neg.t())
    pos_norms = torch.norm(X_pos, p=2, dim=1, keepdim=True)
    neg_norms = torch.norm(X_neg, p=2, dim=1, keepdim=True).t()
    similarity_matrix = dot_products / (pos_norms * neg_norms + 1e-8)
    topk_values, topk_indices = torch.topk(similarity_matrix, k, dim=1)
    return topk_indices, topk_values


def mean_reciprocal_rank(ranks):
    """
    Compute Mean Reciprocal Rank (MRR) from a list of ranks.
    
    Args:
        ranks: List of ranks (1-based)
        
    Returns:
        mrr: Mean reciprocal rank
    """
    reciprocal_ranks = [1.0 / r for r in ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hits_at_k(ranks, k):
    """
    Compute Hits@K from a list of ranks.
    
    Args:
        ranks: List of ranks (1-based)
        k: K value for Hits@K
        
    Returns:
        hits: Hits@K score
    """
    hits = sum(1 for r in ranks if r <= k)
    return hits / len(ranks)


def compute_mrr_hits(X_val, y_val, model, k=100, r=100):
    """
    Compute MRR and Hits@K on validation set using corrupted samples.
    
    Args:
        X_val: Validation features
        y_val: Validation labels
        model: Trained model
        k: Number of negative samples to corrupt with
        
    Returns:
        mrr: Mean Reciprocal Rank
        hits: Dict with Hits@1, Hits@3, Hits@5, Hits@10
    """
    # Split validation set into positive and negative samples
    X_val_pos = X_val[y_val == 1]
    X_val_neg = X_val[y_val == 0]

    corrupted_indices, _ = top_k_cosine_similar(X_val_pos, X_val_neg, k=k)

    ranks = []
    for i in range(X_val_pos.shape[0]):
        pos_sample = X_val_pos[i].unsqueeze(0)
        neg_indices = corrupted_indices[i]
        neg_samples = X_val_neg[neg_indices]
        all_samples = torch.cat([pos_sample, neg_samples], dim=0)

        with torch.no_grad():
            scores, _, _, _ = model(all_samples)

        scores = torch.round(scores * r) / r

        new_scores = []
        seen = set()
        for score in scores:
            if score.item() not in seen:
                new_scores.append(score)
                seen.add(score.item())
        new_scores = torch.tensor(new_scores)

        sorted_indices = torch.argsort(new_scores, descending=True)
        rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    mrr = mean_reciprocal_rank(ranks)
    hits = {k_val: hits_at_k(ranks, k_val) for k_val in [1, 3, 5, 10]}

    return mrr, hits

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

class Tee:
    """Duplicate output to both console and log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


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

def evaluate_single_rule(raw_weights, X_data, y_data, variable_matrix, device):
    """
    Create a single-rule ANDRE model and evaluate its accuracy.
    
    Args:
        raw_weights: List of raw weights (logit space) for all neurons in the rule
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
    
    # Load the raw weights for this single rule
    # In ANDRE_Net2, the structure is: rules[0] is a RuleBlock, which contains activation (ProbSumActivation)
    # The raw_weights are in the activation layer (before softmax)
    with torch.no_grad():
        # Stack all neuron raw weights into a single tensor [input_dim, 3]
        raw_weights_array = torch.tensor(raw_weights, dtype=torch.float32, device=device)  # [input_dim, 3]
        # Directly assign raw weights (raw_weights are already in raw/logit space before softmax)
        single_rule_model.rules[0].activation.raw_weights.data = raw_weights_array
    
    # Evaluate the single rule
    single_rule_model.eval()
    metrics = single_rule_model.evaluate(X_data, y_data)
    
    return metrics