#!/usr/bin/env python3
"""
ANDRE Model Evaluation Script
This script provides utilities to load and evaluate trained ANDRE models.
"""

import argparse
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from ANDRE_Net2 import ANDRE
from utils import create_V_matrix, extract_head_and_connected_vars
from utils import compute_ranking_metrics, print_ranking_metrics


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
        description='ANDRE Model Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model loading options
    parser.add_argument('--model_type', type=str, default='best_model', 
                        choices=['best_model', 'unified_model'],
                        help='Type of model to load: best_model (trained model) or unified_model (from sorted rules)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the saved model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (for best_model type, e.g., "ANDRE_best_model_12-30-45")')
    parser.add_argument('--weights_file', type=str, default='sorted_rules_weights.pt',
                        help='Weights file name (for unified_model type)')
    
    # Data options
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--test_file_x', type=str, default='x_test.dt',
                        help='Test features file name')
    parser.add_argument('--test_file_y', type=str, default='y_test.dt',
                        help='Test labels file name')
    parser.add_argument('--structured_data', type=str2bool, default=True,
                        help='Use structured data with predicates')
    parser.add_argument('--head_predicate', type=str, default='grandparent(X1, X3)',
                        help='Head predicate for structured data')
    
    # Model architecture (needed for reconstruction)
    parser.add_argument('--num_rules', type=int, default=None,
                        help='Number of rules in the model (auto-detected if not provided)')
    parser.add_argument('--input_dim', type=int, default=None,
                        help='Input dimension (auto-detected from data if not provided)')
    
    # Evaluation options
    parser.add_argument('--compute_rule_scores', type=str2bool, default=True,
                        help='Compute per-rule scores')
    parser.add_argument('--compute_ranking_metrics', type=str2bool, default=True,
                        help='Compute MRR and Hits@K metrics')
    parser.add_argument('--rule_confidence_metric', type=str, default='val_accuracy',
                        choices=['val_accuracy', 'val_f1', 'train_accuracy', 'val_precision'],
                        help='Metric to use as rule confidence for ranking')
    parser.add_argument('--save_predictions', type=str2bool, default=True,
                        help='Save predictions to CSV file')
    
    # Device options
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for evaluation')
    
    return parser.parse_args()


def load_test_data(data_path, test_file_x, test_file_y, device):
    """Load test data from pickle files."""
    x_path = os.path.join(data_path, test_file_x)
    y_path = os.path.join(data_path, test_file_y)
    
    print(f"Loading test data from:")
    print(f"  X: {x_path}")
    print(f"  y: {y_path}")
    
    with open(x_path, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_test = pickle.load(f)
    
    # Handle different data formats
    if isinstance(x_test, list):
        x_test_np = np.concatenate(x_test, axis=1)
        X_test = torch.tensor(x_test_np[:, :-1] if x_test_np.shape[1] > len(y_test) else x_test_np, 
                             dtype=torch.float32)
    else:
        X_test = torch.tensor(x_test, dtype=torch.float32)
    
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    print(f"Test data loaded: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
    
    return X_test, y_test


def load_predicates_and_variable_matrix(data_path, head_predicate):
    """Load predicates and create variable matrix for structured data."""
    # Load predicate list
    with open(os.path.join(data_path, 'predicate.list'), 'rb') as f:
        predicate_list = pickle.load(f)
    
    # Load relation variable permutations
    with open(os.path.join(data_path, 'relation_variable.dt'), 'rb') as f:
        relation_variable_permutations = pickle.load(f)
    
    # Load relation names
    with open(os.path.join(data_path, 'all_relation_dic.dt'), 'rb') as f:
        all_relations = pickle.load(f)
    
    # Build body predicates (simplified - you may need to adapt based on your data)
    from main import get_predicate_name
    
    body_predicates = []
    for idx in range(len(predicate_list)):
        predicate_name = get_predicate_name(idx, predicate_list, 
                                            relation_variable_permutations, 
                                            all_relations)
        body_predicates.append(predicate_name)
    
    # Remove head predicate if it's in body predicates
    if head_predicate in body_predicates:
        body_predicates.remove(head_predicate)
    
    # Create variable matrix
    variable_matrix, variables = create_V_matrix(body_predicates)
    head_vars, connected_vars = extract_head_and_connected_vars(head_predicate, variables)
    
    return variable_matrix, variables, head_vars, connected_vars, body_predicates


def load_best_model(model_dir, model_name, num_rules, input_dim, variable_matrix, device):
    """Load a trained ANDRE model."""
    print(f"\nLoading best model from: {model_dir}")
    print(f"Model name: {model_name}")
    
    model = ANDRE(
        num_rules=num_rules,
        pot_body_pred_dim=input_dim,
        variable_matrix=variable_matrix,
        lr=0.005,
        weight_decay=1e-4
    )
    
    model.load_model(save_dir=model_dir, model_name=model_name)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {num_rules} rules")
    
    return model


def load_unified_model(weights_path, input_dim, variable_matrix, device):
    """Load a unified model from saved rule weights."""
    print(f"\nLoading unified model from: {weights_path}")
    
    model = ANDRE.build_from_rule_weights(
        rule_weights_path=weights_path,
        pot_body_pred_dim=input_dim,
        variable_matrix=variable_matrix
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Unified model loaded successfully with {model.num_rules} rules")
    
    return model


def evaluate_model(model, X_test, y_test, args):
    """Evaluate model and return metrics."""
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Basic evaluation
    model.print_evaluation(X_test, y_test)
    metrics = model.evaluate(X_test, y_test)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(X_test)
        binary_predictions = (predictions > 0.5).float()
    
    results = {
        'metrics': metrics,
        'predictions': predictions.cpu().numpy(),
        'binary_predictions': binary_predictions.cpu().numpy()
    }
    
    # Compute per-rule scores if requested
    if args.compute_rule_scores:
        print("\nComputing per-rule scores...")
        with torch.no_grad():
            rule_scores = model.get_rule_scores(X_test, apply_attention=False)
            scores_np = rule_scores.cpu().numpy().T  # [B, R]
            
            # Adjust scores based on labels
            y_test_np = y_test.cpu().numpy().reshape(-1, 1)
            adjusted_scores = scores_np * y_test_np + (1 - scores_np) * (1 - y_test_np)
            
            results['rule_scores'] = scores_np
            results['adjusted_scores'] = adjusted_scores
            
            print(f"Rule scores computed: shape = {scores_np.shape}")
    
    return results


def compute_and_save_ranking_metrics(results, args, model_dir):
    """Compute MRR and Hits@K metrics."""
    if not args.compute_ranking_metrics or 'adjusted_scores' not in results:
        return
    
    print("\n" + "="*70)
    print("COMPUTING RANKING METRICS")
    print("="*70)
    
    # Try to load metadata to get rule confidences
    metadata_path = os.path.join(model_dir, "sorted_rules_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            rules_metadata = json.load(f)
        
        # Extract rule confidences
        confidence_metric_map = {
            'val_accuracy': 'eval_accuracy',
            'val_f1': 'eval_f1',
            'train_accuracy': 'training_accuracy',
            'val_precision': 'eval_precision'
        }
        confidence_key = confidence_metric_map[args.rule_confidence_metric]
        rule_confidences = np.array([rule[confidence_key] for rule in rules_metadata])
        
        print(f"Using '{args.rule_confidence_metric}' as rule confidence metric")
        print(f"Rule confidences: {rule_confidences}")
        
        # Compute metrics
        y_test = results['metrics']['labels']
        adjusted_scores = results['adjusted_scores']
        
        ranking_metrics = compute_ranking_metrics(
            adjusted_scores, y_test, rule_confidences, k_values=[1, 3, 5, 10]
        )
        print_ranking_metrics(ranking_metrics, dataset_name="Test Set")
        
        # Save metrics
        ranking_results = {
            'confidence_metric': args.rule_confidence_metric,
            'rule_confidences': rule_confidences.tolist(),
            'test': {
                'mrr': float(ranking_metrics['mrr']),
                'hits_at_k': {int(k): float(v) for k, v in ranking_metrics['hits_at_k'].items()}
            }
        }
        
        output_path = os.path.join(args.data_path, 'results', 'test_ranking_metrics.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ranking_results, f, indent=2)
        
        print(f"\nSaved ranking metrics to: {output_path}")
    else:
        print(f"Warning: Metadata file not found at {metadata_path}")
        print("Skipping ranking metrics computation.")


def save_predictions(results, X_test, y_test, args):
    """Save predictions to CSV file."""
    if not args.save_predictions:
        return
    
    print("\nSaving predictions...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'prediction_prob': results['predictions'].flatten(),
        'prediction_binary': results['binary_predictions'].flatten(),
        'true_label': y_test.cpu().numpy().flatten()
    })
    
    # Add per-rule scores if available
    if 'rule_scores' in results:
        num_rules = results['rule_scores'].shape[1]
        for i in range(num_rules):
            df[f'rule_{i+1}_score'] = results['rule_scores'][:, i]
    
    # Save to CSV
    output_path = os.path.join(args.data_path, 'results', 'test_predictions.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved predictions to: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    X_test, y_test = load_test_data(args.data_path, args.test_file_x, args.test_file_y, device)
    
    # Get input dimension
    input_dim = args.input_dim if args.input_dim is not None else X_test.shape[1]
    
    # Load variable matrix for structured data
    variable_matrix = None
    if args.structured_data:
        try:
            variable_matrix, variables, head_vars, connected_vars, body_predicates = \
                load_predicates_and_variable_matrix(args.data_path, args.head_predicate)
            print(f"Loaded variable matrix for structured data")
            print(f"Number of predicates: {len(body_predicates)}")
        except FileNotFoundError:
            print("Warning: Predicate files not found. Using unstructured mode.")
            variable_matrix = None
    
    # Load model
    if args.model_type == 'best_model':
        if args.model_name is None:
            raise ValueError("--model_name is required for best_model type")
        
        # Auto-detect num_rules if not provided
        if args.num_rules is None:
            # Try to infer from model directory
            print("Warning: --num_rules not provided. Please specify the number of rules.")
            args.num_rules = 1  # Default fallback
        
        model = load_best_model(
            args.model_dir, args.model_name, args.num_rules, input_dim, variable_matrix, device
        )
    else:  # unified_model
        weights_path = os.path.join(args.model_dir, args.weights_file)
        model = load_unified_model(weights_path, input_dim, variable_matrix, device)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, args)
    
    # Add labels to results for ranking metrics
    results['metrics']['labels'] = y_test.cpu().numpy()
    
    # Compute and save ranking metrics
    compute_and_save_ranking_metrics(results, args, args.model_dir)
    
    # Save predictions
    save_predictions(results, X_test, y_test, args)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
