import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import json
GAMMA1 = 0.5
GAMMA2 = 0.5

def special_randon_tensor(input_dim):
    t = torch.randn(input_dim, 3)
    t[:, 2] = t[:, :2].max(dim=1).values + 2 * torch.rand(input_dim) + 3
    return t

class ProbSumActivation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.randn(input_dim, 3) * 0.1)
        # self.raw_weights = nn.Parameter(special_randon_tensor(input_dim))

    def forward(self, X_basis):
        """
        X_basis: [B, D, 3]
        raw_weights: [D, 3] - stored in logit space (before softmax)
        returns: activated: [B, D], andre_probs: [D, 3] - softmaxed probabilities
        """
        # Convert raw_weights (logits) to probabilities via softmax
        andre_probs = torch.softmax(self.raw_weights, dim=1)              # [D, 3]

        # Expand so batch size dimension aligns
        W = andre_probs.unsqueeze(0)                                       # [1, D, 3]

        # Weighted inputs
        Xw = W * X_basis                                               # [B, D, 3]

        # Compute fuzzy polynomial:
        # X1 + X2 + X3 - (pairwise products) + (triple product)
        S1 = Xw.sum(dim=2)                                             # [B, D]
        pair = (Xw[:, :, 0] * Xw[:, :, 1] +
                Xw[:, :, 0] * Xw[:, :, 2] +
                Xw[:, :, 1] * Xw[:, :, 2])                             # [B, D]
        triple = (Xw[:, :, 0] * Xw[:, :, 1] * Xw[:, :, 2])             # [B, D]

        activated = S1 - pair + triple                                 # [B, D]
        return activated, andre_probs


class RuleBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.activation = ProbSumActivation(input_dim)

    def forward(self, X_basis, beta, lambda_, GAMMA1, V=None):
        """
        X_basis: [B, D, 3]
        V: variable matrix [D, K] or None
        returns: rule_output [B], andre_probs [D,3], entropy, P [D], M [K] or None
        """
        # Apply ProbSum over all predicates at once
        conj, andre_probs = self.activation(X_basis)      # conj: [B, D], andre_probs: [D, 3]

        # Compute entropy for this rule
        entropy = -(andre_probs * (andre_probs + 1e-6).log()).sum()

        # P_j = 1 - w3 (probability that predicate is NOT constant)
        P = 1.0 - andre_probs[:, 2]                       # [D]

        # Compute M with proper gradient tracking
        if V is not None:
            M = torch.matmul(P, V)                    # [D] @ [D,K] = [K]
        else:
            M = None

        # Soft minimum over conjunctions (attention softmin)
        attn = torch.softmax(-beta * conj, dim=1)     # [B, D]
        out = (attn * conj).sum(dim=1)                # [B]
        rule_output = torch.sigmoid(lambda_ * (out - GAMMA1))

        return rule_output, andre_probs, entropy, P, M


class ANDRE(nn.Module):
    def __init__(self, num_rules, pot_body_pred_dim,
                 variable_matrix=None, beta=20, lambda_=10,
                 lr=0.005, weight_decay=1e-5):
        super().__init__()

        self.num_rules = num_rules
        self.input_dim = pot_body_pred_dim
        self.beta = beta
        self.lambda_ = lambda_

        # Register variable matrix as buffer (doesn't require grad but moves with model)
        if variable_matrix is not None:
            self.register_buffer('V', variable_matrix if isinstance(variable_matrix, torch.Tensor) 
                               else torch.tensor(variable_matrix, dtype=torch.float32))
        else:
            self.V = None

        # Create rule blocks
        self.rules = nn.ModuleList([RuleBlock(self.input_dim) for _ in range(num_rules)])

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        """
        x: [B, D]
        returns: activated_output: [B], all_andre_probs: list of softmaxed probabilities
        """
        B, D = x.shape

        # Precompute basis inputs once (major speed gain)
        X1 = x                               # [B, D]
        X2 = 1 - x                            # [B, D]
        X3 = torch.ones_like(x)               # [B, D]
        X_basis = torch.stack([X1, X2, X3], dim=2)  # [B, D, 3]

        rule_vals = []
        all_andre_probs = []
        all_entropies = []
        all_M = []

        # Evaluate each rule as a batch
        for r in self.rules:
            r_out, andre_probs, entropy, P, M = r(X_basis, self.beta, self.lambda_, GAMMA1, self.V)
            rule_vals.append(r_out)                      # r_out: [B]
            all_andre_probs.append(andre_probs.view(-1))     # flatten softmaxed probabilities
            all_entropies.append(entropy)
            all_M.append(M)

        # Stack rules
        rule_vals = torch.stack(rule_vals, dim=1)         # [B, R]

        # Fuzzy disjunction (soft max)
        attn = torch.softmax(self.beta * rule_vals, dim=1)  # [B, R]
        out = (attn * rule_vals).sum(dim=1)                 # [B]
        activated_output = torch.sigmoid(self.lambda_ * (out - GAMMA2))  # [B]

        return activated_output, all_andre_probs, all_entropies, all_M

    
    def evaluate(self, x, y, threshold=0.5):
        """
        Evaluate the model on given data.
        
        Args:
            x: Input features
            y: True labels
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        with torch.no_grad():
            outputs, _, _, _ = self.forward(x)
            
            # Move to CPU and convert to numpy for sklearn metrics
            y_true = y.cpu().numpy()
            y_pred_proba = outputs.cpu().numpy()
            y_pred = (y_pred_proba >= threshold).astype(float)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # ROC-AUC (requires probability scores)
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                # If only one class is present in y_true
                roc_auc = float('nan')
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'specificity': specificity,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'support_positive': int((y_true == 1).sum()),
                'support_negative': int((y_true == 0).sum())
            }
            
        return metrics
    
    def print_evaluation(self, x, y, threshold=0.5):
        """
        Evaluate and print metrics in a readable format.
        
        Args:
            x: Input features
            y: True labels
            threshold: Classification threshold (default: 0.5)
        """
        metrics = self.evaluate(x, y, threshold)
        
        print("=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print("-" * 60)
        print("Confusion Matrix:")
        print(f"  TP: {metrics['true_positives']:4d}  |  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  |  TN: {metrics['true_negatives']:4d}")
        print("-" * 60)
        print(f"Support: Positive={metrics['support_positive']}, Negative={metrics['support_negative']}")
        print("=" * 60)
        
        return metrics
    
    def save_model(self, save_dir, model_name="andre_model"):
        """
        Save ANDRE model weights and configuration to a directory.
        
        Args:
            save_dir: Directory path to save the model
            model_name: Name prefix for saved files (default: "andre_model")
        
        Saves:
            - {model_name}_weights.pt: Model state dict (weights)
            - {model_name}_config.json: Model configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        weights_path = os.path.join(save_dir, f"{model_name}_weights.pt")
        torch.save(self.state_dict(), weights_path)
        
        # Save model configuration
        config = {
            'num_rules': self.num_rules,
            'input_dim': self.input_dim,
            'beta': self.beta,
            'lambda_': self.lambda_,
            'has_variable_matrix': self.V is not None,
            'variable_matrix': self.V.tolist() if self.V is not None else None
        }
        
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved successfully to {save_dir}")
        print(f"  - Weights: {weights_path}")
        print(f"  - Config:  {config_path}")
        
        return weights_path, config_path
    
    @classmethod
    def load_model(cls, save_dir, model_name="andre_model", lr=0.005, weight_decay=1e-5):
        """
        Load ANDRE model from saved weights and configuration.
        
        Args:
            save_dir: Directory path where model was saved
            model_name: Name prefix of saved files (default: "andre_model")
            lr: Learning rate for optimizer (default: 0.005)
            weight_decay: Weight decay for optimizer (default: 1e-5)
        
        Returns:
            Loaded ANDRE model
        """
        # Load configuration
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Reconstruct variable matrix if it exists
        variable_matrix = None
        if config['has_variable_matrix'] and config['variable_matrix'] is not None:
            variable_matrix = torch.tensor(config['variable_matrix'], dtype=torch.float32)
        
        # Create model instance
        model = cls(
            num_rules=config['num_rules'],
            pot_body_pred_dim=config['input_dim'],
            variable_matrix=variable_matrix,
            beta=config['beta'],
            lambda_=config['lambda_'],
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Load weights
        weights_path = os.path.join(save_dir, f"{model_name}_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        
        print(f"Model loaded successfully from {save_dir}")
        print(f"  - Weights: {weights_path}")
        print(f"  - Config:  {config_path}")
        print(f"  - Num Rules: {config['num_rules']}, Input Dim: {config['input_dim']}")
        
        return model

    def export_rule_weights(self, save_dir, file_prefix="rules"):
        """
        Export raw rule weights from each `RuleBlock.activation.raw_weights`.

        Saves two files in `save_dir`:
          - `{file_prefix}_weights.pt`: torch-saved dict of raw tensors
          - `{file_prefix}_weights.json`: same data serialized as plain lists

        Returns tuple(paths)
        """
        os.makedirs(save_dir, exist_ok=True)
        pt_dict = {}
        json_dict = {'rules': []}

        for r_idx, rule_block in enumerate(self.rules):
            activation = rule_block.activation
            # activation.raw_weights: [D, 3]
            raw = activation.raw_weights.detach().cpu()
            D = raw.shape[0]
            rule_list = []
            for n_idx in range(D):
                vec = raw[n_idx]
                key = f"rule_{r_idx}_neuron_{n_idx}"
                pt_dict[key] = vec.clone()
                rule_list.append(vec.tolist())
            json_dict['rules'].append(rule_list)

        pt_path = os.path.join(save_dir, f"{file_prefix}_weights.pt")
        torch.save(pt_dict, pt_path)

        json_path = os.path.join(save_dir, f"{file_prefix}_weights.json")
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

        print(f"Exported rule raw-weights to: {pt_path} and {json_path}")
        return pt_path, json_path

    @staticmethod
    def load_rule_weights(path_or_dict):
        """
        Load rule weights saved by `export_rule_weights`.

        Accepts a path to a .pt file (dict of tensors) or a dict/list structure
        (for example JSON parsed into a dict). Returns nested list: rules -> neurons -> [w0,w1,w2].
        """
        # If given a .pt file
        if isinstance(path_or_dict, str):
            if not os.path.exists(path_or_dict):
                raise FileNotFoundError(path_or_dict)
            loaded = torch.load(path_or_dict)
            rules = {}
            for k, v in loaded.items():
                parts = k.split('_')
                try:
                    r = int(parts[1])
                    n = int(parts[3])
                except Exception:
                    continue
                rules.setdefault(r, {})[n] = v.tolist()

            ordered = []
            for r_idx in sorted(rules.keys()):
                neuron_map = rules[r_idx]
                ordered.append([neuron_map[n_idx] for n_idx in sorted(neuron_map.keys())])
            return ordered

        # If dict-like (JSON loaded)
        if isinstance(path_or_dict, dict):
            if 'rules' in path_or_dict:
                return path_or_dict['rules']

        # If already in nested-list format, return as-is
        return path_or_dict

    @classmethod
    def build_from_rule_weights(cls, rule_weights, pot_body_pred_dim, variable_matrix=None,
                                beta=20, lambda_=10, lr=0.005, weight_decay=1e-5):
        """
        Construct an `ANDRE` instance from `rule_weights`.

        `rule_weights` should be a list of rules (length R), each rule is a list of D neurons,
        and each neuron is an iterable of length 3 (raw weights).
        `pot_body_pred_dim` should match D used during training/data creation.
        """
        num_rules = len(rule_weights)
        model = cls(num_rules=num_rules, pot_body_pred_dim=pot_body_pred_dim,
                    variable_matrix=variable_matrix, beta=beta, lambda_=lambda_,
                    lr=lr, weight_decay=weight_decay)

        # Assign raw_weights into each RuleBlock.activation.raw_weights
        for r_idx, rule in enumerate(rule_weights):
            if r_idx >= len(model.rules):
                break
            activation = model.rules[r_idx].activation
            # activation.raw_weights: [D,3]
            D = activation.raw_weights.shape[0]
            # Prepare a tensor of same shape and fill
            with torch.no_grad():
                for n_idx, neuron_w in enumerate(rule):
                    if n_idx >= D:
                        break
                    vec = torch.tensor(neuron_w, dtype=activation.raw_weights.dtype)
                    activation.raw_weights.data[n_idx].copy_(vec)

        model.eval()
        return model

    def get_neuron_weights(self, softmaxed=True):
        """
        Return activation weights for all rules.

        Returns nested list: rules -> neurons -> [w0,w1,w2]
        """
        out = []
        for rule_block in self.rules:
            raw = rule_block.activation.raw_weights.detach().cpu()
            if softmaxed:
                w = torch.softmax(raw, dim=1).tolist()
            else:
                w = raw.tolist()
            out.append(w)
        return out

    def get_rule_scores(self, x, apply_attention=True):
        """
        Compute per-rule scores for each sample in `x`.

        Returns a tensor of shape (R, B) where R=num_rules and B=batch size.
        If `apply_attention` is True, rule-level attention is applied (same as forward).
        """
        B, D = x.shape
        X1 = x
        X2 = 1 - x
        X3 = torch.ones_like(x)
        X_basis = torch.stack([X1, X2, X3], dim=2)  # [B, D, 3]

        rule_vals = []
        for r_idx, rule_block in enumerate(self.rules):
            r_out, andre_probs, entropy, P, M = rule_block(X_basis, self.beta, self.lambda_, GAMMA1, self.V)
            # r_out: [B]
            rule_vals.append(r_out)

        stacked = torch.stack(rule_vals, dim=0)  # [R, B]

        if apply_attention:
            # compute attention across rules per sample: softmax over R for each sample
            attn = torch.softmax(self.beta * stacked, dim=0)  # [R, B]
            stacked = attn * stacked

        return stacked