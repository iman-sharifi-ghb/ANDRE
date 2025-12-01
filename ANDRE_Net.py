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

def special_randon_tensor():
    t = torch.randn(3)
    t[2] = t[:2].max() + 2 * torch.rand(1).item() + 3
    return t

# Define CustomNeuron and FuzzyDisjunctionNet
class ProbSumActivation(nn.Module):
    def __init__(self):
        super(ProbSumActivation, self).__init__()
        # self.raw_weights = nn.Parameter(torch.randn(3) * 0.1)  # Small init to prevent saturation
        self.raw_weights = nn.Parameter(special_randon_tensor())  # Small init to prevent saturation

    def forward(self, x):
        self.weights = torch.softmax(self.raw_weights, dim=0)
        x1 = x
        x2 = 1 - x
        x3 = torch.ones_like(x)
        X1 = self.weights[0] * x1
        X2 = self.weights[1] * x2
        X3 = self.weights[2] * x3
        activated_input = X1 + X2 + X3 - (X1 * X2 + X1 * X3 + X2 * X3) + (X1 * X2 * X3)
        return activated_input

# Define FuzzyDisjunctionNet
class ANDRE(nn.Module):
    def __init__(self, num_rules, pot_body_pred_dim, variable_matrix=None, 
                 beta=20, lambda_=10, lr=0.005, weight_decay=1e-5):
        super(ANDRE, self).__init__()
        self.num_rules = num_rules
        self.beta = beta
        self.lambda_ = lambda_
        self.input_dim = pot_body_pred_dim
        self.V = variable_matrix
        self.M_matrices = []
        self.rule = nn.ModuleList([
            nn.ModuleList([ProbSumActivation() for _ in range(pot_body_pred_dim)])
            for _ in range(num_rules)
        ])
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        self.saved_weights = []
        self.entropies = []
        self.M_matrices = []
        rule_outputs = []

        for i, subrule in enumerate(self.rule):
            conjunctions = []
            subrule_weights = []
            entropy_sum = 0.0
            P = []
            for j, probsumact in enumerate(subrule):
                xi = x[:, [j % self.input_dim]]
                conjunctions.append(probsumact(xi))
                weights = probsumact.weights
                subrule_weights.append(weights)
                entropy_sum += -(weights * torch.log(weights + 1e-6)).sum()
                P.append(1.0 - weights[2])  # 1 - w3

            subrule_vector = torch.cat(subrule_weights)
            self.saved_weights.append(subrule_vector)
            self.entropies.append(entropy_sum)

            P_tensor = torch.stack(P)
            if self.V is not None:
                M = torch.matmul(P_tensor, self.V)
            else:
                M = None
            self.M_matrices.append(M)

            # Soft min of neuron outputs using attention mechanism
            # find -beta * outputs for all neurons, where beta is a hyperparameter
            attention_weights = torch.softmax(-self.beta * torch.stack(conjunctions), dim=0)
            outputs = [attention_weights[i] * conjunctions[i] for i in range(self.input_dim)]
            activated_outputs = torch.sigmoid(self.lambda_*(sum(outputs) - GAMMA1))  # Apply sigmoid to the sum
            rule_outputs.append(activated_outputs)

        # Fuzzy disjunction: Soft max of rule outputs using attention mechanism
        # find BETA * rule_outputs for all rules, where beta is a hyperparameter
        attention_weights = torch.softmax(self.beta * torch.stack(rule_outputs), dim=0)
        rule_outputs = [attention_weights[i] * rule_outputs[i] for i in range(self.num_rules)]
        activated_output = torch.sigmoid(self.lambda_*(sum(rule_outputs) - GAMMA2))  # Apply sigmoid to the sum

        return activated_output
    
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
            outputs = self.forward(x).squeeze()
            
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