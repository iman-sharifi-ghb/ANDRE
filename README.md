# ANDRE: Attention-based Neural-symbolic Differentiable Rule Extractor

An advanced neural-symbolic learning framework for extracting interpretable logical rules from relational data using differentiable neural networks with attention mechanisms.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🌟 Key Features

- **Neural-Symbolic Integration**: Combines the learning power of neural networks with the interpretability of symbolic logic
- **Differentiable Rule Learning**: End-to-end gradient-based optimization for discovering logical rules
- **Attention-Based Architecture**: Learns which predicates are important through soft attention mechanisms
- **Multi-Objective Training**: Balances accuracy with rule quality through specialized loss functions
- **Comprehensive Evaluation**: Includes ranking metrics (MRR, Hits@K), traditional ML metrics, and rule coverage analysis

---

## 📋 Table of Contents

- [Quick Start Demo](#-quick-start-demo)
- [Core Techniques](#-core-techniques)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [Data Propositionalization](#1-data-propositionalization-with-third_partypy)
  - [Rule Learning](#2-rule-learning-and-evaluation-with-mainpy)
- [Output and Metrics](#-output-and-metrics)
- [Architecture](#-architecture)
- [File Overview](#-file-overview)
- [Citation](#-citation)

---

## 🚀 Quick Start Demo

Try ANDRE on the classic **grandparent** problem with synthetic data:

```bash
python main.py \
    --synthetic_data 1 \
    --structured_data 1 \
    --head_predicate "grandparent(X1, X3)" \
    --rule_counts "4" \
    --num_epochs 500 \
    --batch_size 128 \
    --num_samples 1000 \
    --num_restarts 1 \
    --device cpu \
    --save_model 1 \
    --visualize_weights 0
```

**Expected Output:**

ANDRE automatically discovers the four logical rules for grandparent relationships:

```
[1] grandparent(X1, X3) :- father(X1, X2) and mother(X2, X3).
    Training Acc: 0.6825 | Eval Acc: 0.6900
    Val Coverage: (N_b=53, N_r=53, N_r/N_b=1.0000)
    Train Coverage: (N_b=219, N_r=219, N_r/N_b=1.0000)

[2] grandparent(X1, X3) :- father(X1, X2) and father(X2, X3).
    Training Acc: 0.6725 | Eval Acc: 0.6850
    Val Coverage: (N_b=52, N_r=52, N_r/N_b=1.0000)
    Train Coverage: (N_b=211, N_r=211, N_r/N_b=1.0000)

[3] grandparent(X1, X3) :- mother(X1, X2) and mother(X2, X3).
    Training Acc: 0.6850 | Eval Acc: 0.6500
    Val Coverage: (N_b=49, N_r=47, N_r/N_b=0.9592)
    Train Coverage: (N_b=225, N_r=223, N_r/N_b=0.9911)

[4] grandparent(X1, X3) :- father(X2, X3) and mother(X1, X2).
    Training Acc: 0.6538 | Eval Acc: 0.6500
    Val Coverage: (N_b=45, N_r=45, N_r/N_b=1.0000)
    Train Coverage: (N_b=196, N_r=196, N_r/N_b=1.0000)
```

**Final Unified Rule:**
```
grandparent(X1, X3) :- 
    father(X1, X2) and mother(X2, X3) OR 
    father(X1, X2) and father(X2, X3) OR 
    mother(X1, X2) and mother(X2, X3) OR 
    father(X2, X3) and mother(X1, X2).
```

**Performance:** Training completes in ~11 seconds on CPU with >99% accuracy!

---

## 🔬 Core Techniques

### 1. **Differentiable Soft Logic**
- Converts discrete logical operations (AND, OR, NOT) into continuous, differentiable functions
- Uses soft attention weights instead of hard binary selections
- Enables gradient-based optimization for rule discovery

### 2. **Multi-Component Loss Function**
The training objective balances multiple goals:

```python
Total Loss = BCE Loss + α·Entropy Loss + β·Range Restriction Loss 
             + γ·Connectivity Loss + δ·Digitization Loss + λ·Similarity Penalty
```

- **BCE Loss**: Standard binary cross-entropy for prediction accuracy
- **Entropy Loss**: Encourages decisive predicate selections (high confidence)
- **Range Restriction Loss**: Ensures variables in head predicate appear exactly once in body
- **Connectivity Loss**: Enforces variable connectivity across predicates
- **Digitization Loss**: Pushes weights toward discrete values for interpretability
- **Similarity Penalty**: Prevents duplicate rules by penalizing similar weight patterns

### 3. **Dynamic Loss Scheduling**
Loss weights are scheduled during training to progressively emphasize rule quality:
- Early epochs: Focus on accuracy (high BCE weight)
- Later epochs: Increase structural penalties (connectivity, range restriction)
- Controlled transition ensures stable learning

### 4. **Balanced Batch Sampling**
- Custom sampler ensures equal positive/negative samples per batch
- Prevents class imbalance issues
- Improves model generalization

### 5. **Variable Matrix Representation**
- Encodes predicate-variable relationships as binary matrices
- Enables efficient computation of variable connectivity constraints
- Supports range restriction validation

### 6. **Optimal Threshold Search**
- Automatically finds the best decision threshold on validation set
- Maximizes F1-score rather than using default 0.5
- Reported separately from default threshold metrics

### 7. **Multi-Restart Training**
- Performs multiple training runs with different initializations
- Selects best model based on validation performance
- Helps escape local minima and discover diverse rules

### 8. **Rule Confidence Ranking**
Multiple metrics available for ranking discovered rules:
- **Validation Accuracy**: Individual rule accuracy on held-out data
- **N_r/N_b Ratio**: Coverage quality (correct predictions / total predictions)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Rule correctness when it fires

### 9. **Knowledge Graph Evaluation**
- Computes ranking metrics (MRR, Hits@K) for link prediction
- Evaluates model as a knowledge graph completion system
- Supports corrupted triple generation for realistic evaluation

### 10. **Rule Deduplication and Filtering**
- Automatically removes duplicate rules across restarts
- Validates rules for structural correctness (variable constraints)
- Builds unified models from valid rule sets

### 11. **Training Time Tracking**
- Measures and reports training time per rule count
- Helps analyze computational costs and scalability
- Provides detailed timing summaries

### 12. **Data Perturbation for Binary Datasets**
- Adds small noise to binary data (0/1) for smoother gradients
- Clips values to [0, 1] range
- Improves training stability while preserving semantics

---

## 💾 Installation

### Requirements
- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ANDRE.git
cd ANDRE

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage Guide

ANDRE workflow consists of two main stages:

### 1. Data Propositionalization with `third_party.py`

Transform relational datasets into neural network-friendly feature matrices.

#### Input Format
Place your relational dataset in `.nl` format under `datasets/`:
```
datasets/
  ├── nations/
  │   └── nations.nl
  ├── kinship/
  │   └── kinship.nl
  └── ...
```

#### Command

```bash
python third_party.py \
    --dataset_name nations \
    --dataset_file nations.nl \
    --predicate intergovorgs3 \
    --variable_depth 1
```

**Arguments:**
- `--dataset_name`: Dataset folder name
- `--dataset_file`: Input file name (`.nl` format)
- `--predicate`: Target predicate to learn
- `--variable_depth`: Variable relationship depth (typically 1 or 2)

**Output:** Generates propositionalized data in `datasets/nations/intergovorgs3/`:
- `x_train.dt`: Feature matrix
- `y_train.dt`: Labels
- `predicate.list`: List of body predicates
- `meta_info.dt`: Metadata about the dataset

---

### 2. Rule Learning and Evaluation with `main.py`

Train ANDRE to discover and evaluate logical rules.

#### Synthetic Data Example

```bash
python main.py \
    --synthetic_data 1 \
    --structured_data 1 \
    --head_predicate "grandparent(X1, X3)" \
    --rule_counts "2,4,6" \
    --num_epochs 500 \
    --batch_size 128 \
    --num_restarts 3 \
    --device cuda
```

#### Real Dataset Example

```bash
python main.py \
    --synthetic_data 0 \
    --structured_data 1 \
    --data_path "datasets/nations/intergovorgs3/" \
    --head_predicate "intergovorgs3(X1, X2)" \
    --rule_counts "6,8,10" \
    --num_epochs 500 \
    --batch_size 1024 \
    --num_restarts 1 \
    --device cuda \
    --save_model 1 \
    --val_split 0.05 \
    --rule_confidence_metric nr_nb_ratio
```

#### Key Arguments

**Data Options:**
- `--synthetic_data [0|1]`: Use synthetic (1) or real data (0)
- `--structured_data [0|1]`: Enable structured predicates
- `--data_path <PATH>`: Path to propositionalized data
- `--head_predicate <PRED>`: Target predicate (e.g., `grandparent(X1, X3)`)

**Model Architecture:**
- `--rule_counts <COUNTS>`: Comma-separated rule counts to try (e.g., `"4,6,8"`)
- `--num_rules <N>`: Alternative: single rule count

**Training Parameters:**
- `--num_epochs <N>`: Training epochs (default: 500)
- `--batch_size <N>`: Batch size (default: 128)
- `--num_restarts <N>`: Random restarts (default: 3)
- `--lr <FLOAT>`: Learning rate (default: 0.01)
- `--val_split <FLOAT>`: Validation split ratio (default: 0.2)

**Loss Weights:**
- `--loss_schedule [0|1]`: Enable dynamic loss scheduling
- `--ent_weight <FLOAT>`: Entropy loss weight (default: 0.1)
- `--rr_weight <FLOAT>`: Range restriction weight (default: 1.0)
- `--conn_weight <FLOAT>`: Connectivity weight (default: 5.0)
- `--digit_weight <FLOAT>`: Digitization weight (default: 0.01)
- `--sim_weight <FLOAT>`: Similarity penalty weight (default: 0.2)

**Evaluation:**
- `--rule_confidence_metric <METRIC>`: Ranking metric (`val_accuracy`, `nr_nb_ratio`, `val_f1`)
- `--k <N>`: k for Hits@k metric (default: 100)
- `--r <N>`: r for ranking evaluation (default: 100)

**Output Options:**
- `--save_model [0|1]`: Save trained model
- `--visualize_weights [0|1]`: Generate weight visualizations
- `--device [cpu|cuda|auto]`: Compute device

**Data Preprocessing:**
- `--perturb [0|1]`: Add noise to binary data (default: 1)
- `--np <FLOAT>`: Label noise percentage (default: 0.0)

#### Full Options

```bash
python main.py --help
```

---

## 📊 Output and Metrics

### Console Output

1. **Training Progress**: Loss curves, accuracy per epoch, discovered rules
2. **Individual Rule Evaluation**: Accuracy, precision, recall, F1, coverage stats
3. **Optimal Threshold**: F1-maximizing decision threshold
4. **Ranking Metrics**: MRR and Hits@K for knowledge graph completion
5. **Training Time**: Per-rule-count timing summary

### Saved Files

All outputs are saved in `<data_path>/results/`:

```
results/
├── training_log_<timestamp>.log                 # Complete console output
├── best_model_metrics_<timestamp>.json          # Best model performance
├── threshold_analysis_<timestamp>.json          # Threshold optimization results
└── sorted_rules_<timestamp>/
    ├── sorted_rules_weights.pt                  # PyTorch weights
    ├── sorted_rules_weights.json                # JSON weights
    ├── sorted_rules_metadata.json               # Rule metadata
    ├── unified_model_metrics.json               # Unified model performance
    ├── ranking_metrics.json                     # MRR/Hits@K results
    ├── train_sorted_rule_scores.csv             # Training set predictions
    └── val_sorted_rule_scores.csv               # Validation set predictions
```

### Evaluation Metrics

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Specificity
- Confusion Matrix

**Ranking Metrics:**
- Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10, Hits@K

**Rule Quality Metrics:**
- N_b: Number of samples where rule fires (body satisfied)
- N_r: Number of correct predictions when rule fires
- N_r/N_b: Rule precision (confidence)
- Individual rule accuracy on train/validation sets

---

## 🏗️ Architecture

### ANDRE Model Components

1. **Input Layer**: Accepts propositionalized predicate features
2. **Rule Blocks**: One per rule, each containing:
   - **Attention Layer**: Learns 3-component weights (true, negated, ignore) per predicate
   - **Soft AND**: Differentiable conjunction over selected predicates
3. **Aggregation Layer**: Soft OR over all rules
4. **Output**: Binary prediction with learned threshold

### Variable Tracking

- **Variable Matrix (V)**: Encodes which variables appear in which predicates
- **M Matrix**: Tracks variable occurrence counts across selected predicates
- **Range Restriction**: Ensures head variables appear exactly once
- **Connectivity**: Ensures all variables are connected through predicates

---

## 📁 File Overview

```
ANDRE/
├── main.py                    # Main training and evaluation script
├── ANDRE_Net2.py             # Core model architecture
├── utils.py                  # Utility functions (metrics, rule extraction, visualization)
├── third_party.py            # Data propositionalization pipeline
├── evaluate_model.py         # Model evaluation utilities
├── requirements.txt          # Python dependencies
├── datasets/                 # Dataset directory
│   ├── nations/
│   ├── kinship/
│   ├── fb15k_237/
│   └── ...
├── test_dir/                 # Test outputs
└── saved_models/             # Trained model checkpoints
```

**Key Files:**
- **`main.py`**: Entry point for training, includes all loss functions and training loops
- **`ANDRE_Net2.py`**: Defines the neural-symbolic architecture with soft logic operations
- **`utils.py`**: Helper functions for rule interpretation, metrics, and visualization
- **`third_party.py`**: Converts relational data to propositional features

---

## 📚 Citation

If you use ANDRE in your research, please cite:

```bibtex
@article{ANDRE2026,
  title={ANDRE: Adaptive Neural-symbolic Differentiable Rule Engine},
  author={Your Name et al.},
  journal={Conference/Journal Name},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

## 🙏 Acknowledgments

This work builds upon advances in neural-symbolic AI, differentiable logic, and knowledge graph completion.

