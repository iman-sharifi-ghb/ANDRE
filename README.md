# ANDRE
An Attention-based Neurosymbolic Differentiable Rule Extractor

This repository implements ANDRE, a neural-symbolic learning framework for extracting logical rules from data using neural networks. The workflow consists of two main stages:

1. **Data Propositionalization**: Transforming relational datasets into trainable feature matrices using `third_party.py`.
2. **Rule Learning & Evaluation**: Training the ANDRE model with `main.py` to discover interpretable rules and evaluate their performance.

---

## 1. Extracting Trainable Datasets with `third_party.py`

`third_party.py` processes relational data (e.g., in `.nl` format) and generates propositionalized datasets suitable for neural-symbolic learning.

### Steps:

1. **Prepare your dataset**: Place your relational dataset (e.g., `nations.nl`) in a subfolder under `datasets/`, e.g., `datasets/nations/nations.nl`.

2. **Run the data assembler**:
   
	 ```bash
	 python third_party.py --dataset_name <DATASET_NAME> --dataset_file <DATASET_FILE> --predicate <TARGET_PREDICATE> --variable_depth <DEPTH>
	 ```
	 - `--dataset_name`: Name of the dataset folder (e.g., `nations`)
	 - `--dataset_file`: Name of the dataset file (e.g., `nations.nl`)
	 - `--predicate`: Target predicate to learn (e.g., `locatedIn`)
	 - `--variable_depth`: Depth of variable relationships (e.g., `1`)

	 This will generate propositionalized data (feature matrices, labels, meta-info) in `datasets/<DATASET_NAME>/<TARGET_PREDICATE>/`.

---

## 2. Learning Rules and Evaluation with `main.py`

`main.py` is the entry point for training the ANDRE model, extracting rules, and evaluating their quality.

### Usage:

```bash
python main.py [OPTIONS]
```

#### Key Arguments:
- `--synthetic_data [True|False]`: Use synthetic data or load from file.
- `--data_path <PATH>`: Path to the data directory (for non-synthetic data, e.g., `datasets/nations/locatedIn/`).
- `--head_predicate <PRED>`: Head predicate for structured data (e.g., `locatedIn(X1, X2)`).
- `--num_rules <N>`: Maximum number of rules to learn.
- `--num_epochs <N>`: Number of training epochs.
- `--visualize_weights [True|False]`: Visualize learned weights.
- `--evaluate_individual_rules [True|False]`: Evaluate and sort individual rules by accuracy.
- `--device [auto|cpu|cuda]`: Device to use for training.

For a full list of options, run:
```bash
python main.py --help
```

### Example (using a processed dataset):

```bash
python main.py --synthetic_data False --structured_data True \
	--data_path datasets/nations/locatedIn/ \
	--head_predicate "locatedIn(X1, X2)" \
	--num_rules 4 --num_epochs 500 --visualize_weights True
```

---

## 3. Outputs and Evaluation Metrics

After training, `main.py` will:
- Print evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, specificity) for both training and validation sets.
- Save the best model's metrics as a JSON file in the results directory.
- Compute and display ranking metrics:
	- **MRR (Mean Reciprocal Rank)**
	- **Hits@K** (for K = 1, 3, 5, 10)
- Print the final extracted logical rule(s) in human-readable form.
- Optionally, evaluate and rank individual subrules if `--evaluate_individual_rules` is enabled.

All logs and results are saved in the `results/` subdirectory of your data path.

---

## 4. Dependencies

- Python 3.7+
- PyTorch
- scikit-learn
- numpy, matplotlib, etc.

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 5. File Overview

- `third_party.py`: Data propositionalization and preprocessing.
- `main.py`: Main training, rule extraction, and evaluation script.
- `ANDRE_Net2.py`: Core neural-symbolic model implementation.
- `utils.py`: Utility functions for data handling, rule interpretation, and metrics.

---

## 6. Citation

If you use this codebase, please cite the corresponding paper (add citation here).

---

## 7. License

See LICENSE file for details.

