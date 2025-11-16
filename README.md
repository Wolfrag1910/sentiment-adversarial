# Sentiment Analysis: Classifying text data as positive or negative sentiment

This project implements and evaluates adversarial attacks and defenses for text classification using Convolutional Neural Networks (CNNs) on the IMDB sentiment analysis dataset.

## Project Overview

The project explores the vulnerability of text classification models to adversarial attacks and implements defense mechanisms to improve model robustness. It includes:

- **Baseline Model**: Text CNN for binary sentiment classification
- **Adversarial Attacks**: 
  - **Keyword substitution Attack** (word-level, semantic).
  - **Character-level perturbation** (syntactic noise).
- **Defense Mechanisms**: 
  - **Input sanitization** (pre-processing at inference time).
  - **Adversarial training** (fine-tuning with adversarial examples).

We evaluate clean accuracy and robust accuracy under different attacks / budgets and defense combinations.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- PyTorch
- IMDB dataset (aclImdb)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Wolfrag1910/sentiment-adversarial
cd sentiment-adversarial
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the IMDB dataset:
   - Download from: http://ai.stanford.edu/~amaas/data/sentiment/
   - Extract to: `data/raw/aclImdb/`
   - The directory structure should be:
     ```
     data/raw/aclImdb/
     ├── train/
     │   ├── pos/
     │   └── neg/
     └── test/
         ├── pos/
         └── neg/
     ```

## Running the Code

### 1. Train Baseline Model

Train a clean CNN model on the IMDB dataset:

```bash
python -m src.train --config experiments/configs/imdb_cnn.yaml
```

This will:
- Train a Text CNN model for sentiment classification on the number of epochs configured
- Save the best model checkpoint to `experiments/logs/imdb_cnn.pt`
- Log training metrics to `experiments/logs/imdb_cnn_clean.csv`
- Save the vocabulary from the training set to `experiments/logs/vocab.pkl`

### 2. Evaluate baseline on clean test data

```bash
python -m src.eval --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn.pt
```

### 3. Evaluate Adversarial Attacks (Without Defense)

Test the baseline model against adversarial attacks:

```bash
python src.eval_attacks --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn.pt --output_csv experiments/logs/results_attacks.csv
```

This evaluates:
- **Keyword Substitution Attack**: Replaces important words with synonyms from embedding space
  - for budgets k ∈ {1,2,3}
- **Character Perturbation Attack**: Applies character-level edits (insert/delete/substitute/swap)
  - for budgets m ∈ {1,2,3}

and writes results_attacks.csv with:
- **attack** – attack type (keyword_substitution, char_perturbation)
- **budget** – k or m
- **n_total** – number of test examples considered
- **n_clean_correct** – how many were correct before the attack
- **n_success** – how many successful adversarial examples
- **asr** – attack success rate = n_success / n_clean_correct
- **robust_acc** – robust accuracy = (n_clean_correct - n_success) / n_clean_correct
- **avg_changes** – average number of word/character changes per attacked example

### 4. Evaluate with Input Sanitization Defense

Test the model with input sanitization enabled:

```bash
python src.eval_attacks --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn.pt --sanitize --output_csv experiments/logs/results_attacks_sanitized.csv
```
This applies a pre-processing pipeline before the model sees the input:
1. Unicode NFKC normalization and lowercasing.
2. URL and user-handle stripping (<url>, <user>).
3. Collapsing repeated characters (e.g. “coooool” → “cool”).
4. Basic leetspeak normalization (0→o, 3→e, etc.).
5. Simple spell-correction for OOV tokens using edit-distance-1 and a frequency-based vocabulary.

This defense is very effective against character-level attacks, and slightly improves robustness against keyword-substitution.

### 5. Adversarial Training

Fine-tune the model using adversarial examples:

```bash
python src.train_adv --config experiments/configs/imdb_cnn.yaml
```

This will:
- Load the baseline model from `experiments/logs/imdb_cnn.pt`
- Generate adversarial examples during training
- In each batch:
  - converts a fraction (adv_ratio) of examples into adversarial ones using the chosen attacks and budgets
  - trains on the mixed clean + adversarial batch
- Save the adversarially trained model to `experiments/logs/imdb_cnn_adv.pt`
- Log training metrics to `experiments/logs/imdb_cnn_adv.csv`

### 6. Evaluate Adversarially Trained Model

Test the robustness of the adversarially trained model:

#### Clean accuracy

```bash
python -m src.eval --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn_adv.pt
```

#### Attacks without sanitization
```bash
python src.eval_attacks --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn_adv.pt --output_csv experiments/logs/results_attacks_adv.csv
```

#### Attacks with sanitization
```bash
python -m src.eval_attacks --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn_adv.pt --sanitize --output_csv experiments/logs/results_attacks_adv_sanitized.csv
```

## Algorithms and Methods

### Model Architecture

**Text CNN (Convolutional Neural Network for Text)**
- Embedding layer: Maps tokens to dense vectors (200-dimensional)
- Multiple convolutional filters: Captures n-gram patterns (filter sizes: 3, 4, 5)
- Max-pooling: Extracts most important features from each filter
- Fully connected layer: Maps to binary sentiment classes
- Dropout regularization: Prevents overfitting (p=0.5)
- Trained with cross-entropy loss and Adam optimizer

### Adversarial Attacks

#### 1. Keyword Substitution Attack
A greedy word-level attack that replaces important words with semantically similar alternatives:

**Algorithm**:
1. **Importance Scoring**: Use leave-one-out method to identify words most critical to the prediction
   - Replace each word with `<unk>` token
   - Measure drop in true-label probability
   - Rank words by importance score
2. **Synonym Generation**: For each important word, find nearest neighbors in the model's embedding space
   - Compute cosine similarity between word embeddings
   - Select top-k candidates (k=20)
   - Filter out non-attackable tokens (punctuation, digits, very short words)
3. **Greedy Substitution**: Iteratively replace words to maximize prediction change
   - For each important word, try all synonym candidates
   - Select the substitution that most reduces true-label probability
   - Stop when misclassification occurs or budget is exhausted

**Constraints**:
- Maximum 20% of tokens can be changed
- Only attacks top-8 most important words
- Budgets tested: 1, 2, 3 word substitutions

#### 2. Character Perturbation Attack
A character-level attack that applies small edits to important words:

**Algorithm**:
1. **Importance Scoring**: Same leave-one-out method as keyword attack
2. **Character Edits**: Apply random character-level perturbations
   - **Substitute**: Replace a character with a random letter
   - **Delete**: Remove a character
   - **Insert**: Add a random character at a position
   - **Swap**: Transpose two adjacent characters
3. **Greedy Application**: Apply edits to important words
   - Try one edit per word
   - Keep edits that reduce true-label probability
   - Stop when misclassification occurs or budget is exhausted

**Constraints**:
- Maximum 15% of total characters can be modified
- Only attacks words with 3+ characters
- Budgets tested: 1, 2, 3 character edits

### Defense Mechanisms

#### 1. Input Sanitization
A preprocessing defense that normalizes and corrects adversarial perturbations:

**Steps**:
1. **Unicode Normalization**: Apply NFKC normalization
2. **Lowercasing**: Convert all text to lowercase
3. **URL/Handle Removal**: Replace URLs and @mentions with special tokens
4. **Character Repetition Collapse**: "cooool" → "cool"
5. **Leetspeak Normalization**: "h3ll0" → "hello" (maps 0→o, 1→i, 3→e, 4→a, 5→s, 7→t)
6. **Spell Correction**: For out-of-vocabulary words:
   - Generate all edit-distance-1 candidates (Norvig algorithm)
   - Select the in-vocabulary candidate with highest training frequency
   - Handles typos and character perturbations

#### 2. Adversarial Training
A training-time defense that improves model robustness:

**Algorithm**:
1. **Start from Baseline**: Load pre-trained clean model
2. **Generate Adversarial Examples**: For each training batch:
   - Select 50% of examples to perturb (adv_ratio=0.5)
   - Randomly choose attack type (keyword or character)
   - Randomly choose budget (1 or 2 changes)
   - Generate adversarial example using chosen attack
3. **Train on Adversarial Data**: Update model using adversarial examples
   - Use smaller learning rate (0.0005 vs 0.001) for fine-tuning
   - Apply gradient clipping for stability
   - Train for 3 epochs
4. **Embedding Refresh**: Update embedding matrix after each epoch for keyword attack

**Benefits**:
- Model learns to be robust to perturbations seen during training
- Improves generalization to unseen adversarial examples
- Maintains good performance on clean data

### Evaluation Metrics

- **Attack Success Rate (ASR)**: Percentage of correctly classified examples that are successfully misclassified by the attack
- **Robust Accuracy**: Percentage of correctly classified examples that remain correct after attack
- **Average Changes**: Mean number of modifications per example
- **Clean Accuracy**: Accuracy on unperturbed test data

## Project Structure

```
.
├── data/
│   └── raw/                    # IMDB dataset
├── experiments/
│   ├── configs/
│   │   └── imdb_cnn.yaml      # Configuration file
│   └── logs/                   # Training logs and checkpoints
├── figures/                    # Data visualizations
├── scripts/
│   └── run_clean.sh           # Training script
├── src/
│   ├── attacks/
│   │   └── text_attacks.py    # Adversarial attack implementations
│   ├── defenses/
│   │   ├── __init__.py
│   │   └── sanitization.py    # Input sanitization defense
│   ├── models/
│   │   └── cnn_text.py        # Text CNN model
│   ├── utils/
│   │   ├── common.py          # Utility functions
│   │   └── metrics.py         # Evaluation metrics
│   ├── check_env.py           # Environment checker
│   ├── data.py                # Data loading and preprocessing
│   ├── eval.py                # Model evaluation
│   ├── eval_attacks.py        # Attack evaluation script
│   ├── inspect_data.py        # Data inspection utilities
│   ├── train.py               # Baseline training script
│   ├── train_adv.py           # Adversarial training script
│   └── vocab.py               # Vocabulary management
├── README.md                   # This file
└── requirements.txt           # Python dependencies
```

## Configuration

All hyperparameters are configured in `experiments/configs/imdb_cnn.yaml`:

- **Model parameters**: Embedding dimension, filter sizes, dropout rate
- **Training parameters**: Learning rate, batch size, epochs, early stopping
- **Attack parameters**: Budgets, constraints, number of evaluation examples
- **Defense parameters**: Adversarial training ratio, fine-tuning epochs

Can modify these parameters to experiment with different settings.

- **Reproducibility**: We fix random seeds (Python, NumPy, PyTorch) via the config’s `project.seed` field.

## Additional Notes

### Key Implementation Details

1. **Vocabulary Management**: 
   - Built from training data only to prevent test set leakage
   - Supports configurable minimum frequency and maximum vocabulary size
   - Handles unknown tokens with `<unk>` token

2. **Padding and Truncation**:
   - All sequences padded/truncated to fixed length (400 tokens)
   - Padding index (0) is properly handled in embedding layer

3. **Attack Efficiency**:
   - Importance scoring uses leave-one-out method (linear in sequence length)
   - Greedy search for substitutions (avoids exponential search space)
   - Early stopping when misclassification is achieved

4. **Defense Trade-offs**:
   - Input sanitization: Fast, no retraining needed, but limited effectiveness
   - Adversarial training: More robust but requires retraining and may slightly reduce clean accuracy

### Results

1. **B** = Baseline CNN
2. **B+S** = Baseline + Sanitization
3. **AT** = Adv-trained CNN
4. **AT+S** = Adv-trained + Sanitization

#### Clean accuracy (no attacks)
| Model  | Description     | Test Acc  |
| ------ | --------------- | --------- |
| **B**  | Baseline CNN    | **87.4%** |
| **AT** | Adv-trained CNN | **88.2%** |
- Adversarial training does not hurt clean accuracy – it even slightly improves it.

#### Keyword-substitution attacks
| Budget k | Model    | ASR      | RA        |
| -------- | -------- | -------- | --------- |
| **1**    | **B**    | 8.4      | 91.6      |
|          | **B+S**  | 10.3     | 89.7      |
|          | **AT**   | 9.6      | 90.4      |
|          | **AT+S** | **8.6**  | **91.3**  |
| **2**    | **B**    | 17.0     | 82.9      |
|          | **B+S**  | 19.4     | 80.6      |
|          | **AT**   | 17.1     | 82.9      |
|          | **AT+S** | **16.0** | **83.98** |
| **3**    | **B**    | 27.5     | 72.5      |
|          | **B+S**  | 30.1     | 69.9      |
|          | **AT**   | 29.8     | 70.2      |
|          | **AT+S** | **25.9** | **74.1**  |
- Takeaways for keyword attack
  - Hardest attack overall – all models lose robustness as k increases
  - Sanitization alone (B+S) slightly hurts robustness here (it was designed for character noise, not semantic swaps)
  - Adv training alone (AT) is roughly similar to baseline
  - Best model is clearly AT+S, especially at higher budgets

#### Character-level perturbation attacks

| Budget m | Model    | ASR     | RA       |
| -------- | -------- | ------- | -------- |
| **1**    | **B**    | 8.4     | 91.6     |
|          | **B+S**  | **0.7** | **99.3** |
|          | **AT**   | 6.8     | 93.2     |
|          | **AT+S** | ~1.8    | ~98.2    |
| **2**    | **B**    | 13.6    | 86.4     |
|          | **B+S**  | **1.8** | **98.2** |
|          | **AT**   | 12.2    | 87.8     |
|          | **AT+S** | ~2.5    | ~97.5    |
| **3**    | **B**    | 18.9    | 81.1     |
|          | **B+S**  | **2.1** | **97.9** |
|          | **AT**   | 17.4    | 82.6     |
|          | **AT+S** | ~2.1    | ~97.9    |
- Takeaways for char attack
  - Baseline is noticeably vulnerable to char noise (ASR 8–19%)
  - Sanitization alone (B+S) almost completely kills the attack
  - Adv training alone (AT) improves robustness a bit but nowhere near sanitization
  - AT+S is also extremely robust, but not much better than B+S

### Troubleshooting

- **Out of Memory**: Reduce batch size in config file
- **Slow Training**: Use GPU by setting `device: "cuda"` in config (if available)
- **Missing Data**: Ensure IMDB dataset is properly extracted to `data/raw/aclImdb/`

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
- Alzantot, M. et al. (2018). Generating Natural Language Adversarial Examples
- Goodfellow, I. et al. (2014). Explaining and Harnessing Adversarial Examples
- Norvig, P. How to Write a Spelling Corrector
- Monserrat, V. et al. (2023). White-Box Adversarial	Attacks	Against	SentimentAnalysis Models	using	an	Aspect-Based	Approach
- Tarik, E. et al. (2025). Lexicon-Based Random Substitute and Word-Variant Voting Models for Detecting Textual Adversarial Attacks

## Contact / Authors
- **Samir Kurbissa** - `samir.kurbissa@stud.trans.upb.ro`
- **Adrian-Mihai Costică** - `adrian.costica@stud.trans.upb.ro`
