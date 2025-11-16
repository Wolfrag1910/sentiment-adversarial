"""
Evaluation metrics for model training and testing.

This module provides utilities for accumulating predictions and computing
standard classification metrics (loss, accuracy, F1 score).
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class Accumulator:
    """
    Accumulates predictions and labels across batches for metric computation.
    
    Used during training and evaluation to collect all predictions before
    computing aggregate metrics like accuracy and F1 score.
    """
    
    def __init__(self):
        """Initialize empty accumulators for predictions, labels, and losses."""
        self.y_true = []  # Ground truth labels
        self.y_pred = []  # Predicted labels
        self.losses = []  # Batch losses

    def add_batch(self, logits, y, loss):
        """
        Add a batch of predictions and labels to the accumulator.
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            y: Ground truth labels (batch_size,)
            loss: Scalar loss value for this batch
        """
        self.losses.append(loss)
        
        # Convert logits to predicted class labels
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        self.y_pred.extend(list(preds))
        
        # Store ground truth labels
        self.y_true.extend(list(y.detach().cpu().numpy()))

    def loss(self):
        """
        Compute average loss across all accumulated batches.
        
        Returns:
            Mean loss as a float
        """
        return float(np.mean(self.losses))

    def acc(self):
        """
        Compute accuracy across all accumulated predictions.
        
        Returns:
            Accuracy as a float (0.0 to 1.0)
        """
        return float(accuracy_score(self.y_true, self.y_pred))

    def f1(self):
        """
        Compute macro-averaged F1 score across all accumulated predictions.
        
        Returns:
            Macro F1 score as a float (0.0 to 1.0)
        """
        return float(f1_score(self.y_true, self.y_pred, average="macro"))
