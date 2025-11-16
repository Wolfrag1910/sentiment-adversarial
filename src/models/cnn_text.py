"""
Text CNN model for sentiment classification.

Implements a Convolutional Neural Network architecture for text classification
using multiple filter sizes to capture different n-gram patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification.
    
    Architecture:
        1. Embedding layer: Maps token indices to dense vectors
        2. Multiple parallel convolutional layers with different filter sizes
        3. Max-pooling over time for each convolutional output
        4. Concatenate pooled features
        5. Dropout for regularization
        6. Fully connected layer for classification
    """
    
    def __init__(self, vocab_size, emb_dim, num_classes, filter_sizes, 
                 num_filters, dropout, pad_idx):
        """
        Initialize the Text CNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            emb_dim: Dimension of word embeddings
            num_classes: Number of output classes (2 for binary sentiment)
            filter_sizes: List of convolutional filter sizes (e.g., [3, 4, 5] for trigrams, 4-grams, 5-grams)
            num_filters: Number of filters for each filter size
            dropout: Dropout probability for regularization
            pad_idx: Index of the padding token (embeddings for this index won't be updated)
        """
        super().__init__()
        
        # Embedding layer with padding index
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        # Create multiple convolutional layers with different filter sizes
        # Each conv layer captures different n-gram patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k) for k in filter_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        # Input size is num_filters * number of different filter sizes
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of token indices, shape (batch_size, seq_len)
            
        Returns:
            Logits for each class, shape (batch_size, num_classes)
        """
        # Embed tokens: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        e = self.emb(x)
        
        # Transpose for conv1d: (batch_size, seq_len, emb_dim) -> (batch_size, emb_dim, seq_len)
        # Conv1d expects (batch, channels, length) format
        e = e.transpose(1, 2)
        
        # Apply each convolutional filter
        # Each conv output has shape (batch_size, num_filters, seq_len - filter_size + 1)
        convs = [F.relu(c(e)) for c in self.convs]
        
        # Max-pool over time for each convolutional output
        # This extracts the most important feature from each filter
        # Each pool output has shape (batch_size, num_filters)
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        
        # Concatenate all pooled features
        # Shape: (batch_size, num_filters * len(filter_sizes))
        h = torch.cat(pools, dim=1)
        
        # Apply dropout for regularization
        h = self.dropout(h)
        
        # Final classification layer
        # Shape: (batch_size, num_classes)
        return self.fc(h)
