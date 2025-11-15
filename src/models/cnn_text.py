import torch, torch.nn as nn, torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, filter_sizes, num_filters, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):             # x: (B, T)
        e = self.emb(x).transpose(1,2) # (B, E, T)
        convs = [F.relu(c(e)) for c in self.convs]          # [(B, F, T-k+1), ...]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]  # [(B, F), ...]
        h = torch.cat(pools, dim=1)
        h = self.dropout(h)
        return self.fc(h)             # (B, C)