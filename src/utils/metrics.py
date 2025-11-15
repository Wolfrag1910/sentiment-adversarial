import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class Accumulator:
    def __init__(self):
        self.y_true, self.y_pred, self.losses = [], [], []
    def add_batch(self, logits, y, loss):
        self.losses.append(loss)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        self.y_pred.extend(list(preds))
        self.y_true.extend(list(y.detach().cpu().numpy()))
    def loss(self): return float(np.mean(self.losses))
    def acc(self):  return float(accuracy_score(self.y_true, self.y_pred))
    def f1(self):   return float(f1_score(self.y_true, self.y_pred, average="macro"))
