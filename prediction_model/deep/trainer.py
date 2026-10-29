import numpy as np
import torch
from torch import nn

from dataset import IGNORE_INDEX


def resolve_device(name='auto'):
    """'auto' uses the GPU if a compatible one is available; otherwise, the CPU."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


class NeuralSequenceModel:
    """Trains and runs a neural network through the same methods as the scikit-learn models
    (fit and predict), so it goes through exactly the same evaluation.

    X: (n, length) with one number per nucleotide.
    Y: (n, length) with the correct nucleotide from 0 to 3; IGNORE_INDEX marks positions that don't count.
    """

    def __init__(self, network_class, epochs=200, lr=1e-3, batch_size=8, weight_decay=1e-2,
                 device='auto', seed=0, verbose=False):
        self.network_class = network_class
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.weight_decay, self.seed, self.verbose = weight_decay, seed, verbose
        self.device = resolve_device(device)

    def fit(self, X, Y):
        torch.manual_seed(self.seed)  # the seed sets the initial weights and the batch order
        X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        Y = torch.as_tensor(np.asarray(Y), dtype=torch.long)
        self.network_ = self.network_class(X.shape[1]).to(self.device)

        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.network_.train()
            total = 0.0
            # every epoch, the pairs are shuffled and split into batches
            for batch in torch.randperm(len(X), generator=generator).split(self.batch_size):
                xb, yb = X[batch].to(self.device), Y[batch].to(self.device)
                loss = loss_fn(self.network_(xb).reshape(-1, 4), yb.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(batch)
            scheduler.step()
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f'    epoch {epoch + 1}/{self.epochs}  loss {total / len(X):.4f}')
        return self

    @torch.no_grad()
    def predict(self, X):
        self.network_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        out = [self.network_(xb.to(self.device)).argmax(-1).cpu() for xb in X.split(self.batch_size)]
        return torch.cat(out).numpy()
