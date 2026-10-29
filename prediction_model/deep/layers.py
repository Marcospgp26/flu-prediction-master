"""Building blocks shared by the three networks: splitting DNA into codons and the output layer with copying."""
import torch
from torch import nn

CODON_VOCAB = 6 ** 3  # 3 positions x 6 possible values (padding, A, T, G, C, other) = 216 "words"


def codon_tokens(x):
    """Joins every three nucleotides into a single number, like words in a sentence.

    (batch, length) -> (batch, length / 3). For example, ATG = 1*36 + 2*6 + 3 = 51.
    A codon made only of padding becomes 0. Reading the gene in codons is the usual natural
    language processing tokenization applied to DNA, and each codon codes for one amino acid.
    """
    b, length = x.shape
    x = x.view(b, length // 3, 3)
    return x[..., 0] * 36 + x[..., 1] * 6 + x[..., 2]


class CopyHead(nn.Module):
    """Output layer: 4 scores (A, T, G, C) per position, plus a copy shortcut.

    The shortcut adds a learned weight to the score of the nucleotide that was already in the
    input (the idea behind CopyNet, from natural language processing). Since almost all of the
    gene stays the same from one season to the next, "keep the nucleotide" becomes the default,
    and the network can spend its learning on the positions that change.
    """

    def __init__(self, d_model, nucleotides_per_step=1):
        super().__init__()
        self.per_step = nucleotides_per_step          # 1 for the CNN (one nucleotide at a time), 3 for codon models
        self.proj = nn.Linear(d_model, nucleotides_per_step * 4)
        self.copy_weight = nn.Parameter(torch.zeros(()))  # starts at 0; training decides how much to copy

    def forward(self, h, x):
        b, steps, _ = h.shape
        logits = self.proj(h).view(b, steps * self.per_step, 4)
        # marks the input nucleotide at each position with a 1 (padding and N get no mark)
        valid = ((x >= 1) & (x <= 4)).unsqueeze(-1).to(logits.dtype)
        one_hot = torch.zeros_like(logits).scatter_(-1, (x - 1).clamp(0, 3).unsqueeze(-1), valid)
        return logits + self.copy_weight * one_hot
