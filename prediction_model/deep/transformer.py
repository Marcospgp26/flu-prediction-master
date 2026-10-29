import torch
from torch import nn

from .layers import CODON_VOCAB, CopyHead, codon_tokens


class CodonTransformer(nn.Module):
    """Transformer that reads the gene in codons (3-letter words).

    With attention, each codon can look at every other codon in the gene at once, which lets
    the model learn mutations that tend to happen together, even far apart.
    """

    def __init__(self, seq_len, d_model=128, n_heads=4, n_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(CODON_VOCAB, d_model, padding_idx=0)
        self.position = nn.Embedding(seq_len // 3, d_model)    # the network learns where each codon is
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = CopyHead(d_model, nucleotides_per_step=3)

    def forward(self, x):
        tokens = codon_tokens(x)
        positions = torch.arange(tokens.shape[1], device=x.device)
        h = self.embed(tokens) + self.position(positions)
        h = self.encoder(h, src_key_padding_mask=tokens == 0)  # ignores the padding codons
        return self.head(self.norm(h), x)
