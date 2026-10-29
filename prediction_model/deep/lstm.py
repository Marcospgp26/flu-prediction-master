from torch import nn

from .layers import CODON_VOCAB, CopyHead, codon_tokens


class CodonBiLSTM(nn.Module):
    """Bidirectional LSTM over codons, the classic natural language processing model.

    It reads the gene in both directions (start to end and end to start), keeping a memory
    of what it has read so far.
    """

    def __init__(self, seq_len, d_model=128, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(CODON_VOCAB, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, hidden, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = CopyHead(2 * hidden, nucleotides_per_step=3)   # 2x: one half for each direction

    def forward(self, x):
        h, _ = self.lstm(self.embed(codon_tokens(x)))
        return self.head(self.dropout(h), x)
