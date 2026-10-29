import torch
from torch import nn

from .layers import CopyHead


class ResidualConvBlock(nn.Module):
    """Two convolutions with a shortcut (x + y): if there is nothing to change, the block just passes the input on."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2   # keeps the sequence length the same
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(torch.relu(self.norm1(self.conv1(x))))
        y = self.norm2(self.conv2(y))
        return torch.relu(x + y)


class ResidualCNN(nn.Module):
    """1D convolutional network that reads the DNA one nucleotide at a time.

    Filters of size 9 slide along the sequence. With dilations 1, 2, 4 and 8, each block
    sees further than the previous one, so in the end each position "sees" about 240
    nucleotides on either side.
    """

    def __init__(self, seq_len, channels=128, kernel_size=9, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(6, channels, padding_idx=0)
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels, kernel_size, d, dropout) for d in dilations])
        self.head = CopyHead(channels, nucleotides_per_step=1)

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)       # Conv1d expects (batch, channels, length)
        h = self.blocks(h).transpose(1, 2)
        return self.head(h, x)
