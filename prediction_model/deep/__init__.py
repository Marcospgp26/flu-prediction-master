"""The new neural networks (they need PyTorch).

They all take the old sequence as one number per nucleotide and return, for each position,
a score for A, T, G and C. The nucleotide with the highest score is the prediction.
"""
from .cnn import ResidualCNN
from .lstm import CodonBiLSTM
from .trainer import NeuralSequenceModel, resolve_device
from .transformer import CodonTransformer

# command-line name: (name in the results, network class, learning rate)
DEEP_MODELS = {
    'cnn': ('CNN (dilated residual)', ResidualCNN, 1e-3),
    'transformer': ('Transformer (codon tokens)', CodonTransformer, 5e-4),
    'lstm': ('BiLSTM (codon tokens)', CodonBiLSTM, 1e-3),
}
