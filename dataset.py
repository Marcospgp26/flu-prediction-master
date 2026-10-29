"""Reads the FASTA files and prepares the data for the models.

Each experiment pairs sequence i of an "old" file with sequence i of a "new" file, the same
pairing as the original research. The models learn: old -> new.

From the same pairs we build two representations:
  * chunks: the original research's encoding (15 digits per number), used by the scikit-learn models;
  * tokens: one integer per nucleotide, used by the neural networks.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO

from encoding_data import EncodingData

ROOT = Path(__file__).resolve().parent

# name: (input file = older season, target file = newer season).
# These are the same files as in the original research.
# Note: for H1N1-HA the original research used 2021 as input and 2019-2020 as target.
DATASETS = {
    'h1n1-ha': ('flu-data/H1N1/HA/H1N1-HA24-2021.fasta', 'flu-data/H1N1/HA/H1N1-HA24-before2021.fasta'),
    'h1n1-na': ('flu-data/H1N1/NA/H1N1-NA24-2019And2020.fasta', 'flu-data/H1N1/NA/H1N1-NA24-2021.fasta'),
    'h3n2-ha': ('flu-data/H3N2/HA/H3N2-HA50-2021.fasta', 'flu-data/H3N2/HA/H3N2-HA50-2022.fasta'),
    'h3n2-na': ('flu-data/H3N2/NA/H3N2-NA50-2021.fasta', 'flu-data/H3N2/NA/H3N2-NA50-2022.fasta'),
}

# Token ids for the neural networks. 1 to 4 are the same digits as in the original research.
PAD_TOKEN, OTHER_TOKEN = 0, 5           # 0 = padding at the end of a sequence, 5 = N, R, Y, ...
NT_TOKEN = {'A': 1, 'T': 2, 'G': 3, 'C': 4}
IGNORE_INDEX = -100                     # target positions that are not scored (padding, N, ...)

STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def orf_length(seq, start):
    """Number of codons from `start` until a stop codon (or the end of the sequence)."""
    codons = 0
    for p in range(start, len(seq) - 2, 3):
        if seq[p:p + 3] in STOP_CODONS:
            break
        codons += 1
    return codons


def trim_to_start_codon(seq, search_window=120):
    """Drops the bases before the gene's start codon (ATG).

    In some files the sequences include a few bases before the ATG and in others they don't,
    so position i is not the same place in the gene for every sequence. We pick the ATG near
    the beginning that gives the longest open reading frame (ORF).
    """
    candidates = [p for p in range(min(search_window, len(seq) - 2)) if seq[p:p + 3] == 'ATG']
    if not candidates:
        return seq
    best = max(candidates, key=lambda p: (orf_length(seq, p), -p))
    return seq[best:]


def read_fasta(relative_path):
    path = ROOT / relative_path
    return [str(record.seq).upper() for record in SeqIO.parse(str(path), 'fasta')]


def pad_matrix(rows, width):
    """Puts lists of different lengths into one matrix, filling the gaps with zeros."""
    matrix = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        matrix[i, :len(row)] = row
    return matrix


@dataclass
class PairedData:
    """A set of (old sequence, new sequence) pairs, ready for every model."""
    name: str
    inputs: list
    targets: list

    def __post_init__(self):
        encoder = EncodingData()
        n = len(self.inputs)

        # 1) The original research's encoding, for the scikit-learn models
        x_chunks = [encoder.encoding(s) for s in self.inputs]
        y_chunks = [encoder.encoding(s) for s in self.targets]
        self.chunk_width = max(len(v) for v in x_chunks + y_chunks)
        self.X = pad_matrix(x_chunks, self.chunk_width)
        self.Y = pad_matrix(y_chunks, self.chunk_width)
        self.target_digits = [encoder.to_digits(s) for s in self.targets]

        # 2) One token per nucleotide, for the neural networks. The length is a multiple of 3
        #    so the codon-based models can split the sequence into 3-letter words.
        longest = max(len(s) for s in self.inputs + self.targets)
        self.length = longest + (-longest) % 3
        self.X_tokens = np.full((n, self.length), PAD_TOKEN, dtype=np.int64)
        self.Y_classes = np.full((n, self.length), IGNORE_INDEX, dtype=np.int64)
        for i, (old, new) in enumerate(zip(self.inputs, self.targets)):
            self.X_tokens[i, :len(old)] = [NT_TOKEN.get(c, OTHER_TOKEN) for c in old]
            self.Y_classes[i, :len(new)] = [NT_TOKEN[c] - 1 if c in NT_TOKEN else IGNORE_INDEX for c in new]

        # 3) Positions where the new sequence differs from the old one (the "mutations").
        #    They use the same indexing as target_digits. Copying last season gets all of them wrong.
        self.mutation_sites = []
        for old, new in zip(self.inputs, self.targets):
            sites, k = [], 0
            for p, c in enumerate(new):
                if c not in NT_TOKEN:
                    continue
                if p < len(old) and old[p] in NT_TOKEN and old[p] != c:
                    sites.append(k)
                k += 1
            self.mutation_sites.append(np.array(sites, dtype=np.int64))

    def __len__(self):
        return len(self.inputs)


def load_dataset(name, align=False):
    """Loads one of the DATASETS. With align=True, every sequence is trimmed at its start codon."""
    input_file, target_file = DATASETS[name]
    inputs, targets = read_fasta(input_file), read_fasta(target_file)
    if align:
        inputs = [trim_to_start_codon(s) for s in inputs]
        targets = [trim_to_start_codon(s) for s in targets]
    if len(inputs) != len(targets):
        n = min(len(inputs), len(targets))
        print(f'Warning: {input_file} has {len(inputs)} sequences and {target_file} has {len(targets)}; '
              f'using only the first {n} pairs.')
        inputs, targets = inputs[:n], targets[:n]
    return PairedData(name, inputs, targets)
