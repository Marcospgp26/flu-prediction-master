# Encoding used by the original research.
#
# Each nucleotide becomes a digit (A=1, T=2, G=3, C=4) and every 15 digits become one number.
# For example, a 1701-nucleotide gene becomes a list of 114 numbers.
# Symbols other than A/T/G/C (N, R, Y, ...) are skipped, as in the original code.

CHUNK_SIZE = 15
DIGITS = {'A': '1', 'T': '2', 'G': '3', 'C': '4'}


class EncodingData:

    def to_digits(self, sequence):
        """'ATGC' -> '1234'"""
        return ''.join(DIGITS[n] for n in str(sequence).upper() if n in DIGITS)

    def chunk(self, digits):
        """'123412341234123412' -> [123412341234123.0, 412.0] (chunks of 15 digits)"""
        return [float(digits[i:i + CHUNK_SIZE]) for i in range(0, len(digits), CHUNK_SIZE)]

    def encoding(self, sequence):
        """DNA sequence -> list of numbers, as in the original research."""
        return self.chunk(self.to_digits(sequence))
