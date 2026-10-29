import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.naive_bayes import GaussianNB

NAME = 'Naive Bayes'


class ChunkNaiveBayes(RegressorMixin, BaseEstimator):
    """Naive Bayes used as in the original research.

    Each 15-digit chunk of the old sequence is one sample, and its "class" is the chunk at
    the same position in the new sequence.

    Current scikit-learn rejects decimal class labels ("Unknown label type: continuous"),
    which is why the original code no longer ran. The fix is to convert the chunks to
    integers. 15-digit numbers fit exactly in both float64 and int64, so nothing is lost.
    """

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        self.model_ = GaussianNB().fit(X.reshape(-1, 1), Y.reshape(-1).astype(np.int64))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X.reshape(-1, 1)).astype(np.float64).reshape(X.shape[0], -1)


def build_model(seed=None):
    return ChunkNaiveBayes()
