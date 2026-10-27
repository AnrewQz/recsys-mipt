import numpy as np
from scipy import sparse

from .common import Recommender, top_indices


class UserKNN(Recommender):
    def __init__(self, n_neighbors=40, min_common=1, shrinkage=10, batch_size=256):
        self.n_neighbors = n_neighbors
        self.min_common = min_common
        self.shrinkage = shrinkage
        self.batch_size = batch_size

    def fit(self, data):
        self.data = data
        matrix = data.binary
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        rows, columns, values = [], [], []
        for begin in range(0, data.n_users, self.batch_size):
            end = min(begin + self.batch_size, data.n_users)
            overlap = (matrix[begin:end] @ matrix.T).toarray()
            denominator = norms[begin:end, None] * norms[None, :]
            cosine = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
            similarity = cosine * np.divide(overlap, overlap + self.shrinkage,
                                             out=np.zeros_like(overlap), where=overlap + self.shrinkage > 0)
            similarity[overlap < self.min_common] = 0
            similarity[np.arange(end - begin), np.arange(begin, end)] = 0
            for offset, scores in enumerate(similarity):
                selected = top_indices(scores, self.n_neighbors)
                selected = selected[scores[selected] > 0]
                weights = scores[selected]
                if len(selected):
                    weights = weights / weights.sum()
                rows.extend([begin + offset] * len(selected))
                columns.extend(selected)
                values.extend(weights)
        self.similarity = sparse.csr_matrix((values, (rows, columns)), shape=(data.n_users, data.n_users))
        return self

    def score_users(self, user_indices):
        return (self.similarity[user_indices] @ self.data.binary).toarray()
