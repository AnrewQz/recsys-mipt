import numpy as np
from scipy import linalg

from .common import Recommender


class EASE(Recommender):
    def __init__(self, reg=100, item_batch_size=2048):
        self.reg = reg
        self.item_batch_size = item_batch_size

    def fit(self, data):
        if self.reg <= 0:
            raise ValueError("reg должен быть положительным")
        self.data = data
        self.history = data.binary
        gram = (self.history @ self.history.T).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        self.user_inverse = linalg.cho_solve(factor, np.eye(data.n_users), overwrite_b=True, check_finite=False)
        self.precision_diagonal = np.empty(data.n_items)
        for begin in range(0, data.n_items, self.item_batch_size):
            end = min(begin + self.item_batch_size, data.n_items)
            block = self.history[:, begin:end].T.tocsr()
            projected = block @ self.user_inverse
            leverage = np.asarray(block.multiply(projected).sum(axis=1)).ravel()
            self.precision_diagonal[begin:end] = (1 - leverage) / self.reg
        if np.any(self.precision_diagonal <= 0):
            raise ValueError("Некорректная диагональ обратной матрицы EASE")
        return self

    def score_users(self, user_indices):
        projected = (self.history.T @ self.user_inverse[user_indices].T).T
        return self.history[user_indices].toarray() - projected / self.precision_diagonal
