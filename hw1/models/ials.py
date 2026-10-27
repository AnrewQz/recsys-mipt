import numpy as np

from .common import Recommender


def update_rows(confidence, preference, fixed, reg, batch_size=512):
    confidence = confidence.tocsr()
    preference = preference.tocsr()
    n_factors = fixed.shape[1]
    base = fixed.T.dot(fixed) + reg * np.eye(n_factors)
    result = np.empty((confidence.shape[0], n_factors))
    for begin in range(0, confidence.shape[0], batch_size):
        end = min(begin + batch_size, confidence.shape[0])
        A = np.broadcast_to(base, (end - begin, n_factors, n_factors)).copy()
        b = np.zeros((end - begin, n_factors))
        for offset, row in enumerate(range(begin, end)):
            start, stop = confidence.indptr[row:row + 2]
            indices = confidence.indices[start:stop]
            c = confidence.data[start:stop]
            p = preference.data[start:stop]
            local = fixed[indices]
            A[offset] += local.T.dot((c - 1)[:, None] * local)
            b[offset] = local.T.dot(c * p)
        result[begin:end] = np.linalg.solve(A, b[..., None])[..., 0]
    return result


def als_loss(confidence, preference, users, items, reg):
    coo = confidence.tocoo()
    loss = np.sum(users.T.dot(users) * items.T.dot(items))
    for begin in range(0, coo.nnz, 65536):
        part = slice(begin, begin + 65536)
        predictions = np.einsum("ij,ij->i", users[coo.row[part]], items[coo.col[part]])
        loss += np.sum(coo.data[part] * (preference.data[part] - predictions) ** 2 - predictions ** 2)
    return float(loss + reg * ((users ** 2).sum() + (items ** 2).sum()))


class IALS(Recommender):
    def __init__(self, n_factors=16, n_iters=8, alpha=20, reg=1, random_state=42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.alpha = alpha
        self.reg = reg
        self.random_state = random_state

    def target(self, data):
        return data.binary.copy()

    def confidence(self, data):
        confidence = data.counts.copy()
        confidence.data = 1 + self.alpha * np.log1p(confidence.data)
        return confidence

    def fit(self, data):
        self.data = data
        confidence = self.confidence(data)
        preference = self.target(data)
        confidence_t = confidence.T.tocsr()
        preference_t = preference.T.tocsr()
        rng = np.random.default_rng(self.random_state)
        self.user_factors = np.zeros((data.n_users, self.n_factors))
        self.item_factors = .05 * rng.standard_normal((data.n_items, self.n_factors))
        self.loss_history = []
        for _ in range(self.n_iters):
            self.user_factors = update_rows(confidence, preference, self.item_factors, self.reg)
            self.item_factors = update_rows(confidence_t, preference_t, self.user_factors, self.reg)
            self.loss_history.append(als_loss(confidence, preference, self.user_factors, self.item_factors, self.reg))
        return self

    def score_users(self, user_indices):
        return self.user_factors[user_indices].dot(self.item_factors.T)
