import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from .common import Recommender


class FeatureMap:
    def __init__(self, data):
        self.user_features = sparse.eye(data.n_users, format="csr")
        side = []
        for column in ["category_id", "subcategory_id"]:
            codes, values = pd.factorize(data.metadata[column], sort=True)
            side.append(sparse.csr_matrix((np.ones(data.n_items) / np.sqrt(2),
                                           (np.arange(data.n_items), codes)),
                                          shape=(data.n_items, len(values))))
        self.item_features = sparse.hstack([sparse.eye(data.n_items, format="csr"), *side], format="csr")
        self.user_dim = self.user_features.shape[1]
        self.n_features = self.user_dim + self.item_features.shape[1]

    def pairs(self, users, items):
        return sparse.hstack([self.user_features[users], self.item_features[items]], format="csr")


def sample_negatives(rng, users, history):
    if np.any(np.diff(history.indptr)[users] == history.shape[1]):
        raise ValueError("У пользователя нет ненаблюдаемых товаров")
    negatives = rng.integers(history.shape[1], size=len(users))
    collisions = np.asarray(history[users, negatives]).ravel() > 0
    while np.any(collisions):
        negatives[collisions] = rng.integers(history.shape[1], size=int(collisions.sum()))
        collisions = np.asarray(history[users, negatives]).ravel() > 0
    return negatives


class FactorizationMachine(Recommender):
    def __init__(self, n_factors=16, epochs=5, lr=.01, reg=1e-5, batch_size=8192, random_state=42):
        self.n_factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.random_state = random_state

    def initialize(self, features):
        self.features = features
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(features.n_features)
        self.V = .05 * rng.standard_normal((features.n_features, self.n_factors))
        self.m_w, self.s_w = np.zeros_like(self.w), np.zeros_like(self.w)
        self.m_v, self.s_v = np.zeros_like(self.V), np.zeros_like(self.V)
        self.step = 0

    def forward(self, X):
        squared = X.multiply(X)
        projected = X @ self.V
        scores = np.asarray(X @ self.w).ravel() + .5 * np.sum(
            projected ** 2 - squared @ (self.V ** 2), axis=1
        )
        return scores, projected, squared

    def gradients(self, X, projected, squared, errors):
        grad_w = np.asarray(X.T @ errors).ravel()
        grad_v = X.T @ (errors[:, None] * projected) - np.asarray(squared.T @ errors).ravel()[:, None] * self.V
        return grad_w, np.asarray(grad_v)

    def bpr_gradients(self, X_pos, X_neg):
        pos, projected_pos, squared_pos = self.forward(X_pos)
        neg, projected_neg, squared_neg = self.forward(X_neg)
        delta = pos - neg
        weight = expit(-delta)
        gw_pos, gv_pos = self.gradients(X_pos, projected_pos, squared_pos, -weight)
        gw_neg, gv_neg = self.gradients(X_neg, projected_neg, squared_neg, weight)
        size = X_pos.shape[0]
        loss = float(np.logaddexp(0, -delta).mean())
        loss += .5 * self.reg * (np.sum(self.w ** 2) + np.sum(self.V ** 2))
        return loss, (gw_pos + gw_neg) / size + self.reg * self.w, (gv_pos + gv_neg) / size + self.reg * self.V

    def update(self, grad_w, grad_v):
        self.step += 1
        for parameter, first, second, gradient in [
            (self.w, self.m_w, self.s_w, grad_w),
            (self.V, self.m_v, self.s_v, grad_v),
        ]:
            first *= .9
            first += .1 * gradient
            second *= .999
            second += .001 * gradient ** 2
            parameter -= self.lr * (first / (1 - .9 ** self.step)) / (
                np.sqrt(second / (1 - .999 ** self.step)) + 1e-8
            )

    def fit(self, data):
        self.data = data
        self.initialize(FeatureMap(data))
        positive_users, positive_items = data.binary.nonzero()
        eligible = np.diff(data.binary.indptr)[positive_users] < data.n_items
        positive_users, positive_items = positive_users[eligible], positive_items[eligible]
        if len(positive_users) == 0:
            raise ValueError("Нет пар для обучения FM")
        rng = np.random.default_rng(self.random_state)
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(positive_users))
            total_loss = 0.
            for begin in range(0, len(order), self.batch_size):
                batch = order[begin:begin + self.batch_size]
                users, items = positive_users[batch], positive_items[batch]
                negatives = sample_negatives(rng, users, data.binary)
                positive = self.features.pairs(users, items)
                negative = self.features.pairs(users, negatives)
                loss, grad_w, grad_v = self.bpr_gradients(positive, negative)
                self.update(grad_w, grad_v)
                total_loss += loss * len(batch)
            self.loss_history.append(total_loss / len(order))
        boundary = self.features.user_dim
        U, I = self.features.user_features, self.features.item_features
        Vu, Vi = self.V[:boundary], self.V[boundary:]
        self.user_factors, self.item_factors = U @ Vu, I @ Vi
        self.user_bias = U @ self.w[:boundary] + .5 * np.sum(
            self.user_factors ** 2 - U.multiply(U) @ (Vu ** 2), axis=1)
        self.item_bias = I @ self.w[boundary:] + .5 * np.sum(
            self.item_factors ** 2 - I.multiply(I) @ (Vi ** 2), axis=1)
        return self

    def score_users(self, user_indices):
        return self.user_bias[user_indices, None] + self.item_bias[None, :] + self.user_factors[user_indices].dot(self.item_factors.T)
