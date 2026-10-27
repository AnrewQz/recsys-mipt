import numpy as np
import pandas as pd
from scipy import sparse


def top_indices(scores, k):
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=int)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    better = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(better)]
    selected = np.concatenate([better, tied])
    return selected[np.argsort(-scores[selected], kind="stable")]


class InteractionData:
    def __init__(self, history, metadata):
        if history.empty:
            raise ValueError("История взаимодействий пуста")
        self.history = history[["user_id", "nm_id", "timestamp"]].copy()
        self.user_ids = pd.Index(np.sort(history["user_id"].unique()))
        popularity = history.groupby("nm_id")["user_id"].nunique().sort_values(ascending=False, kind="stable")
        self.item_ids = pd.Index(popularity.index)
        self.popularity = popularity.to_numpy(dtype=float)
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)
        pairs = history.groupby(["user_id", "nm_id"]).agg(
            count=("timestamp", "size")
        ).reset_index()
        users = self.user_ids.get_indexer(pairs["user_id"])
        items = self.item_ids.get_indexer(pairs["nm_id"])
        shape = (self.n_users, self.n_items)
        self.counts = sparse.csr_matrix((pairs["count"].to_numpy(dtype=float), (users, items)), shape=shape)
        self.binary = self.counts.copy()
        self.binary.data[:] = 1
        self.metadata = metadata.loc[metadata["nm_id"].isin(self.item_ids)].set_index("nm_id").reindex(self.item_ids)
        if self.metadata[["category_id", "subcategory_id"]].isna().any().any():
            raise ValueError("Нет метаданных для товаров истории")


class Recommender:
    def recommend(self, user_ids, k=10, batch_size=64):
        if k < 1:
            raise ValueError("k должен быть положительным")
        users = list(user_ids)
        positions = self.data.user_ids.get_indexer(users)
        fallback = self.data.item_ids[:k].tolist()
        result = {}
        for begin in range(0, len(users), batch_size):
            batch_users = users[begin:begin + batch_size]
            batch_positions = positions[begin:begin + batch_size]
            known = batch_positions >= 0
            scores = self.score_users(batch_positions[known]) if known.any() else np.empty((0, self.data.n_items))
            if not np.isfinite(scores).all():
                raise ValueError("Модель вернула нечисловые оценки")
            row = 0
            for user, is_known in zip(batch_users, known):
                if is_known:
                    selected = top_indices(scores[row], k)
                    result[user] = self.data.item_ids[selected].tolist()
                    row += 1
                else:
                    result[user] = fallback.copy()
        return result
