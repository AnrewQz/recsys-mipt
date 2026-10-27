import numpy as np
import pandas as pd
from scipy import sparse

from .ials import IALS


class CustomIALS(IALS):
    def __init__(self, n_factors=16, n_iters=8, alpha=20, reg=1, window_weeks=4, random_state=42):
        super().__init__(n_factors, n_iters, alpha, reg, random_state)
        if window_weeks <= 0:
            raise ValueError("window_weeks должен быть положительным")
        self.window_weeks = window_weeks

    def counts_in_window(self, data):
        cutoff = data.history["timestamp"].max() - pd.Timedelta(weeks=self.window_weeks)
        recent = data.history.loc[data.history["timestamp"] >= cutoff]
        users = data.user_ids.get_indexer(recent["user_id"])
        items = data.item_ids.get_indexer(recent["nm_id"])
        counts = sparse.csr_matrix((np.ones(len(recent)), (users, items)), shape=data.binary.shape)
        stored_users = np.repeat(np.arange(data.n_users), np.diff(data.counts.indptr))
        return np.asarray(counts[stored_users, data.counts.indices]).ravel()

    def target(self, data):
        preference = data.binary.copy()
        preference.data = (self.counts_in_window(data) > 0).astype(float)
        return preference

    def confidence(self, data):
        confidence = data.counts.copy()
        confidence.data = 1 + self.alpha * np.log1p(self.counts_in_window(data))
        return confidence
