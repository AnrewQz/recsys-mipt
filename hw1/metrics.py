import numpy as np
import pandas as pd


def _prepare_ranking(recommended, relevant, k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k должен быть положительным целым числом")
    top = list(recommended)[:k]
    if len(top) != len(set(top)):
        raise ValueError("Товары в рекомендациях не должны повторяться")
    return top, set(relevant)


def recall_at_k(recommended, relevant, k=10):
    top, relevant = _prepare_ranking(recommended, relevant, k)
    if not relevant:
        return np.nan
    return len(set(top) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    top, relevant = _prepare_ranking(recommended, relevant, k)
    if not relevant:
        return np.nan
    hits = np.array([item in relevant for item in top], dtype=float)
    dcg = (hits / np.log2(np.arange(len(top)) + 2)).sum()
    idcg = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return float(dcg / idcg)


def evaluate_ranking(recommendations, relevant_items, k=10):
    _prepare_ranking([], [], k)
    rows = []
    for user, relevant in relevant_items.items():
        relevant = set(relevant)
        if not relevant:
            continue
        recommended = list(recommendations.get(user, []))
        rows.append({
            "user_id": user,
            "relevant_items": len(relevant),
            f"Recall@{k}": recall_at_k(recommended, relevant, k),
            f"NDCG@{k}": ndcg_at_k(recommended, relevant, k),
        })
    if not rows:
        raise ValueError("Нет пользователей с релевантными товарами")
    per_user = pd.DataFrame(rows).set_index("user_id")
    metrics = per_user[[f"Recall@{k}", f"NDCG@{k}"]].mean().to_dict()
    return metrics, per_user
