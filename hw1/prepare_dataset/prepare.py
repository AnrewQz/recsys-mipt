def filter_items(orders, min_interactions=2):
    counts = orders["nm_id"].value_counts()
    items = counts[counts >= min_interactions].index
    return orders.loc[orders["nm_id"].isin(items)].reset_index(drop=True)


def filter_users(orders, min_orders=2):
    counts = orders["user_id"].value_counts()
    users = counts[counts >= min_orders].index
    return orders.loc[orders["user_id"].isin(users)].reset_index(drop=True)
