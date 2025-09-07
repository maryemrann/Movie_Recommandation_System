
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Loading utilities
# -----------------------------
RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]

def load_ratings(path):
    return pd.read_csv(path, sep="\t", names=RATING_COLS, engine="python")

def load_movies(u_item_path):
    # u.item columns: item_id|title|release_date|video_release_date|IMDb_URL|genres...
    cols = ["item_id", "title"] + list(range(22))
    return pd.read_csv(u_item_path, sep="|", names=cols, usecols=[0,1], encoding="latin-1", engine="python")

def get_train_test(data_dir, use_predefined_split=False, split_seed=42, test_size=0.2):
    base_path = os.path.join(data_dir, "u1.base")
    test_path = os.path.join(data_dir, "u1.test")
    ratings_path = os.path.join(data_dir, "u.data")

    if use_predefined_split and os.path.exists(base_path) and os.path.exists(test_path):
        train = load_ratings(base_path)
        test  = load_ratings(test_path)
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        ratings = load_ratings(ratings_path)
        train, test = train_test_split(ratings, test_size=test_size, random_state=split_seed, stratify=ratings["user_id"])
    return train, test

def build_matrices(train_df, data_dir):
    movies = load_movies(os.path.join(data_dir, "u.item"))
    # Pivot on item_id (safer than title), we'll map to titles for display later.
    user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    # Ensure all known items exist as columns (even if absent in train split)
    all_items = movies["item_id"].unique()
    missing = [i for i in all_items if i not in user_item.columns]
    if missing:
        for i in missing: user_item[i] = 0
        user_item = user_item.reindex(sorted(user_item.columns), axis=1)

    movies_map = movies.set_index("item_id")["title"].to_dict()
    return user_item.sort_index(), movies_map

# -----------------------------
# Similarity computations
# -----------------------------
def user_similarity_matrix(user_item_matrix):
    sim = cosine_similarity(user_item_matrix.values)
    # zero self-similarity for convenience when picking neighbors
    np.fill_diagonal(sim, 0.0)
    return sim

def item_similarity_matrix(user_item_matrix):
    sim = cosine_similarity(user_item_matrix.values.T)
    np.fill_diagonal(sim, 0.0)
    return sim

# -----------------------------
# Recommendation functions
# -----------------------------
def recommend_user_based(user_item_matrix, user_sim, user_id, k=10, neighbors=30):
    """Return top-k (item_id, score) for a given user using user-based CF."""
    users = user_item_matrix.index.to_numpy()
    items = user_item_matrix.columns.to_numpy()
    user_to_idx = {u:i for i,u in enumerate(users)}
    uidx = user_to_idx.get(user_id, None)
    if uidx is None:
        raise ValueError(f"user_id {user_id} not found in training data.")

    R = user_item_matrix.values
    # Similar users (top-N by similarity)
    sim_vec = user_sim[uidx]  # shape (n_users,)
    # Pick top neighbors
    if neighbors >= len(sim_vec):
        nbr_idx = np.argsort(sim_vec)[::-1]  # all users sorted
    else:
        nbr_idx = np.argpartition(sim_vec, -neighbors)[-neighbors:]
        nbr_idx = nbr_idx[np.argsort(sim_vec[nbr_idx])[::-1]]  # sort descending

    sims = sim_vec[nbr_idx]  # shape (N,)
    R_nbrs = R[nbr_idx]      # shape (N, n_items)

    # Weighted sum only over observed ratings
    mask = (R_nbrs > 0).astype(float)
    numer = (sims[:, None] * R_nbrs * mask).sum(axis=0)  # (n_items,)
    denom = (np.abs(sims)[:, None] * mask).sum(axis=0)   # (n_items,)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)

    # Remove items already seen by the user
    seen = R[uidx] > 0
    scores[seen] = -np.inf

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return list(zip(items[top_idx], scores[top_idx]))

def recommend_item_based(user_item_matrix, item_sim, user_id, k=10):
    """Return top-k (item_id, score) for a given user using item-based CF."""
    users = user_item_matrix.index.to_numpy()
    items = user_item_matrix.columns.to_numpy()
    user_to_idx = {u:i for i,u in enumerate(users)}
    uidx = user_to_idx.get(user_id, None)
    if uidx is None:
        raise ValueError(f"user_id {user_id} not found in training data.")

    R = user_item_matrix.values
    r_u = R[uidx]  # (n_items,)
    rated_mask = (r_u > 0).astype(float)
    if rated_mask.sum() == 0:
        return []

    # Numerator: similarity to user's rated items weighted by their ratings
    numer = item_sim @ (r_u * rated_mask)  # (n_items,)
    denom = (np.abs(item_sim) @ rated_mask)  # (n_items,)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)

    # Don't recommend already seen
    scores[rated_mask.astype(bool)] = -np.inf

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return list(zip(items[top_idx], scores[top_idx]))

def recommend_svd(user_item_matrix, user_id, k=10, n_components=50, random_state=42):
    """TruncatedSVD reconstruction; return top-k (item_id, score)."""
    users = user_item_matrix.index.to_numpy()
    items = user_item_matrix.columns.to_numpy()
    user_to_idx = {u:i for i,u in enumerate(users)}
    uidx = user_to_idx.get(user_id, None)
    if uidx is None:
        raise ValueError(f"user_id {user_id} not found in training data.")

    R = user_item_matrix.values.astype(float)
    svd = TruncatedSVD(n_components=min(n_components, min(R.shape)-1), random_state=random_state)
    U = svd.fit_transform(R)           # (n_users, k)
    VT = svd.components_               # (k, n_items)
    R_hat = U @ VT                     # (n_users, n_items)
    scores = R_hat[uidx].copy()

    # Remove already seen
    seen = R[uidx] > 0
    scores[seen] = -np.inf

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return list(zip(items[top_idx], scores[top_idx]))

# -----------------------------
# Evaluation (Precision@K)
# -----------------------------
def precision_at_k_all_users(test_df, recommend_fn, k=10, movies_present=None):
    """
    recommend_fn: function(user_id, k) -> list of (item_id, score)
    Returns mean precision@k over users that have at least one relevant item (rating >= 4) in test.
    """
    precisions = []
    for uid, group in test_df.groupby("user_id"):
        relevant = set(group.loc[group["rating"] >= 4, "item_id"])
        if not relevant:
            continue
        # Optionally restrict to items present in training columns
        if movies_present is not None:
            relevant = {i for i in relevant if i in movies_present}
            if not relevant:
                continue
        try:
            recs = recommend_fn(uid, k)
        except Exception:
            continue
        rec_items = [i for i, _ in recs]
        hits = sum(1 for i in rec_items if i in relevant)
        precisions.append(hits / float(k))
    return float(np.mean(precisions)) if precisions else float("nan")

# -----------------------------
# Pretty printing
# -----------------------------
def show_recs(recs, id_to_title, header="Top Recommendations"):
    print("\n" + header)
    print("-" * len(header))
    for rank, (iid, score) in enumerate(recs, start=1):
        title = id_to_title.get(int(iid), f"Item {iid}")
        print(f"{rank:2d}. {title:50s}  score={score:.4f}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="MovieLens 100K Recommender + Precision@K")
    ap.add_argument("--data_dir", required=True, help="Path to ml-100k folder containing u.data, u.item, u1.base, u1.test, ...")
    ap.add_argument("--method", choices=["user", "item", "svd"], default="user", help="Recommender type")
    ap.add_argument("--user_id", type=int, default=1, help="User ID to recommend for")
    ap.add_argument("--k", type=int, default=10, help="Top-K recommendations")
    ap.add_argument("--neighbors", type=int, default=30, help="Neighbors for user-based CF")
    ap.add_argument("--use_predefined_split", action="store_true", help="Use u1.base/u1.test split shipped with MovieLens")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test size if not using predefined split")
    args = ap.parse_args()

    # Train/Test
    train, test = get_train_test(args.data_dir, use_predefined_split=args.use_predefined_split, test_size=args.test_size)
    ui_matrix, id_to_title = build_matrices(train, args.data_dir)

    # Build the chosen recommender and evaluate
    if args.method == "user":
        u_sim = user_similarity_matrix(ui_matrix)
        rec_fn = lambda uid, k=args.k: recommend_user_based(ui_matrix, u_sim, uid, k=k, neighbors=args.neighbors)
        # Demo for chosen user
        recs = rec_fn(args.user_id, args.k)
        show_recs(recs, id_to_title, header=f"User-based CF | User {args.user_id} | Top-{args.k}")
        # Evaluate
        p_at_k = precision_at_k_all_users(test, rec_fn, k=args.k, movies_present=set(ui_matrix.columns))
        print(f"\nPrecision@{args.k} (user-based): {p_at_k:.4f}")
    elif args.method == "item":
        i_sim = item_similarity_matrix(ui_matrix)
        rec_fn = lambda uid, k=args.k: recommend_item_based(ui_matrix, i_sim, uid, k=k)
        recs = rec_fn(args.user_id, args.k)
        show_recs(recs, id_to_title, header=f"Item-based CF | User {args.user_id} | Top-{args.k}")
        p_at_k = precision_at_k_all_users(test, rec_fn, k=args.k, movies_present=set(ui_matrix.columns))
        print(f"\nPrecision@{args.k} (item-based): {p_at_k:.4f}")
    else:  # svd
        rec_fn = lambda uid, k=args.k: recommend_svd(ui_matrix, uid, k=k)
        recs = rec_fn(args.user_id, args.k)
        show_recs(recs, id_to_title, header=f"SVD MF | User {args.user_id} | Top-{args.k}")
        p_at_k = precision_at_k_all_users(test, rec_fn, k=args.k, movies_present=set(ui_matrix.columns))
        print(f"\nPrecision@{args.k} (svd): {p_at_k:.4f}")

if __name__ == "__main__":
    main()
 