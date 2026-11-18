"""
StyleSense recommendation engine.
Loads the saved sparse interaction matrix and recommendation artifacts.
If the KNN artifact cannot be loaded, popularity-based recommendations remain available.
"""
from pathlib import Path
import warnings
import joblib
import pandas as pd
from scipy.sparse import load_npz

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "StyleSense_Artifacts"

def _load_required(filename):
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

def _load_recommendation_assets():
    matrix_path = ARTIFACTS_DIR / "interaction_matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing artifact: {matrix_path}")

    matrix = load_npz(matrix_path).tocsr()
    customers = _load_required("customer_ids.pkl")
    products = [str(x) for x in _load_required("product_ids.pkl")]
    popularity = _load_required("product_popularity.pkl")
    details = _load_required("stylesense_catalog.pkl").copy()

    if not isinstance(popularity, pd.Series):
        popularity = pd.Series(popularity)
    popularity.index = popularity.index.astype(str)
    details["article_id"] = details["article_id"].astype(str)

    # KNN may be unavailable if the local scikit-learn version differs.
    try:
        knn = _load_required("similarity_model.pkl")
        knn_error = None
    except Exception as exc:
        knn = None
        knn_error = str(exc)

    if matrix.shape[0] != len(customers):
        raise ValueError(
            "Recommendation artifact mismatch: interaction matrix rows "
            "do not match customer_ids.pkl."
        )
    if matrix.shape[1] != len(products):
        raise ValueError(
            "Recommendation artifact mismatch: interaction matrix columns "
            "do not match product_ids.pkl."
        )

    return matrix, customers, products, popularity, details, knn, knn_error

try:
    (
        interaction_matrix,
        customer_ids,
        product_ids,
        product_popularity,
        product_details,
        similarity_model,
        similarity_load_error,
    ) = _load_recommendation_assets()
    customer_to_index = {customer: i for i, customer in enumerate(customer_ids)}
    RECOMMENDATION_READY = True
    RECOMMENDATION_ERROR = None
except Exception as exc:
    interaction_matrix = None
    customer_ids = []
    product_ids = []
    product_popularity = pd.Series(dtype=float)
    product_details = pd.DataFrame()
    similarity_model = None
    similarity_load_error = str(exc)
    customer_to_index = {}
    RECOMMENDATION_READY = False
    RECOMMENDATION_ERROR = str(exc)

def _product_rows(article_ids):
    ids = [str(x) for x in article_ids]
    if product_details.empty:
        return pd.DataFrame({"article_id": ids})
    return pd.DataFrame({"article_id": ids}).merge(
        product_details, on="article_id", how="left"
    )

def recommend_products(customer_id, top_n=10):
    """Hybrid recommendations; falls back to popularity if KNN is unavailable."""
    if not RECOMMENDATION_READY:
        raise RuntimeError(RECOMMENDATION_ERROR or "Recommendation artifacts unavailable.")

    top_n = max(1, int(top_n))
    customer_id = str(customer_id)

    if customer_id not in customer_to_index or similarity_model is None:
        ids = product_popularity.head(top_n).index.tolist()
        result = _product_rows(ids)
        result["hybrid_score"] = [
            float(product_popularity.get(x, 0.0)) for x in ids
        ]
        result["recommendation_method"] = "Popularity fallback"
        return result

    customer_index = customer_to_index[customer_id]
    distances, indices = similarity_model.kneighbors(
        interaction_matrix[customer_index], n_neighbors=min(6, len(customer_ids))
    )

    cf_scores = {}
    for similar_index, distance in zip(indices[0][1:], distances[0][1:]):
        similarity = max(0.0, 1.0 - float(distance))
        if similarity <= 0:
            continue
        for product_index in interaction_matrix[int(similar_index)].indices:
            product_id = product_ids[int(product_index)]
            cf_scores[product_id] = cf_scores.get(product_id, 0.0) + similarity

    if cf_scores:
        max_score = max(cf_scores.values()) or 1.0
        cf_scores = {k: v / max_score for k, v in cf_scores.items()}

    already = {
        product_ids[int(i)]
        for i in interaction_matrix[customer_index].indices
    }
    all_products = set(product_popularity.index).union(cf_scores)
    scored = {}
    for product in all_products:
        if product in already:
            continue
        pop = float(product_popularity.get(product, 0.0))
        cf = float(cf_scores.get(product, 0.0))
        scored[product] = 0.5 * pop + 0.5 * cf

    ids = sorted(scored, key=scored.get, reverse=True)[:top_n]
    result = _product_rows(ids)
    result["hybrid_score"] = [scored[x] for x in ids]
    result["recommendation_method"] = "Hybrid KNN + popularity"
    return result

def get_customer_profile(customer_id):
    if not RECOMMENDATION_READY or str(customer_id) not in customer_to_index:
        return None
    customer_index = customer_to_index[str(customer_id)]
    purchased = [product_ids[int(i)] for i in interaction_matrix[customer_index].indices]
    history = _product_rows(purchased)
    categories = (
        history["product_type_name"].value_counts().head(5)
        if "product_type_name" in history else pd.Series(dtype=int)
    )
    colours = (
        history["colour_group_name"].value_counts().head(5)
        if "colour_group_name" in history else pd.Series(dtype=int)
    )
    return {
        "total_products": len(purchased),
        "categories": categories,
        "colours": colours,
        "history": history,
    }
