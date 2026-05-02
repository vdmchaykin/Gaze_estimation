import numpy as np

def poly_expand(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial feature expansion up to degree 2 or 3 for small D.
    For degree=1: returns X
    For degree=2: [x_i, x_i^2, x_i x_j for i<j]
    For degree=3: adds cubic monomials (simple, not exhaustive cross terms of degree 3 with all combinations).
    """
    X = np.asarray(X, dtype=np.float32)
    if degree <= 1:
        return X
    N, D = X.shape
    feats = [X]
    # squares
    feats.append(X**2)
    # pairwise
    pair = []
    for i in range(D):
        for j in range(i+1, D):
            pair.append((X[:, i] * X[:, j])[:, None])
    if pair:
        feats.append(np.concatenate(pair, axis=1).astype(np.float32))
    if degree >= 3:
        # simple cubic terms: x_i^3 and x_i^2 x_j for i!=j
        feats.append(X**3)
        cubic = []
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                cubic.append(((X[:, i]**2) * X[:, j])[:, None])
        if cubic:
            feats.append(np.concatenate(cubic, axis=1).astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)

def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-2):
    """Closed-form ridge for multi-output regression.
    Returns W of shape (D+1, out_dim)
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xb = np.concatenate([X, ones], axis=1)
    D = Xb.shape[1]
    A = Xb.T @ Xb + alpha * np.eye(D, dtype=np.float32)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B).astype(np.float32)
    return W

def predict_ridge(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xb = np.concatenate([X, ones], axis=1)
    return (Xb @ W).astype(np.float32)

def pixel_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    gt_xy = np.asarray(gt_xy, dtype=np.float32)
    return np.linalg.norm(pred_xy - gt_xy, axis=1)
