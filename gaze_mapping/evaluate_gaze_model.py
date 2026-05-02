"""
Evaluation script for gaze mapping model.

Computes comprehensive metrics for gaze estimation:
- Euclidean Error (Mean, Median, Std)
- Mean Absolute Error (MAE) for X and Y
- Root Mean Square Error (RMSE)
- Accuracy at different pixel thresholds
- Per-test breakdown (if test_id is present)

Usage:
    python evaluate_gaze_model.py --train TRAIN.csv --test TEST.csv
    python evaluate_gaze_model.py --crossval  # 5-fold cross-validation
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# ==================== FEATURES ====================
FEATURES = ["L_cx_px", "L_cy_px", "R_cx_px", "R_cy_px"]
EXTRA = ["xm", "ym", "dx", "dy"]
TARGETS = ["gaze_x", "gaze_y"]

# ==================== FEATURE ENGINEERING ====================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: mean eye position and inter-eye difference."""
    df = df.copy()
    df["xm"] = (df["L_cx_px"] + df["R_cx_px"]) / 2.0
    df["ym"] = (df["L_cy_px"] + df["R_cy_px"]) / 2.0
    df["dx"] = df["R_cx_px"] - df["L_cx_px"]
    df["dy"] = df["R_cy_px"] - df["L_cy_px"]
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data: drop NaN, convert types, add features."""
    required_cols = FEATURES + TARGETS
    df = df.dropna(subset=required_cols).copy()
    df = add_features(df)
    return df


# ==================== MODEL ====================
def create_model(degree: int = 2, alpha: float = 1.0) -> Pipeline:
    """Create polynomial Ridge regression model."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])


def train_model(model: Pipeline, X: np.ndarray, Y: np.ndarray) -> Pipeline:
    """Train the gaze mapping model."""
    model.fit(X, Y)
    return model


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Predict gaze coordinates."""
    return model.predict(X)


# ==================== METRICS ====================
def compute_metrics(gt: np.ndarray, pred: np.ndarray, 
                    thresholds: list = [25, 50, 75, 100, 150, 200]) -> Dict:
    """
    Compute comprehensive gaze estimation metrics.
    
    Args:
        gt: Ground truth gaze coordinates (N, 2)
        pred: Predicted gaze coordinates (N, 2)
        thresholds: Pixel thresholds for accuracy computation
    
    Returns:
        Dictionary with all computed metrics
    """
    # Euclidean errors
    errors = np.linalg.norm(pred - gt, axis=1)
    
    # Per-axis errors
    error_x = pred[:, 0] - gt[:, 0]
    error_y = pred[:, 1] - gt[:, 1]
    
    metrics = {
        "n_samples": len(errors),
        
        # Euclidean Error
        "euclidean_mean": np.mean(errors),
        "euclidean_median": np.median(errors),
        "euclidean_std": np.std(errors),
        "euclidean_min": np.min(errors),
        "euclidean_max": np.max(errors),
        "euclidean_q25": np.percentile(errors, 25),
        "euclidean_q75": np.percentile(errors, 75),
        "euclidean_q90": np.percentile(errors, 90),
        "euclidean_q95": np.percentile(errors, 95),
        
        # MAE (Mean Absolute Error)
        "mae_x": np.mean(np.abs(error_x)),
        "mae_y": np.mean(np.abs(error_y)),
        "mae_total": (np.mean(np.abs(error_x)) + np.mean(np.abs(error_y))) / 2,
        
        # RMSE (Root Mean Square Error)
        "rmse_x": np.sqrt(np.mean(error_x**2)),
        "rmse_y": np.sqrt(np.mean(error_y**2)),
        "rmse_total": np.sqrt(np.mean(errors**2)),
        
        # Bias (mean signed error)
        "bias_x": np.mean(error_x),
        "bias_y": np.mean(error_y),
    }
    
    # Accuracy at thresholds (% of predictions within N pixels)
    for t in thresholds:
        metrics[f"acc_{t}px"] = 100.0 * np.mean(errors <= t)
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    print(f"\n Samples: {metrics['n_samples']}")
    
    print("\n── Euclidean Error (pixels) ──")
    print(f"  Mean:     {metrics['euclidean_mean']:.2f} px")
    print(f"  Median:   {metrics['euclidean_median']:.2f} px")
    print(f"  Std:      {metrics['euclidean_std']:.2f} px")
    print(f"  Min/Max:  {metrics['euclidean_min']:.2f} / {metrics['euclidean_max']:.2f} px")
    print(f"  Q25/Q75:  {metrics['euclidean_q25']:.2f} / {metrics['euclidean_q75']:.2f} px")
    print(f"  Q90/Q95:  {metrics['euclidean_q90']:.2f} / {metrics['euclidean_q95']:.2f} px")
    
    print("\n── Per-Axis Error ──")
    print(f"  MAE (X):   {metrics['mae_x']:.2f} px")
    print(f"  MAE (Y):   {metrics['mae_y']:.2f} px")
    print(f"  RMSE (X):  {metrics['rmse_x']:.2f} px")
    print(f"  RMSE (Y):  {metrics['rmse_y']:.2f} px")
    print(f"  Bias (X):  {metrics['bias_x']:+.2f} px")
    print(f"  Bias (Y):  {metrics['bias_y']:+.2f} px")
    
    print("\n── Accuracy at Thresholds ──")
    thresholds = [k for k in metrics.keys() if k.startswith("acc_")]
    for t in sorted(thresholds, key=lambda x: int(x.split("_")[1].replace("px", ""))):
        threshold = t.split("_")[1]
        print(f"  Within {threshold:>5}:  {metrics[t]:.1f}%")
    
    print("=" * 60)


# ==================== VISUALIZATION ====================
def plot_error_distribution(errors: np.ndarray, save_path: Optional[str] = None):
    """Plot histogram of Euclidean errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}px')
    ax1.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f}px')
    ax1.set_xlabel('Euclidean Error (pixels)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    ax2 = axes[1]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax2.plot(sorted_errors, cdf, color='steelblue', linewidth=2)
    ax2.set_xlabel('Euclidean Error (pixels)', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('Cumulative Distribution Function', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add threshold markers
    for threshold in [50, 100, 150]:
        pct = 100 * np.mean(errors <= threshold)
        ax2.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(f'{pct:.0f}%@{threshold}px', xy=(threshold, pct), 
                    xytext=(threshold+10, pct+5), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scatter_predictions(gt: np.ndarray, pred: np.ndarray, save_path: Optional[str] = None):
    """Scatter plot of predictions vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # X coordinate
    ax1 = axes[0]
    ax1.scatter(gt[:, 0], pred[:, 0], alpha=0.3, s=10, c='steelblue')
    min_val, max_val = min(gt[:, 0].min(), pred[:, 0].min()), max(gt[:, 0].max(), pred[:, 0].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Ground Truth X (px)', fontsize=12)
    ax1.set_ylabel('Predicted X (px)', fontsize=12)
    ax1.set_title('X Coordinate: Prediction vs Ground Truth', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Y coordinate
    ax2 = axes[1]
    ax2.scatter(gt[:, 1], pred[:, 1], alpha=0.3, s=10, c='steelblue')
    min_val, max_val = min(gt[:, 1].min(), pred[:, 1].min()), max(gt[:, 1].max(), pred[:, 1].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('Ground Truth Y (px)', fontsize=12)
    ax2.set_ylabel('Predicted Y (px)', fontsize=12)
    ax2.set_title('Y Coordinate: Prediction vs Ground Truth', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_heatmap(gt: np.ndarray, errors: np.ndarray, save_path: Optional[str] = None):
    """Heatmap showing error magnitude at different gaze positions."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(gt[:, 0], gt[:, 1], c=errors, cmap='RdYlGn_r', 
                         s=20, alpha=0.6, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Euclidean Error (px)', fontsize=12)
    
    ax.set_xlabel('Gaze X (px)', fontsize=12)
    ax.set_ylabel('Gaze Y (px)', fontsize=12)
    ax.set_title('Error Distribution Across Gaze Positions', fontsize=14)
    ax.invert_yaxis()  # Typically screen coordinates have Y increasing downward
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== EVALUATION FUNCTIONS ====================
def evaluate_train_test(train_path: str, test_path: str, 
                        degree: int = 2, alpha: float = 1.0,
                        output_dir: Optional[str] = None) -> Dict:
    """
    Train on one dataset, evaluate on another.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        degree: Polynomial degree
        alpha: Ridge regularization
        output_dir: Directory for saving plots
    
    Returns:
        Metrics dictionary
    """
    # Load and prepare data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare features
    X_train = train_df[FEATURES + EXTRA].to_numpy(np.float64)
    Y_train = train_df[TARGETS].to_numpy(np.float64)
    
    X_test = test_df[FEATURES + EXTRA].to_numpy(np.float64)
    Y_test = test_df[TARGETS].to_numpy(np.float64)
    
    # Train and predict
    model = create_model(degree=degree, alpha=alpha)
    model = train_model(model, X_train, Y_train)
    Y_pred = predict(model, X_test)
    
    # Compute metrics
    metrics = compute_metrics(Y_test, Y_pred)
    print_metrics(metrics, title=f"Train: {os.path.basename(train_path)} → Test: {os.path.basename(test_path)}")
    
    # Generate plots
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        errors = np.linalg.norm(Y_pred - Y_test, axis=1)
        
        plot_error_distribution(errors, os.path.join(output_dir, "error_distribution.png"))
        plot_scatter_predictions(Y_test, Y_pred, os.path.join(output_dir, "scatter_predictions.png"))
        plot_error_heatmap(Y_test, errors, os.path.join(output_dir, "error_heatmap.png"))
    
    # Per-test breakdown if test_id exists
    if "test_id" in test_df.columns:
        print("\n── Per-Test Breakdown ──")
        for test_id in test_df["test_id"].unique():
            mask = test_df["test_id"] == test_id
            idx = mask.values
            subset_metrics = compute_metrics(Y_test[idx], Y_pred[idx])
            print(f"  {test_id}: Mean={subset_metrics['euclidean_mean']:.1f}px, "
                  f"Median={subset_metrics['euclidean_median']:.1f}px, N={subset_metrics['n_samples']}")
    
    return metrics


def evaluate_crossval(data_paths: list, n_folds: int = 5, 
                      degree: int = 2, alpha: float = 1.0) -> Dict:
    """
    Cross-validation evaluation.
    
    Args:
        data_paths: List of CSV file paths to combine
        n_folds: Number of folds
        degree: Polynomial degree
        alpha: Ridge regularization
    
    Returns:
        Aggregated metrics dictionary
    """
    # Combine all data
    dfs = [pd.read_csv(p) for p in data_paths]
    combined = pd.concat(dfs, ignore_index=True)
    combined = prepare_data(combined)
    
    print(f"Total samples: {len(combined)}")
    print(f"Running {n_folds}-fold cross-validation...")
    
    X = combined[FEATURES + EXTRA].to_numpy(np.float64)
    Y = combined[TARGETS].to_numpy(np.float64)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_errors = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        model = create_model(degree=degree, alpha=alpha)
        model = train_model(model, X_train, Y_train)
        Y_pred = predict(model, X_test)
        
        errors = np.linalg.norm(Y_pred - Y_test, axis=1)
        all_errors.extend(errors.tolist())
        
        fold_m = compute_metrics(Y_test, Y_pred)
        fold_metrics.append(fold_m)
        print(f"  Fold {fold_idx+1}: Mean={fold_m['euclidean_mean']:.1f}px, Median={fold_m['euclidean_median']:.1f}px")
    
    # Aggregate results
    all_errors = np.array(all_errors)
    
    # Compute aggregated metrics
    aggregated = {
        "n_samples": len(all_errors),
        "euclidean_mean": np.mean(all_errors),
        "euclidean_median": np.median(all_errors),
        "euclidean_std": np.std(all_errors),
        
        # Mean of fold metrics (with std)
        "fold_mean_euclidean_mean": np.mean([m["euclidean_mean"] for m in fold_metrics]),
        "fold_std_euclidean_mean": np.std([m["euclidean_mean"] for m in fold_metrics]),
    }
    
    print_metrics(compute_metrics(np.zeros_like(all_errors).reshape(-1, 2), 
                                   np.column_stack([all_errors, np.zeros_like(all_errors)])),
                  title=f"{n_folds}-Fold Cross-Validation Results")
    
    # Actually recompute properly
    final_metrics = {
        "n_samples": len(all_errors),
        "euclidean_mean": np.mean(all_errors),
        "euclidean_median": np.median(all_errors),
        "euclidean_std": np.std(all_errors),
        "euclidean_min": np.min(all_errors),
        "euclidean_max": np.max(all_errors),
        "euclidean_q25": np.percentile(all_errors, 25),
        "euclidean_q75": np.percentile(all_errors, 75),
        "euclidean_q90": np.percentile(all_errors, 90),
        "euclidean_q95": np.percentile(all_errors, 95),
    }
    for t in [25, 50, 75, 100, 150, 200]:
        final_metrics[f"acc_{t}px"] = 100.0 * np.mean(all_errors <= t)
    
    print("\n" + "=" * 60)
    print(f" {n_folds}-Fold Cross-Validation Results")
    print("=" * 60)
    print(f"\n Samples: {final_metrics['n_samples']}")
    print("\n── Euclidean Error (pixels) ──")
    print(f"  Mean:     {final_metrics['euclidean_mean']:.2f} px")
    print(f"  Median:   {final_metrics['euclidean_median']:.2f} px")
    print(f"  Std:      {final_metrics['euclidean_std']:.2f} px")
    print(f"  Q90/Q95:  {final_metrics['euclidean_q90']:.2f} / {final_metrics['euclidean_q95']:.2f} px")
    print("\n── Accuracy at Thresholds ──")
    for t in [25, 50, 75, 100, 150, 200]:
        print(f"  Within {t:>3}px:  {final_metrics[f'acc_{t}px']:.1f}%")
    print("=" * 60)
    
    return final_metrics


def evaluate_leave_one_session_out(session_dirs: list, degree: int = 2, alpha: float = 1.0) -> Dict:
    """
    Leave-one-session-out cross-validation.
    Train on all sessions except one, test on the held-out session.
    """
    print(f"Leave-One-Session-Out with {len(session_dirs)} sessions...")
    
    session_data = []
    for session_dir in session_dirs:
        # Find CSV files in session directory
        csv_files = [f for f in os.listdir(session_dir) if f.endswith('.csv')]
        if not csv_files:
            continue
        # Use first CSV or combine all
        dfs = [pd.read_csv(os.path.join(session_dir, f)) for f in csv_files 
               if 'timestamp' not in f.lower() or 'world' not in f.lower()]
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = prepare_data(combined)
            if len(combined) > 0:
                session_data.append((session_dir, combined))
    
    if len(session_data) < 2:
        print("Need at least 2 sessions for LOSO evaluation")
        return {}
    
    all_errors = []
    for i, (test_session, test_df) in enumerate(session_data):
        train_dfs = [df for j, (_, df) in enumerate(session_data) if j != i]
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        X_train = train_df[FEATURES + EXTRA].to_numpy(np.float64)
        Y_train = train_df[TARGETS].to_numpy(np.float64)
        X_test = test_df[FEATURES + EXTRA].to_numpy(np.float64)
        Y_test = test_df[TARGETS].to_numpy(np.float64)
        
        model = create_model(degree=degree, alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        errors = np.linalg.norm(Y_pred - Y_test, axis=1)
        mean_err = np.mean(errors)
        all_errors.extend(errors.tolist())
        
        print(f"  Test on {os.path.basename(test_session)}: Mean={mean_err:.1f}px, N={len(errors)}")
    
    all_errors = np.array(all_errors)
    print(f"\nOverall LOSO: Mean={np.mean(all_errors):.1f}px, Median={np.median(all_errors):.1f}px")
    
    return {"euclidean_mean": np.mean(all_errors), "euclidean_median": np.median(all_errors)}


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Evaluate gaze mapping model")
    parser.add_argument("--train", type=str, help="Training CSV file")
    parser.add_argument("--test", type=str, help="Test CSV file")
    parser.add_argument("--crossval", action="store_true", help="Run cross-validation on all data")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0)")
    parser.add_argument("--output", type=str, default="eval_output", help="Output directory for plots")
    parser.add_argument("--data", nargs="+", help="Data files for cross-validation")
    
    args = parser.parse_args()
    
    # Default paths (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.crossval:
        # Cross-validation mode
        if args.data:
            data_paths = args.data
        else:
            # Default: use all available CSVs
            data_paths = [
                os.path.join(script_dir, "2024-11-20_15-32-05-32205695/test_A.csv"),
                os.path.join(script_dir, "2024-11-20_15-32-05-32205695/test_B.csv"),
            ]
        
        evaluate_crossval(data_paths, degree=args.degree, alpha=args.alpha)
    
    elif args.train and args.test:
        # Train-test mode
        evaluate_train_test(
            args.train, args.test,
            degree=args.degree,
            alpha=args.alpha,
            output_dir=args.output
        )
    
    else:
        # Default: Train on test_A, evaluate on test_B (same as current behavior)
        train_path = os.path.join(script_dir, "2024-11-20_15-32-05-32205695/test_A.csv")
        test_path = os.path.join(script_dir, "2024-11-20_15-32-05-32205695/test_B.csv")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Running default evaluation: Train on test_A → Test on test_B")
            evaluate_train_test(
                train_path, test_path,
                degree=args.degree,
                alpha=args.alpha,
                output_dir=args.output
            )
            
            print("\n" + "=" * 60)
            print("Running reverse evaluation: Train on test_B → Test on test_A")
            evaluate_train_test(
                test_path, train_path,
                degree=args.degree,
                alpha=args.alpha,
                output_dir=args.output + "_reverse"
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
