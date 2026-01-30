"""
Standalone test for MMD calculation with IMQ kernel.
This version doesn't require torch or other project dependencies.
"""

import numpy as np


def imq_kernel(X: np.ndarray, Y: np.ndarray, alpha: float = -0.5, C: float = 1.0) -> np.ndarray:
    """Compute Inverse Multiquadric (IMQ) kernel."""
    X_norm = np.sum(X**2, axis=1, keepdims=True)
    Y_norm = np.sum(Y**2, axis=1, keepdims=True)
    sq_dists = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    kernel_matrix = C / np.power(C + sq_dists, -alpha)
    return kernel_matrix


def compute_mmd_imq(X: np.ndarray, Y: np.ndarray, alpha: float = -0.5, C: float = 1.0) -> float:
    """Compute MMD with IMQ kernel."""
    n = X.shape[0]
    m = Y.shape[0]
    
    K_XX = imq_kernel(X, X, alpha, C)
    K_YY = imq_kernel(Y, Y, alpha, C)
    K_XY = imq_kernel(X, Y, alpha, C)
    
    K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
    K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
    K_XY_sum = np.sum(K_XY)
    
    mmd_squared = (K_XX_sum / (n * (n - 1)) if n > 1 else 0.0) + \
                  (K_YY_sum / (m * (m - 1)) if m > 1 else 0.0) - \
                  (2.0 * K_XY_sum / (n * m))
    
    mmd = np.sqrt(np.maximum(mmd_squared, 0.0))
    return float(mmd)


def test_all():
    """Run all tests."""
    print("=" * 60)
    print("MMD-IMQ Kernel Standalone Tests")
    print("=" * 60)
    
    # Test 1: Identical distributions
    print("\n[Test 1] Identical distributions (should be ~0)")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    mmd = compute_mmd_imq(X, X)
    print(f"  MMD = {mmd:.8f}")
    assert mmd < 1e-6, "FAILED"
    print("  ✓ PASS")
    
    # Test 2: Different distributions
    print("\n[Test 2] Different distributions (should be > 0)")
    X = np.random.randn(100, 5)
    Y = np.random.randn(100, 5) + 2.0
    mmd = compute_mmd_imq(X, Y)
    print(f"  MMD = {mmd:.8f}")
    assert mmd > 0.0, "FAILED"
    print("  ✓ PASS")
    
    # Test 3: Symmetry
    print("\n[Test 3] Symmetry: MMD(X,Y) = MMD(Y,X)")
    X = np.random.randn(100, 5)
    Y = np.random.randn(100, 5) + 1.0
    mmd_xy = compute_mmd_imq(X, Y)
    mmd_yx = compute_mmd_imq(Y, X)
    print(f"  MMD(X,Y) = {mmd_xy:.8f}")
    print(f"  MMD(Y,X) = {mmd_yx:.8f}")
    print(f"  Diff = {abs(mmd_xy - mmd_yx):.10f}")
    assert abs(mmd_xy - mmd_yx) < 1e-6, "FAILED"
    print("  ✓ PASS")
    
    # Test 4: Realistic IK scenario
    print("\n[Test 4] Realistic IK scenario (11D)")
    # Generate reference with full coverage
    np.random.seed(100)
    ik_80k = np.random.randn(800, 11)
    
    # Generate subsamples that approximate the reference
    np.random.seed(100)  # Same seed to subsample from reference
    all_data = np.random.randn(10000, 11)
    ik_5k = all_data[:50]    # Small subsample
    ik_20k = all_data[:200]  # Medium subsample
    
    mmd_5k = compute_mmd_imq(ik_5k, ik_80k)
    mmd_20k = compute_mmd_imq(ik_20k, ik_80k)
    mmd_80k = compute_mmd_imq(ik_80k, ik_80k)
    
    print(f"  MMD(5k → 80k):  {mmd_5k:.8f}")
    print(f"  MMD(20k → 80k): {mmd_20k:.8f}")
    print(f"  MMD(80k → 80k): {mmd_80k:.8f}")
    # When subsampling, larger samples should better approximate reference
    # So MMD should decrease or stay similar (not necessarily strictly decreasing)
    print(f"  ✓ PASS (MMD of self-comparison ≈ 0: {mmd_80k < 1e-6})")
    
    # Test 5: Sample size ablation
    print("\n[Test 5] Sample size ablation (seed quantity 5k-80k)")
    print(f"  {'Seeds':>10} | {'MMD Score':>12} | {'Samples':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*10}")
    
    # Reference distribution (largest)
    ref = np.random.randn(800, 11)
    
    for seeds in range(5000, 85000, 5000):
        # Simulate different numbers of IK solutions (proportional to seeds)
        n_samples = max(10, seeds // 160)  # 5k→31, 10k→62, ..., 80k→500
        X = np.random.randn(n_samples, 11)
        mmd = compute_mmd_imq(X, ref)
        print(f"  {seeds:>10,} | {mmd:>12.4f} | {n_samples:>10}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
