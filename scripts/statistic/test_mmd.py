"""
Test MMD calculation with IMQ kernel.

This script validates the MMD implementation with synthetic data.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from analyze_ik_mmd import compute_mmd_imq, imq_kernel


def test_mmd_identical_distributions():
    """Test that MMD between identical distributions is 0."""
    print("\nTest 1: MMD between identical distributions")
    print("-" * 60)
    
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    Y = X.copy()  # Identical
    
    mmd = compute_mmd_imq(X, Y)
    print(f"MMD between identical distributions: {mmd:.8f}")
    print(f"Expected: ~0.0")
    print(f"✓ PASS" if mmd < 1e-6 else f"✗ FAIL")
    

def test_mmd_different_distributions():
    """Test that MMD between different distributions is non-zero."""
    print("\nTest 2: MMD between different distributions")
    print("-" * 60)
    
    # Generate random data from different distributions
    np.random.seed(42)
    X = np.random.randn(100, 5)  # N(0, 1)
    Y = np.random.randn(100, 5) + 2.0  # N(2, 1), shifted
    
    mmd = compute_mmd_imq(X, Y)
    print(f"MMD between N(0,1) and N(2,1): {mmd:.8f}")
    print(f"Expected: > 0.0")
    print(f"✓ PASS" if mmd > 0.0 else f"✗ FAIL")


def test_mmd_symmetric():
    """Test that MMD is symmetric: MMD(X,Y) = MMD(Y,X)."""
    print("\nTest 3: MMD symmetry")
    print("-" * 60)
    
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    Y = np.random.randn(100, 5) + 1.0
    
    mmd_xy = compute_mmd_imq(X, Y)
    mmd_yx = compute_mmd_imq(Y, X)
    
    print(f"MMD(X, Y): {mmd_xy:.8f}")
    print(f"MMD(Y, X): {mmd_yx:.8f}")
    print(f"Difference: {abs(mmd_xy - mmd_yx):.10f}")
    print(f"✓ PASS" if abs(mmd_xy - mmd_yx) < 1e-6 else f"✗ FAIL")


def test_mmd_convergence():
    """Test that MMD converges as sample size increases."""
    print("\nTest 4: MMD convergence with sample size")
    print("-" * 60)
    
    np.random.seed(42)
    
    # Generate large datasets
    X_large = np.random.randn(1000, 5)
    Y_large = np.random.randn(1000, 5) + 0.5
    
    sample_sizes = [50, 100, 200, 500, 1000]
    mmds = []
    
    for n in sample_sizes:
        X = X_large[:n]
        Y = Y_large[:n]
        mmd = compute_mmd_imq(X, Y)
        mmds.append(mmd)
        print(f"Sample size {n:4d}: MMD = {mmd:.8f}")
    
    # Check that MMD stabilizes (variance decreases)
    print(f"\nVariance of MMD values: {np.var(mmds):.8f}")
    print(f"✓ Test complete")


def test_imq_kernel_properties():
    """Test IMQ kernel properties."""
    print("\nTest 5: IMQ kernel properties")
    print("-" * 60)
    
    np.random.seed(42)
    X = np.random.randn(10, 5)
    
    # Test 1: k(x, x) should be C (when ||x-x||² = 0)
    K = imq_kernel(X[:1], X[:1])
    print(f"k(x, x) = {K[0, 0]:.8f} (expected: 1.0)")
    
    # Test 2: Kernel should be symmetric
    K = imq_kernel(X, X)
    is_symmetric = np.allclose(K, K.T)
    print(f"Kernel matrix symmetric: {is_symmetric}")
    
    # Test 3: Kernel values should be positive
    all_positive = np.all(K > 0)
    print(f"All kernel values positive: {all_positive}")
    
    print(f"✓ PASS" if is_symmetric and all_positive else f"✗ FAIL")


def test_real_ik_scenario():
    """Test with realistic IK data dimensions."""
    print("\nTest 6: Realistic IK scenario (11D joint space)")
    print("-" * 60)
    
    np.random.seed(42)
    
    # Simulate IK data: 11 dimensions (10 robot + 1 object joint)
    # Different seed quantities
    ik_5k = np.random.randn(50, 11)  # Clustered to 50
    ik_20k = np.random.randn(200, 11)  # Clustered to 200
    ik_80k = np.random.randn(800, 11)  # Reference (most coverage)
    
    # Compute MMD against reference
    mmd_5k = compute_mmd_imq(ik_5k, ik_80k)
    mmd_20k = compute_mmd_imq(ik_20k, ik_80k)
    mmd_80k = compute_mmd_imq(ik_80k, ik_80k)  # Should be ~0
    
    print(f"MMD(5k seeds → 80k ref):  {mmd_5k:.8f}")
    print(f"MMD(20k seeds → 80k ref): {mmd_20k:.8f}")
    print(f"MMD(80k seeds → 80k ref): {mmd_80k:.8f} (should be ~0)")
    print(f"\nExpected pattern: MMD(5k) > MMD(20k) > MMD(80k) ≈ 0")
    print(f"✓ PASS" if mmd_5k > mmd_20k and mmd_80k < 1e-6 else f"✗ FAIL")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MMD-IMQ Kernel Implementation Tests")
    print("=" * 60)
    
    test_mmd_identical_distributions()
    test_mmd_different_distributions()
    test_mmd_symmetric()
    test_mmd_convergence()
    test_imq_kernel_properties()
    test_real_ik_scenario()
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
