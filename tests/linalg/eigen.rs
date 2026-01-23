//! Tests for eigendecomposition of general (non-symmetric) matrices.

use nalgebra::linalg::Eigen;
use nalgebra::{DMatrix, Matrix2, Matrix3, Matrix4};
use num_complex::Complex;

/// Helper to compare complex numbers by real part, then imaginary part
fn compare_complex(a: &Complex<f64>, b: &Complex<f64>) -> std::cmp::Ordering {
    match a.re.partial_cmp(&b.re) {
        Some(std::cmp::Ordering::Equal) => {
            a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal)
        }
        Some(ord) => ord,
        None => std::cmp::Ordering::Equal,
    }
}

// =============================================================================
// BASIC TESTS
// =============================================================================

#[test]
fn eigen_identity_2x2() {
    // Identity matrix has eigenvalues 1, 1
    let m = Matrix2::<f64>::new(1.0, 0.0, 0.0, 1.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Check eigenvalues
    for i in 0..2 {
        assert!((eigen.eigenvalues[i].re - 1.0_f64).abs() < 1e-10);
        assert!(eigen.eigenvalues[i].im.abs() < 1e-10);
    }

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_diagonal_2x2() {
    // Diagonal matrix with distinct eigenvalues
    let m = Matrix2::<f64>::new(2.0, 0.0, 0.0, 5.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Sort eigenvalues for comparison
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().map(|c| c.re).collect();
    eigenvalues.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    assert!((eigenvalues[0] - 2.0).abs() < 1e-10);
    assert!((eigenvalues[1] - 5.0).abs() < 1e-10);

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_upper_triangular_3x3() {
    // Upper triangular matrix - eigenvalues are on the diagonal
    let m = Matrix3::<f64>::new(1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Sort eigenvalues for comparison
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().map(|c| c.re).collect();
    eigenvalues.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
    assert!((eigenvalues[1] - 4.0).abs() < 1e-10);
    assert!((eigenvalues[2] - 6.0).abs() < 1e-10);

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// SYMMETRIC MATRICES (should also work, eigenvalues are real)
// =============================================================================

#[test]
fn eigen_symmetric_2x2() {
    let m = Matrix2::<f64>::new(4.0, 2.0, 2.0, 1.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Eigenvalues should be real
    for ev in eigen.eigenvalues.iter() {
        assert!(
            ev.im.abs() < 1e-10,
            "Expected real eigenvalue, got imaginary part {}",
            ev.im
        );
    }

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_symmetric_3x3() {
    let m = Matrix3::<f64>::new(2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Eigenvalues should be real
    for ev in eigen.eigenvalues.iter() {
        assert!(
            ev.im.abs() < 1e-10,
            "Expected real eigenvalue, got imaginary part {}",
            ev.im
        );
    }

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// COMPLEX EIGENVALUES
// =============================================================================

#[test]
fn eigen_rotation_2x2() {
    // 90-degree rotation matrix has eigenvalues i and -i
    let m = Matrix2::<f64>::new(0.0, -1.0, 1.0, 0.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Sort by imaginary part to get consistent ordering
    let mut eigenvalues: Vec<Complex<f64>> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| compare_complex(a, b));

    // Should be -i and +i (real parts are 0)
    assert!(eigenvalues[0].re.abs() < 1e-10);
    assert!((eigenvalues[0].im - (-1.0_f64)).abs() < 1e-10);
    assert!(eigenvalues[1].re.abs() < 1e-10);
    assert!((eigenvalues[1].im - 1.0_f64).abs() < 1e-10);

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_complex_conjugate_pair_3x3() {
    // Matrix with one real eigenvalue and one complex conjugate pair
    // This is a block diagonal with [2] and a 2x2 rotation-like block
    let m = Matrix3::<f64>::new(2.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 2.0, 1.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Sort eigenvalues
    let mut eigenvalues: Vec<Complex<f64>> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| compare_complex(a, b));

    // Expected: 2 (real), 1+2i, 1-2i
    // After sorting: 1-2i, 1+2i, 2
    assert!((eigenvalues[0].re - 1.0_f64).abs() < 1e-10);
    assert!((eigenvalues[0].im - (-2.0_f64)).abs() < 1e-10);
    assert!((eigenvalues[1].re - 1.0_f64).abs() < 1e-10);
    assert!((eigenvalues[1].im - 2.0_f64).abs() < 1e-10);
    assert!((eigenvalues[2].re - 2.0_f64).abs() < 1e-10);
    assert!(eigenvalues[2].im.abs() < 1e-10);

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// STOCHASTIC MATRIX (from the original issue)
// =============================================================================

#[test]
fn eigen_stochastic_matrix() {
    // The motivating example from the nalgebra issue
    let m = Matrix3::<f64>::new(0.47, 0.53, 0.0, 0.52, 0.0, 0.48, 0.0, 0.0, 1.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // One eigenvalue should be 1 (Perron-Frobenius)
    let has_unit_eigenvalue = eigen
        .eigenvalues
        .iter()
        .any(|ev| (ev.re - 1.0_f64).abs() < 1e-10 && ev.im.abs() < 1e-10);
    assert!(
        has_unit_eigenvalue,
        "Stochastic matrix should have eigenvalue 1"
    );

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// DYNAMIC MATRICES
// =============================================================================

#[test]
fn eigen_dynamic_matrix() {
    let m = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0]);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Sort eigenvalues
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().map(|c| c.re).collect();
    eigenvalues.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
    assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
    assert!((eigenvalues[2] - 3.0).abs() < 1e-10);

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_dynamic_with_complex() {
    // Dynamic matrix with complex eigenvalues
    let m = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Should have purely imaginary eigenvalues
    for ev in eigen.eigenvalues.iter() {
        assert!(
            ev.re.abs() < 1e-10,
            "Expected zero real part, got {}",
            ev.re
        );
        assert!(
            (ev.im.abs() - 1.0_f64).abs() < 1e-10,
            "Expected |im| = 1, got {}",
            ev.im
        );
    }

    // Verify decomposition
    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn eigen_1x1_matrix() {
    let m = nalgebra::Matrix1::<f64>::new(5.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    assert!((eigen.eigenvalues[0].re - 5.0_f64).abs() < 1e-10);
    assert!(eigen.eigenvalues[0].im.abs() < 1e-10);
}

#[test]
fn eigen_zero_matrix() {
    let m = Matrix2::<f64>::new(0.0, 0.0, 0.0, 0.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // All eigenvalues should be zero
    for ev in eigen.eigenvalues.iter() {
        assert!(ev.re.abs() < 1e-10);
        assert!(ev.im.abs() < 1e-10);
    }
}

#[test]
fn eigen_repeated_eigenvalues() {
    // Matrix with repeated eigenvalue (but still diagonalizable)
    let m = Matrix2::<f64>::new(3.0, 0.0, 0.0, 3.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    for ev in eigen.eigenvalues.iter() {
        assert!((ev.re - 3.0_f64).abs() < 1e-10);
        assert!(ev.im.abs() < 1e-10);
    }

    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// LARGER MATRICES
// =============================================================================

#[test]
fn eigen_4x4_mixed() {
    // 4x4 matrix with 2 real eigenvalues and 1 complex conjugate pair
    let m = Matrix4::<f64>::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,
    );

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    // Count real vs complex eigenvalues
    let real_count = eigen
        .eigenvalues
        .iter()
        .filter(|ev| ev.im.abs() < 1e-10)
        .count();
    let complex_count = eigen
        .eigenvalues
        .iter()
        .filter(|ev| ev.im.abs() >= 1e-10)
        .count();

    assert_eq!(real_count, 2, "Expected 2 real eigenvalues");
    assert_eq!(
        complex_count, 2,
        "Expected 2 complex eigenvalues (conjugate pair)"
    );

    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

#[test]
fn eigen_companion_matrix() {
    // Companion matrix for polynomial x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    // Eigenvalues should be 1, 2, 3
    let m = Matrix3::<f64>::new(0.0, 0.0, 6.0, 1.0, 0.0, -11.0, 0.0, 1.0, 6.0);

    let eigen = Eigen::new(m.clone()).expect("Eigen decomposition failed");

    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().map(|c| c.re).collect();
    eigenvalues.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
    assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
    assert!((eigenvalues[2] - 3.0).abs() < 1e-10);

    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}

// =============================================================================
// CONVENIENCE METHOD TESTS
// =============================================================================

#[test]
fn eigen_method_on_matrix() {
    let m = Matrix3::<f64>::new(1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0);

    // Test the convenience method
    let eigen = m.clone().eigen().expect("Eigen decomposition failed");

    let error = eigen.verify(&m);
    assert!(error < 1e-10, "Verification error too large: {}", error);
}
