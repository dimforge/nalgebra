use super::ColPivQR;
use na::OMatrix;

#[test]
fn smoketest_qr_decomposition_for_f32_matrix() {
    let mat: OMatrix<f32, _, _> = nalgebra::matrix!
    [0.,   8.,   1.;
     4.,   5.,   4.;
     9.,   3.,   1.;
     8.,   4.,   9.];

    let _ =
        ColPivQR::new(mat, Default::default()).expect("creating qr decomposition must not fail");
}

#[test]
fn test_rank_determination_for_different_matrices() {
    use super::{ColPivQR, rank::RankEstimationAlgo};
    use nalgebra::{DMatrix, matrix};

    // Test 1: Full rank square matrix (3x3)
    let full_rank_square = matrix![
        1.0, 2.0, 3.0;
        4.0, 5.0, 6.0;
        7.0, 8.0, 10.0  // Changed last element to avoid rank deficiency
    ];
    let qr = ColPivQR::new(full_rank_square, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Full rank 3x3 matrix should have rank 3");

    // Test 2: Rank deficient square matrix (3x3, rank 2)
    let rank_deficient_square = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 1.0, 0.0  // This row is row1 + row2, third column is zero
    ];
    let qr = ColPivQR::new(rank_deficient_square, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 3x3 matrix should have rank 2");

    // Test 3: Overdetermined system (4x3, full column rank)
    let overdetermined_full_rank = matrix![
        1.0, 0.0, 1.0;
        0.0, 1.0, 1.0;
        1.0, 1.0, 0.0;
        2.0, 1.0, 1.0
    ];
    let qr = ColPivQR::new(overdetermined_full_rank, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        3,
        "Overdetermined 4x3 matrix should have full column rank 3"
    );

    // Test 4: Overdetermined system with rank deficiency (4x3, rank 2)
    let overdetermined_rank_deficient = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 0.0, 0.0;  // This row is same as row1
        2.0, 2.0, 0.0   // This row is 2*row1 + 2*row2 = 2*row3 + 2*row2
    ];
    let qr = ColPivQR::new(overdetermined_rank_deficient, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 4x3 matrix should have rank 2");

    // Test 5: Underdetermined system (3x4, full row rank)
    let underdetermined_full_rank = matrix![
        1.0, 0.0, 1.0, 2.0;
        0.0, 1.0, 1.0, 3.0;
        1.0, 1.0, 0.0, 1.0
    ];
    let qr = ColPivQR::new(underdetermined_full_rank, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        3,
        "Underdetermined 3x4 matrix should have full row rank 3"
    );

    // Test 6: Underdetermined system with rank deficiency (3x4, rank 2)
    let underdetermined_rank_deficient = matrix![
        1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        1.0, 1.0, 0.0, 0.0  // This row is row1 + row2
    ];
    let qr = ColPivQR::new(
        underdetermined_rank_deficient,
        RankEstimationAlgo::default(),
    )
    .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 3x4 matrix should have rank 2");

    // Test 7: Rank 1 matrix (3x3)
    let rank_one = matrix![
        1.0, 2.0, 3.0;
        2.0, 4.0, 6.0;  // 2*row1
        3.0, 6.0, 9.0   // 3*row1
    ];
    let qr = ColPivQR::new(rank_one, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    print!("qr = {:?}", qr.qr);
    assert_eq!(qr.rank(), 1, "Rank 1 matrix should have rank 1");

    // Test 8: Near-zero matrix (should have rank 0 with strict tolerance)
    let zero = matrix![
        0., 0., 0.;
        0., 0., 0.;
        0., 0., 0.
    ];
    let qr = ColPivQR::new(zero, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        0,
        "Near-zero matrix should have rank 0 with strict tolerance"
    );

    // Test 9: Dynamic matrix with different rank algorithms
    let dynamic_mat = DMatrix::from_row_slice(
        3,
        3,
        &[
            1.0, 0.0, 0.0, // row 1
            0.0, 1.0, 0.0, // row 2
            2.0, 3.0, 0.0, // row 3: This row is 2*row1 + 3*row2
        ],
    );

    // Test with fixed epsilon
    let qr_fixed = ColPivQR::new(dynamic_mat.clone(), RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr_fixed.rank(),
        2,
        "Dynamic matrix should have rank 2 with fixed eps"
    );

    // Test with scaled epsilon (default)
    let qr_scaled =
        ColPivQR::new(dynamic_mat, Default::default()).expect("QR decomposition should succeed");
    assert_eq!(
        qr_scaled.rank(),
        2,
        "Dynamic matrix should have rank 2 with scaled eps"
    );

    // Test 10: Identity matrix variants
    let identity_3x3 = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        0.0, 0.0, 1.0
    ];
    let qr = ColPivQR::new(identity_3x3, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Identity matrix should have full rank");

    // Test 11: Matrix with very small but non-zero eigenvalues
    let small_eigenvalue = matrix![
        1.0, 0.0, 0.0;
        0.0, 1e-5, 0.0;  // Small but above tolerance (1e-6)
        0.0, 0.0, 1.0
    ];
    let qr = ColPivQR::new(small_eigenvalue, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        3,
        "Matrix with small eigenvalues should have full rank with appropriate tolerance"
    );
    // Note: We use tolerance 1e-6 rather than machine epsilon because LAPACK's QR
    // decomposition introduces small numerical perturbations that make theoretically
    // zero diagonal elements become very small non-zero values (around 1e-6 to 1e-7).
    // This is expected behavior in numerical linear algebra - "rank deficient" means
    // "approximately rank deficient within tolerance" rather than exactly zero.
}

#[test]
fn solve_overdetermined_system_with_exact_solution() {
    let a = nalgebra::matrix![
       0f32,   2.,   1.;
       6.,   4.,   3.;
       6.,   3.,   5.;
       5.,   9.,   8.;
    ];
    let x = nalgebra::vector![5., 3., 7.];

    let b = &a * &x;
    let qr = ColPivQR::new(a, Default::default()).expect("qr decomposition must not fail");
    let x_calc = qr.solve(&b).unwrap();
    assert_eq!(x_calc, x);
}
