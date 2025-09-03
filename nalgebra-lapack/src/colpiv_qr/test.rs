use super::ColPivQr;
use approx::{assert_abs_diff_eq, assert_relative_eq};
use na::{ColPivQR, DMatrix, Matrix, OMatrix};

#[test]
fn smoketest_qr_decomposition_for_f32_matrix() {
    let mat: OMatrix<f32, _, _> = nalgebra::matrix!
    [0.,   8.,   1.;
     4.,   5.,   4.;
     9.,   3.,   1.;
     8.,   4.,   9.];

    let _ =
        ColPivQr::new(mat, Default::default()).expect("creating qr decomposition must not fail");
}

#[test]
fn qr_decomposition_satisfies_ap_eq_qr_for_full_rank() {
    let a: OMatrix<f32, _, _> = nalgebra::matrix!
    [  65.,   15.,   69.;
       99.,   44.,   63.;
       28.,   50.,   76.;
       99.,   45.,   39.;
        1.,   51.,   93.;
    ];

    let qr = ColPivQr::new(a, Default::default()).unwrap();
    assert_eq!(qr.rank(), 3);
    let q = qr.q();
    let r = qr.r();
    let a_calc = &q * &r;
    let a_perm = {
        let mut tmp = a.clone();
        qr.p().permute_cols_mut(&mut tmp).unwrap();
        tmp
    };

    // A*P == Q*R (within numerical accuracy)
    assert_abs_diff_eq!(a_calc, a_perm, epsilon = 1e-4);

    let expected_r: Matrix<f32, _, _, _> = nalgebra::matrix!
    [
          -157.0860, -106.9733,  -89.7916;
          0.       , -114.7550,  -14.2190;
          0.       ,  0.      ,  -31.9733;
    ];

    assert_relative_eq!(expected_r, r, epsilon = 1e-4);

    // MATLAB gives full Q, we just need the first 3 columns
    let expected_q = nalgebra::matrix!
    [
      -0.439250, -0.156960,  0.834221;//  0.241441, -0.167942;
      -0.401054, -0.488849, -0.032457;// -0.686265,  0.358007;
      -0.483812,  0.207005, -0.297161;// -0.245988, -0.757800;
      -0.248272, -0.631271, -0.429461;//  0.594918, -0.038643;
      -0.592033,  0.543171, -0.174014;//  0.237296,  0.517564;
    ];

    assert_relative_eq!(expected_q, q, epsilon = 1e-4);
}

#[test]
fn qr_decomposition_satisfies_ap_eq_qr_for_for_rank_deficient() {
    let a: Matrix<f32, _, _, _> = nalgebra::dmatrix!
    [
       -105.,  79.,   92.;
        25.,   43.,   9.;
       -19.,   81.,   50.;
       -130.,  58.,   94.;

    ];
    let qr = ColPivQr::new(a.clone(), Default::default()).unwrap();
    assert_eq!(qr.rank(), 2);
    let r = qr.r();
    let q = qr.q();

    // we can compare R with the results from MATLAB, but we can't do that for
    // Q since it's not unique
    let expected_r = nalgebra::dmatrix![
          170.0323,   -95.8582, -132.9453;
          0.      ,   -93.9479, -46.9739;
          0.      ,    0.     , 0.;
    ];
    assert_abs_diff_eq!(r, expected_r, epsilon = 1e-4);

    let a_perm = {
        let mut tmp = a.clone();
        qr.p().permute_cols_mut(&mut tmp).unwrap();
        tmp
    };
    // check AP == QR
    assert_abs_diff_eq!(a_perm, &q * r, epsilon = 1e-4);
}

#[test]
fn test_q_multiplication() {
    let a: Matrix<f32, _, _, _> = nalgebra::matrix!
    [
       29.,    0.,   80.,   10.;
       43.,   95.,   99.,   20.;
       49.,   44.,   65.,   51.;
       45.,   21.,   18.,   90.;
       99.,   20.,   46.,   76.;
    ];

    let qr = ColPivQr::new(a, Default::default()).unwrap();

    // from Octave
    let q_full: Matrix<f32, _, _, _> = nalgebra::matrix![
        -0.52904777,  -0.27274187,  0.74369079, -0.23345502,  0.19530256;
        -0.65469661,  -0.26635373, -0.63457098,  0.03340311,  0.31085678;
        -0.42985131,   0.17855078, -0.06487111, -0.16634077, -0.86687367;
        -0.11903575,   0.75760623, -0.06473310, -0.54939070,  0.32533487;
        -0.30420247,   0.49881860,  0.18932786,  0.78414513,  0.08895075;
    ];

    let b = nalgebra::matrix!
    [
       32.,   93.;
       40.,   92.;
       42.,   65.;
       68.,   32.;
       73.,   30.;
    ];

    let q_mul_b = {
        let mut tmp = b.clone();
        qr.q_mul_mut(&mut tmp).unwrap();
        tmp
    };

    assert_abs_diff_eq!(q_mul_b, q_full * b, epsilon = 1e-4);

    let q_tr_mul_b = {
        let mut tmp = b.clone();
        qr.q_tr_mul_mut(&mut tmp).unwrap();
        tmp
    };

    assert_abs_diff_eq!(q_tr_mul_b, q_full.transpose() * b, epsilon = 1e-4);

    let b = b.transpose();

    let q_mul_b = {
        let mut tmp = b.clone();
        qr.mul_q_mut(&mut tmp).unwrap();
        tmp
    };

    assert_abs_diff_eq!(q_mul_b, b * q_full, epsilon = 1e-4);

    let q_tr_mul_b = {
        let mut tmp = b.clone();
        qr.mul_q_tr_mut(&mut tmp).unwrap();
        tmp
    };

    assert_abs_diff_eq!(q_tr_mul_b, b * q_full.transpose(), epsilon = 1e-4);
}

#[test]
fn test_rank_determination_for_different_matrices() {
    use super::{ColPivQr, rank::RankDeterminationAlgorithm};
    use nalgebra::{DMatrix, matrix};

    // Full rank square matrix (3x3)
    let full_rank_square = matrix![
        1.0, 2.0, 3.0;
        4.0, 5.0, 6.0;
        7.0, 8.0, 10.0
    ];

    let qr = ColPivQr::new(full_rank_square, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Full rank 3x3 matrix should have rank 3");

    // Rank deficient square matrix (3x3, rank 2)
    let rank_deficient_square = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 1.0, 0.0
    ];
    let qr = ColPivQr::new(rank_deficient_square, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 3x3 matrix should have rank 2");

    // Overdetermined system (4x3, full column rank)
    let overdetermined_full_rank = matrix![
        1.0, 0.0, 1.0;
        0.0, 1.0, 1.0;
        1.0, 1.0, 0.0;
        2.0, 1.0, 1.0
    ];
    let qr = ColPivQr::new(
        overdetermined_full_rank,
        RankDeterminationAlgorithm::default(),
    )
    .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        3,
        "Overdetermined 4x3 matrix should have full column rank 3"
    );

    // Overdetermined system with rank deficiency (4x3, rank 2)
    let overdetermined_rank_deficient = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 0.0, 0.0;  // This row is same as row1
        2.0, 2.0, 0.0   // This row is 2*row1 + 2*row2 = 2*row3 + 2*row2
    ];
    let qr = ColPivQr::new(
        overdetermined_rank_deficient,
        RankDeterminationAlgorithm::default(),
    )
    .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 4x3 matrix should have rank 2");

    // Rank 1 matrix (3x3)
    let rank_one = matrix![
        1.0, 2.0, 3.0;
        2.0, 4.0, 6.0;  // 2*row1
        3.0, 6.0, 9.0   // 3*row1
    ];
    let qr = ColPivQr::new(rank_one, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    print!("qr = {:?}", qr.qr);
    assert_eq!(qr.rank(), 1, "Rank 1 matrix should have rank 1");

    // zero matrix should have rank 0
    let zero = matrix![
        0., 0., 0.;
        0., 0., 0.;
        0., 0., 0.
    ];
    let qr = ColPivQr::new(zero, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        0,
        "Near-zero matrix should have rank 0 with strict tolerance"
    );

    // Dynamic matrix with different rank algorithms
    let dynamic_mat = DMatrix::from_row_slice(
        3,
        3,
        &[
            1.0, 0.0, 0.0, // row 1
            0.0, 1.0, 0.0, // row 2
            2.0, 3.0, 0.0, // row 3: This row is 2*row1 + 3*row2
        ],
    );

    let qr = ColPivQr::new(dynamic_mat.clone(), RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        2,
        "Dynamic matrix should have rank 2 with fixed eps"
    );

    // Identity matrix variants
    let identity_3x3 = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        0.0, 0.0, 1.0
    ];
    let qr = ColPivQr::new(identity_3x3, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Identity matrix should have full rank");

    // Matrix with very small but non-zero eigenvalues
    let small_eigenvalue = matrix![
        1.0, 0.0, 0.0;
        0.0, 1e-5, 0.0;
        0.0, 0.0, 1.0
    ];
    let qr = ColPivQr::new(small_eigenvalue, RankDeterminationAlgorithm::default())
        .expect("QR decomposition should succeed");
    assert_eq!(
        qr.rank(),
        3,
        "Matrix with small eigenvalues should have full rank with appropriate tolerance"
    );
}

#[test]
fn solve_full_rank_overdetermined_system_with_single_rhs() {
    let a = nalgebra::matrix![
       0f32,   2.,   1.;
       6.,   4.,   3.;
       6.,   3.,   5.;
       5.,   9.,   8.;
    ];
    let x = nalgebra::vector![8., 6., 2.];

    let b = &a * &x;
    let qr = ColPivQr::new(a, Default::default()).expect("qr decomposition must not fail");
    assert_eq!(qr.rank(), 3);
    let x_calc = qr.solve(&b).unwrap();
    assert_abs_diff_eq!(x_calc, x, epsilon = 1e-6);
}

#[test]
fn solve_rank_deficient_overdetermined_system_with_single_rhs() {
    let a = nalgebra::matrix![
     8.,    2.,    1.;
    14.,    4.,    3.;
     5.,    3.,    5.;
    29.,    9.,    8.;
    ];
    let x = nalgebra::vector![8., 6., 2.];

    let b = &a * &x;
    let qr = ColPivQr::new(a, Default::default()).expect("qr decomposition must not fail");
    assert_eq!(qr.rank(), 2);
    let x_calc = qr.solve(&b).unwrap();
    //@note(geo-ant) for rank deficient problems we cannot expect the
    // original vector x to be reproduced, since the solution is not unique.
    // That's why we don't compare x and x_calc for equality, but we check
    // that the matrices A*x and A*x_calc are equal (within numerical accuracy)
    assert_abs_diff_eq!(a * x_calc, a * x, epsilon = 1e-4);
}
