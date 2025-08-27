use super::ColPivQR;
use approx::assert_abs_diff_eq;
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
fn qr_decomposition_is_logically_correct() {
    let a: OMatrix<f32, _, _> = nalgebra::matrix!
    [  65.,   15.,   69.;
       99.,   44.,   63.;
       28.,   50.,   76.;
       99.,   45.,   39.;
        1.,   51.,   93.;
    ];

    let qr = ColPivQR::new(a, Default::default()).unwrap();
    assert_eq!(qr.rank(), 3);
    let q = qr.q();
    let r = qr.r();
    let a_calc = {
        let mut tmp = q * r;
        qr.p().permute_cols_mut(&mut tmp).unwrap();
        tmp
    };

    assert_abs_diff_eq!(a_calc, a, epsilon = 1e-6);

    // let a_calc = qr.q().transpose() * qr.r();
    // assert_eq!(qr.q(), 2. * qr.q());
    panic!("r = {:?}", qr.r());

    todo!("waaaah")
}

#[test]
fn test_q_multiplication() {
    todo!()
}

#[test]
fn test_rank_determination_for_different_matrices() {
    use super::{ColPivQR, rank::RankEstimationAlgo};
    use nalgebra::{DMatrix, matrix};

    // Full rank square matrix (3x3)
    let full_rank_square = matrix![
        1.0, 2.0, 3.0;
        4.0, 5.0, 6.0;
        7.0, 8.0, 10.0
    ];
    let qr = ColPivQR::new(full_rank_square, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Full rank 3x3 matrix should have rank 3");

    // Rank deficient square matrix (3x3, rank 2)
    let rank_deficient_square = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 1.0, 0.0
    ];
    let qr = ColPivQR::new(rank_deficient_square, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 3x3 matrix should have rank 2");

    // Overdetermined system (4x3, full column rank)
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

    // Overdetermined system with rank deficiency (4x3, rank 2)
    let overdetermined_rank_deficient = matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        1.0, 0.0, 0.0;  // This row is same as row1
        2.0, 2.0, 0.0   // This row is 2*row1 + 2*row2 = 2*row3 + 2*row2
    ];
    let qr = ColPivQR::new(overdetermined_rank_deficient, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 2, "Rank deficient 4x3 matrix should have rank 2");

    // Underdetermined system (3x4, full row rank)
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

    // Underdetermined system with rank deficiency (3x4, rank 2)
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

    // Rank 1 matrix (3x3)
    let rank_one = matrix![
        1.0, 2.0, 3.0;
        2.0, 4.0, 6.0;  // 2*row1
        3.0, 6.0, 9.0   // 3*row1
    ];
    let qr = ColPivQR::new(rank_one, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    print!("qr = {:?}", qr.qr);
    assert_eq!(qr.rank(), 1, "Rank 1 matrix should have rank 1");

    // zero matrix should have rank 0
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

    let qr = ColPivQR::new(dynamic_mat.clone(), RankEstimationAlgo::default())
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
    let qr = ColPivQR::new(identity_3x3, RankEstimationAlgo::default())
        .expect("QR decomposition should succeed");
    assert_eq!(qr.rank(), 3, "Identity matrix should have full rank");

    // Matrix with very small but non-zero eigenvalues
    let small_eigenvalue = matrix![
        1.0, 0.0, 0.0;
        0.0, 1e-5, 0.0;
        0.0, 0.0, 1.0
    ];
    let qr = ColPivQR::new(small_eigenvalue, RankEstimationAlgo::default())
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
    let qr = ColPivQR::new(a, Default::default()).expect("qr decomposition must not fail");
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
    let qr = ColPivQR::new(a, Default::default()).expect("qr decomposition must not fail");
    assert_eq!(qr.rank(), 2);
    let x_calc = qr.solve(&b).unwrap();
    //@note(geo-ant) for rank deficient problems we cannot expect the
    // original vector x to be reproduced, since the solution is not unique.
    // That's why we don't compare x and x_calc for equality, but we check
    // that the matrices A*x and A*x_calc are equal (within numerical accuracy)
    assert_abs_diff_eq!(a * x_calc, a * x, epsilon = 1e-4);
}
