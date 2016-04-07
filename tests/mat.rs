extern crate nalgebra as na;
extern crate rand;

use rand::random;
use na::{Vec1, Vec3, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6, Rot2, Rot3, Persp3, PerspMat3, Ortho3,
         OrthoMat3, DMat, DVec, Row, Col, BaseFloat, Diag, Transpose, RowSlice, ColSlice};

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmat : $t = random();

      match na::inv(&randmat) {
          None    => { },
          Some(i) => assert!(na::approx_eq(&(i * randmat), &na::one()))
      }
    }
  );
);

macro_rules! test_transpose_mat_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmat : $t = random();

      assert!(na::transpose(&na::transpose(&randmat)) == randmat);
    }
  );
);

macro_rules! test_qr_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmat : $t = random();

      let (q, r) = na::qr(&randmat);
      let recomp = q * r;

      assert!(na::approx_eq(&randmat,  &recomp));
    }
  );
);

macro_rules! test_cholesky_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      
      // construct symmetric positive definite matrix
      let mut randmat : $t = random();
      let mut diagmat : $t = Diag::from_diag(&na::diag(&randmat));

      diagmat = na::abs(&diagmat) + 1.0;
      randmat = randmat * diagmat * na::transpose(&randmat);

      let result = na::cholesky(&randmat);

      assert!(result.is_ok());

      let v = result.unwrap();
      let recomp = v * na::transpose(&v);
      assert!(na::approx_eq(&randmat,  &recomp));
    }
  );
);

// NOTE: deactivated untile we get a better convergence rate.
// macro_rules! test_eigen_qr_impl(
//     ($t: ty) => {
//         for _ in 0usize .. 10000 {
//             let randmat : $t = random();
//             // Make it symetric so that we can recompose the matrix to test at the end.
//             let randmat = na::transpose(&randmat) * randmat;
// 
//             let (eigenvectors, eigenvalues) = na::eigen_qr(&randmat, &Float::epsilon(), 100);
// 
//             let diag: $t = Diag::from_diag(&eigenvalues);
// 
//             let recomp = eigenvectors * diag * na::transpose(&eigenvectors);
// 
//             println!("eigenvalues: {}", eigenvalues);
//             println!("   mat: {}", randmat);
//             println!("recomp: {}", recomp);
// 
//             assert!(na::approx_eq_eps(&randmat,  &recomp, &1.0e-2));
//         }
//     }
// )

#[test]
fn test_transpose_mat1() {
    test_transpose_mat_impl!(Mat1<f64>);
}

#[test]
fn test_transpose_mat2() {
    test_transpose_mat_impl!(Mat2<f64>);
}

#[test]
fn test_transpose_mat3() {
    test_transpose_mat_impl!(Mat3<f64>);
}

#[test]
fn test_transpose_mat4() {
    test_transpose_mat_impl!(Mat4<f64>);
}

#[test]
fn test_transpose_mat5() {
    test_transpose_mat_impl!(Mat5<f64>);
}

#[test]
fn test_transpose_mat6() {
    test_transpose_mat_impl!(Mat6<f64>);
}

#[test]
fn test_inv_mat1() {
    test_inv_mat_impl!(Mat1<f64>);
}

#[test]
fn test_inv_mat2() {
    test_inv_mat_impl!(Mat2<f64>);
}

#[test]
fn test_inv_mat3() {
    test_inv_mat_impl!(Mat3<f64>);
}

#[test]
fn test_inv_mat4() {
    test_inv_mat_impl!(Mat4<f64>);
}

#[test]
fn test_inv_mat5() {
    test_inv_mat_impl!(Mat5<f64>);
}

#[test]
fn test_inv_mat6() {
    test_inv_mat_impl!(Mat6<f64>);
}

#[test]
fn test_rotation2() {
    for _ in 0usize .. 10000 {
        let randmat: na::Rot2<f64> = na::one();
        let ang    = Vec1::new(na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());

        assert!(na::approx_eq(&na::rotation(&na::append_rotation(&randmat, &ang)), &ang));
    }
}

#[test]
fn test_index_mat2() {
  let mat: Mat2<f64> = random();

  assert!(mat[(0, 1)] == na::transpose(&mat)[(1, 0)]);
}

#[test]
fn test_inv_rotation3() {
    for _ in 0usize .. 10000 {
        let randmat: Rot3<f64> = na::one();
        let dir:     Vec3<f64> = random();
        let ang            = na::normalize(&dir) * (na::abs(&random::<f64>()) % <f64 as BaseFloat>::pi());
        let rot            = na::append_rotation(&randmat, &ang);

        assert!(na::approx_eq(&(na::transpose(&rot) * rot), &na::one()));
    }
}

#[test]
fn test_rot3_rotation_between() {
    let r1: Rot3<f64> = random();
    let r2: Rot3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot3_angle_between() {
    let r1: Rot3<f64> = random();
    let r2: Rot3<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}

#[test]
fn test_rot2_rotation_between() {
    let r1: Rot2<f64> = random();
    let r2: Rot2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);

    assert!(na::approx_eq(&(delta * r1), &r2))
}

#[test]
fn test_rot2_angle_between() {
    let r1: Rot2<f64> = random();
    let r2: Rot2<f64> = random();

    let delta = na::rotation_between(&r1, &r2);
    let delta_angle = na::angle_between(&r1, &r2);

    assert!(na::approx_eq(&na::norm(&na::rotation(&delta)), &delta_angle))
}

#[test]
fn test_mean_dmat() {
    let mat = DMat::from_row_vec(
        3,
        3,
        &[
            1.0f64, 2.0, 3.0,
            4.0f64, 5.0, 6.0,
            7.0f64, 8.0, 9.0,
        ]
    );

    assert!(na::approx_eq(&na::mean(&mat), &DVec::from_slice(3, &[4.0f64, 5.0, 6.0])));
}

#[test]
fn test_cov_dmat() {
    let mat = DMat::from_row_vec(
        5,
        3,
        &[
            4.0f64, 2.0, 0.60,
            4.2f64, 2.1, 0.59,
            3.9f64, 2.0, 0.58,
            4.3f64, 2.1, 0.62,
            4.1f64, 2.2, 0.63
        ]
    );

    let expected = DMat::from_row_vec(
        3,
        3,
        &[
            0.025f64,   0.0075,  0.00175,
            0.0075f64,  0.007,   0.00135,
            0.00175f64, 0.00135, 0.00043
        ]
    );

    assert!(na::approx_eq(&na::cov(&mat), &expected));
}

#[test]
fn test_transpose_dmat() {
    let mat = DMat::from_row_vec(
        8,
        4,
        &[
            1u32,2,  3,  4,
            5,   6,  7,  8,
            9,   10, 11, 12,
            13,  14, 15, 16,
            17,  18, 19, 20,
            21,  22, 23, 24,
            25,  26, 27, 28,
            29,  30, 31, 32
        ]
    );

    assert!(na::transpose(&na::transpose(&mat)) == mat);
}

#[test]
fn test_row_slice_dmat() {
    let mat = DMat::from_row_vec(
        5,
        4,
        &[
            1u32,2,  3,  4,
            5,   6,  7,  8,
            9,   10, 11, 12,
            13,  14, 15, 16,
            17,  18, 19, 20,
        ]
    );

    assert_eq!(&DVec::from_slice(4, &[1u32, 2, 3, 4]), &mat.row_slice(0, 0, 4));
    assert_eq!(&DVec::from_slice(2, &[1u32, 2]), &mat.row_slice(0, 0, 2));
    assert_eq!(&DVec::from_slice(2, &[10u32, 11]), &mat.row_slice(2, 1, 3));
    assert_eq!(&DVec::from_slice(2, &[19u32, 20]), &mat.row_slice(4, 2, 4));
}

#[test]
fn test_col_slice_dmat() {
    let mat = DMat::from_row_vec(
        8,
        4,
        &[
            1u32,2,  3,  4,
            5,   6,  7,  8,
            9,   10, 11, 12,
            13,  14, 15, 16,
            17,  18, 19, 20,
            21,  22, 23, 24,
            25,  26, 27, 28,
            29,  30, 31, 32
        ]
    );

    assert_eq!(&DVec::from_slice(8, &[1u32, 5, 9, 13, 17, 21, 25, 29]), &mat.col_slice(0, 0, 8));
    assert_eq!(&DVec::from_slice(3, &[1u32, 5, 9]), &mat.col_slice(0, 0, 3));
    assert_eq!(&DVec::from_slice(5, &[11u32, 15, 19, 23, 27]), &mat.col_slice(2, 2, 7));
    assert_eq!(&DVec::from_slice(2, &[28u32, 32]), &mat.col_slice(3, 6, 8));
}

#[test]
fn test_dmat_from_vec() {
    let mat1 = DMat::from_row_vec(
        8,
        4,
        &[
            1i32, 2,  3,  4,
            5,    6,  7,  8,
            9,    10, 11, 12,
            13,   14, 15, 16,
            17,   18, 19, 20,
            21,   22, 23, 24,
            25,   26, 27, 28,
            29,   30, 31, 32
        ]
    );

    let mat2 = DMat::from_col_vec(
        8,
        4,
        &[
            1i32, 5, 9,  13, 17, 21, 25, 29, 
            2i32, 6, 10, 14, 18, 22, 26, 30,
            3i32, 7, 11, 15, 19, 23, 27, 31, 
            4i32, 8, 12, 16, 20, 24, 28, 32
        ]
    );

    println!("mat1: {:?}, mat2: {:?}", mat1, mat2);

    assert!(mat1 == mat2);
}

#[test]
fn test_dmat_addition() {
    let mat1 = DMat::from_row_vec(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMat::from_row_vec(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMat::from_row_vec(
        2,
        2,
        &[
            11.0, 22.0,
            33.0, 44.0
        ]
    );

    assert!((mat1 + mat2) == res);
}

#[test]
fn test_dmat_multiplication() {
   let mat1 = DMat::from_row_vec(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMat::from_row_vec(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMat::from_row_vec(
        2,
        2,
        &[
            70.0, 100.0,
            150.0, 220.0
        ]
    );

    assert!((mat1 * mat2) == res);
}

// Tests multiplication of rectangular (non-square) matrices.
#[test]
fn test_dmat_multiplication_rect() {
    let mat1 = DMat::from_row_vec(
        1,
        2,
        &[
            1.0, 2.0,
        ]
    );

    let mat2 = DMat::from_row_vec(
        2,
        3,
        &[
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]
    );

    let res = DMat::from_row_vec(
        1,
        3,
        &[
            15.0, 18.0, 21.0,
        ]
    );

   assert!((mat1.clone() * mat2.clone()) == res);
   assert!((&mat1 * mat2.clone()) == res);
   assert!((mat1.clone() * &mat2) == res);
   assert!((&mat1 * &mat2) == res);
}

#[test]
fn test_dmat_subtraction() {
    let mat1 = DMat::from_row_vec(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMat::from_row_vec(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMat::from_row_vec(
        2,
        2,
        &[
            -09.0, -18.0,
            -27.0, -36.0
        ]
    );

    assert!((mat1 - mat2) == res);
}

/* FIXME: review qr decomposition to make it work with DMat.
#[test]
fn test_qr() {
    for _ in 0usize .. 10 {
        let dim1: usize = random();
        let dim2: usize = random();
        let rows = min(40, max(dim1, dim2));
        let cols = min(40, min(dim1, dim2));
        let randmat: DMat<f64> = DMat::new_random(rows, cols);
        let (q, r) = na::qr(&randmat);
        let recomp = q * r;

        assert!(na::approx_eq(&randmat,  &recomp));
    }
}
*/

#[test]
fn test_qr_mat1() {
    test_qr_impl!(Mat1<f64>);
}

#[test]
fn test_qr_mat2() {
    test_qr_impl!(Mat2<f64>);
}

#[test]
fn test_qr_mat3() {
    test_qr_impl!(Mat3<f64>);
}

#[test]
fn test_qr_mat4() {
    test_qr_impl!(Mat4<f64>);
}

#[test]
fn test_qr_mat5() {
    test_qr_impl!(Mat5<f64>);
}

#[test]
fn test_qr_mat6() {
    test_qr_impl!(Mat6<f64>);
}

// NOTE: deactivated until we get a better convergence rate.
// #[test]
// fn test_eigen_qr_mat1() {
//     test_eigen_qr_impl!(Mat1<f64>);
// }
// 
// #[test]
// fn test_eigen_qr_mat2() {
//     test_eigen_qr_impl!(Mat2<f64>);
// }
// 
// #[test]
// fn test_eigen_qr_mat3() {
//     test_eigen_qr_impl!(Mat3<f64>);
// }
// 
// #[test]
// fn test_eigen_qr_mat4() {
//     test_eigen_qr_impl!(Mat4<f64>);
// }
// 
// #[test]
// fn test_eigen_qr_mat5() {
//     test_eigen_qr_impl!(Mat5<f64>);
// }
// 
// #[test]
// fn test_eigen_qr_mat6() {
//     test_eigen_qr_impl!(Mat6<f64>);
// }

#[test]
fn test_from_fn() {
    let actual: DMat<usize> = DMat::from_fn(3, 4, |i, j| 10 * i + j);
    let expected: DMat<usize> = DMat::from_row_vec(3, 4, 
                                                  &[ 0_0, 0_1, 0_2, 0_3,
                                                     1_0, 1_1, 1_2, 1_3,
                                                     2_0, 2_1, 2_2, 2_3 ]);

    assert_eq!(actual, expected);
}

#[test]
fn test_row_3() {
    let mat = Mat3::new(0.0f32, 1.0, 2.0,
                        3.0,    4.0, 5.0,
                        6.0,    7.0, 8.0);
    let second_row = mat.row(1);
    let second_col = mat.col(1);

    assert!(second_row == Vec3::new(3.0, 4.0, 5.0));
    assert!(second_col == Vec3::new(1.0, 4.0, 7.0));
}

#[test]
fn test_persp() {
    let mut p  = Persp3::new(42.0f64, 0.5, 1.5, 10.0);
    let mut pm = PerspMat3::new(42.0f64, 0.5, 1.5, 10.0);
    assert!(p.to_mat() == pm.to_mat());
    assert!(p.aspect() == 42.0);
    assert!(p.fov()    == 0.5);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.aspect(), &42.0));
    assert!(na::approx_eq(&pm.fov(),    &0.5));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_fov(0.1);
    pm.set_fov(0.1);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_znear(24.0);
    pm.set_znear(24.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_aspect(23.0);
    pm.set_aspect(23.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    assert!(p.aspect() == 23.0);
    assert!(p.fov()    == 0.1);
    assert!(p.znear()  == 24.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.aspect(), &23.0));
    assert!(na::approx_eq(&pm.fov(),    &0.1));
    assert!(na::approx_eq(&pm.znear(),  &24.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

#[test]
fn test_ortho() {
    let mut p  = Ortho3::new(42.0f64, 0.5, 1.5, 10.0);
    let mut pm = OrthoMat3::new(42.0f64, 0.5, 1.5, 10.0);
    assert!(p.to_mat() == pm.to_mat());
    assert!(p.width()  == 42.0);
    assert!(p.height() == 0.5);
    assert!(p.znear()  == 1.5);
    assert!(p.zfar()   == 10.0);
    assert!(na::approx_eq(&pm.width(),  &42.0));
    assert!(na::approx_eq(&pm.height(), &0.5));
    assert!(na::approx_eq(&pm.znear(),  &1.5));
    assert!(na::approx_eq(&pm.zfar(),   &10.0));

    p.set_width(0.1);
    pm.set_width(0.1);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_znear(24.0);
    pm.set_znear(24.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_zfar(61.0);
    pm.set_zfar(61.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    p.set_height(23.0);
    pm.set_height(23.0);
    assert!(na::approx_eq(&p.to_mat(), pm.as_mat()));

    assert!(p.height() == 23.0);
    assert!(p.width()  == 0.1);
    assert!(p.znear()  == 24.0);
    assert!(p.zfar()   == 61.0);
    assert!(na::approx_eq(&pm.height(), &23.0));
    assert!(na::approx_eq(&pm.width(),  &0.1));
    assert!(na::approx_eq(&pm.znear(),  &24.0));
    assert!(na::approx_eq(&pm.zfar(),   &61.0));
}

#[test]
fn test_cholesky_const() {
    
    let a : Mat3<f64> = Mat3::<f64>::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0);
    let g : Mat3<f64> = Mat3::<f64>::new(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_ok());

    let v = result.unwrap();
    assert!(na::approx_eq(&v, &g));

    let recomp = v * na::transpose(&v);
    assert!(na::approx_eq(&recomp, &a));
}

#[test]
fn test_cholesky_not_spd() {
    
    let a : Mat3<f64> = Mat3::<f64>::new(1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_err());
}

#[test]
fn test_cholesky_not_symmetric() {
    
    let a : Mat2<f64> = Mat2::<f64>::new(1.0, 1.0, -1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_err());
}

#[test]
fn test_cholesky_mat1() {
    test_cholesky_impl!(Mat1<f64>);
}

#[test]
fn test_cholesky_mat2() {
    test_cholesky_impl!(Mat2<f64>);
}

#[test]
fn test_cholesky_mat3() {
    test_cholesky_impl!(Mat3<f64>);
}

#[test]
fn test_cholesky_mat4() {
    test_cholesky_impl!(Mat4<f64>);
}

#[test]
fn test_cholesky_mat5() {
    test_cholesky_impl!(Mat5<f64>);
}

#[test]
fn test_cholesky_mat6() {
    test_cholesky_impl!(Mat6<f64>);
}

#[test]
fn test_transpose_square_mat() {
    let col_major_mat = &[0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 1, 2, 3];
    let num_rows = 4;
    let num_cols = 4;
    let mut mat = DMat::from_col_vec(num_rows, num_cols, col_major_mat);
    mat.transpose_mut();
    for i in 0..num_rows {
        assert_eq!(&[0, 1, 2, 3], mat.row_slice(i, 0, num_cols).as_slice());
    }
}
