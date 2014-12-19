#![feature(macro_rules)]

extern crate "nalgebra" as na;

use std::rand::random;
use na::{Vec1, Vec3, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6, Rot3, Persp3, PerspMat3, Ortho3, OrthoMat3,
         DMat, DVec, Row, Col, BaseFloat};

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    for _ in range(0u, 10000) {
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
    for _ in range(0u, 10000) {
      let randmat : $t = random();

      assert!(na::transpose(&na::transpose(&randmat)) == randmat);
    }
  );
);

macro_rules! test_qr_impl(
  ($t: ty) => (
    for _ in range(0u, 10000) {
      let randmat : $t = random();

      let (q, r) = na::qr(&randmat);
      let recomp = q * r;

      assert!(na::approx_eq(&randmat,  &recomp));
    }
  );
);

// NOTE: deactivated untile we get a better convergence rate.
// macro_rules! test_eigen_qr_impl(
//     ($t: ty) => {
//         for _ in range(0u, 10000) {
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
    for _ in range(0u, 10000) {
        let randmat: na::Rot2<f64> = na::one();
        let ang    = Vec1::new(na::abs(&random::<f64>()) % BaseFloat::pi());

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
    for _ in range(0u, 10000) {
        let randmat: Rot3<f64> = na::one();
        let dir:     Vec3<f64> = random();
        let ang            = na::normalize(&dir) * (na::abs(&random::<f64>()) % BaseFloat::pi());
        let rot            = na::append_rotation(&randmat, &ang);

        assert!(na::approx_eq(&(na::transpose(&rot) * rot), &na::one()));
    }
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

    println!("mat1: {}, mat2: {}", mat1, mat2);

    assert!(mat1 == mat2);
}

/* FIXME: review qr decomposition to make it work with DMat.
#[test]
fn test_qr() {
    for _ in range(0u, 10) {
        let dim1: uint = random();
        let dim2: uint = random();
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
    let actual: DMat<uint> = DMat::from_fn(3, 4, |i, j| 10 * i + j);
    let expected: DMat<uint> = DMat::from_row_vec(3, 4, 
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
