extern crate nalgebra as na;
extern crate rand;

use rand::random;
use na::{Rot2, Rot3, Iso2, Iso3, Sim2, Sim3, Vec3, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6, DMat, DVec,
         Row, Col, Diag, Transpose, RowSlice, ColSlice, Shape};

macro_rules! test_inv_mat_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmat : $t = random();

      match na::inv(&randmat) {
          None    => { },
          Some(i) => {
              assert!(na::approx_eq(&(i * randmat), &na::one()))
          }
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

macro_rules! test_hessenberg_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      
      let randmat : $t = random();

      let (q, h) = na::hessenberg(&randmat);
      let recomp = q * h * na::transpose(&q);

      let (rows, cols) = h.shape();

      // Check if `h` has zero entries below the first subdiagonal
      if cols > 2 {
          for j in 0..(cols-2) {
              for i in (j+2)..rows {
                  assert!(na::approx_eq(&h[(i,j)], &0.0f64));
              }
          }
      }

      assert!(na::approx_eq(&randmat,  &recomp));
    }
  );
);

macro_rules! test_eigen_qr_impl(
    ($t: ty) => {
        for _ in 0usize .. 10000 {
            let randmat : $t = random();
            // Make it symetric so that we can recompose the matrix to test at the end.
            let randmat = na::transpose(&randmat) * randmat;
            let (eigenvectors, eigenvalues) = na::eigen_qr(&randmat, &1e-13, 100);
 
            let diag: $t = Diag::from_diag(&eigenvalues);
            let recomp = eigenvectors * diag * na::transpose(&eigenvectors);
            println!("eigenvalues: {:?}", eigenvalues);
            println!("   mat: {:?}", randmat);
            println!("recomp: {:?}", recomp);

            assert!(na::approx_eq_eps(&randmat,  &recomp, &1.0e-2));
        }

        for _ in 0usize .. 10000 {
            let randmat : $t = random();
            // Take only diagonal part
            let randmat: $t = Diag::from_diag(&randmat.diag());
            let (eigenvectors, eigenvalues) = na::eigen_qr(&randmat, &1e-13, 100);
 
            let diag: $t = Diag::from_diag(&eigenvalues);
            let recomp = eigenvectors * diag * na::transpose(&eigenvectors);
            println!("eigenvalues: {:?}", eigenvalues);
            println!("   mat: {:?}", randmat);
            println!("recomp: {:?}", recomp);

            assert!(na::approx_eq_eps(&randmat,  &recomp, &1.0e-2));
        }
    }
);

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
fn test_inv_rot2() {
    test_inv_mat_impl!(Rot2<f64>);
}

#[test]
fn test_inv_rot3() {
    test_inv_mat_impl!(Rot3<f64>);
}

#[test]
fn test_inv_iso2() {
    test_inv_mat_impl!(Iso2<f64>);
}

#[test]
fn test_inv_iso3() {
    test_inv_mat_impl!(Iso3<f64>);
}

#[test]
fn test_inv_sim2() {
    test_inv_mat_impl!(Sim2<f64>);
}

#[test]
fn test_inv_sim3() {
    test_inv_mat_impl!(Sim3<f64>);
}

#[test]
fn test_index_mat2() {
  let mat: Mat2<f64> = random();

  assert!(mat[(0, 1)] == na::transpose(&mat)[(1, 0)]);
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
fn test_row_dmat() {
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

    assert_eq!(&DVec::from_slice(4, &[1u32,  2,  3,  4]),  &mat.row(0));
    assert_eq!(&DVec::from_slice(4, &[5u32,  6,  7,  8]),  &mat.row(1));
    assert_eq!(&DVec::from_slice(4, &[9u32,  10, 11, 12]), &mat.row(2));
    assert_eq!(&DVec::from_slice(4, &[13u32, 14, 15, 16]), &mat.row(3));
    assert_eq!(&DVec::from_slice(4, &[17u32, 18, 19, 20]), &mat.row(4));
    assert_eq!(&DVec::from_slice(4, &[21u32, 22, 23, 24]), &mat.row(5));
    assert_eq!(&DVec::from_slice(4, &[25u32, 26, 27, 28]), &mat.row(6));
    assert_eq!(&DVec::from_slice(4, &[29u32, 30, 31, 32]), &mat.row(7));
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
fn test_col_dmat() {
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

    assert_eq!(&DVec::from_slice(8, &[1u32, 5, 9,  13, 17, 21, 25, 29]), &mat.col(0));
    assert_eq!(&DVec::from_slice(8, &[2u32, 6, 10, 14, 18, 22, 26, 30]), &mat.col(1));
    assert_eq!(&DVec::from_slice(8, &[3u32, 7, 11, 15, 19, 23, 27, 31]), &mat.col(2));
    assert_eq!(&DVec::from_slice(8, &[4u32, 8, 12, 16, 20, 24, 28, 32]), &mat.col(3));
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

#[test]
fn test_dmat_col() {
    let mat = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(mat.col(1) == DVec::from_slice(3, &[2.0, 5.0, 8.0]));
}

#[test]
fn test_dmat_set_col() {
    let mut mat = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    mat.set_col(1, DVec::from_slice(3, &[12.0, 15.0, 18.0]));

    let expected = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 12.0, 3.0,
            4.0, 15.0, 6.0,
            7.0, 18.0, 9.0,
        ]
    );

    assert!(mat == expected);
}

#[test]
fn test_dmat_row() {
    let mat = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(mat.row(1) == DVec::from_slice(3, &[4.0, 5.0, 6.0]));
}

#[test]
fn test_dmat_set_row() {
    let mut mat = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    mat.set_row(1, DVec::from_slice(3, &[14.0, 15.0, 16.0]));

    let expected = DMat::from_row_vec(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            14.0, 15.0, 16.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(mat == expected);
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

#[test]
fn test_eigen_qr_mat1() {
    test_eigen_qr_impl!(Mat1<f64>);
}

#[test]
fn test_eigen_qr_mat2() {
    test_eigen_qr_impl!(Mat2<f64>);
}

#[test]
fn test_eigen_qr_mat3() {
    test_eigen_qr_impl!(Mat3<f64>);
}

#[test]
fn test_eigen_qr_mat4() {
    test_eigen_qr_impl!(Mat4<f64>);
}

#[test]
fn test_eigen_qr_mat5() {
    test_eigen_qr_impl!(Mat5<f64>);
}

#[test]
fn test_eigen_qr_mat6() {
    test_eigen_qr_impl!(Mat6<f64>);
}

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
fn test_hessenberg_mat1() {
    test_hessenberg_impl!(Mat1<f64>);
}

#[test]
fn test_hessenberg_mat2() {
    test_hessenberg_impl!(Mat2<f64>);
}

#[test]
fn test_hessenberg_mat3() {
    test_hessenberg_impl!(Mat3<f64>);
}

#[test]
fn test_hessenberg_mat4() {
    test_hessenberg_impl!(Mat4<f64>);
}

#[test]
fn test_hessenberg_mat5() {
    test_hessenberg_impl!(Mat5<f64>);
}

#[test]
fn test_hessenberg_mat6() {
    test_hessenberg_impl!(Mat6<f64>);
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
        assert_eq!(&[0, 1, 2, 3], &mat.row_slice(i, 0, num_cols)[..]);
    }
}

#[test]
fn test_outer_dvec() {
    let vec = DVec::from_slice(5, &[ 1.0, 2.0, 3.0, 4.0, 5.0 ]);
    let row = DMat::from_row_vec(1, 5, &vec[..]);

    assert_eq!(row.transpose() * row, na::outer(&vec, &vec))
}
