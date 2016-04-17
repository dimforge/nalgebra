extern crate nalgebra as na;
extern crate rand;

use rand::random;
use na::{Rotation2, Rotation3, Isometry2, Isometry3, Similarity2, Similarity3, Vector3, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6, DMatrix, DVector,
         Row, Column, Diagonal, Transpose, RowSlice, ColumnSlice, Shape};

macro_rules! test_inverse_mat_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmatrix : $t = random();

      match na::inverse(&randmatrix) {
          None    => { },
          Some(i) => {
              assert!(na::approx_eq(&(i * randmatrix), &na::one()))
          }
      }
    }
  );
);

macro_rules! test_transpose_mat_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmatrix : $t = random();

      assert!(na::transpose(&na::transpose(&randmatrix)) == randmatrix);
    }
  );
);

macro_rules! test_qr_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      let randmatrix : $t = random();

      let (q, r) = na::qr(&randmatrix);
      let recomp = q * r;

      assert!(na::approx_eq(&randmatrix,  &recomp));
    }
  );
);

macro_rules! test_cholesky_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      
      // construct symmetric positive definite matrix
      let mut randmatrix : $t = random();
      let mut diagmatrix : $t = Diagonal::from_diag(&na::diagonal(&randmatrix));

      diagmatrix = na::abs(&diagmatrix) + 1.0;
      randmatrix = randmatrix * diagmatrix * na::transpose(&randmatrix);

      let result = na::cholesky(&randmatrix);

      assert!(result.is_ok());

      let v = result.unwrap();
      let recomp = v * na::transpose(&v);
      assert!(na::approx_eq(&randmatrix,  &recomp));
    }
  );
);

macro_rules! test_hessenberg_impl(
  ($t: ty) => (
    for _ in 0usize .. 10000 {
      
      let randmatrix : $t = random();

      let (q, h) = na::hessenberg(&randmatrix);
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

      assert!(na::approx_eq(&randmatrix,  &recomp));
    }
  );
);

macro_rules! test_eigen_qr_impl(
    ($t: ty) => {
        for _ in 0usize .. 10000 {
            let randmatrix : $t = random();
            // Make it symetric so that we can recompose the matrix to test at the end.
            let randmatrix = na::transpose(&randmatrix) * randmatrix;
            let (eigenvectors, eigenvalues) = na::eigen_qr(&randmatrix, &1e-13, 100);
 
            let diagonal: $t = Diagonal::from_diag(&eigenvalues);
            let recomp = eigenvectors * diagonal * na::transpose(&eigenvectors);
            println!("eigenvalues: {:?}", eigenvalues);
            println!("   matrix: {:?}", randmatrix);
            println!("recomp: {:?}", recomp);

            assert!(na::approx_eq_eps(&randmatrix,  &recomp, &1.0e-2));
        }

        for _ in 0usize .. 10000 {
            let randmatrix : $t = random();
            // Take only diagonal part
            let randmatrix: $t = Diagonal::from_diag(&randmatrix.diagonal());
            let (eigenvectors, eigenvalues) = na::eigen_qr(&randmatrix, &1e-13, 100);
 
            let diagonal: $t = Diagonal::from_diag(&eigenvalues);
            let recomp = eigenvectors * diagonal * na::transpose(&eigenvectors);
            println!("eigenvalues: {:?}", eigenvalues);
            println!("   matrix: {:?}", randmatrix);
            println!("recomp: {:?}", recomp);

            assert!(na::approx_eq_eps(&randmatrix,  &recomp, &1.0e-2));
        }
    }
);

#[test]
fn test_transpose_mat1() {
    test_transpose_mat_impl!(Matrix1<f64>);
}

#[test]
fn test_transpose_mat2() {
    test_transpose_mat_impl!(Matrix2<f64>);
}

#[test]
fn test_transpose_mat3() {
    test_transpose_mat_impl!(Matrix3<f64>);
}

#[test]
fn test_transpose_mat4() {
    test_transpose_mat_impl!(Matrix4<f64>);
}

#[test]
fn test_transpose_mat5() {
    test_transpose_mat_impl!(Matrix5<f64>);
}

#[test]
fn test_transpose_mat6() {
    test_transpose_mat_impl!(Matrix6<f64>);
}

#[test]
fn test_inverse_mat1() {
    test_inverse_mat_impl!(Matrix1<f64>);
}

#[test]
fn test_inverse_mat2() {
    test_inverse_mat_impl!(Matrix2<f64>);
}

#[test]
fn test_inverse_mat3() {
    test_inverse_mat_impl!(Matrix3<f64>);
}

#[test]
fn test_inverse_mat4() {
    test_inverse_mat_impl!(Matrix4<f64>);
}

#[test]
fn test_inverse_mat5() {
    test_inverse_mat_impl!(Matrix5<f64>);
}

#[test]
fn test_inverse_mat6() {
    test_inverse_mat_impl!(Matrix6<f64>);
}

#[test]
fn test_inverse_rot2() {
    test_inverse_mat_impl!(Rotation2<f64>);
}

#[test]
fn test_inverse_rot3() {
    test_inverse_mat_impl!(Rotation3<f64>);
}

#[test]
fn test_inverse_iso2() {
    test_inverse_mat_impl!(Isometry2<f64>);
}

#[test]
fn test_inverse_iso3() {
    test_inverse_mat_impl!(Isometry3<f64>);
}

#[test]
fn test_inverse_sim2() {
    test_inverse_mat_impl!(Similarity2<f64>);
}

#[test]
fn test_inverse_sim3() {
    test_inverse_mat_impl!(Similarity3<f64>);
}

#[test]
fn test_index_mat2() {
  let matrix: Matrix2<f64> = random();

  assert!(matrix[(0, 1)] == na::transpose(&matrix)[(1, 0)]);
}


#[test]
fn test_mean_dmatrix() {
    let matrix = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0f64, 2.0, 3.0,
            4.0f64, 5.0, 6.0,
            7.0f64, 8.0, 9.0,
        ]
    );

    assert!(na::approx_eq(&na::mean(&matrix), &DVector::from_slice(3, &[4.0f64, 5.0, 6.0])));
}

#[test]
fn test_covariance_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    let expected = DMatrix::from_row_vector(
        3,
        3,
        &[
            0.025f64,   0.0075,  0.00175,
            0.0075f64,  0.007,   0.00135,
            0.00175f64, 0.00135, 0.00043
        ]
    );

    assert!(na::approx_eq(&na::covariance(&matrix), &expected));
}

#[test]
fn test_transpose_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    assert!(na::transpose(&na::transpose(&matrix)) == matrix);
}

#[test]
fn test_row_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    assert_eq!(&DVector::from_slice(4, &[1u32,  2,  3,  4]),  &matrix.row(0));
    assert_eq!(&DVector::from_slice(4, &[5u32,  6,  7,  8]),  &matrix.row(1));
    assert_eq!(&DVector::from_slice(4, &[9u32,  10, 11, 12]), &matrix.row(2));
    assert_eq!(&DVector::from_slice(4, &[13u32, 14, 15, 16]), &matrix.row(3));
    assert_eq!(&DVector::from_slice(4, &[17u32, 18, 19, 20]), &matrix.row(4));
    assert_eq!(&DVector::from_slice(4, &[21u32, 22, 23, 24]), &matrix.row(5));
    assert_eq!(&DVector::from_slice(4, &[25u32, 26, 27, 28]), &matrix.row(6));
    assert_eq!(&DVector::from_slice(4, &[29u32, 30, 31, 32]), &matrix.row(7));
}

#[test]
fn test_row_slice_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    assert_eq!(&DVector::from_slice(4, &[1u32, 2, 3, 4]), &matrix.row_slice(0, 0, 4));
    assert_eq!(&DVector::from_slice(2, &[1u32, 2]), &matrix.row_slice(0, 0, 2));
    assert_eq!(&DVector::from_slice(2, &[10u32, 11]), &matrix.row_slice(2, 1, 3));
    assert_eq!(&DVector::from_slice(2, &[19u32, 20]), &matrix.row_slice(4, 2, 4));
}

#[test]
fn test_col_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    assert_eq!(&DVector::from_slice(8, &[1u32, 5, 9,  13, 17, 21, 25, 29]), &matrix.column(0));
    assert_eq!(&DVector::from_slice(8, &[2u32, 6, 10, 14, 18, 22, 26, 30]), &matrix.column(1));
    assert_eq!(&DVector::from_slice(8, &[3u32, 7, 11, 15, 19, 23, 27, 31]), &matrix.column(2));
    assert_eq!(&DVector::from_slice(8, &[4u32, 8, 12, 16, 20, 24, 28, 32]), &matrix.column(3));
}

#[test]
fn test_col_slice_dmatrix() {
    let matrix = DMatrix::from_row_vector(
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

    assert_eq!(&DVector::from_slice(8, &[1u32, 5, 9, 13, 17, 21, 25, 29]), &matrix.col_slice(0, 0, 8));
    assert_eq!(&DVector::from_slice(3, &[1u32, 5, 9]), &matrix.col_slice(0, 0, 3));
    assert_eq!(&DVector::from_slice(5, &[11u32, 15, 19, 23, 27]), &matrix.col_slice(2, 2, 7));
    assert_eq!(&DVector::from_slice(2, &[28u32, 32]), &matrix.col_slice(3, 6, 8));
}

#[test]
fn test_dmat_from_vector() {
    let mat1 = DMatrix::from_row_vector(
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

    let mat2 = DMatrix::from_col_vector(
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
    let mat1 = DMatrix::from_row_vector(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMatrix::from_row_vector(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMatrix::from_row_vector(
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
   let mat1 = DMatrix::from_row_vector(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMatrix::from_row_vector(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMatrix::from_row_vector(
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
    let mat1 = DMatrix::from_row_vector(
        1,
        2,
        &[
            1.0, 2.0,
        ]
    );

    let mat2 = DMatrix::from_row_vector(
        2,
        3,
        &[
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]
    );

    let res = DMatrix::from_row_vector(
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
    let mat1 = DMatrix::from_row_vector(
        2,
        2,
        &[
            1.0, 2.0,
            3.0, 4.0
        ]
    );

    let mat2 = DMatrix::from_row_vector(
        2,
        2,
        &[
            10.0, 20.0,
            30.0, 40.0
        ]
    );

    let res = DMatrix::from_row_vector(
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
    let matrix = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(matrix.column(1) == DVector::from_slice(3, &[2.0, 5.0, 8.0]));
}

#[test]
fn test_dmat_set_col() {
    let mut matrix = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    matrix.set_col(1, DVector::from_slice(3, &[12.0, 15.0, 18.0]));

    let expected = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 12.0, 3.0,
            4.0, 15.0, 6.0,
            7.0, 18.0, 9.0,
        ]
    );

    assert!(matrix == expected);
}

#[test]
fn test_dmat_row() {
    let matrix = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(matrix.row(1) == DVector::from_slice(3, &[4.0, 5.0, 6.0]));
}

#[test]
fn test_dmat_set_row() {
    let mut matrix = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
    );

    matrix.set_row(1, DVector::from_slice(3, &[14.0, 15.0, 16.0]));

    let expected = DMatrix::from_row_vector(
        3,
        3,
        &[
            1.0, 2.0, 3.0,
            14.0, 15.0, 16.0,
            7.0, 8.0, 9.0,
        ]
    );

    assert!(matrix == expected);
}

/* FIXME: review qr decomposition to make it work with DMatrix.
#[test]
fn test_qr() {
    for _ in 0usize .. 10 {
        let dim1: usize = random();
        let dim2: usize = random();
        let rows = min(40, max(dim1, dim2));
        let cols = min(40, min(dim1, dim2));
        let randmatrix: DMatrix<f64> = DMatrix::new_random(rows, cols);
        let (q, r) = na::qr(&randmatrix);
        let recomp = q * r;

        assert!(na::approx_eq(&randmatrix,  &recomp));
    }
}
*/

#[test]
fn test_qr_mat1() {
    test_qr_impl!(Matrix1<f64>);
}

#[test]
fn test_qr_mat2() {
    test_qr_impl!(Matrix2<f64>);
}

#[test]
fn test_qr_mat3() {
    test_qr_impl!(Matrix3<f64>);
}

#[test]
fn test_qr_mat4() {
    test_qr_impl!(Matrix4<f64>);
}

#[test]
fn test_qr_mat5() {
    test_qr_impl!(Matrix5<f64>);
}

#[test]
fn test_qr_mat6() {
    test_qr_impl!(Matrix6<f64>);
}

#[test]
fn test_eigen_qr_mat1() {
    test_eigen_qr_impl!(Matrix1<f64>);
}

#[test]
fn test_eigen_qr_mat2() {
    test_eigen_qr_impl!(Matrix2<f64>);
}

#[test]
fn test_eigen_qr_mat3() {
    test_eigen_qr_impl!(Matrix3<f64>);
}

#[test]
fn test_eigen_qr_mat4() {
    test_eigen_qr_impl!(Matrix4<f64>);
}

#[test]
fn test_eigen_qr_mat5() {
    test_eigen_qr_impl!(Matrix5<f64>);
}

#[test]
fn test_eigen_qr_mat6() {
    test_eigen_qr_impl!(Matrix6<f64>);
}

#[test]
fn test_from_fn() {
    let actual: DMatrix<usize> = DMatrix::from_fn(3, 4, |i, j| 10 * i + j);
    let expected: DMatrix<usize> = DMatrix::from_row_vector(3, 4, 
                                                  &[ 0_0, 0_1, 0_2, 0_3,
                                                     1_0, 1_1, 1_2, 1_3,
                                                     2_0, 2_1, 2_2, 2_3 ]);

    assert_eq!(actual, expected);
}

#[test]
fn test_row_3() {
    let matrix = Matrix3::new(0.0f32, 1.0, 2.0,
                        3.0,    4.0, 5.0,
                        6.0,    7.0, 8.0);
    let second_row = matrix.row(1);
    let second_col = matrix.column(1);

    assert!(second_row == Vector3::new(3.0, 4.0, 5.0));
    assert!(second_col == Vector3::new(1.0, 4.0, 7.0));
}

#[test]
fn test_cholesky_const() {
    
    let a : Matrix3<f64> = Matrix3::<f64>::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0);
    let g : Matrix3<f64> = Matrix3::<f64>::new(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_ok());

    let v = result.unwrap();
    assert!(na::approx_eq(&v, &g));

    let recomp = v * na::transpose(&v);
    assert!(na::approx_eq(&recomp, &a));
}

#[test]
fn test_cholesky_not_spd() {
    
    let a : Matrix3<f64> = Matrix3::<f64>::new(1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_err());
}

#[test]
fn test_cholesky_not_symmetric() {
    
    let a : Matrix2<f64> = Matrix2::<f64>::new(1.0, 1.0, -1.0, 1.0);

    let result = na::cholesky(&a);

    assert!(result.is_err());
}

#[test]
fn test_cholesky_mat1() {
    test_cholesky_impl!(Matrix1<f64>);
}

#[test]
fn test_cholesky_mat2() {
    test_cholesky_impl!(Matrix2<f64>);
}

#[test]
fn test_cholesky_mat3() {
    test_cholesky_impl!(Matrix3<f64>);
}

#[test]
fn test_cholesky_mat4() {
    test_cholesky_impl!(Matrix4<f64>);
}

#[test]
fn test_cholesky_mat5() {
    test_cholesky_impl!(Matrix5<f64>);
}

#[test]
fn test_cholesky_mat6() {
    test_cholesky_impl!(Matrix6<f64>);
}

#[test]
fn test_hessenberg_mat1() {
    test_hessenberg_impl!(Matrix1<f64>);
}

#[test]
fn test_hessenberg_mat2() {
    test_hessenberg_impl!(Matrix2<f64>);
}

#[test]
fn test_hessenberg_mat3() {
    test_hessenberg_impl!(Matrix3<f64>);
}

#[test]
fn test_hessenberg_mat4() {
    test_hessenberg_impl!(Matrix4<f64>);
}

#[test]
fn test_hessenberg_mat5() {
    test_hessenberg_impl!(Matrix5<f64>);
}

#[test]
fn test_hessenberg_mat6() {
    test_hessenberg_impl!(Matrix6<f64>);
}

#[test]
fn test_transpose_square_matrix() {
    let col_major_matrix = &[0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 1, 2, 3];
    let num_rows = 4;
    let num_cols = 4;
    let mut matrix = DMatrix::from_col_vector(num_rows, num_cols, col_major_matrix);
    matrix.transpose_mut();
    for i in 0..num_rows {
        assert_eq!(&[0, 1, 2, 3], &matrix.row_slice(i, 0, num_cols)[..]);
    }
}

#[test]
fn test_outer_dvector() {
    let vector = DVector::from_slice(5, &[ 1.0, 2.0, 3.0, 4.0, 5.0 ]);
    let row = DMatrix::from_row_vector(1, 5, &vector[..]);

    assert_eq!(row.transpose() * row, na::outer(&vector, &vector))
}
