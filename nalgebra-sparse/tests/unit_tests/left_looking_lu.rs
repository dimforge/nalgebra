use crate::unit_tests::cholesky::positive_definite;

use nalgebra_sparse::factorization::LeftLookingLUFactorization;

use matrixcompare::{assert_matrix_eq, prop_assert_matrix_eq};
use proptest::prelude::*;

use crate::common::value_strategy;
use nalgebra::proptest::matrix;
use nalgebra_sparse::CscMatrix;

proptest! {
  // Note that positive definite matrices are guaranteed to be invertible.
  // That's why they're used here, but it's not necessary for a matrix to be pd to be
  // invertible.
  #[test]
  fn lu_for_positive_def_matrices(
    matrix in positive_definite()
  ) {
    let lu = LeftLookingLUFactorization::new(&matrix);

    let l = lu.l();
    prop_assert!(l.triplet_iter().all(|(i, j, _)| j <= i));
    let u = lu.u();
    prop_assert!(u.triplet_iter().all(|(i, j, _)| j >= i));
    prop_assert_matrix_eq!(l * u, matrix, comp = abs, tol = 1e-7);
  }

  #[test]
  fn lu_solve_correct_for_positive_def_matrices(
    (matrix, b) in positive_definite()
      .prop_flat_map(|csc| {
        let rhs = matrix(value_strategy::<f64>(), csc.nrows(), 1);
        (Just(csc), rhs)
      })
  ) {
    let lu = LeftLookingLUFactorization::new(&matrix);

    let l = lu.l();
    prop_assert!(l.triplet_iter().all(|(i, j, _)| j <= i));

    let mut x_l = b.clone_owned();
    l.dense_lower_triangular_solve(b.as_slice(), x_l.as_mut_slice(), true);

    prop_assert_matrix_eq!(l * x_l, b, comp=abs, tol=1e-12);

    let u = lu.u();
    prop_assert!(u.triplet_iter().all(|(i, j, _)| j >= i));

    let mut x_u = b.clone_owned();
    u.dense_upper_triangular_solve(b.as_slice(), x_u.as_mut_slice());
    prop_assert_matrix_eq!(u * x_u, b, comp = abs, tol = 1e-7);

    let x = lu.solve(b.as_slice());
    prop_assert_matrix_eq!(&matrix * &x, b, comp=abs, tol=1e-12);
  }
}

#[test]
fn minimized_lu() {
    let major_offsets = vec![0, 1, 3, 4, 5, 8];
    let minor_indices = vec![0, 1, 4, 2, 3, 1, 2, 4];
    let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0];
    assert_eq!(minor_indices.len(), values.len());
    let mat = CscMatrix::try_from_unsorted_csc_data(5, 5, major_offsets, minor_indices, values);
    let mat = mat.unwrap();

    let lu = LeftLookingLUFactorization::new(&mat);

    let l = lu.l();
    assert!(l.triplet_iter().all(|(i, j, _)| j <= i));
    let u = lu.u();
    assert!(u.triplet_iter().all(|(i, j, _)| j >= i));
    assert_matrix_eq!(l * u, mat, comp = abs, tol = 1e-7);
}
