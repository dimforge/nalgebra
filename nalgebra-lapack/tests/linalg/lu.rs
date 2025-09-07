use crate::proptest::*;
use na::{DMatrix, DVector, Dim, Matrix, Matrix4x3, RawStorage, Vector4};
use nl::LU;
use num_traits::Zero;
use proptest::{prop_assert, proptest};
use std::cmp;

fn has_vanishing_diagonal_entries<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> bool
where
    T: Zero + PartialEq,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let dim = mat.nrows().min(mat.ncols());
    let zero = T::zero();
    for j in 0..dim {
        if mat[(j, j)] == zero {
            return true;
        }
    }
    return false;
}

proptest! {
    #[test]
    fn lup(m in dmatrix()) {
        let lup = LU::new(m.clone());
        let l = lup.l();
        let u = lup.u();
        let mut computed1 = &l * &u;
        lup.permute(&mut computed1);

        let computed2 = lup.p() * l * u;

        prop_assert!(relative_eq!(computed1, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(computed2, m, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_static(m in matrix3x5()) {
        let lup = LU::new(m);
        let l = lup.l();
        let u = lup.u();
        let mut computed1 = l * u;
        lup.permute(&mut computed1);

        let computed2 = lup.p() * l * u;

        prop_assert!(relative_eq!(computed1, m, epsilon = 1.0e-7));
        prop_assert!(relative_eq!(computed2, m, epsilon = 1.0e-7));
    }

    #[test]
    fn lu_solve(n in PROPTEST_MATRIX_DIM, nb in PROPTEST_MATRIX_DIM) {
        let n  = cmp::min(n, 25);  // To avoid slowing down the test too much.
        let nb = cmp::min(nb, 25); // To avoid slowing down the test too much.
        let m  = DMatrix::<f64>::new_random(n, n);

        let lup = LU::new(m.clone());
        let b1 = DVector::new_random(n);
        let b2 = DMatrix::new_random(n, nb);

        // this path is taken if the matrix is not detected as singular
        // during the LU decomposition.
        if let (Some(sol1), Some(sol2), Some(tr_sol1), Some(tr_sol2)) = (
            lup.solve(&b1),
            lup.solve(&b2),
            lup.solve_transpose(&b1),
            lup.solve_transpose(&b2)
        ) {
            prop_assert!(relative_eq!(&m * sol1, b1, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(&m * sol2, b2, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-5));
        } else {
            // If any solve returns None, it should indicate that the matrix is
            // singular, which we can verify by checking that one or more
            // diagonal elemenst of U are singular.
            prop_assert!(has_vanishing_diagonal_entries(&lup.u()))
        }
    }

    #[test]
    fn lu_solve_static(m in matrix4(), b1 in vector4(), b2 in matrix4x5()) {
        let lup = LU::new(m);

        if let (Some(sol1), Some(sol2), Some(tr_sol1), Some(tr_sol2)) = (
            lup.solve(&b1),
            lup.solve(&b2),
            lup.solve_transpose(&b1),
            lup.solve_transpose(&b2)
        ) {
            prop_assert!(relative_eq!(m * sol1, b1, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(m * sol2, b2, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(m.transpose() * tr_sol1, b1, epsilon = 1.0e-5));
            prop_assert!(relative_eq!(m.transpose() * tr_sol2, b2, epsilon = 1.0e-5));
        } else {
            prop_assert!(has_vanishing_diagonal_entries(&lup.u()))
        }
    }

    #[test]
    fn lu_inverse(n in PROPTEST_MATRIX_DIM) {
        let n = cmp::min(n, 25); // To avoid slowing down the test too much.
        let m = DMatrix::<f64>::new_random(n, n);

        if let Some(m1) = LU::new(m.clone()).inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-7) && id2.is_identity(1.0e-7));
        }
    }

    #[test]
    fn lu_inverse_static(m in matrix4()) {
        if let Some(m1) = LU::new(m.clone()).inverse() {
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            prop_assert!(id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5))
        }
    }
}
