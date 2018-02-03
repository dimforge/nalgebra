#![cfg_attr(rustfmt, rustfmt_skip)]

use na::Matrix3;

#[test]
fn full_piv_lu_simple() {
    let m = Matrix3::new(
        2.0, -1.0,  0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0);

    let lu = m.full_piv_lu();
    assert_eq!(lu.determinant(), 4.0);

    let (p, l, u, q) = lu.unpack();

    let mut lu = l * u;
    p.inv_permute_rows(&mut lu);
    q.inv_permute_columns(&mut lu);

    assert!(relative_eq!(m, lu, epsilon = 1.0e-7));
}

#[test]
fn full_piv_lu_simple_with_pivot() {
    let m = Matrix3::new(
        0.0, -1.0,  2.0,
       -1.0,  2.0, -1.0,
        2.0, -1.0,  0.0);

    let lu = m.full_piv_lu();
    assert_eq!(lu.determinant(), -4.0);

    let (p, l, u, q) = lu.unpack();

    let mut lu = l * u;
    p.inv_permute_rows(&mut lu);
    q.inv_permute_columns(&mut lu);

    assert!(relative_eq!(m, lu, epsilon = 1.0e-7));
}

#[cfg(feature = "arbitrary")]
mod quickcheck_tests {
    use std::cmp;
    use na::{DMatrix, Matrix4, Matrix4x3, Matrix5x3, Matrix3x5, DVector, Vector4};

    quickcheck! {
        fn full_piv_lu(m: DMatrix<f64>) -> bool {
            let mut m = m;
            if m.len() == 0 {
                m = DMatrix::new_random(1, 1);
            }

            let lu = m.clone().full_piv_lu();
            let (p, l, u, q) = lu.unpack();
            let mut lu = l * u;
            p.inv_permute_rows(&mut lu);
            q.inv_permute_columns(&mut lu);

            relative_eq!(m, lu, epsilon = 1.0e-7)
        }

        fn full_piv_lu_static_3_5(m: Matrix3x5<f64>) -> bool {
            let lu = m.full_piv_lu();
            let (p, l, u, q) = lu.unpack();
            let mut lu = l * u;
            p.inv_permute_rows(&mut lu);
            q.inv_permute_columns(&mut lu);

            relative_eq!(m, lu, epsilon = 1.0e-7)
        }

        fn full_piv_lu_static_5_3(m: Matrix5x3<f64>) -> bool {
            let lu = m.full_piv_lu();
            let (p, l, u, q) = lu.unpack();
            let mut lu = l * u;
            p.inv_permute_rows(&mut lu);
            q.inv_permute_columns(&mut lu);

            relative_eq!(m, lu, epsilon = 1.0e-7)
        }

        fn full_piv_lu_static_square(m: Matrix4<f64>) -> bool {
            let lu = m.full_piv_lu();
            let (p, l, u, q) = lu.unpack();
            let mut lu = l * u;
            p.inv_permute_rows(&mut lu);
            q.inv_permute_columns(&mut lu);

            relative_eq!(m, lu, epsilon = 1.0e-7)
        }

        fn full_piv_lu_solve(n: usize, nb: usize) -> bool {
            if n != 0 && nb != 0 {
                let n  = cmp::min(n, 50);  // To avoid slowing down the test too much.
                let nb = cmp::min(nb, 50); // To avoid slowing down the test too much.
                let m  = DMatrix::<f64>::new_random(n, n);

                let lu = m.clone().full_piv_lu();
                let b1 = DVector::new_random(n);
                let b2 = DMatrix::new_random(n, nb);

                let sol1 = lu.solve(&b1);
                let sol2 = lu.solve(&b2);

                return (sol1.is_none() || relative_eq!(&m * sol1.unwrap(), b1, epsilon = 1.0e-6)) &&
                       (sol2.is_none() || relative_eq!(&m * sol2.unwrap(), b2, epsilon = 1.0e-6))
            }

            return true;
        }

        fn full_piv_lu_solve_static(m: Matrix4<f64>) -> bool {
             let lu = m.full_piv_lu();
             let b1 = Vector4::new_random();
             let b2 = Matrix4x3::new_random();

             let sol1 = lu.solve(&b1);
             let sol2 = lu.solve(&b2);

             return (sol1.is_none() || relative_eq!(&m * sol1.unwrap(), b1, epsilon = 1.0e-6)) &&
                    (sol2.is_none() || relative_eq!(&m * sol2.unwrap(), b2, epsilon = 1.0e-6))
        }

        fn full_piv_lu_inverse(n: usize) -> bool {
            let n = cmp::max(1, cmp::min(n, 15)); // To avoid slowing down the test too much.
            let m = DMatrix::<f64>::new_random(n, n);

            let mut l = m.lower_triangle();
            let mut u = m.upper_triangle();

            // Ensure the matrix is well conditioned for inversion.
            l.fill_diagonal(1.0);
            u.fill_diagonal(1.0);
            let m = l * u;

            let m1  = m.clone().full_piv_lu().try_inverse().unwrap();
            let id1 = &m  * &m1;
            let id2 = &m1 * &m;

            return id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5);
        }

        fn full_piv_lu_inverse_static(m: Matrix4<f64>) -> bool {
            let lu = m.full_piv_lu();

            if let Some(m1)  = lu.try_inverse() {
                let id1 = &m  * &m1;
                let id2 = &m1 * &m;

                id1.is_identity(1.0e-5) && id2.is_identity(1.0e-5)
            }
            else {
                true
            }
        }
    }
}

/*
#[test]
fn swap_rows() {
    let mut m = Matrix5x3::new(
        11.0, 12.0, 13.0,
        21.0, 22.0, 23.0,
        31.0, 32.0, 33.0,
        41.0, 42.0, 43.0,
        51.0, 52.0, 53.0);

    let expected = Matrix5x3::new(
        11.0, 12.0, 13.0,
        41.0, 42.0, 43.0,
        31.0, 32.0, 33.0,
        21.0, 22.0, 23.0,
        51.0, 52.0, 53.0);

    m.swap_rows(1, 3);

    assert_eq!(m, expected);
}

#[test]
fn swap_columns() {
    let mut m = Matrix3x5::new(
        11.0, 12.0, 13.0, 14.0, 15.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        31.0, 32.0, 33.0, 34.0, 35.0);

    let expected = Matrix3x5::new(
        11.0, 14.0, 13.0, 12.0, 15.0,
        21.0, 24.0, 23.0, 22.0, 25.0,
        31.0, 34.0, 33.0, 32.0, 35.0);

    m.swap_columns(1, 3);

    assert_eq!(m, expected);
}

#[test]
fn remove_columns() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected1 = Matrix3x4::new(
        12, 13, 14, 15,
        22, 23, 24, 25,
        32, 33, 34, 35);

    let expected2 = Matrix3x4::new(
        11, 12, 13, 14,
        21, 22, 23, 24,
        31, 32, 33, 34);

    let expected3 = Matrix3x4::new(
        11, 12, 14, 15,
        21, 22, 24, 25,
        31, 32, 34, 35);

    assert_eq!(m.remove_column(0), expected1);
    assert_eq!(m.remove_column(4), expected2);
    assert_eq!(m.remove_column(2), expected3);

    let expected1 = Matrix3::new(
        13, 14, 15,
        23, 24, 25,
        33, 34, 35);

    let expected2 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33);

    let expected3 = Matrix3::new(
        11, 12, 15,
        21, 22, 25,
        31, 32, 35);

    assert_eq!(m.remove_fixed_columns::<U2>(0), expected1);
    assert_eq!(m.remove_fixed_columns::<U2>(3), expected2);
    assert_eq!(m.remove_fixed_columns::<U2>(2), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, U3, Dynamic, _> = m.remove_columns(3, 2);
    assert!(computed.eq(&expected2));
}


#[test]
fn remove_rows() {
    let m = Matrix5x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected1 = Matrix4x3::new(
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected2 = Matrix4x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43);

    let expected3 = Matrix4x3::new(
        11, 12, 13,
        21, 22, 23,
        41, 42, 43,
        51, 52, 53);

    assert_eq!(m.remove_row(0), expected1);
    assert_eq!(m.remove_row(4), expected2);
    assert_eq!(m.remove_row(2), expected3);

    let expected1 = Matrix3::new(
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected2 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33);

    let expected3 = Matrix3::new(
        11, 12, 13,
        21, 22, 23,
        51, 52, 53);

    assert_eq!(m.remove_fixed_rows::<U2>(0), expected1);
    assert_eq!(m.remove_fixed_rows::<U2>(3), expected2);
    assert_eq!(m.remove_fixed_rows::<U2>(2), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, Dynamic, U3, _> = m.remove_rows(3, 2);
    assert!(computed.eq(&expected2));
}


#[test]
fn insert_columns() {
    let m = Matrix5x3::new(
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53);

    let expected1 = Matrix5x4::new(
        0, 11, 12, 13,
        0, 21, 22, 23,
        0, 31, 32, 33,
        0, 41, 42, 43,
        0, 51, 52, 53);

    let expected2 = Matrix5x4::new(
        11, 12, 13, 0,
        21, 22, 23, 0,
        31, 32, 33, 0,
        41, 42, 43, 0,
        51, 52, 53, 0);

    let expected3 = Matrix5x4::new(
        11, 12, 0, 13,
        21, 22, 0, 23,
        31, 32, 0, 33,
        41, 42, 0, 43,
        51, 52, 0, 53);

    assert_eq!(m.insert_column(0, 0), expected1);
    assert_eq!(m.insert_column(3, 0), expected2);
    assert_eq!(m.insert_column(2, 0), expected3);

    let expected1 = Matrix5::new(
        0, 0, 11, 12, 13,
        0, 0, 21, 22, 23,
        0, 0, 31, 32, 33,
        0, 0, 41, 42, 43,
        0, 0, 51, 52, 53);

    let expected2 = Matrix5::new(
        11, 12, 13, 0, 0,
        21, 22, 23, 0, 0,
        31, 32, 33, 0, 0,
        41, 42, 43, 0, 0,
        51, 52, 53, 0, 0);

    let expected3 = Matrix5::new(
        11, 12, 0, 0, 13,
        21, 22, 0, 0, 23,
        31, 32, 0, 0, 33,
        41, 42, 0, 0, 43,
        51, 52, 0, 0, 53);

    assert_eq!(m.insert_fixed_columns::<U2>(0, 0), expected1);
    assert_eq!(m.insert_fixed_columns::<U2>(3, 0), expected2);
    assert_eq!(m.insert_fixed_columns::<U2>(2, 0), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, U5, Dynamic, _> = m.insert_columns(3, 2, 0);
    assert!(computed.eq(&expected2));
}


#[test]
fn insert_rows() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected1 = Matrix4x5::new(
         0,  0,  0,  0,  0,
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected2 = Matrix4x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
         0,  0,  0,  0,  0);

    let expected3 = Matrix4x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
         0,  0,  0,  0,  0,
        31, 32, 33, 34, 35);

    assert_eq!(m.insert_row(0, 0), expected1);
    assert_eq!(m.insert_row(3, 0), expected2);
    assert_eq!(m.insert_row(2, 0), expected3);

    let expected1 = Matrix5::new(
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let expected2 = Matrix5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35,
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0);

    let expected3 = Matrix5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
         0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,
        31, 32, 33, 34, 35);

    assert_eq!(m.insert_fixed_rows::<U2>(0, 0), expected1);
    assert_eq!(m.insert_fixed_rows::<U2>(3, 0), expected2);
    assert_eq!(m.insert_fixed_rows::<U2>(2, 0), expected3);

    // The following is just to verify that the return type dimensions is correctly inferred.
    let computed: Matrix<_, Dynamic, U5, _> = m.insert_rows(3, 2, 0);
    assert!(computed.eq(&expected2));
}

#[test]
fn resize() {
    let m = Matrix3x5::new(
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25,
        31, 32, 33, 34, 35);

    let add_add = DMatrix::from_row_slice(5, 6, &[
        11, 12, 13, 14, 15, 42,
        21, 22, 23, 24, 25, 42,
        31, 32, 33, 34, 35, 42,
        42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42]);

    let del_del = DMatrix::from_row_slice(1, 2, &[11, 12]);

    let add_del = DMatrix::from_row_slice(5, 2, &[
        11, 12,
        21, 22,
        31, 32,
        42, 42,
        42, 42]);

    let del_add = DMatrix::from_row_slice(1, 8, &[
        11, 12, 13, 14, 15, 42, 42, 42]);

    assert_eq!(del_del, m.resize(1, 2, 42));
    assert_eq!(add_add, m.resize(5, 6, 42));
    assert_eq!(add_del, m.resize(5, 2, 42));
    assert_eq!(del_add, m.resize(1, 8, 42));
}
*/