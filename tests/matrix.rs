#[cfg(feature = "arbitrary")]
#[macro_use]
extern crate quickcheck;
#[macro_use]
extern crate approx;
extern crate num_traits as num;
extern crate alga;
extern crate nalgebra as na;

use num::{Zero, One};
use std::fmt::Display;

use alga::linear::FiniteDimInnerSpace;

use na::{DVector, DMatrix,
         Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
         RowVector4,
         Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6,
         Matrix2x3, Matrix3x2, Matrix3x4, Matrix4x3, Matrix2x4, Matrix4x6};


#[test]
fn is_column_major() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);

    let expected = &[ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 ];

    assert_eq!(a.as_slice(), expected);

    let a = Matrix2x3::from_row_slice(&[1.0, 2.0, 3.0,
                                        4.0, 5.0, 6.0]);

    assert_eq!(a.as_slice(), expected);

    let a = Matrix2x3::from_column_slice(&[1.0, 4.0,
                                           2.0, 5.0,
                                           3.0, 6.0]);

    assert_eq!(a.as_slice(), expected);
}

#[test]
fn linear_index() {
    let a = Matrix2x3::new(1, 2, 3,
                           4, 5, 6);

    assert_eq!(a[0], 1);
    assert_eq!(a[1], 4);
    assert_eq!(a[2], 2);
    assert_eq!(a[3], 5);
    assert_eq!(a[4], 3);
    assert_eq!(a[5], 6);

    let b = Vector4::new(1, 2, 3, 4);

    assert_eq!(b[0], 1);
    assert_eq!(b[1], 2);
    assert_eq!(b[2], 3);
    assert_eq!(b[3], 4);

    let c = RowVector4::new(1, 2, 3, 4);

    assert_eq!(c[0], 1);
    assert_eq!(c[1], 2);
    assert_eq!(c[2], 3);
    assert_eq!(c[3], 4);
}

#[test]
fn identity() {
    let id1 = Matrix3::<f64>::identity();
    let id2 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0);
    let id2bis = Matrix3x4::identity();
    let id3 = Matrix4x3::new(1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0);
    let id3bis = Matrix4x3::identity();


    let not_id1 = Matrix3::identity() * 2.0;
    let not_id2 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 1.0);
    let not_id3 = Matrix4x3::new(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0,
                                 0.0, 1.0, 0.0);

    assert_eq!(id2, id2bis);
    assert_eq!(id3, id3bis);
    assert!(id1.is_identity(0.0));
    assert!(id2.is_identity(0.0));
    assert!(id3.is_identity(0.0));
    assert!(!not_id1.is_identity(0.0));
    assert!(!not_id2.is_identity(0.0));
    assert!(!not_id3.is_identity(0.0));
}

#[test]
fn coordinates() {
    let a = Matrix3x4::new(11, 12, 13, 14,
                           21, 22, 23, 24,
                           31, 32, 33, 34);

    assert_eq!(a.m11, 11);
    assert_eq!(a.m12, 12);
    assert_eq!(a.m13, 13);
    assert_eq!(a.m14, 14);

    assert_eq!(a.m21, 21);
    assert_eq!(a.m22, 22);
    assert_eq!(a.m23, 23);
    assert_eq!(a.m24, 24);

    assert_eq!(a.m31, 31);
    assert_eq!(a.m32, 32);
    assert_eq!(a.m33, 33);
    assert_eq!(a.m34, 34);
}

#[test]
fn from_diagonal() {
    let diag = Vector3::new(1, 2, 3);
    let expected = Matrix3::new(
        1, 0, 0,
        0, 2, 0,
        0, 0, 3);
    let a = Matrix3::from_diagonal(&diag);

    assert_eq!(a, expected);
}

#[test]
fn from_rows() {
    let rows = &[
        RowVector4::new(11, 12, 13, 14),
        RowVector4::new(21, 22, 23, 24),
        RowVector4::new(31, 32, 33, 34)
    ];

    let expected = Matrix3x4::new(
        11, 12, 13, 14,
        21, 22, 23, 24,
        31, 32, 33, 34);

    let a = Matrix3x4::from_rows(rows);

    assert_eq!(a, expected);
}

#[test]
fn from_columns() {
    let columns = &[
        Vector3::new(11, 21, 31),
        Vector3::new(12, 22, 32),
        Vector3::new(13, 23, 33),
        Vector3::new(14, 24, 34)
    ];

    let expected = Matrix3x4::new(
        11, 12, 13, 14,
        21, 22, 23, 24,
        31, 32, 33, 34);

    let a = Matrix3x4::from_columns(columns);

    assert_eq!(a, expected);
}

#[test]
fn from_columns_dynamic() {
    let columns = &[
        DVector::from_row_slice(3, &[11, 21, 31]),
        DVector::from_row_slice(3, &[12, 22, 32]),
        DVector::from_row_slice(3, &[13, 23, 33]),
        DVector::from_row_slice(3, &[14, 24, 34])
    ];

    let expected = DMatrix::from_row_slice(3, 4,
        &[ 11, 12, 13, 14,
           21, 22, 23, 24,
           31, 32, 33, 34 ]);

    let a = DMatrix::from_columns(columns);

    assert_eq!(a, expected);
}

#[test]
#[should_panic]
fn from_too_many_rows() {
    let rows = &[
        RowVector4::new(11, 12, 13, 14),
        RowVector4::new(21, 22, 23, 24),
        RowVector4::new(31, 32, 33, 34),
        RowVector4::new(31, 32, 33, 34)
    ];

    let _ = Matrix3x4::from_rows(rows);
}

#[test]
#[should_panic]
fn from_not_enough_columns() {
    let columns = &[
        Vector3::new(11, 21, 31),
        Vector3::new(14, 24, 34)
    ];

    let _ = Matrix3x4::from_columns(columns);
}

#[test]
#[should_panic]
fn from_rows_with_different_dimensions() {
    let columns = &[
        DVector::from_row_slice(3, &[11, 21, 31]),
        DVector::from_row_slice(3, &[12, 22, 32, 33])
    ];

    let _ = DMatrix::from_columns(columns);
}

#[test]
fn to_homogeneous() {
    let a          = Vector3::new(1.0, 2.0, 3.0);
    let expected_a = Vector4::new(1.0, 2.0, 3.0, 0.0);

    let b          = DVector::from_row_slice(3, &[1.0, 2.0, 3.0]);
    let expected_b = DVector::from_row_slice(4, &[1.0, 2.0, 3.0, 0.0]);

    assert_eq!(a.to_homogeneous(), expected_a);
    assert_eq!(b.to_homogeneous(), expected_b);
}

#[test]
fn simple_add() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);

    let b = Matrix2x3::new(10.0, 20.0, 30.0,
                           40.0, 50.0, 60.0);
    let c = DMatrix::from_row_slice(2, 3, &[ 10.0, 20.0, 30.0,
                                             40.0, 50.0, 60.0 ]);

    let expected = Matrix2x3::new(11.0, 22.0, 33.0,
                                  44.0, 55.0, 66.0);

    assert_eq!(expected, &a + &b);
    assert_eq!(expected, &a +  b);
    assert_eq!(expected,  a + &b);
    assert_eq!(expected,  a +  b);

    // Sum of a static matrix with a dynamic one.
    assert_eq!(expected, &a + &c);
    assert_eq!(expected,  a + &c);
    assert_eq!(expected, &c + &a);
    assert_eq!(expected, &c +  a);
}

#[test]
fn simple_scalar_mul() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);

    let expected = Matrix2x3::new(10.0, 20.0, 30.0,
                                  40.0, 50.0, 60.0);

    assert_eq!(expected,  a * 10.0);
    assert_eq!(expected, &a * 10.0);
    assert_eq!(expected, 10.0 *  a);
    assert_eq!(expected, 10.0 * &a);
}

#[test]
fn simple_mul() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);

    let b = Matrix3x4::new(10.0, 20.0,  30.0,  40.0,
                           50.0, 60.0,  70.0,  80.0,
                           90.0, 100.0, 110.0, 120.0);

    let expected = Matrix2x4::new(380.0, 440.0, 500.0,  560.0,
                                  830.0, 980.0, 1130.0, 1280.0);

    assert_eq!(expected, &a * &b);
    assert_eq!(expected,  a * &b);
    assert_eq!(expected, &a *  b);
    assert_eq!(expected,  a *  b);
}

#[test]
fn simple_scalar_conversion() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);
    let expected = Matrix2x3::new(1, 2, 3,
                                  4, 5, 6);

    let a_u32: Matrix2x3<u32> = na::try_convert(a).unwrap(); // f32 -> u32
    let a_f32: Matrix2x3<f32> = na::convert(a_u32);          // u32 -> f32

    assert_eq!(a, a_f32);
    assert_eq!(expected, a_u32);
}

#[test]
#[should_panic]
fn trace_panic() {
    let m = DMatrix::<f32>::new_random(2, 3);
    let _ = m.trace();
}

#[test]
fn trace() {
    let m = Matrix2::new(1.0,  20.0,
                         30.0, 4.0);
    assert_eq!(m.trace(), 5.0);
}

#[test]
fn simple_transpose() {
    let a = Matrix2x3::new(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0);
    let expected = Matrix3x2::new(1.0, 4.0,
                                  2.0, 5.0,
                                  3.0, 6.0);

    assert_eq!(a.transpose(), expected);
}

#[test]
fn simple_transpose_mut() {
    let mut a = Matrix3::new(1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0);
    let expected = Matrix3::new(1.0, 4.0, 7.0,
                                2.0, 5.0, 8.0,
                                3.0, 6.0, 9.0);

    a.transpose_mut();
    assert_eq!(a, expected);
}

#[test]
fn vector_index_mut() {
    let mut v = Vector3::new(1, 2, 3);

    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);

    v[0] = 10;
    v[1] = 20;
    v[2] = 30;

    assert_eq!(v, Vector3::new(10, 20, 30));
}

#[test]
fn components_mut() {
    let mut m2 = Matrix2::from_element(1.0);
    let mut m3 = Matrix3::from_element(1.0);
    let mut m4 = Matrix4::from_element(1.0);
    let mut m5 = Matrix5::from_element(1.0);
    let mut m6 = Matrix6::from_element(1.0);

    m2.m11 = 0.0; m2.m12 = 0.0;
    m2.m21 = 0.0; m2.m22 = 0.0;

    m3.m11 = 0.0; m3.m12 = 0.0; m3.m13 = 0.0;
    m3.m21 = 0.0; m3.m22 = 0.0; m3.m23 = 0.0;
    m3.m31 = 0.0; m3.m32 = 0.0; m3.m33 = 0.0;

    m4.m11 = 0.0; m4.m12 = 0.0; m4.m13 = 0.0; m4.m14 = 0.0;
    m4.m21 = 0.0; m4.m22 = 0.0; m4.m23 = 0.0; m4.m24 = 0.0;
    m4.m31 = 0.0; m4.m32 = 0.0; m4.m33 = 0.0; m4.m34 = 0.0;
    m4.m41 = 0.0; m4.m42 = 0.0; m4.m43 = 0.0; m4.m44 = 0.0;

    m5.m11 = 0.0; m5.m12 = 0.0; m5.m13 = 0.0; m5.m14 = 0.0; m5.m15 = 0.0;
    m5.m21 = 0.0; m5.m22 = 0.0; m5.m23 = 0.0; m5.m24 = 0.0; m5.m25 = 0.0;
    m5.m31 = 0.0; m5.m32 = 0.0; m5.m33 = 0.0; m5.m34 = 0.0; m5.m35 = 0.0;
    m5.m41 = 0.0; m5.m42 = 0.0; m5.m43 = 0.0; m5.m44 = 0.0; m5.m45 = 0.0;
    m5.m51 = 0.0; m5.m52 = 0.0; m5.m53 = 0.0; m5.m54 = 0.0; m5.m55 = 0.0;

    m6.m11 = 0.0; m6.m12 = 0.0; m6.m13 = 0.0; m6.m14 = 0.0; m6.m15 = 0.0; m6.m16 = 0.0;
    m6.m21 = 0.0; m6.m22 = 0.0; m6.m23 = 0.0; m6.m24 = 0.0; m6.m25 = 0.0; m6.m26 = 0.0;
    m6.m31 = 0.0; m6.m32 = 0.0; m6.m33 = 0.0; m6.m34 = 0.0; m6.m35 = 0.0; m6.m36 = 0.0;
    m6.m41 = 0.0; m6.m42 = 0.0; m6.m43 = 0.0; m6.m44 = 0.0; m6.m45 = 0.0; m6.m46 = 0.0;
    m6.m51 = 0.0; m6.m52 = 0.0; m6.m53 = 0.0; m6.m54 = 0.0; m6.m55 = 0.0; m6.m56 = 0.0;
    m6.m61 = 0.0; m6.m62 = 0.0; m6.m63 = 0.0; m6.m64 = 0.0; m6.m65 = 0.0; m6.m66 = 0.0;

    assert!(m2.is_zero());
    assert!(m3.is_zero());
    assert!(m4.is_zero());
    assert!(m5.is_zero());
    assert!(m6.is_zero());


    let mut v1 = Vector1::from_element(1.0);
    let mut v2 = Vector2::from_element(1.0);
    let mut v3 = Vector3::from_element(1.0);
    let mut v4 = Vector4::from_element(1.0);
    let mut v5 = Vector5::from_element(1.0);
    let mut v6 = Vector6::from_element(1.0);

    v1.x = 0.0;
    v2.x = 0.0; v2.y = 0.0;
    v3.x = 0.0; v3.y = 0.0; v3.z = 0.0;
    v4.x = 0.0; v4.y = 0.0; v4.z = 0.0; v4.w = 0.0;
    v5.x = 0.0; v5.y = 0.0; v5.z = 0.0; v5.w = 0.0; v5.a = 0.0;
    v6.x = 0.0; v6.y = 0.0; v6.z = 0.0; v6.w = 0.0; v6.a = 0.0; v6.b = 0.0;

    assert!(v1.is_zero());
    assert!(v2.is_zero());
    assert!(v3.is_zero());
    assert!(v4.is_zero());
    assert!(v5.is_zero());
    assert!(v6.is_zero());

    // Check that the components order is correct.
    m3.m11 = 11.0; m3.m12 = 12.0; m3.m13 = 13.0;
    m3.m21 = 21.0; m3.m22 = 22.0; m3.m23 = 23.0;
    m3.m31 = 31.0; m3.m32 = 32.0; m3.m33 = 33.0;

    let expected_m3 = Matrix3::new(11.0, 12.0, 13.0,
                                   21.0, 22.0, 23.0,
                                   31.0, 32.0, 33.0);
    assert_eq!(expected_m3, m3);
}

#[cfg(feature = "arbitrary")]
quickcheck!{
    /*
     *
     * Transposition.
     *
     */
    fn transpose_transpose_is_self(m: Matrix2x3<f64>) -> bool {
        m.transpose().transpose() == m
    }

    fn transpose_mut_transpose_mut_is_self(m: Matrix3<f64>) -> bool {
        let mut mm = m;
        mm.transpose_mut();
        mm.transpose_mut();
        m == mm
    }

    fn transpose_transpose_is_id_dyn(m: DMatrix<f64>) -> bool {
        m.transpose().transpose() == m
    }

    fn check_transpose_components_dyn(m: DMatrix<f64>) -> bool {
        let tr = m.transpose();
        let (nrows, ncols) = m.shape();

        if nrows != tr.shape().1 || ncols != tr.shape().0 {
            return false
        }

        for i in 0 .. nrows {
            for j in 0 .. ncols {
                if m[(i, j)] != tr[(j, i)] {
                    return false
                }
            }
        }

        true
    }

    fn tr_mul_is_transpose_then_mul(m: Matrix4x6<f64>, v: Vector4<f64>) -> bool {
        m.transpose() * v == m.tr_mul(&v)
    }

    /*
     *
     *
     * Inversion.
     *
     *
     */
    fn self_mul_inv_is_id_dim1(m: Matrix1<f64>) -> bool {
        if let Some(im) = m.try_inverse() {
            let id = Matrix1::one();
            relative_eq!(im * m, id, epsilon = 1.0e-7) &&
            relative_eq!(m * im, id, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn self_mul_inv_is_id_dim2(m: Matrix2<f64>) -> bool {
        if let Some(im) = m.try_inverse() {
            let id = Matrix2::one();
            relative_eq!(im * m, id, epsilon = 1.0e-7) &&
            relative_eq!(m * im, id, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn self_mul_inv_is_id_dim3(m: Matrix3<f64>) -> bool {
        if let Some(im) = m.try_inverse() {
            let id = Matrix3::one();
            relative_eq!(im * m, id, epsilon = 1.0e-7) &&
            relative_eq!(m * im, id, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn self_mul_inv_is_id_dim6(m: Matrix6<f64>) -> bool {
        if let Some(im) = m.try_inverse() {
            let id = Matrix6::one();
            relative_eq!(im * m, id, epsilon = 1.0e-7) &&
            relative_eq!(m * im, id, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    /*
     *
     * Normalization.
     *
     */
    fn normalized_vec_norm_is_one(v: Vector3<f64>) -> bool {
        if let Some(nv) = v.try_normalize(1.0e-10) {
            relative_eq!(nv.norm(), 1.0, epsilon = 1.0e-7)
        }
        else {
            true
        }
    }

    fn normalized_vec_norm_is_one_dyn(v: DVector<f64>) -> bool {
        if let Some(nv) = v.try_normalize(1.0e-10) {
            relative_eq!(nv.norm(), 1.0)
        }
        else {
            true
        }
    }
}

// FIXME: move this to alga ?
macro_rules! finite_dim_inner_space_test(
    ($($Vector: ident, $orthonormal_subspace: ident, $orthonormalization: ident);* $(;)*) => {$(
        #[cfg(feature = "arbitrary")]
        quickcheck!{
            fn $orthonormal_subspace(vs: Vec<$Vector<f64>>) -> bool {
                let mut given_basis = vs.clone();
                let given_basis_dim = $Vector::orthonormalize(&mut given_basis[..]);
                let mut ortho_basis = Vec::new();
                $Vector::orthonormal_subspace_basis(
                    &given_basis[.. given_basis_dim],
                    |e| { ortho_basis.push(*e); true }
                );

                if !is_subspace_basis(&ortho_basis[..]) {
                    return false;
                }

                for v in vs {
                    for b in &ortho_basis {
                        if !relative_eq!(v.dot(b), 0.0, epsilon = 1.0e-7) {
                            println!("Found dot product: {} · {} = {}", v, b, v.dot(b));
                            return false;
                        }
                    }
                }

                true
            }

            fn $orthonormalization(vs: Vec<$Vector<f64>>) -> bool {
                let mut basis = vs.clone();
                let subdim = $Vector::orthonormalize(&mut basis[..]);

                if !is_subspace_basis(&basis[.. subdim]) {
                    return false;
                }

                for mut e in vs {
                    for b in &basis[.. subdim] {
                        e -= e.dot(b) * b
                    }

                    // Any element of `e` must be a linear combination of the basis elements.
                    if !relative_eq!(e.norm(), 0.0, epsilon = 1.0e-7) {
                        println!("Orthonormalization; element decomposition failure: {}", e);
                        println!("... the non-zero norm is: {}", e.norm());
                        return false;
                    }
                }

                true
            }
        }
    )*}
);

finite_dim_inner_space_test!(
    Vector1, orthonormal_subspace_basis1, orthonormalize1;
    Vector2, orthonormal_subspace_basis2, orthonormalize2;
    Vector3, orthonormal_subspace_basis3, orthonormalize3;
    Vector4, orthonormal_subspace_basis4, orthonormalize4;
    Vector5, orthonormal_subspace_basis5, orthonormalize5;
    Vector6, orthonormal_subspace_basis6, orthonormalize6;
);

/*
 *
 * Helper functions.
 *
 */
fn is_subspace_basis<T: FiniteDimInnerSpace<Real = f64> + Display>(vs: &[T]) -> bool {
    for i in 0 .. vs.len() {
        // Basis elements must be normalized.
        if !relative_eq!(vs[i].norm(), 1.0, epsilon = 1.0e-7) {
            println!("Non-zero basis element norm: {}", vs[i].norm());
            return false;
        }

        for j in 0 .. i {
            // Basis elements must be orthogonal.
            if !relative_eq!(vs[i].dot(&vs[j]), 0.0, epsilon = 1.0e-7) {
                println!("Non-orthogonal basis elements: {} · {} = {}", vs[i], vs[j], vs[i].dot(&vs[j]));
                return false
            }
        }
    }

    true
}
