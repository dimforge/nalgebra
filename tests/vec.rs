#![feature(macro_rules)]

extern crate "nalgebra" as na;

use std::rand::random;
use na::{Vec0, Vec1, Vec2, Vec3, Vec4, Vec5, Vec6, Mat3, Iterable, IterableMut};

macro_rules! test_iterator_impl(
    ($t: ty, $n: ty) => (
        for _ in range(0u, 10000) {
            let v: $t      = random();
            let mut mv: $t = v.clone();
            let n: $n      = random();

            let nv: $t = v.iter().map(|e| *e * n).collect();

            for e in mv.iter_mut() {
                *e = *e * n
            }

            assert!(nv == mv && nv == v * n);
        }
    )
);

macro_rules! test_commut_dot_impl(
    ($t: ty) => (
        for _ in range(0u, 10000) {
            let v1 : $t = random();
            let v2 : $t = random();

            assert!(na::approx_eq(&na::dot(&v1, &v2), &na::dot(&v2, &v1)));
        }
    );
);

macro_rules! test_scalar_op_impl(
    ($t: ty, $n: ty) => (
        for _ in range(0u, 10000) {
            let v1 : $t = random();
            let n  : $n = random();

            assert!(na::approx_eq(&((v1 * n) / n), &v1));
            assert!(na::approx_eq(&((v1 / n) * n), &v1));
            assert!(na::approx_eq(&((v1 - n) + n), &v1));
            assert!(na::approx_eq(&((v1 + n) - n), &v1));

            let mut v1 : $t = random();
            let v0 : $t = v1.clone();
            let n  : $n = random();

            v1 = v1 * n;
            v1 = v1 / n;

            assert!(na::approx_eq(&v1, &v0));
        }
    );
);

macro_rules! test_basis_impl(
    ($t: ty) => (
        for _ in range(0u, 10000) {
            na::canonical_basis(|e1: $t| {
                na::canonical_basis(|e2: $t| {
                    assert!(e1 == e2 || na::approx_eq(&na::dot(&e1, &e2), &na::zero()));

                    true
                });

                assert!(na::approx_eq(&na::norm(&e1), &na::one()));

                true
            })
        }
    );
);

macro_rules! test_subspace_basis_impl(
    ($t: ty) => (
        for _ in range(0u, 10000) {
            let v : $t = random();
            let v1     = na::normalize(&v);

            na::orthonormal_subspace_basis(&v1, |e1| {
                // check vectors are orthogonal to v1
                assert!(na::approx_eq(&na::dot(&v1, &e1), &na::zero()));
                // check vectors form an orthonormal basis
                assert!(na::approx_eq(&na::norm(&e1), &na::one()));
                // check vectors form an ortogonal basis
                na::orthonormal_subspace_basis(&v1, |e2| {
                    assert!(e1 == e2 || na::approx_eq(&na::dot(&e1, &e2), &na::zero()));

                    true
                });

                true
            })
        }
    );
);

#[test]
fn test_cross_vec3() {
    for _ in range(0u, 10000) {
        let v1 : Vec3<f64> = random();
        let v2 : Vec3<f64> = random();
        let v3 : Vec3<f64> = na::cross(&v1, &v2);

        assert!(na::approx_eq(&na::dot(&v3, &v2), &na::zero()));
        assert!(na::approx_eq(&na::dot(&v3, &v1), &na::zero()));
    }
}

#[test]
fn test_commut_dot_vec0() {
    test_commut_dot_impl!(Vec0<f64>);
}

#[test]
fn test_commut_dot_vec1() {
    test_commut_dot_impl!(Vec1<f64>);
}

#[test]
fn test_commut_dot_vec2() {
    test_commut_dot_impl!(Vec2<f64>);
}

#[test]
fn test_commut_dot_vec3() {
    test_commut_dot_impl!(Vec3<f64>);
}

#[test]
fn test_commut_dot_vec4() {
    test_commut_dot_impl!(Vec4<f64>);
}

#[test]
fn test_commut_dot_vec5() {
    test_commut_dot_impl!(Vec5<f64>);
}

#[test]
fn test_commut_dot_vec6() {
    test_commut_dot_impl!(Vec6<f64>);
}

#[test]
fn test_basis_vec0() {
    test_basis_impl!(Vec0<f64>);
}

#[test]
fn test_basis_vec1() {
    test_basis_impl!(Vec1<f64>);
}

#[test]
fn test_basis_vec2() {
    test_basis_impl!(Vec2<f64>);
}

#[test]
fn test_basis_vec3() {
    test_basis_impl!(Vec3<f64>);
}

#[test]
fn test_basis_vec4() {
    test_basis_impl!(Vec4<f64>);
}

#[test]
fn test_basis_vec5() {
    test_basis_impl!(Vec5<f64>);
}

#[test]
fn test_basis_vec6() {
    test_basis_impl!(Vec6<f64>);
}

#[test]
fn test_subspace_basis_vec0() {
    test_subspace_basis_impl!(Vec0<f64>);
}

#[test]
fn test_subspace_basis_vec1() {
    test_subspace_basis_impl!(Vec1<f64>);
}

#[test]
fn test_subspace_basis_vec2() {
    test_subspace_basis_impl!(Vec2<f64>);
}

#[test]
fn test_subspace_basis_vec3() {
    test_subspace_basis_impl!(Vec3<f64>);
}

#[test]
fn test_subspace_basis_vec4() {
    test_subspace_basis_impl!(Vec4<f64>);
}

#[test]
fn test_subspace_basis_vec5() {
    test_subspace_basis_impl!(Vec5<f64>);
}

#[test]
fn test_subspace_basis_vec6() {
    test_subspace_basis_impl!(Vec6<f64>);
}

#[test]
fn test_scalar_op_vec0() {
    test_scalar_op_impl!(Vec0<f64>, f64);
}

#[test]
fn test_scalar_op_vec1() {
    test_scalar_op_impl!(Vec1<f64>, f64);
}

#[test]
fn test_scalar_op_vec2() {
    test_scalar_op_impl!(Vec2<f64>, f64);
}

#[test]
fn test_scalar_op_vec3() {
    test_scalar_op_impl!(Vec3<f64>, f64);
}

#[test]
fn test_scalar_op_vec4() {
    test_scalar_op_impl!(Vec4<f64>, f64);
}

#[test]
fn test_scalar_op_vec5() {
    test_scalar_op_impl!(Vec5<f64>, f64);
}

#[test]
fn test_scalar_op_vec6() {
    test_scalar_op_impl!(Vec6<f64>, f64);
}

#[test]
fn test_iterator_vec0() {
    test_iterator_impl!(Vec0<f64>, f64);
}

#[test]
fn test_iterator_vec1() {
    test_iterator_impl!(Vec1<f64>, f64);
}

#[test]
fn test_iterator_vec2() {
    test_iterator_impl!(Vec2<f64>, f64);
}

#[test]
fn test_iterator_vec3() {
    test_iterator_impl!(Vec3<f64>, f64);
}

#[test]
fn test_iterator_vec4() {
    test_iterator_impl!(Vec4<f64>, f64);
}

#[test]
fn test_iterator_vec5() {
    test_iterator_impl!(Vec5<f64>, f64);
}

#[test]
fn test_iterator_vec6() {
    test_iterator_impl!(Vec6<f64>, f64);
}

#[test]
fn test_ord_vec3() {
    // equality
    assert!(Vec3::new(0.5f64, 0.5, 0.5) == Vec3::new(0.5, 0.5, 0.5));
    assert!(!(Vec3::new(1.5f64, 0.5, 0.5) == Vec3::new(0.5, 0.5, 0.5)));
    assert!(Vec3::new(1.5f64, 0.5, 0.5) != Vec3::new(0.5, 0.5, 0.5));

    // comparable
    assert!(na::partial_cmp(&Vec3::new(0.5f64, 0.3, 0.3), &Vec3::new(1.0, 2.0, 1.0)).is_le());
    assert!(na::partial_cmp(&Vec3::new(0.5f64, 0.3, 0.3), &Vec3::new(1.0, 2.0, 1.0)).is_lt());
    assert!(na::partial_cmp(&Vec3::new(2.0f64, 4.0, 2.0), &Vec3::new(1.0, 2.0, 1.0)).is_ge());
    assert!(na::partial_cmp(&Vec3::new(2.0f64, 4.0, 2.0), &Vec3::new(1.0, 2.0, 1.0)).is_gt());

    // not comparable
    assert!(na::partial_cmp(&Vec3::new(0.0f64, 3.0, 0.0), &Vec3::new(1.0, 2.0, 1.0)).is_not_comparable());
}

#[test]
fn test_min_max_vec3() {
    assert_eq!(na::sup(&Vec3::new(1.0f64, 2.0, 3.0), &Vec3::new(3.0, 2.0, 1.0)), Vec3::new(3.0, 2.0, 3.0));
    assert_eq!(na::inf(&Vec3::new(1.0f64, 2.0, 3.0), &Vec3::new(3.0, 2.0, 1.0)), Vec3::new(1.0, 2.0, 1.0));
}

#[test]
fn test_outer_vec3() {
    assert_eq!(
        na::outer(&Vec3::new(1.0f64, 2.0, 3.0), &Vec3::new(4.0, 5.0, 6.0)),
        Mat3::new(
            4.0, 5.0, 6.0,
            8.0, 10.0, 12.0,
            12.0, 15.0, 18.0));
}
