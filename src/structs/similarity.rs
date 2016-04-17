use std::fmt;
use std::ops::{Mul, Neg, MulAssign};

use rand::{Rand, Rng};
use num::One;
use structs::matrix::{Matrix3, Matrix4};
use traits::structure::{Dimension, Column, BaseFloat, BaseNum};
use traits::operations::{Inverse, ApproxEq};
use traits::geometry::{Rotation, Transform, Transformation, Translation, ToHomogeneous};

use structs::vector::{Vector1, Vector2, Vector3};
use structs::point::{Point2, Point3};
use structs::rotation::{Rotation2, Rotation3};
use structs::isometry::{Isometry2, Isometry3};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

// FIXME: the name is not explicit at all but coherent with the other tree-letters names…
/// A two-dimensional similarity transformation.
///
/// This is a composition of a uniform scale, followed by a rotation, followed by a translation.
/// Vectors `Vector2` are not affected by the translational component of this transformation while
/// points `Point2` are.
/// Similarity transformations conserve angles. Distances are multiplied by some constant (the
/// scale factor). The scale factor cannot be zero.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Similarity2<N> {
    /// The uniform scale applicable by this similarity transformation.
    scale: N,
    /// The isometry applicable by this similarity transformation.
    pub isometry: Isometry2<N>
}

/// A three-dimensional similarity transformation.
///
/// This is a composition of a scale, followed by a rotation, followed by a translation.
/// Vectors `Vector3` are not affected by the translational component of this transformation while
/// points `Point3` are.
/// Similarity transformations conserve angles. Distances are multiplied by some constant (the
/// scale factor). The scale factor cannot be zero.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Similarity3<N> {
    /// The uniform scale applicable by this similarity transformation.
    scale: N,
    /// The isometry applicable by this similarity transformation.
    pub isometry: Isometry3<N>
}

sim_impl!(Similarity2, Isometry2, Rotation2, Vector2, Vector1);
dim_impl!(Similarity2, 2);
sim_scale_impl!(Similarity2);
sim_one_impl!(Similarity2);
sim_mul_sim_impl!(Similarity2);
sim_mul_isometry_impl!(Similarity2, Isometry2);
sim_mul_rotation_impl!(Similarity2, Rotation2);
sim_mul_point_vec_impl!(Similarity2, Point2);
sim_mul_point_vec_impl!(Similarity2, Vector2);
transformation_impl!(Similarity2);
sim_transform_impl!(Similarity2, Point2);
sim_inverse_impl!(Similarity2);
sim_to_homogeneous_impl!(Similarity2, Matrix3);
sim_approx_eq_impl!(Similarity2);
sim_rand_impl!(Similarity2);
sim_arbitrary_impl!(Similarity2);
sim_display_impl!(Similarity2);

sim_impl!(Similarity3, Isometry3, Rotation3, Vector3, Vector3);
dim_impl!(Similarity3, 3);
sim_scale_impl!(Similarity3);
sim_one_impl!(Similarity3);
sim_mul_sim_impl!(Similarity3);
sim_mul_isometry_impl!(Similarity3, Isometry3);
sim_mul_rotation_impl!(Similarity3, Rotation3);
sim_mul_point_vec_impl!(Similarity3, Point3);
sim_mul_point_vec_impl!(Similarity3, Vector3);
transformation_impl!(Similarity3);
sim_transform_impl!(Similarity3, Point3);
sim_inverse_impl!(Similarity3);
sim_to_homogeneous_impl!(Similarity3, Matrix4);
sim_approx_eq_impl!(Similarity3);
sim_rand_impl!(Similarity3);
sim_arbitrary_impl!(Similarity3);
sim_display_impl!(Similarity3);
