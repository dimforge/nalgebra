use std::fmt;
use std::ops::{Mul, Neg};

use rand::{Rand, Rng};
use num::One;
use structs::mat::{Mat3, Mat4};
use traits::structure::{Dim, Col, BaseFloat, BaseNum};
use traits::operations::{Inv, ApproxEq};
use traits::geometry::{Rotation, Transform, Transformation, Translation, ToHomogeneous};

use structs::vec::{Vec1, Vec2, Vec3};
use structs::pnt::{Pnt2, Pnt3};
use structs::rot::{Rot2, Rot3};
use structs::iso::{Iso2, Iso3};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

// FIXME: the name is not explicit at all but coherent with the other tree-letters names…
/// A two-dimensional similarity transformation.
///
/// This is a composition of a uniform scale, followed by a rotation, followed by a translation.
/// Vectors `Vec2` are not affected by the translational component of this transformation while
/// points `Pnt2` are.
/// Similarity transformations conserve angles. Distances are multiplied by some constant (the
/// scale factor). The scale factor cannot be zero.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Sim2<N> {
    /// The uniform scale applicable by this similarity transformation.
    scale: N,
    /// The isometry applicable by this similarity transformation.
    pub isometry: Iso2<N>
}

/// A three-dimensional similarity transformation.
///
/// This is a composition of a scale, followed by a rotation, followed by a translation.
/// Vectors `Vec3` are not affected by the translational component of this transformation while
/// points `Pnt3` are.
/// Similarity transformations conserve angles. Distances are multiplied by some constant (the
/// scale factor). The scale factor cannot be zero.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Sim3<N> {
    /// The uniform scale applicable by this similarity transformation.
    scale: N,
    /// The isometry applicable by this similarity transformation.
    pub isometry: Iso3<N>
}

sim_impl!(Sim2, Iso2, Rot2, Vec2, Vec1);
dim_impl!(Sim2, 2);
sim_scale_impl!(Sim2);
sim_one_impl!(Sim2);
sim_mul_sim_impl!(Sim2);
sim_mul_iso_impl!(Sim2, Iso2);
sim_mul_rot_impl!(Sim2, Rot2);
sim_mul_pnt_vec_impl!(Sim2, Pnt2);
sim_mul_pnt_vec_impl!(Sim2, Vec2);
transformation_impl!(Sim2);
sim_transform_impl!(Sim2, Pnt2);
sim_inv_impl!(Sim2);
sim_to_homogeneous_impl!(Sim2, Mat3);
sim_approx_eq_impl!(Sim2);
sim_rand_impl!(Sim2);
sim_arbitrary_impl!(Sim2);
sim_display_impl!(Sim2);

sim_impl!(Sim3, Iso3, Rot3, Vec3, Vec3);
dim_impl!(Sim3, 3);
sim_scale_impl!(Sim3);
sim_one_impl!(Sim3);
sim_mul_sim_impl!(Sim3);
sim_mul_iso_impl!(Sim3, Iso3);
sim_mul_rot_impl!(Sim3, Rot3);
sim_mul_pnt_vec_impl!(Sim3, Pnt3);
sim_mul_pnt_vec_impl!(Sim3, Vec3);
transformation_impl!(Sim3);
sim_transform_impl!(Sim3, Pnt3);
sim_inv_impl!(Sim3);
sim_to_homogeneous_impl!(Sim3, Mat4);
sim_approx_eq_impl!(Sim3);
sim_rand_impl!(Sim3);
sim_arbitrary_impl!(Sim3);
sim_display_impl!(Sim3);
