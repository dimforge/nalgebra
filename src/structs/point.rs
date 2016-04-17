//! Points with dimension known at compile-time.

use std::mem;
use std::fmt;
use std::slice::{Iter, IterMut};
use std::iter::{Iterator, FromIterator, IntoIterator};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::operations::{ApproxEq, PartialOrder, PartialOrdering, Axpy};
use traits::structure::{Cast, Dimension, Indexable, Iterable, IterableMut, PointAsVector, Shape,
                        NumPoint, FloatPoint, BaseFloat, BaseNum, Bounded, Repeat};
use traits::geometry::{Origin, FromHomogeneous, ToHomogeneous};
use structs::vector::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Point of dimension 1.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point1<N> {
    /// First component of the point.
    pub x: N
}

new_impl!(Point1, x);
origin_impl!(Point1, x);
pord_impl!(Point1, x,);
scalar_mul_impl!(Point1, x);
scalar_div_impl!(Point1, x);
scalar_add_impl!(Point1, x);
scalar_sub_impl!(Point1, x);
vec_cast_impl!(Point1, x);
conversion_impl!(Point1, 1);
index_impl!(Point1);
indexable_impl!(Point1, 1);
at_fast_impl!(Point1, 1);
repeat_impl!(Point1, val, x);
dim_impl!(Point1, 1);
container_impl!(Point1);
point_as_vec_impl!(Point1, Vector1, x);
point_sub_impl!(Point1, Vector1);
neg_impl!(Point1, x);
point_add_vec_impl!(Point1, Vector1, x);
point_sub_vec_impl!(Point1, Vector1, x);
approx_eq_impl!(Point1, x);
from_iterator_impl!(Point1, iterator);
bounded_impl!(Point1, x);
axpy_impl!(Point1, x);
iterable_impl!(Point1, 1);
iterable_mut_impl!(Point1, 1);
point_to_homogeneous_impl!(Point1, Point2, y, x);
point_from_homogeneous_impl!(Point1, Point2, y, x);
num_float_point_impl!(Point1, Vector1);
arbitrary_point_impl!(Point1, x);
rand_impl!(Point1, x);
point_display_impl!(Point1);

/// Point of dimension 2.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point2<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N
}

new_impl!(Point2, x, y);
origin_impl!(Point2, x, y);
pord_impl!(Point2, x, y);
scalar_mul_impl!(Point2, x, y);
scalar_div_impl!(Point2, x, y);
scalar_add_impl!(Point2, x, y);
scalar_sub_impl!(Point2, x, y);
vec_cast_impl!(Point2, x, y);
conversion_impl!(Point2, 2);
index_impl!(Point2);
indexable_impl!(Point2, 2);
at_fast_impl!(Point2, 2);
repeat_impl!(Point2, val, x, y);
dim_impl!(Point2, 2);
container_impl!(Point2);
point_as_vec_impl!(Point2, Vector2, x, y);
point_sub_impl!(Point2, Vector2);
neg_impl!(Point2, x, y);
point_add_vec_impl!(Point2, Vector2, x, y);
point_sub_vec_impl!(Point2, Vector2, x, y);
approx_eq_impl!(Point2, x, y);
from_iterator_impl!(Point2, iterator, iterator);
bounded_impl!(Point2, x, y);
axpy_impl!(Point2, x, y);
iterable_impl!(Point2, 2);
iterable_mut_impl!(Point2, 2);
point_to_homogeneous_impl!(Point2, Point3, z, x, y);
point_from_homogeneous_impl!(Point2, Point3, z, x, y);
num_float_point_impl!(Point2, Vector2);
arbitrary_point_impl!(Point2, x, y);
rand_impl!(Point2, x, y);
point_display_impl!(Point2);

/// Point of dimension 3.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point3<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N
}

new_impl!(Point3, x, y, z);
origin_impl!(Point3, x, y, z);
pord_impl!(Point3, x, y, z);
scalar_mul_impl!(Point3, x, y, z);
scalar_div_impl!(Point3, x, y, z);
scalar_add_impl!(Point3, x, y, z);
scalar_sub_impl!(Point3, x, y, z);
vec_cast_impl!(Point3, x, y, z);
conversion_impl!(Point3, 3);
index_impl!(Point3);
indexable_impl!(Point3, 3);
at_fast_impl!(Point3, 3);
repeat_impl!(Point3, val, x, y, z);
dim_impl!(Point3, 3);
container_impl!(Point3);
point_as_vec_impl!(Point3, Vector3, x, y, z);
point_sub_impl!(Point3, Vector3);
neg_impl!(Point3, x, y, z);
point_add_vec_impl!(Point3, Vector3, x, y, z);
point_sub_vec_impl!(Point3, Vector3, x, y, z);
approx_eq_impl!(Point3, x, y, z);
from_iterator_impl!(Point3, iterator, iterator, iterator);
bounded_impl!(Point3, x, y, z);
axpy_impl!(Point3, x, y, z);
iterable_impl!(Point3, 3);
iterable_mut_impl!(Point3, 3);
point_to_homogeneous_impl!(Point3, Point4, w, x, y, z);
point_from_homogeneous_impl!(Point3, Point4, w, x, y, z);
num_float_point_impl!(Point3, Vector3);
arbitrary_point_impl!(Point3, x, y, z);
rand_impl!(Point3, x, y, z);
point_display_impl!(Point3);

/// Point of dimension 4.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point4<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N,
    /// Fourth component of the point.
    pub w: N
}

new_impl!(Point4, x, y, z, w);
origin_impl!(Point4, x, y, z, w);
pord_impl!(Point4, x, y, z, w);
scalar_mul_impl!(Point4, x, y, z, w);
scalar_div_impl!(Point4, x, y, z, w);
scalar_add_impl!(Point4, x, y, z, w);
scalar_sub_impl!(Point4, x, y, z, w);
vec_cast_impl!(Point4, x, y, z, w);
conversion_impl!(Point4, 4);
index_impl!(Point4);
indexable_impl!(Point4, 4);
at_fast_impl!(Point4, 4);
repeat_impl!(Point4, val, x, y, z, w);
dim_impl!(Point4, 4);
container_impl!(Point4);
point_as_vec_impl!(Point4, Vector4, x, y, z, w);
point_sub_impl!(Point4, Vector4);
neg_impl!(Point4, x, y, z, w);
point_add_vec_impl!(Point4, Vector4, x, y, z, w);
point_sub_vec_impl!(Point4, Vector4, x, y, z, w);
approx_eq_impl!(Point4, x, y, z, w);
from_iterator_impl!(Point4, iterator, iterator, iterator, iterator);
bounded_impl!(Point4, x, y, z, w);
axpy_impl!(Point4, x, y, z, w);
iterable_impl!(Point4, 4);
iterable_mut_impl!(Point4, 4);
point_to_homogeneous_impl!(Point4, Point5, a, x, y, z, w);
point_from_homogeneous_impl!(Point4, Point5, a, x, y, z, w);
num_float_point_impl!(Point4, Vector4);
arbitrary_point_impl!(Point4, x, y, z, w);
rand_impl!(Point4, x, y, z, w);
point_display_impl!(Point4);

/// Point of dimension 5.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point5<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N,
    /// Fourth component of the point.
    pub w: N,
    /// Fifth of the point.
    pub a: N
}

new_impl!(Point5, x, y, z, w, a);
origin_impl!(Point5, x, y, z, w, a);
pord_impl!(Point5, x, y, z, w, a);
scalar_mul_impl!(Point5, x, y, z, w, a);
scalar_div_impl!(Point5, x, y, z, w, a);
scalar_add_impl!(Point5, x, y, z, w, a);
scalar_sub_impl!(Point5, x, y, z, w, a);
vec_cast_impl!(Point5, x, y, z, w, a);
conversion_impl!(Point5, 5);
index_impl!(Point5);
indexable_impl!(Point5, 5);
at_fast_impl!(Point5, 5);
repeat_impl!(Point5, val, x, y, z, w, a);
dim_impl!(Point5, 5);
container_impl!(Point5);
point_as_vec_impl!(Point5, Vector5, x, y, z, w, a);
point_sub_impl!(Point5, Vector5);
neg_impl!(Point5, x, y, z, w, a);
point_add_vec_impl!(Point5, Vector5, x, y, z, w, a);
point_sub_vec_impl!(Point5, Vector5, x, y, z, w, a);
approx_eq_impl!(Point5, x, y, z, w, a);
from_iterator_impl!(Point5, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Point5, x, y, z, w, a);
axpy_impl!(Point5, x, y, z, w, a);
iterable_impl!(Point5, 5);
iterable_mut_impl!(Point5, 5);
point_to_homogeneous_impl!(Point5, Point6, b, x, y, z, w, a);
point_from_homogeneous_impl!(Point5, Point6, b, x, y, z, w, a);
num_float_point_impl!(Point5, Vector5);
arbitrary_point_impl!(Point5, x, y, z, w, a);
rand_impl!(Point5, x, y, z, w, a);
point_display_impl!(Point5);

/// Point of dimension 6.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point6<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N,
    /// Fourth component of the point.
    pub w: N,
    /// Fifth of the point.
    pub a: N,
    /// Sixth component of the point.
    pub b: N
}

new_impl!(Point6, x, y, z, w, a, b);
origin_impl!(Point6, x, y, z, w, a, b);
pord_impl!(Point6, x, y, z, w, a, b);
scalar_mul_impl!(Point6, x, y, z, w, a, b);
scalar_div_impl!(Point6, x, y, z, w, a, b);
scalar_add_impl!(Point6, x, y, z, w, a, b);
scalar_sub_impl!(Point6, x, y, z, w, a, b);
vec_cast_impl!(Point6, x, y, z, w, a, b);
conversion_impl!(Point6, 6);
index_impl!(Point6);
indexable_impl!(Point6, 6);
at_fast_impl!(Point6, 6);
repeat_impl!(Point6, val, x, y, z, w, a, b);
dim_impl!(Point6, 6);
container_impl!(Point6);
point_as_vec_impl!(Point6, Vector6, x, y, z, w, a, b);
point_sub_impl!(Point6, Vector6);
neg_impl!(Point6, x, y, z, w, a, b);
point_add_vec_impl!(Point6, Vector6, x, y, z, w, a, b);
point_sub_vec_impl!(Point6, Vector6, x, y, z, w, a, b);
approx_eq_impl!(Point6, x, y, z, w, a, b);
from_iterator_impl!(Point6, iterator, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Point6, x, y, z, w, a, b);
axpy_impl!(Point6, x, y, z, w, a, b);
iterable_impl!(Point6, 6);
iterable_mut_impl!(Point6, 6);
num_float_point_impl!(Point6, Vector6);
arbitrary_point_impl!(Point6, x, y, z, w, a, b);
rand_impl!(Point6, x, y, z, w, a, b);
point_display_impl!(Point6);
