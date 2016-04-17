//! Vectors with dimension known at compile-time.

use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::mem;
use std::slice::{Iter, IterMut};
use std::iter::{Iterator, FromIterator, IntoIterator};
use std::fmt;
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::operations::{ApproxEq, PartialOrder, PartialOrdering, Axpy, Absolute, Mean};
use traits::geometry::{Transform, Rotate, FromHomogeneous, ToHomogeneous, Dot, Norm,
                       Translation, Translate};
use traits::structure::{Basis, Cast, Dimension, Indexable, Iterable, IterableMut, Shape, NumVector,
                        FloatVector, BaseFloat, BaseNum, Bounded, Repeat};
use structs::point::{Point1, Point2, Point3, Point4, Point5, Point6};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Vector of dimension 1.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector1<N> {
    /// First component of the vector.
    pub x: N
}

new_impl!(Vector1, x);
pord_impl!(Vector1, x,);
vec_axis_impl!(Vector1, x);
vec_cast_impl!(Vector1, x);
conversion_impl!(Vector1, 1);
index_impl!(Vector1);
indexable_impl!(Vector1, 1);
at_fast_impl!(Vector1, 1);
repeat_impl!(Vector1, val, x);
dim_impl!(Vector1, 1);
container_impl!(Vector1);
// (specialized); basis_impl!(Vector1, 1);
add_impl!(Vector1, x);
sub_impl!(Vector1, x);
mul_impl!(Vector1, x);
div_impl!(Vector1, x);
scalar_add_impl!(Vector1, x);
scalar_sub_impl!(Vector1, x);
scalar_mul_impl!(Vector1, x);
scalar_div_impl!(Vector1, x);
neg_impl!(Vector1, x);
dot_impl!(Vector1, x);
translation_impl!(Vector1);
norm_impl!(Vector1, x);
approx_eq_impl!(Vector1, x);
zero_one_impl!(Vector1, x);
from_iterator_impl!(Vector1, iterator);
bounded_impl!(Vector1, x);
axpy_impl!(Vector1, x);
iterable_impl!(Vector1, 1);
iterable_mut_impl!(Vector1, 1);
vec_to_homogeneous_impl!(Vector1, Vector2, y, x);
vec_from_homogeneous_impl!(Vector1, Vector2, y, x);
translate_impl!(Vector1, Point1);
rotate_impl!(Vector1);
rotate_impl!(Point1);
transform_impl!(Vector1, Point1);
vec_as_point_impl!(Vector1, Point1, x);
num_float_vec_impl!(Vector1);
absolute_vec_impl!(Vector1, x);
arbitrary_impl!(Vector1, x);
rand_impl!(Vector1, x);
mean_impl!(Vector1);
vec_display_impl!(Vector1);

/// Vector of dimension 2.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector2<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N
}

new_impl!(Vector2, x, y);
pord_impl!(Vector2, x, y);
vec_axis_impl!(Vector2, x, y);
vec_cast_impl!(Vector2, x, y);
conversion_impl!(Vector2, 2);
index_impl!(Vector2);
indexable_impl!(Vector2, 2);
at_fast_impl!(Vector2, 2);
repeat_impl!(Vector2, val, x, y);
dim_impl!(Vector2, 2);
container_impl!(Vector2);
// (specialized); basis_impl!(Vector2, 1);
add_impl!(Vector2, x, y);
sub_impl!(Vector2, x, y);
mul_impl!(Vector2, x, y);
div_impl!(Vector2, x, y);
scalar_add_impl!(Vector2, x, y);
scalar_sub_impl!(Vector2, x, y);
scalar_mul_impl!(Vector2, x, y);
scalar_div_impl!(Vector2, x, y);
neg_impl!(Vector2, x, y);
dot_impl!(Vector2, x, y);
translation_impl!(Vector2);
norm_impl!(Vector2, x, y);
approx_eq_impl!(Vector2, x, y);
zero_one_impl!(Vector2, x, y);
from_iterator_impl!(Vector2, iterator, iterator);
bounded_impl!(Vector2, x, y);
axpy_impl!(Vector2, x, y);
iterable_impl!(Vector2, 2);
iterable_mut_impl!(Vector2, 2);
vec_to_homogeneous_impl!(Vector2, Vector3, z, x, y);
vec_from_homogeneous_impl!(Vector2, Vector3, z, x, y);
translate_impl!(Vector2, Point2);
rotate_impl!(Vector2);
rotate_impl!(Point2);
transform_impl!(Vector2, Point2);
vec_as_point_impl!(Vector2, Point2, x, y);
num_float_vec_impl!(Vector2);
absolute_vec_impl!(Vector2, x, y);
arbitrary_impl!(Vector2, x, y);
rand_impl!(Vector2, x, y);
mean_impl!(Vector2);
vec_display_impl!(Vector2);

/// Vector of dimension 3.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector3<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N
}

new_impl!(Vector3, x, y, z);
pord_impl!(Vector3, x, y, z);
vec_axis_impl!(Vector3, x, y, z);
vec_cast_impl!(Vector3, x, y, z);
conversion_impl!(Vector3, 3);
index_impl!(Vector3);
indexable_impl!(Vector3, 3);
at_fast_impl!(Vector3, 3);
repeat_impl!(Vector3, val, x, y, z);
dim_impl!(Vector3, 3);
container_impl!(Vector3);
// (specialized); basis_impl!(Vector3, 1);
add_impl!(Vector3, x, y, z);
sub_impl!(Vector3, x, y, z);
mul_impl!(Vector3, x, y, z);
div_impl!(Vector3, x, y, z);
scalar_add_impl!(Vector3, x, y, z);
scalar_sub_impl!(Vector3, x, y, z);
scalar_mul_impl!(Vector3, x, y, z);
scalar_div_impl!(Vector3, x, y, z);
neg_impl!(Vector3, x, y, z);
dot_impl!(Vector3, x, y, z);
translation_impl!(Vector3);
norm_impl!(Vector3, x, y ,z);
approx_eq_impl!(Vector3, x, y, z);
zero_one_impl!(Vector3, x, y, z);
from_iterator_impl!(Vector3, iterator, iterator, iterator);
bounded_impl!(Vector3, x, y, z);
axpy_impl!(Vector3, x, y, z);
iterable_impl!(Vector3, 3);
iterable_mut_impl!(Vector3, 3);
vec_to_homogeneous_impl!(Vector3, Vector4, w, x, y, z);
vec_from_homogeneous_impl!(Vector3, Vector4, w, x, y, z);
translate_impl!(Vector3, Point3);
rotate_impl!(Vector3);
rotate_impl!(Point3);
transform_impl!(Vector3, Point3);
vec_as_point_impl!(Vector3, Point3, x, y, z);
num_float_vec_impl!(Vector3);
absolute_vec_impl!(Vector3, x, y, z);
arbitrary_impl!(Vector3, x, y, z);
rand_impl!(Vector3, x, y, z);
mean_impl!(Vector3);
vec_display_impl!(Vector3);


/// Vector of dimension 4.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector4<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N,
    /// Fourth component of the vector.
    pub w: N
}

new_impl!(Vector4, x, y, z, w);
pord_impl!(Vector4, x, y, z, w);
vec_axis_impl!(Vector4, x, y, z, w);
vec_cast_impl!(Vector4, x, y, z, w);
conversion_impl!(Vector4, 4);
index_impl!(Vector4);
indexable_impl!(Vector4, 4);
at_fast_impl!(Vector4, 4);
repeat_impl!(Vector4, val, x, y, z, w);
dim_impl!(Vector4, 4);
container_impl!(Vector4);
basis_impl!(Vector4, 4);
add_impl!(Vector4, x, y, z, w);
sub_impl!(Vector4, x, y, z, w);
mul_impl!(Vector4, x, y, z, w);
div_impl!(Vector4, x, y, z, w);
scalar_add_impl!(Vector4, x, y, z, w);
scalar_sub_impl!(Vector4, x, y, z, w);
scalar_mul_impl!(Vector4, x, y, z, w);
scalar_div_impl!(Vector4, x, y, z, w);
neg_impl!(Vector4, x, y, z, w);
dot_impl!(Vector4, x, y, z, w);
translation_impl!(Vector4);
norm_impl!(Vector4, x, y, z, w);
approx_eq_impl!(Vector4, x, y, z, w);
zero_one_impl!(Vector4, x, y, z, w);
from_iterator_impl!(Vector4, iterator, iterator, iterator, iterator);
bounded_impl!(Vector4, x, y, z, w);
axpy_impl!(Vector4, x, y, z, w);
iterable_impl!(Vector4, 4);
iterable_mut_impl!(Vector4, 4);
vec_to_homogeneous_impl!(Vector4, Vector5, a, x, y, z, w);
vec_from_homogeneous_impl!(Vector4, Vector5, a, x, y, z, w);
translate_impl!(Vector4, Point4);
rotate_impl!(Vector4);
rotate_impl!(Point4);
transform_impl!(Vector4, Point4);
vec_as_point_impl!(Vector4, Point4, x, y, z, w);
num_float_vec_impl!(Vector4);
absolute_vec_impl!(Vector4, x, y, z, w);
arbitrary_impl!(Vector4, x, y, z, w);
rand_impl!(Vector4, x, y, z, w);
mean_impl!(Vector4);
vec_display_impl!(Vector4);

/// Vector of dimension 5.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector5<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N,
    /// Fourth component of the vector.
    pub w: N,
    /// Fifth of the vector.
    pub a: N
}

new_impl!(Vector5, x, y, z, w, a);
pord_impl!(Vector5, x, y, z, w, a);
vec_axis_impl!(Vector5, x, y, z, w, a);
vec_cast_impl!(Vector5, x, y, z, w, a);
conversion_impl!(Vector5, 5);
index_impl!(Vector5);
indexable_impl!(Vector5, 5);
at_fast_impl!(Vector5, 5);
repeat_impl!(Vector5, val, x, y, z, w, a);
dim_impl!(Vector5, 5);
container_impl!(Vector5);
basis_impl!(Vector5, 5);
add_impl!(Vector5, x, y, z, w, a);
sub_impl!(Vector5, x, y, z, w, a);
mul_impl!(Vector5, x, y, z, w, a);
div_impl!(Vector5, x, y, z, w, a);
scalar_add_impl!(Vector5, x, y, z, w, a);
scalar_sub_impl!(Vector5, x, y, z, w, a);
scalar_mul_impl!(Vector5, x, y, z, w, a);
scalar_div_impl!(Vector5, x, y, z, w, a);
neg_impl!(Vector5, x, y, z, w, a);
dot_impl!(Vector5, x, y, z, w, a);
translation_impl!(Vector5);
norm_impl!(Vector5, x, y, z, w, a);
approx_eq_impl!(Vector5, x, y, z, w, a);
zero_one_impl!(Vector5, x, y, z, w, a);
from_iterator_impl!(Vector5, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Vector5, x, y, z, w, a);
axpy_impl!(Vector5, x, y, z, w, a);
iterable_impl!(Vector5, 5);
iterable_mut_impl!(Vector5, 5);
vec_to_homogeneous_impl!(Vector5, Vector6, b, x, y, z, w, a);
vec_from_homogeneous_impl!(Vector5, Vector6, b, x, y, z, w, a);
translate_impl!(Vector5, Point5);
rotate_impl!(Vector5);
rotate_impl!(Point5);
transform_impl!(Vector5, Point5);
vec_as_point_impl!(Vector5, Point5, x, y, z, w, a);
num_float_vec_impl!(Vector5);
absolute_vec_impl!(Vector5, x, y, z, w, a);
arbitrary_impl!(Vector5, x, y, z, w, a);
rand_impl!(Vector5, x, y, z, w, a);
mean_impl!(Vector5);
vec_display_impl!(Vector5);

/// Vector of dimension 6.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector6<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N,
    /// Fourth component of the vector.
    pub w: N,
    /// Fifth of the vector.
    pub a: N,
    /// Sixth component of the vector.
    pub b: N
}

new_impl!(Vector6, x, y, z, w, a, b);
pord_impl!(Vector6, x, y, z, w, a, b);
vec_axis_impl!(Vector6, x, y, z, w, a, b);
vec_cast_impl!(Vector6, x, y, z, w, a, b);
conversion_impl!(Vector6, 6);
index_impl!(Vector6);
indexable_impl!(Vector6, 6);
at_fast_impl!(Vector6, 6);
repeat_impl!(Vector6, val, x, y, z, w, a, b);
dim_impl!(Vector6, 6);
container_impl!(Vector6);
basis_impl!(Vector6, 6);
add_impl!(Vector6, x, y, z, w, a, b);
sub_impl!(Vector6, x, y, z, w, a, b);
mul_impl!(Vector6, x, y, z, w, a, b);
div_impl!(Vector6, x, y, z, w, a, b);
scalar_add_impl!(Vector6, x, y, z, w, a, b);
scalar_sub_impl!(Vector6, x, y, z, w, a, b);
scalar_mul_impl!(Vector6, x, y, z, w, a, b);
scalar_div_impl!(Vector6, x, y, z, w, a, b);
neg_impl!(Vector6, x, y, z, w, a, b);
dot_impl!(Vector6, x, y, z, w, a, b);
translation_impl!(Vector6);
norm_impl!(Vector6, x, y, z, w, a, b);
approx_eq_impl!(Vector6, x, y, z, w, a, b);
zero_one_impl!(Vector6, x, y, z, w, a, b);
from_iterator_impl!(Vector6, iterator, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Vector6, x, y, z, w, a, b);
axpy_impl!(Vector6, x, y, z, w, a, b);
iterable_impl!(Vector6, 6);
iterable_mut_impl!(Vector6, 6);
translate_impl!(Vector6, Point6);
rotate_impl!(Vector6);
rotate_impl!(Point6);
transform_impl!(Vector6, Point6);
vec_as_point_impl!(Vector6, Point6, x, y, z, w, a, b);
num_float_vec_impl!(Vector6);
absolute_vec_impl!(Vector6, x, y, z, w, a, b);
arbitrary_impl!(Vector6, x, y, z, w, a, b);
rand_impl!(Vector6, x, y, z, w, a, b);
mean_impl!(Vector6);
vec_display_impl!(Vector6);
