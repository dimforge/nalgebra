//! Points with dimension known at compile-time.

use std::mem;
use std::fmt;
use std::slice::{Iter, IterMut};
use std::iter::{Iterator, FromIterator, IntoIterator};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::operations::{ApproxEq, POrd, POrdering, Axpy};
use traits::structure::{Cast, Dim, Indexable, Iterable, IterableMut, PntAsVec, Shape,
                        NumPnt, FloatPnt, BaseFloat, BaseNum, Bounded, Repeat};
use traits::geometry::{Orig, FromHomogeneous, ToHomogeneous};
use structs::vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Point of dimension 1.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt1<N> {
    /// First component of the point.
    pub x: N
}

new_impl!(Pnt1, x);
orig_impl!(Pnt1, x);
pord_impl!(Pnt1, x,);
scalar_mul_impl!(Pnt1, x);
scalar_div_impl!(Pnt1, x);
scalar_add_impl!(Pnt1, x);
scalar_sub_impl!(Pnt1, x);
vec_cast_impl!(Pnt1, x);
conversion_impl!(Pnt1, 1);
index_impl!(Pnt1);
indexable_impl!(Pnt1, 1);
at_fast_impl!(Pnt1, 1);
repeat_impl!(Pnt1, val, x);
dim_impl!(Pnt1, 1);
container_impl!(Pnt1);
pnt_as_vec_impl!(Pnt1, Vec1, x);
pnt_sub_impl!(Pnt1, Vec1);
neg_impl!(Pnt1, x);
pnt_add_vec_impl!(Pnt1, Vec1, x);
pnt_sub_vec_impl!(Pnt1, Vec1, x);
approx_eq_impl!(Pnt1, x);
from_iterator_impl!(Pnt1, iterator);
bounded_impl!(Pnt1, x);
axpy_impl!(Pnt1, x);
iterable_impl!(Pnt1, 1);
iterable_mut_impl!(Pnt1, 1);
pnt_to_homogeneous_impl!(Pnt1, Pnt2, y, x);
pnt_from_homogeneous_impl!(Pnt1, Pnt2, y, x);
num_float_pnt_impl!(Pnt1, Vec1);
arbitrary_pnt_impl!(Pnt1, x);
rand_impl!(Pnt1, x);
pnt_display_impl!(Pnt1);

/// Point of dimension 2.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt2<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N
}

new_impl!(Pnt2, x, y);
orig_impl!(Pnt2, x, y);
pord_impl!(Pnt2, x, y);
scalar_mul_impl!(Pnt2, x, y);
scalar_div_impl!(Pnt2, x, y);
scalar_add_impl!(Pnt2, x, y);
scalar_sub_impl!(Pnt2, x, y);
vec_cast_impl!(Pnt2, x, y);
conversion_impl!(Pnt2, 2);
index_impl!(Pnt2);
indexable_impl!(Pnt2, 2);
at_fast_impl!(Pnt2, 2);
repeat_impl!(Pnt2, val, x, y);
dim_impl!(Pnt2, 2);
container_impl!(Pnt2);
pnt_as_vec_impl!(Pnt2, Vec2, x, y);
pnt_sub_impl!(Pnt2, Vec2);
neg_impl!(Pnt2, x, y);
pnt_add_vec_impl!(Pnt2, Vec2, x, y);
pnt_sub_vec_impl!(Pnt2, Vec2, x, y);
approx_eq_impl!(Pnt2, x, y);
from_iterator_impl!(Pnt2, iterator, iterator);
bounded_impl!(Pnt2, x, y);
axpy_impl!(Pnt2, x, y);
iterable_impl!(Pnt2, 2);
iterable_mut_impl!(Pnt2, 2);
pnt_to_homogeneous_impl!(Pnt2, Pnt3, z, x, y);
pnt_from_homogeneous_impl!(Pnt2, Pnt3, z, x, y);
num_float_pnt_impl!(Pnt2, Vec2);
arbitrary_pnt_impl!(Pnt2, x, y);
rand_impl!(Pnt2, x, y);
pnt_display_impl!(Pnt2);

/// Point of dimension 3.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt3<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N
}

new_impl!(Pnt3, x, y, z);
orig_impl!(Pnt3, x, y, z);
pord_impl!(Pnt3, x, y, z);
scalar_mul_impl!(Pnt3, x, y, z);
scalar_div_impl!(Pnt3, x, y, z);
scalar_add_impl!(Pnt3, x, y, z);
scalar_sub_impl!(Pnt3, x, y, z);
vec_cast_impl!(Pnt3, x, y, z);
conversion_impl!(Pnt3, 3);
index_impl!(Pnt3);
indexable_impl!(Pnt3, 3);
at_fast_impl!(Pnt3, 3);
repeat_impl!(Pnt3, val, x, y, z);
dim_impl!(Pnt3, 3);
container_impl!(Pnt3);
pnt_as_vec_impl!(Pnt3, Vec3, x, y, z);
pnt_sub_impl!(Pnt3, Vec3);
neg_impl!(Pnt3, x, y, z);
pnt_add_vec_impl!(Pnt3, Vec3, x, y, z);
pnt_sub_vec_impl!(Pnt3, Vec3, x, y, z);
approx_eq_impl!(Pnt3, x, y, z);
from_iterator_impl!(Pnt3, iterator, iterator, iterator);
bounded_impl!(Pnt3, x, y, z);
axpy_impl!(Pnt3, x, y, z);
iterable_impl!(Pnt3, 3);
iterable_mut_impl!(Pnt3, 3);
pnt_to_homogeneous_impl!(Pnt3, Pnt4, w, x, y, z);
pnt_from_homogeneous_impl!(Pnt3, Pnt4, w, x, y, z);
num_float_pnt_impl!(Pnt3, Vec3);
arbitrary_pnt_impl!(Pnt3, x, y, z);
rand_impl!(Pnt3, x, y, z);
pnt_display_impl!(Pnt3);

/// Point of dimension 4.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt4<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N,
    /// Third component of the point.
    pub z: N,
    /// Fourth component of the point.
    pub w: N
}

new_impl!(Pnt4, x, y, z, w);
orig_impl!(Pnt4, x, y, z, w);
pord_impl!(Pnt4, x, y, z, w);
scalar_mul_impl!(Pnt4, x, y, z, w);
scalar_div_impl!(Pnt4, x, y, z, w);
scalar_add_impl!(Pnt4, x, y, z, w);
scalar_sub_impl!(Pnt4, x, y, z, w);
vec_cast_impl!(Pnt4, x, y, z, w);
conversion_impl!(Pnt4, 4);
index_impl!(Pnt4);
indexable_impl!(Pnt4, 4);
at_fast_impl!(Pnt4, 4);
repeat_impl!(Pnt4, val, x, y, z, w);
dim_impl!(Pnt4, 4);
container_impl!(Pnt4);
pnt_as_vec_impl!(Pnt4, Vec4, x, y, z, w);
pnt_sub_impl!(Pnt4, Vec4);
neg_impl!(Pnt4, x, y, z, w);
pnt_add_vec_impl!(Pnt4, Vec4, x, y, z, w);
pnt_sub_vec_impl!(Pnt4, Vec4, x, y, z, w);
approx_eq_impl!(Pnt4, x, y, z, w);
from_iterator_impl!(Pnt4, iterator, iterator, iterator, iterator);
bounded_impl!(Pnt4, x, y, z, w);
axpy_impl!(Pnt4, x, y, z, w);
iterable_impl!(Pnt4, 4);
iterable_mut_impl!(Pnt4, 4);
pnt_to_homogeneous_impl!(Pnt4, Pnt5, a, x, y, z, w);
pnt_from_homogeneous_impl!(Pnt4, Pnt5, a, x, y, z, w);
num_float_pnt_impl!(Pnt4, Vec4);
arbitrary_pnt_impl!(Pnt4, x, y, z, w);
rand_impl!(Pnt4, x, y, z, w);
pnt_display_impl!(Pnt4);

/// Point of dimension 5.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt5<N> {
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

new_impl!(Pnt5, x, y, z, w, a);
orig_impl!(Pnt5, x, y, z, w, a);
pord_impl!(Pnt5, x, y, z, w, a);
scalar_mul_impl!(Pnt5, x, y, z, w, a);
scalar_div_impl!(Pnt5, x, y, z, w, a);
scalar_add_impl!(Pnt5, x, y, z, w, a);
scalar_sub_impl!(Pnt5, x, y, z, w, a);
vec_cast_impl!(Pnt5, x, y, z, w, a);
conversion_impl!(Pnt5, 5);
index_impl!(Pnt5);
indexable_impl!(Pnt5, 5);
at_fast_impl!(Pnt5, 5);
repeat_impl!(Pnt5, val, x, y, z, w, a);
dim_impl!(Pnt5, 5);
container_impl!(Pnt5);
pnt_as_vec_impl!(Pnt5, Vec5, x, y, z, w, a);
pnt_sub_impl!(Pnt5, Vec5);
neg_impl!(Pnt5, x, y, z, w, a);
pnt_add_vec_impl!(Pnt5, Vec5, x, y, z, w, a);
pnt_sub_vec_impl!(Pnt5, Vec5, x, y, z, w, a);
approx_eq_impl!(Pnt5, x, y, z, w, a);
from_iterator_impl!(Pnt5, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Pnt5, x, y, z, w, a);
axpy_impl!(Pnt5, x, y, z, w, a);
iterable_impl!(Pnt5, 5);
iterable_mut_impl!(Pnt5, 5);
pnt_to_homogeneous_impl!(Pnt5, Pnt6, b, x, y, z, w, a);
pnt_from_homogeneous_impl!(Pnt5, Pnt6, b, x, y, z, w, a);
num_float_pnt_impl!(Pnt5, Vec5);
arbitrary_pnt_impl!(Pnt5, x, y, z, w, a);
rand_impl!(Pnt5, x, y, z, w, a);
pnt_display_impl!(Pnt5);

/// Point of dimension 6.
///
/// The main differance between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Pnt6<N> {
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

new_impl!(Pnt6, x, y, z, w, a, b);
orig_impl!(Pnt6, x, y, z, w, a, b);
pord_impl!(Pnt6, x, y, z, w, a, b);
scalar_mul_impl!(Pnt6, x, y, z, w, a, b);
scalar_div_impl!(Pnt6, x, y, z, w, a, b);
scalar_add_impl!(Pnt6, x, y, z, w, a, b);
scalar_sub_impl!(Pnt6, x, y, z, w, a, b);
vec_cast_impl!(Pnt6, x, y, z, w, a, b);
conversion_impl!(Pnt6, 6);
index_impl!(Pnt6);
indexable_impl!(Pnt6, 6);
at_fast_impl!(Pnt6, 6);
repeat_impl!(Pnt6, val, x, y, z, w, a, b);
dim_impl!(Pnt6, 6);
container_impl!(Pnt6);
pnt_as_vec_impl!(Pnt6, Vec6, x, y, z, w, a, b);
pnt_sub_impl!(Pnt6, Vec6);
neg_impl!(Pnt6, x, y, z, w, a, b);
pnt_add_vec_impl!(Pnt6, Vec6, x, y, z, w, a, b);
pnt_sub_vec_impl!(Pnt6, Vec6, x, y, z, w, a, b);
approx_eq_impl!(Pnt6, x, y, z, w, a, b);
from_iterator_impl!(Pnt6, iterator, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Pnt6, x, y, z, w, a, b);
axpy_impl!(Pnt6, x, y, z, w, a, b);
iterable_impl!(Pnt6, 6);
iterable_mut_impl!(Pnt6, 6);
num_float_pnt_impl!(Pnt6, Vec6);
arbitrary_pnt_impl!(Pnt6, x, y, z, w, a, b);
rand_impl!(Pnt6, x, y, z, w, a, b);
pnt_display_impl!(Pnt6);
