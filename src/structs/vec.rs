//! Vectors with dimensions known at compile-time.

#![allow(missing_docs)] // we allow missing to avoid having to document the dispatch traits.

use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use std::marker::PhantomData;
use std::mem;
use std::slice::{Iter, IterMut};
use std::iter::{Iterator, FromIterator, IntoIterator};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::operations::{ApproxEq, POrd, POrdering, Axpy, Absolute, Mean};
use traits::geometry::{Transform, Rotate, FromHomogeneous, ToHomogeneous, Dot, Norm,
                       Translation, Translate};
use traits::structure::{Basis, Cast, Dim, Indexable, Iterable, IterableMut, Shape, NumVec,
                        FloatVec, BaseFloat, BaseNum, Bounded, Repeat};
use structs::pnt::{Pnt1, Pnt2, Pnt3, Pnt4, Pnt5, Pnt6};

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Vector of dimension 0.
#[repr(C)]
#[derive(Eq, PartialEq, Clone, Debug, Copy)]
pub struct Vec0<N>(pub PhantomData<N>);

impl<N> Vec0<N> {
    /// Creates a new vector.
    #[inline]
    pub fn new() -> Vec0<N> {
        Vec0(PhantomData)
    }
}

impl<N> Repeat<N> for Vec0<N> {
    #[inline]
    fn repeat(_: N) -> Vec0<N> {
        Vec0(PhantomData)
    }
}

/// Vector of dimension 1.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec1<N> {
    /// First component of the vector.
    pub x: N
}

new_impl!(Vec1, x);
pord_impl!(Vec1, x,);
vec_axis_impl!(Vec1, x);
vec_cast_impl!(Vec1, x);
conversion_impl!(Vec1, 1);
index_impl!(Vec1);
indexable_impl!(Vec1, 1);
at_fast_impl!(Vec1, 1);
repeat_impl!(Vec1, val, x);
dim_impl!(Vec1, 1);
container_impl!(Vec1);
// (specialized); basis_impl!(Vec1, 1);
add_impl!(Vec1, x);
sub_impl!(Vec1, x);
mul_impl!(Vec1, x);
div_impl!(Vec1, x);
scalar_add_impl!(Vec1, x);
scalar_sub_impl!(Vec1, x);
scalar_mul_impl!(Vec1, x);
scalar_div_impl!(Vec1, x);
neg_impl!(Vec1, x);
dot_impl!(Vec1, x);
translation_impl!(Vec1);
norm_impl!(Vec1, x);
approx_eq_impl!(Vec1, x);
zero_one_impl!(Vec1, x);
from_iterator_impl!(Vec1, iterator);
bounded_impl!(Vec1, x);
axpy_impl!(Vec1, x);
iterable_impl!(Vec1, 1);
iterable_mut_impl!(Vec1, 1);
vec_to_homogeneous_impl!(Vec1, Vec2, y, x);
vec_from_homogeneous_impl!(Vec1, Vec2, y, x);
translate_impl!(Vec1, Pnt1);
rotate_impl!(Vec1);
rotate_impl!(Pnt1);
transform_impl!(Vec1, Pnt1);
vec_as_pnt_impl!(Vec1, Pnt1, x);
num_float_vec_impl!(Vec1);
absolute_vec_impl!(Vec1, x);
arbitrary_impl!(Vec1, x);
rand_impl!(Vec1, x);
mean_impl!(Vec1);

/// Vector of dimension 2.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec2<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N
}

new_impl!(Vec2, x, y);
pord_impl!(Vec2, x, y);
vec_axis_impl!(Vec2, x, y);
vec_cast_impl!(Vec2, x, y);
conversion_impl!(Vec2, 2);
index_impl!(Vec2);
indexable_impl!(Vec2, 2);
at_fast_impl!(Vec2, 2);
repeat_impl!(Vec2, val, x, y);
dim_impl!(Vec2, 2);
container_impl!(Vec2);
// (specialized); basis_impl!(Vec2, 1);
add_impl!(Vec2, x, y);
sub_impl!(Vec2, x, y);
mul_impl!(Vec2, x, y);
div_impl!(Vec2, x, y);
scalar_add_impl!(Vec2, x, y);
scalar_sub_impl!(Vec2, x, y);
scalar_mul_impl!(Vec2, x, y);
scalar_div_impl!(Vec2, x, y);
neg_impl!(Vec2, x, y);
dot_impl!(Vec2, x, y);
translation_impl!(Vec2);
norm_impl!(Vec2, x, y);
approx_eq_impl!(Vec2, x, y);
zero_one_impl!(Vec2, x, y);
from_iterator_impl!(Vec2, iterator, iterator);
bounded_impl!(Vec2, x, y);
axpy_impl!(Vec2, x, y);
iterable_impl!(Vec2, 2);
iterable_mut_impl!(Vec2, 2);
vec_to_homogeneous_impl!(Vec2, Vec3, z, x, y);
vec_from_homogeneous_impl!(Vec2, Vec3, z, x, y);
translate_impl!(Vec2, Pnt2);
rotate_impl!(Vec2);
rotate_impl!(Pnt2);
transform_impl!(Vec2, Pnt2);
vec_as_pnt_impl!(Vec2, Pnt2, x, y);
num_float_vec_impl!(Vec2);
absolute_vec_impl!(Vec2, x, y);
arbitrary_impl!(Vec2, x, y);
rand_impl!(Vec2, x, y);
mean_impl!(Vec2);

/// Vector of dimension 3.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec3<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N
}

new_impl!(Vec3, x, y, z);
pord_impl!(Vec3, x, y, z);
vec_axis_impl!(Vec3, x, y, z);
vec_cast_impl!(Vec3, x, y, z);
conversion_impl!(Vec3, 3);
index_impl!(Vec3);
indexable_impl!(Vec3, 3);
at_fast_impl!(Vec3, 3);
repeat_impl!(Vec3, val, x, y, z);
dim_impl!(Vec3, 3);
container_impl!(Vec3);
// (specialized); basis_impl!(Vec3, 1);
add_impl!(Vec3, x, y, z);
sub_impl!(Vec3, x, y, z);
mul_impl!(Vec3, x, y, z);
div_impl!(Vec3, x, y, z);
scalar_add_impl!(Vec3, x, y, z);
scalar_sub_impl!(Vec3, x, y, z);
scalar_mul_impl!(Vec3, x, y, z);
scalar_div_impl!(Vec3, x, y, z);
neg_impl!(Vec3, x, y, z);
dot_impl!(Vec3, x, y, z);
translation_impl!(Vec3);
norm_impl!(Vec3, x, y ,z);
approx_eq_impl!(Vec3, x, y, z);
zero_one_impl!(Vec3, x, y, z);
from_iterator_impl!(Vec3, iterator, iterator, iterator);
bounded_impl!(Vec3, x, y, z);
axpy_impl!(Vec3, x, y, z);
iterable_impl!(Vec3, 3);
iterable_mut_impl!(Vec3, 3);
vec_to_homogeneous_impl!(Vec3, Vec4, w, x, y, z);
vec_from_homogeneous_impl!(Vec3, Vec4, w, x, y, z);
translate_impl!(Vec3, Pnt3);
rotate_impl!(Vec3);
rotate_impl!(Pnt3);
transform_impl!(Vec3, Pnt3);
vec_as_pnt_impl!(Vec3, Pnt3, x, y, z);
num_float_vec_impl!(Vec3);
absolute_vec_impl!(Vec3, x, y, z);
arbitrary_impl!(Vec3, x, y, z);
rand_impl!(Vec3, x, y, z);
mean_impl!(Vec3);


/// Vector of dimension 4.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec4<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N,
    /// Third component of the vector.
    pub z: N,
    /// Fourth component of the vector.
    pub w: N
}

new_impl!(Vec4, x, y, z, w);
pord_impl!(Vec4, x, y, z, w);
vec_axis_impl!(Vec4, x, y, z, w);
vec_cast_impl!(Vec4, x, y, z, w);
conversion_impl!(Vec4, 4);
index_impl!(Vec4);
indexable_impl!(Vec4, 4);
at_fast_impl!(Vec4, 4);
repeat_impl!(Vec4, val, x, y, z, w);
dim_impl!(Vec4, 4);
container_impl!(Vec4);
basis_impl!(Vec4, 4);
add_impl!(Vec4, x, y, z, w);
sub_impl!(Vec4, x, y, z, w);
mul_impl!(Vec4, x, y, z, w);
div_impl!(Vec4, x, y, z, w);
scalar_add_impl!(Vec4, x, y, z, w);
scalar_sub_impl!(Vec4, x, y, z, w);
scalar_mul_impl!(Vec4, x, y, z, w);
scalar_div_impl!(Vec4, x, y, z, w);
neg_impl!(Vec4, x, y, z, w);
dot_impl!(Vec4, x, y, z, w);
translation_impl!(Vec4);
norm_impl!(Vec4, x, y, z, w);
approx_eq_impl!(Vec4, x, y, z, w);
zero_one_impl!(Vec4, x, y, z, w);
from_iterator_impl!(Vec4, iterator, iterator, iterator, iterator);
bounded_impl!(Vec4, x, y, z, w);
axpy_impl!(Vec4, x, y, z, w);
iterable_impl!(Vec4, 4);
iterable_mut_impl!(Vec4, 4);
vec_to_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w);
vec_from_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w);
translate_impl!(Vec4, Pnt4);
rotate_impl!(Vec4);
rotate_impl!(Pnt4);
transform_impl!(Vec4, Pnt4);
vec_as_pnt_impl!(Vec4, Pnt4, x, y, z, w);
num_float_vec_impl!(Vec4);
absolute_vec_impl!(Vec4, x, y, z, w);
arbitrary_impl!(Vec4, x, y, z, w);
rand_impl!(Vec4, x, y, z, w);
mean_impl!(Vec4);

/// Vector of dimension 5.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec5<N> {
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

new_impl!(Vec5, x, y, z, w, a);
pord_impl!(Vec5, x, y, z, w, a);
vec_axis_impl!(Vec5, x, y, z, w, a);
vec_cast_impl!(Vec5, x, y, z, w, a);
conversion_impl!(Vec5, 5);
index_impl!(Vec5);
indexable_impl!(Vec5, 5);
at_fast_impl!(Vec5, 5);
repeat_impl!(Vec5, val, x, y, z, w, a);
dim_impl!(Vec5, 5);
container_impl!(Vec5);
basis_impl!(Vec5, 5);
add_impl!(Vec5, x, y, z, w, a);
sub_impl!(Vec5, x, y, z, w, a);
mul_impl!(Vec5, x, y, z, w, a);
div_impl!(Vec5, x, y, z, w, a);
scalar_add_impl!(Vec5, x, y, z, w, a);
scalar_sub_impl!(Vec5, x, y, z, w, a);
scalar_mul_impl!(Vec5, x, y, z, w, a);
scalar_div_impl!(Vec5, x, y, z, w, a);
neg_impl!(Vec5, x, y, z, w, a);
dot_impl!(Vec5, x, y, z, w, a);
translation_impl!(Vec5);
norm_impl!(Vec5, x, y, z, w, a);
approx_eq_impl!(Vec5, x, y, z, w, a);
zero_one_impl!(Vec5, x, y, z, w, a);
from_iterator_impl!(Vec5, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Vec5, x, y, z, w, a);
axpy_impl!(Vec5, x, y, z, w, a);
iterable_impl!(Vec5, 5);
iterable_mut_impl!(Vec5, 5);
vec_to_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a);
vec_from_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a);
translate_impl!(Vec5, Pnt5);
rotate_impl!(Vec5);
rotate_impl!(Pnt5);
transform_impl!(Vec5, Pnt5);
vec_as_pnt_impl!(Vec5, Pnt5, x, y, z, w, a);
num_float_vec_impl!(Vec5);
absolute_vec_impl!(Vec5, x, y, z, w, a);
arbitrary_impl!(Vec5, x, y, z, w, a);
rand_impl!(Vec5, x, y, z, w, a);
mean_impl!(Vec5);

/// Vector of dimension 6.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vec6<N> {
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

new_impl!(Vec6, x, y, z, w, a, b);
pord_impl!(Vec6, x, y, z, w, a, b);
vec_axis_impl!(Vec6, x, y, z, w, a, b);
vec_cast_impl!(Vec6, x, y, z, w, a, b);
conversion_impl!(Vec6, 6);
index_impl!(Vec6);
indexable_impl!(Vec6, 6);
at_fast_impl!(Vec6, 6);
repeat_impl!(Vec6, val, x, y, z, w, a, b);
dim_impl!(Vec6, 6);
container_impl!(Vec6);
basis_impl!(Vec6, 6);
add_impl!(Vec6, x, y, z, w, a, b);
sub_impl!(Vec6, x, y, z, w, a, b);
mul_impl!(Vec6, x, y, z, w, a, b);
div_impl!(Vec6, x, y, z, w, a, b);
scalar_add_impl!(Vec6, x, y, z, w, a, b);
scalar_sub_impl!(Vec6, x, y, z, w, a, b);
scalar_mul_impl!(Vec6, x, y, z, w, a, b);
scalar_div_impl!(Vec6, x, y, z, w, a, b);
neg_impl!(Vec6, x, y, z, w, a, b);
dot_impl!(Vec6, x, y, z, w, a, b);
translation_impl!(Vec6);
norm_impl!(Vec6, x, y, z, w, a, b);
approx_eq_impl!(Vec6, x, y, z, w, a, b);
zero_one_impl!(Vec6, x, y, z, w, a, b);
from_iterator_impl!(Vec6, iterator, iterator, iterator, iterator, iterator, iterator);
bounded_impl!(Vec6, x, y, z, w, a, b);
axpy_impl!(Vec6, x, y, z, w, a, b);
iterable_impl!(Vec6, 6);
iterable_mut_impl!(Vec6, 6);
translate_impl!(Vec6, Pnt6);
rotate_impl!(Vec6);
rotate_impl!(Pnt6);
transform_impl!(Vec6, Pnt6);
vec_as_pnt_impl!(Vec6, Pnt6, x, y, z, w, a, b);
num_float_vec_impl!(Vec6);
absolute_vec_impl!(Vec6, x, y, z, w, a, b);
arbitrary_impl!(Vec6, x, y, z, w, a, b);
rand_impl!(Vec6, x, y, z, w, a, b);
mean_impl!(Vec6);
