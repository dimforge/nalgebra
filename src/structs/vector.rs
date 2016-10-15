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

#[cfg(feature="abstract_algebra")]
use_vector_space_modules!();


/// Vector of dimension 1.
///
/// The main difference between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector1<N> {
    /// First component of the vector.
    pub x: N
}

vector_impl!(Vector1, Point1, 1, x);
vectorlike_impl!(Vector1, 1, x);
from_iterator_impl!(Vector1, iterator);
// (specialized); basis_impl!(Vector1, 1);
vec_to_homogeneous_impl!(Vector1, Vector2, y, x);
vec_from_homogeneous_impl!(Vector1, Vector2, y, x);



/// Vector of dimension 2.
///
/// The main difference between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Vector2<N> {
    /// First component of the vector.
    pub x: N,
    /// Second component of the vector.
    pub y: N
}

vector_impl!(Vector2, Point2, 2, x, y);
vectorlike_impl!(Vector2, 2, x, y);
from_iterator_impl!(Vector2, iterator, iterator);
// (specialized); basis_impl!(Vector2, 1);
vec_to_homogeneous_impl!(Vector2, Vector3, z, x, y);
vec_from_homogeneous_impl!(Vector2, Vector3, z, x, y);



/// Vector of dimension 3.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

vector_impl!(Vector3, Point3, 3, x, y, z);
vectorlike_impl!(Vector3, 3, x, y, z);
from_iterator_impl!(Vector3, iterator, iterator, iterator);
// (specialized); basis_impl!(Vector3, 1);
vec_to_homogeneous_impl!(Vector3, Vector4, w, x, y, z);
vec_from_homogeneous_impl!(Vector3, Vector4, w, x, y, z);


/// Vector of dimension 4.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

vector_impl!(Vector4, Point4, 4, x, y, z, w);
vectorlike_impl!(Vector4, 4, x, y, z, w);
from_iterator_impl!(Vector4, iterator, iterator, iterator, iterator);
basis_impl!(Vector4, 4);
vec_to_homogeneous_impl!(Vector4, Vector5, a, x, y, z, w);
vec_from_homogeneous_impl!(Vector4, Vector5, a, x, y, z, w);



/// Vector of dimension 5.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

vector_impl!(Vector5, Point5, 5, x, y, z, w, a);
vectorlike_impl!(Vector5, 5, x, y, z, w, a);
from_iterator_impl!(Vector5, iterator, iterator, iterator, iterator, iterator);
basis_impl!(Vector5, 5);
vec_to_homogeneous_impl!(Vector5, Vector6, b, x, y, z, w, a);
vec_from_homogeneous_impl!(Vector5, Vector6, b, x, y, z, w, a);


/// Vector of dimension 6.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

vector_impl!(Vector6, Point6, 6, x, y, z, w, a, b);
vectorlike_impl!(Vector6, 6, x, y, z, w, a, b);
from_iterator_impl!(Vector6, iterator, iterator, iterator, iterator, iterator, iterator);

basis_impl!(Vector6, 6);
