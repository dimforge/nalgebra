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

#[cfg(feature="abstract_algebra")]
use_euclidean_space_modules!();


/// Point of dimension 1.
///
/// The main difference between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point1<N> {
    /// First component of the point.
    pub x: N
}

point_impl!(Point1, Vector1, Point2, y | x);
vectorlike_impl!(Point1, 1, x);
from_iterator_impl!(Point1, iterator);

/// Point of dimension 2.
///
/// The main difference between a point and a vector is that a vector is not affected by
/// translations.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Point2<N> {
    /// First component of the point.
    pub x: N,
    /// Second component of the point.
    pub y: N
}

point_impl!(Point2, Vector2, Point3, z | x, y);
vectorlike_impl!(Point2, 2, x, y);
from_iterator_impl!(Point2, iterator, iterator);

/// Point of dimension 3.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

point_impl!(Point3, Vector3, Point4, w | x, y, z);
vectorlike_impl!(Point3, 3, x, y, z);
from_iterator_impl!(Point3, iterator, iterator, iterator);

/// Point of dimension 4.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

point_impl!(Point4, Vector4, Point5, a | x, y, z, w);
vectorlike_impl!(Point4, 4, x, y, z, w);
from_iterator_impl!(Point4, iterator, iterator, iterator, iterator);

/// Point of dimension 5.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

point_impl!(Point5, Vector5, Point6, b | x, y, z, w, a);
vectorlike_impl!(Point5, 5, x, y, z, w, a);
from_iterator_impl!(Point5, iterator, iterator, iterator, iterator, iterator);

/// Point of dimension 6.
///
/// The main difference between a point and a vector is that a vector is not affected by
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

point_impl!(Point6, Vector6 | x, y, z, w, a, b);
vectorlike_impl!(Point6, 6, x, y, z, w, a, b);
from_iterator_impl!(Point6, iterator, iterator, iterator, iterator, iterator, iterator);
