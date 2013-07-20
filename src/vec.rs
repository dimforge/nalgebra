use std::cast;
use std::num::{Zero, One, Algebraic, Bounded};
use std::rand::Rng;
use std::vec::{VecIterator, VecMutIterator};
use std::iterator::{Iterator, IteratorUtil, FromIterator};
use std::cmp::ApproxEq;
use std::uint::iterate;
use traits::iterable::{Iterable, IterableMut};
use traits::basis::Basis;
use traits::dim::Dim;
use traits::dot::Dot;
use traits::sub_dot::SubDot;
use traits::norm::Norm;
use traits::translation::{Translation, Translatable};
use traits::scalar_op::{ScalarMul, ScalarDiv, ScalarAdd, ScalarSub};
use traits::ring::Ring;
use traits::division_ring::DivisionRing;
use traits::homogeneous::{FromHomogeneous, ToHomogeneous};
use traits::indexable::Indexable;

mod vec_impl;

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, Rand, Zero, ToStr)]
pub struct Vec0<N>;

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec1<N>
{ x: N }

new_impl!(Vec1, x)
indexable_impl!(Vec1, 1)
new_repeat_impl!(Vec1, val, x)
dim_impl!(Vec1, 1)
// (specialized) basis_impl!(Vec1, 1)
add_impl!(Vec1, x)
sub_impl!(Vec1, x)
neg_impl!(Vec1, x)
dot_impl!(Vec1, x)
sub_dot_impl!(Vec1, x)
scalar_mul_impl!(Vec1, x)
scalar_div_impl!(Vec1, x)
scalar_add_impl!(Vec1, x)
scalar_sub_impl!(Vec1, x)
translation_impl!(Vec1)
translatable_impl!(Vec1)
norm_impl!(Vec1)
approx_eq_impl!(Vec1, x)
one_impl!(Vec1)
from_iterator_impl!(Vec1, iterator)
bounded_impl!(Vec1)
iterable_impl!(Vec1, 1)
iterable_mut_impl!(Vec1, 1)
to_homogeneous_impl!(Vec1, Vec2, y, x)
from_homogeneous_impl!(Vec1, Vec2, y, x)

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec2<N>
{
  x: N,
  y: N
}

new_impl!(Vec2, x, y)
indexable_impl!(Vec2, 2)
new_repeat_impl!(Vec2, val, x, y)
dim_impl!(Vec2, 2)
// (specialized) basis_impl!(Vec2, 1)
add_impl!(Vec2, x, y)
sub_impl!(Vec2, x, y)
neg_impl!(Vec2, x, y)
dot_impl!(Vec2, x, y)
sub_dot_impl!(Vec2, x, y)
scalar_mul_impl!(Vec2, x, y)
scalar_div_impl!(Vec2, x, y)
scalar_add_impl!(Vec2, x, y)
scalar_sub_impl!(Vec2, x, y)
translation_impl!(Vec2)
translatable_impl!(Vec2)
norm_impl!(Vec2)
approx_eq_impl!(Vec2, x, y)
one_impl!(Vec2)
from_iterator_impl!(Vec2, iterator, iterator)
bounded_impl!(Vec2)
iterable_impl!(Vec2, 2)
iterable_mut_impl!(Vec2, 2)
to_homogeneous_impl!(Vec2, Vec3, z, x, y)
from_homogeneous_impl!(Vec2, Vec3, z, x, y)

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec3<N>
{
  x: N,
  y: N,
  z: N
}

new_impl!(Vec3, x, y, z)
indexable_impl!(Vec3, 3)
new_repeat_impl!(Vec3, val, x, y, z)
dim_impl!(Vec3, 3)
// (specialized) basis_impl!(Vec3, 1)
add_impl!(Vec3, x, y, z)
sub_impl!(Vec3, x, y, z)
neg_impl!(Vec3, x, y, z)
dot_impl!(Vec3, x, y, z)
sub_dot_impl!(Vec3, x, y, z)
scalar_mul_impl!(Vec3, x, y, z)
scalar_div_impl!(Vec3, x, y, z)
scalar_add_impl!(Vec3, x, y, z)
scalar_sub_impl!(Vec3, x, y, z)
translation_impl!(Vec3)
translatable_impl!(Vec3)
norm_impl!(Vec3)
approx_eq_impl!(Vec3, x, y, z)
one_impl!(Vec3)
from_iterator_impl!(Vec3, iterator, iterator, iterator)
bounded_impl!(Vec3)
iterable_impl!(Vec3, 3)
iterable_mut_impl!(Vec3, 3)
to_homogeneous_impl!(Vec3, Vec4, w, x, y, z)
from_homogeneous_impl!(Vec3, Vec4, w, x, y, z)

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec4<N>
{
  x: N,
  y: N,
  z: N,
  w: N
}

new_impl!(Vec4, x, y, z, w)
indexable_impl!(Vec4, 4)
new_repeat_impl!(Vec4, val, x, y, z, w)
dim_impl!(Vec4, 4)
basis_impl!(Vec4, 4)
add_impl!(Vec4, x, y, z, w)
sub_impl!(Vec4, x, y, z, w)
neg_impl!(Vec4, x, y, z, w)
dot_impl!(Vec4, x, y, z, w)
sub_dot_impl!(Vec4, x, y, z, w)
scalar_mul_impl!(Vec4, x, y, z, w)
scalar_div_impl!(Vec4, x, y, z, w)
scalar_add_impl!(Vec4, x, y, z, w)
scalar_sub_impl!(Vec4, x, y, z, w)
translation_impl!(Vec4)
translatable_impl!(Vec4)
norm_impl!(Vec4)
approx_eq_impl!(Vec4, x, y, z, w)
one_impl!(Vec4)
from_iterator_impl!(Vec4, iterator, iterator, iterator, iterator)
bounded_impl!(Vec4)
iterable_impl!(Vec4, 4)
iterable_mut_impl!(Vec4, 4)
to_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w)
from_homogeneous_impl!(Vec4, Vec5, a, x, y, z, w)

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec5<N>
{
  x: N,
  y: N,
  z: N,
  w: N,
  a: N,
}

new_impl!(Vec5, x, y, z, w, a)
indexable_impl!(Vec5, 5)
new_repeat_impl!(Vec5, val, x, y, z, w, a)
dim_impl!(Vec5, 5)
basis_impl!(Vec5, 5)
add_impl!(Vec5, x, y, z, w, a)
sub_impl!(Vec5, x, y, z, w, a)
neg_impl!(Vec5, x, y, z, w, a)
dot_impl!(Vec5, x, y, z, w, a)
sub_dot_impl!(Vec5, x, y, z, w, a)
scalar_mul_impl!(Vec5, x, y, z, w, a)
scalar_div_impl!(Vec5, x, y, z, w, a)
scalar_add_impl!(Vec5, x, y, z, w, a)
scalar_sub_impl!(Vec5, x, y, z, w, a)
translation_impl!(Vec5)
translatable_impl!(Vec5)
norm_impl!(Vec5)
approx_eq_impl!(Vec5, x, y, z, w, a)
one_impl!(Vec5)
from_iterator_impl!(Vec5, iterator, iterator, iterator, iterator, iterator)
bounded_impl!(Vec5)
iterable_impl!(Vec5, 5)
iterable_mut_impl!(Vec5, 5)
to_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a)
from_homogeneous_impl!(Vec5, Vec6, b, x, y, z, w, a)

#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Vec6<N>
{
  x: N,
  y: N,
  z: N,
  w: N,
  a: N,
  b: N
}

new_impl!(Vec6, x, y, z, w, a, b)
indexable_impl!(Vec6, 6)
new_repeat_impl!(Vec6, val, x, y, z, w, a, b)
dim_impl!(Vec6, 6)
basis_impl!(Vec6, 6)
add_impl!(Vec6, x, y, z, w, a, b)
sub_impl!(Vec6, x, y, z, w, a, b)
neg_impl!(Vec6, x, y, z, w, a, b)
dot_impl!(Vec6, x, y, z, w, a, b)
sub_dot_impl!(Vec6, x, y, z, w, a, b)
scalar_mul_impl!(Vec6, x, y, z, w, a, b)
scalar_div_impl!(Vec6, x, y, z, w, a, b)
scalar_add_impl!(Vec6, x, y, z, w, a, b)
scalar_sub_impl!(Vec6, x, y, z, w, a, b)
translation_impl!(Vec6)
translatable_impl!(Vec6)
norm_impl!(Vec6)
approx_eq_impl!(Vec6, x, y, z, w, a, b)
one_impl!(Vec6)
from_iterator_impl!(Vec6, iterator, iterator, iterator, iterator, iterator, iterator)
bounded_impl!(Vec6)
iterable_impl!(Vec6, 6)
iterable_mut_impl!(Vec6, 6)
