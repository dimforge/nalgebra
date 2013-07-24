#[allow(missing_doc)]; // we allow missing to avoid having to document the mij components.

use std::cast;
use std::uint::iterate;
use std::num::{One, Zero};
use std::cmp::ApproxEq;
use std::iterator::IteratorUtil;
use std::vec::{VecIterator, VecMutIterator};
use vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
use traits::dim::Dim;
use traits::ring::Ring;
use traits::inv::Inv;
use traits::division_ring::DivisionRing;
use traits::transpose::Transpose;
use traits::rlmul::{RMul, LMul};
use traits::transformation::Transform;
use traits::homogeneous::{FromHomogeneous, ToHomogeneous};
use traits::indexable::Indexable;
use traits::column::Column;
use traits::iterable::{Iterable, IterableMut};

pub use traits::mat_cast::*;
pub use traits::column::*;
pub use traits::inv::*;
pub use traits::rlmul::*;
pub use traits::rotation::*;
pub use traits::transformation::*;
pub use traits::translation::*;
pub use traits::transpose::*;

mod mat_macros;

/// Square matrix of dimension 1.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat1<N>
{
  m11: N
}

mat_impl!(Mat1, m11)
mat_cast_impl!(Mat1, m11)
one_impl!(Mat1, _1)
iterable_impl!(Mat1, 1)
iterable_mut_impl!(Mat1, 1)
dim_impl!(Mat1, 1)
indexable_impl!(Mat1, 1)
mul_impl!(Mat1, 1)
rmul_impl!(Mat1, Vec1, 1)
lmul_impl!(Mat1, Vec1, 1)
transform_impl!(Mat1, Vec1)
// (specialized) inv_impl!(Mat1, 1)
transpose_impl!(Mat1, 1)
approx_eq_impl!(Mat1)
column_impl!(Mat1, 1)
to_homogeneous_impl!(Mat1, Mat2, 1, 2)
from_homogeneous_impl!(Mat1, Mat2, 1, 2)

/// Square matrix of dimension 2.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat2<N>
{
  m11: N, m12: N,
  m21: N, m22: N
}

mat_impl!(Mat2, m11, m12,
                m21, m22)
mat_cast_impl!(Mat2, m11, m12,
                     m21, m22)
one_impl!(Mat2, _1, _0,
                _0, _1)
iterable_impl!(Mat2, 2)
iterable_mut_impl!(Mat2, 2)
dim_impl!(Mat2, 2)
indexable_impl!(Mat2, 2)
mul_impl!(Mat2, 2)
rmul_impl!(Mat2, Vec2, 2)
lmul_impl!(Mat2, Vec2, 2)
transform_impl!(Mat2, Vec2)
// (specialized) inv_impl!(Mat2, 2)
transpose_impl!(Mat2, 2)
approx_eq_impl!(Mat2)
column_impl!(Mat2, 2)
to_homogeneous_impl!(Mat2, Mat3, 2, 3)
from_homogeneous_impl!(Mat2, Mat3, 2, 3)

/// Square matrix of dimension 3.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat3<N>
{
  m11: N, m12: N, m13: N,
  m21: N, m22: N, m23: N,
  m31: N, m32: N, m33: N
}

mat_impl!(Mat3, m11, m12, m13,
                m21, m22, m23,
                m31, m32, m33)
mat_cast_impl!(Mat3, m11, m12, m13,
                     m21, m22, m23,
                     m31, m32, m33)
one_impl!(Mat3, _1, _0, _0,
                _0, _1, _0,
                _0, _0, _1)
iterable_impl!(Mat3, 3)
iterable_mut_impl!(Mat3, 3)
dim_impl!(Mat3, 3)
indexable_impl!(Mat3, 3)
mul_impl!(Mat3, 3)
rmul_impl!(Mat3, Vec3, 3)
lmul_impl!(Mat3, Vec3, 3)
transform_impl!(Mat3, Vec3)
// (specialized) inv_impl!(Mat3, 3)
transpose_impl!(Mat3, 3)
approx_eq_impl!(Mat3)
column_impl!(Mat3, 3)
to_homogeneous_impl!(Mat3, Mat4, 3, 4)
from_homogeneous_impl!(Mat3, Mat4, 3, 4)

/// Square matrix of dimension 4.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat4<N>
{
  m11: N, m12: N, m13: N, m14: N,
  m21: N, m22: N, m23: N, m24: N,
  m31: N, m32: N, m33: N, m34: N,
  m41: N, m42: N, m43: N, m44: N
}

mat_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
mat_cast_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
one_impl!(Mat4, _1, _0, _0, _0,
                _0, _1, _0, _0,
                _0, _0, _1, _0,
                _0, _0, _0, _1)
iterable_impl!(Mat4, 4)
iterable_mut_impl!(Mat4, 4)
dim_impl!(Mat4, 4)
indexable_impl!(Mat4, 4)
mul_impl!(Mat4, 4)
rmul_impl!(Mat4, Vec4, 4)
lmul_impl!(Mat4, Vec4, 4)
transform_impl!(Mat4, Vec4)
inv_impl!(Mat4, 4)
transpose_impl!(Mat4, 4)
approx_eq_impl!(Mat4)
column_impl!(Mat4, 4)
to_homogeneous_impl!(Mat4, Mat5, 4, 5)
from_homogeneous_impl!(Mat4, Mat5, 4, 5)

/// Square matrix of dimension 5.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat5<N>
{
  m11: N, m12: N, m13: N, m14: N, m15: N,
  m21: N, m22: N, m23: N, m24: N, m25: N,
  m31: N, m32: N, m33: N, m34: N, m35: N,
  m41: N, m42: N, m43: N, m44: N, m45: N,
  m51: N, m52: N, m53: N, m54: N, m55: N
}

mat_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
mat_cast_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
one_impl!(Mat5,
  _1, _0, _0, _0, _0,
  _0, _1, _0, _0, _0,
  _0, _0, _1, _0, _0,
  _0, _0, _0, _1, _0,
  _0, _0, _0, _0, _1
)
iterable_impl!(Mat5, 5)
iterable_mut_impl!(Mat5, 5)
dim_impl!(Mat5, 5)
indexable_impl!(Mat5, 5)
mul_impl!(Mat5, 5)
rmul_impl!(Mat5, Vec5, 5)
lmul_impl!(Mat5, Vec5, 5)
transform_impl!(Mat5, Vec5)
inv_impl!(Mat5, 5)
transpose_impl!(Mat5, 5)
approx_eq_impl!(Mat5)
column_impl!(Mat5, 5)
to_homogeneous_impl!(Mat5, Mat6, 5, 6)
from_homogeneous_impl!(Mat5, Mat6, 5, 6)

/// Square matrix of dimension 6.
#[deriving(Eq, Ord, Encodable, Decodable, Clone, DeepClone, IterBytes, Rand, Zero, ToStr)]
pub struct Mat6<N>
{
  m11: N, m12: N, m13: N, m14: N, m15: N, m16: N,
  m21: N, m22: N, m23: N, m24: N, m25: N, m26: N,
  m31: N, m32: N, m33: N, m34: N, m35: N, m36: N,
  m41: N, m42: N, m43: N, m44: N, m45: N, m46: N,
  m51: N, m52: N, m53: N, m54: N, m55: N, m56: N,
  m61: N, m62: N, m63: N, m64: N, m65: N, m66: N
}

mat_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)
mat_cast_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)
one_impl!(Mat6,
  _1, _0, _0, _0, _0, _0,
  _0, _1, _0, _0, _0, _0,
  _0, _0, _1, _0, _0, _0,
  _0, _0, _0, _1, _0, _0,
  _0, _0, _0, _0, _1, _0,
  _0, _0, _0, _0, _0, _1
)
iterable_impl!(Mat6, 6)
iterable_mut_impl!(Mat6, 6)
dim_impl!(Mat6, 6)
indexable_impl!(Mat6, 6)
mul_impl!(Mat6, 6)
rmul_impl!(Mat6, Vec6, 6)
lmul_impl!(Mat6, Vec6, 6)
transform_impl!(Mat6, Vec6)
inv_impl!(Mat6, 6)
transpose_impl!(Mat6, 6)
approx_eq_impl!(Mat6)
column_impl!(Mat6, 6)
