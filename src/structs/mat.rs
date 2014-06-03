//! Matrices with dimensions known at compile-time.

#![allow(missing_doc)] // we allow missing to avoid having to document the mij components.

use std::mem;
use std::num::{One, Zero};
use traits::operations::ApproxEq;
use std::slice::{Items, MutItems};
use structs::vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6, Vec1MulRhs, Vec4MulRhs,
                   Vec5MulRhs, Vec6MulRhs};
use structs::dvec::DVec;

use traits::structure::{Cast, Row, Col, Iterable, IterableMut, Dim, Indexable,
                        Eye, ColSlice, RowSlice};
use traits::operations::{Absolute, Transpose, Inv, Outer};
use traits::geometry::{ToHomogeneous, FromHomogeneous};


/// Special identity matrix. All its operation are no-ops.
#[deriving(Eq, PartialEq, Decodable, Clone, Rand, Show)]
pub struct Identity;

impl Identity {
    /// Creates a new identity matrix.
    #[inline]
    pub fn new() -> Identity {
        Identity
    }
}

/// Square matrix of dimension 1.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat1<N> {
    pub m11: N
}

eye_impl!(Mat1, 1, m11)

double_dispatch_binop_decl_trait!(Mat1, Mat1MulRhs)
double_dispatch_binop_decl_trait!(Mat1, Mat1DivRhs)
double_dispatch_binop_decl_trait!(Mat1, Mat1AddRhs)
double_dispatch_binop_decl_trait!(Mat1, Mat1SubRhs)
double_dispatch_cast_decl_trait!(Mat1, Mat1Cast)
mul_redispatch_impl!(Mat1, Mat1MulRhs)
div_redispatch_impl!(Mat1, Mat1DivRhs)
add_redispatch_impl!(Mat1, Mat1AddRhs)
sub_redispatch_impl!(Mat1, Mat1SubRhs)
cast_redispatch_impl!(Mat1, Mat1Cast)
mat_impl!(Mat1, m11)
mat_cast_impl!(Mat1, Mat1Cast, m11)
add_impl!(Mat1, Mat1AddRhs, m11)
sub_impl!(Mat1, Mat1SubRhs, m11)

mat_mul_scalar_impl!(Mat1, f64, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, f32, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, i64, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, i32, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, i16, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, i8, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, u64, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, u32, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, u16, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, u8, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, uint, Mat1MulRhs, m11)
mat_mul_scalar_impl!(Mat1, int, Mat1MulRhs, m11)

mat_div_scalar_impl!(Mat1, f64, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, f32, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, i64, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, i32, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, i16, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, i8, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, u64, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, u32, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, u16, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, u8, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, uint, Mat1DivRhs, m11)
mat_div_scalar_impl!(Mat1, int, Mat1DivRhs, m11)

mat_add_scalar_impl!(Mat1, f64, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, f32, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, i64, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, i32, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, i16, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, i8, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, u64, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, u32, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, u16, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, u8, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, uint, Mat1AddRhs, m11)
mat_add_scalar_impl!(Mat1, int, Mat1AddRhs, m11)

mat_sub_scalar_impl!(Mat1, f64, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, f32, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, i64, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, i32, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, i16, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, i8, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, u64, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, u32, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, u16, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, u8, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, uint, Mat1SubRhs, m11)
mat_sub_scalar_impl!(Mat1, int, Mat1SubRhs, m11)

absolute_impl!(Mat1, m11)
one_impl!(Mat1, One::one)
iterable_impl!(Mat1, 1)
iterable_mut_impl!(Mat1, 1)
at_fast_impl!(Mat1, 1)
dim_impl!(Mat1, 1)
indexable_impl!(Mat1, 1)
mat_mul_mat_impl!(Mat1, Mat1MulRhs, 1)
mat_mul_vec_impl!(Mat1, Vec1, Mat1MulRhs, 1)
vec_mul_mat_impl!(Mat1, Vec1, Vec1MulRhs, 1)
// (specialized) inv_impl!(Mat1, 1)
transpose_impl!(Mat1, 1)
approx_eq_impl!(Mat1)
row_impl!(Mat1, Vec1, 1)
col_impl!(Mat1, Vec1, 1)
col_slice_impl!(Mat1, Vec1, 1)
row_slice_impl!(Mat1, Vec1, 1)
to_homogeneous_impl!(Mat1, Mat2, 1, 2)
from_homogeneous_impl!(Mat1, Mat2, 1, 2)
outer_impl!(Vec1, Mat1)

/// Square matrix of dimension 2.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat2<N> {
    pub m11: N, pub m21: N,
    pub m12: N, pub m22: N
}

eye_impl!(Mat2, 2, m11, m22)

double_dispatch_binop_decl_trait!(Mat2, Mat2MulRhs)
double_dispatch_binop_decl_trait!(Mat2, Mat2DivRhs)
double_dispatch_binop_decl_trait!(Mat2, Mat2AddRhs)
double_dispatch_binop_decl_trait!(Mat2, Mat2SubRhs)
double_dispatch_cast_decl_trait!(Mat2, Mat2Cast)
mul_redispatch_impl!(Mat2, Mat2MulRhs)
div_redispatch_impl!(Mat2, Mat2DivRhs)
add_redispatch_impl!(Mat2, Mat2AddRhs)
sub_redispatch_impl!(Mat2, Mat2SubRhs)
cast_redispatch_impl!(Mat2, Mat2Cast)
mat_impl!(Mat2, m11, m12,
                m21, m22)
mat_cast_impl!(Mat2, Mat2Cast, m11, m12,
                               m21, m22)
add_impl!(Mat2, Mat2AddRhs, m11, m12, m21, m22)
sub_impl!(Mat2, Mat2SubRhs, m11, m12, m21, m22)

mat_mul_scalar_impl!(Mat2, f64, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, f32, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, i64, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, i32, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, i16, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, i8, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, u64, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, u32, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, u16, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, u8, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, uint, Mat2MulRhs, m11, m12, m21, m22)
mat_mul_scalar_impl!(Mat2, int, Mat2MulRhs, m11, m12, m21, m22)

mat_div_scalar_impl!(Mat2, f64, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, f32, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, i64, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, i32, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, i16, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, i8, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, u64, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, u32, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, u16, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, u8, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, uint, Mat2DivRhs, m11, m12, m21, m22)
mat_div_scalar_impl!(Mat2, int, Mat2DivRhs, m11, m12, m21, m22)

mat_add_scalar_impl!(Mat2, f64, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, f32, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, i64, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, i32, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, i16, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, i8, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, u64, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, u32, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, u16, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, u8, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, uint, Mat2AddRhs, m11, m12, m21, m22)
mat_add_scalar_impl!(Mat2, int, Mat2AddRhs, m11, m12, m21, m22)

mat_sub_scalar_impl!(Mat2, f64, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, f32, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, i64, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, i32, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, i16, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, i8, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, u64, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, u32, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, u16, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, u8, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, uint, Mat2SubRhs, m11, m12, m21, m22)
mat_sub_scalar_impl!(Mat2, int, Mat2SubRhs, m11, m12, m21, m22)

absolute_impl!(Mat2, m11, m12,
                     m21, m22)
one_impl!(Mat2, One::one,   Zero::zero,
                Zero::zero, One::one)
iterable_impl!(Mat2, 2)
iterable_mut_impl!(Mat2, 2)
dim_impl!(Mat2, 2)
indexable_impl!(Mat2, 2)
at_fast_impl!(Mat2, 2)
// (specialized) mul_impl!(Mat2, 2)
// (specialized) rmul_impl!(Mat2, Vec2, 2)
// (specialized) lmul_impl!(Mat2, Vec2, 2)
// (specialized) inv_impl!(Mat2, 2)
transpose_impl!(Mat2, 2)
approx_eq_impl!(Mat2)
row_impl!(Mat2, Vec2, 2)
col_impl!(Mat2, Vec2, 2)
col_slice_impl!(Mat2, Vec2, 2)
row_slice_impl!(Mat2, Vec2, 2)
to_homogeneous_impl!(Mat2, Mat3, 2, 3)
from_homogeneous_impl!(Mat2, Mat3, 2, 3)
outer_impl!(Vec2, Mat2)

/// Square matrix of dimension 3.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat3<N> {
    pub m11: N, pub m21: N, pub m31: N,
    pub m12: N, pub m22: N, pub m32: N,
    pub m13: N, pub m23: N, pub m33: N
}

eye_impl!(Mat3, 3, m11, m22, m33)

double_dispatch_binop_decl_trait!(Mat3, Mat3MulRhs)
double_dispatch_binop_decl_trait!(Mat3, Mat3DivRhs)
double_dispatch_binop_decl_trait!(Mat3, Mat3AddRhs)
double_dispatch_binop_decl_trait!(Mat3, Mat3SubRhs)
double_dispatch_cast_decl_trait!(Mat3, Mat3Cast)
mul_redispatch_impl!(Mat3, Mat3MulRhs)
div_redispatch_impl!(Mat3, Mat3DivRhs)
add_redispatch_impl!(Mat3, Mat3AddRhs)
sub_redispatch_impl!(Mat3, Mat3SubRhs)
cast_redispatch_impl!(Mat3, Mat3Cast)
mat_impl!(Mat3, m11, m12, m13,
                m21, m22, m23,
                m31, m32, m33)
mat_cast_impl!(Mat3, Mat3Cast, m11, m12, m13,
                               m21, m22, m23,
                               m31, m32, m33)
add_impl!(Mat3, Mat3AddRhs,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
)
sub_impl!(Mat3, Mat3SubRhs,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
)
mat_mul_scalar_impl!(Mat3, f64, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, f32, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, i64, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, i32, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, i16, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, i8, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, u64, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, u32, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, u16, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, u8, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, uint, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_mul_scalar_impl!(Mat3, int, Mat3MulRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)

mat_div_scalar_impl!(Mat3, f64, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, f32, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, i64, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, i32, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, i16, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, i8, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, u64, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, u32, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, u16, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, u8, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, uint, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_div_scalar_impl!(Mat3, int, Mat3DivRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)

mat_add_scalar_impl!(Mat3, f64, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, f32, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, i64, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, i32, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, i16, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, i8, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, u64, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, u32, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, u16, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, u8, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, uint, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_add_scalar_impl!(Mat3, int, Mat3AddRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)

mat_sub_scalar_impl!(Mat3, f64, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, f32, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, i64, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, i32, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, i16, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, i8, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, u64, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, u32, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, u16, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, u8, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, uint, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)
mat_sub_scalar_impl!(Mat3, int, Mat3SubRhs, m11, m12, m13, m21, m22, m23, m31, m32, m33)

absolute_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
)
one_impl!(Mat3, One::one  , Zero::zero, Zero::zero,
                Zero::zero, One::one  , Zero::zero,
                Zero::zero, Zero::zero, One::one)
iterable_impl!(Mat3, 3)
iterable_mut_impl!(Mat3, 3)
dim_impl!(Mat3, 3)
indexable_impl!(Mat3, 3)
at_fast_impl!(Mat3, 3)
// (specialized) mul_impl!(Mat3, 3)
// (specialized) rmul_impl!(Mat3, Vec3, 3)
// (specialized) lmul_impl!(Mat3, Vec3, 3)
// (specialized) inv_impl!(Mat3, 3)
transpose_impl!(Mat3, 3)
approx_eq_impl!(Mat3)
// (specialized) row_impl!(Mat3, Vec3, 3)
// (specialized) col_impl!(Mat3, Vec3, 3)
col_slice_impl!(Mat3, Vec3, 3)
row_slice_impl!(Mat3, Vec3, 3)
to_homogeneous_impl!(Mat3, Mat4, 3, 4)
from_homogeneous_impl!(Mat3, Mat4, 3, 4)
outer_impl!(Vec3, Mat3)

/// Square matrix of dimension 4.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat4<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N
}

eye_impl!(Mat4, 4, m11, m22, m33, m44)

double_dispatch_binop_decl_trait!(Mat4, Mat4MulRhs)
double_dispatch_binop_decl_trait!(Mat4, Mat4DivRhs)
double_dispatch_binop_decl_trait!(Mat4, Mat4AddRhs)
double_dispatch_binop_decl_trait!(Mat4, Mat4SubRhs)
double_dispatch_cast_decl_trait!(Mat4, Mat4Cast)
mul_redispatch_impl!(Mat4, Mat4MulRhs)
div_redispatch_impl!(Mat4, Mat4DivRhs)
add_redispatch_impl!(Mat4, Mat4AddRhs)
sub_redispatch_impl!(Mat4, Mat4SubRhs)
cast_redispatch_impl!(Mat4, Mat4Cast)
mat_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
mat_cast_impl!(Mat4, Mat4Cast,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
add_impl!(Mat4, Mat4AddRhs,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
sub_impl!(Mat4, Mat4SubRhs,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)

mat_mul_scalar_impl!(Mat4, f64, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, f32, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, i64, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, i32, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, i16, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, i8, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, u32, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, u16, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, u8, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, uint, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_mul_scalar_impl!(Mat4, int, Mat4MulRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)

mat_div_scalar_impl!(Mat4, f64, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, f32, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, i64, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, i32, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, i16, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, i8, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, u32, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, u16, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, u8, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, uint, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_div_scalar_impl!(Mat4, int, Mat4DivRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)

mat_add_scalar_impl!(Mat4, f64, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, f32, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, i64, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, i32, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, i16, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, i8, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, u32, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, u16, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, u8, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, uint, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_add_scalar_impl!(Mat4, int, Mat4AddRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)

mat_sub_scalar_impl!(Mat4, f64, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, f32, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, i64, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, i32, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, i16, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, i8, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, u32, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, u16, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, u8, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, uint, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)
mat_sub_scalar_impl!(Mat4, int, Mat4SubRhs, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34,
  m41, m42, m43, m44)

absolute_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
)
one_impl!(Mat4, One::one  , Zero::zero, Zero::zero, Zero::zero,
                Zero::zero, One::one  , Zero::zero, Zero::zero,
                Zero::zero, Zero::zero, One::one  , Zero::zero,
                Zero::zero, Zero::zero, Zero::zero, One::one)
iterable_impl!(Mat4, 4)
iterable_mut_impl!(Mat4, 4)
dim_impl!(Mat4, 4)
indexable_impl!(Mat4, 4)
at_fast_impl!(Mat4, 4)
mat_mul_mat_impl!(Mat4, Mat4MulRhs, 4)
mat_mul_vec_impl!(Mat4, Vec4, Mat4MulRhs, 4)
vec_mul_mat_impl!(Mat4, Vec4, Vec4MulRhs, 4)
inv_impl!(Mat4, 4)
transpose_impl!(Mat4, 4)
approx_eq_impl!(Mat4)
row_impl!(Mat4, Vec4, 4)
col_impl!(Mat4, Vec4, 4)
col_slice_impl!(Mat4, Vec4, 4)
row_slice_impl!(Mat4, Vec4, 4)
to_homogeneous_impl!(Mat4, Mat5, 4, 5)
from_homogeneous_impl!(Mat4, Mat5, 4, 5)
outer_impl!(Vec4, Mat4)

/// Square matrix of dimension 5.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat5<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N
}

eye_impl!(Mat5, 5, m11, m22, m33, m44, m55)

double_dispatch_binop_decl_trait!(Mat5, Mat5MulRhs)
double_dispatch_binop_decl_trait!(Mat5, Mat5DivRhs)
double_dispatch_binop_decl_trait!(Mat5, Mat5AddRhs)
double_dispatch_binop_decl_trait!(Mat5, Mat5SubRhs)
double_dispatch_cast_decl_trait!(Mat5, Mat5Cast)
mul_redispatch_impl!(Mat5, Mat5MulRhs)
div_redispatch_impl!(Mat5, Mat5DivRhs)
add_redispatch_impl!(Mat5, Mat5AddRhs)
sub_redispatch_impl!(Mat5, Mat5SubRhs)
cast_redispatch_impl!(Mat5, Mat5Cast)
mat_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
mat_cast_impl!(Mat5, Mat5Cast,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
absolute_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
one_impl!(Mat5,
  One::one  , Zero::zero, Zero::zero, Zero::zero, Zero::zero,
  Zero::zero, One::one  , Zero::zero, Zero::zero, Zero::zero,
  Zero::zero, Zero::zero, One::one  , Zero::zero, Zero::zero,
  Zero::zero, Zero::zero, Zero::zero, One::one  , Zero::zero,
  Zero::zero, Zero::zero, Zero::zero, Zero::zero, One::one
)
add_impl!(Mat5, Mat5AddRhs,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
sub_impl!(Mat5, Mat5SubRhs,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
)
mat_mul_scalar_impl!(Mat5, f64, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, f32, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, i64, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, i32, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, i16, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, i8, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, u64, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, u32, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, u16, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, u8, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, uint, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_mul_scalar_impl!(Mat5, int, Mat5MulRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)

mat_div_scalar_impl!(Mat5, f64, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, f32, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, i64, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, i32, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, i16, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, i8, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, u64, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, u32, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, u16, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, u8, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, uint, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_div_scalar_impl!(Mat5, int, Mat5DivRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)

mat_add_scalar_impl!(Mat5, f64, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, f32, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, i64, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, i32, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, i16, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, i8, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, u64, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, u32, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, u16, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, u8, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, uint, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_add_scalar_impl!(Mat5, int, Mat5AddRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)

mat_sub_scalar_impl!(Mat5, f64, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, f32, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, i64, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, i32, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, i16, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, i8, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, u64, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, u32, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, u16, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, u8, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, uint, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)
mat_sub_scalar_impl!(Mat5, int, Mat5SubRhs, m11, m12, m13, m14, m15, m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35, m41, m42, m43, m44, m45, m51, m52, m53, m54, m55)

iterable_impl!(Mat5, 5)
iterable_mut_impl!(Mat5, 5)
dim_impl!(Mat5, 5)
indexable_impl!(Mat5, 5)
at_fast_impl!(Mat5, 5)
mat_mul_mat_impl!(Mat5, Mat5MulRhs, 5)
mat_mul_vec_impl!(Mat5, Vec5, Mat5MulRhs, 5)
vec_mul_mat_impl!(Mat5, Vec5, Vec5MulRhs, 5)
inv_impl!(Mat5, 5)
transpose_impl!(Mat5, 5)
approx_eq_impl!(Mat5)
row_impl!(Mat5, Vec5, 5)
col_impl!(Mat5, Vec5, 5)
col_slice_impl!(Mat5, Vec5, 5)
row_slice_impl!(Mat5, Vec5, 5)
to_homogeneous_impl!(Mat5, Mat6, 5, 6)
from_homogeneous_impl!(Mat5, Mat6, 5, 6)
outer_impl!(Vec5, Mat5)

/// Square matrix of dimension 6.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Zero, Show)]
pub struct Mat6<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N, pub m61: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N, pub m62: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N, pub m63: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N, pub m64: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N, pub m65: N,
    pub m16: N, pub m26: N, pub m36: N, pub m46: N, pub m56: N, pub m66: N
}

eye_impl!(Mat6, 6, m11, m22, m33, m44, m55, m66)

double_dispatch_binop_decl_trait!(Mat6, Mat6MulRhs)
double_dispatch_binop_decl_trait!(Mat6, Mat6DivRhs)
double_dispatch_binop_decl_trait!(Mat6, Mat6AddRhs)
double_dispatch_binop_decl_trait!(Mat6, Mat6SubRhs)
double_dispatch_cast_decl_trait!(Mat6, Mat6Cast)
mul_redispatch_impl!(Mat6, Mat6MulRhs)
div_redispatch_impl!(Mat6, Mat6DivRhs)
add_redispatch_impl!(Mat6, Mat6AddRhs)
sub_redispatch_impl!(Mat6, Mat6SubRhs)
cast_redispatch_impl!(Mat6, Mat6Cast)
mat_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)
mat_cast_impl!(Mat6, Mat6Cast,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)
add_impl!(Mat6, Mat6AddRhs,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)
sub_impl!(Mat6, Mat6SubRhs,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
)

mat_mul_scalar_impl!(Mat6, f64, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, f32, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, i64, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, i32, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, i16, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, i8, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, u64, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, u32, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, u16, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, u8, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, uint, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_mul_scalar_impl!(Mat6, int, Mat6MulRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)

mat_div_scalar_impl!(Mat6, f64, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, f32, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, i64, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, i32, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, i16, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, i8, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, u64, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, u32, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, u16, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, u8, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, uint, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_div_scalar_impl!(Mat6, int, Mat6DivRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)

mat_add_scalar_impl!(Mat6, f64, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, f32, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, i64, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, i32, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, i16, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, i8, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, u64, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, u32, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, u16, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, u8, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, uint, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_add_scalar_impl!(Mat6, int, Mat6AddRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)

mat_sub_scalar_impl!(Mat6, f64, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, f32, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, i64, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, i32, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, i16, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, i8, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, u64, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, u32, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, u16, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, u8, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, uint, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)
mat_sub_scalar_impl!(Mat6, int, Mat6SubRhs, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)

absolute_impl!(Mat6, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66)

one_impl!(Mat6,
  One::one  , Zero::zero, Zero::zero, Zero::zero, Zero::zero, Zero::zero,
  Zero::zero, One::one  , Zero::zero, Zero::zero, Zero::zero, Zero::zero,
  Zero::zero, Zero::zero, One::one  , Zero::zero, Zero::zero, Zero::zero,
  Zero::zero, Zero::zero, Zero::zero, One::one  , Zero::zero, Zero::zero,
  Zero::zero, Zero::zero, Zero::zero, Zero::zero, One::one  , Zero::zero,
  Zero::zero, Zero::zero, Zero::zero, Zero::zero, Zero::zero, One::one
)
iterable_impl!(Mat6, 6)
iterable_mut_impl!(Mat6, 6)
dim_impl!(Mat6, 6)
indexable_impl!(Mat6, 6)
at_fast_impl!(Mat6, 6)
mat_mul_mat_impl!(Mat6, Mat6MulRhs, 6)
mat_mul_vec_impl!(Mat6, Vec6, Mat6MulRhs, 6)
vec_mul_mat_impl!(Mat6, Vec6, Vec6MulRhs, 6)
inv_impl!(Mat6, 6)
transpose_impl!(Mat6, 6)
approx_eq_impl!(Mat6)
row_impl!(Mat6, Vec6, 6)
col_impl!(Mat6, Vec6, 6)
col_slice_impl!(Mat6, Vec6, 6)
row_slice_impl!(Mat6, Vec6, 6)
outer_impl!(Vec6, Mat6)
