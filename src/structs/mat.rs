//! Matrices with dimensions known at compile-time.

#![allow(missing_docs)] // we allow missing to avoid having to document the mij components.

use std::mem;
use traits::operations::ApproxEq;
use std::slice::{Items, MutItems};
use structs::vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
use structs::pnt::{Pnt1, Pnt4, Pnt5, Pnt6};
use structs::dvec::{DVec1, DVec2, DVec3, DVec4, DVec5, DVec6};

use traits::structure::{Cast, Row, Col, Iterable, IterableMut, Dim, Indexable,
                        Eye, ColSlice, RowSlice, Diag, Shape, BaseFloat, BaseNum, Zero, One};
use traits::operations::{Absolute, Transpose, Inv, Outer, EigenQR};
use traits::geometry::{ToHomogeneous, FromHomogeneous, Orig};
use linalg;


/// Special identity matrix. All its operation are no-ops.
#[deriving(Eq, PartialEq, Decodable, Clone, Rand, Show, Copy)]
pub struct Identity;

impl Identity {
    /// Creates a new identity matrix.
    #[inline]
    pub fn new() -> Identity {
        Identity
    }
}

/// Square matrix of dimension 1.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat1<N> {
    pub m11: N
}

eye_impl!(Mat1, 1, m11);

mat_impl!(Mat1, m11);
as_array_impl!(Mat1, 1);
mat_cast_impl!(Mat1, m11);
add_impl!(Mat1, m11);
sub_impl!(Mat1, m11);
scalar_add_impl!(Mat1, m11);
scalar_sub_impl!(Mat1, m11);
scalar_mul_impl!(Mat1, m11);
scalar_div_impl!(Mat1, m11);
absolute_impl!(Mat1, m11);
zero_impl!(Mat1, m11);
one_impl!(Mat1, ::one);
iterable_impl!(Mat1, 1);
iterable_mut_impl!(Mat1, 1);
at_fast_impl!(Mat1, 1);
dim_impl!(Mat1, 1);
indexable_impl!(Mat1, 1);
index_impl!(Mat1, 1);
mat_mul_mat_impl!(Mat1, 1);
mat_mul_vec_impl!(Mat1, Vec1, 1, ::zero);
vec_mul_mat_impl!(Mat1, Vec1, 1, ::zero);
mat_mul_pnt_impl!(Mat1, Pnt1, 1, Orig::orig);
pnt_mul_mat_impl!(Mat1, Pnt1, 1, Orig::orig);
// (specialized); inv_impl!(Mat1, 1);
transpose_impl!(Mat1, 1);
approx_eq_impl!(Mat1);
row_impl!(Mat1, Vec1, 1);
col_impl!(Mat1, Vec1, 1);
col_slice_impl!(Mat1, Vec1, DVec1, 1);
row_slice_impl!(Mat1, Vec1, DVec1, 1);
diag_impl!(Mat1, Vec1, 1);
to_homogeneous_impl!(Mat1, Mat2, 1, 2);
from_homogeneous_impl!(Mat1, Mat2, 1, 2);
outer_impl!(Vec1, Mat1);
eigen_qr_impl!(Mat1, Vec1);

/// Square matrix of dimension 2.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat2<N> {
    pub m11: N, pub m21: N,
    pub m12: N, pub m22: N
}

eye_impl!(Mat2, 2, m11, m22);

mat_impl!(Mat2, m11, m12,
                m21, m22);
as_array_impl!(Mat2, 2);
mat_cast_impl!(Mat2, m11, m12,
                     m21, m22);
add_impl!(Mat2, m11, m12, m21, m22);
sub_impl!(Mat2, m11, m12, m21, m22);
scalar_add_impl!(Mat2, m11, m12, m21, m22);
scalar_sub_impl!(Mat2, m11, m12, m21, m22);
scalar_mul_impl!(Mat2, m11, m12, m21, m22);
scalar_div_impl!(Mat2, m11, m12, m21, m22);
absolute_impl!(Mat2, m11, m12,
                     m21, m22);
zero_impl!(Mat2, m11, m12,
                 m21, m22);
one_impl!(Mat2, ::one,  ::zero,
                ::zero, ::one);
iterable_impl!(Mat2, 2);
iterable_mut_impl!(Mat2, 2);
dim_impl!(Mat2, 2);
indexable_impl!(Mat2, 2);
index_impl!(Mat2, 2);
at_fast_impl!(Mat2, 2);
// (specialized); mul_impl!(Mat2, 2);
// (specialized); rmul_impl!(Mat2, Vec2, 2);
// (specialized); lmul_impl!(Mat2, Vec2, 2);
// (specialized); inv_impl!(Mat2, 2);
transpose_impl!(Mat2, 2);
approx_eq_impl!(Mat2);
row_impl!(Mat2, Vec2, 2);
col_impl!(Mat2, Vec2, 2);
col_slice_impl!(Mat2, Vec2, DVec2, 2);
row_slice_impl!(Mat2, Vec2, DVec2, 2);
diag_impl!(Mat2, Vec2, 2);
to_homogeneous_impl!(Mat2, Mat3, 2, 3);
from_homogeneous_impl!(Mat2, Mat3, 2, 3);
outer_impl!(Vec2, Mat2);
eigen_qr_impl!(Mat2, Vec2);

/// Square matrix of dimension 3.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat3<N> {
    pub m11: N, pub m21: N, pub m31: N,
    pub m12: N, pub m22: N, pub m32: N,
    pub m13: N, pub m23: N, pub m33: N
}

eye_impl!(Mat3, 3, m11, m22, m33);

mat_impl!(Mat3, m11, m12, m13,
                m21, m22, m23,
                m31, m32, m33);
as_array_impl!(Mat3, 3);
mat_cast_impl!(Mat3, m11, m12, m13,
                     m21, m22, m23,
                     m31, m32, m33);
add_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
sub_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_add_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_sub_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_mul_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_div_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
absolute_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
zero_impl!(Mat3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
one_impl!(Mat3, ::one , ::zero, ::zero,
                ::zero, ::one , ::zero,
                ::zero, ::zero, ::one);
iterable_impl!(Mat3, 3);
iterable_mut_impl!(Mat3, 3);
dim_impl!(Mat3, 3);
indexable_impl!(Mat3, 3);
index_impl!(Mat3, 3);
at_fast_impl!(Mat3, 3);
// (specialized); mul_impl!(Mat3, 3);
// (specialized); rmul_impl!(Mat3, Vec3, 3);
// (specialized); lmul_impl!(Mat3, Vec3, 3);
// (specialized); inv_impl!(Mat3, 3);
transpose_impl!(Mat3, 3);
approx_eq_impl!(Mat3);
// (specialized); row_impl!(Mat3, Vec3, 3);
// (specialized); col_impl!(Mat3, Vec3, 3);
col_slice_impl!(Mat3, Vec3, DVec3, 3);
row_slice_impl!(Mat3, Vec3, DVec3, 3);
diag_impl!(Mat3, Vec3, 3);
to_homogeneous_impl!(Mat3, Mat4, 3, 4);
from_homogeneous_impl!(Mat3, Mat4, 3, 4);
outer_impl!(Vec3, Mat3);
eigen_qr_impl!(Mat3, Vec3);

/// Square matrix of dimension 4.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat4<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N
}

eye_impl!(Mat4, 4, m11, m22, m33, m44);

mat_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
as_array_impl!(Mat4, 4);
mat_cast_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
add_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
sub_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_add_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_sub_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_mul_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_div_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
absolute_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
zero_impl!(Mat4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
one_impl!(Mat4, ::one  , ::zero, ::zero, ::zero,
                ::zero, ::one  , ::zero, ::zero,
                ::zero, ::zero, ::one  , ::zero,
                ::zero, ::zero, ::zero, ::one);
iterable_impl!(Mat4, 4);
iterable_mut_impl!(Mat4, 4);
dim_impl!(Mat4, 4);
indexable_impl!(Mat4, 4);
index_impl!(Mat4, 4);
at_fast_impl!(Mat4, 4);
mat_mul_mat_impl!(Mat4, 4);
mat_mul_vec_impl!(Mat4, Vec4, 4, ::zero);
vec_mul_mat_impl!(Mat4, Vec4, 4, ::zero);
mat_mul_pnt_impl!(Mat4, Pnt4, 4, Orig::orig);
pnt_mul_mat_impl!(Mat4, Pnt4, 4, Orig::orig);
inv_impl!(Mat4, 4);
transpose_impl!(Mat4, 4);
approx_eq_impl!(Mat4);
row_impl!(Mat4, Vec4, 4);
col_impl!(Mat4, Vec4, 4);
col_slice_impl!(Mat4, Vec4, DVec4, 4);
row_slice_impl!(Mat4, Vec4, DVec4, 4);
diag_impl!(Mat4, Vec4, 4);
to_homogeneous_impl!(Mat4, Mat5, 4, 5);
from_homogeneous_impl!(Mat4, Mat5, 4, 5);
outer_impl!(Vec4, Mat4);
eigen_qr_impl!(Mat4, Vec4);

/// Square matrix of dimension 5.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat5<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N
}

eye_impl!(Mat5, 5, m11, m22, m33, m44, m55);

mat_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
as_array_impl!(Mat5, 5);
mat_cast_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
absolute_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
zero_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
one_impl!(Mat5,
  ::one  , ::zero, ::zero, ::zero, ::zero,
  ::zero, ::one  , ::zero, ::zero, ::zero,
  ::zero, ::zero, ::one  , ::zero, ::zero,
  ::zero, ::zero, ::zero, ::one  , ::zero,
  ::zero, ::zero, ::zero, ::zero, ::one
);
add_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
sub_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_add_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_sub_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_mul_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_div_impl!(Mat5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
iterable_impl!(Mat5, 5);
iterable_mut_impl!(Mat5, 5);
dim_impl!(Mat5, 5);
indexable_impl!(Mat5, 5);
index_impl!(Mat5, 5);
at_fast_impl!(Mat5, 5);
mat_mul_mat_impl!(Mat5, 5);
mat_mul_vec_impl!(Mat5, Vec5, 5, ::zero);
vec_mul_mat_impl!(Mat5, Vec5, 5, ::zero);
mat_mul_pnt_impl!(Mat5, Pnt5, 5, Orig::orig);
pnt_mul_mat_impl!(Mat5, Pnt5, 5, Orig::orig);
inv_impl!(Mat5, 5);
transpose_impl!(Mat5, 5);
approx_eq_impl!(Mat5);
row_impl!(Mat5, Vec5, 5);
col_impl!(Mat5, Vec5, 5);
col_slice_impl!(Mat5, Vec5, DVec5, 5);
row_slice_impl!(Mat5, Vec5, DVec5, 5);
diag_impl!(Mat5, Vec5, 5);
to_homogeneous_impl!(Mat5, Mat6, 5, 6);
from_homogeneous_impl!(Mat5, Mat6, 5, 6);
outer_impl!(Vec5, Mat5);
eigen_qr_impl!(Mat5, Vec5);

/// Square matrix of dimension 6.
#[deriving(Eq, PartialEq, Encodable, Decodable, Clone, Hash, Rand, Show, Copy)]
pub struct Mat6<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N, pub m61: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N, pub m62: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N, pub m63: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N, pub m64: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N, pub m65: N,
    pub m16: N, pub m26: N, pub m36: N, pub m46: N, pub m56: N, pub m66: N
}

eye_impl!(Mat6, 6, m11, m22, m33, m44, m55, m66);

mat_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
as_array_impl!(Mat6, 6);
mat_cast_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
add_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
sub_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_add_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_sub_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_mul_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_div_impl!(Mat6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
absolute_impl!(Mat6, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66);

zero_impl!(Mat6, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66);

one_impl!(Mat6,
  ::one  , ::zero, ::zero, ::zero, ::zero, ::zero,
  ::zero, ::one  , ::zero, ::zero, ::zero, ::zero,
  ::zero, ::zero, ::one  , ::zero, ::zero, ::zero,
  ::zero, ::zero, ::zero, ::one  , ::zero, ::zero,
  ::zero, ::zero, ::zero, ::zero, ::one  , ::zero,
  ::zero, ::zero, ::zero, ::zero, ::zero, ::one
);
iterable_impl!(Mat6, 6);
iterable_mut_impl!(Mat6, 6);
dim_impl!(Mat6, 6);
indexable_impl!(Mat6, 6);
index_impl!(Mat6, 6);
at_fast_impl!(Mat6, 6);
mat_mul_mat_impl!(Mat6, 6);
mat_mul_vec_impl!(Mat6, Vec6, 6, ::zero);
vec_mul_mat_impl!(Mat6, Vec6, 6, ::zero);
mat_mul_pnt_impl!(Mat6, Pnt6, 6, Orig::orig);
pnt_mul_mat_impl!(Mat6, Pnt6, 6, Orig::orig);
inv_impl!(Mat6, 6);
transpose_impl!(Mat6, 6);
approx_eq_impl!(Mat6);
row_impl!(Mat6, Vec6, 6);
col_impl!(Mat6, Vec6, 6);
col_slice_impl!(Mat6, Vec6, DVec6, 6);
row_slice_impl!(Mat6, Vec6, DVec6, 6);
diag_impl!(Mat6, Vec6, 6);
outer_impl!(Vec6, Mat6);
eigen_qr_impl!(Mat6, Vec6);
