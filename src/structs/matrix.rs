//! Matrixrices with dimensions known at compile-time.

#![allow(missing_docs)] // we allow missing to avoid having to document the mij components.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::mem;
use std::slice::{Iter, IterMut};
use rand::{Rand, Rng};
use num::{Zero, One};
use traits::operations::ApproxEq;
use structs::vector::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6};
use structs::point::{Point1, Point4, Point5, Point6};
use structs::dvector::{DVector1, DVector2, DVector3, DVector4, DVector5, DVector6};

use traits::structure::{Cast, Row, Column, Iterable, IterableMut, Dimension, Indexable, Eye, ColumnSlice,
                        RowSlice, Diagonal, DiagMut, Shape, BaseFloat, BaseNum, Repeat};
use traits::operations::{Absolute, Transpose, Inverse, Outer, EigenQR, Mean};
use traits::geometry::{ToHomogeneous, FromHomogeneous, Origin};
use linalg;
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};


/// Special identity matrix. All its operation are no-ops.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Debug, Copy)]
pub struct Identity;

impl Identity {
    /// Creates a new identity matrix.
    #[inline]
    pub fn new() -> Identity {
        Identity
    }
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Identity")
    }
}

/// Square matrix of dimension 1.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix1<N> {
    pub m11: N
}

eye_impl!(Matrix1, 1, m11);

mat_impl!(Matrix1, m11);
repeat_impl!(Matrix1, m11);
conversion_impl!(Matrix1, 1);
mat_cast_impl!(Matrix1, m11);
add_impl!(Matrix1, m11);
sub_impl!(Matrix1, m11);
scalar_add_impl!(Matrix1, m11);
scalar_sub_impl!(Matrix1, m11);
scalar_mul_impl!(Matrix1, m11);
scalar_div_impl!(Matrix1, m11);
absolute_impl!(Matrix1, m11);
zero_impl!(Matrix1, m11);
one_impl!(Matrix1, ::one);
iterable_impl!(Matrix1, 1);
iterable_mut_impl!(Matrix1, 1);
at_fast_impl!(Matrix1, 1);
dim_impl!(Matrix1, 1);
indexable_impl!(Matrix1, 1);
index_impl!(Matrix1, 1);
mat_mul_mat_impl!(Matrix1, 1);
mat_mul_vec_impl!(Matrix1, Vector1, 1, ::zero);
vec_mul_mat_impl!(Matrix1, Vector1, 1, ::zero);
mat_mul_point_impl!(Matrix1, Point1, 1, Origin::origin);
point_mul_mat_impl!(Matrix1, Point1, 1, Origin::origin);
// (specialized); inverse_impl!(Matrix1, 1);
transpose_impl!(Matrix1, 1);
approx_eq_impl!(Matrix1);
row_impl!(Matrix1, Vector1, 1);
column_impl!(Matrix1, Vector1, 1);
column_slice_impl!(Matrix1, Vector1, DVector1, 1);
row_slice_impl!(Matrix1, Vector1, DVector1, 1);
diag_impl!(Matrix1, Vector1, 1);
to_homogeneous_impl!(Matrix1, Matrix2, 1, 2);
from_homogeneous_impl!(Matrix1, Matrix2, 1, 2);
outer_impl!(Vector1, Matrix1);
eigen_qr_impl!(Matrix1, Vector1);
arbitrary_impl!(Matrix1, m11);
rand_impl!(Matrix1, m11);
mean_impl!(Matrix1, Vector1, 1);
mat_display_impl!(Matrix1, 1);

/// Square matrix of dimension 2.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix2<N> {
    pub m11: N, pub m21: N,
    pub m12: N, pub m22: N
}

eye_impl!(Matrix2, 2, m11, m22);

mat_impl!(Matrix2, m11, m12,
                m21, m22);
repeat_impl!(Matrix2, m11, m12,
                   m21, m22);
conversion_impl!(Matrix2, 2);
mat_cast_impl!(Matrix2, m11, m12,
                     m21, m22);
add_impl!(Matrix2, m11, m12, m21, m22);
sub_impl!(Matrix2, m11, m12, m21, m22);
scalar_add_impl!(Matrix2, m11, m12, m21, m22);
scalar_sub_impl!(Matrix2, m11, m12, m21, m22);
scalar_mul_impl!(Matrix2, m11, m12, m21, m22);
scalar_div_impl!(Matrix2, m11, m12, m21, m22);
absolute_impl!(Matrix2, m11, m12,
                     m21, m22);
zero_impl!(Matrix2, m11, m12,
                 m21, m22);
one_impl!(Matrix2, ::one,  ::zero,
                ::zero, ::one);
iterable_impl!(Matrix2, 2);
iterable_mut_impl!(Matrix2, 2);
dim_impl!(Matrix2, 2);
indexable_impl!(Matrix2, 2);
index_impl!(Matrix2, 2);
at_fast_impl!(Matrix2, 2);
// (specialized); mul_impl!(Matrix2, 2);
// (specialized); rmul_impl!(Matrix2, Vector2, 2);
// (specialized); lmul_impl!(Matrix2, Vector2, 2);
// (specialized); inverse_impl!(Matrix2, 2);
transpose_impl!(Matrix2, 2);
approx_eq_impl!(Matrix2);
row_impl!(Matrix2, Vector2, 2);
column_impl!(Matrix2, Vector2, 2);
column_slice_impl!(Matrix2, Vector2, DVector2, 2);
row_slice_impl!(Matrix2, Vector2, DVector2, 2);
diag_impl!(Matrix2, Vector2, 2);
to_homogeneous_impl!(Matrix2, Matrix3, 2, 3);
from_homogeneous_impl!(Matrix2, Matrix3, 2, 3);
outer_impl!(Vector2, Matrix2);
eigen_qr_impl!(Matrix2, Vector2);
arbitrary_impl!(Matrix2, m11, m12, m21, m22);
rand_impl!(Matrix2, m11, m12, m21, m22);
mean_impl!(Matrix2, Vector2, 2);
mat_display_impl!(Matrix2, 2);

/// Square matrix of dimension 3.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix3<N> {
    pub m11: N, pub m21: N, pub m31: N,
    pub m12: N, pub m22: N, pub m32: N,
    pub m13: N, pub m23: N, pub m33: N
}

eye_impl!(Matrix3, 3, m11, m22, m33);

mat_impl!(Matrix3, m11, m12, m13,
                m21, m22, m23,
                m31, m32, m33);
repeat_impl!(Matrix3, m11, m12, m13,
                   m21, m22, m23,
                   m31, m32, m33);
conversion_impl!(Matrix3, 3);
mat_cast_impl!(Matrix3, m11, m12, m13,
                     m21, m22, m23,
                     m31, m32, m33);
add_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
sub_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_add_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_sub_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_mul_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
scalar_div_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
absolute_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
zero_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
one_impl!(Matrix3, ::one , ::zero, ::zero,
                ::zero, ::one , ::zero,
                ::zero, ::zero, ::one);
iterable_impl!(Matrix3, 3);
iterable_mut_impl!(Matrix3, 3);
dim_impl!(Matrix3, 3);
indexable_impl!(Matrix3, 3);
index_impl!(Matrix3, 3);
at_fast_impl!(Matrix3, 3);
// (specialized); mul_impl!(Matrix3, 3);
// (specialized); rmul_impl!(Matrix3, Vector3, 3);
// (specialized); lmul_impl!(Matrix3, Vector3, 3);
// (specialized); inverse_impl!(Matrix3, 3);
transpose_impl!(Matrix3, 3);
approx_eq_impl!(Matrix3);
// (specialized); row_impl!(Matrix3, Vector3, 3);
// (specialized); column_impl!(Matrix3, Vector3, 3);
column_slice_impl!(Matrix3, Vector3, DVector3, 3);
row_slice_impl!(Matrix3, Vector3, DVector3, 3);
diag_impl!(Matrix3, Vector3, 3);
to_homogeneous_impl!(Matrix3, Matrix4, 3, 4);
from_homogeneous_impl!(Matrix3, Matrix4, 3, 4);
outer_impl!(Vector3, Matrix3);
eigen_qr_impl!(Matrix3, Vector3);
arbitrary_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
rand_impl!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
mean_impl!(Matrix3, Vector3, 3);
mat_display_impl!(Matrix3, 3);

/// Square matrix of dimension 4.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix4<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N
}

eye_impl!(Matrix4, 4, m11, m22, m33, m44);

mat_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
repeat_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
conversion_impl!(Matrix4, 4);
mat_cast_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
add_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
sub_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_add_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_sub_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_mul_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
scalar_div_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
absolute_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
zero_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
one_impl!(Matrix4, ::one  , ::zero, ::zero, ::zero,
                ::zero, ::one  , ::zero, ::zero,
                ::zero, ::zero, ::one  , ::zero,
                ::zero, ::zero, ::zero, ::one);
iterable_impl!(Matrix4, 4);
iterable_mut_impl!(Matrix4, 4);
dim_impl!(Matrix4, 4);
indexable_impl!(Matrix4, 4);
index_impl!(Matrix4, 4);
at_fast_impl!(Matrix4, 4);
mat_mul_mat_impl!(Matrix4, 4);
mat_mul_vec_impl!(Matrix4, Vector4, 4, ::zero);
vec_mul_mat_impl!(Matrix4, Vector4, 4, ::zero);
mat_mul_point_impl!(Matrix4, Point4, 4, Origin::origin);
point_mul_mat_impl!(Matrix4, Point4, 4, Origin::origin);
inverse_impl!(Matrix4, 4);
transpose_impl!(Matrix4, 4);
approx_eq_impl!(Matrix4);
row_impl!(Matrix4, Vector4, 4);
column_impl!(Matrix4, Vector4, 4);
column_slice_impl!(Matrix4, Vector4, DVector4, 4);
row_slice_impl!(Matrix4, Vector4, DVector4, 4);
diag_impl!(Matrix4, Vector4, 4);
to_homogeneous_impl!(Matrix4, Matrix5, 4, 5);
from_homogeneous_impl!(Matrix4, Matrix5, 4, 5);
outer_impl!(Vector4, Matrix4);
eigen_qr_impl!(Matrix4, Vector4);
arbitrary_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
rand_impl!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
mean_impl!(Matrix4, Vector4, 4);
mat_display_impl!(Matrix4, 4);

/// Square matrix of dimension 5.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix5<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N
}

eye_impl!(Matrix5, 5, m11, m22, m33, m44, m55);

mat_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
repeat_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
conversion_impl!(Matrix5, 5);
mat_cast_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
absolute_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
zero_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
one_impl!(Matrix5,
  ::one  , ::zero, ::zero, ::zero, ::zero,
  ::zero, ::one  , ::zero, ::zero, ::zero,
  ::zero, ::zero, ::one  , ::zero, ::zero,
  ::zero, ::zero, ::zero, ::one  , ::zero,
  ::zero, ::zero, ::zero, ::zero, ::one
);
add_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
sub_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_add_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_sub_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_mul_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
scalar_div_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
iterable_impl!(Matrix5, 5);
iterable_mut_impl!(Matrix5, 5);
dim_impl!(Matrix5, 5);
indexable_impl!(Matrix5, 5);
index_impl!(Matrix5, 5);
at_fast_impl!(Matrix5, 5);
mat_mul_mat_impl!(Matrix5, 5);
mat_mul_vec_impl!(Matrix5, Vector5, 5, ::zero);
vec_mul_mat_impl!(Matrix5, Vector5, 5, ::zero);
mat_mul_point_impl!(Matrix5, Point5, 5, Origin::origin);
point_mul_mat_impl!(Matrix5, Point5, 5, Origin::origin);
inverse_impl!(Matrix5, 5);
transpose_impl!(Matrix5, 5);
approx_eq_impl!(Matrix5);
row_impl!(Matrix5, Vector5, 5);
column_impl!(Matrix5, Vector5, 5);
column_slice_impl!(Matrix5, Vector5, DVector5, 5);
row_slice_impl!(Matrix5, Vector5, DVector5, 5);
diag_impl!(Matrix5, Vector5, 5);
to_homogeneous_impl!(Matrix5, Matrix6, 5, 6);
from_homogeneous_impl!(Matrix5, Matrix6, 5, 6);
outer_impl!(Vector5, Matrix5);
eigen_qr_impl!(Matrix5, Vector5);
arbitrary_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
rand_impl!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
mean_impl!(Matrix5, Vector5, 5);
mat_display_impl!(Matrix5, 5);

/// Square matrix of dimension 6.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix6<N> {
    pub m11: N, pub m21: N, pub m31: N, pub m41: N, pub m51: N, pub m61: N,
    pub m12: N, pub m22: N, pub m32: N, pub m42: N, pub m52: N, pub m62: N,
    pub m13: N, pub m23: N, pub m33: N, pub m43: N, pub m53: N, pub m63: N,
    pub m14: N, pub m24: N, pub m34: N, pub m44: N, pub m54: N, pub m64: N,
    pub m15: N, pub m25: N, pub m35: N, pub m45: N, pub m55: N, pub m65: N,
    pub m16: N, pub m26: N, pub m36: N, pub m46: N, pub m56: N, pub m66: N
}

eye_impl!(Matrix6, 6, m11, m22, m33, m44, m55, m66);

mat_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
repeat_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
conversion_impl!(Matrix6, 6);
mat_cast_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
add_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
sub_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_add_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_sub_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_mul_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
scalar_div_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
absolute_impl!(Matrix6, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66);

zero_impl!(Matrix6, m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36, m41, m42, m43, m44, m45, m46, m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66);

one_impl!(Matrix6,
  ::one  , ::zero, ::zero, ::zero, ::zero, ::zero,
  ::zero, ::one  , ::zero, ::zero, ::zero, ::zero,
  ::zero, ::zero, ::one  , ::zero, ::zero, ::zero,
  ::zero, ::zero, ::zero, ::one  , ::zero, ::zero,
  ::zero, ::zero, ::zero, ::zero, ::one  , ::zero,
  ::zero, ::zero, ::zero, ::zero, ::zero, ::one
);
iterable_impl!(Matrix6, 6);
iterable_mut_impl!(Matrix6, 6);
dim_impl!(Matrix6, 6);
indexable_impl!(Matrix6, 6);
index_impl!(Matrix6, 6);
at_fast_impl!(Matrix6, 6);
mat_mul_mat_impl!(Matrix6, 6);
mat_mul_vec_impl!(Matrix6, Vector6, 6, ::zero);
vec_mul_mat_impl!(Matrix6, Vector6, 6, ::zero);
mat_mul_point_impl!(Matrix6, Point6, 6, Origin::origin);
point_mul_mat_impl!(Matrix6, Point6, 6, Origin::origin);
inverse_impl!(Matrix6, 6);
transpose_impl!(Matrix6, 6);
approx_eq_impl!(Matrix6);
row_impl!(Matrix6, Vector6, 6);
column_impl!(Matrix6, Vector6, 6);
column_slice_impl!(Matrix6, Vector6, DVector6, 6);
row_slice_impl!(Matrix6, Vector6, DVector6, 6);
diag_impl!(Matrix6, Vector6, 6);
outer_impl!(Vector6, Matrix6);
eigen_qr_impl!(Matrix6, Vector6);
arbitrary_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
rand_impl!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
mean_impl!(Matrix6, Vector6, 6);
mat_display_impl!(Matrix6, 6);
