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
                        RowSlice, Diagonal, DiagonalMut, Shape, BaseFloat, BaseNum, Repeat};
use traits::operations::{Absolute, Transpose, Inverse, Outer, EigenQR, Mean};
use traits::geometry::{ToHomogeneous, FromHomogeneous, Origin};
use linalg;

#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

#[cfg(feature="abstract_algebra")]
use_matrix_group_modules!();


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

matrix_impl!(Matrix1, 1, Vector1, DVector1, m11);
one_impl!(Matrix1, ::one);
dim_impl!(Matrix1, 1);
mat_mul_mat_impl!(Matrix1, 1);
mat_mul_vec_impl!(Matrix1, Vector1, 1, ::zero);
vec_mul_mat_impl!(Matrix1, Vector1, 1, ::zero);
mat_mul_point_impl!(Matrix1, Point1, 1, Origin::origin);
point_mul_mat_impl!(Matrix1, Point1, 1, Origin::origin);
// (specialized); inverse_impl!(Matrix1, 1);
to_homogeneous_impl!(Matrix1, Matrix2, 1, 2);
from_homogeneous_impl!(Matrix1, Matrix2, 1, 2);
eigen_qr_impl!(Matrix1, Vector1);
componentwise_arbitrary!(Matrix1, m11);
componentwise_rand!(Matrix1, m11);
mat_display_impl!(Matrix1, 1);

/// Square matrix of dimension 2.
#[repr(C)]
#[derive(Eq, PartialEq, RustcEncodable, RustcDecodable, Clone, Hash, Debug, Copy)]
pub struct Matrix2<N> {
    pub m11: N, pub m21: N,
    pub m12: N, pub m22: N
}

eye_impl!(Matrix2, 2, m11, m22);

matrix_impl!(Matrix2, 2, Vector2, DVector2, m11, m12,
                                  m21, m22);
one_impl!(Matrix2, ::one,  ::zero,
                ::zero, ::one);
dim_impl!(Matrix2, 2);
// (specialized); mul_impl!(Matrix2, 2);
// (specialized); rmul_impl!(Matrix2, Vector2, 2);
// (specialized); lmul_impl!(Matrix2, Vector2, 2);
// (specialized); inverse_impl!(Matrix2, 2);
to_homogeneous_impl!(Matrix2, Matrix3, 2, 3);
from_homogeneous_impl!(Matrix2, Matrix3, 2, 3);
eigen_qr_impl!(Matrix2, Vector2);
componentwise_arbitrary!(Matrix2, m11, m12, m21, m22);
componentwise_rand!(Matrix2, m11, m12, m21, m22);
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

matrix_impl!(Matrix3, 3, Vector3, DVector3, m11, m12, m13,
                                            m21, m22, m23,
                                            m31, m32, m33);
one_impl!(Matrix3, ::one , ::zero, ::zero,
                ::zero, ::one , ::zero,
                ::zero, ::zero, ::one);
dim_impl!(Matrix3, 3);
// (specialized); mul_impl!(Matrix3, 3);
// (specialized); rmul_impl!(Matrix3, Vector3, 3);
// (specialized); lmul_impl!(Matrix3, Vector3, 3);
// (specialized); inverse_impl!(Matrix3, 3);
to_homogeneous_impl!(Matrix3, Matrix4, 3, 4);
from_homogeneous_impl!(Matrix3, Matrix4, 3, 4);
eigen_qr_impl!(Matrix3, Vector3);
componentwise_arbitrary!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
componentwise_rand!(Matrix3,
    m11, m12, m13,
    m21, m22, m23,
    m31, m32, m33
);
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

matrix_impl!(Matrix4, 4, Vector4, DVector4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
one_impl!(Matrix4, ::one  , ::zero, ::zero, ::zero,
                ::zero, ::one  , ::zero, ::zero,
                ::zero, ::zero, ::one  , ::zero,
                ::zero, ::zero, ::zero, ::one);
dim_impl!(Matrix4, 4);
mat_mul_mat_impl!(Matrix4, 4);
mat_mul_vec_impl!(Matrix4, Vector4, 4, ::zero);
vec_mul_mat_impl!(Matrix4, Vector4, 4, ::zero);
mat_mul_point_impl!(Matrix4, Point4, 4, Origin::origin);
point_mul_mat_impl!(Matrix4, Point4, 4, Origin::origin);
inverse_impl!(Matrix4, 4);
to_homogeneous_impl!(Matrix4, Matrix5, 4, 5);
from_homogeneous_impl!(Matrix4, Matrix5, 4, 5);
eigen_qr_impl!(Matrix4, Vector4);
componentwise_arbitrary!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
componentwise_rand!(Matrix4,
  m11, m12, m13, m14,
  m21, m22, m23, m24,
  m31, m32, m33, m34,
  m41, m42, m43, m44
);
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

matrix_impl!(Matrix5, 5, Vector5, DVector5,
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
dim_impl!(Matrix5, 5);
mat_mul_mat_impl!(Matrix5, 5);
mat_mul_vec_impl!(Matrix5, Vector5, 5, ::zero);
vec_mul_mat_impl!(Matrix5, Vector5, 5, ::zero);
mat_mul_point_impl!(Matrix5, Point5, 5, Origin::origin);
point_mul_mat_impl!(Matrix5, Point5, 5, Origin::origin);
inverse_impl!(Matrix5, 5);
to_homogeneous_impl!(Matrix5, Matrix6, 5, 6);
from_homogeneous_impl!(Matrix5, Matrix6, 5, 6);
eigen_qr_impl!(Matrix5, Vector5);
componentwise_arbitrary!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
componentwise_rand!(Matrix5,
  m11, m12, m13, m14, m15,
  m21, m22, m23, m24, m25,
  m31, m32, m33, m34, m35,
  m41, m42, m43, m44, m45,
  m51, m52, m53, m54, m55
);
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

matrix_impl!(Matrix6, 6, Vector6, DVector6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);

one_impl!(Matrix6,
  ::one  , ::zero, ::zero, ::zero, ::zero, ::zero,
  ::zero, ::one  , ::zero, ::zero, ::zero, ::zero,
  ::zero, ::zero, ::one  , ::zero, ::zero, ::zero,
  ::zero, ::zero, ::zero, ::one  , ::zero, ::zero,
  ::zero, ::zero, ::zero, ::zero, ::one  , ::zero,
  ::zero, ::zero, ::zero, ::zero, ::zero, ::one
);
dim_impl!(Matrix6, 6);
mat_mul_mat_impl!(Matrix6, 6);
mat_mul_vec_impl!(Matrix6, Vector6, 6, ::zero);
vec_mul_mat_impl!(Matrix6, Vector6, 6, ::zero);
mat_mul_point_impl!(Matrix6, Point6, 6, Origin::origin);
point_mul_mat_impl!(Matrix6, Point6, 6, Origin::origin);
inverse_impl!(Matrix6, 6);
eigen_qr_impl!(Matrix6, Vector6);
componentwise_arbitrary!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
componentwise_rand!(Matrix6,
  m11, m12, m13, m14, m15, m16,
  m21, m22, m23, m24, m25, m26,
  m31, m32, m33, m34, m35, m36,
  m41, m42, m43, m44, m45, m46,
  m51, m52, m53, m54, m55, m56,
  m61, m62, m63, m64, m65, m66
);
mat_display_impl!(Matrix6, 6);
