use na::{
    Matrix2, Matrix2x3, Matrix2x4, Matrix3, Matrix3x2, Matrix3x4, Matrix4, Matrix4x2, Matrix4x3,
    Quaternion, SMatrix, SVector,
};

/// A matrix with components of type `T`. It has `R` rows, and `C` columns.
///
/// In this library, vectors, represented as [`TVec`] and
/// friends, are also matrices. Operations that operate on a matrix will
/// also work on a vector.
///
/// # See also:
///
/// * [`TMat2`]
/// * [`TMat2x2`]
/// * [`TMat2x3`]
/// * [`TMat2x4`]
/// * [`TMat3`]
/// * [`TMat3x2`]
/// * [`TMat3x3`]
/// * [`TMat3x4`]
/// * [`TMat4`]
/// * [`TMat4x2`]
/// * [`TMat4x3`]
/// * [`TMat4x4`]
/// * [`TVec`]
pub type TMat<T, const R: usize, const C: usize> = SMatrix<T, R, C>;
/// A column vector with components of type `T`. It has `D` rows (and one column).
///
/// In this library, vectors are represented as a single column matrix, so
/// operations on [`TMat`] are also valid on vectors.
///
/// # See also:
///
/// * [`TMat`]
/// * [`TVec1`]
/// * [`TVec2`]
/// * [`TVec3`]
/// * [`TVec4`]
pub type TVec<T, const R: usize> = SVector<T, R>;
/// A quaternion with components of type `T`.
pub type Qua<T> = Quaternion<T>;

/// A 1D vector with components of type `T`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec1()`](crate::make_vec1)
/// * [`vec1()`](crate::vec1)
/// * [`vec2_to_vec1()`](crate::vec2_to_vec1)
/// * [`vec3_to_vec1()`](crate::vec3_to_vec1)
/// * [`vec4_to_vec1()`](crate::vec4_to_vec1)
///
/// ## Related types:
///
/// * [`BVec1`]
/// * [`DVec1`]
/// * [`IVec1`]
/// * [`I16Vec1`]
/// * [`I32Vec1`]
/// * [`I64Vec1`]
/// * [`I8Vec1`]
/// * [`TVec`]
/// * [`UVec1`]
/// * [`U16Vec1`]
/// * [`U32Vec1`]
/// * [`U64Vec1`]
/// * [`U8Vec1`]
/// * [`Vec1`]
pub type TVec1<T> = TVec<T, 1>;
/// A 2D vector with components of type `T`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec2()`](crate::make_vec2)
/// * [`vec2()`](crate::vec2)
/// * [`vec1_to_vec2()`](crate::vec1_to_vec2)
/// * [`vec3_to_vec2()`](crate::vec3_to_vec2)
/// * [`vec4_to_vec2()`](crate::vec4_to_vec2)
///
/// ## Related types:
///
/// * [`BVec2`]
/// * [`DVec2`]
/// * [`IVec2`]
/// * [`I16Vec2`]
/// * [`I32Vec2`]
/// * [`I64Vec2`]
/// * [`I8Vec2`]
/// * [`TVec`]
/// * [`UVec2`]
/// * [`U16Vec2`]
/// * [`U32Vec2`]
/// * [`U64Vec2`]
/// * [`U8Vec2`]
/// * [`Vec2`]
pub type TVec2<T> = TVec<T, 2>;
/// A 3D vector with components of type `T`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec3()`](crate::make_vec3)
/// * [`vec3()`](crate::vec3)
/// * [`vec1_to_vec3()`](crate::vec1_to_vec3)
/// * [`vec2_to_vec3()`](crate::vec2_to_vec3)
/// * [`vec4_to_vec3()`](crate::vec4_to_vec3)
///
/// ## Related types:
///
/// * [`BVec3`]
/// * [`DVec3`]
/// * [`IVec3`]
/// * [`I16Vec3`]
/// * [`I32Vec3`]
/// * [`I64Vec3`]
/// * [`I8Vec3`]
/// * [`TVec`]
/// * [`UVec3`]
/// * [`U16Vec3`]
/// * [`U32Vec3`]
/// * [`U64Vec3`]
/// * [`U8Vec3`]
/// * [`Vec3`]
pub type TVec3<T> = TVec<T, 3>;
/// A 4D vector with components of type `T`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec4()`](crate::make_vec4)
/// * [`vec4()`](crate::vec4)
/// * [`vec1_to_vec4()`](crate::vec1_to_vec4)
/// * [`vec2_to_vec4()`](crate::vec2_to_vec4)
/// * [`vec3_to_vec4()`](crate::vec3_to_vec4)
///
/// ## Related types:
///
/// * [`BVec4`]
/// * [`DVec4`]
/// * [`IVec4`]
/// * [`I16Vec4`]
/// * [`I32Vec4`]
/// * [`I64Vec4`]
/// * [`I8Vec4`]
/// * [`UVec4`]
/// * [`U16Vec4`]
/// * [`U32Vec4`]
/// * [`U64Vec4`]
/// * [`U8Vec4`]
/// * [`Vec4`]
pub type TVec4<T> = TVec<T, 4>;
/// A 1D vector with boolean components.
pub type BVec1 = TVec1<bool>;
/// A 2D vector with boolean components.
pub type BVec2 = TVec2<bool>;
/// A 3D vector with boolean components.
pub type BVec3 = TVec3<bool>;
/// A 4D vector with boolean components.
pub type BVec4 = TVec4<bool>;
/// A 1D vector with `f64` components.
pub type DVec1 = TVec1<f64>;
/// A 2D vector with `f64` components.
pub type DVec2 = TVec2<f64>;
/// A 3D vector with `f64` components.
pub type DVec3 = TVec3<f64>;
/// A 4D vector with `f64` components.
pub type DVec4 = TVec4<f64>;
/// A 1D vector with `i32` components.
pub type IVec1 = TVec1<i32>;
/// A 2D vector with `i32` components.
pub type IVec2 = TVec2<i32>;
/// A 3D vector with `i32` components.
pub type IVec3 = TVec3<i32>;
/// A 4D vector with `i32` components.
pub type IVec4 = TVec4<i32>;
/// A 1D vector with `u32` components.
pub type UVec1 = TVec1<u32>;
/// A 2D vector with `u32` components.
pub type UVec2 = TVec2<u32>;
/// A 3D vector with `u32` components.
pub type UVec3 = TVec3<u32>;
/// A 4D vector with `u32` components.
pub type UVec4 = TVec4<u32>;
/// A 1D vector with `f32` components.
pub type Vec1 = TVec1<f32>;
/// A 2D vector with `f32` components.
pub type Vec2 = TVec2<f32>;
/// A 3D vector with `f32` components.
pub type Vec3 = TVec3<f32>;
/// A 4D vector with `f32` components.
pub type Vec4 = TVec4<f32>;

/// A 1D vector with `u64` components.
pub type U64Vec1 = TVec1<u64>;
/// A 2D vector with `u64` components.
pub type U64Vec2 = TVec2<u64>;
/// A 3D vector with `u64` components.
pub type U64Vec3 = TVec3<u64>;
/// A 4D vector with `u64` components.
pub type U64Vec4 = TVec4<u64>;
/// A 1D vector with `i64` components.
pub type I64Vec1 = TVec1<i64>;
/// A 2D vector with `i64` components.
pub type I64Vec2 = TVec2<i64>;
/// A 3D vector with `i64` components.
pub type I64Vec3 = TVec3<i64>;
/// A 4D vector with `i64` components.
pub type I64Vec4 = TVec4<i64>;

/// A 1D vector with `u32` components.
pub type U32Vec1 = TVec1<u32>;
/// A 2D vector with `u32` components.
pub type U32Vec2 = TVec2<u32>;
/// A 3D vector with `u32` components.
pub type U32Vec3 = TVec3<u32>;
/// A 4D vector with `u32` components.
pub type U32Vec4 = TVec4<u32>;
/// A 1D vector with `i32` components.
pub type I32Vec1 = TVec1<i32>;
/// A 2D vector with `i32` components.
pub type I32Vec2 = TVec2<i32>;
/// A 3D vector with `i32` components.
pub type I32Vec3 = TVec3<i32>;
/// A 4D vector with `i32` components.
pub type I32Vec4 = TVec4<i32>;

/// A 1D vector with `u16` components.
pub type U16Vec1 = TVec1<u16>;
/// A 2D vector with `u16` components.
pub type U16Vec2 = TVec2<u16>;
/// A 3D vector with `u16` components.
pub type U16Vec3 = TVec3<u16>;
/// A 4D vector with `u16` components.
pub type U16Vec4 = TVec4<u16>;
/// A 1D vector with `i16` components.
pub type I16Vec1 = TVec1<i16>;
/// A 2D vector with `i16` components.
pub type I16Vec2 = TVec2<i16>;
/// A 3D vector with `i16` components.
pub type I16Vec3 = TVec3<i16>;
/// A 4D vector with `i16` components.
pub type I16Vec4 = TVec4<i16>;

/// A 1D vector with `u8` components.
pub type U8Vec1 = TVec1<u8>;
/// A 2D vector with `u8` components.
pub type U8Vec2 = TVec2<u8>;
/// A 3D vector with `u8` components.
pub type U8Vec3 = TVec3<u8>;
/// A 4D vector with `u8` components.
pub type U8Vec4 = TVec4<u8>;
/// A 1D vector with `i8` components.
pub type I8Vec1 = TVec1<i8>;
/// A 2D vector with `i8` components.
pub type I8Vec2 = TVec2<i8>;
/// A 3D vector with `i8` components.
pub type I8Vec3 = TVec3<i8>;
/// A 4D vector with `i8` components.
pub type I8Vec4 = TVec4<i8>;

/// A 2x2 matrix with components of type `T`.
pub type TMat2<T> = Matrix2<T>;
/// A 2x2 matrix with components of type `T`.
pub type TMat2x2<T> = Matrix2<T>;
/// A 2x3 matrix with components of type `T`.
pub type TMat2x3<T> = Matrix2x3<T>;
/// A 2x4 matrix with components of type `T`.
pub type TMat2x4<T> = Matrix2x4<T>;
/// A 3x3 matrix with components of type `T`.
pub type TMat3<T> = Matrix3<T>;
/// A 3x2 matrix with components of type `T`.
pub type TMat3x2<T> = Matrix3x2<T>;
/// A 3x3 matrix with components of type `T`.
pub type TMat3x3<T> = Matrix3<T>;
/// A 3x4 matrix with components of type `T`.
pub type TMat3x4<T> = Matrix3x4<T>;
/// A 4x4 matrix with components of type `T`.
pub type TMat4<T> = Matrix4<T>;
/// A 4x2 matrix with components of type `T`.
pub type TMat4x2<T> = Matrix4x2<T>;
/// A 4x3 matrix with components of type `T`.
pub type TMat4x3<T> = Matrix4x3<T>;
/// A 4x4 matrix with components of type `T`.
pub type TMat4x4<T> = Matrix4<T>;
/// A 2x2 matrix with components of type `T`.
pub type DMat2 = Matrix2<f64>;
/// A 2x2 matrix with `f64` components.
pub type DMat2x2 = Matrix2<f64>;
/// A 2x3 matrix with `f64` components.
pub type DMat2x3 = Matrix2x3<f64>;
/// A 2x4 matrix with `f64` components.
pub type DMat2x4 = Matrix2x4<f64>;
/// A 3x3 matrix with `f64` components.
pub type DMat3 = Matrix3<f64>;
/// A 3x2 matrix with `f64` components.
pub type DMat3x2 = Matrix3x2<f64>;
/// A 3x3 matrix with `f64` components.
pub type DMat3x3 = Matrix3<f64>;
/// A 3x4 matrix with `f64` components.
pub type DMat3x4 = Matrix3x4<f64>;
/// A 4x4 matrix with `f64` components.
pub type DMat4 = Matrix4<f64>;
/// A 4x2 matrix with `f64` components.
pub type DMat4x2 = Matrix4x2<f64>;
/// A 4x3 matrix with `f64` components.
pub type DMat4x3 = Matrix4x3<f64>;
/// A 4x4 matrix with `f64` components.
pub type DMat4x4 = Matrix4<f64>;
/// A 2x2 matrix with `f32` components.
pub type Mat2 = Matrix2<f32>;
/// A 2x2 matrix with `f32` components.
pub type Mat2x2 = Matrix2<f32>;
/// A 2x3 matrix with `f32` components.
pub type Mat2x3 = Matrix2x3<f32>;
/// A 2x4 matrix with `f32` components.
pub type Mat2x4 = Matrix2x4<f32>;
/// A 3x3 matrix with `f32` components.
pub type Mat3 = Matrix3<f32>;
/// A 3x2 matrix with `f32` components.
pub type Mat3x2 = Matrix3x2<f32>;
/// A 3x3 matrix with `f32` components.
pub type Mat3x3 = Matrix3<f32>;
/// A 3x4 matrix with `f32` components.
pub type Mat3x4 = Matrix3x4<f32>;
/// A 4x2 matrix with `f32` components.
pub type Mat4x2 = Matrix4x2<f32>;
/// A 4x3 matrix with `f32` components.
pub type Mat4x3 = Matrix4x3<f32>;
/// A 4x4 matrix with `f32` components.
pub type Mat4x4 = Matrix4<f32>;
/// A 4x4 matrix with `f32` components.
pub type Mat4 = Matrix4<f32>;

/// A quaternion with f32 components.
pub type Quat = Qua<f32>;
/// A quaternion with f64 components.
pub type DQuat = Qua<f64>;
