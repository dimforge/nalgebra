use na::{MatrixMN, VectorN, Vector1, Vector2, Vector3, Vector4,
         Matrix2, Matrix3, Matrix4,
         Matrix2x3, Matrix3x2, Matrix4x2,
         Matrix2x4, Matrix3x4, Matrix4x3,
         Quaternion};

/// A matrix with components of type `N`. It has `R` rows, and `C` columns.
///
/// # See also:
///
/// * [`TMat2`](type.TMat2.html)
/// * [`TMat2x2`](type.TMat2x2.html)
/// * [`TMat2x3`](type.TMat2x3.html)
/// * [`TMat2x4`](type.TMat2x4.html)
/// * [`TMat3`](type.TMat3.html)
/// * [`TMat3x2`](type.TMat3x2.html)
/// * [`TMat3x3`](type.TMat3x3.html)
/// * [`TMat3x4`](type.TMat3x4.html)
/// * [`TMat4`](type.TMat4.html)
/// * [`TMat4x2`](type.TMat4x2.html)
/// * [`TMat4x3`](type.TMat4x3.html)
/// * [`TMat4x4`](type.TMat4x4.html)
pub type TMat<N, R, C> = MatrixMN<N, R, C>;
/// A column vector with components of type `N`. It has `D` rows (and one column).
///
/// # See also:
///
/// * [`TVec1`](type.TVec1.html)
/// * [`TVec2`](type.TVec2.html)
/// * [`TVec3`](type.TVec3.html)
/// * [`TVec4`](type.TVec4.html)
pub type TVec<N, R> = VectorN<N, R>;
/// A quaternion with components of type `N`.
pub type Qua<N> = Quaternion<N>;

/// A 1D vector with components of type `N`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec1`](fn.make_vec1.html)
/// * [`vec1`](fn.vec1.html)
/// * [`vec2_to_vec1`](fn.vec2_to_vec1.html)
/// * [`vec3_to_vec1`](fn.vec3_to_vec1.html)
/// * [`vec4_to_vec1`](fn.vec4_to_vec1.html)
///
/// ## Related types:
///
/// * [`BVec1`](type.BVec1.html)
/// * [`DVec1`](type.DVec1.html)
/// * [`IVec1`](type.IVec1.html)
/// * [`I16Vec1`](type.I16Vec1.html)
/// * [`I32Vec1`](type.I32Vec1.html)
/// * [`I64Vec1`](type.I64Vec1.html)
/// * [`I8Vec1`](type.I8Vec1.html)
/// * [`TVec`](type.TVec.html)
/// * [`UVec1`](type.UVec1.html)
/// * [`U16Vec1`](type.U16Vec1.html)
/// * [`U32Vec1`](type.U32Vec1.html)
/// * [`U64Vec1`](type.U64Vec1.html)
/// * [`U8Vec1`](type.U8Vec1.html)
/// * [`Vec1`](type.Vec1.html)
pub type TVec1<N> = Vector1<N>;
/// A 2D vector with components of type `N`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec2`](fn.make_vec2.html)
/// * [`vec2`](fn.vec2.html)
/// * [`vec1_to_vec2`](fn.vec1_to_vec2.html)
/// * [`vec3_to_vec2`](fn.vec3_to_vec2.html)
/// * [`vec4_to_vec2`](fn.vec4_to_vec2.html)
///
/// ## Related types:
///
/// * [`vec2`](fn.vec2.html)
/// * [`BVec2`](type.BVec2.html)
/// * [`DVec2`](type.DVec2.html)
/// * [`IVec2`](type.IVec2.html)
/// * [`I16Vec2`](type.I16Vec2.html)
/// * [`I32Vec2`](type.I32Vec2.html)
/// * [`I64Vec2`](type.I64Vec2.html)
/// * [`I8Vec2`](type.I8Vec2.html)
/// * [`TVec`](type.TVec.html)
/// * [`UVec2`](type.UVec2.html)
/// * [`U16Vec2`](type.U16Vec2.html)
/// * [`U32Vec2`](type.U32Vec2.html)
/// * [`U64Vec2`](type.U64Vec2.html)
/// * [`U8Vec2`](type.U8Vec2.html)
/// * [`Vec2`](type.Vec2.html)
pub type TVec2<N> = Vector2<N>;
/// A 3D vector with components of type `N`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec3`](fn.make_vec3.html)
/// * [`vec3`](fn.vec3.html)
/// * [`vec1_to_vec3`](fn.vec1_to_vec3.html)
/// * [`vec2_to_vec3`](fn.vec2_to_vec3.html)
/// * [`vec4_to_vec3`](fn.vec4_to_vec3.html)
///
/// ## Related types:
///
/// * [`vec3`](fn.vec3.html)
/// * [`BVec3`](type.BVec3.html)
/// * [`DVec3`](type.DVec3.html)
/// * [`IVec3`](type.IVec3.html)
/// * [`I16Vec3`](type.I16Vec3.html)
/// * [`I32Vec3`](type.I32Vec3.html)
/// * [`I64Vec3`](type.I64Vec3.html)
/// * [`I8Vec3`](type.I8Vec3.html)
/// * [`TVec`](type.TVec.html)
/// * [`UVec3`](type.UVec3.html)
/// * [`U16Vec3`](type.U16Vec3.html)
/// * [`U32Vec3`](type.U32Vec3.html)
/// * [`U64Vec3`](type.U64Vec3.html)
/// * [`U8Vec3`](type.U8Vec3.html)
/// * [`Vec3`](type.Vec3.html)
pub type TVec3<N> = Vector3<N>;
/// A 4D vector with components of type `N`.
///
/// # See also:
///
/// ## Constructors:
///
/// * [`make_vec4`](fn.make_vec4.html)
/// * [`vec4`](fn.vec4.html)
/// * [`vec1_to_vec4`](fn.vec1_to_vec4.html)
/// * [`vec2_to_vec4`](fn.vec2_to_vec4.html)
/// * [`vec3_to_vec4`](fn.vec3_to_vec4.html)
///
/// ## Related types:
///
/// * [`vec4`](fn.vec4.html)
/// * [`BVec4`](type.BVec4.html)
/// * [`DVec4`](type.DVec4.html)
/// * [`IVec4`](type.IVec4.html)
/// * [`I16Vec4`](type.I16Vec4.html)
/// * [`I32Vec4`](type.I32Vec4.html)
/// * [`I64Vec4`](type.I64Vec4.html)
/// * [`I8Vec4`](type.I8Vec4.html)
/// * [`UVec4`](type.UVec4.html)
/// * [`U16Vec4`](type.U16Vec4.html)
/// * [`U32Vec4`](type.U32Vec4.html)
/// * [`U64Vec4`](type.U64Vec4.html)
/// * [`U8Vec4`](type.U8Vec4.html)
/// * [`Vec4`](type.Vec4.html)
pub type TVec4<N> = Vector4<N>;
/// A 1D vector with boolean components.
pub type BVec1 = Vector1<bool>;
/// A 2D vector with boolean components.
pub type BVec2 = Vector2<bool>;
/// A 3D vector with boolean components.
pub type BVec3 = Vector3<bool>;
/// A 4D vector with boolean components.
pub type BVec4 = Vector4<bool>;
/// A 1D vector with `f64` components.
pub type DVec1 = Vector1<f64>;
/// A 2D vector with `f64` components.
pub type DVec2 = Vector2<f64>;
/// A 3D vector with `f64` components.
pub type DVec3 = Vector3<f64>;
/// A 4D vector with `f64` components.
pub type DVec4 = Vector4<f64>;
/// A 1D vector with `i32` components.
pub type IVec1 = Vector1<i32>;
/// A 2D vector with `i32` components.
pub type IVec2 = Vector2<i32>;
/// A 3D vector with `i32` components.
pub type IVec3 = Vector3<i32>;
/// A 4D vector with `i32` components.
pub type IVec4 = Vector4<i32>;
/// A 1D vector with `u32` components.
pub type UVec1 = Vector1<u32>;
/// A 2D vector with `u32` components.
pub type UVec2 = Vector2<u32>;
/// A 3D vector with `u32` components.
pub type UVec3 = Vector3<u32>;
/// A 4D vector with `u32` components.
pub type UVec4 = Vector4<u32>;
/// A 1D vector with `f32` components.
pub type Vec1  = Vector1<f32>;
/// A 2D vector with `f32` components.
pub type Vec2  = Vector2<f32>;
/// A 3D vector with `f32` components.
pub type Vec3  = Vector3<f32>;
/// A 4D vector with `f32` components.
pub type Vec4  = Vector4<f32>;

/// A 1D vector with `u64` components.
pub type U64Vec1 = Vector1<u64>;
/// A 2D vector with `u64` components.
pub type U64Vec2 = Vector2<u64>;
/// A 3D vector with `u64` components.
pub type U64Vec3 = Vector3<u64>;
/// A 4D vector with `u64` components.
pub type U64Vec4 = Vector4<u64>;
/// A 1D vector with `i64` components.
pub type I64Vec1 = Vector1<i64>;
/// A 2D vector with `i64` components.
pub type I64Vec2 = Vector2<i64>;
/// A 3D vector with `i64` components.
pub type I64Vec3 = Vector3<i64>;
/// A 4D vector with `i64` components.
pub type I64Vec4 = Vector4<i64>;

/// A 1D vector with `u32` components.
pub type U32Vec1 = Vector1<u32>;
/// A 2D vector with `u32` components.
pub type U32Vec2 = Vector2<u32>;
/// A 3D vector with `u32` components.
pub type U32Vec3 = Vector3<u32>;
/// A 4D vector with `u32` components.
pub type U32Vec4 = Vector4<u32>;
/// A 1D vector with `i32` components.
pub type I32Vec1 = Vector1<i32>;
/// A 2D vector with `i32` components.
pub type I32Vec2 = Vector2<i32>;
/// A 3D vector with `i32` components.
pub type I32Vec3 = Vector3<i32>;
/// A 4D vector with `i32` components.
pub type I32Vec4 = Vector4<i32>;

/// A 1D vector with `u16` components.
pub type U16Vec1 = Vector1<u16>;
/// A 2D vector with `u16` components.
pub type U16Vec2 = Vector2<u16>;
/// A 3D vector with `u16` components.
pub type U16Vec3 = Vector3<u16>;
/// A 4D vector with `u16` components.
pub type U16Vec4 = Vector4<u16>;
/// A 1D vector with `i16` components.
pub type I16Vec1 = Vector1<i16>;
/// A 2D vector with `i16` components.
pub type I16Vec2 = Vector2<i16>;
/// A 3D vector with `i16` components.
pub type I16Vec3 = Vector3<i16>;
/// A 4D vector with `i16` components.
pub type I16Vec4 = Vector4<i16>;

/// A 1D vector with `u8` components.
pub type U8Vec1 = Vector1<u8>;
/// A 2D vector with `u8` components.
pub type U8Vec2 = Vector2<u8>;
/// A 3D vector with `u8` components.
pub type U8Vec3 = Vector3<u8>;
/// A 4D vector with `u8` components.
pub type U8Vec4 = Vector4<u8>;
/// A 1D vector with `i8` components.
pub type I8Vec1 = Vector1<i8>;
/// A 2D vector with `i8` components.
pub type I8Vec2 = Vector2<i8>;
/// A 3D vector with `i8` components.
pub type I8Vec3 = Vector3<i8>;
/// A 4D vector with `i8` components.
pub type I8Vec4 = Vector4<i8>;


/// A 2x2 matrix with components of type `N`.
pub type TMat2<N> = Matrix2<N>;
/// A 2x2 matrix with components of type `N`.
pub type TMat2x2<N> = Matrix2<N>;
/// A 2x3 matrix with components of type `N`.
pub type TMat2x3<N> = Matrix2x3<N>;
/// A 2x4 matrix with components of type `N`.
pub type TMat2x4<N> = Matrix2x4<N>;
/// A 3x3 matrix with components of type `N`.
pub type TMat3<N> = Matrix3<N>;
/// A 3x2 matrix with components of type `N`.
pub type TMat3x2<N> = Matrix3x2<N>;
/// A 3x3 matrix with components of type `N`.
pub type TMat3x3<N> = Matrix3<N>;
/// A 3x4 matrix with components of type `N`.
pub type TMat3x4<N> = Matrix3x4<N>;
/// A 4x4 matrix with components of type `N`.
pub type TMat4<N> = Matrix4<N>;
/// A 4x2 matrix with components of type `N`.
pub type TMat4x2<N> = Matrix4x2<N>;
/// A 4x3 matrix with components of type `N`.
pub type TMat4x3<N> = Matrix4x3<N>;
/// A 4x4 matrix with components of type `N`.
pub type TMat4x4<N> = Matrix4<N>;
/// A 2x2 matrix with components of type `N`.
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
/// A 2x2 matrix with `f32` components.
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
pub type Quat = Quaternion<f32>;
/// A quaternion with f64 components.
pub type DQuat = Quaternion<f64>;
