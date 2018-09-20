use na::{MatrixMN, VectorN, Vector1, Vector2, Vector3, Vector4,
         Matrix2, Matrix3, Matrix4,
         Matrix2x3, Matrix3x2, Matrix4x2,
         Matrix2x4, Matrix3x4, Matrix4x3,
         Quaternion};

pub type Mat<N, R, C> = MatrixMN<N, R, C>;
pub type Vec<N, R> = VectorN<N, R>;
pub type Qua<N> = Quaternion<N>;

pub type BVec1 = Vector1<bool>;
pub type BVec2 = Vector2<bool>;
pub type BVec3 = Vector3<bool>;
pub type BVec4 = Vector4<bool>;
pub type DVec1 = Vector1<f64>;
pub type DVec2 = Vector2<f64>;
pub type DVec3 = Vector3<f64>;
pub type DVec4 = Vector4<f64>;
pub type IVec1 = Vector1<i32>;
pub type IVec2 = Vector2<i32>;
pub type IVec3 = Vector3<i32>;
pub type IVec4 = Vector4<i32>;
pub type UVec1 = Vector1<u32>;
pub type UVec2 = Vector2<u32>;
pub type UVec3 = Vector3<u32>;
pub type UVec4 = Vector4<u32>;
pub type Vec1  = Vector1<f32>;
pub type Vec2  = Vector2<f32>;
pub type Vec3  = Vector3<f32>;
pub type Vec4  = Vector4<f32>;

pub type U64Vec1 = Vector1<u64>;
pub type U64Vec2 = Vector2<u64>;
pub type U64Vec3 = Vector3<u64>;
pub type U64Vec4 = Vector4<u64>;
pub type I64Vec1 = Vector1<i64>;
pub type I64Vec2 = Vector2<i64>;
pub type I64Vec3 = Vector3<i64>;
pub type I64Vec4 = Vector4<i64>;

pub type U32Vec1 = Vector1<u32>;
pub type U32Vec2 = Vector2<u32>;
pub type U32Vec3 = Vector3<u32>;
pub type U32Vec4 = Vector4<u32>;
pub type I32Vec1 = Vector1<i32>;
pub type I32Vec2 = Vector2<i32>;
pub type I32Vec3 = Vector3<i32>;
pub type I32Vec4 = Vector4<i32>;

pub type U16Vec1 = Vector1<u16>;
pub type U16Vec2 = Vector2<u16>;
pub type U16Vec3 = Vector3<u16>;
pub type U16Vec4 = Vector4<u16>;
pub type I16Vec1 = Vector1<i16>;
pub type I16Vec2 = Vector2<i16>;
pub type I16Vec3 = Vector3<i16>;
pub type I16Vec4 = Vector4<i16>;

pub type U8Vec1 = Vector1<u8>;
pub type U8Vec2 = Vector2<u8>;
pub type U8Vec3 = Vector3<u8>;
pub type U8Vec4 = Vector4<u8>;
pub type I8Vec1 = Vector1<i8>;
pub type I8Vec2 = Vector2<i8>;
pub type I8Vec3 = Vector3<i8>;
pub type I8Vec4 = Vector4<i8>;


pub type DMat2 = Matrix2<f64>;
pub type DMat2x2 = Matrix2<f64>;
pub type DMat2x3 = Matrix2x3<f64>;
pub type DMat2x4 = Matrix2x4<f64>;
pub type DMat3 = Matrix3<f64>;
pub type DMat3x2 = Matrix3x2<f64>;
pub type DMat3x3 = Matrix3<f64>;
pub type DMat3x4 = Matrix3x4<f64>;
pub type DMat4 = Matrix4<f64>;
pub type DMat4x2 = Matrix4x2<f64>;
pub type DMat4x3 = Matrix4x3<f64>;
pub type DMat4x4 = Matrix4<f64>;
pub type Mat2 = Matrix2<f32>;
pub type Mat2x2 = Matrix2<f32>;
pub type Mat2x3 = Matrix2x3<f32>;
pub type Mat2x4 = Matrix2x4<f32>;
pub type Mat3 = Matrix3<f32>;
pub type Mat3x2 = Matrix3x2<f32>;
pub type Mat3x3 = Matrix3<f32>;
pub type Mat3x4 = Matrix3x4<f32>;
pub type Mat4x2 = Matrix4x2<f32>;
pub type Mat4x3 = Matrix4x3<f32>;
pub type Mat4x4 = Matrix4<f32>;
pub type Mat4 = Matrix4<f32>;

pub type Quat = Quaternion<f32>;
pub type DQuat = Quaternion<f64>;
