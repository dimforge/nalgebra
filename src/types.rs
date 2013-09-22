//! Useful type aliases.

use vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
use mat::{Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
use adaptors::rotmat::Rotmat;
use adaptors::transform::Transform;

// 1D
/// 1-dimensional `f64`-valued vector.
pub type Vec1f64 = Vec1<f64>;
/// 1-dimensional `f32`-valued vector.
pub type Vec1f32 = Vec1<f32>;
/// 1-dimensional `float`-valued vector.
pub type Vec1flt = Vec1<float>;

/// 1-dimensional `f64`-valued matrix.
pub type Mat1f64 = Mat1<f64>;
/// 1-dimensional `f32`-valued matrix.
pub type Mat1f32 = Mat1<f32>;
/// 1-dimensional `float`-valued matrix.
pub type Mat1flt = Mat1<float>;

// /// 1-dimensional `f64`-valued rotation matrix.
// pub type Rot1f64 = Rotmat<Mat1<f64>>;
// /// 1-dimensional `f32`-valued rotation matrix.
// pub type Rot1f32 = Rotmat<Mat1<f32>>;
// /// 1-dimensional `float`-valued rotation matrix.
// pub type Rot1flt = Rotmat<Mat1<float>>;
// 
// /// 1-dimensional `f64`-valued isometric transform.
// pub type Iso1f64 = Transform<Rot1f64, Vec1f64>;
// /// 1-dimensional `f32`-valued isometric transform.
// pub type Iso1f32 = Transform<Rot1f32, Vec1f32>;
// /// 1-dimensional `float`-valued isometric transform.
// pub type Iso1flt = Transform<Rot1flt, Vec1flt>;

/// 1-dimensional `f64`-valued general transform.
pub type Aff1f64 = Transform<Vec1f64, Mat1f64>;
/// 1-dimensional `f32`-valued general transform.
pub type Aff1f32 = Transform<Vec1f32, Mat1f32>;
/// 1-dimensional `float`-valued general transform.
pub type Aff1flt = Transform<Vec1flt, Mat1flt>;

// 2D
/// 2-dimensional `f64`-valued vector.
pub type Vec2f64 = Vec2<f64>;
/// 2-dimensional `f32`-valued vector.
pub type Vec2f32 = Vec2<f32>;
/// 2-dimensional `float`-valued vector.
pub type Vec2flt = Vec2<float>;

/// 2-dimensional `f64`-valued matrix.
pub type Mat2f64 = Mat2<f64>;
/// 2-dimensional `f32`-valued matrix.
pub type Mat2f32 = Mat2<f32>;
/// 2-dimensional `float`-valued matrix.
pub type Mat2flt = Mat2<float>;

/// 2-dimensional `f64`-valued rotation matrix.
pub type Rot2f64 = Rotmat<Mat2<f64>>;
/// 2-dimensional `f32`-valued rotation matrix.
pub type Rot2f32 = Rotmat<Mat2<f32>>;
/// 2-dimensional `float`-valued rotation matrix.
pub type Rot2flt = Rotmat<Mat2<float>>;

/// 2-dimensional `f64`-valued isometric transform.
pub type Iso2f64 = Transform<Vec2f64, Rot2f64>;
/// 2-dimensional `f32`-valued isometric transform.
pub type Iso2f32 = Transform<Vec2f32, Rot2f32>;
/// 2-dimensional `float`-valued isometric transform.
pub type Iso2flt = Transform<Vec2flt, Rot2flt>;

/// 2-dimensional `f64`-valued general transform.
pub type Aff2f64 = Transform<Vec2f64, Mat2f64>;
/// 2-dimensional `f32`-valued general transform.
pub type Aff2f32 = Transform<Vec2f32, Mat2f32>;
/// 2-dimensional `float`-valued general transform.
pub type Aff2flt = Transform<Vec2flt, Mat2flt>;

// 3D
/// 3-dimensional `f64`-valued vector.
pub type Vec3f64 = Vec3<f64>;
/// 3-dimensional `f32`-valued vector.
pub type Vec3f32 = Vec3<f32>;
/// 3-dimensional `float`-valued vector.
pub type Vec3flt = Vec3<float>;

/// 3-dimensional `f64`-valued matrix.
pub type Mat3f64 = Mat3<f64>;
/// 3-dimensional `f32`-valued matrix.
pub type Mat3f32 = Mat3<f32>;
/// 3-dimensional `float`-valued matrix.
pub type Mat3flt = Mat3<float>;

/// 3-dimensional `f64`-valued rotation matrix.
pub type Rot3f64 = Rotmat<Mat3<f64>>;
/// 3-dimensional `f32`-valued rotation matrix.
pub type Rot3f32 = Rotmat<Mat3<f32>>;
/// 3-dimensional `float`-valued rotation matrix.
pub type Rot3flt = Rotmat<Mat3<float>>;

/// 3-dimensional `f64`-valued isometric transform.
pub type Iso3f64 = Transform<Vec3f64, Rot3f64>;
/// 3-dimensional `f32`-valued isometric transform.
pub type Iso3f32 = Transform<Vec3f32, Rot3f32>;
/// 3-dimensional `float`-valued isometric transform.
pub type Iso3flt = Transform<Vec3flt, Rot3flt>;

/// 3-dimensional `f64`-valued general transform.
pub type Aff3f64 = Transform<Vec3f64, Mat3f64>;
/// 3-dimensional `f32`-valued general transform.
pub type Aff3f32 = Transform<Vec3f32, Mat3f32>;
/// 3-dimensional `float`-valued general transform.
pub type Aff3flt = Transform<Vec3flt, Mat3flt>;

// 4D
/// 4-dimensional `f64`-valued vector.
pub type Vec4f64 = Vec4<f64>;
/// 4-dimensional `f32`-valued vector.
pub type Vec4f32 = Vec4<f32>;
/// 4-dimensional `float`-valued vector.
pub type Vec4flt = Vec4<float>;

/// 4-dimensional `f64`-valued matrix.
pub type Mat4f64 = Mat4<f64>;
/// 4-dimensional `f32`-valued matrix.
pub type Mat4f32 = Mat4<f32>;
/// 4-dimensional `float`-valued matrix.
pub type Mat4flt = Mat4<float>;

// /// 4-dimensional `f64`-valued rotation matrix.
// pub type Rot4f64 = Rotmat<Mat4<f64>>;
// /// 4-dimensional `f32`-valued rotation matrix.
// pub type Rot4f32 = Rotmat<Mat4<f32>>;
// /// 4-dimensional `float`-valued rotation matrix.
// pub type Rot4flt = Rotmat<Mat4<float>>;
// 
// /// 4-dimensional `f64`-valued isometric transform.
// pub type Iso4f64 = Transform<Rot4f64, Vec4f64>;
// /// 4-dimensional `f32`-valued isometric transform.
// pub type Iso4f32 = Transform<Rot4f32, Vec4f32>;
// /// 4-dimensional `float`-valued isometric transform.
// pub type Iso4flt = Transform<Rot4flt, Vec4flt>;

/// 4-dimensional `f64`-valued general transform.
pub type Aff4f64 = Transform<Vec4f64, Mat4f64>;
/// 4-dimensional `f32`-valued general transform.
pub type Aff4f32 = Transform<Vec4f32, Mat4f32>;
/// 4-dimensional `float`-valued general transform.
pub type Aff4flt = Transform<Vec4flt, Mat4flt>;

// 5D
/// 5-dimensional `f64`-valued vector.
pub type Vec5f64 = Vec5<f64>;
/// 5-dimensional `f32`-valued vector.
pub type Vec5f32 = Vec5<f32>;
/// 5-dimensional `float`-valued vector.
pub type Vec5flt = Vec5<float>;

/// 5-dimensional `f64`-valued matrix.
pub type Mat5f64 = Mat5<f64>;
/// 5-dimensional `f32`-valued matrix.
pub type Mat5f32 = Mat5<f32>;
/// 5-dimensional `float`-valued matrix.
pub type Mat5flt = Mat5<float>;

// /// 5-dimensional `f64`-valued rotation matrix.
// pub type Rot5f64 = Rotmat<Mat5<f64>>;
// /// 5-dimensional `f32`-valued rotation matrix.
// pub type Rot5f32 = Rotmat<Mat5<f32>>;
// /// 5-dimensional `float`-valued rotation matrix.
// pub type Rot5flt = Rotmat<Mat5<float>>;
// 
// /// 5-dimensional `f64`-valued isometric transform.
// pub type Iso5f64 = Transform<Rot5f64, Vec5f64>;
// /// 5-dimensional `f32`-valued isometric transform.
// pub type Iso5f32 = Transform<Rot5f32, Vec5f32>;
// /// 5-dimensional `float`-valued isometric transform.
// pub type Iso5flt = Transform<Rot5flt, Vec5flt>;

/// 5-dimensional `f64`-valued general transform.
pub type Aff5f64 = Transform<Vec5f64, Mat5f64>;
/// 5-dimensional `f32`-valued general transform.
pub type Aff5f32 = Transform<Vec5f32, Mat5f32>;
/// 5-dimensional `float`-valued general transform.
pub type Aff5flt = Transform<Vec5flt, Mat5flt>;

// 6D
/// 6-dimensional `f64`-valued vector.
pub type Vec6f64 = Vec6<f64>;
/// 6-dimensional `f32`-valued vector.
pub type Vec6f32 = Vec6<f32>;
/// 6-dimensional `float`-valued vector.
pub type Vec6flt = Vec6<float>;

/// 6-dimensional `f64`-valued matrix.
pub type Mat6f64 = Mat6<f64>;
/// 6-dimensional `f32`-valued matrix.
pub type Mat6f32 = Mat6<f32>;
/// 6-dimensional `float`-valued matrix.
pub type Mat6flt = Mat6<float>;

// /// 6-dimensional `f64`-valued rotation matrix.
// pub type Rot6f64 = Rotmat<Mat6<f64>>;
// /// 6-dimensional `f32`-valued rotation matrix.
// pub type Rot6f32 = Rotmat<Mat6<f32>>;
// /// 6-dimensional `float`-valued rotation matrix.
// pub type Rot6flt = Rotmat<Mat6<float>>;
// 
// /// 6-dimensional `f64`-valued isometric transform.
// pub type Iso6f64 = Transform<Rot6f64, Vec6f64>;
// /// 6-dimensional `f32`-valued isometric transform.
// pub type Iso6f32 = Transform<Rot6f32, Vec6f32>;
// /// 6-dimensional `float`-valued isometric transform.
// pub type Iso6flt = Transform<Rot6flt, Vec6flt>;

/// 6-dimensional `f64`-valued general transform.
pub type Aff6f64 = Transform<Vec6f64, Mat6f64>;
/// 6-dimensional `f32`-valued general transform.
pub type Aff6f32 = Transform<Vec6f32, Mat6f32>;
/// 6-dimensional `float`-valued general transform.
pub type Aff6flt = Transform<Vec6flt, Mat6flt>;
