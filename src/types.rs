use vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
use mat::{Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
use adaptors::rotmat::Rotmat;
use adaptors::transform::Transform;

// 1D
pub type Vec1f64 = Vec1<f64>;
pub type Vec1f32 = Vec1<f32>;
pub type Vec1flt = Vec1<float>;

pub type Mat1f64 = Mat1<f64>;
pub type Mat1f32 = Mat1<f32>;
pub type Mat1flt = Mat1<float>;

pub type Rot1f64 = Rotmat<Mat1<f64>>;
pub type Rot1f32 = Rotmat<Mat1<f32>>;
pub type Rot1flt = Rotmat<Mat1<float>>;

pub type Iso1f64 = Transform<Rot1f64, Vec1f64>;
pub type Iso1f32 = Transform<Rot1f32, Vec1f32>;
pub type Iso1flt = Transform<Rot1flt, Vec1flt>;

pub type Aff1f64 = Transform<Mat1f64, Vec1f64>;
pub type Aff1f32 = Transform<Mat1f32, Vec1f32>;
pub type Aff1flt = Transform<Mat1flt, Vec1flt>;

// 2D
pub type Vec2f64 = Vec2<f64>;
pub type Vec2f32 = Vec2<f32>;
pub type Vec2flt = Vec2<float>;

pub type Mat2f64 = Mat2<f64>;
pub type Mat2f32 = Mat2<f32>;
pub type Mat2flt = Mat2<float>;

pub type Rot2f64 = Rotmat<Mat2<f64>>;
pub type Rot2f32 = Rotmat<Mat2<f32>>;
pub type Rot2flt = Rotmat<Mat2<float>>;

pub type Iso2f64 = Transform<Rot2f64, Vec2f64>;
pub type Iso2f32 = Transform<Rot2f32, Vec2f32>;
pub type Iso2flt = Transform<Rot2flt, Vec2flt>;

pub type Aff2f64 = Transform<Mat2f64, Vec2f64>;
pub type Aff2f32 = Transform<Mat2f32, Vec2f32>;
pub type Aff2flt = Transform<Mat2flt, Vec2flt>;

// 3D
pub type Vec3f64 = Vec3<f64>;
pub type Vec3f32 = Vec3<f32>;
pub type Vec3flt = Vec3<float>;

pub type Mat3f64 = Mat3<f64>;
pub type Mat3f32 = Mat3<f32>;
pub type Mat3flt = Mat3<float>;

pub type Rot3f64 = Rotmat<Mat3<f64>>;
pub type Rot3f32 = Rotmat<Mat3<f32>>;
pub type Rot3flt = Rotmat<Mat3<float>>;

pub type Iso3f64 = Transform<Rot3f64, Vec3f64>;
pub type Iso3f32 = Transform<Rot3f32, Vec3f32>;
pub type Iso3flt = Transform<Rot3flt, Vec3flt>;

pub type Aff3f64 = Transform<Mat3f64, Vec3f64>;
pub type Aff3f32 = Transform<Mat3f32, Vec3f32>;
pub type Aff3flt = Transform<Mat3flt, Vec3flt>;

// 4D
pub type Vec4f64 = Vec4<f64>;
pub type Vec4f32 = Vec4<f32>;
pub type Vec4flt = Vec4<float>;

pub type Mat4f64 = Mat4<f64>;
pub type Mat4f32 = Mat4<f32>;
pub type Mat4flt = Mat4<float>;

pub type Rot4f64 = Rotmat<Mat4<f64>>;
pub type Rot4f32 = Rotmat<Mat4<f32>>;
pub type Rot4flt = Rotmat<Mat4<float>>;

pub type Iso4f64 = Transform<Rot4f64, Vec4f64>;
pub type Iso4f32 = Transform<Rot4f32, Vec4f32>;
pub type Iso4flt = Transform<Rot4flt, Vec4flt>;

pub type Aff4f64 = Transform<Mat4f64, Vec4f64>;
pub type Aff4f32 = Transform<Mat4f32, Vec4f32>;
pub type Aff4flt = Transform<Mat4flt, Vec4flt>;

// 5D
pub type Vec5f64 = Vec5<f64>;
pub type Vec5f32 = Vec5<f32>;
pub type Vec5flt = Vec5<float>;

pub type Mat5f64 = Mat5<f64>;
pub type Mat5f32 = Mat5<f32>;
pub type Mat5flt = Mat5<float>;

pub type Rot5f64 = Rotmat<Mat5<f64>>;
pub type Rot5f32 = Rotmat<Mat5<f32>>;
pub type Rot5flt = Rotmat<Mat5<float>>;

pub type Iso5f64 = Transform<Rot5f64, Vec5f64>;
pub type Iso5f32 = Transform<Rot5f32, Vec5f32>;
pub type Iso5flt = Transform<Rot5flt, Vec5flt>;

pub type Aff5f64 = Transform<Mat5f64, Vec5f64>;
pub type Aff5f32 = Transform<Mat5f32, Vec5f32>;
pub type Aff5flt = Transform<Mat5flt, Vec5flt>;

// 6D
pub type Vec6f64 = Vec6<f64>;
pub type Vec6f32 = Vec6<f32>;
pub type Vec6flt = Vec6<float>;

pub type Mat6f64 = Mat6<f64>;
pub type Mat6f32 = Mat6<f32>;
pub type Mat6flt = Mat6<float>;

pub type Rot6f64 = Rotmat<Mat6<f64>>;
pub type Rot6f32 = Rotmat<Mat6<f32>>;
pub type Rot6flt = Rotmat<Mat6<float>>;

pub type Iso6f64 = Transform<Rot6f64, Vec6f64>;
pub type Iso6f32 = Transform<Rot6f32, Vec6f32>;
pub type Iso6flt = Transform<Rot6flt, Vec6flt>;

pub type Aff6f64 = Transform<Mat6f64, Vec6f64>;
pub type Aff6f32 = Transform<Mat6f32, Vec6f32>;
pub type Aff6flt = Transform<Mat6flt, Vec6flt>;
