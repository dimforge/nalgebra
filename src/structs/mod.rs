//! Data structures and implementations.

pub use self::dmat::DMat;
pub use self::dvec::{DVec, DVec1, DVec2, DVec3, DVec4, DVec5, DVec6};
pub use self::vec::{Vec0, Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
pub use self::pnt::{Pnt0, Pnt1, Pnt2, Pnt3, Pnt4, Pnt5, Pnt6};
pub use self::mat::{Identity, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
pub use self::rot::{Rot2, Rot3, Rot4};
pub use self::iso::{Iso2, Iso3, Iso4};
pub use self::persp::{Persp3, PerspMat3};
pub use self::ortho::{Ortho3, OrthoMat3};

pub use self::vec::{Vec1MulRhs, Vec2MulRhs, Vec3MulRhs, Vec4MulRhs, Vec5MulRhs, Vec6MulRhs,
                    Vec1DivRhs, Vec2DivRhs, Vec3DivRhs, Vec4DivRhs, Vec5DivRhs, Vec6DivRhs,
                    Vec1AddRhs, Vec2AddRhs, Vec3AddRhs, Vec4AddRhs, Vec5AddRhs, Vec6AddRhs,
                    Vec1SubRhs, Vec2SubRhs, Vec3SubRhs, Vec4SubRhs, Vec5SubRhs, Vec6SubRhs};
pub use self::pnt::{Pnt1MulRhs, Pnt2MulRhs, Pnt3MulRhs, Pnt4MulRhs, Pnt5MulRhs, Pnt6MulRhs,
                    Pnt1DivRhs, Pnt2DivRhs, Pnt3DivRhs, Pnt4DivRhs, Pnt5DivRhs, Pnt6DivRhs,
                    Pnt1AddRhs, Pnt2AddRhs, Pnt3AddRhs, Pnt4AddRhs, Pnt5AddRhs, Pnt6AddRhs,
                    Pnt1SubRhs, Pnt2SubRhs, Pnt3SubRhs, Pnt4SubRhs, Pnt5SubRhs, Pnt6SubRhs};
pub use self::mat::{Mat1MulRhs, Mat2MulRhs, Mat3MulRhs, Mat4MulRhs, Mat5MulRhs, Mat6MulRhs,
                    Mat1DivRhs, Mat2DivRhs, Mat3DivRhs, Mat4DivRhs, Mat5DivRhs, Mat6DivRhs,
                    Mat1AddRhs, Mat2AddRhs, Mat3AddRhs, Mat4AddRhs, Mat5AddRhs, Mat6AddRhs,
                    Mat1SubRhs, Mat2SubRhs, Mat3SubRhs, Mat4SubRhs, Mat5SubRhs, Mat6SubRhs};

mod metal;
mod dmat;
mod dvec_macros;
mod dvec;
mod vec_macros;
mod vec;
mod pnt_macros;
mod pnt;
mod mat_macros;
mod mat;
mod rot_macros;
mod rot;
mod iso_macros;
mod iso;
mod persp;
mod ortho;

// specialization for some 1d, 2d and 3d operations
#[doc(hidden)]
mod spec {
    mod identity;
    mod mat;
    mod vec0;
    mod vec;
    mod primitives;
    // mod complex;
}
