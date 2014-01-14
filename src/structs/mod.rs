//! Data structures and implementations.

pub use self::dmat::DMat;
pub use self::dvec::DVec;
pub use self::vec::{Vec0, Vec1, Vec2, Vec3, PVec3, Vec4, Vec5, Vec6};
pub use self::mat::{Identity, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
pub use self::rot::{Rot2, Rot3, Rot4};
pub use self::iso::{Iso2, Iso3, Iso4};

mod dmat;
mod dvec;
mod vec;
mod mat;
mod rot;
mod iso;

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
