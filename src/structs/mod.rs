//! Data structures and implementations.

pub use self::dmat::{DMat, DMat1, DMat2, DMat3, DMat4, DMat5, DMat6};
pub use self::dvec::{DVec, DVec1, DVec2, DVec3, DVec4, DVec5, DVec6};
pub use self::vec::{Vec1, Vec2, Vec3, Vec4, Vec5, Vec6};
pub use self::pnt::{Pnt1, Pnt2, Pnt3, Pnt4, Pnt5, Pnt6};
pub use self::mat::{Identity, Mat1, Mat2, Mat3, Mat4, Mat5, Mat6};
pub use self::rot::{Rot2, Rot3};
pub use self::iso::{Iso2, Iso3};
pub use self::sim::{Sim2, Sim3};
pub use self::persp::{Persp3, PerspMat3};
pub use self::ortho::{Ortho3, OrthoMat3};
pub use self::quat::{Quat, UnitQuat};

#[cfg(feature="generic_sizes")]
pub use self::vecn::VecN;

mod dmat_macros;
mod dmat;
mod vecn_macros;
#[cfg(feature="generic_sizes")]
mod vecn;
mod dvec_macros;
mod dvec;
mod vec_macros;
mod vec;
mod pnt_macros;
mod pnt;
mod quat;
mod mat_macros;
mod mat;
mod rot_macros;
mod rot;
mod iso_macros;
mod iso;
mod sim_macros;
mod sim;
mod persp;
mod ortho;

// Specialization for some 1d, 2d and 3d operations.
#[doc(hidden)]
mod spec {
    mod identity;
    mod mat;
    mod vec;
    mod primitives;
    // mod complex;
}
