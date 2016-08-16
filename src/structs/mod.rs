//! Data structures and implementations.

pub use self::dmatrix::{DMatrix, DMatrix1, DMatrix2, DMatrix3, DMatrix4, DMatrix5, DMatrix6};
pub use self::dvector::{DVector, DVector1, DVector2, DVector3, DVector4, DVector5, DVector6};
pub use self::vector::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6};
pub use self::point::{Point1, Point2, Point3, Point4, Point5, Point6};
pub use self::matrix::{Identity, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6};
pub use self::rotation::{Rotation2, Rotation3};
pub use self::isometry::{Isometry2, Isometry3};
pub use self::similarity::{Similarity2, Similarity3};
pub use self::perspective::{Perspective3, PerspectiveMatrix3};
pub use self::orthographic::{Orthographic3, OrthographicMatrix3};
pub use self::quaternion::{Quaternion, UnitQuaternion};
pub use self::unit::Unit;

#[cfg(feature="generic_sizes")]
pub use self::vectorn::VectorN;

mod common_macros;
mod dmatrix_macros;
mod dmatrix;
mod vectorn_macros;
#[cfg(feature="generic_sizes")]
mod vectorn;
mod dvector_macros;
mod dvector;
mod vector_macros;
mod vector;
mod point_macros;
mod point;
mod quaternion;
mod matrix_macros;
mod matrix;
mod rotation_macros;
mod rotation;
mod isometry_macros;
mod isometry;
mod similarity_macros;
mod similarity;
mod perspective;
mod orthographic;
mod unit;

// Specialization for some 1d, 2d and 3d operations.
#[doc(hidden)]
mod specializations {
    mod identity;
    mod matrix;
    mod vector;
    mod primitives;
    // mod complex;
}
