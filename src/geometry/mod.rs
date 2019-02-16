//! [Reexported at the root of this crate.] Data structures for points and usual transformations
//! (rotations, isometries, etc.)

mod op_macros;

mod point;
mod point_alga;
mod point_alias;
mod point_construction;
mod point_conversion;
mod point_coordinates;
mod point_ops;

mod rotation;
mod rotation_alga; // FIXME: implement Rotation methods.
mod rotation_alias;
mod rotation_construction;
mod rotation_conversion;
mod rotation_ops;
mod rotation_specialization;

mod quaternion;
mod quaternion_alga;
mod quaternion_construction;
mod quaternion_conversion;
mod quaternion_coordinates;
mod quaternion_ops;

mod unit_complex;
mod unit_complex_alga;
mod unit_complex_construction;
mod unit_complex_conversion;
mod unit_complex_ops;

mod translation;
mod translation_alga;
mod translation_alias;
mod translation_construction;
mod translation_conversion;
mod translation_coordinates;
mod translation_ops;

mod isometry;
mod isometry_alga;
mod isometry_alias;
mod isometry_construction;
mod isometry_conversion;
mod isometry_ops;

mod similarity;
mod similarity_alga;
mod similarity_alias;
mod similarity_construction;
mod similarity_conversion;
mod similarity_ops;

mod swizzle;

mod transform;
mod transform_alga;
mod transform_alias;
mod transform_construction;
mod transform_conversion;
mod transform_ops;

mod reflection;

mod orthographic;
mod perspective;

pub use self::point::*;
pub use self::point_alias::*;

pub use self::rotation::*;
pub use self::rotation_alias::*;

pub use self::quaternion::*;

pub use self::unit_complex::*;

pub use self::translation::*;
pub use self::translation_alias::*;

pub use self::isometry::*;
pub use self::isometry_alias::*;

pub use self::similarity::*;
pub use self::similarity_alias::*;

pub use self::transform::*;
pub use self::transform_alias::*;

pub use self::reflection::*;

pub use self::orthographic::Orthographic3;
pub use self::perspective::Perspective3;
