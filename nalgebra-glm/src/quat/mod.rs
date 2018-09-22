//! Definition and operations on quaternions.

pub use self::gtc_quaternion::*;
pub use self::gtx_quaternion::*;
pub use self::gtx_rotate_normalized_axis::*;
pub use self::quaternion_common::*;
pub use self::quaternion_geometric::*;
pub use self::quaternion_relational::*;
pub use self::quaternion_transform::*;
pub use self::quaternion_trigonometric::*;

mod gtc_quaternion;
mod gtx_quaternion;
mod gtx_rotate_normalized_axis;
mod quaternion_common;
mod quaternion_geometric;
mod quaternion_relational;
mod quaternion_transform;
mod quaternion_trigonometric;