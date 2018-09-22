//! (Reexported) Additional features not specified by GLSL specification

pub use self::matrix_clip_space::{ortho, perspective};
pub use self::matrix_projection::{pick_matrix, project, project_no, project_zo, unproject, unproject_no, unproject_zo};
pub use self::matrix_relationnal::{equal_columns, equal_columns_eps, equal_columns_eps_vec, not_equal_columns, not_equal_columns_eps, not_equal_columns_eps_vec};
pub use self::matrix_transform::{identity, look_at, look_at_lh, rotate, scale, look_at_rh, translate, rotate_x, rotate_y, rotate_z};
pub use self::scalar_common::{max3_scalar, max4_scalar, min3_scalar, min4_scalar};
pub use self::scalar_constants::{epsilon, pi};
pub use self::vector_common::{max, max2, max3, max4, min, min2, min3, min4};
pub use self::vector_relational::{equal_eps, equal_eps_vec, not_equal_eps, not_equal_eps_vec};
pub use self::quaternion_common::{quat_conjugate, quat_inverse, quat_lerp, quat_slerp};
pub use self::quaternion_geometric::{quat_cross, quat_dot, quat_length, quat_magnitude, quat_normalize};
pub use self::quaternion_relational::{quat_equal, quat_equal_eps, quat_not_equal, quat_not_equal_eps};
pub use self::quaternion_transform::{quat_exp, quat_log, quat_pow, quat_rotate};
pub use self::quaternion_trigonometric::{quat_angle, quat_angle_axis, quat_axis};


mod matrix_clip_space;
mod matrix_projection;
mod matrix_relationnal;
mod matrix_transform;
mod scalar_common;
mod scalar_constants;
mod vector_common;
mod vector_relational;
mod quaternion_common;
mod quaternion_geometric;
mod quaternion_relational;
mod quaternion_transform;
mod quaternion_trigonometric;