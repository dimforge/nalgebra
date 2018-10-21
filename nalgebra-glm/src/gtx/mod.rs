//! (Reexported) Experimental features not specified by GLSL specification.

pub use self::component_wise::{comp_add, comp_max, comp_min, comp_mul};
//pub use self::euler_angles::*;
pub use self::exterior_product::cross2d;
pub use self::handed_coordinate_space::{left_handed, right_handed};
pub use self::matrix_cross_product::{matrix_cross, matrix_cross3};
pub use self::matrix_operation::{
    diagonal2x2, diagonal2x3, diagonal2x4, diagonal3x2, diagonal3x3, diagonal3x4, diagonal4x2,
    diagonal4x3, diagonal4x4,
};
pub use self::norm::{distance2, l1_distance, l1_norm, l2_distance, l2_norm, length2, magnitude2};
pub use self::normal::triangle_normal;
pub use self::normalize_dot::{fast_normalize_dot, normalize_dot};
pub use self::quaternion::{
    mat3_to_quat, quat_cross_vec, quat_extract_real_component, quat_fast_mix, quat_identity,
    quat_inv_cross_vec, quat_length2, quat_magnitude2, quat_rotate_vec, quat_rotate_vec3,
    quat_rotation, quat_short_mix, quat_to_mat3, quat_to_mat4, to_quat,
};
pub use self::rotate_normalized_axis::{quat_rotate_normalized_axis, rotate_normalized_axis};
pub use self::rotate_vector::{
    orientation, rotate_vec2, rotate_vec3, rotate_vec4, rotate_x_vec3, rotate_x_vec4,
    rotate_y_vec3, rotate_y_vec4, rotate_z_vec3, rotate_z_vec4, slerp,
};
pub use self::transform::{rotation, rotation2d, scaling, scaling2d, translation, translation2d};
pub use self::transform2::{
    proj, proj2d, reflect, reflect2d, scale_bias, scale_bias_matrix, shear2d_x, shear2d_y, shear_x,
    shear_y, shear_z,
};
pub use self::transform2d::{rotate2d, scale2d, translate2d};
pub use self::vector_angle::angle;
pub use self::vector_query::{
    are_collinear, are_collinear2d, are_orthogonal, is_comp_null, is_normalized, is_null,
};

mod component_wise;
//mod euler_angles;
mod exterior_product;
mod handed_coordinate_space;
mod matrix_cross_product;
mod matrix_operation;
mod norm;
mod normal;
mod normalize_dot;
mod quaternion;
mod rotate_normalized_axis;
mod rotate_vector;
mod transform;
mod transform2;
mod transform2d;
mod vector_angle;
mod vector_query;
