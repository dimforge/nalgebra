

extern crate num_traits as num;
#[macro_use]
extern crate approx;
extern crate alga;
extern crate nalgebra as na;

pub use aliases::*;
pub use constructors::*;
pub use common::{abs, ceil, clamp, clamp2, clamp3, float_bits_to_int, float_bits_to_int_vec, float_bits_to_uint, float_bits_to_uint_vec, floor, fract, int_bits_to_float, int_bits_to_float_vec, mix, modf, modf_vec, round, sign, smoothstep, step, step_scalar, step_vec, trunc, uint_bits_to_float, uint_bits_to_float_scalar};
pub use geometric::{reflect_vec, cross, distance, dot, faceforward, length, magnitude, normalize, refract_vec};
pub use matrix::{transpose, determinant, inverse, matrix_comp_mult, outer_product};
pub use traits::{Dimension, Number, Alloc};
pub use trigonometric::{acos, acosh, asin, asinh, atan, atan2, atanh, cos, cosh, degrees, radians, sin, sinh, tan, tanh};
pub use vector_relational::{all, any, equal, greater_than, greater_than_equal, less_than, less_than_equal, not, not_equal};
pub use exponential::{exp, exp2, inversesqrt, log, log2, pow, sqrt};

pub use gtx::{
    comp_add, comp_max, comp_min, comp_mul,
    cross2d,
    left_handed, right_handed,
    matrix_cross, matrix_cross3,
    diagonal2x2, diagonal2x3, diagonal2x4, diagonal3x2, diagonal3x3, diagonal3x4, diagonal4x2, diagonal4x3, diagonal4x4,
    distance2, l1_distance, l1_norm, l2_distance, l2_norm, length2, magnitude2,
    triangle_normal,
    fast_normalize_dot, normalize_dot,
    quat_rotate_normalized_axis, rotate_normalized_axis,
    orientation, rotate_vec2, rotate_vec3, rotate_vec4, rotate_x, rotate_x_vec3, rotate_y, rotate_y_vec3, rotate_z, rotate_z_vec3, slerp,
    rotation, scaling, translation,
    proj, proj2d, reflect, reflect2d, scale_bias, scale_bias_matrix, shear2d_x, shear_x, shear_y, shear_y_mat3, shear_z,
    rotate2d, scale2d, translate2d,
    angle,
    are_collinear, are_collinear2d, are_orthogonal, is_comp_null, is_normalized, is_null,
    quat_to_mat3, quat_rotate_vec, quat_cross_vec, mat3_to_quat, quat_extract_real_component, quat_fast_mix, quat_inv_cross_vec, quat_length2, quat_magnitude2, quat_identity, quat_rotate_vec3, quat_rotation, quat_short_mix, quat_to_mat4, to_quat
};
pub use gtc::{
    e, two_pi, euler, four_over_pi, golden_ratio, half_pi, ln_ln_two, ln_ten, ln_two, one, one_over_pi, one_over_root_two, one_over_two_pi, quarter_pi, root_five, root_half_pi, root_ln_four, root_pi, root_three, root_two, root_two_pi, third, three_over_two_pi, two_over_pi, two_over_root_pi, two_thirds, zero,
    column, row, set_column, set_row,
    affine_inverse, inverse_transpose,
    make_mat2, make_mat2x2, make_mat2x3, make_mat2x4, make_mat3, make_mat3x2, make_mat3x3, make_mat3x4, make_mat4, make_mat4x2, make_mat4x3, make_mat4x4, make_quat, make_vec1, make_vec2, make_vec3, make_vec4, value_ptr, value_ptr_mut, vec1_to_vec2, vec1_to_vec3, vec1_to_vec4, vec2_to_vec1, vec2_to_vec2, vec2_to_vec3, vec2_to_vec4, vec3_to_vec1, vec3_to_vec2, vec3_to_vec3, vec3_to_vec4, vec4_to_vec1, vec4_to_vec2, vec4_to_vec3, vec4_to_vec4,
    quat_cast, quat_euler_angles, quat_greater_than, quat_greater_than_equal, quat_less_than, quat_less_than_equal, quat_look_at, quat_look_at_lh, quat_look_at_rh, quat_pitch, quat_roll, quat_yaw
};
pub use ext::{
    ortho, perspective,
    pick_matrix, project, project_no, project_zo, unproject, unproject_no, unproject_zo,
    equal_columns, equal_columns_eps, equal_columns_eps_vec, not_equal_columns, not_equal_columns_eps, not_equal_columns_eps_vec,
    identity, look_at, look_at_lh, rotate, scale, look_at_rh, translate,
    max3_scalar, max4_scalar, min3_scalar, min4_scalar,
    epsilon, pi,
    max, max2, max3, max4, min, min2, min3, min4,
    equal_eps, equal_eps_vec, not_equal_eps, not_equal_eps_vec,
    quat_conjugate, quat_inverse, quat_lerp, quat_slerp,
    quat_cross, quat_dot, quat_length, quat_magnitude, quat_normalize,
    quat_equal, quat_equal_eps, quat_not_equal, quat_not_equal_eps,
    quat_exp, quat_log, quat_pow, quat_rotate,
    quat_angle, quat_angle_axis, quat_axis
};

mod aliases;
mod constructors;
mod common;
mod matrix;
mod geometric;
mod traits;
mod trigonometric;
mod vector_relational;
mod exponential;
//mod integer;
//mod packing;

mod ext;
mod gtc;
mod gtx;