//! (Reexported) Recommended features not specified by GLSL specification

//pub use self::bitfield::*;
pub use self::constants::{
    e, euler, four_over_pi, golden_ratio, half_pi, ln_ln_two, ln_ten, ln_two, one, one_over_pi,
    one_over_root_two, one_over_two_pi, quarter_pi, root_five, root_half_pi, root_ln_four, root_pi,
    root_three, root_two, root_two_pi, third, three_over_two_pi, two_over_pi, two_over_root_pi,
    two_pi, two_thirds, zero,
};
//pub use self::integer::*;
pub use self::matrix_access::{column, row, set_column, set_row};
pub use self::matrix_inverse::{affine_inverse, inverse_transpose};
//pub use self::packing::*;
//pub use self::reciprocal::*;
//pub use self::round::*;
pub use self::type_ptr::{
    make_mat2, make_mat2x2, make_mat2x3, make_mat2x4, make_mat3, make_mat3x2, make_mat3x3,
    make_mat3x4, make_mat4, make_mat4x2, make_mat4x3, make_mat4x4, make_quat, make_vec1, make_vec2,
    make_vec3, make_vec4, mat2_to_mat3, mat2_to_mat4, mat3_to_mat2, mat3_to_mat4, mat4_to_mat2,
    mat4_to_mat3, value_ptr, value_ptr_mut, vec1_to_vec2, vec1_to_vec3, vec1_to_vec4, vec2_to_vec1,
    vec2_to_vec2, vec2_to_vec3, vec2_to_vec4, vec3_to_vec1, vec3_to_vec2, vec3_to_vec3,
    vec3_to_vec4, vec4_to_vec1, vec4_to_vec2, vec4_to_vec3, vec4_to_vec4,
};
//pub use self::ulp::*;
pub use self::quaternion::{
    quat_cast, quat_euler_angles, quat_greater_than, quat_greater_than_equal, quat_less_than,
    quat_less_than_equal, quat_look_at, quat_look_at_lh, quat_look_at_rh, quat_pitch, quat_roll,
    quat_yaw,
};

//mod bitfield;
mod constants;
mod epsilon;
//mod integer;
mod matrix_access;
mod matrix_inverse;
//mod packing;
//mod reciprocal;
//mod round;
mod type_ptr;
//mod ulp;
mod quaternion;
