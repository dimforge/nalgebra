use na::{Real, Unit, Rotation3, Matrix4, U3, U4};

use traits::Number;
use aliases::{Vec, Mat};


pub fn rotate<N: Real>(angle: N, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Rotation3::from_axis_angle(&Unit::new_normalize(*v), angle).to_homogeneous()
}

pub fn scale<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_nonuniform_scaling(v)
}

pub fn translate<N: Number>(v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Matrix4::new_translation(v)
}
