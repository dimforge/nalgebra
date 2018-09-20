use na::{DefaultAllocator, Scalar, Real, U3, U4, Unit, Rotation3, Point3};

use traits::{Dimension, Number, Alloc};
use aliases::{Mat, Vec};

pub fn identity<N: Number, D: Dimension>() -> Mat<N, D, D>
    where DefaultAllocator: Alloc<N, D, D> {
    Mat::<N, D, D>::identity()
}

/// Same as `look_at_rh`
pub fn look_at<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    look_at_rh(eye, center, up)
}

pub fn look_at_lh<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Mat::look_at_lh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

pub fn look_at_rh<N: Real>(eye: &Vec<N, U3>, center: &Vec<N, U3>, up: &Vec<N, U3>) -> Mat<N, U4, U4> {
    Mat::look_at_rh(&Point3::from_coordinates(*eye), &Point3::from_coordinates(*center), up)
}

pub fn rotate<N: Real>(m: &Mat<N, U4, U4>, angle: N, axis: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m * Rotation3::from_axis_angle(&Unit::new_normalize(*axis), angle).to_homogeneous()
}

pub fn scale<N: Number>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m.append_nonuniform_scaling(v)
}

pub fn translate<N: Number>(m: &Mat<N, U4, U4>, v: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m.append_translation(v)
}
