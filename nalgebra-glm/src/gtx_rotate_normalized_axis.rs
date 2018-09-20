use na::{Real, Rotation3, UnitQuaternion, Unit, U3, U4};

use aliases::{Vec, Mat, Qua};

pub fn rotateNormalizedAxis<N: Real>(m: &Mat<N, U4, U4>, angle: N, axis: &Vec<N, U3>) -> Mat<N, U4, U4> {
    m * Rotation3::from_axis_angle(&Unit::new_unchecked(*axis), angle).to_homogeneous()
}

pub fn rotateNormalizedAxis2<N: Real>(q: &Qua<N>, angle: N, axis: &Vec<N, U3>) -> Qua<N> {
    q * UnitQuaternion::from_axis_angle(&Unit::new_unchecked(*axis), angle).unwrap()
}