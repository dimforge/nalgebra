use na::{Matrix2, Matrix2x3, Matrix2x4, Matrix3, Matrix3x2, Matrix3x4, Matrix4, Matrix4x2, Matrix4x3,
         U2, U3, U4};

use traits::Number;
use aliases::{Vec, Mat};


pub fn diagonal2x2<N: Number>(v: &Vec<N, U2>) -> Mat<N, U2, U2> {
    Matrix2::from_diagonal(v)
}

pub fn diagonal2x3<N: Number>(v: &Vec<N, U2>) -> Mat<N, U2, U3> {
    Matrix2x3::from_partial_diagonal(v.as_slice())
}

pub fn diagonal2x4<N: Number>(v: &Vec<N, U2>) -> Mat<N, U2, U4> {
    Matrix2x4::from_partial_diagonal(v.as_slice())
}

pub fn diagonal3x2<N: Number>(v: &Vec<N, U3>) -> Mat<N, U3, U2> {
    Matrix3x2::from_partial_diagonal(v.as_slice())
}

pub fn diagonal3x3<N: Number>(v: &Vec<N, U3>) -> Mat<N, U3, U3> {
    Matrix3::from_diagonal(v)
}

pub fn diagonal3x4<N: Number>(v: &Vec<N, U3>) -> Mat<N, U3, U4> {
    Matrix3x4::from_partial_diagonal(v.as_slice())
}

pub fn diagonal4x2<N: Number>(v: &Vec<N, U4>) -> Mat<N, U4, U2> {
    Matrix4x2::from_partial_diagonal(v.as_slice())
}

pub fn diagonal4x3<N: Number>(v: &Vec<N, U4>) -> Mat<N, U4, U3> {
    Matrix4x3::from_partial_diagonal(v.as_slice())
}

pub fn diagonal4x4<N: Number>(v: &Vec<N, U4>) -> Mat<N, U4, U4> {
    Matrix4::from_diagonal(v)
}
