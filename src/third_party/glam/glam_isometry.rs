use crate::{Isometry2, Isometry3};
use glam::{DMat3, DMat4, Mat3, Mat4};

impl From<Isometry2<f32>> for Mat3 {
    fn from(iso: Isometry2<f32>) -> Mat3 {
        iso.to_homogeneous().into()
    }
}
impl From<Isometry3<f32>> for Mat4 {
    fn from(iso: Isometry3<f32>) -> Mat4 {
        iso.to_homogeneous().into()
    }
}

impl From<Isometry2<f64>> for DMat3 {
    fn from(iso: Isometry2<f64>) -> DMat3 {
        iso.to_homogeneous().into()
    }
}
impl From<Isometry3<f64>> for DMat4 {
    fn from(iso: Isometry3<f64>) -> DMat4 {
        iso.to_homogeneous().into()
    }
}

#[cfg(feature = "convert-glam-unchecked")]
mod unchecked {
    use crate::{Isometry2, Isometry3, Matrix3, Matrix4};
    use glam::{DMat3, DMat4, Mat3, Mat4};

    impl From<Mat3> for Isometry2<f32> {
        fn from(mat3: Mat3) -> Isometry2<f32> {
            crate::convert_unchecked(Matrix3::from(mat3))
        }
    }

    impl From<Mat4> for Isometry3<f32> {
        fn from(mat4: Mat4) -> Isometry3<f32> {
            crate::convert_unchecked(Matrix4::from(mat4))
        }
    }

    impl From<DMat3> for Isometry2<f64> {
        fn from(mat3: DMat3) -> Isometry2<f64> {
            crate::convert_unchecked(Matrix3::from(mat3))
        }
    }

    impl From<DMat4> for Isometry3<f64> {
        fn from(mat4: DMat4) -> Isometry3<f64> {
            crate::convert_unchecked(Matrix4::from(mat4))
        }
    }
}
