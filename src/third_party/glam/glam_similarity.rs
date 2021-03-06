use crate::{Similarity2, Similarity3};
use glam::{DMat3, DMat4, Mat3, Mat4};

impl From<Similarity2<f32>> for Mat3 {
    fn from(iso: Similarity2<f32>) -> Mat3 {
        iso.to_homogeneous().into()
    }
}
impl From<Similarity3<f32>> for Mat4 {
    fn from(iso: Similarity3<f32>) -> Mat4 {
        iso.to_homogeneous().into()
    }
}

impl From<Similarity2<f64>> for DMat3 {
    fn from(iso: Similarity2<f64>) -> DMat3 {
        iso.to_homogeneous().into()
    }
}
impl From<Similarity3<f64>> for DMat4 {
    fn from(iso: Similarity3<f64>) -> DMat4 {
        iso.to_homogeneous().into()
    }
}

#[cfg(feature = "convert-glam-unchecked")]
mod unchecked {
    use crate::{Matrix3, Matrix4, Similarity2, Similarity3};
    use glam::{DMat3, DMat4, Mat3, Mat4};

    impl From<Mat3> for Similarity2<f32> {
        fn from(mat3: Mat3) -> Similarity2<f32> {
            crate::convert_unchecked(Matrix3::from(mat3))
        }
    }

    impl From<Mat4> for Similarity3<f32> {
        fn from(mat4: Mat4) -> Similarity3<f32> {
            crate::convert_unchecked(Matrix4::from(mat4))
        }
    }

    impl From<DMat3> for Similarity2<f64> {
        fn from(mat3: DMat3) -> Similarity2<f64> {
            crate::convert_unchecked(Matrix3::from(mat3))
        }
    }

    impl From<DMat4> for Similarity3<f64> {
        fn from(mat4: DMat4) -> Similarity3<f64> {
            crate::convert_unchecked(Matrix4::from(mat4))
        }
    }
}
