use crate::{Isometry2, Isometry3};
use glam::{DMat3, DMat4, DQuat, DVec3, Mat3, Mat4, Quat, Vec3};

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

impl From<Isometry3<f32>> for (Vec3, Quat) {
    fn from(iso: Isometry3<f32>) -> (Vec3, Quat) {
        (iso.translation.into(), iso.rotation.into())
    }
}

impl From<Isometry3<f64>> for (DVec3, DQuat) {
    fn from(iso: Isometry3<f64>) -> (DVec3, DQuat) {
        (iso.translation.into(), iso.rotation.into())
    }
}

impl From<Isometry2<f32>> for (Vec3, Quat) {
    fn from(iso: Isometry2<f32>) -> (Vec3, Quat) {
        let tra = Vec3::new(iso.translation.x, iso.translation.y, 0.0);
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), iso.rotation.angle());
        (tra, rot)
    }
}

impl From<Isometry2<f64>> for (DVec3, DQuat) {
    fn from(iso: Isometry2<f64>) -> (DVec3, DQuat) {
        let tra = DVec3::new(iso.translation.x, iso.translation.y, 0.0);
        let rot = DQuat::from_axis_angle(DVec3::new(0.0, 0.0, 1.0), iso.rotation.angle());
        (tra, rot)
    }
}

#[cfg(feature = "convert-glam-unchecked")]
mod unchecked {
    use crate::{Isometry2, Isometry3, Matrix3, Matrix4};
    use glam::{DMat3, DMat4, DQuat, DVec2, DVec3, Mat3, Mat4, Quat, Vec2, Vec3};

    impl From<(Vec3, Quat)> for Isometry3<f32> {
        fn from((tra, rot): (Vec3, Quat)) -> Self {
            Isometry3::from_parts(tra.into(), rot.into())
        }
    }

    impl From<(DVec3, DQuat)> for Isometry3<f64> {
        fn from((tra, rot): (DVec3, DQuat)) -> Self {
            Isometry3::from_parts(tra.into(), rot.into())
        }
    }

    impl From<(Vec3, Quat)> for Isometry2<f32> {
        fn from((tra, rot): (Vec3, Quat)) -> Self {
            Isometry2::new([tra.x, tra.y].into(), rot.to_axis_angle().1)
        }
    }

    impl From<(DVec3, DQuat)> for Isometry2<f64> {
        fn from((tra, rot): (DVec3, DQuat)) -> Self {
            Isometry2::new([tra.x, tra.y].into(), rot.to_axis_angle().1)
        }
    }

    impl From<(Vec2, Quat)> for Isometry2<f32> {
        fn from((tra, rot): (Vec2, Quat)) -> Self {
            Isometry2::new([tra.x, tra.y].into(), rot.to_axis_angle().1)
        }
    }

    impl From<(DVec2, DQuat)> for Isometry2<f64> {
        fn from((tra, rot): (DVec2, DQuat)) -> Self {
            Isometry2::new([tra.x, tra.y].into(), rot.to_axis_angle().1)
        }
    }

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
