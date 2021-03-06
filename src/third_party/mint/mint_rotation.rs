use crate::{RealField, Rotation3};

impl<N: RealField> From<mint::EulerAngles<N, mint::IntraXYZ>> for Rotation3<N> {
    fn from(euler: mint::EulerAngles<N, mint::IntraXYZ>) -> Self {
        Self::from_euler_angles(euler.a, euler.b, euler.c)
    }
}
