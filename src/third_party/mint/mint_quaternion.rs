use crate::{Quaternion, Scalar, SimdValue, UnitQuaternion};

impl<N: Scalar> From<mint::Quaternion<N>> for Quaternion<N> {
    fn from(q: mint::Quaternion<N>) -> Self {
        Self::new(q.s, q.v.x, q.v.y, q.v.z)
    }
}

impl<N: Scalar> Into<mint::Quaternion<N>> for Quaternion<N> {
    fn into(self) -> mint::Quaternion<N> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0].inlined_clone(),
                y: self[1].inlined_clone(),
                z: self[2].inlined_clone(),
            },
            s: self[3].inlined_clone(),
        }
    }
}

impl<N: Scalar + SimdValue> Into<mint::Quaternion<N>> for UnitQuaternion<N> {
    fn into(self) -> mint::Quaternion<N> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0].inlined_clone(),
                y: self[1].inlined_clone(),
                z: self[2].inlined_clone(),
            },
            s: self[3].inlined_clone(),
        }
    }
}
