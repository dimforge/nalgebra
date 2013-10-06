use std::num::{One, Zero};
use structs::mat;
use traits::operations::{Inv, Transpose};
use traits::geometry::{Translation, Translate, Rotation, Rotate, Transformation, Transform};

impl One for mat::Identity {
    #[inline]
    fn one() -> mat::Identity {
        mat::Identity::new()
    }
}

impl Inv for mat::Identity {
    fn inverted(&self) -> Option<mat::Identity> {
        Some(mat::Identity::new())
    }

    fn invert(&mut self) -> bool {
        true
    }
}

impl<T: Clone> Mul<T, T> for mat::Identity {
    #[inline]
    fn mul(&self, other: &T) -> T {
        other.clone()
    }
}

impl Transpose for mat::Identity {
    #[inline]
    fn transposed(&self) -> mat::Identity {
        mat::Identity::new()
    }

    #[inline]
    fn transpose(&mut self) {
    }
}

impl<V: Zero> Translation<V> for mat::Identity {
    #[inline]
    fn translation(&self) -> V {
        Zero::zero()
    }

    #[inline]
    fn inv_translation(&self) -> V {
        Zero::zero()
    }

    #[inline]
    fn translate_by(&mut self, _: &V) {
        fail!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn translated(&self, _: &V) -> mat::Identity {
        fail!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn set_translation(&mut self, _: V) {
        fail!("Attempted to translate the identity matrix.")
    }
}

impl<V: Clone> Translate<V> for mat::Identity {
    #[inline]
    fn translate(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inv_translate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<V: Zero> Rotation<V> for mat::Identity {
    #[inline]
    fn rotation(&self) -> V {
        Zero::zero()
    }

    #[inline]
    fn inv_rotation(&self) -> V {
        Zero::zero()
    }

    #[inline]
    fn rotate_by(&mut self, _: &V) {
        fail!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn rotated(&self, _: &V) -> mat::Identity {
        fail!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn set_rotation(&mut self, _: V) {
        fail!("Attempted to rotate the identity matrix.")
    }
}

impl<V: Clone> Rotate<V> for mat::Identity {
    #[inline]
    fn rotate(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inv_rotate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<M: One> Transformation<M> for mat::Identity {
    #[inline]
    fn transformation(&self) -> M {
        One::one()
    }

    #[inline]
    fn inv_transformation(&self) -> M {
        One::one()
    }

    #[inline]
    fn transform_by(&mut self, _: &M) {
        fail!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn transformed(&self, _: &M) -> mat::Identity {
        fail!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn set_transformation(&mut self, _: M) {
        fail!("Attempted to transform the identity matrix.")
    }
}

impl<V: Clone> Transform<V> for mat::Identity {
    #[inline]
    fn transform(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inv_transform(&self, v: &V) -> V {
        v.clone()
    }
}
