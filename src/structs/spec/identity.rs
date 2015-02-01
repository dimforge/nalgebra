use std::ops::Mul;
use structs::mat;
use traits::operations::{Inv, Transpose};
use traits::structure::{Zero, One};
use traits::geometry::{Translation, Translate, Rotation, Rotate, Transformation, Transform,
                       AbsoluteRotate};

impl One for mat::Identity {
    #[inline]
    fn one() -> mat::Identity {
        mat::Identity::new()
    }
}

impl Inv for mat::Identity {
    fn inv(&self) -> Option<mat::Identity> {
        Some(mat::Identity::new())
    }

    fn inv_mut(&mut self) -> bool {
        true
    }
}

impl<T: Clone> Mul<T> for mat::Identity {
    type Output = T;

    #[inline]
    fn mul(self, other: T) -> T {
        other
    }
}

impl Transpose for mat::Identity {
    #[inline]
    fn transpose(&self) -> mat::Identity {
        mat::Identity::new()
    }

    #[inline]
    fn transpose_mut(&mut self) {
    }
}

impl<V: Zero> Translation<V> for mat::Identity {
    #[inline]
    fn translation(&self) -> V {
        ::zero()
    }

    #[inline]
    fn inv_translation(&self) -> V {
        ::zero()
    }

    #[inline]
    fn append_translation_mut(&mut self, _: &V) {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn append_translation(&self, _: &V) -> mat::Identity {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn prepend_translation_mut(&mut self, _: &V) {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn prepend_translation(&self, _: &V) -> mat::Identity {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn set_translation(&mut self, _: V) {
        panic!("Attempted to translate the identity matrix.")
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
        ::zero()
    }

    #[inline]
    fn inv_rotation(&self) -> V {
        ::zero()
    }

    #[inline]
    fn append_rotation_mut(&mut self, _: &V) {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn append_rotation(&self, _: &V) -> mat::Identity {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn prepend_rotation_mut(&mut self, _: &V) {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn prepend_rotation(&self, _: &V) -> mat::Identity {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn set_rotation(&mut self, _: V) {
        panic!("Attempted to rotate the identity matrix.")
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

impl<V: Clone> AbsoluteRotate<V> for mat::Identity {
    #[inline]
    fn absolute_rotate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<M: One> Transformation<M> for mat::Identity {
    #[inline]
    fn transformation(&self) -> M {
        ::one()
    }

    #[inline]
    fn inv_transformation(&self) -> M {
        ::one()
    }

    #[inline]
    fn append_transformation_mut(&mut self, _: &M) {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn append_transformation(&self, _: &M) -> mat::Identity {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn prepend_transformation_mut(&mut self, _: &M) {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn prepend_transformation(&self, _: &M) -> mat::Identity {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn set_transformation(&mut self, _: M) {
        panic!("Attempted to transform the identity matrix.")
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
