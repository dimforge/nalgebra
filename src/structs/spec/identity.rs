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
    fn inv_cpy(&self) -> Option<mat::Identity> {
        Some(mat::Identity::new())
    }

    fn inv(&mut self) -> bool {
        true
    }
}

impl<T: Clone> Mul<T, T> for mat::Identity {
    #[inline]
    fn mul(self, other: T) -> T {
        other
    }
}

impl Transpose for mat::Identity {
    #[inline]
    fn transpose_cpy(&self) -> mat::Identity {
        mat::Identity::new()
    }

    #[inline]
    fn transpose(&mut self) {
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
    fn append_translation(&mut self, _: &V) {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn append_translation_cpy(&self, _: &V) -> mat::Identity {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn prepend_translation(&mut self, _: &V) {
        panic!("Attempted to translate the identity matrix.")
    }

    #[inline]
    fn prepend_translation_cpy(&self, _: &V) -> mat::Identity {
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
    fn append_rotation(&mut self, _: &V) {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn append_rotation_cpy(&self, _: &V) -> mat::Identity {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn prepend_rotation(&mut self, _: &V) {
        panic!("Attempted to rotate the identity matrix.")
    }

    #[inline]
    fn prepend_rotation_cpy(&self, _: &V) -> mat::Identity {
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
    fn append_transformation(&mut self, _: &M) {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn append_transformation_cpy(&self, _: &M) -> mat::Identity {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn prepend_transformation(&mut self, _: &M) {
        panic!("Attempted to transform the identity matrix.")
    }

    #[inline]
    fn prepend_transformation_cpy(&self, _: &M) -> mat::Identity {
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
