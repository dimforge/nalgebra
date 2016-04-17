use std::ops::Mul;
use num::One;
use structs::matrix::Identity;
use traits::operations::{Inverse, Transpose};
use traits::geometry::{Translate, Rotate, Transform, AbsoluteRotate};

impl One for Identity {
    #[inline]
    fn one() -> Identity {
        Identity::new()
    }
}

impl Inverse for Identity {
    fn inverse(&self) -> Option<Identity> {
        Some(Identity::new())
    }

    fn inverse_mut(&mut self) -> bool {
        true
    }
}

impl<T: Clone> Mul<T> for Identity {
    type Output = T;

    #[inline]
    fn mul(self, other: T) -> T {
        other
    }
}

impl Transpose for Identity {
    #[inline]
    fn transpose(&self) -> Identity {
        Identity::new()
    }

    #[inline]
    fn transpose_mut(&mut self) {
    }
}

impl<V: Clone> Translate<V> for Identity {
    #[inline]
    fn translate(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inverse_translate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<V: Clone> Rotate<V> for Identity {
    #[inline]
    fn rotate(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inverse_rotate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<V: Clone> AbsoluteRotate<V> for Identity {
    #[inline]
    fn absolute_rotate(&self, v: &V) -> V {
        v.clone()
    }
}

impl<V: Clone> Transform<V> for Identity {
    #[inline]
    fn transform(&self, v: &V) -> V {
        v.clone()
    }

    #[inline]
    fn inverse_transform(&self, v: &V) -> V {
        v.clone()
    }
}
