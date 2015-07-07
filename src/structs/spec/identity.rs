use std::ops::Mul;
use num::One;
use structs::mat;
use traits::operations::{Inv, Transpose};
use traits::geometry::{Translate, Rotate, Transform, AbsoluteRotate};

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
