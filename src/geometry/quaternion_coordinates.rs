use std::mem;
use std::ops::{Deref, DerefMut};

use simba::simd::SimdValue;

use crate::base::coordinates::IJKW;
use crate::Scalar;

use crate::geometry::Quaternion;

impl<N: Scalar + SimdValue> Deref for Quaternion<N> {
    type Target = IJKW<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<N: Scalar + SimdValue> DerefMut for Quaternion<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
