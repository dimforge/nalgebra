use std::mem;
use std::ops::{Deref, DerefMut};

use simba::simd::SimdRealField;

use crate::base::coordinates::IJKW;

use crate::geometry::Quaternion;

impl<N: SimdRealField> Deref for Quaternion<N> {
    type Target = IJKW<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<N: SimdRealField> DerefMut for Quaternion<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
