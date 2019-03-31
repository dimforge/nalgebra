use std::mem;
use std::ops::{Deref, DerefMut};

use alga::general::RealField;

use crate::base::coordinates::IJKW;

use crate::geometry::Quaternion;

impl<N: RealField> Deref for Quaternion<N> {
    type Target = IJKW<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<N: RealField> DerefMut for Quaternion<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
