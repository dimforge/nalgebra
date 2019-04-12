use {
    alga::general::RealField,
    std::{mem, ops::{Deref, DerefMut}},
    crate::{
        base::coordinates::IJKW,
        geometry::Quaternion
    }
};

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
