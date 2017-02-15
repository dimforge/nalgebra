use std::mem;
use std::ops::{Deref, DerefMut};

use alga::general::Real;

use core::coordinates::IJKW;
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;
use core::dimension::{U1, U4};

use geometry::QuaternionBase;


impl<N, S> Deref for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    type Target = IJKW<N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<N, S> DerefMut for QuaternionBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U1>,
          S::Alloc: OwnedAllocator<N, U4, U1, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
