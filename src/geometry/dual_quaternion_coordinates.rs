// use std::mem;
// use std::ops::{Deref, DerefMut};

// use alga::general::Real;

// use base::coordinates::IJKW;

// use geometry::DualQuaternion;

// impl<N: Real> Deref for DualQuaternion<N> {
//     type Target = IJKW<N>;

//     #[inline]
//     fn deref(&self) -> &Self::Target {
//         unsafe { mem::transmute(self) }
//     }
// }

// impl<N: Real> DerefMut for DualQuaternion<N> {
//     #[inline]
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         unsafe { mem::transmute(self) }
//     }
// }
