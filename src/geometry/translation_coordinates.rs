use std::mem;
use std::ops::{Deref, DerefMut};

use base::allocator::Allocator;
use base::coordinates::{X, XY, XYZ, XYZW, XYZWA, XYZWAB};
use base::dimension::{U1, U2, U3, U4, U5, U6};
use base::{DefaultAllocator, Scalar};

use geometry::Translation;

/*
 *
 * Give coordinates to Translation{1 .. 6}
 *
 */

macro_rules! deref_impl(
    ($D: ty, $Target: ident $(, $comps: ident)*) => {
        impl<N: Scalar> Deref for Translation<N, $D>
            where DefaultAllocator: Allocator<N, $D> {
            type Target = $Target<N>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { mem::transmute(self) }
            }
        }

        impl<N: Scalar> DerefMut for Translation<N, $D>
            where DefaultAllocator: Allocator<N, $D> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { mem::transmute(self) }
            }
        }
    }
);

deref_impl!(U1, X, x);
deref_impl!(U2, XY, x, y);
deref_impl!(U3, XYZ, x, y, z);
deref_impl!(U4, XYZW, x, y, z, w);
deref_impl!(U5, XYZWA, x, y, z, w, a);
deref_impl!(U6, XYZWAB, x, y, z, w, a, b);
