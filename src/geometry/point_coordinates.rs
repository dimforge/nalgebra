use {
    std::{mem, ops::{Deref, DerefMut}},
    crate::{
        base::{
            allocator::Allocator,
            coordinates::{X, XY, XYZ, XYZW, XYZWA, XYZWAB},
            dimension::{U1, U2, U3, U4, U5, U6},
            DefaultAllocator, Scalar
        },
        geometry::Point
    }
};

/*
 *
 * Give coordinates to Point{1 .. 6}
 *
 */

macro_rules! deref_impl(
    ($D: ty, $Target: ident $(, $comps: ident)*) => {
        impl<N: Scalar> Deref for Point<N, $D>
            where DefaultAllocator: Allocator<N, $D> {
            type Target = $Target<N>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { mem::transmute(self) }
            }
        }

        impl<N: Scalar> DerefMut for Point<N, $D>
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
