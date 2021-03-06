use crate::base::storage::{Storage, StorageMut};
use crate::{Point, Scalar, VectorN, U2, U3};
use std::convert::{AsMut, AsRef};

macro_rules! impl_from_into_mint_1D(
    ($($NRows: ident => $PT:ident, $VT:ident [$SZ: expr]);* $(;)*) => {$(
        impl<N> From<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn from(p: mint::$PT<N>) -> Self {
                Self {
                    coords: VectorN::from(mint::$VT::from(p)),
                }
            }
        }

        impl<N> Into<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn into(self) -> mint::$PT<N> {
                let mint_vec: mint::$VT<N> = self.coords.into();
                mint::$PT::from(mint_vec)
            }
        }

        impl<N> AsRef<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn as_ref(&self) -> &mint::$PT<N> {
                unsafe {
                    &*(self.coords.data.ptr() as *const mint::$PT<N>)
                }
            }
        }

        impl<N> AsMut<mint::$PT<N>> for Point<N, $NRows>
        where N: Scalar {
            #[inline]
            fn as_mut(&mut self) -> &mut mint::$PT<N> {
                unsafe {
                    &mut *(self.coords.data.ptr_mut() as *mut mint::$PT<N>)
                }
            }
        }
    )*}
);

// Implement for points of dimension 2, 3.
impl_from_into_mint_1D!(
    U2 => Point2, Vector2[2];
    U3 => Point3, Vector3[3];
);
