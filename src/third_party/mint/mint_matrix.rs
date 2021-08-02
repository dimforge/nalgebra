use std::convert::{AsMut, AsRef, From, Into};
use std::mem;
use std::ptr;

use crate::base::allocator::Allocator;
use crate::base::dimension::{U1, U2, U3, U4};
use crate::base::storage::{IsContiguous, RawStorage, RawStorageMut};
use crate::base::{DefaultAllocator, Matrix, OMatrix, Scalar};

macro_rules! impl_from_into_mint_1D(
    ($($NRows: ident => $VT:ident [$SZ: expr]);* $(;)*) => {$(
        impl<T> From<mint::$VT<T>> for OMatrix<T, $NRows, U1>
        where T: Scalar,
              DefaultAllocator: Allocator<T, $NRows, U1> {
            #[inline]
            fn from(v: mint::$VT<T>) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&v.x, (*res.as_mut_ptr()).data.ptr_mut(), $SZ);

                    res.assume_init()
                }
            }
        }

        impl<T, S> Into<mint::$VT<T>> for Matrix<T, $NRows, U1, S>
        where T: Scalar,
              S: RawStorage<T, $NRows, U1> + IsContiguous {
            #[inline]
            fn into(self) -> mint::$VT<T> {
                // SAFETY: this is OK thanks to the IsContiguous bound.
                unsafe {
                    let mut res: mint::$VT<T> = mem::MaybeUninit::uninit().assume_init();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res.x, $SZ);
                    res
                }
            }
        }

        impl<T, S> AsRef<mint::$VT<T>> for Matrix<T, $NRows, U1, S>
        where T: Scalar,
              S: RawStorage<T, $NRows, U1> + IsContiguous {
            #[inline]
            fn as_ref(&self) -> &mint::$VT<T> {
                // SAFETY: this is OK thanks to the IsContiguous bound.
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<T, S> AsMut<mint::$VT<T>> for Matrix<T, $NRows, U1, S>
        where T: Scalar,
              S: RawStorageMut<T, $NRows, U1> + IsContiguous {
            #[inline]
            fn as_mut(&mut self) -> &mut mint::$VT<T> {
                // SAFETY: this is OK thanks to the IsContiguous bound.
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);

// Implement for vectors of dimension 2 .. 4.
impl_from_into_mint_1D!(
    U2 => Vector2[2];
    U3 => Vector3[3];
    U4 => Vector4[4];
);

macro_rules! impl_from_into_mint_2D(
    ($(($NRows: ty, $NCols: ty) => $MV:ident{ $($component:ident),* }[$SZRows: expr]);* $(;)*) => {$(
        impl<T> From<mint::$MV<T>> for OMatrix<T, $NRows, $NCols>
        where T: Scalar,
              DefaultAllocator: Allocator<T, $NRows, $NCols> {
            #[inline]
            fn from(m: mint::$MV<T>) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    let mut ptr = (*res.as_mut_ptr()).data.ptr_mut();
                    $(
                        ptr::copy_nonoverlapping(&m.$component.x, ptr, $SZRows);
                        ptr = ptr.offset($SZRows);
                    )*
                    let _ = ptr;
                    res.assume_init()
                }
            }
        }

        impl<T> Into<mint::$MV<T>> for OMatrix<T, $NRows, $NCols>
        where T: Scalar,
              DefaultAllocator: Allocator<T, $NRows, $NCols> {
            #[inline]
            fn into(self) -> mint::$MV<T> {
                unsafe {
                    let mut res: mint::$MV<T> = mem::MaybeUninit::uninit().assume_init();
                    let mut ptr = self.data.ptr();
                    $(
                        ptr::copy_nonoverlapping(ptr, &mut res.$component.x, $SZRows);
                        ptr = ptr.offset($SZRows);
                    )*
                    let _ = ptr;
                    res
                }
            }
        }
    )*}
);

// Implement for matrices with shape 2x2 .. 4x4.
impl_from_into_mint_2D!(
    (U2, U2) => ColumnMatrix2{x, y}[2];
    (U2, U3) => ColumnMatrix2x3{x, y, z}[2];
    (U3, U3) => ColumnMatrix3{x, y, z}[3];
    (U3, U4) => ColumnMatrix3x4{x, y, z, w}[3];
    (U4, U4) => ColumnMatrix4{x, y, z, w}[4];
);
