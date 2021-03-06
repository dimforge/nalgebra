use std::convert::{AsMut, AsRef, From, Into};
use std::mem;
use std::ptr;

use crate::base::allocator::Allocator;
use crate::base::dimension::{U1, U2, U3, U4};
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixMN, Scalar};

macro_rules! impl_from_into_mint_1D(
    ($($NRows: ident => $VT:ident [$SZ: expr]);* $(;)*) => {$(
        impl<N> From<mint::$VT<N>> for MatrixMN<N, $NRows, U1>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, U1> {
            #[inline]
            fn from(v: mint::$VT<N>) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&v.x, (*res.as_mut_ptr()).data.ptr_mut(), $SZ);

                    res.assume_init()
                }
            }
        }

        impl<N, S> Into<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, U1> {
            #[inline]
            fn into(self) -> mint::$VT<N> {
                unsafe {
                    let mut res: mint::$VT<N> = mem::MaybeUninit::uninit().assume_init();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res.x, $SZ);
                    res
                }
            }
        }

        impl<N, S> AsRef<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorage<N, $NRows, U1> {
            #[inline]
            fn as_ref(&self) -> &mint::$VT<N> {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N, S> AsMut<mint::$VT<N>> for Matrix<N, $NRows, U1, S>
        where N: Scalar,
              S: ContiguousStorageMut<N, $NRows, U1> {
            #[inline]
            fn as_mut(&mut self) -> &mut mint::$VT<N> {
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
        impl<N> From<mint::$MV<N>> for MatrixMN<N, $NRows, $NCols>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn from(m: mint::$MV<N>) -> Self {
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

        impl<N> Into<mint::$MV<N>> for MatrixMN<N, $NRows, $NCols>
        where N: Scalar,
              DefaultAllocator: Allocator<N, $NRows, $NCols> {
            #[inline]
            fn into(self) -> mint::$MV<N> {
                unsafe {
                    let mut res: mint::$MV<N> = mem::MaybeUninit::uninit().assume_init();
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
