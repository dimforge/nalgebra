use std::ptr;
use std::mem;
use std::convert::{From, Into, AsRef, AsMut};
use typenum::{self, Cmp, Equal};
use alga::general::{SubsetOf, SupersetOf};

use core::{Scalar, Matrix};
use core::dimension::{Dim, DimName, DimNameMul, DimNameProd, U1, U2, U3, U4, U5, U6};
use core::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};
use core::storage::{Storage, StorageMut, OwnedStorage};
use core::iter::{MatrixIter, MatrixIterMut};
use core::allocator::{OwnedAllocator, SameShapeAllocator};


// FIXME: too bad this won't work allo slice conversions.
impl<N1, N2, R1, C1, R2, C2, SA, SB> SubsetOf<Matrix<N2, R2, C2, SB>> for Matrix<N1, R1, C1, SA>
    where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
          N1: Scalar,
          N2: Scalar + SupersetOf<N1>,
          SA: OwnedStorage<N1, R1, C1>,
          SB: OwnedStorage<N2, R2, C2>,
          SB::Alloc: OwnedAllocator<N2, R2, C2, SB>,
          SA::Alloc: OwnedAllocator<N1, R1, C1, SA> +
                     SameShapeAllocator<N1, R1, C1, R2, C2, SA>,
          ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
    #[inline]
    fn to_superset(&self) -> Matrix<N2, R2, C2, SB> {
        let (nrows, ncols) = self.shape();
        let nrows2 = R2::from_usize(nrows);
        let ncols2 = C2::from_usize(ncols);

        let mut res = unsafe { Matrix::<N2, R2, C2, SB>::new_uninitialized_generic(nrows2, ncols2) };
        for i in 0 .. nrows {
            for j in 0 .. ncols {
                unsafe {
                    *res.get_unchecked_mut(i, j) = N2::from_subset(self.get_unchecked(i, j))
                }
            }
        }

        res
    }

    #[inline]
    fn is_in_subset(m: &Matrix<N2, R2, C2, SB>) -> bool {
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    unsafe fn from_superset_unchecked(m: &Matrix<N2, R2, C2, SB>) -> Self {
        let (nrows2, ncols2) = m.shape();
        let nrows = R1::from_usize(nrows2);
        let ncols = C1::from_usize(ncols2);

        let mut res = Self::new_uninitialized_generic(nrows, ncols);
        for i in 0 .. nrows2 {
            for j in 0 .. ncols2 {
                *res.get_unchecked_mut(i, j) = m.get_unchecked(i, j).to_subset_unchecked()
            }
        }

        res
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> IntoIterator for &'a Matrix<N, R, C, S> {
    type Item     = &'a N;
    type IntoIter = MatrixIter<'a, N, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, S: StorageMut<N, R, C>> IntoIterator for &'a mut Matrix<N, R, C, S> {
    type Item     = &'a mut N;
    type IntoIter = MatrixIterMut<'a, N, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}


macro_rules! impl_from_into_asref_1D(
    ($($Dim: ident => $SZ: expr);* $(;)*) => {$(
        impl<N, R, C, S> From<[N; $SZ]> for Matrix<N, R, C, S>
        where N: Scalar,
              R: DimName + DimNameMul<C>,
              C: DimName,
              S: OwnedStorage<N, R, C>,
              S::Alloc: OwnedAllocator<N, R, C, S>,
              <DimNameProd<R, C> as DimName>::Value: Cmp<typenum::$Dim, Output = Equal> {
            #[inline]
            fn from(arr: [N; $SZ]) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&arr[0], res.data.ptr_mut(), $SZ);

                    res
                }
            }
        }

        impl<N, R, C, S> Into<[N; $SZ]> for Matrix<N, R, C, S>
        where N: Scalar,
              R: DimName + DimNameMul<C>,
              C: DimName,
              S: OwnedStorage<N, R, C>,
              S::Alloc: OwnedAllocator<N, R, C, S>,
              <DimNameProd<R, C> as DimName>::Value: Cmp<typenum::$Dim, Output = Equal> {
            #[inline]
            fn into(self) -> [N; $SZ] {
                unsafe {
                    let mut res: [N; $SZ] = mem::uninitialized();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res[0], $SZ);

                    res
                }
            }
        }

        impl<N, R, C, S> AsRef<[N; $SZ]> for Matrix<N, R, C, S>
        where N: Scalar,
              R: DimName + DimNameMul<C>,
              C: DimName,
              S: OwnedStorage<N, R, C>,
              S::Alloc: OwnedAllocator<N, R, C, S>,
              <DimNameProd<R, C> as DimName>::Value: Cmp<typenum::$Dim, Output = Equal> {
            #[inline]
            fn as_ref(&self) -> &[N; $SZ] {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N, R, C, S> AsMut<[N; $SZ]> for Matrix<N, R, C, S>
        where N: Scalar,
              R: DimName + DimNameMul<C>,
              C: DimName,
              S: OwnedStorage<N, R, C>,
              S::Alloc: OwnedAllocator<N, R, C, S>,
              <DimNameProd<R, C> as DimName>::Value: Cmp<typenum::$Dim, Output = Equal> {
            #[inline]
            fn as_mut(&mut self) -> &mut [N; $SZ] {
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);

// Implement for vectors of dimension 1 .. 36 and matrices with shape 1x1 .. 6x6.
impl_from_into_asref_1D!(
    U1  => 1;  U2  => 2;  U3  => 3;  U4  => 4;  U5  => 5;  U6  => 6;
    U7  => 7;  U8  => 8;  U9  => 9;  U10 => 10; U11 => 11; U12 => 12;
    U13 => 13; U14 => 14; U15 => 15; U16 => 16; U17 => 17; U18 => 18;
    U19 => 19; U20 => 20; U21 => 21; U22 => 22; U23 => 23; U24 => 24;
    U25 => 25; U26 => 26; U27 => 27; U28 => 28; U29 => 29; U30 => 30;
    U31 => 31; U32 => 32; U33 => 33; U34 => 34; U35 => 35; U36 => 36;
);



macro_rules! impl_from_into_asref_2D(
    ($(($NRows: ty, $NCols: ty) => ($SZRows: expr, $SZCols: expr));* $(;)*) => {$(
        impl<N, S> From<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: OwnedStorage<N, $NRows, $NCols>,
              S::Alloc: OwnedAllocator<N, $NRows, $NCols, S> {
            #[inline]
            fn from(arr: [[N; $SZRows]; $SZCols]) -> Self {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    ptr::copy_nonoverlapping(&arr[0][0], res.data.ptr_mut(), $SZRows * $SZCols);

                    res
                }
            }
        }

        impl<N, S> Into<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: OwnedStorage<N, $NRows, $NCols>,
              S::Alloc: OwnedAllocator<N, $NRows, $NCols, S> {
            #[inline]
            fn into(self) -> [[N; $SZRows]; $SZCols] {
                unsafe {
                    let mut res: [[N; $SZRows]; $SZCols] = mem::uninitialized();
                    ptr::copy_nonoverlapping(self.data.ptr(), &mut res[0][0], $SZRows * $SZCols);

                    res
                }
            }
        }

        impl<N, S> AsRef<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: OwnedStorage<N, $NRows, $NCols>,
              S::Alloc: OwnedAllocator<N, $NRows, $NCols, S> {
            #[inline]
            fn as_ref(&self) -> &[[N; $SZRows]; $SZCols] {
                unsafe {
                    mem::transmute(self.data.ptr())
                }
            }
        }

        impl<N, S> AsMut<[[N; $SZRows]; $SZCols]> for Matrix<N, $NRows, $NCols, S>
        where N: Scalar,
              S: OwnedStorage<N, $NRows, $NCols>,
              S::Alloc: OwnedAllocator<N, $NRows, $NCols, S> {
            #[inline]
            fn as_mut(&mut self) -> &mut [[N; $SZRows]; $SZCols] {
                unsafe {
                    mem::transmute(self.data.ptr_mut())
                }
            }
        }
    )*}
);


// Implement for matrices with shape 1x1 .. 4x4.
impl_from_into_asref_2D!(
    (U1, U1) => (1, 1); (U1, U2) => (1, 2); (U1, U3) => (1, 3); (U1, U4) => (1, 4); (U1, U5) => (1, 5); (U1, U6) => (1, 6);
    (U2, U1) => (2, 1); (U2, U2) => (2, 2); (U2, U3) => (2, 3); (U2, U4) => (2, 4); (U2, U5) => (2, 5); (U2, U6) => (2, 6);
    (U3, U1) => (3, 1); (U3, U2) => (3, 2); (U3, U3) => (3, 3); (U3, U4) => (3, 4); (U3, U5) => (3, 5); (U3, U6) => (3, 6);
    (U4, U1) => (4, 1); (U4, U2) => (4, 2); (U4, U3) => (4, 3); (U4, U4) => (4, 4); (U4, U5) => (4, 5); (U4, U6) => (4, 6);
    (U5, U1) => (5, 1); (U5, U2) => (5, 2); (U5, U3) => (5, 3); (U5, U4) => (5, 4); (U5, U5) => (5, 5); (U5, U6) => (5, 6);
    (U6, U1) => (6, 1); (U6, U2) => (6, 2); (U6, U3) => (6, 3); (U6, U4) => (6, 4); (U6, U5) => (6, 5); (U6, U6) => (6, 6);
);
