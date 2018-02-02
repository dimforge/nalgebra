use core::{MatrixSliceMN, MatrixSliceMutMN, Scalar};
use core::dimension::{Dim, DimName, Dynamic, U1};
use core::matrix_slice::{SliceStorage, SliceStorageMut};

/*
 *
 * Slice constructors.
 *
 */
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMN<'a, N, R, C, RStride, CStride> {
    #[inline]
    pub unsafe fn new_with_strides_generic_unchecked(
        data: &'a [N],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorage::from_raw_parts(
            data.as_ptr().offset(start as isize),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    #[inline]
    pub fn new_with_strides_generic(
        data: &'a [N],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        unsafe { Self::new_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride) }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMutMN<'a, N, R, C, RStride, CStride> {
    #[inline]
    pub unsafe fn new_with_strides_generic_mut_unchecked(
        data: &'a mut [N],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorageMut::from_raw_parts(
            data.as_mut_ptr().offset(start as isize),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    #[inline]
    pub fn new_with_strides_generic_mut(
        data: &'a mut [N],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        unsafe {
            Self::new_with_strides_generic_mut_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim> MatrixSliceMN<'a, N, R, C> {
    #[inline]
    pub unsafe fn new_generic_unchecked(data: &'a [N], start: usize, nrows: R, ncols: C) -> Self {
        Self::new_with_strides_generic_unchecked(data, start, nrows, ncols, U1, nrows)
    }

    #[inline]
    pub fn new_generic(data: &'a [N], nrows: R, ncols: C) -> Self {
        Self::new_with_strides_generic(data, nrows, ncols, U1, nrows)
    }
}

impl<'a, N: Scalar, R: Dim, C: Dim> MatrixSliceMutMN<'a, N, R, C> {
    #[inline]
    pub unsafe fn new_generic_mut_unchecked(
        data: &'a mut [N],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        Self::new_with_strides_generic_mut_unchecked(data, start, nrows, ncols, U1, nrows)
    }

    #[inline]
    pub fn new_generic_mut(data: &'a mut [N], nrows: R, ncols: C) -> Self {
        Self::new_with_strides_generic_mut(data, nrows, ncols, U1, nrows)
    }
}

macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, N: Scalar, $($DimIdent: $DimBound),*> MatrixSliceMN<'a, N, $($Dims),*> {
            #[inline]
            pub fn new(data: &'a [N], $($args: usize),*) -> Self {
                Self::new_generic(data, $($gargs),*)
            }

            #[inline]
            pub unsafe fn new_unchecked(data: &'a [N], start: usize, $($args: usize),*) -> Self {
                Self::new_generic_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, N: Scalar, $($DimIdent: $DimBound, )*> MatrixSliceMN<'a, N, $($Dims,)* Dynamic, Dynamic> {
            #[inline]
            pub fn new_with_strides(data: &'a [N], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::new_with_strides_generic(data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            #[inline]
            pub unsafe fn new_with_strides_unchecked(data: &'a [N], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::new_with_strides_generic_unchecked(data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<N, ..., S>
                   => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                   R::name(), C::name();         // Arguments for `_generic` constructors.
                   ); // Arguments for non-generic constructors.

impl_constructors!(R, Dynamic;
                   => R: DimName;
                   R::name(), Dynamic::new(ncols);
                   ncols);

impl_constructors!(Dynamic, C;
                   => C: DimName;
                   Dynamic::new(nrows), C::name();
                   nrows);

impl_constructors!(Dynamic, Dynamic;
                   ;
                   Dynamic::new(nrows), Dynamic::new(ncols);
                   nrows, ncols);

macro_rules! impl_constructors_mut(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, N: Scalar, $($DimIdent: $DimBound),*> MatrixSliceMutMN<'a, N, $($Dims),*> {

            #[inline]
            pub fn new(data: &'a mut [N], $($args: usize),*) -> Self {
                Self::new_generic_mut(data, $($gargs),*)
            }

            #[inline]
            pub unsafe fn new_unchecked(data: &'a mut [N], start: usize, $($args: usize),*) -> Self {
                Self::new_generic_mut_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, N: Scalar, $($DimIdent: $DimBound, )*> MatrixSliceMutMN<'a, N, $($Dims,)* Dynamic, Dynamic> {
            #[inline]
            pub fn new_with_strides(data: &'a mut [N], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::new_with_strides_generic_mut(
                    data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            #[inline]
            pub unsafe fn new_with_strides_unchecked(data: &'a mut [N], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::new_with_strides_generic_mut_unchecked(
                    data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors_mut!(R, C;                         // Arguments for Matrix<N, ..., S>
                       => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                       R::name(), C::name();         // Arguments for `_generic` constructors.
                       ); // Arguments for non-generic constructors.

impl_constructors_mut!(R, Dynamic;
                       => R: DimName;
                       R::name(), Dynamic::new(ncols);
                       ncols);

impl_constructors_mut!(Dynamic, C;
                       => C: DimName;
                       Dynamic::new(nrows), C::name();
                       nrows);

impl_constructors_mut!(Dynamic, Dynamic;
                       ;
                       Dynamic::new(nrows), Dynamic::new(ncols);
                       nrows, ncols);
