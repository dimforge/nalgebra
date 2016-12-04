use std::marker::PhantomData;

use core::{Scalar, Matrix};
use core::dimension::{Dim, DimName, Dynamic, DimMul, DimProd, U1};
use core::iter::MatrixIter;
use core::storage::{Storage, StorageMut, Owned};
use core::allocator::Allocator;

macro_rules! slice_storage_impl(
    ($Storage: ident as $SRef: ty; $T: ident.$get_addr: ident ($Ptr: ty as $Ref: ty)) => {
        pub struct $T<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim, Alloc> {
            ptr:       $Ptr,
            shape:     (R, C),
            strides:   (RStride, CStride),
            _phantoms: PhantomData<($Ref, Alloc)>,
        }

        // Dynamic and () are arbitrary. It's just to be able to call the constructors with
        // `Slice::`
        impl<'a, N: Scalar, R: Dim, C: Dim> $T<'a, N, R, C, Dynamic, Dynamic, ()> {
            /// Create a new matrix slice without bound checking.
            #[inline]
            pub unsafe fn new_unchecked<RStor, CStor, S>(storage: $SRef, start: (usize, usize), shape: (R, C))
                -> $T<'a, N, R, C, S::RStride, S::CStride, S::Alloc>
                where RStor: Dim,
                      CStor: Dim,
                      S:     $Storage<N, RStor, CStor> {

                let strides = storage.strides();
                $T::new_with_strides_unchecked(storage, start, shape, strides)
            }

            /// Create a new matrix slice without bound checking.
            #[inline]
            pub unsafe fn new_with_strides_unchecked<S, RStor, CStor, RStride, CStride>(storage: $SRef,
                                                                                        start:   (usize, usize),
                                                                                        shape:   (R, C),
                                                                                        strides: (RStride, CStride))
                -> $T<'a, N, R, C, RStride, CStride, S::Alloc>
                where RStor: Dim,
                      CStor: Dim,
                      S: $Storage<N, RStor, CStor>,
                      RStride: Dim,
                      CStride: Dim {

                $T {
                    ptr:       storage.$get_addr(start.0, start.1),
                    shape:     shape,
                    strides:   (strides.0, strides.1),
                    _phantoms: PhantomData
                }
            }
        }
    }
);

slice_storage_impl!(Storage as &'a S; SliceStorage.get_address_unchecked(*const N as &'a N));
slice_storage_impl!(StorageMut as &'a mut S; SliceStorageMut.get_address_unchecked_mut(*mut N as &'a mut N));


impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim, Alloc> Copy
for SliceStorage<'a, N, R, C, RStride, CStride, Alloc> { }

impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim, Alloc> Clone
for SliceStorage<'a, N, R, C, RStride, CStride, Alloc> {
    #[inline]
    fn clone(&self) -> Self {
        SliceStorage {
            ptr:       self.ptr,
            shape:     self.shape,
            strides:   self.strides,
            _phantoms: PhantomData,
        }
    }
}

macro_rules! storage_impl(
    ($($T: ident),* $(,)*) => {$(
        unsafe impl<'a, N, R: Dim, C: Dim, RStride: Dim, CStride: Dim, Alloc> Storage<N, R, C>
            for $T<'a, N, R, C, RStride, CStride, Alloc>
            where N:     Scalar,
                  Alloc: Allocator<N, R, C> {

            type RStride = RStride;
            type CStride = CStride;
            type Alloc   = Alloc;

            #[inline]
            fn into_owned(self) -> Owned<N, R, C, Self::Alloc> {
                self.clone_owned()
            }

            #[inline]
            fn clone_owned(&self) -> Owned<N, R, C, Self::Alloc> {
                let (nrows, ncols) = self.shape();
                let it = MatrixIter::new(self).cloned();
                Alloc::allocate_from_iterator(nrows, ncols, it)
            }

            #[inline]
            fn ptr(&self) -> *const N {
                self.ptr
            }

            #[inline]
            fn shape(&self) -> (R, C) {
                self.shape
            }

            #[inline]
            fn strides(&self) -> (Self::RStride, Self::CStride) {
                self.strides
            }
        }
    )*}
);

storage_impl!(SliceStorage, SliceStorageMut);

unsafe impl<'a, N, R: Dim, C: Dim, RStride: Dim, CStride: Dim, Alloc> StorageMut<N, R, C>
    for SliceStorageMut<'a, N, R, C, RStride, CStride, Alloc>
    where N: Scalar,
          Alloc: Allocator<N, R, C> {
    #[inline]
    fn ptr_mut(&mut self) -> *mut N {
        self.ptr
    }
}


impl<N: Scalar, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    #[inline]
    fn assert_slice_index(&self, start: (usize, usize), shape: (usize, usize), steps: (usize, usize)) {
        let my_shape = self.shape();
        assert!(start.0 + (shape.0 - 1) * steps.0 <= my_shape.0, "Matrix slicing out of bounds.");
        assert!(start.1 + (shape.1 - 1) * steps.1 <= my_shape.1, "Matrix slicing out of bounds.");
    }
}


macro_rules! matrix_slice_impl(
    ($me: ident: $Me: ty, $MatrixSlice: ident, $SliceStorage: ident, $Storage: ident, $data: expr;
     $row: ident,
     $rows: ident,
     $rows_with_step: ident,
     $fixed_rows: ident,
     $fixed_rows_with_step: ident,
     $rows_generic: ident,
     $column: ident,
     $columns: ident,
     $columns_with_step: ident,
     $fixed_columns: ident,
     $fixed_columns_with_step: ident,
     $columns_generic: ident,
     $slice: ident,
     $slice_with_steps: ident,
     $fixed_slice: ident,
     $fixed_slice_with_steps: ident,
     $generic_slice: ident,
     $generic_slice_with_steps: ident) => {
        /// A matrix slice.
        pub type $MatrixSlice<'a, N, R, C, RStride, CStride, Alloc>
        = Matrix<N, R, C, $SliceStorage<'a, N, R, C, RStride, CStride, Alloc>>;

        impl<N: Scalar, R: Dim, C: Dim, S: $Storage<N, R, C>> Matrix<N, R, C, S> {
            /*
             *
             * Row slicing.
             *
             */
            /// Returns a slice containing the i-th column of this matrix.
            #[inline]
            pub fn $row($me: $Me, i: usize) -> $MatrixSlice<N, U1, C, S::RStride, S::CStride, S::Alloc> {
                $me.$fixed_rows::<U1>(i)
            }

            /// Extracts from this matrix a set of consecutive rows.
            #[inline]
            pub fn $rows($me: $Me, first_row: usize, nrows: usize)
                -> $MatrixSlice<N, Dynamic, C, S::RStride, S::CStride, S::Alloc> {

                let my_shape = $me.data.shape();
                $me.assert_slice_index((first_row, 0), (nrows, my_shape.1.value()), (1, 1));
                let shape = (Dynamic::new(nrows), my_shape.1);

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, (first_row, 0), shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /// Extracts from this matrix a set of consecutive rows regularly spaced by `step` rows.
            #[inline]
            pub fn $rows_with_step($me: $Me, first_row: usize, nrows: usize, step: usize)
                -> $MatrixSlice<N, Dynamic, C, Dynamic, S::CStride, S::Alloc> {

                $me.$rows_generic(first_row, Dynamic::new(nrows), Dynamic::new(step))
            }

            /// Extracts a compile-time number of consecutive rows from this matrix.
            #[inline]
            pub fn $fixed_rows<RSlice>($me: $Me, first_row: usize)
                -> $MatrixSlice<N, RSlice, C, S::RStride, S::CStride, S::Alloc>
                where RSlice: DimName {

                let my_shape = $me.data.shape();
                $me.assert_slice_index((first_row, 0), (RSlice::dim(), my_shape.1.value()), (1, 1));
                let shape = (RSlice::name(), my_shape.1);

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, (first_row, 0), shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /// Extracts from this matrix a compile-time number of rows regularly spaced by `step` rows.
            #[inline]
            pub fn $fixed_rows_with_step<RSlice>($me: $Me, first_row: usize, step: usize)
                -> $MatrixSlice<N, RSlice, C, Dynamic, S::CStride, S::Alloc>
                where RSlice: DimName {

                $me.$rows_generic(first_row, RSlice::name(), Dynamic::new(step))
            }

            /// Extracts from this matrix `nrows` rows regularly spaced by `step` rows. Both argument may
            /// or may not be values known at compile-time.
            #[inline]
            pub fn $rows_generic<RSlice, RStep>($me: $Me, row_start: usize, nrows: RSlice, step: RStep)
                -> $MatrixSlice<N, RSlice, C, DimProd<RStep, S::RStride>, S::CStride, S::Alloc>
                where RSlice: Dim,
                      RStep: DimMul<S::RStride> {

                let my_shape   = $me.data.shape();
                let my_strides = $me.data.strides();
                $me.assert_slice_index((row_start, 0), (nrows.value(), my_shape.1.value()), (step.value(), 1));

                let strides = (step.mul(my_strides.0), my_strides.1);
                let shape   = (nrows, my_shape.1);

                unsafe {
                    let data = $SliceStorage::new_with_strides_unchecked($data, (row_start, 0), shape, strides);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /*
             *
             * Column slicing.
             *
             */
            /// Returns a slice containing the i-th column of this matrix.
            #[inline]
            pub fn $column($me: $Me, i: usize) -> $MatrixSlice<N, R, U1, S::RStride, S::CStride, S::Alloc> {
                $me.$fixed_columns::<U1>(i)
            }

            /// Extracts from this matrix a set of consecutive columns.
            #[inline]
            pub fn $columns($me: $Me, first_col: usize, ncols: usize)
                -> $MatrixSlice<N, R, Dynamic, S::RStride, S::CStride, S::Alloc> {

                let my_shape = $me.data.shape();
                $me.assert_slice_index((0, first_col), (my_shape.0.value(), ncols), (1, 1));
                let shape = (my_shape.0, Dynamic::new(ncols));

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, (0, first_col), shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /// Extracts from this matrix a set of consecutive columns regularly spaced by `step` columns.
            #[inline]
            pub fn $columns_with_step($me: $Me, first_col: usize, ncols: usize, step: usize)
                -> $MatrixSlice<N, R, Dynamic, S::RStride, Dynamic, S::Alloc> {

                $me.$columns_generic(first_col, Dynamic::new(ncols), Dynamic::new(step))
            }

            /// Extracts a compile-time number of consecutive columns from this matrix.
            #[inline]
            pub fn $fixed_columns<CSlice>($me: $Me, first_col: usize)
                -> $MatrixSlice<N, R, CSlice, S::RStride, S::CStride, S::Alloc>
                where CSlice: DimName {

                let my_shape = $me.data.shape();
                $me.assert_slice_index((0, first_col), (my_shape.0.value(), CSlice::dim()), (1, 1));
                let shape = (my_shape.0, CSlice::name());

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, (0, first_col), shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /// Extracts from this matrix a compile-time number of columns regularly spaced by `step`
            /// columns.
            #[inline]
            pub fn $fixed_columns_with_step<CSlice>($me: $Me, first_col: usize, step: usize)
                -> $MatrixSlice<N, R, CSlice, S::RStride, Dynamic, S::Alloc>
                where CSlice: DimName {

                $me.$columns_generic(first_col, CSlice::name(), Dynamic::new(step))
            }

            /// Extracts from this matrix `ncols` columns regularly spaced by `step` columns. Both argument may
            /// or may not be values known at compile-time.
            #[inline]
            pub fn $columns_generic<CSlice, CStep>($me: $Me, first_col: usize, ncols: CSlice, step: CStep)
                -> $MatrixSlice<N, R, CSlice, S::RStride, DimProd<CStep, S::CStride>, S::Alloc>
                where CSlice: Dim,
                      CStep: DimMul<S::CStride> {

                let my_shape   = $me.data.shape();
                let my_strides = $me.data.strides();

                $me.assert_slice_index((0, first_col), (my_shape.0.value(), ncols.value()), (1, step.value()));

                let strides = (my_strides.0, step.mul(my_strides.1));
                let shape   = (my_shape.0, ncols);

                unsafe {
                    let data = $SliceStorage::new_with_strides_unchecked($data, (0, first_col), shape, strides);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /*
             *
             * General slicing.
             *
             */
            /// Slices this matrix starting at its component `(irow, icol)` and with `(nrows, ncols)`
            /// consecutive elements.
            #[inline]
            pub fn $slice($me: $Me, start: (usize, usize), shape: (usize, usize))
                -> $MatrixSlice<N, Dynamic, Dynamic, S::RStride, S::CStride, S::Alloc> {

                $me.assert_slice_index(start, shape, (1, 1));
                let shape = (Dynamic::new(shape.0), Dynamic::new(shape.1));

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, start, shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            #[inline]
            pub fn $slice_with_steps($me: $Me, start: (usize, usize), shape: (usize, usize), steps: (usize, usize))
                -> $MatrixSlice<N, Dynamic, Dynamic, Dynamic, Dynamic, S::Alloc> {
                let shape = (Dynamic::new(shape.0), Dynamic::new(shape.1));
                let steps = (Dynamic::new(steps.0), Dynamic::new(steps.1));

                $me.$generic_slice_with_steps(start, shape, steps)
            }

            /// Slices this matrix starting at its component `(irow, icol)` and with `(R::dim(),
            /// CSlice::dim())` consecutive components.
            #[inline]
            pub fn $fixed_slice<RSlice, CSlice>($me: $Me, irow: usize, icol: usize)
                -> $MatrixSlice<N, RSlice, CSlice, S::RStride, S::CStride, S::Alloc>
                where RSlice: DimName,
                      CSlice: DimName {

                $me.assert_slice_index((irow, icol), (RSlice::dim(), CSlice::dim()), (1, 1));
                let shape = (RSlice::name(), CSlice::name());

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, (irow, icol), shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            #[inline]
            pub fn $fixed_slice_with_steps<RSlice, CSlice>($me: $Me, start: (usize, usize), steps: (usize, usize))
                -> $MatrixSlice<N, RSlice, CSlice, Dynamic, Dynamic, S::Alloc>
                where RSlice: DimName,
                      CSlice: DimName {
                let shape = (RSlice::name(), CSlice::name());
                let steps = (Dynamic::new(steps.0), Dynamic::new(steps.1));
                $me.$generic_slice_with_steps(start, shape, steps)
            }

            /// Creates a slice that may or may not have a fixed size and stride.
            #[inline]
            pub fn $generic_slice<RSlice, CSlice>($me: $Me, start: (usize, usize), shape: (RSlice, CSlice))
                -> $MatrixSlice<N, RSlice, CSlice, S::RStride, S::CStride, S::Alloc>
                where RSlice: Dim,
                      CSlice: Dim {

                $me.assert_slice_index(start, (shape.0.value(), shape.1.value()), (1, 1));

                unsafe {
                    let data = $SliceStorage::new_unchecked($data, start, shape);
                    Matrix::from_data_statically_unchecked(data)
                }
            }

            /// Creates a slice that may or may not have a fixed size and stride.
            #[inline]
            pub fn $generic_slice_with_steps<RSlice, CSlice, RStep, CStep>($me: $Me,
                                                                           start: (usize, usize),
                                                                           shape: (RSlice, CSlice),
                                                                           steps: (RStep, CStep))
                -> $MatrixSlice<N, RSlice, CSlice, DimProd<RStep, S::RStride>, DimProd<CStep, S::CStride>, S::Alloc>
                where RSlice: Dim,
                      CSlice: Dim,
                      RStep: DimMul<S::RStride>,
                      CStep: DimMul<S::CStride> {

                $me.assert_slice_index(start, (shape.0.value(), shape.1.value()), (steps.0.value(), steps.1.value()));

                let my_strides = $me.data.strides();
                let strides    = (steps.0.mul(my_strides.0), steps.1.mul(my_strides.1));

                unsafe {
                    let data = $SliceStorage::new_with_strides_unchecked($data, start, shape, strides);
                    Matrix::from_data_statically_unchecked(data)
                }
            }
        }
    }
);

matrix_slice_impl!(
     self: &Self, MatrixSlice, SliceStorage, Storage, &self.data;
     row,
     rows,
     rows_with_step,
     fixed_rows,
     fixed_rows_with_step,
     rows_generic,
     column,
     columns,
     columns_with_step,
     fixed_columns,
     fixed_columns_with_step,
     columns_generic,
     slice,
     slice_with_steps,
     fixed_slice,
     fixed_slice_with_steps,
     generic_slice,
     generic_slice_with_steps);


matrix_slice_impl!(
     self: &mut Self, MatrixSliceMut, SliceStorageMut, StorageMut, &mut self.data;
     row_mut,
     rows_mut,
     rows_with_step_mut,
     fixed_rows_mut,
     fixed_rows_with_step_mut,
     rows_generic_mut,
     column_mut,
     columns_mut,
     columns_with_step_mut,
     fixed_columns_mut,
     fixed_columns_with_step_mut,
     columns_generic_mut,
     slice_mut,
     slice_with_steps_mut,
     fixed_slice_mut,
     fixed_slice_with_steps_mut,
     generic_slice_mut,
     generic_slice_with_steps_mut);
