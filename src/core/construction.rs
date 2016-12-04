#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use std::iter;
use num::{Zero, One, Bounded};
use rand::{self, Rand, Rng};
use typenum::{self, Cmp, Greater};

use alga::general::{ClosedAdd, ClosedMul};

use core::{Scalar, Matrix, SquareMatrix, ColumnVector, Unit};
use core::dimension::{Dim, DimName, Dynamic, U1, U2, U3, U4, U5, U6};
use core::allocator::{Allocator, OwnedAllocator};
use core::storage::{Storage, OwnedStorage};

/*
 *
 * Generic constructors.
 *
 */
impl<N: Scalar, R: Dim, C: Dim, S: OwnedStorage<N, R, C>> Matrix<N, R, C, S>
    // XXX: needed because of a compiler bug. See the rust compiler issue #26026.
    where S::Alloc: OwnedAllocator<N, R, C, S> {
    /// Creates a new uninitialized matrix. If the matrix has a compile-time dimension, this panics
    /// if `nrows != R::to_usize()` or `ncols != C::to_usize()`.
    #[inline]
    pub unsafe fn new_uninitialized_generic(nrows: R, ncols: C) -> Matrix<N, R, C, S> {
        Matrix::from_data(S::Alloc::allocate_uninitialized(nrows, ncols))
    }

    /// Creates a matrix with all its elements set to `elem`.
    #[inline]
    pub fn from_element_generic(nrows: R, ncols: C, elem: N) -> Matrix<N, R, C, S> {
        let len = nrows.value() * ncols.value();
        Matrix::from_iterator_generic(nrows, ncols, iter::repeat(elem).take(len))
    }

    /// Creates a matrix with all its elements filled by an iterator.
    #[inline]
    pub fn from_iterator_generic<I>(nrows: R, ncols: C, iter: I) -> Matrix<N, R, C, S>
        where I: IntoIterator<Item = N> {
        Matrix::from_data(S::Alloc::allocate_from_iterator(nrows, ncols, iter))
    }

    /// Creates a matrix with its elements filled with the components provided by a slice in
    /// row-major order.
    ///
    /// The order of elements in the slice must follow the usual mathematic writing, i.e.,
    /// row-by-row.
    #[inline]
    pub fn from_row_slice_generic(nrows: R, ncols: C, slice: &[N]) -> Matrix<N, R, C, S> {
        assert!(slice.len() == nrows.value() * ncols.value(),
                "Matrix init. error: the slice did not contain the right number of elements.");

        let mut res  = unsafe { Self::new_uninitialized_generic(nrows, ncols) };
        let mut iter = slice.iter();

        for i in 0 .. nrows.value() {
            for j in 0 .. ncols.value() {
                unsafe {
                    *res.get_unchecked_mut(i, j) = *iter.next().unwrap()
                }
            }
        }

        res
    }

    /// Creates a matrix with its elements filled with the components provided by a slice. The
    /// components must have the same layout as the matrix data storage (i.e. row-major or column-major).
    #[inline]
    pub fn from_column_slice_generic(nrows: R, ncols: C, slice: &[N]) -> Matrix<N, R, C, S> {
        Matrix::from_iterator_generic(nrows, ncols, slice.iter().cloned())
    }

    /// Creates a matrix filled with the results of a function applied to each of its component
    /// coordinates.
    #[inline]
    pub fn from_fn_generic<F>(nrows: R, ncols: C, mut f: F) -> Matrix<N, R, C, S>
        where F: FnMut(usize, usize) -> N {
        let mut res = unsafe { Self::new_uninitialized_generic(nrows, ncols) };

        for i in 0 .. nrows.value() {
            for j in 0 .. ncols.value() {
                unsafe { *res.get_unchecked_mut(i, j) = f(i, j) }
            }
        }

        res
    }

    /// Creates a new indentity matrix.
    ///
    /// If the matrix is not square, the largest square submatrix starting at index `(0, 0)` is set
    /// to the identity matrix. All other entries are set to zero.
    #[inline]
    pub fn identity_generic(nrows: R, ncols: C) -> Matrix<N, R, C, S>
        where N: Zero + One {
        Self::from_diagonal_element_generic(nrows, ncols, N::one())
    }

    /// Creates a new matrix with its diagonal filled with copies of `elt`.
    ///
    /// If the matrix is not square, the largest square submatrix starting at index `(0, 0)` is set
    /// to the identity matrix. All other entries are set to zero.
    #[inline]
    pub fn from_diagonal_element_generic(nrows: R, ncols: C, elt: N) -> Matrix<N, R, C, S>
        where N: Zero + One {
        let mut res = unsafe { Self::new_uninitialized_generic(nrows, ncols) };
        res.fill(N::zero());

        for i in 0 .. ::min(nrows.value(), ncols.value()) {
            unsafe { *res.get_unchecked_mut(i, i) = elt }
        }

        res
    }

    #[inline]
    pub fn from_rows<SB>(rows: &[Matrix<N, U1, C, SB>]) -> Matrix<N, R, C, S>
        where SB: Storage<N, U1, C> {

        assert!(rows.len() > 0, "At least one row must be given.");
        let nrows = R::try_to_usize().unwrap_or(rows.len());
        let ncols = rows[0].len();
        assert!(rows.len() == nrows, "Invalid number of rows provided to build this matrix.");

        if C::try_to_usize().is_none() {
            assert!(rows.iter().all(|r| r.len() == ncols),
            "The rows provided must all have the same dimension.");
        }

        // FIXME: optimize that.
        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |i, j| rows[i][(0, j)])
    }

    #[inline]
    pub fn from_columns<SB>(columns: &[ColumnVector<N, R, SB>]) -> Matrix<N, R, C, S>
        where SB: Storage<N, R, U1> {

        assert!(columns.len() > 0, "At least one column must be given.");
        let ncols = C::try_to_usize().unwrap_or(columns.len());
        let nrows = columns[0].len();
        assert!(columns.len() == ncols, "Invalid number of columns provided to build this matrix.");

        if R::try_to_usize().is_none() {
            assert!(columns.iter().all(|r| r.len() == nrows),
            "The columns provided must all have the same dimension.");
        }

        // FIXME: optimize that.
        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |i, j| columns[j][i])
    }
}

impl<N, R: Dim, C: Dim, S> Matrix<N, R, C, S>
    where N: Scalar + Rand,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    /// Creates a matrix filled with random values.
    #[inline]
    pub fn new_random_generic(nrows: R, ncols: C) -> Matrix<N, R, C, S> {
        Matrix::from_fn_generic(nrows, ncols, |_, _| rand::random())
    }
}

impl<N, D: Dim, S> SquareMatrix<N, D, S>
    where N: Scalar + Zero,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    /// Creates a square matrix with its diagonal set to `diag` and all other entries set to 0.
    #[inline]
    pub fn from_diagonal<SB: Storage<N, D, U1>>(diag: &ColumnVector<N, D, SB>) -> Self {
        let (dim, _) = diag.data.shape();
        let mut res = Self::from_element_generic(dim, dim, N::zero());

        for i in 0 .. diag.len() {
            unsafe { *res.get_unchecked_mut(i, i) = *diag.get_unchecked(i, 0); }
        }

        res
    }
}


/*
 *
 * Generate constructors with varying number of arguments, depending on the object type.
 *
 */
macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<N: Scalar, $($DimIdent: $DimBound, )* S> Matrix<N $(, $Dims)*, S>
            where S: OwnedStorage<N $(, $Dims)*>,
                  S::Alloc: OwnedAllocator<N $(, $Dims)*, S> {

            /// Creates a new uninitialized matrix.
            #[inline]
            pub unsafe fn new_uninitialized($($args: usize),*) -> Matrix<N $(, $Dims)*, S> {
                Self::new_uninitialized_generic($($gargs),*)
            }

            /// Creates a matrix with all its elements set to `elem`.
            #[inline]
            pub fn from_element($($args: usize,)* elem: N) -> Matrix<N $(, $Dims)*, S> {
                Self::from_element_generic($($gargs, )* elem)
            }

            /// Creates a matrix with all its elements filled by an iterator.
            #[inline]
            pub fn from_iterator<I>($($args: usize,)* iter: I) -> Matrix<N $(, $Dims)*, S>
                where I: IntoIterator<Item = N> {
                Self::from_iterator_generic($($gargs, )* iter)
            }

            /// Creates a matrix with its elements filled with the components provided by a slice
            /// in row-major order.
            ///
            /// The order of elements in the slice must follow the usual mathematic writing, i.e.,
            /// row-by-row.
            #[inline]
            pub fn from_row_slice($($args: usize,)* slice: &[N]) -> Matrix<N $(, $Dims)*, S> {
                Self::from_row_slice_generic($($gargs, )* slice)
            }

            /// Creates a matrix with its elements filled with the components provided by a slice
            /// in column-major order.
            #[inline]
            pub fn from_column_slice($($args: usize,)* slice: &[N]) -> Matrix<N $(, $Dims)*, S> {
                Self::from_column_slice_generic($($gargs, )* slice)
            }

            /// Creates a matrix filled with the results of a function applied to each of its
            /// component coordinates.
            // FIXME: don't take a dimension of the matrix is statically sized.
            #[inline]
            pub fn from_fn<F>($($args: usize,)* f: F) -> Matrix<N $(, $Dims)*, S>
                where F: FnMut(usize, usize) -> N {
                Self::from_fn_generic($($gargs, )* f)
            }

            #[inline]
            pub fn identity($($args: usize,)*) -> Matrix<N $(, $Dims)*, S>
                where N: Zero + One {
                Self::identity_generic($($gargs),* )
            }

            #[inline]
            pub fn from_diagonal_element($($args: usize,)* elt: N) -> Matrix<N $(, $Dims)*, S>
                where N: Zero + One {
                Self::from_diagonal_element_generic($($gargs, )* elt)
            }
        }

        impl<N: Scalar + Rand, $($DimIdent: $DimBound, )* S> Matrix<N $(, $Dims)*, S>
            where S: OwnedStorage<N $(, $Dims)*>,
                  S::Alloc: OwnedAllocator<N $(, $Dims)*, S> {

            /// Creates a matrix filled with random values.
            #[inline]
            pub fn new_random($($args: usize),*) -> Matrix<N $(, $Dims)*, S> {
                Self::new_random_generic($($gargs),*)
            }
        }
    }
);

// FIXME: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<N, ..., S>
                   => R: DimName, => C: DimName; // Type parameters for impl<N, ..., S>
                   R::name(), C::name();         // Arguments for `_generic` constructors.
                   );                            // Arguments for non-generic constructors.

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

/*
 *
 * Zero, One, Rand traits.
 *
 */
impl<N, R: DimName, C: DimName, S> Zero for Matrix<N, R, C, S>
    where N: Scalar + Zero + ClosedAdd,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn zero() -> Self {
        Self::from_element(N::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(|e| e.is_zero())
    }
}

impl<N, D: DimName, S> One for Matrix<N, D, D, S>
    where N: Scalar + Zero + One + ClosedMul + ClosedAdd,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn one() -> Self {
        Self::identity()
    }
}

impl<N, R: DimName, C: DimName, S> Bounded for Matrix<N, R, C, S>
    where N: Scalar + Bounded,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn max_value() -> Self {
        Self::from_element(N::max_value())
    }

    #[inline]
    fn min_value() -> Self {
        Self::from_element(N::min_value())
    }
}

impl<N: Scalar + Rand, R: Dim, C: Dim, S> Rand for Matrix<N, R, C, S>
    where S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn rand<G: Rng>(rng: &mut G) -> Self {
        let nrows = R::try_to_usize().unwrap_or(rng.gen_range(0, 10));
        let ncols = C::try_to_usize().unwrap_or(rng.gen_range(0, 10));

        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |_, _| rng.gen())
    }
}


#[cfg(feature = "arbitrary")]
impl<N, R, C, S> Arbitrary for Matrix<N, R, C, S>
    where R: Dim, C: Dim,
          N: Scalar + Arbitrary + Send,
          S: OwnedStorage<N, R, C> + Send,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let nrows = R::try_to_usize().unwrap_or(g.gen_range(0, 10));
        let ncols = C::try_to_usize().unwrap_or(g.gen_range(0, 10));

        Self::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |_, _| N::arbitrary(g))
    }
}


/*
 *
 * Constructors for small matrices and vectors.
 *
 */
macro_rules! componentwise_constructors_impl(
    ($($R: ty, $C: ty, $($args: ident:($irow: expr,$icol: expr)),*);* $(;)*) => {$(
        impl<N, S> Matrix<N, $R, $C, S>
            where N: Scalar,
                  S: OwnedStorage<N, $R, $C>,
                  S::Alloc: OwnedAllocator<N, $R, $C, S> {
            /// Initializes this matrix from its components.
            #[inline]
            pub fn new($($args: N),*) -> Matrix<N, $R, $C, S> {
                unsafe {
                    let mut res = Self::new_uninitialized();
                    $( *res.get_unchecked_mut($irow, $icol) = $args; )*

                    res
                }
            }
        }
    )*}
);

componentwise_constructors_impl!(
    /*
     * Square matrices 1 .. 6.
     */
    U2, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1);
    U3, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2);
    U4, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3);
    U5, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4);
    U6, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4), m56:(4,5),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3), m65:(5,4), m66:(5,5);

    /*
     * Rectangular matrices with 2 rows.
     */
    U2, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2);
    U2, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3);
    U2, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4);
    U2, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5);

    /*
     * Rectangular matrices with 3 rows.
     */
    U3, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1);
    U3, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3);
    U3, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4);
    U3, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5);

    /*
     * Rectangular matrices with 4 rows.
     */
    U4, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1);
    U4, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2);
    U4, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4);
    U4, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5);

    /*
     * Rectangular matrices with 5 rows.
     */
    U5, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1),
            m51:(4,0), m52:(4,1);
    U5, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2),
            m51:(4,0), m52:(4,1), m53:(4,2);
    U5, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3);
    U5, U6, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4), m16:(0,5),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4), m26:(1,5),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4), m36:(2,5),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4), m46:(3,5),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4), m56:(4,5);

    /*
     * Rectangular matrices with 6 rows.
     */
    U6, U2, m11:(0,0), m12:(0,1),
            m21:(1,0), m22:(1,1),
            m31:(2,0), m32:(2,1),
            m41:(3,0), m42:(3,1),
            m51:(4,0), m52:(4,1),
            m61:(5,0), m62:(5,1);
    U6, U3, m11:(0,0), m12:(0,1), m13:(0,2),
            m21:(1,0), m22:(1,1), m23:(1,2),
            m31:(2,0), m32:(2,1), m33:(2,2),
            m41:(3,0), m42:(3,1), m43:(3,2),
            m51:(4,0), m52:(4,1), m53:(4,2),
            m61:(5,0), m62:(5,1), m63:(5,2);
    U6, U4, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3);
    U6, U5, m11:(0,0), m12:(0,1), m13:(0,2), m14:(0,3), m15:(0,4),
            m21:(1,0), m22:(1,1), m23:(1,2), m24:(1,3), m25:(1,4),
            m31:(2,0), m32:(2,1), m33:(2,2), m34:(2,3), m35:(2,4),
            m41:(3,0), m42:(3,1), m43:(3,2), m44:(3,3), m45:(3,4),
            m51:(4,0), m52:(4,1), m53:(4,2), m54:(4,3), m55:(4,4),
            m61:(5,0), m62:(5,1), m63:(5,2), m64:(5,3), m65:(5,4);

    /*
     * Row vectors 1 .. 6.
     */
    U1, U1, x:(0,0);
    U1, U2, x:(0,0), y:(0,1);
    U1, U3, x:(0,0), y:(0,1), z:(0,2);
    U1, U4, x:(0,0), y:(0,1), z:(0,2), w:(0,3);
    U1, U5, x:(0,0), y:(0,1), z:(0,2), w:(0,3), a:(0,4);
    U1, U6, x:(0,0), y:(0,1), z:(0,2), w:(0,3), a:(0,4), b:(0,5);

    /*
     * Column vectors 1 .. 6.
     */
    U2, U1, x:(0,0), y:(1,0);
    U3, U1, x:(0,0), y:(1,0), z:(2,0);
    U4, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0);
    U5, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0), a:(4,0);
    U6, U1, x:(0,0), y:(1,0), z:(2,0), w:(3,0), a:(4,0), b:(5,0);
);

/*
 *
 * Axis constructors.
 *
 */
impl<N, R: DimName, S> ColumnVector<N, R, S>
where N: Scalar + Zero + One,
      S: OwnedStorage<N, R, U1>,
      S::Alloc: OwnedAllocator<N, R, U1, S> {
    /// The column vector with a 1 as its first component, and zero elsewhere.
    #[inline]
    pub fn x() -> Self
      where R::Value: Cmp<typenum::U0, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(0, 0) = N::one(); }

        res
    }

    /// The column vector with a 1 as its second component, and zero elsewhere.
    #[inline]
    pub fn y() -> Self
      where R::Value: Cmp<typenum::U1, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(1, 0) = N::one(); }

        res
    }

    /// The column vector with a 1 as its third component, and zero elsewhere.
    #[inline]
    pub fn z() -> Self
      where R::Value: Cmp<typenum::U2, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(2, 0) = N::one(); }

        res
    }

    /// The column vector with a 1 as its fourth component, and zero elsewhere.
    #[inline]
    pub fn w() -> Self
      where R::Value: Cmp<typenum::U3, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(3, 0) = N::one(); }

        res
    }

    /// The column vector with a 1 as its fifth component, and zero elsewhere.
    #[inline]
    pub fn a() -> Self
      where R::Value: Cmp<typenum::U4, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(4, 0) = N::one(); }

        res
    }

    /// The column vector with a 1 as its sixth component, and zero elsewhere.
    #[inline]
    pub fn b() -> Self
      where R::Value: Cmp<typenum::U5, Output = Greater> {
        let mut res = Self::from_element(N::zero());
        unsafe { *res.get_unchecked_mut(5, 0) = N::one(); }

        res
    }

    /// The unit column vector with a 1 as its first component, and zero elsewhere.
    #[inline]
    pub fn x_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U0, Output = Greater> {
          Unit::new_unchecked(Self::x())
    }

    /// The unit column vector with a 1 as its second component, and zero elsewhere.
    #[inline]
    pub fn y_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U1, Output = Greater> {
          Unit::new_unchecked(Self::y())
    }

    /// The unit column vector with a 1 as its third component, and zero elsewhere.
    #[inline]
    pub fn z_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U2, Output = Greater> {
          Unit::new_unchecked(Self::z())
    }

    /// The unit column vector with a 1 as its fourth component, and zero elsewhere.
    #[inline]
    pub fn w_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U3, Output = Greater> {
          Unit::new_unchecked(Self::w())
    }

    /// The unit column vector with a 1 as its fifth component, and zero elsewhere.
    #[inline]
    pub fn a_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U4, Output = Greater> {
          Unit::new_unchecked(Self::a())
    }

    /// The unit column vector with a 1 as its sixth component, and zero elsewhere.
    #[inline]
    pub fn b_axis() -> Unit<Self>
      where R::Value: Cmp<typenum::U5, Output = Greater> {
          Unit::new_unchecked(Self::b())
    }
}
