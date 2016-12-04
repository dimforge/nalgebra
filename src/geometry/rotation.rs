use num::{Zero, One};
use std::fmt;
use approx::ApproxEq;

use alga::general::Real;

use core::{SquareMatrix, Scalar, OwnedSquareMatrix};
use core::dimension::{DimName, DimNameSum, DimNameAdd, U1};
use core::storage::{Storage, StorageMut};
use core::allocator::Allocator;


/// A rotation matrix with an owned storage.
pub type OwnedRotation<N, D, A> = RotationBase<N, D, <A as Allocator<N, D, D>>::Buffer>;

/// A rotation matrix.
#[repr(C)]
#[derive(Hash, Debug, Clone, Copy)]
pub struct RotationBase<N: Scalar, D: DimName, S: Storage<N, D, D>> {
    matrix: SquareMatrix<N, D, S>
}

impl<N: Scalar, D: DimName, S: Storage<N, D, D>> RotationBase<N, D, S>
    where N: Scalar,
          S: Storage<N, D, D> {
    /// A reference to the underlying matrix representation of this rotation.
    #[inline]
    pub fn matrix(&self) -> &SquareMatrix<N, D, S> {
        &self.matrix
    }

    /// A mutable reference to the underlying matrix representation of this rotation.
    ///
    /// This is unsafe because this allows the user to replace the matrix by another one that is
    /// non-square, non-inversible, or non-orthonormal. If one of those properties is broken,
    /// subsequent method calls may be UB.
    #[inline]
    pub unsafe fn matrix_mut(&mut self) -> &mut SquareMatrix<N, D, S> {
        &mut self.matrix
    }

    /// Unwraps the underlying matrix.
    #[inline]
    pub fn unwrap(self) -> SquareMatrix<N, D, S> {
        self.matrix
    }

    /// Converts this rotation into its equivalent homogeneous transformation matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, DimNameSum<D, U1>, S::Alloc>
        where N: Zero + One,
              D: DimNameAdd<U1>,
              S::Alloc: Allocator<N, DimNameSum<D, U1>, DimNameSum<D, U1>> {
        let mut res = OwnedSquareMatrix::<N, _, S::Alloc>::identity();
        res.fixed_slice_mut::<D, D>(0, 0).copy_from(&self.matrix);

        res
    }
}

impl<N: Scalar, D: DimName, S: Storage<N, D, D>> RotationBase<N, D, S> {
    /// Creates a new rotation from the given square matrix.
    ///
    /// The matrix squareness is checked but not its orthonormality.
    #[inline]
    pub fn from_matrix_unchecked(matrix: SquareMatrix<N, D, S>) -> RotationBase<N, D, S> {
        assert!(matrix.is_square(), "Unable to create a rotation from a non-square matrix.");

        RotationBase {
            matrix: matrix
        }
    }

    /// Transposes `self`.
    #[inline]
    pub fn transpose(&self) -> OwnedRotation<N, D, S::Alloc> {
        RotationBase::from_matrix_unchecked(self.matrix.transpose())
    }

    /// Inverts `self`.
    #[inline]
    pub fn inverse(&self) -> OwnedRotation<N, D, S::Alloc> {
        self.transpose()
    }
}


impl<N: Scalar, D: DimName, S: StorageMut<N, D, D>> RotationBase<N, D, S> {
    /// Transposes `self` in-place.
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.matrix.transpose_mut()
    }

    /// Inverts `self` in-place.
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.transpose_mut()
    }
}

impl<N: Scalar + Eq, D: DimName, S: Storage<N, D, D>> Eq for RotationBase<N, D, S> { }

impl<N: Scalar + PartialEq, D: DimName, S: Storage<N, D, D>> PartialEq for RotationBase<N, D, S> {
    #[inline]
    fn eq(&self, right: &RotationBase<N, D, S>) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, D: DimName, S> ApproxEq for RotationBase<N, D, S>
    where N: Scalar + ApproxEq,
          S: Storage<N, D, D>,
          N::Epsilon: Copy {
    type Epsilon = N::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        N::default_epsilon()
    }

    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        N::default_max_relative()
    }

    #[inline]
    fn default_max_ulps() -> u32 {
        N::default_max_ulps()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.matrix.relative_eq(&other.matrix, epsilon, max_relative)
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<N, D: DimName, S> fmt::Display for RotationBase<N, D, S>
    where N: Real + fmt::Display,
          S: Storage<N, D, D>,
          S::Alloc: Allocator<usize, D, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        try!(writeln!(f, "RotationBase matrix {{"));
        try!(write!(f, "{:.*}", precision, self.matrix));
        writeln!(f, "}}")
    }
}

//          //         /*
//          //          *
//          //          * Absolute
//          //          *
//          //          */
//          //         impl<N: Absolute> Absolute for $t<N> {
//          //             type AbsoluteValue = $submatrix<N::AbsoluteValue>;
//          //
//          //             #[inline]
//          //             fn abs(m: &$t<N>) -> $submatrix<N::AbsoluteValue> {
//          //                 Absolute::abs(&m.submatrix)
//          //             }
//          //         }
