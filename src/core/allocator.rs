//! Abstract definition of a matrix data storage allocator.

use std::any::Any;

use core::Scalar;
use core::constraint::{SameNumberOfRows, SameNumberOfColumns, ShapeConstraint};
use core::dimension::{Dim, U1};
use core::storage::{Storage, OwnedStorage};

/// A matrix allocator of a memory buffer that may contain `R::to_usize() * C::to_usize()`
/// elements of type `N`.
///
/// An allocator is said to be:
///   − static:  if `R` and `C` both implement `DimName`.
///   − dynamic: if either one (or both) of `R` or `C` is equal to `Dynamic`.
///
/// Every allocator must be both static and dynamic. Though not all implementations may share the
/// same `Buffer` type.
pub trait Allocator<N: Scalar, R: Dim, C: Dim>: Any + Sized {
    /// The type of buffer this allocator can instanciate.
    type Buffer: OwnedStorage<N, R, C, Alloc = Self>;

    /// Allocates a buffer with the given number of rows and columns without initializing its content.
    unsafe fn allocate_uninitialized(nrows: R, ncols: C) -> Self::Buffer;

    /// Allocates a buffer initialized with the content of the given iterator.
    fn allocate_from_iterator<I: IntoIterator<Item = N>>(nrows: R, ncols: C, iter: I) -> Self::Buffer;
}

/// A matrix data allocator dedicated to the given owned matrix storage.
pub trait OwnedAllocator<N: Scalar, R: Dim, C: Dim, S: OwnedStorage<N, R, C, Alloc = Self>>:
    Allocator<N, R, C, Buffer = S> {
}

impl<N, R, C, T, S> OwnedAllocator<N, R, C, S> for T
    where N: Scalar, R: Dim, C: Dim,
          T: Allocator<N, R, C, Buffer = S>,
          S: OwnedStorage<N, R, C, Alloc = T> {
}

/// The number of rows of the result of a componentwise operation on two matrices.
pub type SameShapeR<R1, R2> = <ShapeConstraint as SameNumberOfRows<R1, R2>>::Representative;

/// The number of columns of the result of a componentwise operation on two matrices.
pub type SameShapeC<C1, C2> = <ShapeConstraint as SameNumberOfColumns<C1, C2>>::Representative;

// FIXME: Bad name.
/// Restricts the given number of rows and columns to be respectively the same. Can only be used
/// when `Self = SA::Alloc`.
pub trait SameShapeAllocator<N, R1, C1, R2, C2, SA>:
        Allocator<N, R1, C1> +
        Allocator<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
    where R1: Dim, R2: Dim, C1: Dim, C2: Dim,
          N: Scalar,
          SA: Storage<N, R1, C1, Alloc = Self>,
          ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
}

impl<N, R1, R2, C1, C2, SA> SameShapeAllocator<N, R1, C1, R2, C2, SA> for SA::Alloc
    where R1: Dim, R2: Dim, C1: Dim, C2: Dim,
          N: Scalar,
          SA: Storage<N, R1, C1>,
          SA::Alloc:
            Allocator<N, R1, C1> +
            Allocator<N, SameShapeR<R1, R2>, SameShapeC<C1, C2>>,
          ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
}

// XXX: Bad name.
/// Restricts the given number of rows to be equal. Can only be used when `Self = SA::Alloc`.
pub trait SameShapeColumnVectorAllocator<N, R1, R2, SA>:
        Allocator<N, R1, U1> +
        Allocator<N, SameShapeR<R1, R2>, U1> +
        SameShapeAllocator<N, R1, U1, R2, U1, SA>
    where R1: Dim, R2: Dim,
          N: Scalar,
          SA: Storage<N, R1, U1, Alloc = Self>,
          ShapeConstraint: SameNumberOfRows<R1, R2> {
}

impl<N, R1, R2, SA> SameShapeColumnVectorAllocator<N, R1, R2, SA> for SA::Alloc
    where R1: Dim, R2: Dim,
          N: Scalar,
          SA: Storage<N, R1, U1>,
          SA::Alloc:
            Allocator<N, R1, U1> +
            Allocator<N, SameShapeR<R1, R2>, U1>,
          ShapeConstraint: SameNumberOfRows<R1, R2> {
}
