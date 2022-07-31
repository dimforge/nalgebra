//! Abstract definition of a matrix data storage allocator.

use std::any::Any;

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, U1};
use crate::base::{DefaultAllocator, Scalar};
use crate::storage::{IsContiguous, RawStorageMut};
use crate::StorageMut;
use std::fmt::Debug;
use std::mem::MaybeUninit;

/// A matrix allocator of a memory buffer that may contain `R::to_usize() * C::to_usize()`
/// elements of type `T`.
///
/// An allocator is said to be:
///   − static:  if `R` and `C` both implement `DimName`.
///   − dynamic: if either one (or both) of `R` or `C` is equal to `Dynamic`.
///
/// Every allocator must be both static and dynamic. Though not all implementations may share the
/// same `Buffer` type.
pub trait Allocator<T, R: Dim, C: Dim = U1>: Any + Sized {
    /// The type of buffer this allocator can instanciate.
    type Buffer: StorageMut<T, R, C> + IsContiguous + Clone + Debug;
    /// The type of buffer with uninitialized components this allocator can instanciate.
    type BufferUninit: RawStorageMut<MaybeUninit<T>, R, C> + IsContiguous;

    /// Allocates a buffer with the given number of rows and columns without initializing its content.
    fn allocate_uninit(nrows: R, ncols: C) -> Self::BufferUninit;

    /// Assumes a data buffer to be initialized.
    ///
    /// # Safety
    /// The user must make sure that every single entry of the buffer has been initialized,
    /// or Undefined Behavior will immediately occur.    
    unsafe fn assume_init(uninit: Self::BufferUninit) -> Self::Buffer;

    /// Allocates a buffer initialized with the content of the given iterator.
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer;

    #[inline]
    /// Allocates a buffer initialized with the content of the given row-major order iterator.
    fn allocate_from_row_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer {
        let mut res = Self::allocate_uninit(nrows, ncols);
        let mut count = 0;

        unsafe {
            // OK because the allocated buffer is guaranteed to be contiguous.
            let res_ptr = res.as_mut_slice_unchecked();

            for (k, e) in iter
                .into_iter()
                .take(ncols.value() * nrows.value())
                .enumerate()
            {
                let i = k / ncols.value();
                let j = k % ncols.value();
                // result[(i, j)] = e;
                *res_ptr.get_unchecked_mut(i + j * nrows.value()) = MaybeUninit::new(e);
                count += 1;
            }

            assert!(
                count == nrows.value() * ncols.value(),
                "Matrix init. from row iterator: iterator not long enough."
            );

            <Self as Allocator<T, R, C>>::assume_init(res)
        }
    }
}

/// A matrix reallocator. Changes the size of the memory buffer that initially contains (`RFrom` ×
/// `CFrom`) elements to a smaller or larger size (`RTo`, `CTo`).
pub trait Reallocator<T: Scalar, RFrom: Dim, CFrom: Dim, RTo: Dim, CTo: Dim>:
    Allocator<T, RFrom, CFrom> + Allocator<T, RTo, CTo>
{
    /// Reallocates a buffer of shape `(RTo, CTo)`, possibly reusing a previously allocated buffer
    /// `buf`. Data stored by `buf` are linearly copied to the output:
    ///
    /// # Safety
    /// The following invariants must be respected by the implementors of this method:
    /// * The copy is performed as if both were just arrays (without taking into account the matrix structure).
    /// * If the underlying buffer is being shrunk, the removed elements must **not** be dropped
    ///   by this method. Dropping them is the responsibility of the caller.
    unsafe fn reallocate_copy(
        nrows: RTo,
        ncols: CTo,
        buf: <Self as Allocator<T, RFrom, CFrom>>::Buffer,
    ) -> <Self as Allocator<T, RTo, CTo>>::BufferUninit;
}

/// The number of rows of the result of a componentwise operation on two matrices.
pub type SameShapeR<R1, R2> = <ShapeConstraint as SameNumberOfRows<R1, R2>>::Representative;

/// The number of columns of the result of a componentwise operation on two matrices.
pub type SameShapeC<C1, C2> = <ShapeConstraint as SameNumberOfColumns<C1, C2>>::Representative;

// TODO: Bad name.
/// Restricts the given number of rows and columns to be respectively the same.
pub trait SameShapeAllocator<T, R1, C1, R2, C2>:
    Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
where
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

impl<T, R1, R2, C1, C2> SameShapeAllocator<T, R1, C1, R2, C2> for DefaultAllocator
where
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    DefaultAllocator: Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

// XXX: Bad name.
/// Restricts the given number of rows to be equal.
pub trait SameShapeVectorAllocator<T, R1, R2>:
    Allocator<T, R1> + Allocator<T, SameShapeR<R1, R2>> + SameShapeAllocator<T, R1, U1, R2, U1>
where
    R1: Dim,
    R2: Dim,
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}

impl<T, R1, R2> SameShapeVectorAllocator<T, R1, R2> for DefaultAllocator
where
    R1: Dim,
    R2: Dim,
    DefaultAllocator: Allocator<T, R1, U1> + Allocator<T, SameShapeR<R1, R2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}
