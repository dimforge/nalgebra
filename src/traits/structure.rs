//! Traits giving structural informations on linear algebra objects or the space they live in.

use std::num::{Zero, Bounded};
use std::slice::{Items, MutItems};
use traits::operations::{RMul, LMul, ScalarAdd, ScalarSub};
use traits::geometry::{Dot, Norm, UniformSphereSample};

/// Traits of objects which can be created from an object of type `T`.
pub trait Cast<T> {
    /// Converts an element of type `T` to an element of type `Self`.
    fn from(t: T) -> Self;
}

/// Trait of matrices.
///
/// A matrix has rows and columns and are able to multiply them.
pub trait Mat<R, C> : Row<R> + Col<C> + RMul<R> + LMul<C> { }

impl<M: Row<R> + Col<C> + RMul<R> + LMul<C>, R, C> Mat<R, C> for M {
}

// XXX: we keep ScalarAdd and ScalarSub here to avoid trait impl conflict (overriding) between the
// different Add/Sub traits. This is _so_ unfortunate…

/// Trait grouping most common operations on vectors.
pub trait Vec<N>: Dim + Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero + Eq + Mul<N, Self>
                  + Div<N, Self> + Dot<N> {
}

/// Trait of vector with components implementing the `Float` trait.
pub trait FloatVec<N: Float>: Vec<N> + Norm<N> {
}

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait VecExt<N>: Vec<N> + Indexable<uint, N> + Iterable<N> +
                     UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded
{ }

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait FloatVecExt<N: Float>: FloatVec<N> + VecExt<N> + Basis + Round { }

impl<N, V: Dim + Sub<V, V> + Add<V, V> + Neg<V> + Zero + Eq + Mul<N, V> + Div<N, V> + Dot<N>>
Vec<N> for V { }

impl<N: Float, V: Vec<N> + Norm<N>> FloatVec<N> for V { }

impl<N,
     V: Vec<N> + Indexable<uint, N> + Iterable<N> +
        UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded>
VecExt<N> for V { }

impl<N: Float, V: FloatVec<N> + VecExt<N> + Basis + Round> FloatVecExt<N> for V { }

// FIXME: return an iterator instead
/// Traits of objects which can form a basis (typically vectors).
pub trait Basis {
    /// Iterates through the canonical basis of the space in which this object lives.
    fn canonical_basis(|Self| -> bool);

    /// Iterates through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis(&Self, |Self| -> bool);
}

/// Trait to access rows of a matrix or a vector.
pub trait Row<R> {
    /// The number of column of `self`.
    fn nrows(&self) -> uint;
    /// Reads the `i`-th row of `self`.
    fn row(&self, i: uint) -> R;
    /// Writes the `i`-th row of `self`.
    fn set_row(&mut self, i: uint, R);

    // FIXME: add iterators on rows: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access columns of a matrix or vector.
pub trait Col<C> {
    /// The number of column of this matrix or vector.
    fn ncols(&self) -> uint;

    /// Reads the `i`-th column of `self`.
    fn col(&self, i: uint) -> C;

    /// Writes the `i`-th column of `self`.
    fn set_col(&mut self, i: uint, C);

    // FIXME: add iterators on columns: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait of objects having a spacial dimension known at compile time.
pub trait Dim {
    /// The dimension of the object.
    fn dim(unused_self: Option<Self>) -> uint;
}

// FIXME: this trait should not be on nalgebra.
// however, it is needed because std::ops::Index is (strangely) to poor: it
// does not have a function to set values.
// Also, using Index with tuples crashes.
/// This is a workaround of current Rust limitations.
///
/// It exists because the `Index` trait cannot be used to express write access.
/// Thus, this is the same as the `Index` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<Index, Res> {
    /// Reads the `i`-th element of `self`.
    fn at(&self, i: Index) -> Res;
    /// Writes to the `i`-th element of `self`.
    fn set(&mut self, i: Index, Res);
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: Index, j: Index);

    /// Reads the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_at(&self, i: Index) -> Res;
    /// Writes to the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_set(&mut self, i: Index, Res);
}

/// This is a workaround of current Rust limitations.
///
/// Traits of objects which can be iterated through like a vector.
pub trait Iterable<N> {
    /// Gets a vector-like read-only iterator.
    fn iter<'l>(&'l self) -> Items<'l, N>;
}

/// This is a workaround of current Rust limitations.
///
/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut<N> {
    /// Gets a vector-like read-write iterator.
    fn mut_iter<'l>(&'l mut self) -> MutItems<'l, N>;
}
