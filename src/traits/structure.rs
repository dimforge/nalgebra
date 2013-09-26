use std::num::Zero;
use std::vec::{VecIterator, VecMutIterator};
use traits::operations::{RMul, LMul, ScalarAdd, ScalarSub};
use traits::geometry::{Dot, Norm, UniformSphereSample};

/// Trait of matrices.
///
/// A matrix has rows and columns and are able to multiply them.
pub trait Mat<R, C> : Row<R> + Col<C> + RMul<R> + LMul<C> { }

impl<M: Row<R> + Col<C> + RMul<R> + LMul<C>, R, C> Mat<R, C> for M {
}

/// Trait of matrices which can be converted to another matrix.
///
/// Use this to change easily the type of a matrix components.
pub trait MatCast<M> {
    /// Converts `m` to have the type `M`.
    fn from(m: Self) -> M;
}

// XXX: we keep ScalarAdd and ScalarSub here to avoid trait impl conflict (overriding) between the
// different Add/Sub traits. This is _so_ unfortunate…

// NOTE: cant call that `Vector` because it conflicts with std::Vector
/// Trait grouping most common operations on vectors.
pub trait Vec<N>: Dim + Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero + Eq + Mul<N, Self>
                  + Div<N, Self> + Dot<N> {
}

/// Trait of vector with components implementing the `Algebraic` trait.
pub trait AlgebraicVec<N: Algebraic>: Vec<N> + Norm<N> {
}

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait VecExt<N>: Vec<N> + Basis + Indexable<uint, N> + Iterable<N> + Round +
                     UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded + Orderable
{ }

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait AlgebraicVecExt<N: Algebraic>: AlgebraicVec<N> + VecExt<N> { }

impl<N, V: Dim + Sub<V, V> + Add<V, V> + Neg<V> + Zero + Eq + Mul<N, V> + Div<N, V> + Dot<N>>
Vec<N> for V { }

impl<N: Algebraic, V: Vec<N> + Norm<N>> AlgebraicVec<N> for V { }

impl<N,
     V: Vec<N> + Basis + Indexable<uint, N> + Iterable<N> + Round +
        UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded + Orderable>
VecExt<N> for V { }

impl<N: Algebraic, V: AlgebraicVec<N> + VecExt<N>> AlgebraicVecExt<N> for V { }

/// Trait of vectors which can be converted to another vector. Used to change the type of a vector
/// components.
pub trait VecCast<V> {
    /// Converts `v` to have the type `V`.
    fn from(v: Self) -> V;
}

// FIXME: return an iterator instead
/// Traits of objects which can form a basis (typically vectors).
pub trait Basis {
    /// Iterates through the canonical basis of the space in which this object lives.
    fn canonical_basis(&fn(Self) -> bool);

    /// Iterates through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis(&self, &fn(Self) -> bool);

    /// Creates the canonical basis of the space in which this object lives.
    fn canonical_basis_list() -> ~[Self] {
        let mut res = ~[];

        do Basis::canonical_basis |elem| {
            res.push(elem);

            true
        }

        res
    }

    /// Creates a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis_list(&self) -> ~[Self] {
        let mut res = ~[];

        do self.orthonormal_subspace_basis |elem| {
            res.push(elem);

            true
        }

        res
    }
}

/// Trait to access rows of a matrix or a vector.
pub trait Row<R> {
    /// The number of column of `self`.
    fn num_rows(&self) -> uint;
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
    fn num_cols(&self) -> uint;

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
}

/// This is a workaround of current Rust limitations.
///
/// Traits of objects which can be iterated through like a vector.
pub trait Iterable<N> {
    /// Gets a vector-like read-only iterator.
    fn iter<'l>(&'l self) -> VecIterator<'l, N>;
}

/// This is a workaround of current Rust limitations.
///
/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut<N> {
    /// Gets a vector-like read-write iterator.
    fn mut_iter<'l>(&'l mut self) -> VecMutIterator<'l, N>;
}
