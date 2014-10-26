//! Traits giving structural informations on linear algebra objects or the space they live in.

use std::num::Zero;
use std::slice::{Items, MutItems};
use traits::operations::{RMul, LMul, Axpy};
use traits::geometry::{Dot, Norm, Orig};

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

/// Trait for constructing the identity matrix
pub trait Eye {
    /// Return the identity matrix of specified dimension
    fn new_identity(dim: uint) -> Self;
}

// XXX: we keep ScalarAdd and ScalarSub here to avoid trait impl conflict (overriding) between the
// different Add/Sub traits. This is _so_ unfortunate…

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

/// Trait to access part of a column of a matrix
pub trait ColSlice<C> {
    /// Returns a view to a slice of a column of a matrix.
    fn col_slice(&self, col_id: uint, row_start: uint, row_end: uint) -> C;
}

/// Trait to access part of a row of a matrix
pub trait RowSlice<R> {
    /// Returns a view to a slice of a row of a matrix.
    fn row_slice(&self, row_id: uint, col_start: uint, col_end: uint) -> R;
}

/// Trait of objects having a spacial dimension known at compile time.
pub trait Dim {
    /// The dimension of the object.
    fn dim(unused_self: Option<Self>) -> uint;
}

/// Trait to get the diagonal of square matrices.
pub trait Diag<V> {
    /// Creates a new matrix with the given diagonal.
    fn from_diag(diag: &V) -> Self;

    /// Sets the diagonal of this matrix.
    fn set_diag(&mut self, diag: &V);

    /// The diagonal of this matrix.
    fn diag(&self) -> V;
}

/// The shape of an indexable object.
pub trait Shape<I, Res>: Index<I, Res> {
    /// Returns the shape of an indexable object.
    fn shape(&self) -> I;
}

/// This is a workaround of current Rust limitations.
///
/// It exists because the `I` trait cannot be used to express write access.
/// Thus, this is the same as the `I` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<I, Res>: Shape<I, Res> + IndexMut<I, Res> {
    #[deprecated = "use the Index `[]` overloaded operator instead"]
    /// Reads the `i`-th element of `self`.
    fn at(&self, i: I) -> Res;
    #[deprecated = "use the IndexMut `[]` overloaded operator instead"]
    /// Writes to the `i`-th element of `self`.
    fn set(&mut self, i: I, Res);
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: I, j: I);

    /// Reads the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_at(&self, i: I) -> Res;
    /// Writes to the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_set(&mut self, i: I, Res);
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
    fn iter_mut<'l>(&'l mut self) -> MutItems<'l, N>;
}

/*
 * Vec related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
#[deprecated = "This will be removed in the future. Use point + vector operations instead."]
pub trait VecAsPnt<P> {
    /// Converts this point to its associated vector.
    fn to_pnt(self) -> P;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_pnt<'a>(&'a self) -> &'a P;
}

/// Trait grouping most common operations on vectors.
pub trait NumVec<N>: Dim + Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero + PartialEq +
                     Mul<N, Self> + Div<N, Self> + Dot<N> + Axpy<N> + Index<uint, N> {
}

/// Trait of vector with components implementing the `Float` trait.
pub trait FloatVec<N: Float>: NumVec<N> + Norm<N> + Basis {
}

/*
 * Pnt related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
pub trait PntAsVec<V> {
    /// Converts this point to its associated vector.
    fn to_vec(self) -> V;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_vec<'a>(&'a self) -> &'a V;

    // NOTE: this is used in some places to overcome some limitations untill the trait reform is
    // done on rustc.
    /// Sets the coordinates of this point to match those of a given vector.
    fn set_coords(&mut self, coords: V);
}

/// Trait grouping most common operations on points.
// XXX: the vector space element `V` should be an associated type. Though this would prevent V from
// having bounds (they are not supported yet). So, for now, we will just use a type parameter.
pub trait NumPnt<N, V>:
          PntAsVec<V> + Dim + Sub<Self, V> + Orig + Neg<Self> + PartialEq + Mul<N, Self> +
          Div<N, Self> + Add<V, Self> + Axpy<N> + Index<uint, N> { // FIXME: + Sub<V, Self>
}

/// Trait of points with components implementing the `Float` trait.
pub trait FloatPnt<N: Float, V: Norm<N>>: NumPnt<N, V> {
    /// Computes the square distance between two points.
    #[inline]
    fn sqdist(a: &Self, b: &Self) -> N {
        Norm::sqnorm(&(*a - *b))
    }

    /// Computes the distance between two points.
    #[inline]
    fn dist(a: &Self, b: &Self) -> N {
        Norm::norm(&(*a - *b))
    }
}
