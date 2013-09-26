//! Low level operations on vectors and matrices.


/// Trait of objects having an absolute value.
/// This is useful if the object does not have the same type as its absolute value.
pub trait Absolute<A> {
    /// Computes some absolute value of this object.
    /// Typically, this will make all component of a matrix or vector positive.
    fn absolute(&self) -> A;
}

/// Trait of objects having an inverse. Typically used to implement matrix inverse.
pub trait Inv {
    /// Returns the inverse of `self`.
    fn inverse(&self) -> Option<Self>;
    /// In-place version of `inverse`.
    fn inplace_inverse(&mut self) -> bool;
}

/// Trait of objects which can be transposed. Note that, for the moment, this
/// does not allow the implementation by non-square matrix (or anything which
/// is not stable by transposition).
pub trait Transpose {
    /// Computes the transpose of a matrix.
    fn transposed(&self) -> Self;

    /// In-place version of `transposed`.
    fn transpose(&mut self);
}

/// Traits of objects having an outer product.
pub trait Outer<V, M> {
    /// Computes the outer product: `self * other`
    fn outer(&self, other: &V) -> M;
}

/// Trait for computing the covariance of a set of data.
pub trait Cov<M> {
    /// Computes the covariance of the obsevations stored by `self`:
    ///
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn cov(&self) -> M;
}

/// Trait for computing the covariance of a set of data.
pub trait Mean<N> {
    /// Computes the mean of the observations stored by `self`.
    /// 
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn mean(&self) -> N;
}

// XXX: those two traits should not exist since there is generalized operator overloading of Add
// and Sub.
// However, using the same trait multiple time as a trait bound (ex: impl<T: Add<N, V> + Add<V, V>)
// does not work properly, mainly because the way we are doing generalized operator overloading is
// verry hacky.
//
// Hopefully, this will be fixed on a future version of the language!
/// Trait of objects having a right multiplication with another element.
pub trait RMul<V> {
    /// Computes `self * v`
    fn rmul(&self, v: &V) -> V;
}

impl<M: Mul<T, T>, T> RMul<T> for M {
    fn rmul(&self, v: &T) -> T {
        self * *v
    }
}

/// Trait of objects having a left multiplication with another element.
pub trait LMul<V> {
    /// Computes `v * self`
    fn lmul(&self, &V) -> V;
}

impl<T: Mul<M, T>, M> LMul<T> for M {
    fn lmul(&self, v: &T) -> T {
        v * *self
    }
}

// XXX: those traits should not exist since there is generalized operator overloading of Add and
// Sub.
// However, using the same trait multiple time as a trait bound (ex: impl<T: Add<N, V> + Add<V, V>)
// does not work properly, mainly because the way we are doing generalized operator overloading is
// verry hacky.
//
// Hopefully, this will be fixed on a future version of the language!
/// Trait of objects having an addition with a scalar.
pub trait ScalarAdd<N> {
    /// Gets the result of `self + n`.
    fn add_s(&self, n: &N) -> Self;
}

impl<N, T: Add<N, T>> ScalarAdd<N> for T {
    /// Gets the result of `self + n`.
    fn add_s(&self, n: &N) -> T {
        *self + *n
    }
}

/**
 * Trait of objects having a subtraction with a scalar.
 */
pub trait ScalarSub<N> {
    /// Gets the result of `self - n`.
    fn sub_s(&self, n: &N) -> Self;
}

impl<N, T: Sub<N, T>> ScalarSub<N> for T {
    /// Gets the result of `self - n`.
    fn sub_s(&self, n: &N) -> T {
        *self - *n
    }
}
