//! Low level operations on vectors and matrices.


/// Trait of objects having an absolute value.
/// This is useful if the object does not have the same type as its absolute value.
pub trait Absolute<A> {
    /// Computes some absolute value of this object.
    /// Typically, this will make all component of a matrix or vector positive.
    fn abs(&Self) -> A;
}

/// Trait of objects having an inverse. Typically used to implement matrix inverse.
pub trait Inv {
    /// Returns the inverse of `self`.
    fn inv_cpy(m: &Self) -> Option<Self>;

    /// In-place version of `inverse`.
    fn inv(&mut self) -> bool;
}

/// Trait of objects which can be transposed.
pub trait Transpose {
    /// Computes the transpose of a matrix.
    fn transpose_cpy(m: &Self) -> Self;

    /// In-place version of `transposed`.
    fn transpose(&mut self);
}

/// Traits of objects having an outer product.
pub trait Outer<M> {
    /// Computes the outer product: `self * other`
    fn outer(a: &Self, b: &Self) -> M;
}

/// Trait for computing the covariance of a set of data.
pub trait Cov<M> {
    /// Computes the covariance of the obsevations stored by `self`:
    ///
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn cov(&Self) -> M;

    /// Computes the covariance of the obsevations stored by `self`:
    ///
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn cov_to(m: &Self, out: &mut M) {
        *out = Cov::cov(m)
    }
}

/// Trait for computing the covariance of a set of data.
pub trait Mean<N> {
    /// Computes the mean of the observations stored by `self`.
    /// 
    ///   * For matrices, observations are stored in its rows.
    ///   * For vectors, observations are stored in its components (thus are 1-dimensional).
    fn mean(&Self) -> N;
}

// /// Cholesky decomposition.
// pub trait Chol {
//     /// Performs the cholesky decomposition on `self`. The resulting upper-triangular matrix is
//     /// returned. Returns `None` if the matrix is not symetric positive-definite.
//     fn chol(self) -> Option<Self>;
// 
//     /// Performs the cholesky decomposition on `self`. The resulting upper-triangular matrix is
//     /// written to a given parameter. If the decomposition fails `false` is returned; the state of
//     /// the output parameter is undefined.
//     fn chol_to(&self, out: &mut Self) -> bool {
//         match self.chol() {
//             None => false,
//             Some(decomp) => {
//                 *out = decomp;
//                 true
//             }
//         }
//     }
// }

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
