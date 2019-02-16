use num::Signed;
use std::cmp::PartialOrd;

use allocator::Allocator;
use ::{Real, Scalar};
use storage::{Storage, StorageMut};
use base::{DefaultAllocator, Matrix, Dim, MatrixMN};
use constraint::{SameNumberOfRows, SameNumberOfColumns, ShapeConstraint};


// FIXME: this should be be a trait on alga?
/// A trait for abstract matrix norms.
///
/// This may be moved to the alga crate in the future.
pub trait Norm<N: Scalar> {
    /// Apply this norm to the given matrix.
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N
        where R: Dim, C: Dim, S: Storage<N, R, C>;
    /// Use the metric induced by this norm to compute the metric distance between the two given matrices.
    fn metric_distance<R1, C1, S1, R2, C2, S2>(&self, m1: &Matrix<N, R1, C1, S1>, m2: &Matrix<N, R2, C2, S2>) -> N
        where R1: Dim, C1: Dim, S1: Storage<N, R1, C1>,
              R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>;
}

/// Euclidean norm.
pub struct EuclideanNorm;
/// Lp norm.
pub struct LpNorm(pub i32);
/// L-infinite norm aka. Chebytchev norm aka. uniform norm aka. suppremum norm.
pub struct UniformNorm;

impl<N: Real> Norm<N> for EuclideanNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N
        where R: Dim, C: Dim, S: Storage<N, R, C> {
        m.norm_squared().sqrt()
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(&self, m1: &Matrix<N, R1, C1, S1>, m2: &Matrix<N, R2, C2, S2>) -> N
        where R1: Dim, C1: Dim, S1: Storage<N, R1, C1>,
              R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
        m1.zip_fold(m2, N::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff * diff
        }).sqrt()
    }
}

impl<N: Real> Norm<N> for LpNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N
        where R: Dim, C: Dim, S: Storage<N, R, C> {
        m.fold(N::zero(), |a, b| {
            a + b.abs().powi(self.0)
        }).powf(::convert(1.0 / (self.0 as f64)))
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(&self, m1: &Matrix<N, R1, C1, S1>, m2: &Matrix<N, R2, C2, S2>) -> N
        where R1: Dim, C1: Dim, S1: Storage<N, R1, C1>,
              R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
        m1.zip_fold(m2, N::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.abs().powi(self.0)
        }).powf(::convert(1.0 / (self.0 as f64)))
    }
}

impl<N: Scalar + PartialOrd + Signed> Norm<N> for UniformNorm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N
        where R: Dim, C: Dim, S: Storage<N, R, C> {
        m.amax()
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(&self, m1: &Matrix<N, R1, C1, S1>, m2: &Matrix<N, R2, C2, S2>) -> N
        where R1: Dim, C1: Dim, S1: Storage<N, R1, C1>,
              R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
        m1.zip_fold(m2, N::zero(), |acc, a, b| {
            let val = (a - b).abs();
            if val > acc {
                val
            } else {
                acc
            }
        })
    }
}


impl<N: Real, R: Dim, C: Dim, S: Storage<N, R, C>> Matrix<N, R, C, S> {
    /// The squared L2 norm of this vector.
    #[inline]
    pub fn norm_squared(&self) -> N {
        let mut res = N::zero();

        for i in 0..self.ncols() {
            let col = self.column(i);
            res += col.dot(&col)
        }

        res
    }

    /// The L2 norm of this matrix.
    ///
    /// Use `.apply_norm` to apply a custom norm.
    #[inline]
    pub fn norm(&self) -> N {
        self.norm_squared().sqrt()
    }

    /// Compute the distance between `self` and `rhs` using the metric induced by the euclidean norm.
    ///
    /// Use `.apply_metric_distance` to apply a custom norm.
    #[inline]
    pub fn metric_distance<R2, C2, S2>(&self, rhs: &Matrix<N, R2, C2, S2>) -> N
        where R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        self.apply_metric_distance(rhs, &EuclideanNorm)
    }

    /// Uses the given `norm` to compute the norm of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Vector3, UniformNorm, LpNorm, EuclideanNorm};
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.apply_norm(&UniformNorm), 3.0);
    /// assert_eq!(v.apply_norm(&LpNorm(1)), 6.0);
    /// assert_eq!(v.apply_norm(&EuclideanNorm), v.norm());
    /// ```
    #[inline]
    pub fn apply_norm(&self, norm: &impl Norm<N>) -> N {
        norm.norm(self)
    }

    /// Uses the metric induced by the given `norm` to compute the metric distance between `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra::{Vector3, UniformNorm, LpNorm, EuclideanNorm};
    ///
    /// let v1 = Vector3::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3::new(10.0, 20.0, 30.0);
    ///
    /// assert_eq!(v1.apply_metric_distance(&v2, &UniformNorm), 27.0);
    /// assert_eq!(v1.apply_metric_distance(&v2, &LpNorm(1)), 27.0 + 18.0 + 9.0);
    /// assert_eq!(v1.apply_metric_distance(&v2, &EuclideanNorm), (v1 - v2).norm());
    /// ```
    #[inline]
    pub fn apply_metric_distance<R2, C2, S2>(&self, rhs: &Matrix<N, R2, C2, S2>, norm: &impl Norm<N>) -> N
        where R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        norm.metric_distance(self,rhs)
    }

    /// The Lp norm of this matrix.
    #[inline]
    pub fn lp_norm(&self, p: i32) -> N {
        self.apply_norm(&LpNorm(p))
    }

    /// A synonym for the norm of this matrix.
    ///
    /// Aka the length.
    ///
    /// This function is simply implemented as a call to `norm()`
    #[inline]
    pub fn magnitude(&self) -> N {
        self.norm()
    }

    /// A synonym for the squared norm of this matrix.
    ///
    /// Aka the squared length.
    ///
    /// This function is simply implemented as a call to `norm_squared()`
    #[inline]
    pub fn magnitude_squared(&self) -> N {
        self.norm_squared()
    }

    /// Returns a normalized version of this matrix.
    #[inline]
    pub fn normalize(&self) -> MatrixMN<N, R, C>
        where DefaultAllocator: Allocator<N, R, C> {
        self / self.norm()
    }

    /// Returns a normalized version of this matrix unless its norm as smaller or equal to `eps`.
    #[inline]
    pub fn try_normalize(&self, min_norm: N) -> Option<MatrixMN<N, R, C>>
        where DefaultAllocator: Allocator<N, R, C> {
        let n = self.norm();

        if n <= min_norm {
            None
        } else {
            Some(self / n)
        }
    }
}

impl<N: Real, R: Dim, C: Dim, S: StorageMut<N, R, C>> Matrix<N, R, C, S> {
    /// Normalizes this matrix in-place and returns its norm.
    #[inline]
    pub fn normalize_mut(&mut self) -> N {
        let n = self.norm();
        *self /= n;

        n
    }

    /// Normalizes this matrix in-place or does nothing if its norm is smaller or equal to `eps`.
    ///
    /// If the normalization succeeded, returns the old normal of this matrix.
    #[inline]
    pub fn try_normalize_mut(&mut self, min_norm: N) -> Option<N> {
        let n = self.norm();

        if n <= min_norm {
            None
        } else {
            *self /= n;
            Some(n)
        }
    }
}
