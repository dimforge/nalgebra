use num::{Signed, Zero};
use std::cmp::PartialOrd;

use allocator::Allocator;
use ::{Real, Scalar};
use storage::{Storage, StorageMut};
use base::{DefaultAllocator, Matrix, Dim, MatrixMN};
use constraint::{SameNumberOfRows, SameNumberOfColumns, ShapeConstraint};


// FIXME: this should be be a trait on alga?
pub trait Norm<N: Scalar> {
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N
        where R: Dim, C: Dim, S: Storage<N, R, C>;
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
    #[inline]
    pub fn norm(&self) -> N {
        self.norm_squared().sqrt()
    }

    /// Computes the metric distance between `self` and `rhs` using the Euclidean metric.
    #[inline]
    pub fn metric_distance<R2, C2, S2>(&self, rhs: &Matrix<N, R2, C2, S2>) -> N
        where R2: Dim, C2: Dim, S2: Storage<N, R2, C2>,
              ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2> {
        self.apply_metric_distance(rhs, &EuclideanNorm)
    }

    #[inline]
    pub fn apply_norm(&self, norm: &impl Norm<N>) -> N {
        norm.norm(self)
    }

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