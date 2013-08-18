use std::num::Zero;
use traits::dim::Dim;
use traits::basis::Basis;
use traits::indexable::Indexable;
use traits::iterable::Iterable;
use traits::sample::UniformSphereSample;
use traits::scalar_op::{ScalarAdd, ScalarSub};

// NOTE: cant call that `Vector` because it conflicts with std::Vector
/// Trait grouping most common operations on vectors.
pub trait Vec<N>: Dim + Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero + Eq + Mul<N, Self>
                     + Div<N, Self>
{
    /// Computes the dot (inner) product of two vectors.
    #[inline]
    fn dot(&self, &Self) -> N;

    /**
     * Short-cut to compute the projection of a point on a vector, but without
     * computing intermediate vectors.
     * This must be equivalent to:
     *
     *   (a - b).dot(c)
     *
     */
    #[inline]
    fn sub_dot(&self, b: &Self, c: &Self) -> N {
        (*self - *b).dot(c)
    }
}

/// Trait of vector with components implementing the `Algebraic` trait.
pub trait AlgebraicVec<N: Algebraic>: Vec<N> {
    /// Computes the norm a an object.
    #[inline]
    fn norm(&self) -> N {
        self.sqnorm().sqrt()
    }

    /**
     * Computes the squared norm of an object. Usually faster than computing the
     * norm itself.
     */
    #[inline]
    fn sqnorm(&self) -> N {
        self.dot(self)
    }

    /// Gets the normalized version of the argument.
    #[inline]
    fn normalized(&self) -> Self {
        self / self.norm()
    }

    /// In-place version of `normalized`.
    #[inline]
    fn normalize(&mut self) -> N {
        let norm = self.norm();

        *self = *self / norm;

        norm
    }
}

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait VecExt<N>: Vec<N> + Basis + Indexable<uint, N> + Iterable<N> + Round +
                     UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded + Orderable
{ }

/// Trait grouping uncommon, low-level and borderline (from the mathematical point of view)
/// operations on vectors.
pub trait AlgebraicVecExt<N: Algebraic>: AlgebraicVec<N> + VecExt<N>
{ }

impl<N,
     V: Vec<N> + Basis + Indexable<uint, N> + Iterable<N> + Round +
        UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded + Orderable>
VecExt<N> for V;

impl<N: Algebraic, V: AlgebraicVec<N> + VecExt<N>>
AlgebraicVecExt<N> for V;
