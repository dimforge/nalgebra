use std::num::Zero;
use traits::dim::Dim;
use traits::basis::Basis;
use traits::indexable::Indexable;
use traits::iterable::Iterable;
use traits::sample::UniformSphereSample;
use traits::scalar_op::{ScalarAdd, ScalarSub};
use traits::dot::Dot;
use traits::norm::Norm;

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
pub trait AlgebraicVecExt<N: Algebraic>: AlgebraicVec<N> + VecExt<N>
{ }

impl<N, V: Dim + Sub<V, V> + Add<V, V> + Neg<V> + Zero + Eq + Mul<N, V> + Div<N, V> + Dot<N>>
Vec<N> for V;

impl<N: Algebraic, V: Vec<N> + Norm<N>> AlgebraicVec<N> for V;

impl<N,
     V: Vec<N> + Basis + Indexable<uint, N> + Iterable<N> + Round +
        UniformSphereSample + ScalarAdd<N> + ScalarSub<N> + Bounded + Orderable>
VecExt<N> for V;

impl<N: Algebraic, V: AlgebraicVec<N> + VecExt<N>> AlgebraicVecExt<N> for V;
