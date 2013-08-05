use std::num::Zero;
use traits::division_ring::DivisionRing;
use traits::scalar_op::{ScalarMul, ScalarDiv};

/// Trait of elements of a vector space. A vector space is an algebraic
/// structure, the elements of which have addition, substraction, negation,
/// scalar multiplication (the scalar being a element of a `DivisionRing`), and
/// has a distinct element (`Zero`) neutral wrt the addition.
pub trait VectorSpace<N>
: Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero +
  ScalarMul<N> + ScalarDiv<N> { }

impl<V: Sub<V, V> + Add<V, V> + Neg<V> + Zero + ScalarMul<N> + ScalarDiv<N>,
     N: DivisionRing> VectorSpace<N> for V;
