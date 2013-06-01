use std::num::Zero;
use traits::division_ring::DivisionRing;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv};

/// Trait of elements of a vector space. A vector space is an algebraic
/// structure, the elements of which have addition, substraction, negation,
/// scalar multiplication (the scalar being a element of a `DivisionRing`), and
/// has a distinct element (`Zero`) neutral wrt the addition.
pub trait VectorSpace<T: DivisionRing>
: Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Zero +
  ScalarMul<T> + ScalarDiv<T>
{ }
