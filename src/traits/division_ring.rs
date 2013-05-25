use traits::ring::Ring;

/**
 * Trait of elements of a division ring. A division ring is an algebraic
 * structure like a ring, but with the division operator.
 */
pub trait DivisionRing : Ring + Div<Self, Self>
{ }

impl<T: Ring + Div<T, T>> DivisionRing for T;
