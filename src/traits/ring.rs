use core::num::{One, Zero};

/**
 * Trait of elements of a ring. A rings is an algebraic structure, the
 * elements of which have all the common numeric operation but the division:
 * addition, subtraction, multiplication and distinct elements (`One` and
 * `Zero`) respectively neutral and absorbant wrt the multiplication.
 */
pub trait Ring :
Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Mul<Self, Self> + One + Zero
{ }

impl<T: Sub<T, T> + Add<T, T> + Neg<T> + Mul<T, T> + One + Zero>
Ring for T;
