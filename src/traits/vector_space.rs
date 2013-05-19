use core::num::Zero;
use traits::division_ring::DivisionRing;
use traits::workarounds::scalar_op::{ScalarMul, ScalarDiv};

pub trait VectorSpace<T: DivisionRing> : Sub<Self, Self> + Add<Self, Self> +
                                         Neg<Self> + Zero + ScalarMul<T> + ScalarDiv<T>
{ }
