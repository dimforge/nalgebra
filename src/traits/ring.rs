use core::num::{One, Zero};

pub trait Ring :
Sub<Self, Self> + Add<Self, Self> + Neg<Self> + Mul<Self, Self> + One + Zero
{ }
