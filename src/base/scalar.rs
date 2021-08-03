use std::any::Any;
use std::fmt::Debug;

/// The basic scalar type for all structures of `nalgebra`.
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: 'static + Clone + PartialEq + Debug {
    #[inline(always)]
    /// Performance hack: Clone doesn't get inlined for Copy types in debug mode, so make it inline anyway.
    fn inlined_clone(&self) -> Self {
        self.clone()
    }
}

impl<T: Copy + PartialEq + Debug + Any> Scalar for T {
    #[inline(always)]
    fn inlined_clone(&self) -> T {
        *self
    }
}
