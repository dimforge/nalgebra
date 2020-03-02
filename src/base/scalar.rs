use std::any::Any;
use std::any::TypeId;
use std::fmt::Debug;

/// The basic scalar type for all structures of `nalgebra`.
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: Clone + PartialEq + Debug + Any {
    #[inline]
    /// Tests if `Self` the same as the type `T`
    ///
    /// Typically used to test of `Self` is a f32 or a f64 with `N::is::<f32>()`.
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }

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
