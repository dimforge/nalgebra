use std::any::TypeId;
use std::fmt::Debug;
use std::any::Any;

/// The basic scalar type for all structures of `nalgebra`.
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: Copy + PartialEq + Debug + Any {
    #[inline]
    /// Tests if `Self` the same as the type `T`
    ///
    /// Typically used to test of `Self` is a f32 or a f64 with `N::is::<f32>()`.
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }
}
impl<T: Copy + PartialEq + Debug + Any> Scalar for T {}
