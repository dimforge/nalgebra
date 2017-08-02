use std::any::TypeId;
use std::fmt::Debug;
use std::any::Any;

/// The basic scalar type for all structures of `nalgebra`.
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: Copy + PartialEq + Debug + Any {
    #[inline]
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }
}
impl<T: Copy + PartialEq + Debug + Any> Scalar for T { }
