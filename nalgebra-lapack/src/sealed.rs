#[doc(hidden)]
#[allow(dead_code)]
/// helps to seal traits to this crate
pub trait Sealed {}

// we need to implement sealed for all the numeric types that we want to support
// here.
impl Sealed for f32 {}
impl Sealed for f64 {}
