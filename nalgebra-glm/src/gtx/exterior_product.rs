use crate::aliases::TVec2;
use crate::traits::Number;

/// The 2D perpendicular product between two vectors.
pub fn cross2d<T: Number>(v: &TVec2<T>, u: &TVec2<T>) -> T {
    v.perp(u)
}
