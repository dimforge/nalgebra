use crate::aliases::{
    TMat2, TMat2x3, TMat2x4, TMat3, TMat3x2, TMat3x4, TMat4, TMat4x2, TMat4x3, TVec2, TVec3, TVec4,
};
use crate::traits::Number;

/// Builds a 2x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal2x2<T: Number>(v: &TVec2<T>) -> TMat2<T> {
    TMat2::from_diagonal(v)
}

/// Builds a 2x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal2x3<T: Number>(v: &TVec2<T>) -> TMat2x3<T> {
    TMat2x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 2x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal2x4<T: Number>(v: &TVec2<T>) -> TMat2x4<T> {
    TMat2x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal3x2<T: Number>(v: &TVec2<T>) -> TMat3x2<T> {
    TMat3x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 3x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal3x3<T: Number>(v: &TVec3<T>) -> TMat3<T> {
    TMat3::from_diagonal(v)
}

/// Builds a 3x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal3x4<T: Number>(v: &TVec3<T>) -> TMat3x4<T> {
    TMat3x4::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x2 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x3()`]
/// * [`diagonal4x4()`]
pub fn diagonal4x2<T: Number>(v: &TVec2<T>) -> TMat4x2<T> {
    TMat4x2::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x3 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x4()`]
pub fn diagonal4x3<T: Number>(v: &TVec3<T>) -> TMat4x3<T> {
    TMat4x3::from_partial_diagonal(v.as_slice())
}

/// Builds a 4x4 diagonal matrix.
///
/// # See also:
///
/// * [`diagonal2x2()`]
/// * [`diagonal2x3()`]
/// * [`diagonal2x4()`]
/// * [`diagonal3x2()`]
/// * [`diagonal3x3()`]
/// * [`diagonal3x4()`]
/// * [`diagonal4x2()`]
/// * [`diagonal4x3()`]
pub fn diagonal4x4<T: Number>(v: &TVec4<T>) -> TMat4<T> {
    TMat4::from_diagonal(v)
}
