use crate::aliases::TVec;
use crate::traits::Number;

/// Checks that all the vector components are `true`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(true, false);
/// assert_eq!(false, glm::all(&vec));
///
/// let vec = glm::vec2(true, true);
/// assert_eq!(true, glm::all(&vec));
/// ```
///
/// # See also:
///
/// * [`any()`]
/// * [`not()`]
pub fn all<const D: usize>(v: &TVec<bool, D>) -> bool {
    v.iter().all(|x| *x)
}

/// Checks that at least one of the vector components is `true`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(true, false);
/// assert_eq!(true, glm::any(&vec));
///
/// let vec = glm::vec2(true, true);
/// assert_eq!(true, glm::any(&vec));
///
/// let vec = glm::vec2(false, false);
/// assert_eq!(false, glm::any(&vec));
/// ```
///
/// # See also:
///
/// * [`all()`]
/// * [`not()`]
pub fn any<const D: usize>(v: &TVec<bool, D>) -> bool {
    v.iter().any(|x| *x)
}

/// Component-wise equality comparison.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec2(true, false),
///            glm::equal(&glm::vec2(1.0, 3.0),
///                       &glm::vec2(1.0, 2.0)));
/// ```
///
/// # See also:
///
/// * [`greater_than()`]
/// * [`greater_than_equal()`]
/// * [`less_than()`]
/// * [`less_than_equal()`]
/// * [`not()`]
/// * [`not_equal()`]
pub fn equal<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x == y)
}

/// Component-wise `>` comparison.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec2(false, true),
///            glm::greater_than(&glm::vec2(1.0, 3.0),
///                              &glm::vec2(1.0, 2.0)));
/// ```
///
/// # See also:
///
/// * [`equal()`]
/// * [`greater_than_equal()`]
/// * [`less_than()`]
/// * [`less_than_equal()`]
/// * [`not()`]
/// * [`not_equal()`]
pub fn greater_than<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x > y)
}

/// Component-wise `>=` comparison.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec3(true, true, false),
///            glm::greater_than_equal(&glm::vec3(1.0, 3.0, 4.0),
///                                    &glm::vec3(1.0, 2.0, 5.0)));
/// ```
///
/// # See also:
///
/// * [`equal()`]
/// * [`greater_than()`]
/// * [`less_than()`]
/// * [`less_than_equal()`]
/// * [`not()`]
/// * [`not_equal()`]
pub fn greater_than_equal<T: Number, const D: usize>(
    x: &TVec<T, D>,
    y: &TVec<T, D>,
) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x >= y)
}

/// Component-wise `<` comparison.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec3(false, false, true),
///            glm::less_than(&glm::vec3(1.0, 3.0, 4.0),
///                           &glm::vec3(1.0, 2.0, 5.0)));
/// ```
///
/// # See also:
///
/// * [`equal()`]
/// * [`greater_than()`]
/// * [`greater_than_equal()`]
/// * [`less_than_equal()`]
/// * [`not()`]
/// * [`not_equal()`]
pub fn less_than<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x < y)
}

/// Component-wise `<=` comparison.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec3(true, false, true),
///            glm::less_than_equal(&glm::vec3(1.0, 3.0, 4.0),
///                                 &glm::vec3(1.0, 2.0, 5.0)));
/// ```
///
/// # See also:
///
/// * [`equal()`]
/// * [`greater_than()`]
/// * [`greater_than_equal()`]
/// * [`less_than()`]
/// * [`not()`]
/// * [`not_equal()`]
pub fn less_than_equal<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x <= y)
}

/// Component-wise not `!`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec2(true, false);
/// assert_eq!(glm::vec2(false, true), glm::not(&vec));
/// ```
///
/// # See also:
///
/// * [`all()`]
/// * [`any()`]
/// * [`equal()`]
/// * [`greater_than()`]
/// * [`greater_than_equal()`]
/// * [`less_than()`]
/// * [`less_than_equal()`]
/// * [`not_equal()`]
pub fn not<const D: usize>(v: &TVec<bool, D>) -> TVec<bool, D> {
    v.map(|x| !x)
}

/// Component-wise not-equality `!=`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::vec2(false, true),
///            glm::not_equal(&glm::vec2(1.0, 3.0),
///                           &glm::vec2(1.0, 2.0)));
/// ```
///
/// # See also:
///
/// * [`equal()`]
/// * [`greater_than()`]
/// * [`greater_than_equal()`]
/// * [`less_than()`]
/// * [`less_than_equal()`]
/// * [`not()`]
pub fn not_equal<T: Number, const D: usize>(x: &TVec<T, D>, y: &TVec<T, D>) -> TVec<bool, D> {
    x.zip_map(y, |x, y| x != y)
}
