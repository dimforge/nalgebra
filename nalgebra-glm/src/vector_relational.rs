use na::DefaultAllocator;

use aliases::TVec;
use traits::{Alloc, Dimension, Number};

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
/// * [`any`](fn.any.html)
/// * [`not`](fn.not.html)
pub fn all<D: Dimension>(v: &TVec<bool, D>) -> bool
where DefaultAllocator: Alloc<bool, D> {
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
/// * [`all`](fn.all.html)
/// * [`not`](fn.not.html)
pub fn any<D: Dimension>(v: &TVec<bool, D>) -> bool
where DefaultAllocator: Alloc<bool, D> {
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
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
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
/// * [`equal`](fn.equal.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn greater_than<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
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
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn greater_than_equal<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
) -> TVec<bool, D>
where
    DefaultAllocator: Alloc<N, D>,
{
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
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn less_than<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

/// Component-wise `>=` comparison.
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
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn less_than_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
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
/// * [`all`](fn.all.html)
/// * [`any`](fn.any.html)
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn not<D: Dimension>(v: &TVec<bool, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<bool, D> {
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
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
pub fn not_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
