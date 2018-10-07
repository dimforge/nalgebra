use na::{DefaultAllocator};

use aliases::TVec;
use traits::{Number, Alloc, Dimension};

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
/// # See also:
///
/// * [`equal`](fn.equal.html)
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
/// # See also:
///
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
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
/// # See also:
///
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn greater_than_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x >= y)
}

/// Component-wise `<` comparison.
///
/// # See also:
///
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn less_than<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x < y)
}

/// Component-wise `>=` comparison.
///
/// # See also:
///
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
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
/// * [`not`](fn.not.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn not<D: Dimension>(v: &TVec<bool, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<bool, D> {
    v.map(|x| !x)
}

/// Component-wise not-equality `!=`.
///
/// # See also:
///
/// * [`equal`](fn.equal.html)
/// * [`greater_than`](fn.greater_than.html)
/// * [`greater_than_equal`](fn.greater_than_equal.html)
/// * [`less_than`](fn.less_than.html)
/// * [`less_than_equal`](fn.less_than_equal.html)
/// * [`not_equal`](fn.not_equal.html)
pub fn not_equal<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<bool, D>
    where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x != y)
}
