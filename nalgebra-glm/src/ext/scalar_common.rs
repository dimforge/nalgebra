use na;

use traits::Number;

/// Returns the maximum among three values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(3.0, glm::max3_scalar(1.0, 2.0, 3.0));
/// assert_eq!(2, glm::max3_scalar(0, 1, 2));
/// ```
///
/// # See also:
///
/// * [`max4_scalar`](fn.max4_scalar.html)
/// * [`min3_scalar`](fn.min3_scalar.html)
/// * [`min4_scalar`](fn.min4_scalar.html)
pub fn max3_scalar<N: Number>(a: N, b: N, c: N) -> N {
    na::sup(&na::sup(&a, &b), &c)
}

/// Returns the maximum among four values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(4.0, glm::max4_scalar(2.0, 4.0, 1.0, 3.0));
/// assert_eq!(7, glm::max4_scalar(1, 0, 7, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar`](fn.max3_scalar.html)
/// * [`min3_scalar`](fn.min3_scalar.html)
/// * [`min4_scalar`](fn.min4_scalar.html)
pub fn max4_scalar<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::sup(&na::sup(&a, &b), &na::sup(&c, &d))
}

/// Returns the minimum among three values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(1.0, glm::min3_scalar(1.0, 2.0, 3.0));
/// assert_eq!(0, glm::min3_scalar(0, 1, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar`](fn.max3_scalar.html)
/// * [`max4_scalar`](fn.max4_scalar.html)
/// * [`min4_scalar`](fn.min4_scalar.html)
pub fn min3_scalar<N: Number>(a: N, b: N, c: N) -> N {
    na::inf(&na::inf(&a, &b), &c)
}

/// Returns the minimum among four values.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(1.0, glm::min4_scalar(2.0, 4.0, 1.0, 3.0));
/// assert_eq!(0, glm::min4_scalar(1, 0, 7, 2));
/// ```
///
/// # See also:
///
/// * [`max3_scalar`](fn.max3_scalar.html)
/// * [`max4_scalar`](fn.max4_scalar.html)
/// * [`min3_scalar`](fn.min3_scalar.html)
pub fn min4_scalar<N: Number>(a: N, b: N, c: N, d: N) -> N {
    na::inf(&na::inf(&a, &b), &na::inf(&c, &d))
}
