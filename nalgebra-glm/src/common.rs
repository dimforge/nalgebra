use na::{self, DefaultAllocator, Real};
use num::FromPrimitive;
use std::mem;

use aliases::{TMat, TVec};
use traits::{Alloc, Dimension, Number};

/// For each matrix or vector component `x` if `x >= 0`; otherwise, it returns `-x`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec3(-1.0, 0.0, 2.0);
/// assert_eq!(glm::vec3(1.0, 0.0, 2.0), glm::abs(&vec));
///
/// let mat = glm::mat2(-0.0, 1.0, -3.0, 2.0);
/// assert_eq!(glm::mat2(0.0, 1.0, 3.0, 2.0), glm::abs(&mat));
/// ```
///
/// # See also:
///
/// * [`sign`](fn.sign.html)
pub fn abs<N: Number, R: Dimension, C: Dimension>(x: &TMat<N, R, C>) -> TMat<N, R, C>
where DefaultAllocator: Alloc<N, R, C> {
    x.abs()
}

/// For each matrix or vector component returns a value equal to the nearest integer that is greater than or equal to `x`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec3(-1.5, 0.5, 2.8);
/// assert_eq!(glm::vec3(-1.0, 1.0, 3.0), glm::ceil(&vec));
/// ```
///
/// # See also:
///
/// * [`ceil`](fn.ceil.html)
/// * [`floor`](fn.floor.html)
/// * [`fract`](fn.fract.html)
/// * [`round`](fn.round.html)
/// * [`trunc`](fn.trunc.html)
pub fn ceil<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.ceil())
}

/// Returns `min(max(x, min_val), max_val)`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// // Works with integers:
/// assert_eq!(3, glm::clamp_scalar(1, 3, 5));
/// assert_eq!(4, glm::clamp_scalar(4, 3, 5));
/// assert_eq!(5, glm::clamp_scalar(7, 3, 5));
///
/// // And it works with floats:
/// assert_eq!(3.25, glm::clamp_scalar(1.3, 3.25, 5.5));
/// assert_eq!(4.5, glm::clamp_scalar(4.5, 3.25, 5.5));
/// assert_eq!(5.5, glm::clamp_scalar(7.8, 3.25, 5.5));
/// ```
///
/// # See also:
///
/// * [`clamp`](fn.clamp.html)
/// * [`clamp_vec`](fn.clamp_vec.html)
pub fn clamp_scalar<N: Number>(x: N, min_val: N, max_val: N) -> N {
    na::clamp(x, min_val, max_val)
}

/// Returns `min(max(x[i], min_val), max_val)` for each component in `x`
/// using the values `min_val and `max_val` as bounds.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// // Works with integers:
/// assert_eq!(glm::vec3(3, 4, 5),
///            glm::clamp(&glm::vec3(1, 4, 7), 3, 5));
///
/// // And it works with floats:
/// assert_eq!(glm::vec3(3.25, 4.5, 5.5),
///            glm::clamp(&glm::vec3(1.3, 4.5, 7.8), 3.25, 5.5));
/// ```
///
/// # See also:
///
/// * [`clamp_scalar`](fn.clamp_scalar.html)
/// * [`clamp_vec`](fn.clamp_vec.html)
pub fn clamp<N: Number, D: Dimension>(x: &TVec<N, D>, min_val: N, max_val: N) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| na::clamp(x, min_val, max_val))
}

/// Returns `min(max(x[i], min_val[i]), max_val[i])` for each component in `x`
/// using the components of `min_val` and `max_val` as bounds.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let min_bounds = glm::vec2(1.0, 3.0);
/// let max_bounds = glm::vec2(5.0, 6.0);
/// assert_eq!(glm::vec2(1.0, 6.0),
///            glm::clamp_vec(&glm::vec2(0.0, 7.0),
///                           &min_bounds,
///                           &max_bounds));
/// assert_eq!(glm::vec2(2.0, 6.0),
///            glm::clamp_vec(&glm::vec2(2.0, 7.0),
///                           &min_bounds,
///                           &max_bounds));
/// assert_eq!(glm::vec2(1.0, 4.0),
///            glm::clamp_vec(&glm::vec2(0.0, 4.0),
///                           &min_bounds,
///                           &max_bounds));
/// ```
///
/// # See also:
///
/// * [`clamp_scalar`](fn.clamp_scalar.html)
/// * [`clamp`](fn.clamp.html)
pub fn clamp_vec<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    min_val: &TVec<N, D>,
    max_val: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.zip_zip_map(min_val, max_val, |a, min, max| na::clamp(a, min, max))
}

/// Returns a signed integer value representing the encoding of a floating-point value.
///
/// The floating-point value's bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn float_bits_to_int(v: f32) -> i32 {
    unsafe { mem::transmute(v) }
}

/// Returns a signed integer value representing the encoding of each component of `v`.
///
/// The floating point value's bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn float_bits_to_int_vec<D: Dimension>(v: &TVec<f32, D>) -> TVec<i32, D>
where DefaultAllocator: Alloc<f32, D> {
    v.map(float_bits_to_int)
}

/// Returns an unsigned integer value representing the encoding of a floating-point value.
///
/// The floating-point value's bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn float_bits_to_uint(v: f32) -> u32 {
    unsafe { mem::transmute(v) }
}

/// Returns an unsigned integer value representing the encoding of each component of `v`.
///
/// The floating point value's bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn float_bits_to_uint_vec<D: Dimension>(v: &TVec<f32, D>) -> TVec<u32, D>
where DefaultAllocator: Alloc<f32, D> {
    v.map(float_bits_to_uint)
}

/// Returns componentwise a value equal to the nearest integer that is less then or equal to `x`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec3(-1.5, 0.5, 2.8);
/// assert_eq!(glm::vec3(-2.0, 0.0, 2.0), glm::floor(&vec));
/// ```
///
/// # See also:
///
/// * [`ceil`](fn.ceil.html)
/// * [`fract`](fn.fract.html)
/// * [`round`](fn.round.html)
/// * [`trunc`](fn.trunc.html)
pub fn floor<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.floor())
}

//// FIXME: should be implemented for TVec/TMat?
//pub fn fma<N: Number>(a: N, b: N, c: N) -> N {
//    // FIXME: use an actual FMA
//    a * b + c
//}

/// Returns the fractional part of each component of `x`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec3(-1.5, 0.5, 2.25);
/// assert_eq!(glm::vec3(-0.5, 0.5, 0.25), glm::fract(&vec));
/// ```
///
/// # See also:
///
/// * [`ceil`](fn.ceil.html)
/// * [`floor`](fn.floor.html)
/// * [`round`](fn.round.html)
/// * [`trunc`](fn.trunc.html)
pub fn fract<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.fract())
}

//// FIXME: should be implemented for TVec/TMat?
///// Returns the (significant, exponent) of this float number.
//pub fn frexp<N: Real>(x: N, exp: N) -> (N, N) {
//    // FIXME: is there a better approach?
//    let e = x.log2().ceil();
//    (x * (-e).exp2(), e)
//}

/// Returns a floating-point value corresponding to a signed integer encoding of a floating-point value.
///
/// If an inf or NaN is passed in, it will not signal, and the resulting floating point value is unspecified. Otherwise, the bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn int_bits_to_float(v: i32) -> f32 {
    f32::from_bits(v as u32)
}

/// For each components of `v`, returns a floating-point value corresponding to a signed integer encoding of a floating-point value.
///
/// If an inf or NaN is passed in, it will not signal, and the resulting floating point value is unspecified. Otherwise, the bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn int_bits_to_float_vec<D: Dimension>(v: &TVec<i32, D>) -> TVec<f32, D>
where DefaultAllocator: Alloc<f32, D> {
    v.map(int_bits_to_float)
}

//pub fn isinf<N: Scalar, D: Dimension>(x: &TVec<N, D>) -> TVec<bool, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//
//}
//
//pub fn isnan<N: Scalar, D: Dimension>(x: &TVec<N, D>) -> TVec<bool, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//
//}

///// Returns the (significant, exponent) of this float number.
//pub fn ldexp<N: Real>(x: N, exp: N) -> N {
//    // FIXME: is there a better approach?
//    x * (exp).exp2()
//}

/// Returns `x * (1.0 - a) + y * a`, i.e., the linear blend of the scalars x and y using the scalar value a.
///
/// The value for a is not restricted to the range `[0, 1]`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::mix_scalar(2.0, 20.0, 0.1), 3.8);
/// ```
///
/// # See also:
///
/// * [`mix`](fn.mix.html)
/// * [`mix_vec`](fn.mix_vec.html)
pub fn mix_scalar<N: Number>(x: N, y: N, a: N) -> N {
    x * (N::one() - a) + y * a
}

/// Returns `x * (1.0 - a) + y * a`, i.e., the linear blend of the vectors x and y using the scalar value a.
///
/// The value for a is not restricted to the range `[0, 1]`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let x = glm::vec3(1.0, 2.0, 3.0);
/// let y = glm::vec3(10.0, 20.0, 30.0);
/// assert_eq!(glm::mix(&x, &y, 0.1), glm::vec3(1.9, 3.8, 5.7));
/// ```
///
/// # See also:
///
/// * [`mix_scalar`](fn.mix_scalar.html)
/// * [`mix_vec`](fn.mix_vec.html)
pub fn mix<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, a: N) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x * (N::one() - a) + y * a
}

/// Returns `x * (1.0 - a) + y * a`, i.e., the component-wise linear blend of `x` and `y` using the components of
/// the vector `a` as coefficients.
///
/// The value for a is not restricted to the range `[0, 1]`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let x = glm::vec3(1.0, 2.0, 3.0);
/// let y = glm::vec3(10.0, 20.0, 30.0);
/// let a = glm::vec3(0.1, 0.2, 0.3);
/// assert_eq!(glm::mix_vec(&x, &y, &a), glm::vec3(1.9, 5.6, 11.1));
/// ```
///
/// # See also:
///
/// * [`mix_scalar`](fn.mix_scalar.html)
/// * [`mix`](fn.mix.html)
pub fn mix_vec<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    a: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    x.component_mul(&(TVec::<N, D>::repeat(N::one()) - a)) + y.component_mul(&a)
}

/// Returns `x * (1.0 - a) + y * a`, i.e., the linear blend of the scalars x and y using the scalar value a.
///
/// The value for a is not restricted to the range `[0, 1]`.
/// This is an alias for `mix_scalar`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// assert_eq!(glm::lerp_scalar(2.0, 20.0, 0.1), 3.8);
/// ```
///
/// # See also:
///
/// * [`lerp`](fn.lerp.html)
/// * [`lerp_vec`](fn.lerp_vec.html)
pub fn lerp_scalar<N: Number>(x: N, y: N, a: N) -> N {
    mix_scalar(x, y, a)
}

/// Returns `x * (1.0 - a) + y * a`, i.e., the linear blend of the vectors x and y using the scalar value a.
///
/// The value for a is not restricted to the range `[0, 1]`.
/// This is an alias for `mix`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let x = glm::vec3(1.0, 2.0, 3.0);
/// let y = glm::vec3(10.0, 20.0, 30.0);
/// assert_eq!(glm::lerp(&x, &y, 0.1), glm::vec3(1.9, 3.8, 5.7));
/// ```
///
/// # See also:
///
/// * [`lerp_scalar`](fn.lerp_scalar.html)
/// * [`lerp_vec`](fn.lerp_vec.html)
pub fn lerp<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>, a: N) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    mix(x, y, a)
}

/// Returns `x * (1.0 - a) + y * a`, i.e., the component-wise linear blend of `x` and `y` using the components of
/// the vector `a` as coefficients.
///
/// The value for a is not restricted to the range `[0, 1]`.
/// This is an alias for `mix_vec`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let x = glm::vec3(1.0, 2.0, 3.0);
/// let y = glm::vec3(10.0, 20.0, 30.0);
/// let a = glm::vec3(0.1, 0.2, 0.3);
/// assert_eq!(glm::lerp_vec(&x, &y, &a), glm::vec3(1.9, 5.6, 11.1));
/// ```
///
/// # See also:
///
/// * [`lerp_scalar`](fn.lerp_scalar.html)
/// * [`lerp`](fn.lerp.html)
pub fn lerp_vec<N: Number, D: Dimension>(
    x: &TVec<N, D>,
    y: &TVec<N, D>,
    a: &TVec<N, D>,
) -> TVec<N, D>
where
    DefaultAllocator: Alloc<N, D>,
{
    mix_vec(x, y, a)
}

/// Component-wise modulus.
///
/// Returns `x - y * floor(x / y)` for each component in `x` using the corresponding component of `y`.
///
/// # See also:
///
/// * [`modf`](fn.modf.html)
pub fn modf_vec<N: Number, D: Dimension>(x: &TVec<N, D>, y: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.zip_map(y, |x, y| x % y)
}

/// Modulus between two values.
///
/// # See also:
///
/// * [`modf_vec`](fn.modf_vec.html)
pub fn modf<N: Number>(x: N, i: N) -> N {
    x % i
}

/// Component-wise rounding.
///
/// Values equal to `0.5` are rounded away from `0.0`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec4(-1.5, 0.6, 1.5, -3.2);
/// assert_eq!(glm::vec4(-2.0, 1.0, 2.0, -3.0), glm::round(&vec));
/// ```
///
/// # See also:
///
/// * [`ceil`](fn.ceil.html)
/// * [`floor`](fn.floor.html)
/// * [`fract`](fn.fract.html)
/// * [`trunc`](fn.trunc.html)
pub fn round<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.round())
}

//pub fn roundEven<N: Scalar, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
//    where DefaultAllocator: Alloc<N, D> {
//        unimplemented!()
//}

/// For each vector component `x`:  1 if `x > 0`, 0 if `x == 0`, or -1 if `x < 0`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec4(-2.0, 0.0, -0.0, 2.0);
/// assert_eq!(glm::vec4(-1.0, 0.0, 0.0, 1.0), glm::sign(&vec));
/// ```
///
/// # See also:
///
/// * [`abs`](fn.abs.html)
///
pub fn sign<N: Number, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| if x.is_zero() { N::zero() } else { x.signum() })
}

/// Returns 0.0 if `x <= edge0` and `1.0 if x >= edge1` and performs smooth Hermite interpolation between 0 and 1 when `edge0 < x < edge1`.
///
/// This is useful in cases where you would want a threshold function with a smooth transition.
/// This is equivalent to: `let result = clamp((x - edge0) / (edge1 - edge0), 0, 1); return t * t * (3 - 2 * t);` Results are undefined if `edge0 >= edge1`.
pub fn smoothstep<N: Number>(edge0: N, edge1: N, x: N) -> N {
    let _3: N = FromPrimitive::from_f64(3.0).unwrap();
    let _2: N = FromPrimitive::from_f64(2.0).unwrap();
    let t = na::clamp((x - edge0) / (edge1 - edge0), N::zero(), N::one());
    t * t * (_3 - t * _2)
}

/// Returns 0.0 if `x < edge`, otherwise it returns 1.0.
pub fn step_scalar<N: Number>(edge: N, x: N) -> N {
    if edge > x {
        N::zero()
    } else {
        N::one()
    }
}

/// Returns 0.0 if `x[i] < edge`, otherwise it returns 1.0.
pub fn step<N: Number, D: Dimension>(edge: N, x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| step_scalar(edge, x))
}

/// Returns 0.0 if `x[i] < edge[i]`, otherwise it returns 1.0.
pub fn step_vec<N: Number, D: Dimension>(edge: &TVec<N, D>, x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    edge.zip_map(x, step_scalar)
}

/// Returns a value equal to the nearest integer to `x` whose absolute value is not larger than the absolute value of `x`.
///
/// # Examples:
///
/// ```
/// # use nalgebra_glm as glm;
/// let vec = glm::vec3(-1.5, 0.5, 2.8);
/// assert_eq!(glm::vec3(-1.0, 0.0, 2.0), glm::trunc(&vec));
/// ```
///
/// # See also:
///
/// * [`ceil`](fn.ceil.html)
/// * [`floor`](fn.floor.html)
/// * [`fract`](fn.fract.html)
/// * [`round`](fn.round.html)
pub fn trunc<N: Real, D: Dimension>(x: &TVec<N, D>) -> TVec<N, D>
where DefaultAllocator: Alloc<N, D> {
    x.map(|x| x.trunc())
}

/// Returns a floating-point value corresponding to a unsigned integer encoding of a floating-point value.
///
/// If an `inf` or `NaN` is passed in, it will not signal, and the resulting floating point value is unspecified. Otherwise, the bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float`](fn.uint_bits_to_float.html)
pub fn uint_bits_to_float_scalar(v: u32) -> f32 {
    f32::from_bits(v)
}

/// For each component of `v`, returns a floating-point value corresponding to a unsigned integer encoding of a floating-point value.
///
/// If an inf or NaN is passed in, it will not signal, and the resulting floating point value is unspecified. Otherwise, the bit-level representation is preserved.
///
/// # See also:
///
/// * [`float_bits_to_int`](fn.float_bits_to_int.html)
/// * [`float_bits_to_int_vec`](fn.float_bits_to_int_vec.html)
/// * [`float_bits_to_uint`](fn.float_bits_to_uint.html)
/// * [`float_bits_to_uint_vec`](fn.float_bits_to_uint_vec.html)
/// * [`int_bits_to_float`](fn.int_bits_to_float.html)
/// * [`int_bits_to_float_vec`](fn.int_bits_to_float_vec.html)
/// * [`uint_bits_to_float_scalar`](fn.uint_bits_to_float_scalar.html)
pub fn uint_bits_to_float<D: Dimension>(v: &TVec<u32, D>) -> TVec<f32, D>
where DefaultAllocator: Alloc<f32, D> {
    v.map(uint_bits_to_float_scalar)
}
