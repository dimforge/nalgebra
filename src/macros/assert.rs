/// Asserts approximate equality within a given tolerance of two values with the
/// `ApproxEq` trait.
#[macro_export]
macro_rules! assert_approx_eq_eps(
    ($given: expr, $expected: expr, $eps: expr) => ({
        let eps = &($eps);
        let (given_val, expected_val) = (&($given), &($expected));
        if !ApproxEq::approx_eq_eps(given_val, expected_val, eps) {
            panic!("assertion failed: `left ≈ right` (left: `{}`, right: `{}`, tolerance: `{}`)",
                *given_val, *expected_val, *eps
            )
        }
    })
);

/// Asserts approximate equality of two values with the `ApproxEq` trait.
#[macro_export]
macro_rules! assert_approx_eq(
    ($given: expr, $expected: expr) => ({
        let (given_val, expected_val) = (&($given), &($expected));
        if !ApproxEq::approx_eq(given_val, expected_val) {
            panic!("assertion failed: `left ≈ right` (left: `{}`, right: `{}`, tolerance: `{}`)",
                *given_val, *expected_val,
                ApproxEq::approx_epsilon(Some(*given_val))
            )
        }
    })
);
