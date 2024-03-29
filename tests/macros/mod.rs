mod matrix;
mod stack;

/// Wrapper for `assert_eq` that also asserts that the types are the same
macro_rules! assert_eq_and_type {
    ($left:expr, $right:expr $(,)?) => {
        {
            fn check_statically_same_type<T>(_: &T, _: &T) {}
            check_statically_same_type(&$left, &$right);
        }
        assert_eq!($left, $right);
    };
}

pub(crate) use assert_eq_and_type;