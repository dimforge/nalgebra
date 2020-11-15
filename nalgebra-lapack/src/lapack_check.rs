#![macro_use]

macro_rules! lapack_check(
    ($info: expr) => (
        // TODO: return a richer error.
        if $info != 0 {
            return None;
        }
        // if $info < 0 {
        //     return Err(Error::from(ErrorKind::LapackIllegalArgument(-$info)));
        // } else if $info > 0 {
        //     return Err(Error::from(ErrorKind::LapackFailure($info)));
        // }
    );
);

macro_rules! lapack_panic(
    ($info: expr) => (
        assert!($info == 0, "Lapack error.");
    );
);

macro_rules! lapack_test(
    ($info: expr) => (
        $info == 0
    );
);
