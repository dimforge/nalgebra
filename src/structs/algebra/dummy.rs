#![macro_use]

macro_rules! vector_space_impl(
    ($t: ident, $dimension: expr, $($compN: ident),+) => { }
);

macro_rules! special_orthogonal_group_impl(
    ($t: ident, $point: ident, $vector: ident) => { }
);

macro_rules! euclidean_space_impl(
    ($t: ident, $vector: ident) => { }
);

macro_rules! matrix_group_approx_impl(
    ($t: ident, $($compN: ident),+) => { }
);
