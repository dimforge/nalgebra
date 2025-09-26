//! Macros for `nalgebra`.
//!
//! This crate is not intended for direct consumption. Instead, the macros are re-exported by
//! `nalgebra` if the `macros` feature is enabled (enabled by default).

#![deny(
    nonstandard_style,
    unused,
    missing_docs,
    rust_2018_idioms,
    rust_2024_compatibility,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    clippy::all
)]

mod matrix_vector_impl;
mod stack_impl;

use matrix_vector_impl::{Matrix, Vector};

use crate::matrix_vector_impl::{dmatrix_impl, dvector_impl, matrix_impl, vector_impl};
use proc_macro::TokenStream;
use quote::quote;
use stack_impl::stack_impl;
use syn::parse_macro_input;

/// Construct a fixed-size matrix directly from data.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// This macro facilitates easy construction of matrices when the entries of the matrix are known
/// (either as constants or expressions). This macro produces an instance of `SMatrix`. This means
/// that the data of the matrix is stored on the stack, and its dimensions are fixed at
/// compile-time. If you want to construct a dynamic matrix, use [`dmatrix!`] instead.
///
/// `matrix!` is intended to be both the simplest and most efficient way to construct (small)
/// matrices, and can also be used in *const fn* contexts.
///
/// The syntax is MATLAB-like. Column elements are separated by a comma (`,`), and a semi-colon
/// (`;`) designates that a new row begins.
///
/// # Examples
/// ```
/// use nalgebra::matrix;
///
/// // Produces a Matrix3<_> == SMatrix<_, 3, 3>
/// let a = matrix![1, 2, 3;
///                 4, 5, 6;
///                 7, 8, 9];
/// ```
///
/// You can construct matrices with arbitrary expressions for its elements:
///
/// ```
/// use nalgebra::{matrix, Matrix2};
/// let theta = 0.45f64;
///
/// let r = matrix![theta.cos(), - theta.sin();
///                 theta.sin(),   theta.cos()];
/// ```
#[proc_macro]
pub fn matrix(stream: TokenStream) -> TokenStream {
    matrix_impl(stream)
}

/// Construct a dynamic matrix directly from data.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// The syntax is exactly the same as for [`matrix!`], but instead of producing instances of
/// `SMatrix`, it produces instances of `DMatrix`. At the moment it is not usable
/// in `const fn` contexts.
///
/// # Example
/// ```
/// use nalgebra::dmatrix;
///
/// // Produces a DMatrix<_>
/// let a = dmatrix![1, 2, 3;
///                  4, 5, 6;
///                  7, 8, 9];
/// ```
#[proc_macro]
pub fn dmatrix(stream: TokenStream) -> TokenStream {
    dmatrix_impl(stream)
}

/// Construct a fixed-size column vector directly from data.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// Similarly to [`matrix!`], this macro facilitates easy construction of fixed-size vectors.
/// However, whereas the [`matrix!`] macro expects each row to be separated by a semi-colon,
/// the syntax of this macro is instead similar to `vec!`, in that the elements of the vector
/// are simply listed consecutively.
///
/// `vector!` is intended to be the most readable and performant way of constructing small,
/// fixed-size vectors, and it is usable in `const fn` contexts.
///
/// # Example
/// ```
/// use nalgebra::vector;
///
/// // Produces a Vector3<_> == SVector<_, 3>
/// let v = vector![1, 2, 3];
/// ```
#[proc_macro]
pub fn vector(stream: TokenStream) -> TokenStream {
    vector_impl(stream)
}

/// Construct a dynamic column vector directly from data.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// The syntax is exactly the same as for [`vector!`], but instead of producing instances of
/// `SVector`, it produces instances of `DVector`. At the moment it is not usable
/// in `const fn` contexts.
///
/// # Example
/// ```
/// use nalgebra::dvector;
///
/// // Produces a DVector<_>
/// let v = dvector![1, 2, 3];
/// ```
#[proc_macro]
pub fn dvector(stream: TokenStream) -> TokenStream {
    dvector_impl(stream)
}

/// Construct a fixed-size point directly from data.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// Similarly to [`vector!`], this macro facilitates easy construction of points.
///
/// `point!` is intended to be the most readable and performant way of constructing small,
/// points, and it is usable in `const fn` contexts.
///
/// # Example
/// ```
/// use nalgebra::point;
///
/// // Produces a Point3<_>
/// let v = point![1, 2, 3];
/// ```
#[proc_macro]
pub fn point(stream: TokenStream) -> TokenStream {
    let vector = parse_macro_input!(stream as Vector);
    let len = vector.len();
    let array_tokens = vector.to_array_tokens();
    let output = quote! {
        nalgebra::Point::<_, #len> {
            coords: nalgebra::SVector::<_, #len>
                        ::from_array_storage(nalgebra::ArrayStorage([#array_tokens]))
        }
    };
    proc_macro::TokenStream::from(output)
}

/// Construct a new matrix by stacking matrices in a block matrix.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// This macro facilitates the construction of
/// [block matrices](https://en.wikipedia.org/wiki/Block_matrix)
/// by stacking blocks (matrices) using the same MATLAB-like syntax as the [`matrix!`] and
/// [`dmatrix!`] macros:
///
/// ```rust
/// # use nalgebra::stack;
/// #
/// # fn main() {
/// # let [a, b, c, d] = std::array::from_fn(|_| nalgebra::Matrix1::new(0));
/// // a, b, c and d are matrices
/// let block_matrix = stack![ a, b;
///                            c, d ];
/// # }
/// ```
///
/// The resulting matrix is stack-allocated if the dimension of each block row and column
/// can be determined at compile-time, otherwise it is heap-allocated.
/// This is the case if, for every row, there is at least one matrix with a fixed number of rows,
/// and, for every column, there is at least one matrix with a fixed number of columns.
///
/// [`stack!`] also supports special syntax to indicate zero blocks in a matrix:
///
/// ```rust
/// # use nalgebra::stack;
/// #
/// # fn main() {
/// # let [a, b, c, d] = std::array::from_fn(|_| nalgebra::Matrix1::new(0));
/// // a and d are matrices
/// let block_matrix = stack![ a, 0;
///                            0, d ];
/// # }
/// ```
/// Here, the `0` literal indicates a zero matrix of implicitly defined size.
/// In order to infer the size of the zero blocks, there must be at least one matrix
/// in every row and column of the matrix.
/// In other words, no row or column can consist entirely of implicit zero blocks.
///
/// # Panics
///
/// Panics if dimensions are inconsistent and it cannot be determined at compile-time.
///
/// # Examples
///
/// ```
/// use nalgebra::{matrix, SMatrix, stack};
///
/// let a = matrix![1, 2;
///                 3, 4];
/// let b = matrix![5, 6;
///                 7, 8];
/// let c = matrix![9, 10];
///
/// let block_matrix = stack![ a, b;
///                            c, 0 ];
///
/// assert_eq!(block_matrix, matrix![1,  2,  5,  6;
///                                  3,  4,  7,  8;
///                                  9, 10,  0,  0]);
///
/// // Verify that the resulting block matrix is stack-allocated
/// let _: SMatrix<_, 3, 4> = block_matrix;
/// ```
///
/// The example above shows how stacking stack-allocated matrices results in a stack-allocated
/// block matrix. If all row and column dimensions can not be determined at compile-time,
/// the result is instead a dynamically allocated matrix:
///
/// ```
/// use nalgebra::{dmatrix, DMatrix, Dyn, matrix, OMatrix, SMatrix, stack, U3};
///
/// # let a = matrix![1, 2; 3, 4]; let c = matrix![9, 10];
/// // a and c as before, but b is a dynamic matrix this time
/// let b = dmatrix![5, 6;
///                  7, 8];
///
/// // In this case, the number of rows can be statically inferred to be 3 (U3),
/// // but the number of columns cannot, hence it is dynamic
/// let block_matrix: OMatrix<_, U3, Dyn> = stack![ a, b;
///                                                 c, 0 ];
///
/// // If necessary, a fully dynamic matrix (DMatrix) can be obtained by reshaping
/// let dyn_block_matrix: DMatrix<_> = block_matrix.reshape_generic(Dyn(3), Dyn(4));
/// ```
/// Note that explicitly annotating the types of `block_matrix` and `dyn_block_matrix` is
/// only made for illustrative purposes, and is not generally necessary.
///
#[proc_macro]
pub fn stack(stream: TokenStream) -> TokenStream {
    let matrix = parse_macro_input!(stream as Matrix);
    proc_macro::TokenStream::from(stack_impl(matrix).unwrap_or_else(syn::Error::into_compile_error))
}
