//! Macros for `nalgebra`.
//!
//! This crate is not intended for direct consumption. Instead, the macros are re-exported by
//! `nalgebra` if the `macros` feature is enabled (enabled by default).

#![deny(
    nonstandard_style,
    unused,
    missing_docs,
    rust_2018_idioms,
    rust_2018_compatibility,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    clippy::all,
    clippy::pedantic
)]

mod matrix_vector_impl;
mod stack_impl;

use matrix_vector_impl::{Matrix, Vector};

use crate::matrix_vector_impl::{dmatrix_impl, dvector_impl, matrix_impl, vector_impl};
use proc_macro::TokenStream;
use quote::quote;
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

/// Constructs a new matrix by concatenating the given matrices.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// The syntax is similar to the [`matrix_vector_macros!`] and [`dmatrix!`]) macros. However the elements should
/// be of type `Matrix`, `&Matrix` or be one of the litterals `0` or `1`. The elements of type `Matrix` and `&Matrix` are
/// concatenated as expected. The litteral `0` is expanded to the zero.Note that at least one element
/// in each row and column must be an expression of type `Matrix` or `&Matrix`.
///
/// All elements in the same row need to have the same number of rows and simillary for the
/// elements in the same column. This is checked at compile time as long as all elements have
/// dimensions of type `Const<N>`. If one or more elements have dimensions of type `Dynamic` then
/// it is checked at runtime that all dimensions match.
///
/// If at least one element of a row has `Const<N>` number of rows then that row has a whole will
/// have `Const<N>` number of rows. However if at least one row has has `Dynamic` number of rows
/// then the entire matrix will have `Dynamic` number of rows. Same for columns.
///
/// # Examples
///
/// ```
/// use nalgebra::{stack, matrix};
///
/// let a = matrix![1,2;3,4;];
///
/// let m1 = stack![
///     a, 0;
///     0, &matrix![5,6;7,8;];
/// ];
///
/// let m2 = matrix![
///     1,2,0,0;
///     3,4,0,0;
///     0,0,5,6;
///     0,0,7,8;
/// ];
///
/// assert_eq!(m1, m2);
/// ```
///
/// ```
/// use nalgebra::{stack, matrix, Matrix5x6};
///
/// let a: Matrix5x6<_> = stack![
///     0, matrix![1;2], 0;
///     0, 0, matrix![3,4;5,6;];
///     matrix![7,8,9;], 0, 0;
/// ];
///
/// let b = matrix![
///     0, 0, 0, 1, 0, 0;
///     0, 0, 0, 2, 0, 0;
///     0, 0, 0, 0, 3, 4;
///     0, 0, 0, 0, 5, 6;
///     7, 8, 9, 0, 0, 0;
/// ];
///
/// assert_eq!(a, b);
/// ```
///
/// ```
/// use nalgebra::{stack, matrix, dmatrix, DMatrix};
///
/// let a: DMatrix<_> = stack![
///     dmatrix![1,2;3,4;], 0;
///     0, matrix![5,6;7,8;];
/// ];
///
/// let b = dmatrix![
///     1,2,0,0;
///     3,4,0,0;
///     0,0,5,6;
///     0,0,7,8;
/// ];
///
/// assert_eq!(a, b);
/// ```
///
/// ```
/// use nalgebra::{stack, matrix, dmatrix, Matrix4};
///
/// let a: Matrix4<_> = stack![
///     &dmatrix![1,2;3,4;], &matrix![5,6;7,8;];
///     &matrix![9,10;11,12;], 0;
/// ];
///
/// let b = matrix![
///     1,2,5,6;
///     3,4,7,8;
///     9,10,0,0;
///     11,12,0,0;
/// ];
///
/// assert_eq!(a, b);
/// ```
#[proc_macro]
pub fn stack(stream: TokenStream) -> TokenStream {
    let matrix = parse_macro_input!(stream as Matrix);
    proc_macro::TokenStream::from(match stack_impl::stack_impl("__11f075cdd4a86538", matrix) {
        Ok(res) => res,
        Err(err) => err.into_compile_error(),
    })
}
