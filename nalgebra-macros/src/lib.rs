//! Macros for `nalgebra`.
//!
//! This crate is not intended for direct consumption. Instead, the macros are re-exported by
//! `nalgebra` if the `macros` feature is enabled (enabled by default).

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote, ToTokens, TokenStreamExt};
use syn::parse::{Error, Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::Expr;
use syn::{parse_macro_input, Token};

use proc_macro2::{Delimiter, Spacing, TokenStream as TokenStream2, TokenTree};
use proc_macro2::{Group, Punct};

struct Matrix {
    // Represent the matrix as a row-major vector of vectors of expressions
    rows: Vec<Vec<Expr>>,
    ncols: usize,
}

impl Matrix {
    fn nrows(&self) -> usize {
        self.rows.len()
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    /// Produces a stream of tokens representing this matrix as a column-major nested array.
    fn to_col_major_nested_array_tokens(&self) -> TokenStream2 {
        let mut result = TokenStream2::new();
        for j in 0..self.ncols() {
            let mut col = TokenStream2::new();
            let col_iter = (0..self.nrows()).map(move |i| &self.rows[i][j]);
            col.append_separated(col_iter, Punct::new(',', Spacing::Alone));
            result.append(Group::new(Delimiter::Bracket, col));
            result.append(Punct::new(',', Spacing::Alone));
        }
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, result)))
    }

    /// Produces a stream of tokens representing this matrix as a column-major flat array
    /// (suitable for representing e.g. a `DMatrix`).
    fn to_col_major_flat_array_tokens(&self) -> TokenStream2 {
        let mut data = TokenStream2::new();
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self.rows[i][j].to_tokens(&mut data);
                data.append(Punct::new(',', Spacing::Alone));
            }
        }
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, data)))
    }
}

type MatrixRowSyntax = Punctuated<Expr, Token![,]>;

impl Parse for Matrix {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut rows = Vec::new();
        let mut ncols = None;

        while !input.is_empty() {
            let row_span = input.span();
            let row = MatrixRowSyntax::parse_separated_nonempty(input)?;

            if let Some(ncols) = ncols {
                if row.len() != ncols {
                    let row_idx = rows.len();
                    let error_msg = format!(
                        "Unexpected number of entries in row {}. Expected {}, found {} entries.",
                        row_idx,
                        ncols,
                        row.len()
                    );
                    return Err(Error::new(row_span, error_msg));
                }
            } else {
                ncols = Some(row.len());
            }
            rows.push(row.into_iter().collect());

            // We've just read a row, so if there are more tokens, there must be a semi-colon,
            // otherwise the input is malformed
            if !input.is_empty() {
                input.parse::<Token![;]>()?;
            }
        }

        Ok(Self {
            rows,
            ncols: ncols.unwrap_or(0),
        })
    }
}

/// Construct a fixed-size matrix directly from data.
///
/// **Note: Requires the `macro` feature to be enabled (enabled by default)**.
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
///
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
    let matrix = parse_macro_input!(stream as Matrix);

    let row_dim = matrix.nrows();
    let col_dim = matrix.ncols();

    let array_tokens = matrix.to_col_major_nested_array_tokens();

    //  TODO: Use quote_spanned instead??
    let output = quote! {
        nalgebra::SMatrix::<_, #row_dim, #col_dim>
            ::from_array_storage(nalgebra::ArrayStorage(#array_tokens))
    };

    proc_macro::TokenStream::from(output)
}

/// Construct a dynamic matrix directly from data.
///
/// **Note: Requires the `macro` feature to be enabled (enabled by default)**.
///
/// The syntax is exactly the same as for [`matrix!`], but instead of producing instances of
/// `SMatrix`, it produces instances of `DMatrix`. At the moment it is not usable
/// in `const fn` contexts.
///
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
    let matrix = parse_macro_input!(stream as Matrix);

    let row_dim = matrix.nrows();
    let col_dim = matrix.ncols();

    let array_tokens = matrix.to_col_major_flat_array_tokens();

    //  TODO: Use quote_spanned instead??
    let output = quote! {
        nalgebra::DMatrix::<_>
            ::from_vec_storage(nalgebra::VecStorage::new(
                nalgebra::Dynamic::new(#row_dim),
                nalgebra::Dynamic::new(#col_dim),
                vec!#array_tokens))
    };

    proc_macro::TokenStream::from(output)
}

struct Vector {
    elements: Vec<Expr>,
}

impl Vector {
    fn to_array_tokens(&self) -> TokenStream2 {
        let mut data = TokenStream2::new();
        data.append_separated(&self.elements, Punct::new(',', Spacing::Alone));
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, data)))
    }

    fn len(&self) -> usize {
        self.elements.len()
    }
}

impl Parse for Vector {
    fn parse(input: ParseStream) -> Result<Self> {
        // The syntax of a vector is just the syntax of a single matrix row
        if input.is_empty() {
            Ok(Self {
                elements: Vec::new(),
            })
        } else {
            let elements = MatrixRowSyntax::parse_separated_nonempty(input)?
                .into_iter()
                .collect();
            Ok(Self { elements })
        }
    }
}

/// Construct a fixed-size column vector directly from data.
///
/// **Note: Requires the `macro` feature to be enabled (enabled by default)**.
///
/// Similarly to [`matrix!`], this macro facilitates easy construction of fixed-size vectors.
/// However, whereas the [`matrix!`] macro expects each row to be separated by a semi-colon,
/// the syntax of this macro is instead similar to `vec!`, in that the elements of the vector
/// are simply listed consecutively.
///
/// `vector!` is intended to be the most readable and performant way of constructing small,
/// fixed-size vectors, and it is usable in `const fn` contexts.
///
/// ## Examples
///
/// ```
/// use nalgebra::vector;
///
/// // Produces a Vector3<_> == SVector<_, 3>
/// let v = vector![1, 2, 3];
/// ```
#[proc_macro]
pub fn vector(stream: TokenStream) -> TokenStream {
    let vector = parse_macro_input!(stream as Vector);
    let len = vector.len();
    let array_tokens = vector.to_array_tokens();
    let output = quote! {
        nalgebra::SVector::<_, #len>
            ::from_array_storage(nalgebra::ArrayStorage([#array_tokens]))
    };
    proc_macro::TokenStream::from(output)
}

/// Construct a dynamic column vector directly from data.
///
/// **Note: Requires the `macro` feature to be enabled (enabled by default)**.
///
/// The syntax is exactly the same as for [`vector!`], but instead of producing instances of
/// `SVector`, it produces instances of `DVector`. At the moment it is not usable
/// in `const fn` contexts.
///
/// ```
/// use nalgebra::dvector;
///
/// // Produces a DVector<_>
/// let v = dvector![1, 2, 3];
/// ```
#[proc_macro]
pub fn dvector(stream: TokenStream) -> TokenStream {
    let vector = parse_macro_input!(stream as Vector);
    let len = vector.len();
    let array_tokens = vector.to_array_tokens();
    let output = quote! {
        nalgebra::DVector::<_>
            ::from_vec_storage(nalgebra::VecStorage::new(
                nalgebra::Dynamic::new(#len),
                nalgebra::Const::<1>,
                vec!#array_tokens))
    };
    proc_macro::TokenStream::from(output)
}

/// Construct a fixed-size point directly from data.
///
/// **Note: Requires the `macro` feature to be enabled (enabled by default)**.
///
/// Similarly to [`vector!`], this macro facilitates easy construction of points.
///
/// `point!` is intended to be the most readable and performant way of constructing small,
/// points, and it is usable in `const fn` contexts.
///
/// ## Examples
///
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
