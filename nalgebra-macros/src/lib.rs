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

use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens, TokenStreamExt};
use syn::parse::{Error, Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Token};
use syn::{Expr, Lit};

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
    fn parse(input: ParseStream<'_>) -> Result<Self> {
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
    let matrix = parse_macro_input!(stream as Matrix);

    let row_dim = matrix.nrows();
    let col_dim = matrix.ncols();

    let array_tokens = matrix.to_col_major_flat_array_tokens();

    //  TODO: Use quote_spanned instead??
    let output = quote! {
        nalgebra::DMatrix::<_>
            ::from_vec_storage(nalgebra::VecStorage::new(
                nalgebra::Dyn(#row_dim),
                nalgebra::Dyn(#col_dim),
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
    fn parse(input: ParseStream<'_>) -> Result<Self> {
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
    let vector = parse_macro_input!(stream as Vector);
    let len = vector.len();
    let array_tokens = vector.to_array_tokens();
    let output = quote! {
        nalgebra::DVector::<_>
            ::from_vec_storage(nalgebra::VecStorage::new(
                nalgebra::Dyn(#len),
                nalgebra::Const::<1>,
                vec!#array_tokens))
    };
    proc_macro::TokenStream::from(output)
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

enum ConcatElem {
    Zero,
    Expr(Expr),
}

impl ConcatElem {
    fn from_expr(expr: Expr) -> Self {
        if let Expr::Lit(syn::ExprLit {
            lit: Lit::Int(ilit),
            ..
        }) = &expr
        {
            if ilit.base10_digits() == "0" {
                return ConcatElem::Zero;
            }
        }
        ConcatElem::Expr(expr)
    }

    fn is_expr(&self) -> bool {
        matches!(self, ConcatElem::Expr(_))
    }
}

/// Constructs a new matrix by concatenating the given matrices.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// The syntax is similar to the [`matrix!`] and [`dmatrix!`]) macros. However the elements should
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
/// use nalgebra::{cat, matrix};
///
/// let a = matrix![1,2;3,4;];
///
/// let m1 = cat![
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
/// use nalgebra::{cat, matrix, Matrix5x6};
///
/// let a: Matrix5x6<_> = cat![
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
/// use nalgebra::{cat, matrix, dmatrix, DMatrix};
///
/// let a: DMatrix<_> = cat![
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
/// use nalgebra::{cat, matrix, dmatrix, Matrix4};
///
/// let a: Matrix4<_> = cat![
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
pub fn cat(stream: TokenStream) -> TokenStream {
    let matrix = parse_macro_input!(stream as Matrix);
    proc_macro::TokenStream::from(cat_impl("__11f075cdd4a86538", matrix))
}

#[allow(clippy::too_many_lines)]
fn cat_impl(prefix: &str, matrix: Matrix) -> TokenStream2 {
    let n_macro_rows = matrix.nrows();
    let n_macro_cols = matrix.ncols();

    let rows = matrix
        .rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(ConcatElem::from_expr)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut output = quote! {};

    for (i, row) in rows.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if let ConcatElem::Expr(expr) = cell {
                let ident = format_ident!("{}_cat_{}_{}", prefix, i, j);
                let ident_shape = format_ident!("{}_cat_{}_{}_shape", prefix, i, j);
                output.extend(std::iter::once(quote_spanned! {expr.span()=>
                    let #ident = #expr;
                    let #ident_shape = #ident.shape_generic();
                }));
            }
        }
    }

    for (i, row) in rows.iter().enumerate() {
        let size = row.iter().enumerate().filter(|(_, c)| c.is_expr()).map(|(j, _)| {
            let ident_shape = format_ident!("{}_cat_{}_{}_shape", prefix, i, j);
            quote!{ #ident_shape.0 }
        }).reduce(|a, b| quote!{
            <nalgebra::constraint::ShapeConstraint as nalgebra::constraint::DimEq<_, _>>::representative(#a, #b)
                .expect("The concatenated matrices do not have the same number of columns")
        }).expect("At least one element in each row must be an expression of type `Matrix`");

        let size_ident = format_ident!("{}_cat_row_{}_size", prefix, i);
        let offset_ident = format_ident!("{}_cat_row_{}_offset", prefix, i);

        let offset = if i == 0 {
            quote! { 0 }
        } else {
            let last_offset_ident = format_ident!("{}_cat_row_{}_offset", prefix, i - 1);
            let last_size_ident = format_ident!("{}_cat_row_{}_size", prefix, i - 1);
            quote! { #last_offset_ident + <_ as nalgebra::Dim>::value(&#last_size_ident) }
        };

        output.extend(std::iter::once(quote! {
            let #size_ident = #size;
            let #offset_ident = #offset;
        }));
    }

    for j in 0..n_macro_cols {
        let size = (0..n_macro_rows).filter(|i| rows[*i][j].is_expr()).map(|i| {
            let ident_shape = format_ident!("{}_cat_{}_{}_shape", prefix, i, j);
            quote!{ #ident_shape.1 }
        }).reduce(|a, b| quote!{
            <nalgebra::constraint::ShapeConstraint as nalgebra::constraint::DimEq<_, _>>::representative(#a, #b)
                .expect("The concatenated matrices do not have the same number of rows")
        }).expect("At least one element in each column must be an expression of type `Matrix`");

        let size_ident = format_ident!("{}_cat_col_{}_size", prefix, j);
        let offset_ident = format_ident!("{}_cat_col_{}_offset", prefix, j);

        let offset = if j == 0 {
            quote! { 0 }
        } else {
            let last_offset_ident = format_ident!("{}_cat_col_{}_offset", prefix, j - 1);
            let last_size_ident = format_ident!("{}_cat_col_{}_size", prefix, j - 1);
            quote! { #last_offset_ident + <_ as nalgebra::Dim>::value(&#last_size_ident) }
        };

        output.extend(std::iter::once(quote! {
            let #size_ident = #size;
            let #offset_ident = #offset;
        }));
    }

    let num_rows = (0..n_macro_rows)
        .map(|i| {
            let ident = format_ident!("{}_cat_row_{}_size", prefix, i);
            quote! { #ident }
        })
        .reduce(|a, b| {
            quote! {
                <_ as nalgebra::DimAdd<_>>::add(#a, #b)
            }
        })
        .expect("More than zero rows in concatenation");

    let num_cols = (0..n_macro_cols)
        .map(|j| {
            let ident = format_ident!("{}_cat_col_{}_size", prefix, j);
            quote! { #ident }
        })
        .reduce(|a, b| {
            quote! {
                <_ as nalgebra::DimAdd<_>>::add(#a, #b)
            }
        })
        .unwrap();

    // It should be possible to use `uninitialized_generic` here instead
    // however that would mean that the macro needs to generate unsafe code
    // which does not seem like a great idea.
    output.extend(std::iter::once(quote! {
        let mut matrix = nalgebra::Matrix::zeros_generic(#num_rows, #num_cols);
    }));

    for (i, row) in rows.into_iter().enumerate() {
        for (j, cell) in row.into_iter().enumerate() {
            let row_size = format_ident!("{}_cat_row_{}_size", prefix, i);
            let col_size = format_ident!("{}_cat_col_{}_size", prefix, j);
            let row_offset = format_ident!("{}_cat_row_{}_offset", prefix, i);
            let col_offset = format_ident!("{}_cat_col_{}_offset", prefix, j);
            match cell {
                ConcatElem::Zero => (),
                ConcatElem::Expr(_) => {
                    let expr_ident = format_ident!("{}_cat_{}_{}", prefix, i, j);
                    output.extend(std::iter::once(quote! {
                        let start = (#row_offset, #col_offset);
                        let shape = (#row_size, #col_size);
                        let input_view = #expr_ident.generic_view((0, 0), shape);
                        let mut output_view = matrix.generic_view_mut(start, shape);
                        output_view.copy_from(&input_view);
                    }));
                }
            }
        }
    }

    quote! {
        {
            #output
            matrix
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cat_simple_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0;
            0, b;
        ];

        let result = cat_impl("", input);

        let expected = quote! {{
            let _cat_0_0 = a;
            let _cat_0_0_shape = _cat_0_0.shape_generic();
            let _cat_1_1 = b;
            let _cat_1_1_shape = _cat_1_1.shape_generic();
            let _cat_row_0_size = _cat_0_0_shape.0;
            let _cat_row_0_offset = 0;
            let _cat_row_1_size = _cat_1_1_shape.0;
            let _cat_row_1_offset = _cat_row_0_offset + <_ as nalgebra::Dim>::value(&_cat_row_0_size);
            let _cat_col_0_size = _cat_0_0_shape.1;
            let _cat_col_0_offset = 0;
            let _cat_col_1_size = _cat_1_1_shape.1;
            let _cat_col_1_offset = _cat_col_0_offset + <_ as nalgebra::Dim>::value(&_cat_col_0_size);
            let mut matrix = nalgebra::Matrix::zeros_generic(
                <_ as nalgebra::DimAdd<_>>::add(_cat_row_0_size, _cat_row_1_size),
                <_ as nalgebra::DimAdd<_>>::add(_cat_col_0_size, _cat_col_1_size)
            );
            let start = (_cat_row_0_offset, _cat_col_0_offset);
            let shape = (_cat_row_0_size, _cat_col_0_size);
            let input_view = _cat_0_0.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (_cat_row_1_offset, _cat_col_1_offset);
            let shape = (_cat_row_1_size, _cat_col_1_size);
            let input_view = _cat_1_1.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            matrix
        }};

        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn cat_complex_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0, b;
            0, c, d;
            e, 0, 0;
        ];

        let result = cat_impl("", input);

        let expected = quote! {{
            let _cat_0_0 = a;
            let _cat_0_0_shape = _cat_0_0.shape_generic();
            let _cat_0_2 = b;
            let _cat_0_2_shape = _cat_0_2.shape_generic();
            let _cat_1_1 = c;
            let _cat_1_1_shape = _cat_1_1.shape_generic();
            let _cat_1_2 = d;
            let _cat_1_2_shape = _cat_1_2.shape_generic();
            let _cat_2_0 = e;
            let _cat_2_0_shape = _cat_2_0.shape_generic();
            let _cat_row_0_size = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: DimEq < _ , _ >> :: representative (_cat_0_0_shape . 0 , _cat_0_2_shape . 0) . expect ("The concatenated matrices do not have the same number of columns") ;
            let _cat_row_0_offset = 0;
            let _cat_row_1_size = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: DimEq < _ , _ >> :: representative (_cat_1_1_shape . 0 , _cat_1_2_shape . 0) . expect ("The concatenated matrices do not have the same number of columns") ;
            let _cat_row_1_offset = _cat_row_0_offset + <_ as nalgebra::Dim>::value(&_cat_row_0_size);
            let _cat_row_2_size = _cat_2_0_shape.0;
            let _cat_row_2_offset = _cat_row_1_offset + <_ as nalgebra::Dim>::value(&_cat_row_1_size);
            let _cat_col_0_size = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: DimEq < _ , _ >> :: representative (_cat_0_0_shape . 1 , _cat_2_0_shape . 1) . expect ("The concatenated matrices do not have the same number of rows") ;
            let _cat_col_0_offset = 0;
            let _cat_col_1_size = _cat_1_1_shape.1;
            let _cat_col_1_offset = _cat_col_0_offset + <_ as nalgebra::Dim>::value(&_cat_col_0_size);
            let _cat_col_2_size = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: DimEq < _ , _ >> :: representative (_cat_0_2_shape . 1 , _cat_1_2_shape . 1) . expect ("The concatenated matrices do not have the same number of rows") ;
            let _cat_col_2_offset = _cat_col_1_offset + <_ as nalgebra::Dim>::value(&_cat_col_1_size);
            let mut matrix = nalgebra::Matrix::zeros_generic(
                <_ as nalgebra::DimAdd<_>>::add(
                    <_ as nalgebra::DimAdd<_>>::add(_cat_row_0_size, _cat_row_1_size),
                    _cat_row_2_size
                ),
                <_ as nalgebra::DimAdd<_>>::add(
                    <_ as nalgebra::DimAdd<_>>::add(_cat_col_0_size, _cat_col_1_size),
                    _cat_col_2_size
                )
            );
            let start = (_cat_row_0_offset, _cat_col_0_offset);
            let shape = (_cat_row_0_size, _cat_col_0_size);
            let input_view = _cat_0_0.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (_cat_row_0_offset, _cat_col_2_offset);
            let shape = (_cat_row_0_size, _cat_col_2_size);
            let input_view = _cat_0_2.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (_cat_row_1_offset, _cat_col_1_offset);
            let shape = (_cat_row_1_size, _cat_col_1_size);
            let input_view = _cat_1_1.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (_cat_row_1_offset, _cat_col_2_offset);
            let shape = (_cat_row_1_size, _cat_col_2_size);
            let input_view = _cat_1_2.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (_cat_row_2_offset, _cat_col_0_offset);
            let shape = (_cat_row_2_size, _cat_col_0_size);
            let input_view = _cat_2_0.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            matrix
        }};

        assert_eq!(format!("{}", result), format!("{}", expected));
    }
}
