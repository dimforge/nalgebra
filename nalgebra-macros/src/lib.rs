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
use quote::{quote, format_ident, ToTokens, TokenStreamExt};
use syn::parse::{Error, Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{Expr, Lit};
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
            let elements = MatrixRowSyntax::parse_terminated(input)?
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
    One,
    Expr(Expr),
}

impl ConcatElem {
    fn from_expr(expr: Expr) -> Self {
        match &expr {
            Expr::Lit(syn::ExprLit { lit: Lit::Int(ilit), .. }) => {
                if ilit.base10_digits() == "0" {
                    return ConcatElem::Zero
                } else if ilit.base10_digits() == "1" {
                    return ConcatElem::One
                }
            }, 
            _ => (),
        };
        ConcatElem::Expr(expr)
    }

    fn is_expr(&self) -> bool {
        match self {
            ConcatElem::Zero => false,
            ConcatElem::One => false,
            ConcatElem::Expr(_) => true
        }
    }
}

/// Constructs a new matrix by concatenating the given matrices.
///
/// **Note: Requires the `macros` feature to be enabled (enabled by default)**.
///
/// The syntax is similar to the [`matrix!`](./macro.matrix.html) and
/// [`dmatrix`](./macro.dmatrix.html) macros. However the elements should
/// be of type [`&Matrix`](./base/struct.Matrix.html) or be one of the litterals
/// `0` or `1`. The elements of type `&Matrix` are concatenated as expected. The
/// litteral `0` is expanded to the zero. The litteral `1` is expanded to the 
/// identity matrix. Note that at least one element in each row and column must
/// be an expression of type `Matrix`.
///
/// All elements in the same row needs to have the same number of rows and simillary
/// for the elements in the same column. This is checked at compile time as long as
/// all elements have dimensions of type `Const<N>`. If one or more elements have
/// dimensions of type `Dynamic` then it is checked at runtime that all dimensions
/// match.
///
/// If at least one element of a row has `Const<N>` number of rows then that row has 
/// a whole will have `Const<N>` number of rows. However if at least one row has
/// has `Dynamic` number of rows then the entire matrix will have `Dynamic` number of
/// rows. Same for columns.
///
/// # Examples
///
/// ```
/// use nalgebra::{cat, matrix};
///
/// let a = cat![
///     &matrix![1,2;3,4;], 0;
///     0, &matrix![5,6;7,8;]
/// ];
///
/// let b = matrix![
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
/// use nalgebra::{cat, matrix, Matrix5x6};
///
/// let a: Matrix5x6<_> = cat![
///     1, &matrix![1;2], 0;
///     0, 1, &matrix![3,4;5,6;];
///     &matrix![7,8,9;], 0, 0;
/// ];
///
/// let b = matrix![
///     1, 0, 0, 1, 0, 0;
///     0, 1, 0, 2, 0, 0;
///     0, 0, 0, 1, 3, 4;
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
///     &dmatrix![1,2;3,4;], 0;
///     0, &matrix![5,6;7,8;];
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
    let n_macro_rows = matrix.nrows();
    let n_macro_cols = matrix.ncols();

    let rows = matrix.rows.into_iter().map(|row| {
        row.into_iter().map(|c| ConcatElem::from_expr(c)).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    let mut output = quote!{};

    for (i, row) in rows.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if let ConcatElem::Expr(expr) = cell {
                let ident = format_ident!("cat_{}_{}", i, j);
                let ident_shape = format_ident!("cat_{}_{}_shape", i, j);
                output.extend(std::iter::once(quote!{
                    let #ident = #expr;
                    let #ident_shape = #ident.shape_generic();
                }));
            }
        }
    }

    for i in 0..n_macro_rows {
        let size = (0..n_macro_cols).filter(|j| rows[i][*j].is_expr()).map(|j| {
            let ident_shape = format_ident!("cat_{}_{}_shape", i, j);
            quote!{ #ident_shape.0 }
        }).reduce(|a, b| quote!{
            <_ as nalgebra::DimUnify<_>>::unify(#a, #b)
                .expect("The concatenated matrices do not have the same number of columns")
        }).expect("At least one element in each row must be an expression of type `Matrix`");

        let size_ident = format_ident!("cat_row_{}_size", i);
        let offset_ident = format_ident!("cat_row_{}_offset", i);

        let offset = if i == 0 {
            quote!{ 0 }
        } else {
            let last_offset_ident = format_ident!("cat_row_{}_offset", i - 1);
            let last_size_ident = format_ident!("cat_row_{}_size", i - 1);
            quote!{ #last_offset_ident + <_ as nalgebra::Dim>::value(&#last_size_ident) }
        };

        output.extend(std::iter::once(quote!{
            let #size_ident = #size;
            let #offset_ident = #offset;
        }));
    }

    for j in 0..n_macro_cols {
        let size = (0..n_macro_rows).filter(|i| rows[*i][j].is_expr()).map(|i| {
            let ident_shape = format_ident!("cat_{}_{}_shape", i, j);
            quote!{ #ident_shape.1 }
        }).reduce(|a, b| quote!{
            <_ as nalgebra::DimUnify<_>>::unify(#a, #b)
                .expect("The concatenated matrices do not have the same number of rows")
        }).expect("At least one element in each column must be an expression of type `Matrix`");

        let size_ident = format_ident!("cat_col_{}_size", j);
        let offset_ident = format_ident!("cat_col_{}_offset", j);

        let offset = if j == 0 {
            quote!{ 0 }
        } else {
            let last_offset_ident = format_ident!("cat_col_{}_offset", j - 1);
            let last_size_ident = format_ident!("cat_col_{}_size", j - 1);
            quote!{ #last_offset_ident + <_ as nalgebra::Dim>::value(&#last_size_ident) }
        };

        output.extend(std::iter::once(quote!{
            let #size_ident = #size;
            let #offset_ident = #offset;
        }));
    }

    let nrows = (0..n_macro_rows).map(|i| {
        let ident = format_ident!("cat_row_{}_size", i);
        quote!{ #ident }
    }).reduce(|a, b| quote!{
        <_ as nalgebra::DimAdd<_>>::add(#a, #b)
    }).expect("More than zero rows in concatenation");

    let ncols = (0..n_macro_cols).map(|j| {
        let ident = format_ident!("cat_col_{}_size", j);
        quote!{ #ident }
    }).reduce(|a, b| quote!{
        <_ as nalgebra::DimAdd<_>>::add(#a, #b)
    }).unwrap();

    // It should be possible to use `uninitialized_generic` here instead
    // however that would mean that the macro needs to generate unsafe code
    // which does not seem like a great idea.
    output.extend(std::iter::once(quote!{
        let mut matrix = nalgebra::Matrix::zeros_generic(#nrows, #ncols);
    }));

    for (i, row) in rows.into_iter().enumerate() {
        for (j, cell) in row.into_iter().enumerate() {
            let row_size = format_ident!("cat_row_{}_size", i);
            let col_size = format_ident!("cat_col_{}_size", j);
            let row_offset = format_ident!("cat_row_{}_offset", i);
            let col_offset = format_ident!("cat_col_{}_offset", j);
            match cell {
                ConcatElem::Zero => (),
                ConcatElem::One => {
                    // FIXME: should be possible to use Matrix::fill_diagonal here,
                    // but how to access `One::one()` hygienically?
                    output.extend(std::iter::once(quote!{
                        let start = (#row_offset, #col_offset);
                        let shape = (#row_size, #col_size);
                        let mut slice = matrix.generic_slice_mut(start, shape);
                        slice.copy_from(&nalgebra::Matrix::identity_generic(shape.0, shape.1));
                    }));
                },
                ConcatElem::Expr(_) => {
                    let expr_ident = format_ident!("cat_{}_{}", i, j);
                    output.extend(std::iter::once(quote!{
                        let start = (#row_offset, #col_offset);
                        let shape = (#row_size, #col_size);
                        let mut slice = matrix.generic_slice_mut(start, shape);
                        slice.copy_from(#expr_ident);
                    }));
                },
            }
        }
    }

    proc_macro::TokenStream::from(quote!{
        {
            #output
            matrix
        }
    })
}
