use proc_macro::TokenStream;
use quote::{ToTokens, TokenStreamExt, quote};
use std::ops::Index;
use syn::Expr;
use syn::parse::{Error, Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Token, parse_macro_input};

use proc_macro2::{Delimiter, Spacing, TokenStream as TokenStream2, TokenTree};
use proc_macro2::{Group, Punct};

/// A matrix of expressions
pub struct Matrix {
    // Represent the matrix data in row-major format
    data: Vec<Expr>,
    nrows: usize,
    ncols: usize,
}

impl Index<(usize, usize)> for Matrix {
    type Output = Expr;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        let linear_idx = self.ncols * row + col;
        &self.data[linear_idx]
    }
}

impl Matrix {
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Produces a stream of tokens representing this matrix as a column-major nested array.
    pub fn to_col_major_nested_array_tokens(&self) -> TokenStream2 {
        let mut result = TokenStream2::new();
        for j in 0..self.ncols() {
            let mut col = TokenStream2::new();
            let col_iter = (0..self.nrows()).map(|i| &self[(i, j)]);
            col.append_separated(col_iter, Punct::new(',', Spacing::Alone));
            result.append(Group::new(Delimiter::Bracket, col));
            result.append(Punct::new(',', Spacing::Alone));
        }
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, result)))
    }

    /// Produces a stream of tokens representing this matrix as a column-major flat array
    /// (suitable for representing e.g. a `DMatrix`).
    pub fn to_col_major_flat_array_tokens(&self) -> TokenStream2 {
        let mut data = TokenStream2::new();
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self[(i, j)].to_tokens(&mut data);
                data.append(Punct::new(',', Spacing::Alone));
            }
        }
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, data)))
    }
}

type MatrixRowSyntax = Punctuated<Expr, Token![,]>;

impl Parse for Matrix {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut data = Vec::new();
        let mut ncols = None;
        let mut nrows = 0;

        while !input.is_empty() {
            let row = MatrixRowSyntax::parse_separated_nonempty(input)?;
            let row_span = row.span();

            if let Some(ncols) = ncols {
                if row.len() != ncols {
                    let error_msg = format!(
                        "Unexpected number of entries in row {}. Expected {}, found {} entries.",
                        nrows,
                        ncols,
                        row.len()
                    );
                    return Err(Error::new(row_span, error_msg));
                }
            } else {
                ncols = Some(row.len());
            }
            data.extend(row.into_iter());
            nrows += 1;

            // We've just read a row, so if there are more tokens, there must be a semi-colon,
            // otherwise the input is malformed
            if !input.is_empty() {
                input.parse::<Token![;]>()?;
            }
        }

        Ok(Self {
            data,
            nrows,
            ncols: ncols.unwrap_or(0),
        })
    }
}

pub struct Vector {
    elements: Vec<Expr>,
}

impl Vector {
    pub fn to_array_tokens(&self) -> TokenStream2 {
        let mut data = TokenStream2::new();
        data.append_separated(&self.elements, Punct::new(',', Spacing::Alone));
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, data)))
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

impl Parse for Vector {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
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

pub fn matrix_impl(stream: TokenStream) -> TokenStream {
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

pub fn dmatrix_impl(stream: TokenStream) -> TokenStream {
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

pub fn vector_impl(stream: TokenStream) -> TokenStream {
    let vector = parse_macro_input!(stream as Vector);
    let len = vector.len();
    let array_tokens = vector.to_array_tokens();
    let output = quote! {
        nalgebra::SVector::<_, #len>
            ::from_array_storage(nalgebra::ArrayStorage([#array_tokens]))
    };
    proc_macro::TokenStream::from(output)
}

pub fn dvector_impl(stream: TokenStream) -> TokenStream {
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
