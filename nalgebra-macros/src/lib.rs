extern crate proc_macro;

use syn::{Expr};
use syn::parse::{Parse, ParseStream, Result, Error};
use syn::punctuated::{Punctuated};
use syn::{parse_macro_input, Token};
use quote::{quote, TokenStreamExt};
use proc_macro::TokenStream;

use proc_macro2::{TokenStream as TokenStream2, Delimiter, TokenTree, Spacing};
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

    /// Produces a stream of tokens representing this matrix as a column-major array.
    fn col_major_array_tokens(&self) -> TokenStream2 {
        let mut result = TokenStream2::new();
        for j in 0 .. self.ncols() {
            let mut col = TokenStream2::new();
            let col_iter = (0 .. self.nrows())
                .map(move |i| &self.rows[i][j]);
            col.append_separated(col_iter, Punct::new(',', Spacing::Alone));
            result.append(Group::new(Delimiter::Bracket, col));
            result.append(Punct::new(',', Spacing::Alone));
        }
        TokenStream2::from(TokenTree::Group(Group::new(Delimiter::Bracket, result)))
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
                        row.len());
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
            ncols: ncols.unwrap_or(0)
        })
    }
}

#[proc_macro]
pub fn matrix(stream: TokenStream) -> TokenStream {
    let matrix = parse_macro_input!(stream as Matrix);

    let row_dim = matrix.nrows();
    let col_dim = matrix.ncols();

    let array_tokens = matrix.col_major_array_tokens();

    //  TODO: Use quote_spanned instead??
    // TODO: Avoid use of unsafe here
    let output = quote! {
        unsafe {
        nalgebra::SMatrix::<_, #row_dim, #col_dim>
            ::from_data_statically_unchecked(nalgebra::ArrayStorage(#array_tokens))
        }
    };

    proc_macro::TokenStream::from(output)
}