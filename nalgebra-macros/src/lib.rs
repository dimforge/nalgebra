extern crate proc_macro;

use syn::{Expr};
use syn::parse::{Parse, ParseStream, Result, Error};
use syn::punctuated::{Punctuated};
use syn::{parse_macro_input, Token};
use quote::{quote, format_ident};
use proc_macro::TokenStream;

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

    fn to_col_major_repr(&self) -> Vec<Expr> {
        let mut data = Vec::with_capacity(self.nrows() * self.ncols());
        for j in 0 .. self.ncols() {
            for i in 0 .. self.nrows() {
                data.push(self.rows[i][j].clone());
            }
        }
        data
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

    let dim_ident = |dim| format_ident!("U{}", dim);
    let row_dim = dim_ident(matrix.nrows());
    let col_dim = dim_ident(matrix.ncols());
    let entries_col_major = matrix.to_col_major_repr();

    //  TODO: Use quote_spanned instead??
    // TODO: Construct directly from array?
    let output = quote! {
        nalgebra::MatrixMN::<_, nalgebra::dimension::#row_dim, nalgebra::dimension::#col_dim>
            ::from_column_slice(&[#(#entries_col_major),*])
    };

    proc_macro::TokenStream::from(output)
}