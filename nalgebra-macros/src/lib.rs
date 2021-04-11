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
type MatrixSyntax = Punctuated<MatrixRowSyntax, Token![;]>;

impl Parse for Matrix {
    fn parse(input: ParseStream) -> Result<Self> {
        let span = input.span();
        // TODO: Handle empty matrix case
        let ast = MatrixSyntax::parse_separated_nonempty_with(input,
            |input| MatrixRowSyntax::parse_separated_nonempty(input))?;
        let ncols = ast.first().map(|row| row.len())
            .unwrap_or(0);

        let mut rows = Vec::new();

        for row in ast {
            if row.len() != ncols {
                // TODO: Is this the correct span?
                // Currently it returns the span corresponding to the first element in the macro
                // invocation, but it would be nice if it returned the span of the first element
                // in the first row that has an unexpected number of columns
                return Err(Error::new(span, "Unexpected number of columns. TODO"))
            }
            rows.push(row.into_iter().collect());
        }

        Ok(Self {
            rows,
            ncols
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