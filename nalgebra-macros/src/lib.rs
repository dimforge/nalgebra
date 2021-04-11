extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Literal, Ident, Punct, Spacing, Group, Delimiter};
use std::iter::FromIterator;

struct MatrixEntries {
    entries: Vec<Vec<TokenTree>>
}

impl MatrixEntries {
    fn new() -> Self {
        Self {
            entries: Vec::new()
        }
    }

    fn begin_new_row(&mut self) {
        self.entries.push(Vec::new());
    }

    fn push_entry(&mut self, entry: TokenTree) {
        if self.entries.is_empty() {
            self.entries.push(Vec::new());
        }
        let mut last_row = self.entries.last_mut().unwrap();
        last_row.push(entry);
    }

    fn build_stream(&self) -> TokenStream {
        let num_rows = self.entries.len();
        let num_cols = self.entries.first()
            .map(|first_row| first_row.len())
            .unwrap_or(0);

        // First check that dimensions are consistent
        for (i, row) in self.entries.iter().enumerate() {
            if row.len() != num_cols {
                panic!("Unexpected number of columns in row {}: {}. Expected {}", i, row.len(), num_cols);
            }
        }

        let mut array_tokens = Vec::new();

        // Collect entries in column major order
        for i in 0 .. num_rows {
            for j in 0 .. num_cols {
                let entry = &self.entries[i][j];
                array_tokens.push(entry.clone());
                array_tokens.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
            }
        }

        let row_dim = format!("U{}", num_rows);
        let col_dim = format!("U{}", num_cols);
        // let imports = format!("use nalgebra::\{Matrix, {}, {}\};", row_dim, col_dim);
        // let constructor = format!("Matrix::<_, {}, {}>::from_slice", row_dim, col_dim);
        // let array_group = Group::new(Delimiter::Bracket, TokenStream::from_iter(array_tokens.into_iter()));

        let array_stream = TokenStream::from_iter(array_tokens);

        // TODO: Build this up without parsing?
        format!(r"{{
            nalgebra::MatrixMN::<_, nalgebra::{row_dim}, nalgebra::{col_dim}>::from_column_slice(&[
                {array_tokens}
            ])
        }}", row_dim=row_dim, col_dim=col_dim, array_tokens=array_stream.to_string()).parse().unwrap()


        // let mut outer_group = Group::new(Delimiter::Brace,
        //
        // );


        // TODO: Outer group




        // todo!()
    }
}

#[proc_macro]
pub fn matrix(stream: TokenStream) -> TokenStream {
    let mut entries = MatrixEntries::new();
    for tree in stream {
        match tree {
            // TokenTree::Ident(ident) => entries.push_entry(tree),
            // TokenTree::Literal(literal) => entries.push_entry(tree),
            TokenTree::Punct(punct) if punct == ';' => entries.begin_new_row(),
            TokenTree::Punct(punct) if punct == ',' => {},
            // TokenTree::Punct(punct) => panic!("Unexpected punctuation: '{}'", punct),
            // TokenTree::Group(_) => panic!("Unexpected token group"),
            _ => entries.push_entry(tree)
        }
    }

    entries.build_stream()
}