use crate::Matrix;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Error, Expr, Lit};

#[allow(clippy::too_many_lines)]
pub fn stack_impl(matrix: Matrix) -> syn::Result<TokenStream2> {
    // The prefix is used to construct variable names
    // that are extremely unlikely to collide with variable names used in e.g. expressions
    // by the user. Although we could use a long, pseudo-random string, this makes the generated
    // code very painful to parse, so we settle for something more semantic that is still
    // very unlikely to collide
    let prefix = "___na";
    let n_block_rows = matrix.nrows();
    let n_block_cols = matrix.ncols();

    let mut output = quote! {};

    // First assign data and shape for each matrix entry to variables
    // (this is important so that we, for example, don't evaluate an expression more than once)
    for i in 0..n_block_rows {
        for j in 0..n_block_cols {
            let expr = &matrix[(i, j)];
            if !is_literal_zero(expr) {
                let ident_block = format_ident!("{prefix}_stack_{i}_{j}_block");
                let ident_shape = format_ident!("{prefix}_stack_{i}_{j}_shape");
                output.extend(std::iter::once(quote_spanned! {expr.span()=>
                    let ref #ident_block = #expr;
                    let #ident_shape = #ident_block.shape_generic();
                }));
            }
        }
    }

    // Determine the number of rows (dimension) in each block row,
    // and write out variables that define block row dimensions and offsets into the
    // output matrix
    for i in 0..n_block_rows {
        // The dimension of the block row is the result of trying to unify the row shape of
        // all blocks in the block row
        let dim = (0 ..n_block_cols)
            .filter_map(|j| {
                let expr = &matrix[(i, j)];
                if !is_literal_zero(expr) {
                    let mut ident_shape = format_ident!("{prefix}_stack_{i}_{j}_shape");
                    ident_shape.set_span(ident_shape.span().located_at(expr.span()));
                    Some(quote_spanned!{expr.span()=> #ident_shape.0 })
                } else {
                    None
                }
            }).reduce(|a, b| {
                let expect_msg = format!("All blocks in block row {i} must have the same number of rows");
                quote_spanned!{b.span()=>
                    <nalgebra::constraint::ShapeConstraint as nalgebra::constraint::SameNumberOfRows<_, _>>::representative(#a, #b)
                        .expect(#expect_msg)
                }
            }).ok_or(Error::new(Span::call_site(), format!("Block row {i} cannot consist entirely of implicit zero blocks.")))?;

        let dim_ident = format_ident!("{prefix}_stack_row_{i}_dim");
        let offset_ident = format_ident!("{prefix}_stack_row_{i}_offset");

        let offset = if i == 0 {
            quote! { 0 }
        } else {
            let prev_offset_ident = format_ident!("{prefix}_stack_row_{}_offset", i - 1);
            let prev_dim_ident = format_ident!("{prefix}_stack_row_{}_dim", i - 1);
            quote! { #prev_offset_ident + <_ as nalgebra::Dim>::value(&#prev_dim_ident) }
        };

        output.extend(std::iter::once(quote! {
            let #dim_ident = #dim;
            let #offset_ident = #offset;
        }));
    }

    // Do the same thing for the block columns
    for j in 0..n_block_cols {
        let dim = (0 ..n_block_rows)
            .filter_map(|i| {
                let expr = &matrix[(i, j)];
                if !is_literal_zero(expr) {
                    let mut ident_shape = format_ident!("{prefix}_stack_{i}_{j}_shape");
                    ident_shape.set_span(ident_shape.span().located_at(expr.span()));
                    Some(quote_spanned!{expr.span()=> #ident_shape.1 })
                } else {
                    None
                }
            }).reduce(|a, b| {
                let expect_msg = format!("All blocks in block column {j} must have the same number of columns");
                quote_spanned!{b.span()=>
                        <nalgebra::constraint::ShapeConstraint as nalgebra::constraint::SameNumberOfColumns<_, _>>::representative(#a, #b)
                            .expect(#expect_msg)
                }
            }).ok_or(Error::new(Span::call_site(), format!("Block column {j} cannot consist entirely of implicit zero blocks.")))?;

        let dim_ident = format_ident!("{prefix}_stack_col_{j}_dim");
        let offset_ident = format_ident!("{prefix}_stack_col_{j}_offset");

        let offset = if j == 0 {
            quote! { 0 }
        } else {
            let prev_offset_ident = format_ident!("{prefix}_stack_col_{}_offset", j - 1);
            let prev_dim_ident = format_ident!("{prefix}_stack_col_{}_dim", j - 1);
            quote! { #prev_offset_ident + <_ as nalgebra::Dim>::value(&#prev_dim_ident) }
        };

        output.extend(std::iter::once(quote! {
            let #dim_ident = #dim;
            let #offset_ident = #offset;
        }));
    }

    // Determine number of rows and cols in output matrix,
    // by adding together dimensions of all block rows/cols
    let num_rows = (0..n_block_rows)
        .map(|i| {
            let ident = format_ident!("{prefix}_stack_row_{i}_dim");
            quote! { #ident }
        })
        .reduce(|a, b| {
            quote! {
                <_ as nalgebra::DimAdd<_>>::add(#a, #b)
            }
        })
        .unwrap_or(quote! { nalgebra::dimension::U0 });

    let num_cols = (0..n_block_cols)
        .map(|j| {
            let ident = format_ident!("{prefix}_stack_col_{j}_dim");
            quote! { #ident }
        })
        .reduce(|a, b| {
            quote! {
                <_ as nalgebra::DimAdd<_>>::add(#a, #b)
            }
        })
        .unwrap_or(quote! { nalgebra::dimension::U0 });

    // It should be possible to use `uninitialized_generic` here instead
    // however that would mean that the macro needs to generate unsafe code
    // which does not seem like a great idea.
    output.extend(std::iter::once(quote! {
        let mut matrix = nalgebra::Matrix::zeros_generic(#num_rows, #num_cols);
    }));

    for i in 0..n_block_rows {
        for j in 0..n_block_cols {
            let row_dim = format_ident!("{prefix}_stack_row_{i}_dim");
            let col_dim = format_ident!("{prefix}_stack_col_{j}_dim");
            let row_offset = format_ident!("{prefix}_stack_row_{i}_offset");
            let col_offset = format_ident!("{prefix}_stack_col_{j}_offset");
            let expr = &matrix[(i, j)];
            if !is_literal_zero(expr) {
                let expr_ident = format_ident!("{prefix}_stack_{i}_{j}_block");
                output.extend(std::iter::once(quote! {
                    let start = (#row_offset, #col_offset);
                    let shape = (#row_dim, #col_dim);
                    let input_view = #expr_ident.generic_view((0, 0), shape);
                    let mut output_view = matrix.generic_view_mut(start, shape);
                    output_view.copy_from(&input_view);
                }));
            }
        }
    }

    Ok(quote! {
        {
            #output
            matrix
        }
    })
}

fn is_literal_zero(expr: &Expr) -> bool {
    matches!(expr,
        Expr::Lit(syn::ExprLit { lit: Lit::Int(integer_literal), .. })
        if integer_literal.base10_digits() == "0")
}

#[cfg(test)]
mod tests {
    use crate::Matrix;
    use crate::stack_impl::stack_impl;
    use quote::quote;

    #[test]
    fn stack_simple_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0;
            0, b;
        ];

        let result = stack_impl(input).unwrap();

        let expected = quote! {{
            let ref ___na_stack_0_0_block = a;
            let ___na_stack_0_0_shape = ___na_stack_0_0_block.shape_generic();
            let ref ___na_stack_1_1_block = b;
            let ___na_stack_1_1_shape = ___na_stack_1_1_block.shape_generic();
            let ___na_stack_row_0_dim = ___na_stack_0_0_shape.0;
            let ___na_stack_row_0_offset = 0;
            let ___na_stack_row_1_dim = ___na_stack_1_1_shape.0;
            let ___na_stack_row_1_offset = ___na_stack_row_0_offset + <_ as nalgebra::Dim>::value(&___na_stack_row_0_dim);
            let ___na_stack_col_0_dim = ___na_stack_0_0_shape.1;
            let ___na_stack_col_0_offset = 0;
            let ___na_stack_col_1_dim = ___na_stack_1_1_shape.1;
            let ___na_stack_col_1_offset = ___na_stack_col_0_offset + <_ as nalgebra::Dim>::value(&___na_stack_col_0_dim);
            let mut matrix = nalgebra::Matrix::zeros_generic(
                <_ as nalgebra::DimAdd<_>>::add(___na_stack_row_0_dim, ___na_stack_row_1_dim),
                <_ as nalgebra::DimAdd<_>>::add(___na_stack_col_0_dim, ___na_stack_col_1_dim)
            );
            let start = (___na_stack_row_0_offset, ___na_stack_col_0_offset);
            let shape = (___na_stack_row_0_dim, ___na_stack_col_0_dim);
            let input_view = ___na_stack_0_0_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (___na_stack_row_1_offset, ___na_stack_col_1_offset);
            let shape = (___na_stack_row_1_dim, ___na_stack_col_1_dim);
            let input_view = ___na_stack_1_1_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            matrix
        }};

        assert_eq!(format!("{result}"), format!("{}", expected));
    }

    #[test]
    fn stack_complex_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0, b;
            0, c, d;
            e, 0, 0;
        ];

        let result = stack_impl(input).unwrap();

        let expected = quote! {{
            let ref ___na_stack_0_0_block = a;
            let ___na_stack_0_0_shape = ___na_stack_0_0_block.shape_generic();
            let ref ___na_stack_0_2_block = b;
            let ___na_stack_0_2_shape = ___na_stack_0_2_block.shape_generic();
            let ref ___na_stack_1_1_block = c;
            let ___na_stack_1_1_shape = ___na_stack_1_1_block.shape_generic();
            let ref ___na_stack_1_2_block = d;
            let ___na_stack_1_2_shape = ___na_stack_1_2_block.shape_generic();
            let ref ___na_stack_2_0_block = e;
            let ___na_stack_2_0_shape = ___na_stack_2_0_block.shape_generic();
            let ___na_stack_row_0_dim = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: SameNumberOfRows < _ , _ >> :: representative (___na_stack_0_0_shape . 0 , ___na_stack_0_2_shape . 0) . expect ("All blocks in block row 0 must have the same number of rows") ;
            let ___na_stack_row_0_offset = 0;
            let ___na_stack_row_1_dim = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: SameNumberOfRows < _ , _ >> :: representative (___na_stack_1_1_shape . 0 , ___na_stack_1_2_shape . 0) . expect ("All blocks in block row 1 must have the same number of rows") ;
            let ___na_stack_row_1_offset = ___na_stack_row_0_offset + <_ as nalgebra::Dim>::value(&___na_stack_row_0_dim);
            let ___na_stack_row_2_dim = ___na_stack_2_0_shape.0;
            let ___na_stack_row_2_offset = ___na_stack_row_1_offset + <_ as nalgebra::Dim>::value(&___na_stack_row_1_dim);
            let ___na_stack_col_0_dim = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: SameNumberOfColumns < _ , _ >> :: representative (___na_stack_0_0_shape . 1 , ___na_stack_2_0_shape . 1) . expect ("All blocks in block column 0 must have the same number of columns") ;
            let ___na_stack_col_0_offset = 0;
            let ___na_stack_col_1_dim = ___na_stack_1_1_shape.1;
            let ___na_stack_col_1_offset = ___na_stack_col_0_offset + <_ as nalgebra::Dim>::value(&___na_stack_col_0_dim);
            let ___na_stack_col_2_dim = < nalgebra :: constraint :: ShapeConstraint as nalgebra :: constraint :: SameNumberOfColumns < _ , _ >> :: representative (___na_stack_0_2_shape . 1 , ___na_stack_1_2_shape . 1) . expect ("All blocks in block column 2 must have the same number of columns") ;
            let ___na_stack_col_2_offset = ___na_stack_col_1_offset + <_ as nalgebra::Dim>::value(&___na_stack_col_1_dim);
            let mut matrix = nalgebra::Matrix::zeros_generic(
                <_ as nalgebra::DimAdd<_>>::add(
                    <_ as nalgebra::DimAdd<_>>::add(___na_stack_row_0_dim, ___na_stack_row_1_dim),
                    ___na_stack_row_2_dim
                ),
                <_ as nalgebra::DimAdd<_>>::add(
                    <_ as nalgebra::DimAdd<_>>::add(___na_stack_col_0_dim, ___na_stack_col_1_dim),
                    ___na_stack_col_2_dim
                )
            );
            let start = (___na_stack_row_0_offset, ___na_stack_col_0_offset);
            let shape = (___na_stack_row_0_dim, ___na_stack_col_0_dim);
            let input_view = ___na_stack_0_0_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (___na_stack_row_0_offset, ___na_stack_col_2_offset);
            let shape = (___na_stack_row_0_dim, ___na_stack_col_2_dim);
            let input_view = ___na_stack_0_2_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (___na_stack_row_1_offset, ___na_stack_col_1_offset);
            let shape = (___na_stack_row_1_dim, ___na_stack_col_1_dim);
            let input_view = ___na_stack_1_1_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (___na_stack_row_1_offset, ___na_stack_col_2_offset);
            let shape = (___na_stack_row_1_dim, ___na_stack_col_2_dim);
            let input_view = ___na_stack_1_2_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            let start = (___na_stack_row_2_offset, ___na_stack_col_0_offset);
            let shape = (___na_stack_row_2_dim, ___na_stack_col_0_dim);
            let input_view = ___na_stack_2_0_block.generic_view((0,0), shape);
            let mut output_view = matrix.generic_view_mut(start, shape);
            output_view.copy_from(&input_view);
            matrix
        }};

        assert_eq!(format!("{result}"), format!("{}", expected));
    }
}
