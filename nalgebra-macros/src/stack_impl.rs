use crate::Matrix;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Error, Expr, Lit};

// The implementation of the stack macro. The prefix is used to construct variable names
// that are extremely unlikely to collide with variable names used in e.g. expressions
// by the user.
#[allow(clippy::too_many_lines)]
pub fn stack_impl(prefix: &str, matrix: Matrix) -> syn::Result<TokenStream2> {
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
        }).ok_or(Error::new(Span::call_site(), "At least one element in each row must be an expression of type `Matrix`"))?;

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
        }).ok_or(Error::new(Span::call_site(), "At least one element in each column must be an expression of type `Matrix`"))?;

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
        .ok_or(Error::new(
            Span::call_site(),
            "`stack` macro cannot be used without any arguments",
        ))?;

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
        .ok_or(Error::new(
            Span::call_site(),
            "`stack` macro cannot be used without any arguments",
        ))?;

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

    Ok(quote! {
        {
            #output
            matrix
        }
    })
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

#[cfg(test)]
mod tests {
    use crate::stack_impl::stack_impl;
    use crate::Matrix;
    use quote::quote;

    #[test]
    fn stack_simple_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0;
            0, b;
        ];

        let result = stack_impl("", input).unwrap();

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
    fn stack_complex_generation() {
        let input: Matrix = syn::parse_quote![
            a, 0, b;
            0, c, d;
            e, 0, 0;
        ];

        let result = stack_impl("", input).unwrap();

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
