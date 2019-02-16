use std::fs;
use std::path::Path;

use pest::Parser;
use sparse::CsMatrix;
use Real;

#[derive(Parser)]
#[grammar = "io/matrix_market.pest"]
struct MatrixMarketParser;

// FIXME: return an Error instead of an Option.
/// Parses a Matrix Market file at the given path, and returns the corresponding sparse matrix.
pub fn cs_matrix_from_matrix_market<N: Real, P: AsRef<Path>>(path: P) -> Option<CsMatrix<N>> {
    let file = fs::read_to_string(path).ok()?;
    cs_matrix_from_matrix_market_str(&file)
}

// FIXME: return an Error instead of an Option.
/// Parses a Matrix Market file described by the given string, and returns the corresponding sparse matrix.
pub fn cs_matrix_from_matrix_market_str<N: Real>(data: &str) -> Option<CsMatrix<N>> {
    let file = MatrixMarketParser::parse(Rule::Document, data)
        .unwrap()
        .next()?;
    let mut shape = (0, 0, 0);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut data: Vec<N> = Vec::new();

    for line in file.into_inner() {
        match line.as_rule() {
            Rule::Header => {}
            Rule::Shape => {
                let mut inner = line.into_inner();
                shape.0 = inner.next()?.as_str().parse::<usize>().ok()?;
                shape.1 = inner.next()?.as_str().parse::<usize>().ok()?;
                shape.2 = inner.next()?.as_str().parse::<usize>().ok()?;
            }
            Rule::Entry => {
                let mut inner = line.into_inner();
                // NOTE: indices are 1-based.
                rows.push(inner.next()?.as_str().parse::<usize>().ok()? - 1);
                cols.push(inner.next()?.as_str().parse::<usize>().ok()? - 1);
                data.push(::convert(inner.next()?.as_str().parse::<f64>().ok()?));
            }
            _ => return None, // FIXME: return an Err instead.
        }
    }

    Some(CsMatrix::from_triplet(
        shape.0, shape.1, &rows, &cols, &data,
    ))
}
