extern crate nalgebra_sparse;
use nalgebra_sparse::CsrMatrix;
use std::fs::{self, DirEntry};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(feature = "io")]
use nalgebra_sparse::io::load_coo_from_matrix_market_file;
fn main() {
    #[cfg(feature = "io")]
    {
        let mut file_iter = fs::read_dir("./data").unwrap();
        for f in file_iter {
            println!("Benchmark file {:?}", f);
            let f = f.unwrap().path();
            let sparse_input_matrix = load_coo_from_matrix_market_file::<f64, _>(&f).unwrap();
            let sparse_input_matrix = CsrMatrix::from(&sparse_input_matrix);
            let spmm_result = &sparse_input_matrix * &sparse_input_matrix;
            let now = Instant::now();
            let spmm_result = &sparse_input_matrix * &sparse_input_matrix;
            let spmm_time = now.elapsed().as_millis();
            println!("SGEMM time was {}", spmm_time);
            let sum: f64 = spmm_result.triplet_iter().map(|(_, _, v)| v).sum();
            println!("sum of product is {}", sum);
        }
    }
    #[cfg(not(feature = "io"))]
    {
        panic!("Run with IO feature only");
    }
}
