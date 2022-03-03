use criterion::criterion_main;
mod matrix;
mod ops;
pub use matrix::*;

#[cfg(not(all(feature = "io")))]
compile_error!(
    "Please enable the `io` features in order to compile and run the benchmark.
     Example: `cargo bench --features io`"
);

criterion_main!(ops::spmm);
