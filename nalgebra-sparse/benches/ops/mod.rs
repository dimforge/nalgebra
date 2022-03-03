use crate::{INT_MATRICES, REAL_MATRICES};
use criterion::{criterion_group, Criterion};
use nalgebra::DMatrix;
use nalgebra_sparse::io::load_coo_from_matrix_market_file;
use nalgebra_sparse::ops::serial::{spmm_csr_dense, spmm_csr_pattern, spmm_csr_prealloc};
use nalgebra_sparse::{self, csr::CsrMatrix, ops::Op};
use num_traits::identities::Zero;
use paste::paste;
use rand::Rng;

macro_rules! spmm_dense_bench {
($matrix_collection:ident,$matrix_type:ty, $matrix_type_name:tt,$scalar_type:ty, $dimension: tt) => {
    paste! {
        fn [<spmm_dense_ $matrix_collection:lower _ $matrix_type_name _ $scalar_type _ $dimension >](cr: &mut Criterion) {

            let group_name = stringify!([<spmm_dense_ $matrix_collection:lower _ $scalar_type _ $dimension>]);
            let mut group = cr.benchmark_group(group_name);

            for matrix_name in $matrix_collection{

                let sparse_matrix_file = format!("data/{}.mtx", matrix_name);
                let sparse_matrix = load_coo_from_matrix_market_file::<$scalar_type, _>(sparse_matrix_file).expect(
                        &format!("Can't load the matrix data/{}.mtx.",matrix_name )
                    );
                let matrix_a = $matrix_type::from(&sparse_matrix);

                let mut rng = rand::thread_rng();
                let mut matrix_c = DMatrix::<$scalar_type>::new_random(matrix_a.nrows(), $dimension);
                let matrix_b = DMatrix::<$scalar_type>::new_random(matrix_a.ncols(), $dimension);

                let alpha:$scalar_type = rng.gen();
                let beta:$scalar_type = rng.gen();

                let bench_name = format!("{}_{}",group_name,matrix_name);

                group.bench_function(&bench_name, move |ben| {
                    ben.iter(|| {
                        [<spmm_ $matrix_type_name _dense>](
                            alpha,
                            &mut matrix_c,
                            beta,
                            Op::NoOp(&matrix_a),
                            Op::NoOp(&matrix_b));
                        })});
            }
            group.finish();

    }
}
};
}

macro_rules! spmm_prealloc_bench {
($matrix_collection:ident,$matrix_type:ty, $matrix_type_name:tt,$scalar_type:ty) => {
    paste! {
        fn [<spmm_prealloc_ $matrix_collection:lower _ $matrix_type_name _ $scalar_type >](cr: &mut Criterion) {

            let group_name = stringify!([<spmm_prealloc_ $matrix_collection:lower _ $scalar_type>]);
            let mut group = cr.benchmark_group(group_name);

            for matrix_name in $matrix_collection{

                let sparse_matrix_file = format!("data/{}.mtx", matrix_name);
                let sparse_matrix = load_coo_from_matrix_market_file::<$scalar_type, _>(sparse_matrix_file).expect(
                        &format!("Can't load the matrix data/{}.mtx.",matrix_name )
                    );

                let matrix_a = $matrix_type::from(&sparse_matrix);

                let mut rng = rand::thread_rng();
                let pattern = [<spmm_ $matrix_type_name _pattern>](matrix_a.pattern(), matrix_a.transpose().pattern());
                let values = vec![$scalar_type::zero(); pattern.nnz()];
                let mut matrix_c = $matrix_type::<$scalar_type>::try_from_pattern_and_values(pattern, values).unwrap();

                let matrix_b = matrix_a.transpose();
                let alpha:$scalar_type = rng.gen();
                let beta:$scalar_type = rng.gen();

                let bench_name = format!("{}_{}",group_name,matrix_name);

                group.bench_function(&bench_name, move |ben| {
                    ben.iter(|| {
                        [<spmm_ $matrix_type_name _prealloc>](
                            alpha,
                            &mut matrix_c,
                            beta,
                            Op::NoOp(&matrix_a),
                            Op::NoOp(&matrix_b)).expect("unexpected benchmark error");
                        })});
            }
            group.finish();

    }
}
};
}

spmm_dense_bench!(REAL_MATRICES, CsrMatrix, csr, f64, 1);
spmm_dense_bench!(INT_MATRICES, CsrMatrix, csr, i128, 1);
spmm_prealloc_bench!(INT_MATRICES, CsrMatrix, csr, i32);
spmm_prealloc_bench!(REAL_MATRICES, CsrMatrix, csr, f64);
criterion_group!(
    spmm,
    spmm_dense_real_matrices_csr_f64_1,
    spmm_dense_int_matrices_csr_i128_1,
    spmm_prealloc_real_matrices_csr_f64,
    spmm_prealloc_int_matrices_csr_i32,
);
