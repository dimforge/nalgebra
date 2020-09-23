use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::ops::spmv_coo;
use nalgebra::DVector;

#[test]
fn spmv_coo_agrees_with_dense_gemv() {
    let x = DVector::from_column_slice(&[2, 3, 4, 5]);

    let i = vec![0, 0, 1, 1, 2, 2];
    let j = vec![0, 3, 0, 1, 1, 3];
    let v = vec![3, 2, 1, 2, 3, 1];
    let a = CooMatrix::try_from_triplets(3, 4, i, j, v).unwrap();

    let betas = [0, 1, 2];
    let alphas = [0, 1, 2];

    for &beta in &betas {
        for &alpha in &alphas {
            let mut y = DVector::from_column_slice(&[2, 5, 3]);
            let mut y_dense = y.clone();
            spmv_coo(beta, &mut y, alpha, &a, &x);

            y_dense.gemv(alpha, &a.to_dense(), &x, beta);

            assert_eq!(y, y_dense);
        }
    }
}