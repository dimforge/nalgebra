use approx::assert_relative_eq;
use na::{
    Complex, ComplexField, DMatrix, DVector, DefaultAllocator, Dim, DimMin, Dyn, LBLT, OMatrix,
    OVector, RealField, allocator::Allocator,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

fn relative_norm<T, N, M>(matrix1: &OMatrix<T, N, M>, matrix2: &OMatrix<T, N, M>) -> T::RealField
where
    T: ComplexField,
    N: Dim,
    M: Dim,
    DefaultAllocator: Allocator<N, M>,
{
    (matrix1 - matrix2).norm() / matrix1.norm().max(matrix2.norm())
}

fn reconstruct<T: Copy + ComplexField, N: Dim>(lblt: &LBLT<T, N>) -> OMatrix<T, N, N>
where
    T::RealField: Copy,
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    let l_permuted = lblt.l_permuted();
    &l_permuted * lblt.d() * l_permuted.adjoint()
}

/// Samples a Haar-random isometry using a QR factorization of a complex Gaussian matrix.
fn random_isometry<T, N, M>(n: N, m: M, rng: &mut impl Rng) -> OMatrix<Complex<T>, N, M>
where
    T: Copy + RealField,
    N: DimMin<M, Output = M>,
    M: Dim,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<N, M> + Allocator<M, M>,
    StandardNormal: Distribution<T>,
{
    let gaussian = OMatrix::from_fn_generic(n, m, |_, _| {
        Complex::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
    });

    let qr = gaussian.qr();
    let mut q = qr.q();
    let r = qr.r();

    for j in 0..m.value() {
        let phase = r[(j, j)] / r[(j, j)].abs();
        for el in q.column_mut(j) {
            *el *= phase;
        }
    }
    q
}

/// Sample a random Hermitian matrix with the given real diagonal spectrum.
fn random_hermitian_from_diag<T, N>(
    diag: &OVector<T, N>,
    rng: &mut impl Rng,
) -> OMatrix<Complex<T>, N, N>
where
    T: Copy + RealField,
    N: DimMin<N, Output = N>,
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
    StandardNormal: Distribution<T>,
{
    let n = diag.shape_generic().0;
    let u = random_isometry(n, n, rng);
    let d = OMatrix::from_diagonal(&diag.map(|x| Complex::new(x, T::zero())));
    &u * d * u.adjoint()
}

// This indefinite spectrum reliably exercises diverse 1x1/2x2 pivot patterns across sizes.
#[test]
fn alternating_unit_spectrum() {
    let mut rng = StdRng::seed_from_u64(0);

    for n in 2..=10 {
        let n_i32 = i32::try_from(n).unwrap();
        let diag = DVector::from_iterator(n, (0..n_i32).map(|i| (-1.0).powi(i)));

        for _ in 0..10 {
            let matrix: DMatrix<Complex<f64>> = random_hermitian_from_diag(&diag, &mut rng);
            let lblt = matrix.clone().lblt();
            assert!(relative_norm(&matrix, &reconstruct(&lblt)) < 1e-12);

            assert_relative_eq!(
                lblt.determinant(),
                (-1.0).powi(n_i32 / 2),
                max_relative = 1e-12
            );

            let b = random_isometry(Dyn(n), Dyn(n), &mut rng);
            assert!((&matrix * &lblt.solve(&b).unwrap() - &b).norm() / matrix.norm() < 1e-12);
        }
    }
}

// An alternating-sign geometric spectrum combines extreme scaling with indefiniteness.
// After Haar randomization this reliably triggers a wide variety of 1×1 and 2×2 pivots.
#[test]
fn alternating_geometric_spectrum() {
    let mut rng = StdRng::seed_from_u64(0);

    for n in 2..=10 {
        let n_i32 = i32::try_from(n).unwrap();
        let diag = DVector::from_iterator(n, (0..n_i32).map(|i| (-10.0).powi(i)));

        for _ in 0..10 {
            let matrix: DMatrix<Complex<f64>> = random_hermitian_from_diag(&diag, &mut rng);
            let lblt = matrix.clone().lblt();
            assert!(relative_norm(&matrix, &reconstruct(&lblt)) < 1e-12);

            assert_relative_eq!(
                lblt.determinant(),
                (-10.0).powi(n_i32 * (n_i32 - 1) / 2),
                max_relative = 10.0.powi(-11 + (n_i32 / 2))
            );

            let b = random_isometry(Dyn(n), Dyn(n), &mut rng);
            assert!((&matrix * &lblt.solve(&b).unwrap() - &b).norm() / matrix.norm() < 1e-12);
        }
    }
}
