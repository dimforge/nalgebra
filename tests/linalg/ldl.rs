use approx::assert_relative_eq;
use na::{
    Complex, ComplexField, DMatrix, DVector, DefaultAllocator, Dim, DimMin, Dyn, LDL, OMatrix,
    OVector, RealField, allocator::Allocator,
};
use num_traits::{One, Zero};
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

fn reconstruct<T: Copy + ComplexField, N: Dim>(ldl: &LDL<T, N>) -> OMatrix<T, N, N>
where
    T::RealField: Copy,
    DefaultAllocator: Allocator<N, N>,
{
    let l_permuted = ldl.l_permuted();
    &l_permuted * ldl.d() * l_permuted.adjoint()
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

#[test]
fn zero_matrix() {
    for n in 1..=5 {
        let ldl = DMatrix::<f64>::zeros(n, n).ldl();

        assert_eq!(ldl.l_permuted(), DMatrix::identity(n, n));
        assert_eq!(ldl.d(), DMatrix::zeros(n, n));
        assert_eq!(ldl.zero_pivot(), Some(1));
        assert_eq!(ldl.pivots(), &(1..=n.cast_signed()).collect::<Vec<_>>());
        assert!(ldl.determinant().is_zero());

        assert!(ldl.solve(&DVector::from_element(n, 1.0)).is_none());
    }
}

#[test]
fn identity_matrix() {
    for n in 1..=5 {
        let identity = DMatrix::<f64>::identity(n, n);
        let ldl = identity.clone().ldl();

        assert_eq!(ldl.l_permuted(), identity);
        assert_eq!(ldl.d(), identity);
        assert_eq!(ldl.zero_pivot(), None);
        assert_eq!(ldl.pivots(), &(1..=n.cast_signed()).collect::<Vec<_>>());
        assert!(ldl.determinant().is_one());
    }
}

#[test]
fn exchange_matrix() {
    for n in 1..=15 {
        let exchange = DMatrix::from_fn(n, n, |i, j| if i + j + 1 == n { 1.0 } else { 0.0 });
        let ldl = exchange.clone().ldl();

        let mut expected = Vec::with_capacity(n);
        let m = (n + 2) / 4;
        for r in 0..m {
            let pivot = -(n - 2 * r).cast_signed();
            expected.push(pivot);
            expected.push(pivot);
        }
        if !n.is_multiple_of(2) {
            expected.push((2 * m + 1).cast_signed());
        }

        for r in m..(n / 2) {
            let pivot = -(2 * r + 2 + n % 2).cast_signed();
            expected.push(pivot);
            expected.push(pivot);
        }

        assert_eq!(exchange, reconstruct(&ldl));
        assert_eq!(ldl.pivots(), expected);
    }
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
            let ldl = matrix.clone().ldl();
            assert!(relative_norm(&matrix, &reconstruct(&ldl)) < 1e-12);

            assert_relative_eq!(
                ldl.determinant(),
                (-1.0).powi(n_i32 / 2),
                max_relative = 1e-12
            );

            let b = random_isometry(Dyn(n), Dyn(n), &mut rng);
            assert!((&matrix * &ldl.solve(&b).unwrap() - &b).norm() / matrix.norm() < 1e-12);
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
            let ldl = matrix.clone().ldl();
            assert!(relative_norm(&matrix, &reconstruct(&ldl)) < 1e-12);

            assert_relative_eq!(
                ldl.determinant(),
                (-10.0).powi(n_i32 * (n_i32 - 1) / 2),
                max_relative = 10.0.powi(-11 + (n_i32 / 2))
            );

            let b = random_isometry(Dyn(n), Dyn(n), &mut rng);
            assert!((&matrix * &ldl.solve(&b).unwrap() - &b).norm() / matrix.norm() < 1e-12);
        }
    }
}
