use crate::{
    ComplexField, DefaultAllocator, Dim, Matrix, OMatrix, RealField, Storage, allocator::Allocator,
};
use num_traits::{FromPrimitive, One, Zero};

/// Bunch–Kaufman LDL^H factorization of a Hermitian matrix with symmetric pivoting.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<N, N>,
         OMatrix<T, N, N>: Serialize,
         Vec<isize>: Serialize,
         Option<usize>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<N, N>,
         OMatrix<T, N, N>: Deserialize<'de>,
         Vec<isize>: Deserialize<'de>,
         Option<usize>: Deserialize<'de>"))
)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[derive(Clone, Debug)]
pub struct LDL<T: ComplexField, N: Dim>
where
    DefaultAllocator: Allocator<N, N>,
{
    matrix: OMatrix<T, N, N>,
    pivots: Vec<isize>,
    zero_pivot: Option<usize>,
}

impl<T: Copy + ComplexField, N: Dim> LDL<T, N>
where
    T::RealField: Copy,
    DefaultAllocator: Allocator<N, N>,
{
    /// Compute the factorization of a complex Hermitian matrix using the Bunch-Kaufman
    /// block-diagonal pivoting method:
    ///
    /// P A P^T = L * D * L^H
    ///
    /// where L is a product of permutation and unit lower triangular matrices, U^H is the
    /// conjugate transpose of U, and D is Hermitian and block diagonal with 1-by-1 and
    /// 2-by-2 diagonal blocks.
    pub fn new(mut matrix: OMatrix<T, N, N>) -> Self {
        assert!(matrix.is_square());
        let n = matrix.nrows();

        let mut pivots = vec![0; n];
        let mut zero_pivot = None;

        // Bunch–Kaufman pivot threshold
        let alpha = T::RealField::from_f64((1.0 + 17.0.sqrt()) / 8.0).unwrap();

        // current pivot position
        let mut k = 0;

        while k < n {
            let mut block_size = 1;

            // Ensure the diagonal element is real
            matrix[(k, k)] = T::from_real(matrix[(k, k)].real());
            let diag_abs = matrix[(k, k)].real().abs();

            // Row index and magnitude of the largest off-diagonal entry in the active part
            // of column k.
            let (imax, colmax) = if k + 1 < n {
                let mut imax = k + 1;
                let mut colmax = matrix[(imax, k)].norm1();
                for i in (k + 2)..n {
                    let magnitude = matrix[(i, k)].norm1();
                    if magnitude > colmax {
                        imax = i;
                        colmax = magnitude;
                    }
                }
                (Some(imax), colmax)
            } else {
                // If k is the last column, there is no off-diagonal candidate.
                (None, T::RealField::zero())
            };

            if diag_abs.max(colmax) == T::RealField::zero() {
                // Column k is zero: store a 1x1 pivot, and skip all other logic.
                if zero_pivot.is_none() {
                    zero_pivot = Some(k + 1);
                }
                pivots[k] = (k + 1).cast_signed();
                k += 1;
                continue;
            }

            let pivot_index: usize;

            if diag_abs >= alpha * colmax {
                // The diagonal dominates column k strongly enough to use A[k, k] as a 1x1 pivot
                // without any row/column interchange.
                pivot_index = k;
            } else {
                let imax = imax.unwrap();

                let mut rowmax = T::RealField::zero();
                for j in k..imax {
                    rowmax = rowmax.max(matrix[(imax, j)].norm1());
                }
                for j in (imax + 1)..matrix.nrows() {
                    rowmax = rowmax.max(matrix[(j, imax)].norm1());
                }

                if diag_abs >= alpha * colmax * (colmax / rowmax) {
                    // Even though A[k, k] is not diagonally dominant, it is still large enough
                    // compared with the candidate row/column growth, so keep a 1x1 pivot at k.
                    pivot_index = k;
                } else {
                    pivot_index = imax;

                    if matrix[(imax, imax)].real().abs() < alpha * rowmax {
                        // The candidate diagonal at imax is also too small relative to its row
                        // maximum, so use a 2x2 pivot block involving k and k+1.
                        block_size = 2;
                    }
                }
            }

            let pivot_target: usize = k + block_size - 1;

            if pivot_index != pivot_target {
                // Hermitian two-sided interchange for the chosen pivot.
                for i in (pivot_index + 1)..matrix.nrows() {
                    // Swap entries below both indices
                    matrix.swap((i, pivot_target), (i, pivot_index));
                }

                for j in (pivot_target + 1)..pivot_index {
                    // Swap the strip between the two indices.
                    matrix.swap((j, pivot_target), (pivot_index, j));
                    matrix[(j, pivot_target)] = matrix[(j, pivot_target)].conjugate();
                    matrix[(pivot_index, j)] = matrix[(pivot_index, j)].conjugate();
                }

                // The cross entry between the swapped indices remains in the same slot.
                matrix[(pivot_index, pivot_target)] =
                    matrix[(pivot_index, pivot_target)].conjugate();

                // Swap the diagonal entries.
                matrix.swap((pivot_target, pivot_target), (pivot_index, pivot_index));

                if k + 1 == pivot_target {
                    // For a 2x2 pivot block, move the off-diagonal block entry.
                    matrix.swap((k + 1, k), (pivot_index, k));
                }
            }

            if block_size == 1 {
                // 1x1 pivot block D(k)
                if k + 1 < n {
                    let inv_diag = T::RealField::one() / matrix[(k, k)].real();

                    for j in (k + 1)..n {
                        let jk_conj = matrix[(j, k)].conjugate();

                        for i in j..n {
                            // Rank-1 Hermitian update of the trailing submatrix.
                            matrix[(i, j)] =
                                matrix[(i, j)] + matrix[(i, k)].scale(-inv_diag) * jk_conj;
                        }

                        // Keep the Hermitian diagonal explicitly real.
                        matrix[(j, j)] = T::from_real(matrix[(j, j)].real());
                    }

                    // Normalize column k so that it stores the multipliers of L.
                    for i in (k + 1)..n {
                        matrix[(i, k)] = matrix[(i, k)].scale(inv_diag);
                    }
                }

                pivots[k] = (pivot_index + 1).cast_signed();
            } else {
                // 2x2 pivot block D(k:k+1)
                if k + 2 < n {
                    // Form the scaled inverse-coefficient data for the 2x2 Hermitian pivot block.
                    let d = matrix[(k + 1, k)].abs();
                    let d11 = matrix[(k + 1, k + 1)].real() / d;
                    let d22 = matrix[(k, k)].real() / d;
                    let d21 = matrix[(k + 1, k)].unscale(d);
                    let scale = T::RealField::one() / (d * (d11 * d22 - T::RealField::one()));

                    for j in (k + 2)..n {
                        // These are the two transformed entries for row j. Together they represent
                        // the action of inv(D(k:k+1)) on the stored columns k and k+1.
                        let work1 =
                            (matrix[(j, k)].scale(d11) - matrix[(j, k + 1)] * d21).scale(scale);
                        let work2 = (matrix[(j, k + 1)].scale(d22)
                            - matrix[(j, k)] * d21.conjugate())
                        .scale(scale);

                        for i in j..n {
                            // Rank-2 Hermitian update of the trailing submatrix.
                            matrix[(i, j)] = matrix[(i, j)]
                                - matrix[(i, k)] * work1.conjugate()
                                - matrix[(i, k + 1)] * work2.conjugate();
                        }

                        matrix[(j, k)] = work1;
                        matrix[(j, k + 1)] = work2;

                        // Keep the Hermitian diagonal explicitly real.
                        matrix[(j, j)] = T::from_real(matrix[(j, j)].real());
                    }
                }

                pivots[k] = -(pivot_index + 1).cast_signed();
                pivots[k + 1] = -(pivot_index + 1).cast_signed();
            }

            k += block_size;
        }

        Self {
            matrix,
            pivots,
            zero_pivot,
        }
    }

    /// The pivot data for this factorization.
    ///
    /// This uses the LAPACK `ipiv` convention:
    /// a positive entry denotes a 1x1 pivot block, while a repeated negative entry
    /// denotes a 2x2 pivot block. The stored pivot indices are 1-based.
    #[inline]
    pub fn pivots(&self) -> &[isize] {
        &self.pivots
    }

    /// The first exactly zero pivot, if one was encountered.
    ///
    /// This follows the LAPACK convention and is therefore 1-based.
    /// A value of `None` means no zero pivot was detected.
    #[inline]
    pub const fn zero_pivot(&self) -> Option<usize> {
        self.zero_pivot
    }

    /// The permutation-aware factor of this decomposition.
    ///
    /// This is the accumulated product of the stepwise permutation and unit-lower
    /// factors. In general, once the permutations are absorbed, the result is no
    /// longer lower triangular.
    pub fn l_permuted(&self) -> OMatrix<T, N, N> {
        let n = self.matrix.nrows();
        let (nrows, ncols) = self.matrix.shape_generic();
        let mut l_permuted = OMatrix::identity_generic(nrows, ncols);

        let mut k = 0;
        while k < n {
            let pivot_index = self.pivots[k].unsigned_abs() - 1;

            if self.pivots[k] > 0 {
                // Right-multiply by the permutation: swap the affected columns.
                l_permuted.swap_columns(k, pivot_index);

                // Right-multiply by the unit-lower factor for this 1x1 step.
                for row in 0..n {
                    for i in (k + 1)..n {
                        l_permuted[(row, k)] =
                            l_permuted[(row, k)] + l_permuted[(row, i)] * self.matrix[(i, k)];
                    }
                }

                k += 1;
            } else {
                // Right-multiply by the permutation: swap the affected columns.
                l_permuted.swap_columns(k + 1, pivot_index);

                // Right-multiply by the unit-lower factor for this 2x2 step.
                for row in 0..n {
                    for i in (k + 2)..n {
                        l_permuted[(row, k)] =
                            l_permuted[(row, k)] + l_permuted[(row, i)] * self.matrix[(i, k)];
                        l_permuted[(row, k + 1)] = l_permuted[(row, k + 1)]
                            + l_permuted[(row, i)] * self.matrix[(i, k + 1)];
                    }
                }

                k += 2;
            }
        }

        l_permuted
    }

    /// The block diagonal matrix of this decomposition.
    pub fn d(&self) -> OMatrix<T, N, N> {
        let n = self.matrix.nrows();
        let (nrows, ncols) = self.matrix.shape_generic();
        let mut d = OMatrix::zeros_generic(nrows, ncols);

        let mut k = 0;
        while k < n {
            d[(k, k)] = self.matrix[(k, k)];
            if self.pivots[k] < 0 {
                d[(k + 1, k)] = self.matrix[(k + 1, k)];
                d[(k, k + 1)] = self.matrix[(k + 1, k)].conjugate();
                d[(k + 1, k + 1)] = self.matrix[(k + 1, k + 1)];
                k += 1;
            }
            k += 1;
        }

        d
    }

    /// Solves the linear system A * x = b using this factorization.
    pub fn solve<M: Dim, S>(&self, b: &Matrix<T, N, M, S>) -> Option<OMatrix<T, N, M>>
    where
        S: Storage<T, N, M>,
        DefaultAllocator: Allocator<N, M>,
    {
        let mut result = b.clone_owned();

        if self.solve_mut(&mut result) {
            Some(result)
        } else {
            None
        }
    }

    /// Solves the linear system A * x = b in place, overwriting `b` with the solution.
    pub fn solve_mut<M: Dim>(&self, b: &mut OMatrix<T, N, M>) -> bool
    where
        DefaultAllocator: Allocator<N, M>,
    {
        assert_eq!(self.matrix.nrows(), b.nrows());

        if self.zero_pivot.is_some() {
            return false;
        }

        let (n, m) = b.shape();

        // Solve L * y = P^T * b using the stored pivot sequence and multipliers.
        let mut k = 0;
        while k < n {
            let pivot_index = self.pivots[k].unsigned_abs() - 1;

            if self.pivots[k] > 0 {
                b.swap_rows(k, pivot_index);

                for j in 0..m {
                    for i in (k + 1)..n {
                        b[(i, j)] = b[(i, j)] - self.matrix[(i, k)] * b[(k, j)];
                    }
                }

                k += 1;
            } else {
                b.swap_rows(k + 1, pivot_index);

                for j in 0..m {
                    for i in (k + 2)..n {
                        b[(i, j)] = b[(i, j)]
                            - self.matrix[(i, k)] * b[(k, j)]
                            - self.matrix[(i, k + 1)] * b[(k + 1, j)];
                    }
                }

                k += 2;
            }
        }

        // Solve D * z = y, handling 1x1 and 2x2 diagonal blocks.
        let mut k = 0;
        while k < n {
            if self.pivots[k] > 0 {
                for j in 0..m {
                    b[(k, j)] = b[(k, j)].unscale(self.matrix[(k, k)].real());
                }
                k += 1;
            } else {
                let d11 = self.matrix[(k, k)].real();
                let d22 = self.matrix[(k + 1, k + 1)].real();
                let d21 = self.matrix[(k + 1, k)];

                let det = d11 * d22 - d21.modulus_squared();

                for j in 0..m {
                    let b_k = b[(k, j)];
                    let b_k1 = b[(k + 1, j)];

                    b[(k, j)] = (b_k.scale(d22) - b_k1 * d21.conjugate()).unscale(det);
                    b[(k + 1, j)] = (b_k1.scale(d11) - b_k * d21).unscale(det);
                }
                k += 2;
            }
        }

        // Solve L^H * x = z, undoing the pivot sequence in reverse order.
        let mut k = n;
        while k > 0 {
            let k1 = k - 1;

            for j in 0..m {
                for i in k..n {
                    b[(k1, j)] = b[(k1, j)] - self.matrix[(i, k1)].conjugate() * b[(i, j)];
                }
            }

            if self.pivots[k1] > 0 {
                k -= 1;
            } else {
                let k2 = k - 2;
                for j in 0..m {
                    for i in k..n {
                        b[(k2, j)] = b[(k2, j)] - self.matrix[(i, k2)].conjugate() * b[(i, j)];
                    }
                }
                k -= 2;
            }

            b.swap_rows(k1, self.pivots[k1].unsigned_abs() - 1);
        }

        true
    }

    /// Computes the determinant of the decomposed matrix.
    pub fn determinant(&self) -> T::RealField {
        let n = self.matrix.nrows();
        let mut determinant = T::RealField::one();

        let mut k = 0;
        while k < n {
            if self.pivots[k] > 0 {
                determinant *= self.matrix[(k, k)].real();
                k += 1;
            } else {
                determinant *= self.matrix[(k, k)].real() * self.matrix[(k + 1, k + 1)].real()
                    - self.matrix[(k + 1, k)].modulus_squared();
                k += 2;
            }
        }

        determinant
    }
}
