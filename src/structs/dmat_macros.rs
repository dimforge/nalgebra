#![macro_use]

macro_rules! dmat_impl(
    ($dmat: ident, $dvec: ident) => (
        impl<N: Zero + Clone + Copy> $dmat<N> {
            /// Builds a matrix filled with zeros.
            ///
            /// # Arguments
            ///   * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
            ///   components.
            #[inline]
            pub fn new_zeros(nrows: usize, ncols: usize) -> $dmat<N> {
                $dmat::from_elem(nrows, ncols, ::zero())
            }

            /// Tests if all components of the matrix are zeroes.
            #[inline]
            pub fn is_zero(&self) -> bool {
                self.mij.iter().all(|e| e.is_zero())
            }

            /// Set this matrix components to zero.
            #[inline]
            pub fn reset(&mut self) {
                for mij in self.mij.iter_mut() {
                    *mij = ::zero();
                }
            }
        }

        impl<N: Zero + Copy + Rand> $dmat<N> {
            /// Builds a matrix filled with random values.
            #[inline]
            pub fn new_random(nrows: usize, ncols: usize) -> $dmat<N> {
                $dmat::from_fn(nrows, ncols, |_, _| rand::random())
            }
        }

        impl<N: One + Zero + Clone + Copy> $dmat<N> {
            /// Builds a matrix filled with a given constant.
            #[inline]
            pub fn new_ones(nrows: usize, ncols: usize) -> $dmat<N> {
                $dmat::from_elem(nrows, ncols, ::one())
            }
        }

        impl<N> $dmat<N> {
            /// The number of row on the matrix.
            #[inline]
            pub fn nrows(&self) -> usize {
                self.nrows
            }

            /// The number of columns on the matrix.
            #[inline]
            pub fn ncols(&self) -> usize {
                self.ncols
            }

            /// Gets a reference to this matrix data.
            /// The returned vector contains the matrix data in column-major order.
            #[inline]
            pub fn as_vec(&self) -> &[N] {
                &self.mij
            }

            /// Gets a mutable reference to this matrix data.
            /// The returned vector contains the matrix data in column-major order.
            #[inline]
            pub fn as_mut_vec(&mut self) -> &mut [N] {
                 &mut self.mij[..]
            }
        }

        // FIXME: add a function to modify the dimension (to avoid useless allocations)?

        impl<N: One + Zero + Clone + Copy> Eye for $dmat<N> {
            /// Builds an identity matrix.
            ///
            /// # Arguments
            /// * `dim` - The dimension of the matrix. A `dim`-dimensional matrix contains `dim * dim`
            /// components.
            #[inline]
            fn new_identity(dim: usize) -> $dmat<N> {
                let mut res = $dmat::new_zeros(dim, dim);

                for i in 0..dim {
                    let _1: N  = ::one();
                    res[(i, i)]  = _1;
                }

                res
            }
        }

        impl<N> $dmat<N> {
            #[inline(always)]
            fn offset(&self, i: usize, j: usize) -> usize {
                i + j * self.nrows
            }

        }

        impl<N: Copy> Indexable<(usize, usize), N> for $dmat<N> {
            /// Just like `set` without bounds checking.
            #[inline]
            unsafe fn unsafe_set(&mut self, rowcol: (usize, usize), val: N) {
                let (row, col) = rowcol;
                let offset = self.offset(row, col);
                *self.mij[..].get_unchecked_mut(offset) = val
            }

            /// Just like `at` without bounds checking.
            #[inline]
            unsafe fn unsafe_at(&self, rowcol: (usize,  usize)) -> N {
                let (row, col) = rowcol;

                *self.mij.get_unchecked(self.offset(row, col))
            }

            #[inline]
            fn swap(&mut self, rowcol1: (usize, usize), rowcol2: (usize, usize)) {
                let (row1, col1) = rowcol1;
                let (row2, col2) = rowcol2;
                let offset1 = self.offset(row1, col1);
                let offset2 = self.offset(row2, col2);
                let count = self.mij.len();
                assert!(offset1 < count);
                assert!(offset2 < count);
                self.mij[..].swap(offset1, offset2);
            }

        }

        impl<N> Shape<(usize, usize)> for $dmat<N> {
            #[inline]
            fn shape(&self) -> (usize, usize) {
                (self.nrows, self.ncols)
            }
        }

        impl<N> Index<(usize, usize)> for $dmat<N> {
            type Output = N;

            fn index(&self, (i, j): (usize, usize)) -> &N {
                assert!(i < self.nrows);
                assert!(j < self.ncols);

                unsafe {
                    self.mij.get_unchecked(self.offset(i, j))
                }
            }
        }

        impl<N> IndexMut<(usize, usize)> for $dmat<N> {
            fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut N {
                assert!(i < self.nrows);
                assert!(j < self.ncols);

                let offset = self.offset(i, j);

                unsafe {
                    self.mij[..].get_unchecked_mut(offset)
                }
            }
        }

        impl<N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<$dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn mul(self, right: $dmat<N>) -> $dmat<N> {
                (&self) * (&right)
            }
        }

        impl<'a, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<&'a $dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn mul(self, right: &'a $dmat<N>) -> $dmat<N> {
                (&self) * right
            }
        }

        impl<'a, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<$dmat<N>> for &'a $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn mul(self, right: $dmat<N>) -> $dmat<N> {
                self * (&right)
            }
        }

        impl<'a, 'b, N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero> Mul<&'b $dmat<N>> for &'a $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn mul(self, right: &$dmat<N>) -> $dmat<N> {
                assert!(self.ncols == right.nrows);

                let mut res = unsafe { $dmat::new_uninitialized(self.nrows, right.ncols) };

                for i in 0..self.nrows {
                    for j in 0..right.ncols {
                        let mut acc: N = ::zero();

                        unsafe {
                            for k in 0..self.ncols {
                                acc = acc
                                    + self.unsafe_at((i, k)) * right.unsafe_at((k, j));
                            }

                            res.unsafe_set((i, j), acc);
                        }
                    }
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero> Mul<$dvec<N>> for $dmat<N> {
            type Output = $dvec<N>;

            fn mul(self, right: $dvec<N>) -> $dvec<N> {
                assert!(self.ncols == right.len());

                let mut res : $dvec<N> = unsafe { $dvec::new_uninitialized(self.nrows) };

                for i in 0..self.nrows {
                    let mut acc: N = ::zero();

                    for j in 0..self.ncols {
                        unsafe {
                            acc = acc + self.unsafe_at((i, j)) * right.unsafe_at(j);
                        }
                    }

                    unsafe {
                        res.unsafe_set(i, acc);
                    }
                }

                res
            }
        }


        impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero> Mul<$dmat<N>> for $dvec<N> {
            type Output = $dvec<N>;

            fn mul(self, right: $dmat<N>) -> $dvec<N> {
                assert!(right.nrows == self.len());

                let mut res : $dvec<N> = unsafe { $dvec::new_uninitialized(right.ncols) };

                for i in 0..right.ncols {
                    let mut acc: N = ::zero();

                    for j in 0..right.nrows {
                        unsafe {
                            acc = acc + self.unsafe_at(j) * right.unsafe_at((j, i));
                        }
                    }

                    unsafe {
                        res.unsafe_set(i, acc);
                    }
                }

                res
            }
        }

        impl<N: BaseNum + Clone> Inv for $dmat<N> {
            #[inline]
            fn inv(&self) -> Option<$dmat<N>> {
                let mut res: $dmat<N> = self.clone();
                if res.inv_mut() {
                    Some(res)
                }
                else {
                    None
                }
            }

            fn inv_mut(&mut self) -> bool {
                assert!(self.nrows == self.ncols);

                let dim              = self.nrows;
                let mut res: $dmat<N> = Eye::new_identity(dim);

                // inversion using Gauss-Jordan elimination
                for k in 0..dim {
                    // search a non-zero value on the k-th column
                    // FIXME: would it be worth it to spend some more time searching for the
                    // max instead?

                    let mut n0 = k; // index of a non-zero entry

                    while n0 != dim {
                        if unsafe { self.unsafe_at((n0, k)) } != ::zero() {
                            break;
                        }

                        n0 = n0 + 1;
                    }

                    if n0 == dim {
                        return false
                    }

                    // swap pivot line
                    if n0 != k {
                        for j in 0..dim {
                            let off_n0_j = self.offset(n0, j);
                            let off_k_j  = self.offset(k, j);

                            self.mij[..].swap(off_n0_j, off_k_j);
                            res.mij[..].swap(off_n0_j, off_k_j);
                        }
                    }

                    unsafe {
                        let pivot = self.unsafe_at((k, k));

                        for j in k..dim {
                            let selfval = self.unsafe_at((k, j)) / pivot;
                            self.unsafe_set((k, j), selfval);
                        }

                        for j in 0..dim {
                            let resval = res.unsafe_at((k, j)) / pivot;
                            res.unsafe_set((k, j), resval);
                        }

                        for l in 0..dim {
                            if l != k {
                                let normalizer = self.unsafe_at((l, k));

                                for j in k..dim {
                                    let selfval = self.unsafe_at((l, j)) - self.unsafe_at((k, j)) * normalizer;
                                    self.unsafe_set((l, j), selfval);
                                }

                                for j in 0..dim {
                                    let resval = res.unsafe_at((l, j)) - res.unsafe_at((k, j)) * normalizer;
                                    res.unsafe_set((l, j), resval);
                                }
                            }
                        }
                    }
                }

                *self = res;

                true
            }
        }

        impl<N: Clone + Copy> Transpose for $dmat<N> {
            #[inline]
            fn transpose(&self) -> $dmat<N> {
                if self.nrows == self.ncols {
                    let mut res = self.clone();

                    res.transpose_mut();

                    res
                }
                else {
                    let mut res = unsafe { $dmat::new_uninitialized(self.ncols, self.nrows) };

                    for i in 0..self.nrows {
                        for j in 0..self.ncols {
                            unsafe {
                                res.unsafe_set((j, i), self.unsafe_at((i, j)))
                            }
                        }
                    }

                    res
                }
            }

            #[inline]
            fn transpose_mut(&mut self) {
                if self.nrows == self.ncols {
                    let n = self.nrows;
                    for i in 0..n - 1 {
                        for j in i + 1..n {
                            let off_i_j = self.offset(i, j);
                            let off_j_i = self.offset(j, i);

                            self.mij[..].swap(off_i_j, off_j_i);
                        }
                    }
                }
                else {
                    // FIXME: implement a better algorithm which does that in-place.
                    *self = Transpose::transpose(self);
                }
            }
        }

        impl<N: BaseNum + Cast<f64> + Clone> Mean<$dvec<N>> for $dmat<N> {
            fn mean(&self) -> $dvec<N> {
                let mut res: $dvec<N> = $dvec::new_zeros(self.ncols);
                let normalizer: N     = Cast::from(1.0f64 / self.nrows as f64);

                for i in 0 .. self.nrows {
                    for j in 0 .. self.ncols {
                        unsafe {
                            let acc = res.unsafe_at(j) + self.unsafe_at((i, j)) * normalizer;
                            res.unsafe_set(j, acc);
                        }
                    }
                }

                res
            }
        }

        impl<N: BaseNum + Cast<f64> + Clone> Cov<$dmat<N>> for $dmat<N> {
            // FIXME: this could be heavily optimized, removing all temporaries by merging loops.
            fn cov(&self) -> $dmat<N> {
                assert!(self.nrows > 1);

                let mut centered = unsafe { $dmat::new_uninitialized(self.nrows, self.ncols) };
                let mean = self.mean();

                // FIXME: use the rows iterator when available
                for i in 0 .. self.nrows {
                    for j in 0 .. self.ncols {
                        unsafe {
                            centered.unsafe_set((i, j), self.unsafe_at((i, j)) - mean.unsafe_at(j));
                        }
                    }
                }

                // FIXME: return a triangular matrix?
                let fnormalizer: f64 = Cast::from(self.nrows() - 1);
                let normalizer: N    = Cast::from(fnormalizer);

                // FIXME: this will do 2 allocations for temporaries!
                (Transpose::transpose(&centered) * centered) / normalizer
            }
        }

        impl<N: Copy + Zero> Col<$dvec<N>> for $dmat<N> {
            #[inline]
            fn ncols(&self) -> usize {
                self.ncols
            }

            #[inline]
            fn set_col(&mut self, col_id: usize, col: $dvec<N>) {
                assert!(col_id < self.ncols);
                assert!(col.len() == self.nrows);

                for row_id in 0 .. self.nrows {
                    unsafe {
                        self.unsafe_set((row_id, col_id), col.unsafe_at(row_id));
                    }
                }
            }

            fn col(&self, col_id: usize) -> $dvec<N> {
                assert!(col_id < self.ncols);

                let start = self.offset(0, col_id);
                let stop  = self.offset(self.nrows, col_id);
                $dvec::from_slice(self.nrows, &self.mij[start .. stop])
            }
        }

        impl<N: Copy + Clone + Zero> ColSlice<$dvec<N>> for $dmat<N> {
            fn col_slice(&self, col_id :usize, row_start: usize, row_end: usize) -> $dvec<N> {
                assert!(col_id < self.ncols);
                assert!(row_start < row_end);
                assert!(row_end <= self.nrows);

                // We can init from slice thanks to the matrix being column-major.
                let start = self.offset(row_start, col_id);
                let stop  = self.offset(row_end, col_id);
                let slice = $dvec::from_slice(row_end - row_start, &self.mij[start .. stop]);

                slice
            }
        }

        impl<N: Copy + Zero> Row<$dvec<N>> for $dmat<N> {
            #[inline]
            fn nrows(&self) -> usize {
                self.nrows
            }

            #[inline]
            fn set_row(&mut self, row_id: usize, row: $dvec<N>) {
                assert!(row_id < self.nrows);
                assert!(row.len() == self.ncols);

                for col_id in 0 .. self.ncols {
                    unsafe {
                        self.unsafe_set((row_id, col_id), row.unsafe_at(col_id));
                    }
                }
            }

            #[inline]
            fn row(&self, row_id: usize) -> $dvec<N> {
                assert!(row_id < self.nrows);

                let mut slice : $dvec<N> = unsafe {
                    $dvec::new_uninitialized(self.ncols)
                };

                for col_id in 0 .. self.ncols {
                    unsafe {
                        slice.unsafe_set(col_id, self.unsafe_at((row_id, col_id)));
                    }
                }
                slice
            }
        }

        impl<N: Copy> RowSlice<$dvec<N>> for $dmat<N> {
            fn row_slice(&self, row_id :usize, col_start: usize, col_end: usize) -> $dvec<N> {
                assert!(row_id < self.nrows);
                assert!(col_start < col_end);
                assert!(col_end <= self.ncols);

                let mut slice : $dvec<N> = unsafe {
                    $dvec::new_uninitialized(col_end - col_start)
                };
                let mut slice_idx = 0;
                for col_id in col_start .. col_end {
                    unsafe {
                        slice.unsafe_set(slice_idx, self.unsafe_at((row_id, col_id)));
                    }
                    slice_idx += 1;
                }

                slice
            }
        }

        impl<N: Copy + Clone + Zero> Diag<$dvec<N>> for $dmat<N> {
            #[inline]
            fn from_diag(diag: &$dvec<N>) -> $dmat<N> {
                let mut res = $dmat::new_zeros(diag.len(), diag.len());

                res.set_diag(diag);

                res
            }

            #[inline]
            fn diag(&self) -> $dvec<N> {
                let smallest_dim = cmp::min(self.nrows, self.ncols);

                let mut diag: $dvec<N> = $dvec::new_zeros(smallest_dim);

                for i in 0..smallest_dim {
                    unsafe { diag.unsafe_set(i, self.unsafe_at((i, i))) }
                }

                diag
            }
        }

        impl<N: Copy + Clone + Zero> DiagMut<$dvec<N>> for $dmat<N> {
            #[inline]
            fn set_diag(&mut self, diag: &$dvec<N>) {
                let smallest_dim = cmp::min(self.nrows, self.ncols);

                assert!(diag.len() == smallest_dim);

                for i in 0..smallest_dim {
                    unsafe { self.unsafe_set((i, i), diag.unsafe_at(i)) }
                }
            }
        }

        impl<N: ApproxEq<N>> ApproxEq<N> for $dmat<N> {
            #[inline]
            fn approx_epsilon(_: Option<$dmat<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_ulps(_: Option<$dmat<N>>) -> u32 {
                ApproxEq::approx_ulps(None::<N>)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$dmat<N>, epsilon: &N) -> bool {
                let mut zip = self.mij.iter().zip(other.mij.iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$dmat<N>, ulps: u32) -> bool {
                let mut zip = self.mij.iter().zip(other.mij.iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
            }
        }

        impl<N: Debug + Copy> Debug for $dmat<N> {
            fn fmt(&self, form:&mut Formatter) -> Result {
                for i in 0..self.nrows() {
                    for j in 0..self.ncols() {
                        let _ = write!(form, "{:?} ", self[(i, j)]);
                    }
                    let _ = write!(form, "\n");
                }
                write!(form, "\n")
            }
        }

        impl<N: Copy + Mul<N, Output = N>> Mul<N> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn mul(self, right: N) -> $dmat<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij * right;
                }

                res
            }
        }

        impl Mul<$dmat<f32>> for f32 {
            type Output = $dmat<f32>;

            #[inline]
            fn mul(self, right: $dmat<f32>) -> $dmat<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self * *mij;
                }

                res
            }
        }

        impl Mul<$dmat<f64>> for f64 {
            type Output = $dmat<f64>;

            #[inline]
            fn mul(self, right: $dmat<f64>) -> $dmat<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self * *mij;
                }

                res
            }
        }

        impl<N: Copy + Div<N, Output = N>> Div<N> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn div(self, right: N) -> $dmat<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij / right;
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N>> Add<N> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn add(self, right: N) -> $dmat<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij + right;
                }

                res
            }
        }

        impl Add<$dmat<f32>> for f32 {
            type Output = $dmat<f32>;

            #[inline]
            fn add(self, right: $dmat<f32>) -> $dmat<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self + *mij;
                }

                res
            }
        }

        impl Add<$dmat<f64>> for f64 {
            type Output = $dmat<f64>;

            #[inline]
            fn add(self, right: $dmat<f64>) -> $dmat<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self + *mij;
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N>> Add<$dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn add(self, right: $dmat<N>) -> $dmat<N> {
                self + (&right)
            }
        }

        impl<'a, N: Copy + Add<N, Output = N>> Add<$dmat<N>> for &'a $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn add(self, right: $dmat<N>) -> $dmat<N> {
                right + self
            }
        }

        impl<'a, N: Copy + Add<N, Output = N>> Add<&'a $dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn add(self, right: &'a $dmat<N>) -> $dmat<N> {
                assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                        "Unable to add matrices with different dimensions.");

                let mut res = self;

                for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
                    *mij = *mij + *right_ij;
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N>> Sub<N> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn sub(self, right: N) -> $dmat<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij - right;
                }

                res
            }
        }

        impl Sub<$dmat<f32>> for f32 {
            type Output = $dmat<f32>;

            #[inline]
            fn sub(self, right: $dmat<f32>) -> $dmat<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self - *mij;
                }

                res
            }
        }

        impl Sub<$dmat<f64>> for f64 {
            type Output = $dmat<f64>;

            #[inline]
            fn sub(self, right: $dmat<f64>) -> $dmat<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self - *mij;
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N>> Sub<$dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn sub(self, right: $dmat<N>) -> $dmat<N> {
                self - (&right)
            }
        }

        impl<'a, N: Copy + Sub<N, Output = N>> Sub<$dmat<N>> for &'a $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn sub(self, right: $dmat<N>) -> $dmat<N> {
                right - self
            }
        }

        impl<'a, N: Copy + Sub<N, Output = N>> Sub<&'a $dmat<N>> for $dmat<N> {
            type Output = $dmat<N>;

            #[inline]
            fn sub(self, right: &'a $dmat<N>) -> $dmat<N> {
                assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                        "Unable to subtract matrices with different dimensions.");

                let mut res = self;

                for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
                    *mij = *mij - *right_ij;
                }

                res
            }
        }

        #[cfg(feature="arbitrary")]
        impl<N: Copy + Zero + Arbitrary> Arbitrary for $dmat<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $dmat<N> {
                $dmat::from_fn(
                    Arbitrary::arbitrary(g), Arbitrary::arbitrary(g),
                    |_, _| Arbitrary::arbitrary(g)
                )
            }
        }
    )
);

macro_rules! small_dmat_impl (
    ($dmat: ident, $dvec: ident, $dim: expr, $($idx: expr),*) => (
        impl<N: PartialEq> PartialEq for $dmat<N> {
            #[inline]
            fn eq(&self, other: &$dmat<N>) -> bool {
                if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
                    return false; // FIXME: fail instead?
                }

                for (a, b) in self.mij[0 .. self.nrows() * self.ncols()].iter().zip(
                                other.mij[0 .. self.nrows() * self.ncols()].iter()) {
                    if *a != *b {
                        return false;
                    }
                }

                true
            }
        }

        impl<N: Clone> Clone for $dmat<N> {
            fn clone(&self) -> $dmat<N> {
                let mij: [N; $dim * $dim] = [ $( self.mij[$idx].clone(), )* ];

                $dmat {
                    nrows: self.nrows,
                    ncols: self.ncols,
                    mij:   mij,
                }
            }
        }

        dmat_impl!($dmat, $dvec);
    )
);

macro_rules! small_dmat_from_impl(
    ($dmat: ident, $dim: expr, $($zeros: expr),*) => (
        impl<N: Zero + Clone + Copy> $dmat<N> {
            /// Builds a matrix filled with a given constant.
            #[inline]
            pub fn from_elem(nrows: usize, ncols: usize, elem: N) -> $dmat<N> {
                assert!(nrows <= $dim);
                assert!(ncols <= $dim);

                let mut mij: [N; $dim * $dim] = [ $( $zeros, )* ];

                for n in &mut mij[.. nrows * ncols] {
                    *n = elem;
                }

                $dmat {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }

            /// Builds a matrix filled with the components provided by a vector.
            /// The vector contains the matrix data in row-major order.
            /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `$dmat` stores its data
            /// in column-major order.
            ///
            /// The vector must have at least `nrows * ncols` elements.
            #[inline]
            pub fn from_row_vec(nrows: usize, ncols: usize, vec: &[N]) -> $dmat<N> {
                let mut res = $dmat::from_col_vec(ncols, nrows, vec);
        
                // we transpose because the buffer is row_major
                res.transpose_mut();
        
                res
            }

            /// Builds a matrix filled with the components provided by a vector.
            /// The vector contains the matrix data in column-major order.
            /// Note that `from_col_vec` is a lot faster than `from_row_vec` since a `$dmat` stores its data
            /// in column-major order.
            ///
            /// The vector must have at least `nrows * ncols` elements.
            #[inline]
            pub fn from_col_vec(nrows: usize, ncols: usize, vec: &[N]) -> $dmat<N> {
                assert!(nrows * ncols == vec.len());

                let mut mij: [N; $dim * $dim] = [ $( $zeros, )* ];

                for (n, val) in mij[.. nrows * ncols].iter_mut().zip(vec.iter()) {
                    *n = *val;
                }
        
                $dmat {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }

            /// Builds a matrix using an initialization function.
            #[inline(always)]
            pub fn from_fn<F: FnMut(usize, usize) -> N>(nrows: usize, ncols: usize, mut f: F) -> $dmat<N> {
                assert!(nrows <= $dim);
                assert!(ncols <= $dim);

                let mut mij: [N; $dim * $dim] = [ $( $zeros, )* ];

                for i in 0 .. nrows {
                    for j in 0 .. ncols {
                        mij[i + j * nrows] = f(i, j)
                    }
                }

                $dmat {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }
        }

        impl<N: Copy> $dmat<N> {
            /// Creates a new matrix with uninitialized components (with `mem::uninitialized()`).
            #[inline]
            pub unsafe fn new_uninitialized(nrows: usize, ncols: usize) -> $dmat<N> {
                assert!(nrows <= $dim);
                assert!(ncols <= $dim);

                $dmat {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mem::uninitialized()
                }
            }
        }
    )
);
