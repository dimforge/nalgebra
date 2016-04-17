#![macro_use]

macro_rules! dmat_impl(
    ($dmatrix: ident, $dvector: ident) => (
        impl<N: Zero + Clone + Copy> $dmatrix<N> {
            /// Builds a matrix filled with zeros.
            ///
            /// # Arguments
            ///   * `dimension` - The dimension of the matrix. A `dimension`-dimensional matrix contains `dimension * dimension`
            ///   components.
            #[inline]
            pub fn new_zeros(nrows: usize, ncols: usize) -> $dmatrix<N> {
                $dmatrix::from_elem(nrows, ncols, ::zero())
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

        impl<N: Zero + Copy + Rand> $dmatrix<N> {
            /// Builds a matrix filled with random values.
            #[inline]
            pub fn new_random(nrows: usize, ncols: usize) -> $dmatrix<N> {
                $dmatrix::from_fn(nrows, ncols, |_, _| rand::random())
            }
        }

        impl<N: One + Zero + Clone + Copy> $dmatrix<N> {
            /// Builds a matrix filled with a given constant.
            #[inline]
            pub fn new_ones(nrows: usize, ncols: usize) -> $dmatrix<N> {
                $dmatrix::from_elem(nrows, ncols, ::one())
            }
        }

        impl<N> $dmatrix<N> {
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
            pub fn as_vector(&self) -> &[N] {
                &self.mij
            }

            /// Gets a mutable reference to this matrix data.
            /// The returned vector contains the matrix data in column-major order.
            #[inline]
            pub fn as_mut_vector(&mut self) -> &mut [N] {
                 &mut self.mij[..]
            }
        }

        // FIXME: add a function to modify the dimension (to avoid useless allocations)?

        impl<N: One + Zero + Clone + Copy> Eye for $dmatrix<N> {
            /// Builds an identity matrix.
            ///
            /// # Arguments
            /// * `dimension` - The dimension of the matrix. A `dimension`-dimensional matrix contains `dimension * dimension`
            /// components.
            #[inline]
            fn new_identity(dimension: usize) -> $dmatrix<N> {
                let mut res = $dmatrix::new_zeros(dimension, dimension);

                for i in 0..dimension {
                    let _1: N  = ::one();
                    res[(i, i)]  = _1;
                }

                res
            }
        }

        impl<N> $dmatrix<N> {
            #[inline(always)]
            fn offset(&self, i: usize, j: usize) -> usize {
                i + j * self.nrows
            }

        }

        impl<N: Copy> Indexable<(usize, usize), N> for $dmatrix<N> {
            /// Just like `set` without bounds checking.
            #[inline]
            unsafe fn unsafe_set(&mut self, rowcol: (usize, usize), val: N) {
                let (row, column) = rowcol;
                let offset = self.offset(row, column);
                *self.mij[..].get_unchecked_mut(offset) = val
            }

            /// Just like `at` without bounds checking.
            #[inline]
            unsafe fn unsafe_at(&self, rowcol: (usize,  usize)) -> N {
                let (row, column) = rowcol;

                *self.mij.get_unchecked(self.offset(row, column))
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

        impl<N> Shape<(usize, usize)> for $dmatrix<N> {
            #[inline]
            fn shape(&self) -> (usize, usize) {
                (self.nrows, self.ncols)
            }
        }

        impl<N> Index<(usize, usize)> for $dmatrix<N> {
            type Output = N;

            fn index(&self, (i, j): (usize, usize)) -> &N {
                assert!(i < self.nrows);
                assert!(j < self.ncols);

                unsafe {
                    self.mij.get_unchecked(self.offset(i, j))
                }
            }
        }

        impl<N> IndexMut<(usize, usize)> for $dmatrix<N> {
            fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut N {
                assert!(i < self.nrows);
                assert!(j < self.ncols);

                let offset = self.offset(i, j);

                unsafe {
                    self.mij[..].get_unchecked_mut(offset)
                }
            }
        }

        /*
         *
         * Multiplications matrix/matrix.
         *
         */
        impl<N> Mul<$dmatrix<N>> for $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            type Output = $dmatrix<N>;

            #[inline]
            fn mul(self, right: $dmatrix<N>) -> $dmatrix<N> {
                (&self) * (&right)
            }
        }

        impl<'a, N> Mul<&'a $dmatrix<N>> for $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            type Output = $dmatrix<N>;

            #[inline]
            fn mul(self, right: &'a $dmatrix<N>) -> $dmatrix<N> {
                (&self) * right
            }
        }

        impl<'a, N> Mul<$dmatrix<N>> for &'a $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            type Output = $dmatrix<N>;

            #[inline]
            fn mul(self, right: $dmatrix<N>) -> $dmatrix<N> {
                self * (&right)
            }
        }

        impl<'a, 'b, N> Mul<&'b $dmatrix<N>> for &'a $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            type Output = $dmatrix<N>;

            #[inline]
            fn mul(self, right: &$dmatrix<N>) -> $dmatrix<N> {
                assert!(self.ncols == right.nrows);

                let mut res = unsafe { $dmatrix::new_uninitialized(self.nrows, right.ncols) };

                for i in 0 .. self.nrows {
                    for j in 0 .. right.ncols {
                        let mut acc: N = ::zero();

                        unsafe {
                            for k in 0 .. self.ncols {
                                acc = acc + self.unsafe_at((i, k)) * right.unsafe_at((k, j));
                            }

                            res.unsafe_set((i, j), acc);
                        }
                    }
                }

                res
            }
        }

        impl<N> MulAssign<$dmatrix<N>> for $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            #[inline]
            fn mul_assign(&mut self, right: $dmatrix<N>) {
                self.mul_assign(&right)
            }
        }

        impl<'a, N> MulAssign<&'a $dmatrix<N>> for $dmatrix<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            #[inline]
            fn mul_assign(&mut self, right: &'a $dmatrix<N>) {
                assert!(self.ncols == right.nrows);

                // FIXME: optimize when both matrices have the same layout.
                let res = &*self * right;
                *self = res;
            }
        }


        /*
         *
         * Multiplication matrix/vector.
         *
         */
        impl<N> Mul<$dvector<N>> for $dmatrix<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: $dvector<N>) -> $dvector<N> {
                (&self) * (&right)
            }
        }

        impl<'a, N> Mul<$dvector<N>> for &'a $dmatrix<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: $dvector<N>) -> $dvector<N> {
                self * (&right)
            }
        }

        impl<'a, N> Mul<&'a $dvector<N>> for $dmatrix<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: &'a $dvector<N>) -> $dvector<N> {
                (&self) * right
            }
        }

        impl<'a, 'b, N> Mul<&'b $dvector<N>> for &'a $dmatrix<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: &'b $dvector<N>) -> $dvector<N> {
                assert!(self.ncols == right.len());

                let mut res : $dvector<N> = unsafe { $dvector::new_uninitialized(self.nrows) };

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

        impl<N> Mul<$dmatrix<N>> for $dvector<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: $dmatrix<N>) -> $dvector<N> {
                (&self) * (&right)
            }
        }

        impl<'a, N> Mul<$dmatrix<N>> for &'a $dvector<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: $dmatrix<N>) -> $dvector<N> {
                self * (&right)
            }
        }

        impl<'a, N> Mul<&'a $dmatrix<N>> for $dvector<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: &'a $dmatrix<N>) -> $dvector<N> {
                (&self) * right
            }
        }


        impl<'a, 'b, N> Mul<&'b $dmatrix<N>> for &'a $dvector<N>
            where N: Copy + Add<N, Output = N> + Mul<N, Output = N> + Zero {
            type Output = $dvector<N>;

            fn mul(self, right: &'b $dmatrix<N>) -> $dvector<N> {
                assert!(right.nrows == self.len());

                let mut res : $dvector<N> = unsafe { $dvector::new_uninitialized(right.ncols) };

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

        impl<N> MulAssign<$dmatrix<N>> for $dvector<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            #[inline]
            fn mul_assign(&mut self, right: $dmatrix<N>) {
                self.mul_assign(&right)
            }
        }

        impl<'a, N> MulAssign<&'a $dmatrix<N>> for $dvector<N>
            where N: Copy + Mul<N, Output = N> + Add<N, Output = N> + Zero {
            #[inline]
            fn mul_assign(&mut self, right: &'a $dmatrix<N>) {
                assert!(right.nrows == self.len());

                let res = &*self * right;
                *self = res;
            }
        }

        /*
         *
         * Addition matrix/matrix.
         *
         */
        impl<N: Copy + Add<N, Output = N>> Add<$dmatrix<N>> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn add(self, right: $dmatrix<N>) -> $dmatrix<N> {
                self + (&right)
            }
        }

        impl<'a, N: Copy + Add<N, Output = N>> Add<$dmatrix<N>> for &'a $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn add(self, right: $dmatrix<N>) -> $dmatrix<N> {
                right + self
            }
        }

        impl<'a, N: Copy + Add<N, Output = N>> Add<&'a $dmatrix<N>> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn add(self, right: &'a $dmatrix<N>) -> $dmatrix<N> {
                let mut res = self;

                for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
                    *mij = *mij + *right_ij;
                }

                res
            }
        }

        impl<N: Copy + AddAssign<N>> AddAssign<$dmatrix<N>> for $dmatrix<N> {
            #[inline]
            fn add_assign(&mut self, right: $dmatrix<N>) {
                self.add_assign(&right)
            }
        }

        impl<'a, N: Copy + AddAssign<N>> AddAssign<&'a $dmatrix<N>> for $dmatrix<N> {
            #[inline]
            fn add_assign(&mut self, right: &'a $dmatrix<N>) {
                assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                        "Unable to add matrices with different dimensions.");

                for (mij, right_ij) in self.mij.iter_mut().zip(right.mij.iter()) {
                    *mij += *right_ij;
                }
            }
        }

        /*
         *
         * Subtraction matrix/scalar.
         *
         */
        impl<N: Copy + Sub<N, Output = N>> Sub<N> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn sub(self, right: N) -> $dmatrix<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij - right;
                }

                res
            }
        }

        impl<'a, N: Copy + SubAssign<N>> SubAssign<N> for $dmatrix<N> {
            #[inline]
            fn sub_assign(&mut self, right: N) {
                for mij in self.mij.iter_mut() {
                    *mij -= right
                }
            }
        }

        impl Sub<$dmatrix<f32>> for f32 {
            type Output = $dmatrix<f32>;

            #[inline]
            fn sub(self, right: $dmatrix<f32>) -> $dmatrix<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self - *mij;
                }

                res
            }
        }

        impl Sub<$dmatrix<f64>> for f64 {
            type Output = $dmatrix<f64>;

            #[inline]
            fn sub(self, right: $dmatrix<f64>) -> $dmatrix<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self - *mij;
                }

                res
            }
        }

        /*
         *
         * Subtraction matrix/matrix.
         *
         */
        impl<N: Copy + Sub<N, Output = N>> Sub<$dmatrix<N>> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn sub(self, right: $dmatrix<N>) -> $dmatrix<N> {
                self - (&right)
            }
        }

        impl<'a, N: Copy + Sub<N, Output = N>> Sub<$dmatrix<N>> for &'a $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn sub(self, right: $dmatrix<N>) -> $dmatrix<N> {
                right - self
            }
        }

        impl<'a, N: Copy + Sub<N, Output = N>> Sub<&'a $dmatrix<N>> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn sub(self, right: &'a $dmatrix<N>) -> $dmatrix<N> {
                assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                        "Unable to subtract matrices with different dimensions.");

                let mut res = self;

                for (mij, right_ij) in res.mij.iter_mut().zip(right.mij.iter()) {
                    *mij = *mij - *right_ij;
                }

                res
            }
        }

        impl<N: Copy + SubAssign<N>> SubAssign<$dmatrix<N>> for $dmatrix<N> {
            #[inline]
            fn sub_assign(&mut self, right: $dmatrix<N>) {
                self.sub_assign(&right)
            }
        }

        impl<'a, N: Copy + SubAssign<N>> SubAssign<&'a $dmatrix<N>> for $dmatrix<N> {
            #[inline]
            fn sub_assign(&mut self, right: &'a $dmatrix<N>) {
                assert!(self.nrows == right.nrows && self.ncols == right.ncols,
                        "Unable to subtract matrices with different dimensions.");

                for (mij, right_ij) in self.mij.iter_mut().zip(right.mij.iter()) {
                    *mij -= *right_ij;
                }
            }
        }

        /*
         *
         * Inversion.
         *
         */
        impl<N: BaseNum + Clone> Inverse for $dmatrix<N> {
            #[inline]
            fn inverse(&self) -> Option<$dmatrix<N>> {
                let mut res: $dmatrix<N> = self.clone();
                if res.inverse_mut() {
                    Some(res)
                }
                else {
                    None
                }
            }

            fn inverse_mut(&mut self) -> bool {
                assert!(self.nrows == self.ncols);

                let dimension              = self.nrows;
                let mut res: $dmatrix<N> = Eye::new_identity(dimension);

                // inversion using Gauss-Jordan elimination
                for k in 0..dimension {
                    // search a non-zero value on the k-th column
                    // FIXME: would it be worth it to spend some more time searching for the
                    // max instead?

                    let mut n0 = k; // index of a non-zero entry

                    while n0 != dimension {
                        if unsafe { self.unsafe_at((n0, k)) } != ::zero() {
                            break;
                        }

                        n0 = n0 + 1;
                    }

                    if n0 == dimension {
                        return false
                    }

                    // swap pivot line
                    if n0 != k {
                        for j in 0..dimension {
                            let off_n0_j = self.offset(n0, j);
                            let off_k_j  = self.offset(k, j);

                            self.mij[..].swap(off_n0_j, off_k_j);
                            res.mij[..].swap(off_n0_j, off_k_j);
                        }
                    }

                    unsafe {
                        let pivot = self.unsafe_at((k, k));

                        for j in k..dimension {
                            let selfval = self.unsafe_at((k, j)) / pivot;
                            self.unsafe_set((k, j), selfval);
                        }

                        for j in 0..dimension {
                            let resval = res.unsafe_at((k, j)) / pivot;
                            res.unsafe_set((k, j), resval);
                        }

                        for l in 0..dimension {
                            if l != k {
                                let normalizer = self.unsafe_at((l, k));

                                for j in k..dimension {
                                    let selfval = self.unsafe_at((l, j)) - self.unsafe_at((k, j)) * normalizer;
                                    self.unsafe_set((l, j), selfval);
                                }

                                for j in 0..dimension {
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

        impl<N: Clone + Copy> Transpose for $dmatrix<N> {
            #[inline]
            fn transpose(&self) -> $dmatrix<N> {
                if self.nrows == self.ncols {
                    let mut res = self.clone();

                    res.transpose_mut();

                    res
                }
                else {
                    let mut res = unsafe { $dmatrix::new_uninitialized(self.ncols, self.nrows) };

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

        impl<N: BaseNum + Cast<f64> + Clone> Mean<$dvector<N>> for $dmatrix<N> {
            fn mean(&self) -> $dvector<N> {
                let mut res: $dvector<N> = $dvector::new_zeros(self.ncols);
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

        impl<N: BaseNum + Cast<f64> + Clone> Covariance<$dmatrix<N>> for $dmatrix<N> {
            // FIXME: this could be heavily optimized, removing all temporaries by merging loops.
            fn covariance(&self) -> $dmatrix<N> {
                assert!(self.nrows > 1);

                let mut centered = unsafe { $dmatrix::new_uninitialized(self.nrows, self.ncols) };
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

        impl<N: Copy + Zero> Column<$dvector<N>> for $dmatrix<N> {
            #[inline]
            fn ncols(&self) -> usize {
                self.ncols
            }

            #[inline]
            fn set_col(&mut self, col_id: usize, column: $dvector<N>) {
                assert!(col_id < self.ncols);
                assert!(column.len() == self.nrows);

                for row_id in 0 .. self.nrows {
                    unsafe {
                        self.unsafe_set((row_id, col_id), column.unsafe_at(row_id));
                    }
                }
            }

            fn column(&self, col_id: usize) -> $dvector<N> {
                assert!(col_id < self.ncols);

                let start = self.offset(0, col_id);
                let stop  = self.offset(self.nrows, col_id);
                $dvector::from_slice(self.nrows, &self.mij[start .. stop])
            }
        }

        impl<N: Copy + Clone + Zero> ColumnSlice<$dvector<N>> for $dmatrix<N> {
            fn col_slice(&self, col_id :usize, row_start: usize, row_end: usize) -> $dvector<N> {
                assert!(col_id < self.ncols);
                assert!(row_start < row_end);
                assert!(row_end <= self.nrows);

                // We can init from slice thanks to the matrix being column-major.
                let start = self.offset(row_start, col_id);
                let stop  = self.offset(row_end, col_id);
                let slice = $dvector::from_slice(row_end - row_start, &self.mij[start .. stop]);

                slice
            }
        }

        impl<N: Copy + Zero> Row<$dvector<N>> for $dmatrix<N> {
            #[inline]
            fn nrows(&self) -> usize {
                self.nrows
            }

            #[inline]
            fn set_row(&mut self, row_id: usize, row: $dvector<N>) {
                assert!(row_id < self.nrows);
                assert!(row.len() == self.ncols);

                for col_id in 0 .. self.ncols {
                    unsafe {
                        self.unsafe_set((row_id, col_id), row.unsafe_at(col_id));
                    }
                }
            }

            #[inline]
            fn row(&self, row_id: usize) -> $dvector<N> {
                assert!(row_id < self.nrows);

                let mut slice : $dvector<N> = unsafe {
                    $dvector::new_uninitialized(self.ncols)
                };

                for col_id in 0 .. self.ncols {
                    unsafe {
                        slice.unsafe_set(col_id, self.unsafe_at((row_id, col_id)));
                    }
                }
                slice
            }
        }

        impl<N: Copy> RowSlice<$dvector<N>> for $dmatrix<N> {
            fn row_slice(&self, row_id :usize, col_start: usize, col_end: usize) -> $dvector<N> {
                assert!(row_id < self.nrows);
                assert!(col_start < col_end);
                assert!(col_end <= self.ncols);

                let mut slice : $dvector<N> = unsafe {
                    $dvector::new_uninitialized(col_end - col_start)
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

        impl<N: Copy + Clone + Zero> Diagonal<$dvector<N>> for $dmatrix<N> {
            #[inline]
            fn from_diagonal(diagonal: &$dvector<N>) -> $dmatrix<N> {
                let mut res = $dmatrix::new_zeros(diagonal.len(), diagonal.len());

                res.set_diagonal(diagonal);

                res
            }

            #[inline]
            fn diagonal(&self) -> $dvector<N> {
                let smallest_dim = cmp::min(self.nrows, self.ncols);

                let mut diagonal: $dvector<N> = $dvector::new_zeros(smallest_dim);

                for i in 0..smallest_dim {
                    unsafe { diagonal.unsafe_set(i, self.unsafe_at((i, i))) }
                }

                diagonal
            }
        }

        impl<N: Copy + Clone + Zero> DiagMut<$dvector<N>> for $dmatrix<N> {
            #[inline]
            fn set_diagonal(&mut self, diagonal: &$dvector<N>) {
                let smallest_dim = cmp::min(self.nrows, self.ncols);

                assert!(diagonal.len() == smallest_dim);

                for i in 0..smallest_dim {
                    unsafe { self.unsafe_set((i, i), diagonal.unsafe_at(i)) }
                }
            }
        }

        impl<N: ApproxEq<N>> ApproxEq<N> for $dmatrix<N> {
            #[inline]
            fn approx_epsilon(_: Option<$dmatrix<N>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_ulps(_: Option<$dmatrix<N>>) -> u32 {
                ApproxEq::approx_ulps(None::<N>)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$dmatrix<N>, epsilon: &N) -> bool {
                let mut zip = self.mij.iter().zip(other.mij.iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$dmatrix<N>, ulps: u32) -> bool {
                let mut zip = self.mij.iter().zip(other.mij.iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
            }
        }

        impl<N: Debug + Copy> Debug for $dmatrix<N> {
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

        /*
         *
         * Multpilication matrix/scalar.
         *
         */
        impl<N: Copy + Mul<N, Output = N>> Mul<N> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn mul(self, right: N) -> $dmatrix<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij * right;
                }

                res
            }
        }

        impl Mul<$dmatrix<f32>> for f32 {
            type Output = $dmatrix<f32>;

            #[inline]
            fn mul(self, right: $dmatrix<f32>) -> $dmatrix<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self * *mij;
                }

                res
            }
        }

        impl Mul<$dmatrix<f64>> for f64 {
            type Output = $dmatrix<f64>;

            #[inline]
            fn mul(self, right: $dmatrix<f64>) -> $dmatrix<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self * *mij;
                }

                res
            }
        }

        /*
         *
         * Division matrix/scalar.
         *
         */
        impl<N: Copy + Div<N, Output = N>> Div<N> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn div(self, right: N) -> $dmatrix<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij / right;
                }

                res
            }
        }


        /*
         *
         * Addition matrix/scalar.
         *
         */
        impl<N: Copy + Add<N, Output = N>> Add<N> for $dmatrix<N> {
            type Output = $dmatrix<N>;

            #[inline]
            fn add(self, right: N) -> $dmatrix<N> {
                let mut res = self;

                for mij in res.mij.iter_mut() {
                    *mij = *mij + right;
                }

                res
            }
        }

        impl Add<$dmatrix<f32>> for f32 {
            type Output = $dmatrix<f32>;

            #[inline]
            fn add(self, right: $dmatrix<f32>) -> $dmatrix<f32> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self + *mij;
                }

                res
            }
        }

        impl Add<$dmatrix<f64>> for f64 {
            type Output = $dmatrix<f64>;

            #[inline]
            fn add(self, right: $dmatrix<f64>) -> $dmatrix<f64> {
                let mut res = right;

                for mij in res.mij.iter_mut() {
                    *mij = self + *mij;
                }

                res
            }
        }

        #[cfg(feature="arbitrary")]
        impl<N: Copy + Zero + Arbitrary> Arbitrary for $dmatrix<N> {
            fn arbitrary<G: Gen>(g: &mut G) -> $dmatrix<N> {
                $dmatrix::from_fn(
                    Arbitrary::arbitrary(g), Arbitrary::arbitrary(g),
                    |_, _| Arbitrary::arbitrary(g)
                )
            }
        }
    )
);

macro_rules! small_dmat_impl (
    ($dmatrix: ident, $dvector: ident, $dimension: expr, $($idx: expr),*) => (
        impl<N: PartialEq> PartialEq for $dmatrix<N> {
            #[inline]
            fn eq(&self, other: &$dmatrix<N>) -> bool {
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

        impl<N: Clone> Clone for $dmatrix<N> {
            fn clone(&self) -> $dmatrix<N> {
                let mij: [N; $dimension * $dimension] = [ $( self.mij[$idx].clone(), )* ];

                $dmatrix {
                    nrows: self.nrows,
                    ncols: self.ncols,
                    mij:   mij,
                }
            }
        }

        dmat_impl!($dmatrix, $dvector);
    )
);

macro_rules! small_dmat_from_impl(
    ($dmatrix: ident, $dimension: expr, $($zeros: expr),*) => (
        impl<N: Zero + Clone + Copy> $dmatrix<N> {
            /// Builds a matrix filled with a given constant.
            #[inline]
            pub fn from_elem(nrows: usize, ncols: usize, elem: N) -> $dmatrix<N> {
                assert!(nrows <= $dimension);
                assert!(ncols <= $dimension);

                let mut mij: [N; $dimension * $dimension] = [ $( $zeros, )* ];

                for n in &mut mij[.. nrows * ncols] {
                    *n = elem;
                }

                $dmatrix {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }

            /// Builds a matrix filled with the components provided by a vector.
            /// The vector contains the matrix data in row-major order.
            /// Note that `from_col_vector` is a lot faster than `from_row_vector` since a `$dmatrix` stores its data
            /// in column-major order.
            ///
            /// The vector must have at least `nrows * ncols` elements.
            #[inline]
            pub fn from_row_vector(nrows: usize, ncols: usize, vector: &[N]) -> $dmatrix<N> {
                let mut res = $dmatrix::from_col_vector(ncols, nrows, vector);
        
                // we transpose because the buffer is row_major
                res.transpose_mut();
        
                res
            }

            /// Builds a matrix filled with the components provided by a vector.
            /// The vector contains the matrix data in column-major order.
            /// Note that `from_col_vector` is a lot faster than `from_row_vector` since a `$dmatrix` stores its data
            /// in column-major order.
            ///
            /// The vector must have at least `nrows * ncols` elements.
            #[inline]
            pub fn from_col_vector(nrows: usize, ncols: usize, vector: &[N]) -> $dmatrix<N> {
                assert!(nrows * ncols == vector.len());

                let mut mij: [N; $dimension * $dimension] = [ $( $zeros, )* ];

                for (n, val) in mij[.. nrows * ncols].iter_mut().zip(vector.iter()) {
                    *n = *val;
                }
        
                $dmatrix {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }

            /// Builds a matrix using an initialization function.
            #[inline(always)]
            pub fn from_fn<F: FnMut(usize, usize) -> N>(nrows: usize, ncols: usize, mut f: F) -> $dmatrix<N> {
                assert!(nrows <= $dimension);
                assert!(ncols <= $dimension);

                let mut mij: [N; $dimension * $dimension] = [ $( $zeros, )* ];

                for i in 0 .. nrows {
                    for j in 0 .. ncols {
                        mij[i + j * nrows] = f(i, j)
                    }
                }

                $dmatrix {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mij
                }
            }
        }

        impl<N: Copy> $dmatrix<N> {
            /// Creates a new matrix with uninitialized components (with `mem::uninitialized()`).
            #[inline]
            pub unsafe fn new_uninitialized(nrows: usize, ncols: usize) -> $dmatrix<N> {
                assert!(nrows <= $dimension);
                assert!(ncols <= $dimension);

                $dmatrix {
                    nrows: nrows,
                    ncols: ncols,
                    mij:   mem::uninitialized()
                }
            }
        }
    )
);
