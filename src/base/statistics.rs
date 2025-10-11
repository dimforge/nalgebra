use crate::allocator::Allocator;
use crate::storage::RawStorage;
use crate::{Const, DefaultAllocator, Dim, Matrix, OVector, RowOVector, Scalar, U1, VectorView};
use num::{One, Zero};
use simba::scalar::{ClosedAddAssign, ClosedMulAssign, Field, SupersetOf};
use std::mem::MaybeUninit;

/// # Folding on columns and rows
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Applies a function to each column of the matrix and returns the results as a row vector.
    ///
    /// This function takes a closure `f` that processes each column independently and produces
    /// a single scalar value. The results are collected into a row vector where each element
    /// corresponds to the result of applying `f` to the respective column.
    ///
    /// This is particularly useful for computing column-wise statistics or applying custom
    /// aggregation operations across the columns of a matrix.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a column view and returns a scalar value
    ///
    /// # Returns
    ///
    /// A row vector containing the results of applying `f` to each column
    ///
    /// # Examples
    ///
    /// ## Computing column maximums
    ///
    /// ```
    /// # use nalgebra::Matrix3x4;
    /// let matrix = Matrix3x4::new(
    ///     1.0, 5.0, 9.0,  2.0,
    ///     3.0, 6.0, 10.0, 4.0,
    ///     7.0, 8.0, 11.0, 1.0,
    /// );
    ///
    /// // Find the maximum value in each column
    /// let col_max = matrix.compress_rows(|col| {
    ///     col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    /// });
    ///
    /// assert_eq!(col_max[(0, 0)], 7.0);  // max of column 0
    /// assert_eq!(col_max[(0, 1)], 8.0);  // max of column 1
    /// assert_eq!(col_max[(0, 2)], 11.0); // max of column 2
    /// assert_eq!(col_max[(0, 3)], 4.0);  // max of column 3
    /// ```
    ///
    /// ## Computing column sums
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let matrix = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// );
    ///
    /// // Sum each column
    /// let col_sums = matrix.compress_rows(|col| col.sum());
    /// assert_eq!(col_sums[(0, 0)], 5.0);  // 1 + 4
    /// assert_eq!(col_sums[(0, 1)], 7.0);  // 2 + 5
    /// assert_eq!(col_sums[(0, 2)], 9.0);  // 3 + 6
    /// ```
    ///
    /// ## Computing custom statistics
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let data = Matrix3x2::new(
    ///     10.0, 20.0,
    ///     30.0, 40.0,
    ///     50.0, 60.0,
    /// );
    ///
    /// // Compute weighted average for each column
    /// let weights = [0.5, 0.3, 0.2];
    /// let weighted_avg = data.compress_rows(|col| {
    ///     col.iter().zip(&weights).map(|(val, w)| val * w).sum()
    /// });
    ///
    /// assert_eq!(weighted_avg[(0, 0)], 24.0);  // 10*0.5 + 30*0.3 + 50*0.2 = 5+9+10 = 24
    /// assert_eq!(weighted_avg[(0, 1)], 34.0);  // 20*0.5 + 40*0.3 + 60*0.2 = 10+12+12 = 34
    /// ```
    ///
    /// # See Also
    ///
    /// * [`compress_rows_tr`](Self::compress_rows_tr) - Same operation but returns a column vector
    /// * [`compress_columns`](Self::compress_columns) - Applies a folding function across columns
    /// * [`row_sum`](Self::row_sum) - Specific case for computing column sums
    /// * [`row_mean`](Self::row_mean) - Specific case for computing column means
    #[inline]
    #[must_use]
    pub fn compress_rows(
        &self,
        f: impl Fn(VectorView<'_, T, R, S::RStride, S::CStride>) -> T,
    ) -> RowOVector<T, C>
    where
        DefaultAllocator: Allocator<U1, C>,
    {
        let ncols = self.shape_generic().1;
        let mut res = Matrix::uninit(Const::<1>, ncols);

        for i in 0..ncols.value() {
            // TODO: avoid bound checking of column.
            // Safety: all indices are in range.
            unsafe {
                *res.get_unchecked_mut((0, i)) = MaybeUninit::new(f(self.column(i)));
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Applies a function to each column of the matrix and returns the results as a column vector.
    ///
    /// This function is identical to [`compress_rows`](Self::compress_rows), but returns the
    /// result as a column vector instead of a row vector. It's equivalent to calling
    /// `self.compress_rows(f).transpose()`, but is more efficient as it avoids the transpose
    /// operation.
    ///
    /// This is convenient when you need the results in column vector form for further
    /// vector operations or when interfacing with APIs that expect column vectors.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a column view and returns a scalar value
    ///
    /// # Returns
    ///
    /// A column vector containing the results of applying `f` to each column
    ///
    /// # Examples
    ///
    /// ## Computing column norms
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let matrix = Matrix3x2::new(
    ///     3.0, 1.0,
    ///     4.0, 0.0,
    ///     0.0, 0.0,
    /// );
    ///
    /// // Compute the L2 norm of each column
    /// let norms = matrix.compress_rows_tr(|col| col.norm());
    ///
    /// assert_eq!(norms[0], 5.0);  // sqrt(3^2 + 4^2) = 5
    /// assert_eq!(norms[1], 1.0);  // sqrt(1^2) = 1
    /// ```
    ///
    /// ## Column-wise counting
    ///
    /// ```
    /// # use nalgebra::Matrix4x3;
    /// let data = Matrix4x3::new(
    ///     1.0, 0.0, 5.0,
    ///     2.0, 0.0, 6.0,
    ///     3.0, 7.0, 0.0,
    ///     4.0, 8.0, 0.0,
    /// );
    ///
    /// // Count non-zero elements in each column
    /// let non_zero_counts = data.compress_rows_tr(|col| {
    ///     col.iter().filter(|&&x| x != 0.0).count() as f64
    /// });
    ///
    /// assert_eq!(non_zero_counts[0], 4.0);  // All elements non-zero
    /// assert_eq!(non_zero_counts[1], 2.0);  // Two non-zero elements
    /// assert_eq!(non_zero_counts[2], 2.0);  // Two non-zero elements
    /// ```
    ///
    /// ## Using with further vector operations
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let matrix = Matrix2x3::new(
    ///     2.0, 4.0, 6.0,
    ///     8.0, 10.0, 12.0,
    /// );
    ///
    /// // Get column means as a column vector
    /// let col_means = matrix.compress_rows_tr(|col| col.mean());
    ///
    /// // Now we can easily compute the mean of means (overall mean)
    /// let overall_mean = col_means.mean();
    /// assert_eq!(overall_mean, 7.0);  // (5.0 + 7.0 + 9.0) / 3
    /// ```
    ///
    /// # See Also
    ///
    /// * [`compress_rows`](Self::compress_rows) - Same operation but returns a row vector
    /// * [`compress_columns`](Self::compress_columns) - Applies a folding function across columns
    /// * [`row_sum_tr`](Self::row_sum_tr) - Specific case for computing column sums as column vector
    /// * [`row_mean_tr`](Self::row_mean_tr) - Specific case for computing column means as column vector
    #[inline]
    #[must_use]
    pub fn compress_rows_tr(
        &self,
        f: impl Fn(VectorView<'_, T, R, S::RStride, S::CStride>) -> T,
    ) -> OVector<T, C>
    where
        DefaultAllocator: Allocator<C>,
    {
        let ncols = self.shape_generic().1;
        let mut res = Matrix::uninit(ncols, Const::<1>);

        for i in 0..ncols.value() {
            // TODO: avoid bound checking of column.
            // Safety: all indices are in range.
            unsafe {
                *res.vget_unchecked_mut(i) = MaybeUninit::new(f(self.column(i)));
            }
        }

        // Safety: res is now fully initialized.
        unsafe { res.assume_init() }
    }

    /// Accumulates values across columns using a folding function with an initial accumulator.
    ///
    /// This function iterates through each column of the matrix and applies the provided
    /// function `f` to update an accumulator vector. The function `f` receives a mutable
    /// reference to the accumulator and a view of the current column, allowing it to update
    /// the accumulator in place.
    ///
    /// Unlike [`compress_rows`](Self::compress_rows) which processes columns independently,
    /// `compress_columns` maintains state across iterations via the accumulator, making it
    /// ideal for computing running statistics or performing reduction operations that
    /// require maintaining intermediate state.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial accumulator vector
    /// * `f` - A function that updates the accumulator given the current column
    ///
    /// # Returns
    ///
    /// The final accumulator vector after processing all columns
    ///
    /// # Examples
    ///
    /// ## Computing row-wise sums across columns
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector3};
    /// let matrix = Matrix3x4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    /// );
    ///
    /// // Sum all elements in each row
    /// let row_sums = matrix.compress_columns(
    ///     Vector3::zeros(),
    ///     |acc, col| *acc += col
    /// );
    ///
    /// assert_eq!(row_sums[0], 10.0);  // 1 + 2 + 3 + 4
    /// assert_eq!(row_sums[1], 26.0);  // 5 + 6 + 7 + 8
    /// assert_eq!(row_sums[2], 42.0);  // 9 + 10 + 11 + 12
    /// ```
    ///
    /// ## Finding row-wise minimums
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let matrix = Matrix2x3::new(
    ///     5.0, 2.0, 8.0,
    ///     3.0, 9.0, 1.0,
    /// );
    ///
    /// // Find minimum value in each row
    /// let row_mins = matrix.compress_columns(
    ///     Vector2::repeat(f64::INFINITY),
    ///     |acc, col| {
    ///         for i in 0..acc.len() {
    ///             acc[i] = acc[i].min(col[i]);
    ///         }
    ///     }
    /// );
    ///
    /// assert_eq!(row_mins[0], 2.0);  // min(5, 2, 8)
    /// assert_eq!(row_mins[1], 1.0);  // min(3, 9, 1)
    /// ```
    ///
    /// ## Computing element-wise products across columns
    ///
    /// ```
    /// # use nalgebra::{Matrix2x4, Vector2};
    /// let matrix = Matrix2x4::new(
    ///     2.0, 3.0, 4.0, 5.0,
    ///     1.0, 2.0, 3.0, 4.0,
    /// );
    ///
    /// // Multiply all values in each row together
    /// let row_products = matrix.compress_columns(
    ///     Vector2::repeat(1.0),
    ///     |acc, col| acc.component_mul_assign(&col)
    /// );
    ///
    /// assert_eq!(row_products[0], 120.0);  // 2 * 3 * 4 * 5
    /// assert_eq!(row_products[1], 24.0);   // 1 * 2 * 3 * 4
    /// ```
    ///
    /// ## Accumulating weighted values
    ///
    /// ```
    /// # use nalgebra::{Matrix3x3, Vector3};
    /// let data = Matrix3x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    ///
    /// // Weight each column differently (0.5, 0.3, 0.2)
    /// let weights = [0.5, 0.3, 0.2];
    /// let weighted = data.compress_columns(
    ///     Vector3::zeros(),
    ///     |acc, col| {
    ///         let weight = weights[0]; // In real use, track column index
    ///         *acc += col * weight;
    ///     }
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// * [`compress_rows`](Self::compress_rows) - Applies a function independently to each column
    /// * [`compress_rows_tr`](Self::compress_rows_tr) - Like compress_rows but returns column vector
    /// * [`column_sum`](Self::column_sum) - Specific case for summing across columns
    /// * [`column_mean`](Self::column_mean) - Specific case for computing row-wise means
    #[inline]
    #[must_use]
    pub fn compress_columns(
        &self,
        init: OVector<T, R>,
        f: impl Fn(&mut OVector<T, R>, VectorView<'_, T, R, S::RStride, S::CStride>),
    ) -> OVector<T, R>
    where
        DefaultAllocator: Allocator<R>,
    {
        let mut res = init;

        for i in 0..self.ncols() {
            f(&mut res, self.column(i))
        }

        res
    }
}

/// # Common statistics operations
impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> Matrix<T, R, C, S> {
    /*
     *
     * Sum computation.
     *
     */
    /// Computes the sum of all elements in the matrix.
    ///
    /// This function adds together every element in the matrix, regardless of its position,
    /// and returns the total. The sum is computed by iterating through all elements and
    /// accumulating them using addition. For an empty matrix, the sum is zero.
    ///
    /// # Returns
    ///
    /// The sum of all matrix elements as a scalar value of type `T`
    ///
    /// # Examples
    ///
    /// ## Basic usage with floating-point values
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.sum(), 21.0);  // 1 + 2 + 3 + 4 + 5 + 6 = 21
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::Matrix3x3;
    /// let matrix = Matrix3x3::new(
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9
    /// );
    /// assert_eq!(matrix.sum(), 45);  // Sum of numbers 1 through 9
    /// ```
    ///
    /// ## Sum of a vector
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(10.0, 20.0, 30.0, 40.0);
    /// assert_eq!(v.sum(), 100.0);
    /// ```
    ///
    /// ## Practical example: computing total sales
    ///
    /// ```
    /// # use nalgebra::Matrix4x3;
    /// // Sales data: 4 products across 3 regions
    /// let sales = Matrix4x3::new(
    ///     100.0, 150.0, 200.0,  // Product A
    ///     80.0,  90.0,  110.0,  // Product B
    ///     120.0, 130.0, 140.0,  // Product C
    ///     95.0,  105.0, 115.0,  // Product D
    /// );
    ///
    /// let total_sales = sales.sum();
    /// assert_eq!(total_sales, 1435.0);
    /// ```
    ///
    /// ## Edge case: empty matrix
    ///
    /// ```
    /// # use nalgebra::Matrix0x0;
    /// let empty: Matrix0x0<f64> = Matrix0x0::from_row_slice(&[]);
    /// assert_eq!(empty.sum(), 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_sum`](Self::row_sum) - Computes the sum of each column (returns row vector)
    /// * [`column_sum`](Self::column_sum) - Computes the sum of each row (returns column vector)
    /// * [`mean`](Self::mean) - Computes the average of all elements
    /// * [`product`](Self::product) - Computes the product of all elements
    #[inline]
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: ClosedAddAssign + Zero,
    {
        self.iter().cloned().fold(T::zero(), |a, b| a + b)
    }

    /// Computes the sum of each column and returns the results as a row vector.
    ///
    /// This function calculates the sum of all elements in each column independently.
    /// For a matrix with `n` columns, the result is a row vector with `n` elements,
    /// where each element is the sum of the corresponding column.
    ///
    /// Note: Despite the name "row_sum", this function sums **columns** and returns
    /// the result as a row vector. Use [`column_sum`](Self::column_sum) to sum **rows**
    /// and get a column vector result.
    ///
    /// # Returns
    ///
    /// A row vector where element `i` contains the sum of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_sum(), RowVector3::new(5.0, 7.0, 9.0));
    /// // Column 0: 1 + 4 = 5
    /// // Column 1: 2 + 5 = 7
    /// // Column 2: 3 + 6 = 9
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, RowVector2};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.row_sum(), RowVector2::new(9, 12));
    /// // Column 0: 1 + 3 + 5 = 9
    /// // Column 1: 2 + 4 + 6 = 12
    /// ```
    ///
    /// ## Analyzing scientific data
    ///
    /// ```
    /// # use nalgebra::{Matrix5x3, RowVector3};
    /// // Temperature measurements at 3 sensors over 5 time periods
    /// let temperatures = Matrix5x3::new(
    ///     20.5, 21.0, 19.5,  // Time 1
    ///     21.0, 21.5, 20.0,  // Time 2
    ///     21.5, 22.0, 20.5,  // Time 3
    ///     22.0, 22.5, 21.0,  // Time 4
    ///     22.5, 23.0, 21.5,  // Time 5
    /// );
    ///
    /// // Total temperature recorded at each sensor
    /// let sensor_totals = temperatures.row_sum();
    /// assert_eq!(sensor_totals[0], 107.5);  // Sensor 1 total
    /// assert_eq!(sensor_totals[1], 110.0);  // Sensor 2 total
    /// assert_eq!(sensor_totals[2], 102.5);  // Sensor 3 total
    /// ```
    ///
    /// ## Financial data analysis
    ///
    /// ```
    /// # use nalgebra::{Matrix4x3, RowVector3};
    /// // Quarterly revenue for 3 products over 4 quarters
    /// let revenue = Matrix4x3::new(
    ///     100.0, 150.0, 200.0,  // Q1
    ///     110.0, 160.0, 210.0,  // Q2
    ///     120.0, 170.0, 220.0,  // Q3
    ///     130.0, 180.0, 230.0,  // Q4
    /// );
    ///
    /// // Annual revenue per product
    /// let annual_revenue = revenue.row_sum();
    /// assert_eq!(annual_revenue[0], 460.0);  // Product 1
    /// assert_eq!(annual_revenue[1], 660.0);  // Product 2
    /// assert_eq!(annual_revenue[2], 860.0);  // Product 3
    /// ```
    ///
    /// ## Converting to column vector if needed
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    ///
    /// // Use row_sum_tr() for direct column vector result
    /// let col_vec = m.row_sum_tr();
    /// assert_eq!(col_vec, Vector3::new(5.0, 7.0, 9.0));
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_sum_tr`](Self::row_sum_tr) - Same operation but returns a column vector
    /// * [`column_sum`](Self::column_sum) - Sums rows and returns a column vector
    /// * [`sum`](Self::sum) - Computes the sum of all elements
    /// * [`row_mean`](Self::row_mean) - Computes the mean of each column
    #[inline]
    #[must_use]
    pub fn row_sum(&self) -> RowOVector<T, C>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.sum())
    }

    /// Computes the sum of each column and returns the results as a column vector.
    ///
    /// This function is identical to [`row_sum`](Self::row_sum), but returns the result
    /// as a column vector instead of a row vector. It's equivalent to calling
    /// `self.row_sum().transpose()`, but is more efficient.
    ///
    /// For a matrix with `n` columns, the result is a column vector with `n` elements,
    /// where each element is the sum of the corresponding column.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the sum of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_sum_tr(), Vector3::new(5.0, 7.0, 9.0));
    /// // Column 0: 1 + 4 = 5
    /// // Column 1: 2 + 5 = 7
    /// // Column 2: 3 + 6 = 9
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector2};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.row_sum_tr(), Vector2::new(9, 12));
    /// ```
    ///
    /// ## Use case: Further vector operations
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector4};
    /// let data = Matrix3x4::new(
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    /// );
    ///
    /// // Get column sums as a vector
    /// let col_sums = data.row_sum_tr();
    /// assert_eq!(col_sums, Vector4::new(15.0, 18.0, 21.0, 24.0));
    ///
    /// // Now we can easily do vector operations
    /// let normalized = col_sums / col_sums.sum();
    /// // Each element is the proportion of the total sum
    /// ```
    ///
    /// ## Combining with other operations
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     10.0, 20.0, 30.0,
    ///     5.0,  10.0, 15.0
    /// );
    ///
    /// let col_sums = m.row_sum_tr();
    /// let col_means = m.row_mean_tr();
    ///
    /// // Verify relationship: sum = mean * count
    /// assert_eq!(col_sums[0], col_means[0] * 2.0);
    /// assert_eq!(col_sums[1], col_means[1] * 2.0);
    /// assert_eq!(col_sums[2], col_means[2] * 2.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_sum`](Self::row_sum) - Same operation but returns a row vector
    /// * [`column_sum`](Self::column_sum) - Sums rows and returns a column vector
    /// * [`sum`](Self::sum) - Computes the sum of all elements
    /// * [`row_mean_tr`](Self::row_mean_tr) - Computes the mean of each column as column vector
    #[inline]
    #[must_use]
    pub fn row_sum_tr(&self) -> OVector<T, C>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.sum())
    }

    /// Computes the sum of each row and returns the results as a column vector.
    ///
    /// This function calculates the sum of all elements in each row independently.
    /// For a matrix with `m` rows, the result is a column vector with `m` elements,
    /// where each element is the sum of the corresponding row.
    ///
    /// Note: Despite the name "column_sum", this function sums **rows** and returns
    /// the result as a column vector. Use [`row_sum`](Self::row_sum) to sum **columns**
    /// and get a row vector result.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the sum of row `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.column_sum(), Vector2::new(6.0, 15.0));
    /// // Row 0: 1 + 2 + 3 = 6
    /// // Row 1: 4 + 5 + 6 = 15
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.column_sum(), Vector3::new(3, 7, 11));
    /// // Row 0: 1 + 2 = 3
    /// // Row 1: 3 + 4 = 7
    /// // Row 2: 5 + 6 = 11
    /// ```
    ///
    /// ## Analyzing student scores
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector3};
    /// // Test scores for 3 students across 4 exams
    /// let scores = Matrix3x4::new(
    ///     85.0, 90.0, 88.0, 92.0,  // Student 1
    ///     78.0, 82.0, 80.0, 85.0,  // Student 2
    ///     92.0, 95.0, 93.0, 96.0,  // Student 3
    /// );
    ///
    /// // Total score for each student
    /// let total_scores = scores.column_sum();
    /// assert_eq!(total_scores[0], 355.0);  // Student 1: 85+90+88+92
    /// assert_eq!(total_scores[1], 325.0);  // Student 2: 78+82+80+85
    /// assert_eq!(total_scores[2], 376.0);  // Student 3: 92+95+93+96
    /// ```
    ///
    /// ## Financial portfolio analysis
    ///
    /// ```
    /// # use nalgebra::{Matrix4x5, Vector4};
    /// // Investment returns for 4 assets over 5 months
    /// let returns = Matrix4x5::new(
    ///     100.0, 120.0, 110.0, 130.0, 125.0,  // Asset 1
    ///     80.0,  85.0,  90.0,  88.0,  92.0,   // Asset 2
    ///     150.0, 160.0, 155.0, 165.0, 170.0,  // Asset 3
    ///     95.0,  98.0,  100.0, 102.0, 105.0,  // Asset 4
    /// );
    ///
    /// // Total return for each asset
    /// let total_returns = returns.column_sum();
    /// assert_eq!(total_returns[0], 585.0);  // Asset 1
    /// assert_eq!(total_returns[1], 435.0);  // Asset 2
    /// assert_eq!(total_returns[2], 800.0);  // Asset 3
    /// assert_eq!(total_returns[3], 500.0);  // Asset 4
    /// ```
    ///
    /// ## Combining row and column operations
    ///
    /// ```
    /// # use nalgebra::Matrix3x3;
    /// let matrix = Matrix3x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// );
    ///
    /// let row_totals = matrix.column_sum();    // Sum across each row
    /// let col_totals = matrix.row_sum_tr();    // Sum down each column
    ///
    /// // The sum of row totals equals the sum of column totals
    /// assert_eq!(row_totals.sum(), col_totals.sum());
    /// assert_eq!(row_totals.sum(), 45.0);  // Total of all elements
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_sum`](Self::row_sum) - Sums columns and returns a row vector
    /// * [`row_sum_tr`](Self::row_sum_tr) - Sums columns and returns a column vector
    /// * [`sum`](Self::sum) - Computes the sum of all elements
    /// * [`column_mean`](Self::column_mean) - Computes the mean of each row
    #[inline]
    #[must_use]
    pub fn column_sum(&self) -> OVector<T, R>
    where
        T: ClosedAddAssign + Zero,
        DefaultAllocator: Allocator<R>,
    {
        let nrows = self.shape_generic().0;
        self.compress_columns(OVector::zeros_generic(nrows, Const::<1>), |out, col| {
            *out += col;
        })
    }

    /*
     *
     * Product computation.
     *
     */
    /// Computes the product of all elements in the matrix.
    ///
    /// This function multiplies together every element in the matrix, regardless of its
    /// position, and returns the result. The product is computed by iterating through all
    /// elements and accumulating them using multiplication. For an empty matrix, the
    /// product is one (the multiplicative identity).
    ///
    /// # Returns
    ///
    /// The product of all matrix elements as a scalar value of type `T`
    ///
    /// # Examples
    ///
    /// ## Basic usage with floating-point values
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.product(), 720.0);  // 1 * 2 * 3 * 4 * 5 * 6 = 720
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::Matrix2x2;
    /// let matrix = Matrix2x2::new(
    ///     2, 3,
    ///     4, 5
    /// );
    /// assert_eq!(matrix.product(), 120);  // 2 * 3 * 4 * 5 = 120
    /// ```
    ///
    /// ## Product of a vector
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let v = Vector4::new(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(v.product(), 120.0);
    /// ```
    ///
    /// ## Computing factorial using a vector
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// // Compute 5! = 1 * 2 * 3 * 4 * 5
    /// let factorial_5 = Vector5::new(1, 2, 3, 4, 5);
    /// assert_eq!(factorial_5.product(), 120);
    /// ```
    ///
    /// ## Edge case: matrix containing zero
    ///
    /// ```
    /// # use nalgebra::Matrix2x2;
    /// let matrix = Matrix2x2::new(
    ///     1.0, 2.0,
    ///     0.0, 4.0  // Contains a zero
    /// );
    /// assert_eq!(matrix.product(), 0.0);  // Any zero makes the product zero
    /// ```
    ///
    /// ## Practical example: compound growth rates
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// // Quarterly growth multipliers (1.05 = 5% growth)
    /// let growth = Vector4::new(1.05, 1.03, 1.04, 1.06);
    ///
    /// // Total compound growth over the year
    /// let annual_growth = growth.product();
    /// // Approximately 1.188 or 18.8% total growth
    /// assert!((annual_growth - 1.188).abs() < 0.001);
    /// ```
    ///
    /// ## Edge case: empty matrix
    ///
    /// ```
    /// # use nalgebra::Matrix0x0;
    /// let empty: Matrix0x0<f64> = Matrix0x0::from_row_slice(&[]);
    /// assert_eq!(empty.product(), 1.0);  // Identity element for multiplication
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_product`](Self::row_product) - Computes the product of each column
    /// * [`column_product`](Self::column_product) - Computes the product of each row
    /// * [`sum`](Self::sum) - Computes the sum of all elements
    /// * [`mean`](Self::mean) - Computes the average of all elements
    #[inline]
    #[must_use]
    pub fn product(&self) -> T
    where
        T: ClosedMulAssign + One,
    {
        self.iter().cloned().fold(T::one(), |a, b| a * b)
    }

    /// Computes the product of each column and returns the results as a row vector.
    ///
    /// This function calculates the product of all elements in each column independently.
    /// For a matrix with `n` columns, the result is a row vector with `n` elements,
    /// where each element is the product of the corresponding column.
    ///
    /// Note: Despite the name "row_product", this function computes the product of
    /// **columns** and returns the result as a row vector. Use [`column_product`](Self::column_product)
    /// to compute the product of **rows** and get a column vector result.
    ///
    /// # Returns
    ///
    /// A row vector where element `i` contains the product of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_product(), RowVector3::new(4.0, 10.0, 18.0));
    /// // Column 0: 1 * 4 = 4
    /// // Column 1: 2 * 5 = 10
    /// // Column 2: 3 * 6 = 18
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, RowVector2};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.row_product(), RowVector2::new(15, 48));
    /// // Column 0: 1 * 3 * 5 = 15
    /// // Column 1: 2 * 4 * 6 = 48
    /// ```
    ///
    /// ## Computing geometric means preparation
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, RowVector2};
    /// let data = Matrix3x2::new(
    ///     2.0, 4.0,
    ///     8.0, 2.0,
    ///     4.0, 8.0,
    /// );
    ///
    /// // Product is first step in geometric mean calculation
    /// let products = data.row_product();
    /// let geom_mean_col0 = products[0].powf(1.0 / 3.0);  // Cube root
    /// let geom_mean_col1 = products[1].powf(1.0 / 3.0);
    ///
    /// assert_eq!(geom_mean_col0, 4.0);  // ∛(2*8*4) = ∛64 = 4
    /// assert_eq!(geom_mean_col1, 4.0);  // ∛(4*2*8) = ∛64 = 4
    /// ```
    ///
    /// ## Probability calculations
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    /// // Probability of success in independent trials
    /// let success_probs = Matrix2x3::new(
    ///     0.9, 0.8, 0.95,  // Trial 1 probabilities for 3 events
    ///     0.85, 0.9, 0.9,  // Trial 2 probabilities for 3 events
    /// );
    ///
    /// // Joint probability of both trials succeeding for each event
    /// let joint_probs = success_probs.row_product();
    /// assert_eq!(joint_probs[0], 0.765);  // 0.9 * 0.85
    /// assert_eq!(joint_probs[1], 0.72);   // 0.8 * 0.9
    /// assert_eq!(joint_probs[2], 0.855);  // 0.95 * 0.9
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_product_tr`](Self::row_product_tr) - Same operation but returns a column vector
    /// * [`column_product`](Self::column_product) - Computes product of each row
    /// * [`product`](Self::product) - Computes the product of all elements
    /// * [`row_sum`](Self::row_sum) - Computes the sum of each column
    #[inline]
    #[must_use]
    pub fn row_product(&self) -> RowOVector<T, C>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.product())
    }

    /// Computes the product of each column and returns the results as a column vector.
    ///
    /// This function is identical to [`row_product`](Self::row_product), but returns the
    /// result as a column vector instead of a row vector. It's equivalent to calling
    /// `self.row_product().transpose()`, but is more efficient.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the product of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_product_tr(), Vector3::new(4.0, 10.0, 18.0));
    /// // Column 0: 1 * 4 = 4
    /// // Column 1: 2 * 5 = 10
    /// // Column 2: 3 * 6 = 18
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector2};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.row_product_tr(), Vector2::new(15, 48));
    /// ```
    ///
    /// ## Computing geometric means
    ///
    /// ```
    /// # use nalgebra::{Matrix4x3, Vector3};
    /// let data = Matrix4x3::new(
    ///     1.0, 2.0, 4.0,
    ///     2.0, 4.0, 2.0,
    ///     4.0, 2.0, 1.0,
    ///     2.0, 4.0, 8.0,
    /// );
    ///
    /// let products = data.row_product_tr();
    /// // Compute geometric mean for each column
    /// let n = 4.0;
    /// let geom_means = products.map(|p| p.powf(1.0 / n));
    ///
    /// assert_eq!(geom_means[0], 2.0);  // ⁴√(1*2*4*2) = ⁴√16 = 2
    /// assert_eq!(geom_means[1], 2.0 * 2.0f64.sqrt());  // ⁴√(2*4*2*4) = ⁴√64
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_product`](Self::row_product) - Same operation but returns a row vector
    /// * [`column_product`](Self::column_product) - Computes product of each row
    /// * [`product`](Self::product) - Computes the product of all elements
    /// * [`row_sum_tr`](Self::row_sum_tr) - Computes the sum of each column as column vector
    #[inline]
    #[must_use]
    pub fn row_product_tr(&self) -> OVector<T, C>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.product())
    }

    /// Computes the product of each row and returns the results as a column vector.
    ///
    /// This function calculates the product of all elements in each row independently.
    /// For a matrix with `m` rows, the result is a column vector with `m` elements,
    /// where each element is the product of the corresponding row.
    ///
    /// Note: Despite the name "column_product", this function computes the product of
    /// **rows** and returns the result as a column vector. Use [`row_product`](Self::row_product)
    /// to compute the product of **columns** and get a row vector result.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the product of row `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.column_product(), Vector2::new(6.0, 120.0));
    /// // Row 0: 1 * 2 * 3 = 6
    /// // Row 1: 4 * 5 * 6 = 120
    /// ```
    ///
    /// ## Working with integer matrices
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, Vector3};
    /// let matrix = Matrix3x2::new(
    ///     1, 2,
    ///     3, 4,
    ///     5, 6
    /// );
    /// assert_eq!(matrix.column_product(), Vector3::new(2, 12, 30));
    /// // Row 0: 1 * 2 = 2
    /// // Row 1: 3 * 4 = 12
    /// // Row 2: 5 * 6 = 30
    /// ```
    ///
    /// ## Computing compound effects
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector3};
    /// // Efficiency multipliers for 3 machines across 4 factors
    /// let efficiency = Matrix3x4::new(
    ///     0.95, 0.98, 0.99, 0.97,  // Machine 1
    ///     0.90, 0.95, 0.93, 0.96,  // Machine 2
    ///     0.98, 0.99, 0.97, 0.98,  // Machine 3
    /// );
    ///
    /// // Overall efficiency for each machine (product of all factors)
    /// let overall = efficiency.column_product();
    /// // Machine 1: 0.95 * 0.98 * 0.99 * 0.97 ≈ 0.891
    /// assert!((overall[0] - 0.8915).abs() < 0.001);
    /// ```
    ///
    /// ## Scale factors in graphics
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// // Scaling transformations applied sequentially
    /// let scales = Matrix2x3::new(
    ///     2.0, 1.5, 0.5,  // X-axis scales
    ///     1.0, 2.0, 0.5,  // Y-axis scales
    /// );
    ///
    /// // Net scale factor for each axis
    /// let net_scale = scales.column_product();
    /// assert_eq!(net_scale[0], 1.5);  // X: 2.0 * 1.5 * 0.5
    /// assert_eq!(net_scale[1], 1.0);  // Y: 1.0 * 2.0 * 0.5
    /// ```
    ///
    /// ## Probability chains
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// // Success probabilities for sequential events
    /// let prob_chain = Matrix2x3::new(
    ///     0.9, 0.8, 0.95,  // Chain A: 3 events in sequence
    ///     0.7, 0.85, 0.9,  // Chain B: 3 events in sequence
    /// );
    ///
    /// // Overall success probability for each chain
    /// let overall_prob = prob_chain.column_product();
    /// assert_eq!(overall_prob[0], 0.684);  // 0.9 * 0.8 * 0.95
    /// assert_eq!(overall_prob[1], 0.5355); // 0.7 * 0.85 * 0.9
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_product`](Self::row_product) - Computes the product of each column
    /// * [`row_product_tr`](Self::row_product_tr) - Computes product of columns as column vector
    /// * [`product`](Self::product) - Computes the product of all elements
    /// * [`column_sum`](Self::column_sum) - Computes the sum of each row
    #[inline]
    #[must_use]
    pub fn column_product(&self) -> OVector<T, R>
    where
        T: ClosedMulAssign + One,
        DefaultAllocator: Allocator<R>,
    {
        let nrows = self.shape_generic().0;
        self.compress_columns(
            OVector::repeat_generic(nrows, Const::<1>, T::one()),
            |out, col| {
                out.component_mul_assign(&col);
            },
        )
    }

    /*
     *
     * Variance computation.
     *
     */
    /// Computes the variance of all elements in the matrix.
    ///
    /// Variance is a statistical measure of how spread out the values are from their mean.
    /// It represents the average of the squared differences from the mean. A higher
    /// variance indicates that the data points are more spread out, while a lower variance
    /// indicates they are closer to the mean.
    ///
    /// The formula used is: `variance = Σ(xᵢ - mean)² / n`
    ///
    /// where `xᵢ` are the individual elements, `mean` is the average of all elements,
    /// and `n` is the total number of elements.
    ///
    /// For an empty matrix, the variance is zero.
    ///
    /// # Returns
    ///
    /// The variance of all matrix elements as a scalar value of type `T`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// // Mean = 3.5
    /// // Variance = [(1-3.5)² + (2-3.5)² + ... + (6-3.5)²] / 6
    /// assert_relative_eq!(m.variance(), 35.0 / 12.0, epsilon = 1.0e-8);
    /// ```
    ///
    /// ## Understanding variance with uniform data
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// // All values are the same - no variation
    /// let uniform = Vector5::from_element(42.0);
    /// assert_eq!(uniform.variance(), 0.0);
    /// ```
    ///
    /// ## Comparing spread of different datasets
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// // Tightly clustered data
    /// let tight = Vector4::new(10.0, 11.0, 9.0, 10.0);
    /// let tight_var = tight.variance();
    ///
    /// // More spread out data with same mean
    /// let spread = Vector4::new(5.0, 15.0, 5.0, 15.0);
    /// let spread_var = spread.variance();
    ///
    /// // The spread data has much higher variance
    /// assert!(spread_var > tight_var);
    /// assert!(spread_var > 20.0);
    /// assert!(tight_var < 1.0);
    /// ```
    ///
    /// ## Analyzing measurement precision
    ///
    /// ```
    /// # use nalgebra::Vector6;
    /// // Measurements from a precise instrument
    /// let precise = Vector6::new(100.1, 100.0, 99.9, 100.0, 100.1, 99.9);
    /// let precise_var = precise.variance();
    ///
    /// // Measurements from a less precise instrument
    /// let imprecise = Vector6::new(98.0, 102.0, 99.0, 101.0, 97.0, 103.0);
    /// let imprecise_var = imprecise.variance();
    ///
    /// // Lower variance indicates better precision
    /// assert!(precise_var < imprecise_var);
    /// ```
    ///
    /// ## Financial risk assessment
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// // Daily returns for two investments (in percentage)
    /// let stable_asset = Vector5::new(1.0, 0.5, 1.2, 0.8, 1.0);
    /// let volatile_asset = Vector5::new(5.0, -3.0, 8.0, -2.0, 4.0);
    ///
    /// let stable_var = stable_asset.variance();
    /// let volatile_var = volatile_asset.variance();
    ///
    /// // Higher variance indicates higher risk
    /// assert!(volatile_var > stable_var);
    /// ```
    ///
    /// ## Relationship to standard deviation
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// let data = Vector4::new(2.0, 4.0, 6.0, 8.0);
    /// let var = data.variance();
    /// let std_dev = var.sqrt();
    ///
    /// // Standard deviation is the square root of variance
    /// assert_eq!(std_dev * std_dev, var);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`mean`](Self::mean) - Computes the average of all elements
    /// * [`row_variance`](Self::row_variance) - Computes variance of each column
    /// * [`column_variance`](Self::column_variance) - Computes variance of each row
    /// * Standard deviation can be computed as `matrix.variance().sqrt()`
    #[inline]
    #[must_use]
    pub fn variance(&self) -> T
    where
        T: Field + SupersetOf<f64>,
    {
        if self.is_empty() {
            T::zero()
        } else {
            let n_elements: T = crate::convert(self.len() as f64);
            let mean = self.mean();

            self.iter().cloned().fold(T::zero(), |acc, x| {
                acc + (x.clone() - mean.clone()) * (x - mean.clone())
            }) / n_elements
        }
    }

    /// Computes the variance of each column and returns the results as a row vector.
    ///
    /// For each column, this function calculates how spread out the values are from
    /// their column mean. Variance measures the average of the squared differences
    /// from the mean, providing insight into the data dispersion within each column.
    ///
    /// For a matrix with `n` columns, the result is a row vector with `n` elements,
    /// where each element is the variance of the corresponding column.
    ///
    /// Note: Despite the name "row_variance", this function computes the variance of
    /// **columns** and returns the result as a row vector.
    ///
    /// # Returns
    ///
    /// A row vector where element `i` contains the variance of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_variance(), RowVector3::new(2.25, 2.25, 2.25));
    /// // Each column has variance 2.25
    /// // Column 0: mean=2.5, var=[(1-2.5)² + (4-2.5)²]/2 = 2.25
    /// ```
    ///
    /// ## Analyzing data consistency across features
    ///
    /// ```
    /// # use nalgebra::Matrix4x3;
    /// // Measurements for 3 features across 4 samples
    /// let data = Matrix4x3::new(
    ///     10.0, 100.0, 1.0,
    ///     12.0, 105.0, 1.1,
    ///     11.0, 95.0,  0.9,
    ///     13.0, 110.0, 1.0,
    /// );
    ///
    /// let variances = data.row_variance();
    /// // Feature 0 has low variance (values 10-13)
    /// // Feature 1 has high variance (values 95-110)
    /// // Feature 2 has very low variance (values 0.9-1.1)
    /// assert!(variances[1] > variances[0]);
    /// assert!(variances[0] > variances[2]);
    /// ```
    ///
    /// ## Quality control for manufacturing
    ///
    /// ```
    /// # use nalgebra::Matrix5x2;
    /// // Measurements from 2 production lines over 5 batches
    /// let measurements = Matrix5x2::new(
    ///     10.0, 10.5,
    ///     10.1, 12.0,
    ///     9.9,  9.5,
    ///     10.0, 13.0,
    ///     10.1, 11.0,
    /// );
    ///
    /// let line_variance = measurements.row_variance();
    /// // Line 0 has low variance (more consistent)
    /// // Line 1 has high variance (less consistent, may need adjustment)
    /// assert!(line_variance[1] > line_variance[0]);
    /// ```
    ///
    /// ## Comparing variability
    ///
    /// ```
    /// # use nalgebra::{Matrix3x2, RowVector2};
    /// let data = Matrix3x2::new(
    ///     5.0, 10.0,
    ///     5.0, 5.0,
    ///     5.0, 15.0,
    /// );
    ///
    /// let vars = data.row_variance();
    /// // Column 0: all values are 5.0, variance = 0
    /// // Column 1: values vary (5, 10, 15), variance > 0
    /// assert_eq!(vars[0], 0.0);
    /// assert!(vars[1] > 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_variance_tr`](Self::row_variance_tr) - Same operation but returns a column vector
    /// * [`column_variance`](Self::column_variance) - Computes variance of each row
    /// * [`variance`](Self::variance) - Computes variance of all elements
    /// * [`row_mean`](Self::row_mean) - Computes the mean of each column
    #[inline]
    #[must_use]
    pub fn row_variance(&self) -> RowOVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.variance())
    }

    /// Computes the variance of each column and returns the results as a column vector.
    ///
    /// This function is identical to [`row_variance`](Self::row_variance), but returns
    /// the result as a column vector instead of a row vector. It's more efficient than
    /// calling `self.row_variance().transpose()`.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the variance of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_variance_tr(), Vector3::new(2.25, 2.25, 2.25));
    /// ```
    ///
    /// ## Feature scaling analysis
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// // Two features with different scales
    /// let data = Matrix3x2::new(
    ///     1.0, 1000.0,
    ///     2.0, 2000.0,
    ///     3.0, 3000.0,
    /// );
    ///
    /// let variances = data.row_variance_tr();
    /// // Feature 1 (column 1) has much higher variance due to scale
    /// assert!(variances[1] > variances[0]);
    /// ```
    ///
    /// ## Using with vector operations
    ///
    /// ```
    /// # use nalgebra::Matrix4x3;
    /// let data = Matrix4x3::new(
    ///     1.0, 5.0, 10.0,
    ///     2.0, 6.0, 12.0,
    ///     3.0, 7.0, 14.0,
    ///     4.0, 8.0, 16.0,
    /// );
    ///
    /// let variances = data.row_variance_tr();
    /// // Compute standard deviations from variances
    /// let std_devs = variances.map(|v| v.sqrt());
    ///
    /// // Standard deviation is sqrt of variance
    /// assert!(std_devs[0] > 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_variance`](Self::row_variance) - Same operation but returns a row vector
    /// * [`column_variance`](Self::column_variance) - Computes variance of each row
    /// * [`variance`](Self::variance) - Computes variance of all elements
    /// * [`row_mean_tr`](Self::row_mean_tr) - Computes mean of each column as column vector
    #[inline]
    #[must_use]
    pub fn row_variance_tr(&self) -> OVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.variance())
    }

    /// Computes the variance of each row and returns the results as a column vector.
    ///
    /// For each row, this function calculates how spread out the values are from
    /// their row mean. Variance measures the average of the squared differences
    /// from the mean, providing insight into the data dispersion within each row.
    ///
    /// For a matrix with `m` rows, the result is a column vector with `m` elements,
    /// where each element is the variance of the corresponding row.
    ///
    /// Note: Despite the name "column_variance", this function computes the variance
    /// of **rows** and returns the result as a column vector.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the variance of row `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// // Row 0: mean=2.0, var=[(1-2)² + (2-2)² + (3-2)²]/3 = 2/3
    /// // Row 1: mean=5.0, var=[(4-5)² + (5-5)² + (6-5)²]/3 = 2/3
    /// assert_relative_eq!(m.column_variance(), Vector2::new(2.0 / 3.0, 2.0 / 3.0), epsilon = 1.0e-8);
    /// ```
    ///
    /// ## Analyzing individual observations
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector3};
    /// // Three individuals measured on 4 variables
    /// let measurements = Matrix3x4::new(
    ///     10.0, 10.5, 11.0, 10.2,  // Person 1: consistent values
    ///     5.0,  15.0, 8.0,  12.0,  // Person 2: varying values
    ///     20.0, 20.1, 19.9, 20.0,  // Person 3: very consistent
    /// );
    ///
    /// let row_vars = measurements.column_variance();
    /// // Person 2 has highest variance (most variable)
    /// // Person 3 has lowest variance (most consistent)
    /// assert!(row_vars[1] > row_vars[0]);
    /// assert!(row_vars[0] > row_vars[2]);
    /// ```
    ///
    /// ## Sensor stability analysis
    ///
    /// ```
    /// # use nalgebra::{Matrix4x5, Vector4};
    /// // 4 sensors taking 5 readings each
    /// let readings = Matrix4x5::new(
    ///     100.0, 100.1, 99.9,  100.0, 100.0,  // Sensor 1: stable
    ///     100.0, 105.0, 95.0,  110.0, 90.0,   // Sensor 2: unstable
    ///     50.0,  50.0,  50.0,  50.0,  50.0,   // Sensor 3: perfectly stable
    ///     75.0,  76.0,  74.0,  75.5,  74.5,   // Sensor 4: moderately stable
    /// );
    ///
    /// let sensor_variance = readings.column_variance();
    /// // Sensor 3 has zero variance (all readings identical)
    /// // Sensor 2 has highest variance (readings vary significantly)
    /// assert_eq!(sensor_variance[2], 0.0);
    /// assert!(sensor_variance[1] > sensor_variance[0]);
    /// assert!(sensor_variance[1] > sensor_variance[3]);
    /// ```
    ///
    /// ## Comparing variability patterns
    ///
    /// ```
    /// # use nalgebra::{Matrix2x4, Vector2};
    /// let data = Matrix2x4::new(
    ///     1.0, 2.0, 3.0, 4.0,  // Evenly spaced
    ///     2.0, 2.0, 2.0, 2.0,  // No variation
    /// );
    ///
    /// let vars = data.column_variance();
    /// // Row 0 has variance (values differ)
    /// // Row 1 has zero variance (all values equal)
    /// assert!(vars[0] > 0.0);
    /// assert_eq!(vars[1], 0.0);
    /// ```
    ///
    /// ## Time series volatility
    ///
    /// ```
    /// # use nalgebra::{Matrix3x5, Vector3};
    /// // Price changes for 3 stocks over 5 days
    /// let price_changes = Matrix3x5::new(
    ///     0.5,  -0.3,  0.2,  -0.1,  0.4,   // Stock A: moderate volatility
    ///     2.0,  -1.5,  3.0,  -2.0,  1.5,   // Stock B: high volatility
    ///     0.1,  -0.05, 0.08, -0.03, 0.07,  // Stock C: low volatility
    /// );
    ///
    /// let volatility = price_changes.column_variance();
    /// // Higher variance indicates higher volatility/risk
    /// assert!(volatility[1] > volatility[0]);
    /// assert!(volatility[0] > volatility[2]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_variance`](Self::row_variance) - Computes variance of each column
    /// * [`row_variance_tr`](Self::row_variance_tr) - Computes variance of columns as column vector
    /// * [`variance`](Self::variance) - Computes variance of all elements
    /// * [`column_mean`](Self::column_mean) - Computes the mean of each row
    #[inline]
    #[must_use]
    pub fn column_variance(&self) -> OVector<T, R>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<R>,
    {
        let (nrows, ncols) = self.shape_generic();

        let mut mean = self.column_mean();
        mean.apply(|e| *e = -(e.clone() * e.clone()));

        let denom = T::one() / crate::convert::<_, T>(ncols.value() as f64);
        self.compress_columns(mean, |out, col| {
            for i in 0..nrows.value() {
                unsafe {
                    let val = col.vget_unchecked(i);
                    *out.vget_unchecked_mut(i) += denom.clone() * val.clone() * val.clone()
                }
            }
        })
    }

    /*
     *
     * Mean computation.
     *
     */
    /// Computes the mean (average) of all elements in the matrix.
    ///
    /// The mean is a fundamental statistical measure representing the central tendency
    /// of a dataset. It is calculated by summing all elements and dividing by the total
    /// number of elements.
    ///
    /// The formula used is: `mean = (Σ xᵢ) / n`
    ///
    /// where `xᵢ` are the individual elements and `n` is the total number of elements.
    ///
    /// For an empty matrix, the mean is zero.
    ///
    /// # Returns
    ///
    /// The mean of all matrix elements as a scalar value of type `T`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.mean(), 3.5);  // (1+2+3+4+5+6) / 6 = 21 / 6 = 3.5
    /// ```
    ///
    /// ## Computing average of a vector
    ///
    /// ```
    /// # use nalgebra::Vector5;
    /// let scores = Vector5::new(85.0, 90.0, 78.0, 92.0, 88.0);
    /// let average = scores.mean();
    /// assert_eq!(average, 86.6);  // (85+90+78+92+88) / 5
    /// ```
    ///
    /// ## Analyzing temperature data
    ///
    /// ```
    /// # use nalgebra::Vector7;
    /// // Daily temperatures for a week
    /// let temps = Vector7::new(22.5, 24.0, 23.5, 21.0, 20.5, 23.0, 24.5);
    /// let avg_temp = temps.mean();
    /// // Average temperature is about 22.7°C
    /// assert!((avg_temp - 22.71).abs() < 0.01);
    /// ```
    ///
    /// ## Financial data: average return
    ///
    /// ```
    /// # use nalgebra::Vector4;
    /// // Quarterly returns (in percentage)
    /// let returns = Vector4::new(5.0, -2.0, 8.0, 3.0);
    /// let avg_return = returns.mean();
    /// assert_eq!(avg_return, 3.5);  // Average return of 3.5%
    /// ```
    ///
    /// ## Relationship to sum
    ///
    /// ```
    /// # use nalgebra::Matrix2x2;
    /// let m = Matrix2x2::new(
    ///     10.0, 20.0,
    ///     30.0, 40.0
    /// );
    /// // Mean is sum divided by count
    /// assert_eq!(m.mean(), m.sum() / 4.0);
    /// assert_eq!(m.mean(), 25.0);
    /// ```
    ///
    /// ## Sensor calibration
    ///
    /// ```
    /// # use nalgebra::Vector10;
    /// // 10 measurements from a sensor (should be around 100)
    /// let measurements = Vector10::new(
    ///     99.8, 100.2, 99.9, 100.1, 100.0,
    ///     99.7, 100.3, 99.8, 100.2, 100.0
    /// );
    /// let sensor_mean = measurements.mean();
    /// // Check if sensor is properly calibrated (mean should be close to 100)
    /// assert!((sensor_mean - 100.0).abs() < 0.5);
    /// ```
    ///
    /// ## Comparing datasets
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let group_a = Vector3::new(10.0, 15.0, 20.0);
    /// let group_b = Vector3::new(30.0, 35.0, 40.0);
    ///
    /// assert_eq!(group_a.mean(), 15.0);
    /// assert_eq!(group_b.mean(), 35.0);
    /// // Group B has higher average than Group A
    /// assert!(group_b.mean() > group_a.mean());
    /// ```
    ///
    /// # See Also
    ///
    /// * [`sum`](Self::sum) - Computes the sum of all elements
    /// * [`row_mean`](Self::row_mean) - Computes the mean of each column
    /// * [`column_mean`](Self::column_mean) - Computes the mean of each row
    /// * [`variance`](Self::variance) - Measures spread around the mean
    #[inline]
    #[must_use]
    pub fn mean(&self) -> T
    where
        T: Field + SupersetOf<f64>,
    {
        if self.is_empty() {
            T::zero()
        } else {
            self.sum() / crate::convert(self.len() as f64)
        }
    }

    /// Computes the mean (average) of each column and returns the results as a row vector.
    ///
    /// For each column, this function calculates the average by summing all elements in
    /// that column and dividing by the number of rows. This provides a measure of the
    /// central tendency for each column independently.
    ///
    /// For a matrix with `n` columns, the result is a row vector with `n` elements,
    /// where each element is the mean of the corresponding column.
    ///
    /// Note: Despite the name "row_mean", this function computes the mean of **columns**
    /// and returns the result as a row vector.
    ///
    /// # Returns
    ///
    /// A row vector where element `i` contains the mean of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, RowVector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_mean(), RowVector3::new(2.5, 3.5, 4.5));
    /// // Column 0: (1 + 4) / 2 = 2.5
    /// // Column 1: (2 + 5) / 2 = 3.5
    /// // Column 2: (3 + 6) / 2 = 4.5
    /// ```
    ///
    /// ## Analyzing feature averages
    ///
    /// ```
    /// # use nalgebra::{Matrix4x3, RowVector3};
    /// // 4 observations of 3 features
    /// let data = Matrix4x3::new(
    ///     10.0, 100.0, 1.0,
    ///     12.0, 105.0, 1.2,
    ///     11.0, 95.0,  0.9,
    ///     13.0, 110.0, 1.1,
    /// );
    ///
    /// let feature_means = data.row_mean();
    /// assert_eq!(feature_means[0], 11.5);   // Feature 1 average
    /// assert_eq!(feature_means[1], 102.5);  // Feature 2 average
    /// assert_eq!(feature_means[2], 1.05);   // Feature 3 average
    /// ```
    ///
    /// ## Sensor average readings
    ///
    /// ```
    /// # use nalgebra::{Matrix5x3, RowVector3};
    /// // 5 readings from 3 sensors
    /// let readings = Matrix5x3::new(
    ///     20.0, 21.0, 19.5,
    ///     21.0, 21.5, 20.0,
    ///     20.5, 21.0, 19.8,
    ///     21.5, 22.0, 20.2,
    ///     20.0, 21.5, 19.5,
    /// );
    ///
    /// let sensor_averages = readings.row_mean();
    /// // Each sensor's average reading over the 5 measurements
    /// assert_eq!(sensor_averages[0], 20.6);
    /// assert_eq!(sensor_averages[1], 21.4);
    /// assert_eq!(sensor_averages[2], 19.8);
    /// ```
    ///
    /// ## Baseline establishment
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// // Baseline measurements for 2 metrics over 3 trials
    /// let baseline = Matrix3x2::new(
    ///     100.0, 50.0,
    ///     102.0, 48.0,
    ///     98.0,  52.0,
    /// );
    ///
    /// let baseline_means = baseline.row_mean();
    /// // These means can be used for normalization
    /// assert_eq!(baseline_means[0], 100.0);
    /// assert_eq!(baseline_means[1], 50.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_mean_tr`](Self::row_mean_tr) - Same operation but returns a column vector
    /// * [`column_mean`](Self::column_mean) - Computes the mean of each row
    /// * [`mean`](Self::mean) - Computes the mean of all elements
    /// * [`row_sum`](Self::row_sum) - Computes the sum of each column
    #[inline]
    #[must_use]
    pub fn row_mean(&self) -> RowOVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<U1, C>,
    {
        self.compress_rows(|col| col.mean())
    }

    /// Computes the mean (average) of each column and returns the results as a column vector.
    ///
    /// This function is identical to [`row_mean`](Self::row_mean), but returns the result
    /// as a column vector instead of a row vector. It's more efficient than calling
    /// `self.row_mean().transpose()`.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the mean of column `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.row_mean_tr(), Vector3::new(2.5, 3.5, 4.5));
    /// ```
    ///
    /// ## Data normalization
    ///
    /// ```
    /// # use nalgebra::Matrix3x2;
    /// let data = Matrix3x2::new(
    ///     10.0, 100.0,
    ///     20.0, 200.0,
    ///     30.0, 300.0,
    /// );
    ///
    /// let means = data.row_mean_tr();
    /// // Can be used to center the data (subtract mean from each column)
    /// assert_eq!(means[0], 20.0);   // Mean of first column
    /// assert_eq!(means[1], 200.0);  // Mean of second column
    /// ```
    ///
    /// ## Computing deviations from mean
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector3};
    /// let data = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    ///
    /// let col_means = data.row_mean_tr();
    ///
    /// // Each column's mean can be used for further analysis
    /// // For example, calculating how far each value is from its column mean
    /// let deviation_row0_col0 = data[(0, 0)] - col_means[0];
    /// assert_eq!(deviation_row0_col0, -1.5);  // 1.0 - 2.5 = -1.5
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_mean`](Self::row_mean) - Same operation but returns a row vector
    /// * [`column_mean`](Self::column_mean) - Computes the mean of each row
    /// * [`mean`](Self::mean) - Computes the mean of all elements
    /// * [`row_sum_tr`](Self::row_sum_tr) - Computes the sum of each column as column vector
    #[inline]
    #[must_use]
    pub fn row_mean_tr(&self) -> OVector<T, C>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<C>,
    {
        self.compress_rows_tr(|col| col.mean())
    }

    /// Computes the mean (average) of each row and returns the results as a column vector.
    ///
    /// For each row, this function calculates the average by summing all elements in
    /// that row and dividing by the number of columns. This provides a measure of the
    /// central tendency for each row independently.
    ///
    /// For a matrix with `m` rows, the result is a column vector with `m` elements,
    /// where each element is the mean of the corresponding row.
    ///
    /// Note: Despite the name "column_mean", this function computes the mean of **rows**
    /// and returns the result as a column vector.
    ///
    /// # Returns
    ///
    /// A column vector where element `i` contains the mean of row `i`
    ///
    /// # Examples
    ///
    /// ## Basic usage
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2};
    /// let m = Matrix2x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0
    /// );
    /// assert_eq!(m.column_mean(), Vector2::new(2.0, 5.0));
    /// // Row 0: (1 + 2 + 3) / 3 = 2.0
    /// // Row 1: (4 + 5 + 6) / 3 = 5.0
    /// ```
    ///
    /// ## Student grade averages
    ///
    /// ```
    /// # use nalgebra::{Matrix3x4, Vector3};
    /// // 3 students, 4 exam scores each
    /// let scores = Matrix3x4::new(
    ///     85.0, 90.0, 88.0, 92.0,  // Student 1
    ///     78.0, 82.0, 80.0, 85.0,  // Student 2
    ///     92.0, 95.0, 93.0, 96.0,  // Student 3
    /// );
    ///
    /// let averages = scores.column_mean();
    /// assert_eq!(averages[0], 88.75);  // Student 1 average
    /// assert_eq!(averages[1], 81.25);  // Student 2 average
    /// assert_eq!(averages[2], 94.0);   // Student 3 average
    /// ```
    ///
    /// ## Portfolio performance
    ///
    /// ```
    /// # use nalgebra::{Matrix4x5, Vector4};
    /// // 4 assets with returns over 5 months
    /// let returns = Matrix4x5::new(
    ///     2.0,  3.0,  -1.0, 4.0,  2.0,   // Asset 1
    ///     1.5,  2.5,  1.0,  2.0,  1.5,   // Asset 2
    ///     3.0,  -2.0, 5.0,  1.0,  -1.0,  // Asset 3
    ///     0.5,  1.0,  0.8,  1.2,  0.9,   // Asset 4
    /// );
    ///
    /// let avg_returns = returns.column_mean();
    /// // Average return per asset
    /// assert_eq!(avg_returns[0], 2.0);   // Asset 1
    /// assert_eq!(avg_returns[1], 1.7);   // Asset 2
    /// assert_eq!(avg_returns[2], 1.2);   // Asset 3
    /// assert!((avg_returns[3] - 0.88).abs() < 0.01);  // Asset 4
    /// ```
    ///
    /// ## Sensor daily averages
    ///
    /// ```
    /// # use nalgebra::{Matrix2x24, Vector2};
    /// // 2 sensors, 24 hourly readings
    /// let hourly_temps = Matrix2x24::from_fn(|i, j| {
    ///     if i == 0 {
    ///         20.0 + (j as f64) * 0.5  // Sensor 1: gradually warming
    ///     } else {
    ///         25.0  // Sensor 2: constant
    ///     }
    /// });
    ///
    /// let daily_avg = hourly_temps.column_mean();
    /// // Sensor 1 averages around 25.75 (starts at 20, ends at 31.5)
    /// // Sensor 2 averages 25.0 (constant)
    /// assert!((daily_avg[0] - 25.75).abs() < 0.01);
    /// assert_eq!(daily_avg[1], 25.0);
    /// ```
    ///
    /// ## Comparing row averages
    ///
    /// ```
    /// # use nalgebra::{Matrix3x5, Vector3};
    /// let data = Matrix3x5::new(
    ///     10.0, 12.0, 11.0, 13.0, 9.0,   // Low average
    ///     50.0, 52.0, 51.0, 53.0, 49.0,  // Medium average
    ///     90.0, 92.0, 91.0, 93.0, 89.0,  // High average
    /// );
    ///
    /// let row_avgs = data.column_mean();
    /// assert_eq!(row_avgs[0], 11.0);
    /// assert_eq!(row_avgs[1], 51.0);
    /// assert_eq!(row_avgs[2], 91.0);
    /// // Clear difference between rows
    /// assert!(row_avgs[2] > row_avgs[1]);
    /// assert!(row_avgs[1] > row_avgs[0]);
    /// ```
    ///
    /// ## Combining with variance for complete picture
    ///
    /// ```
    /// # use nalgebra::{Matrix2x4, Vector2};
    /// let data = Matrix2x4::new(
    ///     10.0, 10.0, 10.0, 10.0,  // Consistent values
    ///     5.0,  15.0, 8.0,  12.0,  // Varying values but same mean
    /// );
    ///
    /// let means = data.column_mean();
    /// let vars = data.column_variance();
    ///
    /// // Both rows have same mean (10.0)
    /// assert_eq!(means[0], 10.0);
    /// assert_eq!(means[1], 10.0);
    /// // But different variances
    /// assert_eq!(vars[0], 0.0);   // No variance
    /// assert!(vars[1] > 0.0);     // Has variance
    /// ```
    ///
    /// # See Also
    ///
    /// * [`row_mean`](Self::row_mean) - Computes the mean of each column
    /// * [`row_mean_tr`](Self::row_mean_tr) - Computes mean of columns as column vector
    /// * [`mean`](Self::mean) - Computes the mean of all elements
    /// * [`column_sum`](Self::column_sum) - Computes the sum of each row
    /// * [`column_variance`](Self::column_variance) - Computes the variance of each row
    #[inline]
    #[must_use]
    pub fn column_mean(&self) -> OVector<T, R>
    where
        T: Field + SupersetOf<f64>,
        DefaultAllocator: Allocator<R>,
    {
        let (nrows, ncols) = self.shape_generic();
        let denom = T::one() / crate::convert::<_, T>(ncols.value() as f64);
        self.compress_columns(OVector::zeros_generic(nrows, Const::<1>), |out, col| {
            out.axpy(denom.clone(), &col, T::one())
        })
    }
}
