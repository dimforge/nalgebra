# Quick reference

Free functions are noted with a leading `::` while methods start with a dot.
Most types are type aliases. Refer to the [API
documentation](../rustdoc_nalgebra) for details about the functions arguments
and type parameters.

* [Matrices and vectors](#matrices-and-vectors)
    * [Construction](#construction), [Common methods](#common-methods), [Slicing](#slicing), [Resizing](#resizing)
    * [Blas operations](#blas-operations)
    * [Decompositions](#decompositions)
    * [Computer graphics](#computer-graphics)
* [Geometry](#geometry)
    * [Points and transformations](#geometry)
    * [Projections](#projections)
* [Data storages and allocators](#data-storage-and-allocators)

Serialization with [serde](https://serde.rs) can be enabled by enabling the
**serde-serialize** feature.

### Matrices and vectors

* Within this reference, `N` is the scalar type, `R`, `C` and `D` are type-level integers.  
* Matrices are stored in column-major order.
* Vectors are type aliases for matrices with only one column or one row.
* Overloaded operators: `*`, `/`, `+`, `-` (binary and unary), and corresponding assignement operators.
* Comparison operators `==`, `>=`, `<=`, `>`, `<`,  using column-major lexicographic ordering.
* Mutable and non-mutable indexing: `the_vector[usize]` and `the_matrix[(usize, usize)]`.
* All angles are in radian.

-----

`Matrix<...>`                                       <span style="float:right;">Generic matrix type.</span><br/>
`Matrix1<N> .. Matrix6<N>`, `MatrixN<N, D>`         <span style="float:right;">Statically-sized square matrix.</span><br/>
`Matrix1x2<N> .. Matrix6x5<N>`, `MatrixMN<N, R, C>` <span style="float:right;">Statically-sized rectangular matrix.</span><br/>
`DMatrix<N>`                                        <span style="float:right;">Dynamically-sized matrix.</span>

<br/>

`Vector1<N> .. Vector6<N>`, `VectorN<N, D>` <span style="float:right;">Statically-sized column vector.</span><br/>
`DVector<N>`                                <span style="float:right;">Dynamically-sized column vector.</span>

<br/>

`RowVector1<N> .. RowVector6<N>`, `RowVectorN<N, D>` <span style="float:right;">Statically-sized row vector.</span><br/>
`RowDVector<N>`                                      <span style="float:right;">Dynamically-sized row vector.</span>

<br/>

`Unit<T>`                                            <span style="float:right;">Wrapper that ensures the underlying value of type `T` is normalized.</span>

<br/>

`MatrixSlice1<N> .. MatrixSlice6<N>`, `MatrixSliceN<N, D>`         <span style="float:right;">Statically-sized square matrix slice.</span><br/>
`MatrixSlice1x2<N> .. MatrixSlice6x5<N>`, `MatrixSliceMN<N, R, C>` <span style="float:right;">Statically-sized rectangular matrix slice.</span><br/>
`MatrixSlice1xX<N> .. MatrixSlice6xX<N>`                           <span style="float:right;">Rectangular matrix slice with a dynamic number of columns.</span><br/>
`MatrixSliceXx1<N> .. MatrixSliceXx6<N>`                           <span style="float:right;">Rectangular matrix slice with a dynamic number of rows.</span><br/>
`DMatrixSlice<N>`                                                  <span style="float:right;">Dynamically-sized matrix slice.</span><br/>

Add `Mut` before the dimension numbers for mutable slice types, e.g., `MatrixSliceMut1x2<N>`, `DMatrixSliceMut<N>`.

<br/>

`VectorSlice1<N> .. VectorSlice6<N>`, `VectorSliceN<N, D>` <span style="float:right;">Statically-sized column vector.</span><br/>
`DVectorSlice<N>`                                          <span style="float:right;">Dynamically-sized column vector.</span><br/>

Add `Mut` before the dimension numbers for mutable slice types, e.g., `VectorSliceMut3<N>`, `DVectorSliceMut<N>`.

<br/>

-----

#### Construction
The exact number of argument, and the existence of some constructors depend on
the result matrix type. For example, dynamically-sized or 2D vectors do not
have a `::z()` constructors. In the following "Low-dimensional" means with
dimensions 1 to 6 for vectors and 1x1 to 6x6 for matrices.


`::x(), ::y(), ::z(), ::w(), ::a(), ::b()` <span style="float:right;">Low-dimensional vector with one component set to 1 and others to 0.</span><br/>
`::x_axis() .. ::b_axis()`                 <span style="float:right;">Low-dimensional vector wrapped in `Unit<...>`.</span><br/>
`::new(x, y, .. b)`                        <span style="float:right;">Low-dimensional vector constructed from its components values.</span><br/>
`::new(m11, m12, .. m66)`                  <span style="float:right;">Low-dimensional matrix constructed from its components values.</span><br/>

<br/>

In the following, ellipsis `...` are either no arguments, one `usize` argument
(the number of rows or the number of columns), or two `usize` arguments (the
number of rows and columns), depending on the number of dimensions unknown at
compile-time of the matrix being created.

`::new_uninitialized(...)`       <span style="float:right;">_[unsafe]_ Matrix with uninitialized components.</span><br/>
`::new_random(...)`              <span style="float:right;">Matrix filled with random values.</span><br/>
`::identity(...)`                <span style="float:right;">The identity matrix.</span><br/>
`::zeros(...)`                   <span style="float:right;">Matrix filled with zeros.</span><br/>
`::repeat(..., value)`           <span style="float:right;">Matrix filled with the given value.</span><br/>
`::from_element(..., value)`     <span style="float:right;">Same as `.from_element`.</span><br/>
`::from_iterator(..., iterator)` <span style="float:right;">Matrix filled with the content of the given iterator.</span><br/>
`::from_row_slice(..., array)`   <span style="float:right;">Matrix filled with the content of `array` given in **row-major** order.</span><br/>
`::from_column_slice(..., array)`         <span style="float:right;">Matrix filled with the content of `array` given in **column-major** order.</span><br/>
`::from_fn(..., closure)`                 <span style="float:right;">Matrix filled with the result of a closure called for each entry.</span><br/>
`::from_diagonal(..., vector)`            <span style="float:right;">Diagonal square matrix with the given diagonal vector.</span><br/>
`::from_diagonal_element(..., value)`     <span style="float:right;">Diagonal square matrix with the diagonal filled with one value.</span><br/>
`::from_partial_diagonal(..., &[values])` <span style="float:right;">Rectangular matrix with diagonal filled with the given values.</span><br/>
`::from_rows(..., &[vectors])`            <span style="float:right;">Matrix formed by the concatenation of the given rows.</span><br/>
`::from_columns(..., &[vectors])`         <span style="float:right;">Matrix formed by the concatenation of the given columns.</span><br/>

<br/>

`Zero::zero()`         <span style="float:right;">Matrix filled with zeroes.</span><br/>
`One::one()`           <span style="float:right;">The identity matrix.</span><br/>
`Bounded::min_value()` <span style="float:right;">Matrix filled with the min value of the scalar type.</span><br/>
`Bounded::max_value()` <span style="float:right;">Matrix filled with the max value of the scalar type.</span><br/>
`Rand::rand(rng)`      <span style="float:right;">Matrix filled with random values.</span><br/>

----

#### Common methods

`.len()`     <span style="float:right;">The number of components.</span><br/>
`.shape()`   <span style="float:right;">The number of rows and columns.</span><br/>
`.nrows()`   <span style="float:right;">The number of rows.</span><br/>
`.ncols()`   <span style="float:right;">The number of columns.</span><br/>
`.strides()` <span style="float:right;">Number of skipped original rows/columns in-between each row/column of a slice.</span><br/>

<br/>

`.iter()`                  <span style="float:right;">An iterator through the matrix components in column-major order.</span><br/>
`.iter_mut()`              <span style="float:right;">A mutable iterator through the matrix components in column-major order.</span><br/>
`.get_unchecked(i, j)`     <span style="float:right;">_[unsafe]_ Component at row `i` and column `j`. No bound checking.</span><br/>
`.get_unchecked_mut(i, j)` <span style="float:right;">_[unsafe]_ Mutable component at row `i` and column `j`. No bound checking.</span><br/>
`.swap_unchecked(i, j)`    <span style="float:right;">_[unsafe]_ Swaps two components. No bound checking.</span><br/>
`.as_slice()`              <span style="float:right;">Reference to the internal column-major array of component.</span><br/>
`.as_mut_slice()`          <span style="float:right;">Mutable reference to the internal column-major array of component.</span><br/>

<br/>

`.upper_triangle()`         <span style="float:right;">Extracts the upper triangle, including the diagonal.</span><br/>
`.lower_triangle()`         <span style="float:right;">Extracts the lower triangle, including the diagonal.</span><br/>
`.swap_rows(id_row1, id_row2)`    <span style="float:right;">Swaps two rows.</span><br/>
`.swap_columns(id_col1, id_col2)` <span style="float:right;">Swaps two columns.</span><br/>

<br/>

`.copy_from(matrix)`          <span style="float:right;">Copies the content of another matrix with the same shape.</span><br/>
`.fill(value)`                <span style="float:right;">Sets all components to `value`.</span><br/>
`.fill_diagonal(value)`       <span style="float:right;">Fills the matrix diagonal with a single value.</span><br/>
`.fill_lower_triangle(value)` <span style="float:right;">Fills some sub-diagonals below the main diagonal with a value.</span><br/>
`.fill_upper_triangle(value)` <span style="float:right;">Fills some sub-diagonals above the main diagonal with a value.</span><br/>
`.map(f)`                     <span style="float:right;">Applies `f` to each component and stores the results on a new matrix.</span><br/>
`.apply(f)`                   <span style="float:right;">Applies in-place `f` to each component of the matrix.</span><br/>
`.zip_map(m2, f)`             <span style="float:right;">Applies `f` to pairs of components from `self` and `m2` into a new matrix.</span><br/>

<br/>

`.relative_eq(abolutes_eps, relative_eps)` <span style="float:right;">Componentwise approximate matrix equality.</span><br />
`.component_mul(rhs)`                      <span style="float:right;">Componentwise multiplication (aka. Hadamard product).</span><br />
`.component_mul_assign(rhs)`               <span style="float:right;">In-place componentwise multiplication (aka. Hadamard product).</span><br />
`.component_div(rhs)`                      <span style="float:right;">Componentwise division.</span><br />
`.component_div_assign(rhs)`               <span style="float:right;">In-place componentwise division.</span><br />

<br/>

`.transpose()`                    <span style="float:right;">Matrix transposition.</span><br />
`.transpose_mut()`                <span style="float:right;">In-place matrix transposition.</span><br />
`.transpose_to(output)`           <span style="float:right;">Transposes a matrix to the given output.</span><br />
`.conjugate_transpose()`          <span style="float:right;">Complex matrix transposed conjugate.</span><br />
`.conjugate_transpose_mut()`      <span style="float:right;">In-place complex matrix transposed conjugate.</span><br />
`.conjugate_transpose_to(output)` <span style="float:right;">Conjugate-transposes a complex matrix to the given output matrix.</span><br />
`.try_inverse()`                  <span style="float:right;">Matrix inverse. Returns `None` if it fails.</span><br />
`.try_inverse_mut()`              <span style="float:right;">In-place matrix inverse. Returns `false` if it fails.</span><br />

<br/>

`.diagonal()`          <span style="float:right;">The matrix diagonal.</span><br />
`.abs()`               <span style="float:right;">The absolute value of this matrix components.</span><br />
`.determinant()`       <span style="float:right;">The matrix determinant.</span><br />
`.trace()`             <span style="float:right;">The trace of this matrix.</span><br />
`.norm()`              <span style="float:right;">The L2 norm.</span><br />
`.norm_squared()`      <span style="float:right;">The squared L2 norm.</span><br />
`.normalize()`         <span style="float:right;">Normalization using the L2 norm.</span><br />
`.normalize_mut()`     <span style="float:right;">In-place normalization using the L2 norm.</span><br />
`.try_normalize()`     <span style="float:right;">Normalization. Returns `None` if the norm is too small.</span><br />
`.try_normalize_mut()` <span style="float:right;">In-place normalization.  Returns `None` if the norm is too small.</span><br />

<br/>

`.dot(rhs)`       <span style="float:right;">Vector dot product.</span><br />
`.tr_dot(rhs)`    <span style="float:right;">Vector dot product between `self.transpose()` and `rhs`.</span><br />
`.perp(rhs)`      <span style="float:right;">2D cross product, i.e., determinant of the matrix formed by two 2D column vectors.</span><br />
`.cross(rhs)`      <span style="float:right;">3D cross product.</span><br />
`.kronecker(rhs)`  <span style="float:right;">Matrix tensor (kronecker) product.</span><br />
`.angle(rhs)`      <span style="float:right;">Smallest angle between two vectors.</span><br />

<br/>

`.is_square()`             <span style="float:right;">Tests if `self` is square.</span><br />
`.is_identity()`           <span style="float:right;">Tests if `self` is the identity matrix.</span><br />
`.is_orthogonal()`         <span style="float:right;">Tests if `self.transpose() * self` is the identity.</span><br />
`.is_special_orthogonal()` <span style="float:right;">Tests if `self.is_orthogonal()` and has a determinant equal to 1.</span><br />
`.is_invertible()`         <span style="float:right;">Tests if `self` is invertible.</span><br />

---------

#### Slicing
Slice are references to sub-matrices. They do not own their data and cannot be
converted to arrays as their data buffer may not be contiguous in memory.
Mutable slices are obtained using the same methods suffixed by `_mut`, e.g.,
`.row_mut(i)`. They can also be constructed from a data array `&[N]` using the
constructors of matrix slice type aliases, e.g., `MatrixSlice3::new(data)`.


`.row(i)`                             <span style="float:right;">A matrix row.</span><br />
`.rows(i, nrows)`                     <span style="float:right;">Several consecutive rows.</span><br />
`.rows_with_step(i, nrows, step)`     <span style="float:right;">Several non-consecutive rows.</span><br />
`.fixed_rows::<D>(i)`                 <span style="float:right;">A compile-time number of consecutive rows.</span><br />
`.fixed_rows_with_step::<D>(i, step)` <span style="float:right;">A compile-time number of non-consecutive rows.</span><br />

<br/>

`.column(j)`                             <span style="float:right;">A matrix column.</span><br />
`.columns(j, ncols)`                     <span style="float:right;">Several consecutive columns.</span><br />
`.columns_with_step(j, ncols, step)`     <span style="float:right;">Several non-consecutive columns.</span><br />
`.fixed_columns::<D>(j)`                 <span style="float:right;">A compile-time number of consecutive columns.</span><br />
`.fixed_columns_with_step::<D>(j, step)` <span style="float:right;">A compile-time number of non-consecutive columns.</span><br />

<br/>

`.slice((i, j), (nrows, ncols))`                           <span style="float:right;">Consecutive rows and columns.</span><br />
`.slice_with_steps((i, j), (nrows, ncols), (rstep, cstep)` <span style="float:right;">Non consecutive rows and columns.</span><br />
`.fixed_slice::<R, C>((i, j))`                             <span style="float:right;">Compile-time number of consecutive rows and columns.</span><br />
`.fixed_slice_with_steps::<R, C>((i, j), (rstep, cstep))`  <span style="float:right;">Compile-time number of non consecutive rows and columns.</span><br />

-----

#### Resizing
The dimension of a matrix can be modified by inserting or removing rows or
columns and by changing its dimensions. The input is always consumed to produce
the output. Inserted rows/columns are filled by a user-provided value.

`.remove_row(i)`             <span style="float:right;">Removes one row.</span><br />
`.remove_rows(i, nrows)`     <span style="float:right;">Removes several consecutive rows.</span><br />
`.remove_fixed_rows::<D>(i)` <span style="float:right;">Removes a compile-time number of consecutive rows.</span><br />

<br/>

`.remove_column(j)`             <span style="float:right;">Removes one column.</span><br />
`.remove_columns(j, ncols)`     <span style="float:right;">Removes several consecutive columns.</span><br />
`.remove_fixed_columns::<D>(j)` <span style="float:right;">Removes a compile-time number of consecutive columns.</span><br />

<br/>

`.insert_row(i, val)`             <span style="float:right;">Adds one row filled with `val`.</span><br />
`.insert_rows(i, nrows, val)`     <span style="float:right;">Adds several consecutive rows filled with `val`.</span><br />
`.insert_fixed_rows::<D>(i, val)` <span style="float:right;">Adds a compile-time number of consecutive rows filled with `val`.</span><br />

<br/>

`.insert_column(j, val)`             <span style="float:right;">Adds one column.</span><br />
`.insert_columns(j, ncols, val)`     <span style="float:right;">Adds several consecutive columns.</span><br />
`.insert_fixed_columns::<D>(j, val)` <span style="float:right;">Adds a compile-time number of consecutive columns.</span><br />

<br/>

`.resize(nrows, ncols, val)` <span style="float:right;">Resizes the output matrix to `nrows` rows and `ncols` columns.</span><br />
`.fixed_resize<R, C>(val)`   <span style="float:right;">Resizes the output matrix to `R` rows and `C` columns.</span><br />

Resizing methods keep the original component values, i.e., `self[(i, j)] ==
output[(i, j)]`. Additional rows and columns are filled with `val`.

-----

#### Blas andoperations
**nalgebra** implements some Blas operations in pure Rust. In the following,
the variables $\mathbf{v}$ and $\mathbf{V}$ designs the `self` argument.

`.iamax()`                 <span style="float:right;">Returns the index of the vector component with the greatest absolute value.</span><br />
`.iamax_full()`            <span style="float:right;">Returns the index of the matrix component with the greatest absolute value.</span><br />
`.dot(x)`                  <span style="float:right;">Computes the scalar product $\left<\mathbf{v}, \mathbf{x}\right>$.</span><br />
`.axpy(alpha, x, beta)`    <span style="float:right;">Computes $\mathbf{v} = \alpha \mathbf{x} + \beta \mathbf{v}$.</span><br />
`.gemv(alpha, A, x, beta)` <span style="float:right;">Computes $V = \alpha A \mathbf{x} + \beta V$ with a matrix and vector $a$ and $\mathbf{x}$.</span><br />
`.ger(alpha, x, y, beta)`  <span style="float:right;">Computes $V = \alpha \mathbf{x}^T \mathbf{y} + \beta V$ where $\mathbf{x}$ and $\mathbf{y}$ are vectors.</span><br />
`.gemm(alpha, A, B, beta)` <span style="float:right;">Computes $V = \alpha A B + \beta V$ where $A$ and $B$ are matrices.</span><br />
`.gemv_symm(...)`          <span style="float:right;">Is the same as `.gemv` except that `self` is assumed symmetric.</span><br />
`.ger_symm(...)`           <span style="float:right;">Is the same as `.ger` except that `self` is assumed symmetric.</span><br />
`.gemv_tr(...)`          <span style="float:right;">Is the same as `.gemv` except that the transpose of `A` is considered.</span><br />

Other operations that work like blas operations (i.e., in-place and real coefficients) are implemented:

`.iamax()`                 <span style="float:right;">Returns the index of the vector component with the smallest absolute value.</span><br />
`.cmpy(alpha, a, b, beta)` <span style="float:right;">Computes the component-wise multiplication: $\mathbf{v}_i = \alpha \mathbf{a}_i * \mathbf{b}_i + \beta \mathbf{v}_i$.</span><br/>
`.cdpy(alpha, a, b, beta)` <span style="float:right;">Computes the component-wise division: $\mathbf{v}_i = \alpha \mathbf{a}_i / \mathbf{b}_i + \beta \mathbf{v}_i$.</span><br/>
`.quadform(alpha, M, B, beta)`    <span style="float:right;">Computes the quadratic form $V = \alpha B^TMB + \beta V$.</span><br />
`.quadform_tr(alpha, A, M, beta)` <span style="float:right;">Computes the quadratic form $V = \alpha AMA^T + \beta V$.</span><br />

-----

#### Decompositions
All matrix decompositions are implemented in Rust and operate on Real matrices
only. Refer to [Lapack integration](#nalgebra-lapack) for lapack-based decompositions.

`.bidiagonalize()`            <span style="float:right;">Bidiagonalization of a general matrix.</span><br />
`.symmetric_tridiagonalize()` <span style="float:right;">Tridiagonalization of a general matrix.</span><br />
`.cholesky()`                 <span style="float:right;">Cholesky factorization of a Symmetric-Definite-Positive square matrix.</span><br />
`.qr()`                       <span style="float:right;">QR decomposition.</span><br />
`.lu()`                       <span style="float:right;">LU decomposition with partial (row) pivoting.</span><br />
`.full_piv_lu()`              <span style="float:right;">LU decomposition with full pivoting.</span><br />
`.hessenberg()`               <span style="float:right;">Hessenberg form computation for a square matrix.</span><br />
`.real_schur()`               <span style="float:right;">Real Schur decomposition of a square matrix.</span><br />
`.symmetric_eigen()`          <span style="float:right;">Eigenvalue and eigenvectors computation of a symmetric matrix.</span><br />
`.svd()`                      <span style="float:right;">Singular Value Decomposition.</span><br />

Iterative methods may take extra parameters to control their convergence: an
error tolenence `eps` (set to machine epsilon by default) and maximum number of
iteration (set to infinite by default):

`.try_real_schur(eps, max_niter)`      <span style="float:right;">Real Schur decomposition of a square matrix.</span><br />
`.try_symmetric_eigen(eps, max_niter)` <span style="float:right;">Eigenvalue and eigenvectors computation of a symmetric matrix.</span><br />
`.try_svd(eps, max_niter)`             <span style="float:right;">Singular Value Decomposition.</span><br />

------

#### Lapack integration 
Lapack-based decompositions are available using the **nalgebra-lapack** crate.
Refer the the [dedicated
section](decompositions_and_lapack/#lapack-integration) for details regarding
its use and the choice of backend (OpenBLAS, netlib, or Accelerate) The
following factorization are implemented:

`Cholesky::new(matrix)`       <span style="float:right;">Cholesky factorization of a Symmetric-Definite-Positive matrix.</span><br />
`QR::new(matrix)`             <span style="float:right;">QR decomposition.</span><br />
`LU::new(matrix)`             <span style="float:right;">LU decomposition with partial (row) pivoting.</span><br />
`Hessenberg::new(matrix)`     <span style="float:right;">Hessenberg form computation.</span><br />
`RealSchur::new(matrix)`      <span style="float:right;">Real Schur decomposition of a square matrix.</span><br />
`Eigen::new(matrix, compute_left, compute_right)` <span style="float:right;">Eigendecomposition of a symmetric matrix.</span><br />
`SymmetricEigen::new(matrix)`                     <span style="float:right;">Eigendecomposition of a symmetric matrix.</span><br />
`SVD::new(matrix)`                                <span style="float:right;">Singular Value Decomposition.</span><br />

------

#### Computer graphics
Those are constructors and methods useful for the computer graphics community
that work with homogeneous matrix coordinates, e.g., 4x4 matrices for 3D
transformations. Homogeneous matrix coordinates are expected to be multiplied
by homogeneous vectors (the vector goes on the right-hand-side of the
operator).

`::new_scaling(factor)`                     <span style="float:right;">An uniform scaling matrix.</span><br />
`::new_nonuniform_scaling(vector)`          <span style="float:right;">A nonuniform scaling matrix.</span><br />
`::new_translation(vector)`                 <span style="float:right;">A translation matrix.</span><br />
`::new_rotation(angle)`                     <span style="float:right;">A 2D rotation matrix from an angle.</span><br />
`::new_rotation(axisangle)`                 <span style="float:right;">A 3D rotation matrix from an axis multiplied by an angle.</span><br />
`::new_rotation_wrt_point(axiangle, point)` <span style="float:right;">An 3D isometry matrix that lets the given point invariant.</span><br />
`::from_scaled_axis(axisangle)`             <span style="float:right;">A 3D rotation matrix from an axis multiplied by an angle.</span><br />
`::from_euler_angles(roll, pitch, yaw)`     <span style="float:right;">A 3D rotation matrix from euler angles (roll → pitch → yaw).</span><br />
`::from_axis_angle(axis, angle)`            <span style="float:right;">A 3D rotation matrix from an axis and an angle.</span><br />
`::new_orthographic(left, right, top, bottom, znear, zfar)` <span style="float:right;">A 3D orthographic projection matrix.</span><br />
`::new_perspective(aspect, fovy, znear, zfar)` <span style="float:right;">A 3D perspective projection matrix.</span><br />
`::new_observer_frame(eye, target, up)` <span style="float:right;">3D local coordinate system of a player looking toward `target`.</span><br />
`::look_at_rh(eye, target, up)`         <span style="float:right;">A 3D right-handed look-at matrix.</span><br />
`::look_at_lh(eye, target, up)`         <span style="float:right;">A 3D left-handed look-at matrix.</span><br />

<br/>

`.append_scaling(factor)`                 <span style="float:right;">Applies an uniform scaling after `self`.</span><br />
`.append_scaling_mut(factor)`             <span style="float:right;">Applies in-place an uniform scaling after `self`.</span><br />
`.prepend_scaling(factor)`                <span style="float:right;">Applies an uniform scaling before`self`.</span><br />
`.prepend_scaling_mut(factor)`            <span style="float:right;">Applies in-place an uniform scaling before `self`.</span><br />
`.append_nonuniform_scaling(vector)`      <span style="float:right;">Applies a non-uniform scaling after `self`.</span><br />
`.append_nonuniform_scaling_mut(vector)`  <span style="float:right;">Applies in-place a non-uniform scaling after `self`.</span><br />
`.prepend_nonuniform_scaling(vector)`     <span style="float:right;">Applies a non-uniform scaling before`self`.</span><br />
`.prepend_nonuniform_scaling_mut(vector)` <span style="float:right;">Applies in-place a non-uniform scaling before `self`.</span><br />
`.append_translation(vector)`             <span style="float:right;">Applies a translation after `self`.</span><br />
`.append_translation_mut(vector)`         <span style="float:right;">Applies in-place a translation after `self`.</span><br />
`.prepend_translation(vector)`            <span style="float:right;">Applies a translation before `self`.</span><br />
`.prepend_translation_mut(vector)`        <span style="float:right;">Applies in-place a translation before `self`.</span><br />

-----

### Geometry

* Most geometric entities are just wrappers for a matrix (or a vector).
* Points have a very different semantic than vectors.
* Transformation types are preferred over raw matrices as they allow
  optimization because of their intrinsic properties.
* Use `.to_homogeneous()` pour obtain the corresponding homogeneous raw matrix
  of any geometric entity.
* Transformations of different types can be multiplied together. The result
  type is automatically inferred.
* Transforming a point or vector can be done by multiplication. The
  transformation goes on the left-hand-side.
* In the following, notations like `Transform2/3<N>` means "either
  `Transform2<N>` or `Transform3<N>`".

`Point2/3<N>`       <span style="float:right;">A location in space.</span><br/>
`Quaternion<N>`     <span style="float:right;">A general quaternion.</span><br/>
`Rotation2/3<N>`    <span style="float:right;">A rotation matrix.</span><br/>
`UnitComplex<N>`    <span style="float:right;">A 2D rotation represented as a unit complex number.</span><br/>
`UnitQuaternion<N>` <span style="float:right;">A 3D rotation represented as a unit quaternion.</span><br/>
`Translation2/3<N>` <span style="float:right;">A translation (stored as a vector).</span><br/>
`Isometry2/3<N>`    <span style="float:right;">A 2D/3D isometry containing an unit complex/quaternion.</span><br/>
`Similarity2/3<N>`  <span style="float:right;">A 2D/3D similarity containing an unit complex/quaternion.</span><br/>
`Affine2/3<N>`      <span style="float:right;">An affine transformation stored as an homogeneous matrix.</span><br/>
`Projective2/3<N>`  <span style="float:right;">An invertible transformation stored as an homogeneous matrix.</span><br/>
`Transform2/3<N>`   <span style="float:right;">A general transformation stored as an homogeneous matrix.</span><br/>

-----

#### Projections

Projections follow the behavior expected by Computer Graphics community, i.e.,
they are invertible transformation from a convex shape to a unit cube centered
at the origin.

`Perspective3<N>`  <span style="float:right;">3D perspective projection matrix using homogeneous coordinates.</span><br/>
`Orthographic3<N>` <span style="float:right;">3D orthographic projection matrix using homogeneous coordinates.</span><br/>

-----

#### Base types
Base types are generic wrt. the dimension and/or the data storage type. They
should not be used directly, prefer type aliases shown in the previous section
instead.

`Point<N>`                      <span style="float:right;">A location in space.</span><br/>
`Rotation<N, Dim>`              <span style="float:right;">A rotation matrix.</span><br/>
`Translation<N, Dim>`           <span style="float:right;">A translation vector.</span><br/>
`Isometry<N, Dim, Rotation>`    <span style="float:right;">An isometry containing an abstract rotation type.</span><br/>
`Similarity<N, Dim, Rotation>`  <span style="float:right;">A similarity containing an abstract rotation type.</span><br/>
`Transform<N, Dim, Affine>`     <span style="float:right;">An affine transformation stored as an homogeneous matrix.</span><br/>
`Transform<N, Dim, Projective>` <span style="float:right;">An invertible transformation stored as an homogeneous matrix.</span><br/>
`Transform<N, Dim, General>`    <span style="float:right;">A general transformation stored as an homogeneous matrix.</span><br/>
`Perspective<N>`                <span style="float:right;">A 3D perspective projection matrix.</span><br/>
`Orthographic<N>`               <span style="float:right;">A 3D orthographic projection matrix.</span><br/>

-----

### Data storage and allocators

* Data storages provide access to a matrix shape and its components.
* The last type parameter of the generic `Matrix<...>` type is the data storage.
* Allocators provide a way to allocate a data storage type that depends on
  whether the matrix shape is statically known. The `Allocator` trait should
  not be implemented manually by the user. Only one implementor exists:
  `DefaultAllocator`.


#### Traits
`Storage<...>`              <span style="float:right;">Implemented by buffers that may store matrix elements non-contiguously.</span><br/>
`StorageMut<...>`           <span style="float:right;">Implemented by mutable buffers that may store matrix elements non-contiguously.</span><br/>
`ContiguousStorage<...>`    <span style="float:right;">Implemented by buffers storing matrix components contiguously.</span><br/>
`ContiguousStorageMut<...>` <span style="float:right;">Implemented by mutable buffers storing matrix components contiguously.</span><br/>


-----

#### Implementors
`MatrixArray<...>`      <span style="float:right;">A stack-allocated owned data storage.</span><br/>
`MatrixVec<...>`        <span style="float:right;">A heap-allocated owned data storage.</span><br/>
`MatrixSlice<...>`      <span style="float:right;">A non-mutable reference to a piece of another data storage.</span><br/>
`MatrixSliceMut<...>`   <span style="float:right;">A mutable reference to a piece of another data storage.</span><br/>
`DefaultAllocator<...>` <span style="float:right;">Allocates `MatrixArray` for statically sized matrices, and `MatrixVec` otherwise.</span><br/>
