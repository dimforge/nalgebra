# Change Log
All notable changes to `nalgebra`, starting with the version 0.6.0 will be
documented here.

This project adheres to [Semantic Versioning](http://semver.org/).

## [0.17.0] - WIP

### Added
  * Add swizzling up to dimension 3. For example, you can do `v.zxy()` as an equivalent to `Vector3::new(v.z, v.x, v.w)`.
  * Add `.copy_from_slice` to copy matrix components from a slice in column-major order.
  * Add `.dot` to quaternions.
  * Add `.zip_zip_map` for iterating on three matrices simultaneously, and applying a closure to them.
  * Add `.slerp` and `.try_slerp` to unit vectors.
  * Add `.to_projective` and `.as_projective` to `Perspective3` and `Orthographic3` in order to
  use them as `Projective3` structures.

## [0.16.0]
All dependencies have been updated to their latest versions.

## Modified
  * Adjust `UnitQuaternion`s, `Rotation3`s, and `Rotation2`s generated from the `Standard` distribution to be uniformly distributed.
### Added
  * Add a feature `stdweb` to activate the dependency feature `rand/stdweb`.
  * Add blas-like methods `.imin()` and `.imax()` that return the index of the minimum and maximum entry of a vector.
  * Add construction of a `Point` from an array by implementing the `From` trait.
  * Add support for generating uniformly distributed random unit column vectors using the `Standard` distribution.

## [0.15.0]
The most notable change of this release is the support for using part of the library without the rust standard
library (i.e. it supports `#![no_std]`). See the corresponding [documentation](http://nalgebra.org/wasm_and_embedded_programming/).
### Modified
  * Rename the `core` module to `base` to avoid conflicts with the `core` crate implicitly imported when
    `#![no_std]` is enabled.
  * Constructors of the `MatrixSlice*` types have been renamed from `new_*` to `from_slice_*`. This was
    necessary to avoid the `incoherent_fundamental_impls` lint that is going to become a hard error.
### Added
  * Add `UnitQuaternion` constructor `::new_eps(...)` and `::from_scaled_axis_eps(...)` that return the
    identity if the magnitude of the input axisangle is smaller than the epsilon provided.
  * Add methods `.rotation_between_axis(...)` and `.scaled_rotation_between_axis(...)` to `UnitComplex`
    to compute the rotation matrix between two 2D **unit** vectors.
  * Add methods `.axis_angle()` to `UnitComplex` and `UnitQuaternion` in order to retrieve both the
    unit rotation axis and the rotation angle simultaneously.
  * Add functions to construct a random matrix with a user-defined distribution: `::from_distribution(...)`.

## [0.14.0]
### Modified
  * Allow the `Isometry * Unit<Vector>` multiplication.
### Added
  * Add blas-like operations: `.quadform(...)` and `.quadform_tr(...)` to compute respectively
    the quadratic forms `self = alpha * A.transpose() * B * A + beta * self` and
    `alpha * A * B * A.transpose() + beta * self`. Here, `A, B` are matrices with
    `B` square, and `alpha, beta` are reals.
  * Add blas-like operations: `.gemv_tr(...)` that behaves like `.gemv` except that the
    provided matrix is assumed to be transposed.
  * Add blas-like operations: `cmpy, cdpy` for component-wise multiplications and
    division with scalar factors:
        - `self <- alpha * self + beta * a * b`
        - `self <- alpha * self + beta * a / b`
  * `.cross_matrix()` returns the cross-product matrix of a given 3D vector, i.e.,
    the matrix `M` such that for all vector `v` we have
    `M * v == self.cross(&v)`.
  * `.iamin()` that returns the index of the vector entry with
    smallest absolute value.
  * The `mint` feature that can be enabled in order to allow conversions from
    and to types of the [mint](https://crates.io/crates/mint) crate.
  * Aliases for matrix and vector slices. Their are named by adding `Slice`
    before the dimension numbers, i.e., a 3x5 matrix slice with dimensions known
    at compile-time is called `MatrixSlice3x5`. A vector slice with dimensions
    unknown at compile-time is called `DVectorSlice`.
  * Add functions for constructing matrix slices from a slice `&[N]`, e.g.,
    `MatrixSlice2::new(...)` and `MatrixSlice2::new_with_strides(...)`.
  * The `::repeat(...)` constructor that is an alternative name to
    `::from_element(...)`.
  * `UnitQuaternion::scaled_rotation_between_axis(...)` and
    `UnitQuaternion::rotation_between_axis(...)` that take Unit vectors instead of
    Vector as arguments.



## [0.13.0]

The **nalgebra-lapack** crate has been updated. This now includes a broad range
matrix decompositions using LAPACK bindings.

This adds support for serialization using the
[abomonation](https://crates.io/crates/abomonation) crate.

### Breaking semantic change
  * The implementation of slicing with steps now matches the documentation.
    Before, step identified the number to add to pass from one column/row index
    to the next one. This made 0 step invalid. Now (and on the documentation so
    far), the step is the number of ignored row/columns between each
    row/column. Thus, a step of 0 means that no row/column is ignored.  For
    example, a step of, say, 3 on previous versions should now bet set to 2.

### Modified
  * The trait `Axpy` has been replaced by a metod `.axpy`.
  * The alias `MatrixNM` is now deprecated. Use `MatrixMN` instead (we
    reordered M and N to be in alphabetical order).
  * In-place componentwise multiplication and division
    `.component_mul_mut(...)` and `.component_div_mut(...)` have bee deprecated
    for a future renaming. Use `.component_mul_assign(...)` and
    `.component_div_assign(...)` instead.

### Added
  * `alga::general::Real` is now re-exported by nalgebra.
    elements.)
  * `::zeros(...)` that creates a matrix filled with zeroes.
  * `::from_partial_diagonal(...)` that creates a matrix from diagonal elements.
    The matrix can be rectangular. If not enough elements are provided, the rest
    of the diagonal is set to 0.
  * `.conjugate_transpose()` computes the transposed conjugate of a
    complex matrix.
  * `.conjugate_transpose_to(...)` computes the transposed conjugate of a
    complex matrix. The result written into a user-provided matrix.
  * `.transpose_to(...)` is the same as `.transpose()` but stores the result in
    the provided matrix.
  * `.conjugate_transpose_to(...)` is the same as `.conjugate_transpose()` but
    stores the result in the provided matrix.
  * Implements `IntoIterator` for `&Matrix`, `&mut Matrix` and `Matrix`.
  * `.mul_to(...)` multiplies two matrices and stores the result to the given buffer.
  * `.tr_mul_to(...)` left-multiplies `self.transpose()` to another matrix and stores the result to the given buffer.
  * `.add_scalar(...)` that adds a scalar to each component of a matrix.
  * `.add_scalar_mut(...)` that adds in-place a scalar to each component of a matrix.
  * `.kronecker(a, b)` computes the kronecker product (i.e. matrix tensor
    product) of two matrices.
  * `.apply(f)` replaces each component of a matrix with the results of the
    closure `f` called on each of them.

Pure Rust implementation of some Blas operations:

  * `.iamax()` returns the index of the maximum value of a vector.
  * `.axpy(...)` computes `self = a * x + b * self`.
  * `.gemv(...)` computes `self = alpha * a * x + beta * self` with a matrix and vector `a` and `x`.
  * `.ger(...)` computes `self = alpha * x^t * y + beta * self` where `x` and `y` are vectors.
  * `.gemm(...)` computes `self = alpha * a * b + beta * self` where `a` and `b` are matrices.
  * `.gemv_symm(...)` is the same as `.gemv` except that `self` is assumed symmetric.
  * `.ger_symm(...)` is the same as `.ger` except that `self` is assumed symmetric.

New slicing methods:
  * `.rows_range(...)` that retrieves a reference to a range of rows.
  * `.rows_range_mut(...)` that retrieves a mutable reference to a range of rows.
  * `.columns_range(...)` that retrieves a reference to a range of columns.
  * `.columns_range_mut(...)` that retrieves a mutable reference to a range of columns.

Matrix decompositions implemented in pure Rust:
  * Cholesky, SVD, LU, QR, Hessenberg, Schur, Symmetric eigendecompositions,
    Bidiagonal, Symmetric tridiagonal
  * Computation of householder reflectors and givens rotations.

Matrix edition:
  * `.upper_triangle()` extracts the upper triangle of a matrix, including the diagonal.
  * `.lower_triangle()` extracts the lower triangle of a matrix, including the diagonal.
  * `.fill(...)` fills the matrix with a single value.
  * `.fill_with_identity(...)` fills the matrix with the identity.
  * `.fill_diagonal(...)` fills the matrix diagonal with a single value.
  * `.fill_row(...)` fills a selected matrix row with a single value.
  * `.fill_column(...)` fills a selected matrix column with a single value.
  * `.set_diagonal(...)` sets the matrix diagonal.
  * `.set_row(...)` sets a selected row.
  * `.set_column(...)` sets a selected column.
  * `.fill_lower_triangle(...)` fills some sub-diagonals bellow the main diagonal with a value.
  * `.fill_upper_triangle(...)` fills some sub-diagonals above the main diagonal with a value.
  * `.swap_rows(...)` swaps two rows.
  * `.swap_columns(...)` swaps two columns.

Column removal:
  * `.remove_column(...)` removes one column.
  * `.remove_fixed_columns<D>(...)` removes `D` columns.
  * `.remove_columns(...)` removes a number of columns known at run-time.

Row removal:
  * `.remove_row(...)` removes one row.
  * `.remove_fixed_rows<D>(...)` removes `D` rows.
  * `.remove_rows(...)` removes a number of rows known at run-time.

Column insertion:
  * `.insert_column(...)` adds one column at the given position.
  * `.insert_fixed_columns<D>(...)` adds `D` columns at the given position.
  * `.insert_columns(...)` adds at the given position a number of columns known at run-time.

Row insertion:
  * `.insert_row(...)` adds one row at the given position.
  * `.insert_fixed_rows<D>(...)` adds `D` rows at the given position.
  * `.insert_rows(...)` adds at the given position a number of rows known at run-time.

## [0.12.0]
The main change of this release is the update of the dependency serde to 1.0.

### Added
 * `.trace()` that computes the trace of a matrix (the sum of its diagonal
   elements.)

## [0.11.0]
The [website](http://nalgebra.org) has been fully rewritten and gives a good
overview of all the added/modified features.

This version is a major rewrite of the library. Major changes are:
  * Algebraic traits are now defined by the [alga](https://crates.io/crates/alga) crate.
  All other mathematical traits, except `Axpy` have been removed from
  **nalgebra**.
  * Methods are now preferred to free functions because they do not require any
    trait to be used any more.
  * Most algebraic entities can be parametrized by type-level integers
    to specify their dimensions. Using `Dynamic` instead of a type-level
    integer indicates that the dimension known at run-time only.
  * Statically-sized **rectangular** matrices.
  * More transformation types have been added: unit-sized complex numbers (for
    2D rotations), affine/projective/general transformations with `Affine2/3`,
    `Projective2/3`, and `Transform2/3`.
  * Serde serialization is now supported instead of `rustc_serialize`. Enable
    it with the `serde-serialize` feature.
  * Matrix **slices** are now implemented.

### Added
Lots of features including rectangular matrices, slices, and Serde
serialization. Refer to the brand new [website](http://nalgebra.org) for more
details. The following free-functions have been added as well:
  * `::id()` that returns the universal [identity element](http://nalgebra.org/performance_tricks/#the-id-type)
    of type `Id`.
  * `::inf_sup()` that returns both the infimum and supremum of a value at the
    same time.
  * `::partial_sort2()` that attempts to sort two values in increasing order.
  * `::wrap()` that moves a value to the given interval by adding or removing
    the interval width to it.

### Modified
  * `::cast`            -> `::convert`
  * `point.as_vector()` -> `point.coords`
  * `na::origin`        -> `P::origin()`
  * `na::is_zero`       -> `.is_zero()` (from num::Zero)
  * `.transform`        -> `.transform_point`/`.transform_vector`
  * `.translate`        -> `.translate_point`
  * `::dimension::<P>`  -> `::dimension::<P::Vector>`
  * `::angle_between`   -> `::angle`

Componentwise multiplication and division has been replaced by methods:
  * multiplication -> `.componentwise_mul`, `.componentwise_mul_mut`.
  * division       -> `.componentwise_div`, `.componentwise_div_mut`.

The following free-functions are now replaced by methods (with the same names)
only:
`::cross`, `::cholesky`, `::determinant`, `::diagonal`, `::eigen_qr` (becomes
`.eig`), `::hessenberg`, `::qr`, `::to_homogeneous`, `::to_rotation_matrix`,
`::transpose`, `::shape`.


The following free-functions are now replaced by static methods only:
  * `::householder_matrix` under the name `::new_householder_generic`
  * `::identity`
  * `::new_identity` under the name `::identity`
  * `::from_homogeneous`
  * `::repeat` under the name `::from_element`

The following free-function are now replaced methods accessible through traits
only:
  * `::transform` -> methods `.transform_point` and `.transform_vector` of the `alga::linear::Transformation` trait.
  * `::inverse_transform` -> methods `.inverse_transform_point` and
    `.inverse_transform_vector` of the `alga::linear::ProjectiveTransformation`
    trait.
  * `::translate`, `::inverse_translate`, `::rotate`, `::inverse_rotate` ->
    methods from the `alga::linear::Similarity` trait instead. Those have the
    same names but end with `_point` or `_vector`, e.g., `.translate_point` and
    `.translate_vector`.
  * `::orthonormal_subspace_basis` -> method with the same name from
    `alga::linear::FiniteDimInnerSpace`.
  * `::canonical_basis_element` and `::canonical_basis` -> methods with the
    same names from `alga::linear::FiniteDimVectorSpace`.
  * `::rotation_between` -> method with the same name from the
    `alga::linear::Rotation` trait.
  * `::is_zero` -> method with the same name from `num::Zero`.



### Removed
  * The free functions `::prepend_rotation`, `::append_rotation`,
    `::append_rotation_wrt_center`, `::append_rotation_wrt_point`,
    `::append_transformation`, and `::append_translation ` have been removed.
    Instead create the rotation or translation object explicitly and use
    multiplication to compose it with anything else.

  * The free function `::outer` has been removed. Use column-vector ×
    row-vector multiplication instead.

  * `::approx_eq`, `::approx_eq_eps` have been removed. Use the `relative_eq!`
    macro from the [approx](https://crates.io/crates/approx) crate instead.

  * `::covariance` has been removed. There is no replacement for now.
  * `::mean` has been removed. There is no replacement for now.
  * `::sample_sphere` has been removed. There is no replacement for now.
  * `::cross_matrix` has been removed. There is no replacement for now.
  * `::absolute_rotate` has been removed. There is no replacement for now.
  * `::rotation`, `::transformation`, `::translation`, `::inverse_rotation`,
    `::inverse_transformation`, `::inverse_translation` have been removed. Use
    the appropriate methods/field of each transformation type, e.g.,
    `rotation.angle()` and `rotation.axis()`.

## [0.10.0]
### Added
Binary operations are now allowed between references as well. For example
`Vector3<f32> + &Vector3<f32>` is now possible.

### Modified
Removed unused parameters to methods from the `ApproxEq` trait. Those were
required before rust 1.0 to help type inference. The are not needed any more
since it now allowed to write for a type `T` that implements `ApproxEq`:
`<T as ApproxEq>::approx_epsilon()`. This replaces the old form:
`ApproxEq::approx_epsilon(None::<T>)`.

## [0.9.0]
### Modified
  * Renamed:
    - `::from_col_vector` -> `::from_column_vector`
    - `::from_col_iter` -> `::from_column_iter`
    - `.col_slice` -> `.column_slice`
    - `.set_col` -> `.set_column`
    - `::canonical_basis_with_dim` -> `::canonical_basis_with_dimension`
    - `::from_elem` -> `::from_element`
    - `DiagMut` -> `DiagonalMut`
    - `UnitQuaternion::new` becomes `UnitQuaternion::from_scaled_axis` or
      `UnitQuaternion::from_axisangle`. The new `::new` method now requires a
      not-normalized quaternion.

Methods names starting with `new_with_` now start with `from_`. This is more
idiomatic in Rust.

The `Norm` trait now uses an associated type instead of a type parameter.
Other similar trait changes are to be expected in the future, e.g., for the
`Diagonal` trait.

Methods marked `unsafe` for reasons unrelated to memory safety are no
longer unsafe. Instead, their name end with `_unchecked`. In particular:
* `Rotation3::new_with_matrix` -> `Rotation3::from_matrix_unchecked`
* `PerspectiveMatrix3::new_with_matrix` -> `PerspectiveMatrix3::from_matrix_unchecked`
* `OrthographicMatrix3::new_with_matrix` -> `OrthographicMatrix3::from_matrix_unchecked`

### Added
- A `Unit<T>` type that wraps normalized values. In particular,
  `UnitQuaternion<N>` is now an alias for `Unit<Quaternion<N>>`.
- `.ln()`, `.exp()` and `.powf(..)` for quaternions and unit quaternions.
- `::from_parts(...)` to build a quaternion from its scalar and vector
  parts.
- The `Norm` trait now has a `try_normalize()` that returns `None` if the
norm is too small.
- The `BaseFloat` and `FloatVector` traits now inherit from `ApproxEq` as
  well. It is clear that performing computations with floats requires
  approximate equality.

Still WIP: add implementations of abstract algebra traits from the `algebra`
crate for vectors, rotations and points. To enable them, activate the
`abstract_algebra` feature.

## [0.8.0]
### Modified
  * Almost everything (types, methods, and traits) now use full names instead
    of abbreviations (e.g. `Vec3` becomes `Vector3`). Most changes are abvious.
    Note however that:
    - `::sqnorm` becomes `::norm_squared`.
    - `::sqdist` becomes `::distance_squared`.
    - `::abs`, `::min`, etc. did not change as this is a common name for
      absolute values on, e.g., the libc.
    - Dynamically sized structures keep the `D` prefix, e.g., `DMat` becomes
      `DMatrix`.
  * All files with abbreviated names have been renamed to their full version,
    e.g., `vec.rs` becomes `vector.rs`.

## [0.7.0]
### Added
  * Added implementation of assignment operators (+=, -=, etc.) for
    everything.
### Modified
  * Points and vectors are now linked to each other with associated types
    (on the PointAsVector trait).


## [0.6.0]
**Announcement:** a users forum has been created for `nalgebra`, `ncollide`, and `nphysics`. See
you [there](http://users.nphysics.org)!

### Added
  * Added a dependency to [generic-array](https://crates.io/crates/generic-array). Feature-gated:
    requires `features="generic_sizes"`.
  * Added statically sized vectors with user-defined sizes: `VectorN`. Feature-gated: requires
    `features="generic_sizes"`.
  * Added similarity transformations (an uniform scale followed by a rotation followed by a
    translation): `Similarity2`, `Similarity3`.

### Removed
  * Removed zero-sized elements `Vector0`, `Point0`.
  * Removed 4-dimensional transformations `Rotation4` and `Isometry4` (which had an implementation to incomplete to be useful).

### Modified
  * Vectors are now multipliable with isometries. This will result into a pure rotation (this is how
  vectors differ from point semantically: they design directions so they are not translatable).
  * `{Isometry3, Rotation3}::look_at` reimplemented and renamed to `::look_at_rh` and `::look_at_lh` to agree
  with the computer graphics community (in particular, the GLM library). Use the `::look_at_rh`
  variant to build a view matrix that
  may be successfully used with `Persp` and `Ortho`.
  * The old `{Isometry3, Rotation3}::look_at` implementations are now called `::new_observer_frame`.
  * Rename every `fov` on `Persp` to `fovy`.
  * Fixed the perspective and orthographic projection matrices.
