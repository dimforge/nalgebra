# Change Log
All notable changes to `nalgebra`, starting with the version 0.6.0 will be
documented here.

This project adheres to [Semantic Versioning](http://semver.org/).

## [0.13.0] - WIP
### Added
  * `.kronecker(a, b)` computes the kronecker product (i.e. matrix tensor
    product) of two matrices.
  * `.set_row(i, row)` sets the i-th row of the matrix.
  * `.set_column(j, column)` sets the i-th column of the matrix.


## [0.12.0]
The main change of this release is the update of the dependency serde to 1.0.

### Added
 * `.trace()` that computes the trace of a matrix (i.e., the sum of its
   diagonal elements.)

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
  * Added implementation of assignement operators (+=, -=, etc.) for
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
