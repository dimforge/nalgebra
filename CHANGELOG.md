# Change Log
All notable changes to `nalgebra`, starting with the version 0.6.0 will be
documented here.

This project adheres to [Semantic Versioning](http://semver.org/).

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
