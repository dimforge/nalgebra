# Change Log
All notable changes to `nalgebra`, starting with the version 0.6.0 will be
documented here.

This project adheres to [Semantic Versioning](http://semver.org/).

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
