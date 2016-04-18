[![Build Status](https://travis-ci.org/sebcrozet/nalgebra.svg?branch=master)](https://travis-ci.org/sebcrozet/nalgebra)

nalgebra
========

**nalgebra** is a low-dimensional linear algebra library written for Rust targeting:

* General-purpose linear algebra (still lacks a lot of features…)
* Real time computer graphics.
* Real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org/doc/nalgebra).

## Using **nalgebra**
All the functionality of **nalgebra** is grouped in one place: the root module `nalgebra::`.  This
module re-exports everything and includes free functions for all traits methods performing
out-of-place operations.

Thus, you can import the whole prelude using:

```.ignore
use nalgebra::*;
```

However, the recommended way to use **nalgebra** is to import types and traits
explicitly, and call free-functions using the `na::` prefix:

```.rust
extern crate nalgebra as na;
use na::{Vector3, Rotation3, Rotation};

fn main() {
    let     a = Vector3::new(1.0f64, 1.0, 1.0);
    let mut b = Rotation3::new(na::zero());

    b.append_rotation_mut(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```


## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* Vectors with predefined static sizes: `Vector1`, `Vector2`, `Vector3`, `Vector4`, `Vector5`, `Vector6`.
* Vector with a user-defined static size: `VectorN` (available only with the `generic_sizes` feature).
* Points with static sizes: `Point1`, `Point2`, `Point3`, `Point4`, `Point5`, `Point6`.
* Square matrices with static sizes: `Matrix1`, `Matrix2`, `Matrix3`, `Matrix4`, `Matrix5`, `Matrix6 `.
* Rotation matrices: `Rotation2`, `Rotation3`
* Quaternions: `Quaternion`, `UnitQuaternion`.
* Isometries (translation ⨯ rotation): `Isometry2`, `Isometry3`
* Similarity transformations (translation ⨯ rotation ⨯ uniform scale): `Similarity2`, `Similarity3`.
* 3D projections for computer graphics: `Persp3`, `PerspMatrix3`, `Ortho3`, `OrthoMatrix3`.
* Dynamically sized heap-allocated vector: `DVector`.
* Dynamically sized stack-allocated vectors with a maximum size: `DVector1` to `DVector6`.
* Dynamically sized heap-allocated (square or rectangular) matrix: `DMatrix`.
* Linear algebra and data analysis operators: `Covariance`, `Mean`, `qr`, `cholesky`.
* Almost one trait per functionality: useful for generic programming.
