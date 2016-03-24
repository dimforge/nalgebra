[![Build Status](https://travis-ci.org/sebcrozet/nalgebra.svg?branch=master)](https://travis-ci.org/sebcrozet/nalgebra)

nalgebra
========

**nalgebra** is a low-dimensional linear algebra library written for Rust targeting:

* general-purpose linear algebra (still lacks a lot of features…).
* real time computer graphics.
* real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org/doc/nalgebra).

## Using **nalgebra**
All the functionality of **nalgebra** is grouped in one place: the root module `nalgebra::`.  This
module re-exports everything and includes free functions for all traits methods performing
out-of-place operations.

* You can import the whole prelude using:

```.ignore
use nalgebra::*;
```

The preferred way to use **nalgebra** is to import types and traits explicitly, and call
free-functions using the `na::` prefix:

```.rust
extern crate nalgebra as na;
use na::{Vec3, Rot3, Rotation};

fn main() {
    let     a = Vec3::new(1.0f64, 1.0, 1.0);
    let mut b = Rot3::new(na::zero());

    b.append_rotation_mut(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```


## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* Vectors with predefined static sizes: `Vec1`, `Vec2`, `Vec3`, `Vec4`, `Vec5`, `Vec6`.
* Vector with a user-defined static size: `VecN`.
* Points with static sizes: `Pnt1`, `Pnt2`, `Pnt3`, `Pnt4`, `Pnt5`, `Pnt6`.
* Square matrices with static sizes: `Mat1`, `Mat2`, `Mat3`, `Mat4`, `Mat5`, `Mat6 `.
* Rotation matrices: `Rot2`, `Rot3`
* Quaternions: `Quat`, `UnitQuat`.
* Isometries (translation ⨯ rotation): `Iso2`, `Iso3`
* Similarity transformations (translation ⨯ rotation ⨯ uniform scale): `Sim2`, `Sim3`.
* 3D projections for computer graphics: `Persp3`, `PerspMat3`, `Ortho3`, `OrthoMat3`.
* Dynamically sized heap-allocated vector: `DVec`.
* Dynamically sized stack-allocated vectors with a maximum size: `DVec1` to `DVec6`.
* Dynamically sized heap-allocated (square or rectangular) matrix: `DMat`.
* Linear algebra and data analysis operators: `Cov`, `Mean`, `qr`, `cholesky`.
* Almost one trait per functionality: useful for generic programming.
