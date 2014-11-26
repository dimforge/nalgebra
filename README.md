# nalgebra

**nalgebra** is a low-dimensional linear algebra library written for Rust targeting:

* general-purpose linear algebra (still lacks a lot of features…).
* real time computer graphics.
* real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org).

## Using **nalgebra**
All the functionality of **nalgebra** is grouped in one place: the root module `nalgebra::`.
This module re-exports everything and includes free functions for all traits methods doing
out-of-place modifications.

* You can import the whole prelude using:

```.ignore
use nalgebra::*;
```

The preferred way to use **nalgebra** is to import types and traits explicitly, and call
free-functions using the `na::` prefix:

```.rust
extern crate "nalgebra" as na;
use na::{Vec3, Rot3, Rotation};

fn main() {
    let     a = Vec3::new(1.0f64, 1.0, 1.0);
    let mut b = Rot3::new(na::zero());

    b.append_rotation(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```

## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* Vectors with static sizes: `Vec0`, `Vec1`, `Vec2`, `Vec3`, `Vec4`, `Vec5`, `Vec6`.
* Points with static sizes: `Pnt0`, `Pnt1`, `Pnt2`, `Pnt3`, `Pnt4`, `Pnt5`, `Pnt6`.
* Square matrices with static sizes: `Mat1`, `Mat2`, `Mat3`, `Mat4`, `Mat5`, `Mat6 `.
* Rotation matrices: `Rot2`, `Rot3`, `Rot4`.
* Quaternions: `Quat`, `UnitQuat`.
* Isometries: `Iso2`, `Iso3`, `Iso4`.
* 3D projections for computer graphics: `Persp3`, `PerspMat3`, `Ortho3`, `OrthoMat3`.
* Dynamically sized vector: `DVec`.
* Dynamically sized (square or rectangular) matrix: `DMat`.
* A few methods for data analysis: `Cov`, `Mean`.
* Almost one trait per functionality: useful for generic programming.
* Operator overloading using multidispatch.
