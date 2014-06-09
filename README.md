# nalgebra

**nalgebra** is a linear algebra library written for Rust targeting:

* general-purpose linear algebra (still misses a lot of features…).
* real time computer graphics.
* real time computer physics.

An on-line version of this documentation is available [here](http://nalgebra.org).

## Using **nalgebra**
All the functionalities of **nalgebra** are grouped in one place: the `na` module.
This module re-exports everything and includes free functions for all traits methods doing
out-of-place modifications.

* You can import the whole prelude using:

```
use nalgebra::na::*;
```

The preferred way to use **nalgebra** is to import types and traits explicitly, and call
free-functions using the `na::` prefix:

```.rust
extern crate nalgebra;
use nalgebra::na::{Vec3, Rot3, Rotation};
use nalgebra::na;

fn main() {
    let     a = Vec3::new(1.0f64, 1.0, 1.0);
    let mut b = Rot3::new(na::zero());

    b.append_rotation(&a);

    assert!(na::approx_eq(&na::rotation(&b), &a));
}
```

## Features
**nalgebra** is meant to be a general-purpose linear algebra library (but is very far from that…),
and keeps an optimized set of tools for computational graphics and physics. Those features include:

* Vectors with static sizes: `Vec0`, `Vec1`, `Vec2`, `Vec3`, `Vec4`, `Vec5`, `Vec6`.
* Square matrices with static sizes: `Mat1`, `Mat2`, `Mat3`, `Mat4`, `Mat5`, `Mat6 `.
* Rotation matrices: `Rot2`, `Rot3`, `Rot4`.
* Isometries: `Iso2`, `Iso3`, `Iso4`.
* Dynamically sized vector: `DVec`.
* Dynamically sized (square or rectangular) matrix: `DMat`.
* A few methods for data analysis: `Cov`, `Mean`.
* Some matrix factorization algorithms: QR decomposition, ...
* Almost one trait per functionality: useful for generic programming.
* Operator overloading using the double trait dispatch
  [trick](http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/).
  For example, the following works:

```rust
extern crate nalgebra;
use nalgebra::na::{Vec3, Mat3};
use nalgebra::na;

fn main() {
    let v: Vec3<f64> = na::zero();
    let m: Mat3<f64> = na::one();

    let _ = m * v;   // matrix-vector multiplication.
    let _ = v * m;   // vector-matrix multiplication.
    let _ = m * m;   // matrix-matrix multiplication.
    let _ = v * 2.0; // vector-scalar multiplication.
}
```

## Compilation
You will need the last rust compiler from the master branch.
If you encounter problems, make sure you have the last version before creating an issue.

    git clone git://github.com/sebcrozet/nalgebra.git
    cd nalgebra
    make

You can build the documentation on the `doc` folder using:

```
make doc
```

## **nalgebra** in use
Here are some projects using **nalgebra**.
Feel free to add your project to this list if you happen to use **nalgebra**!

* [nphysics](http://nphysics-dev.org): a real-time physics engine.
* [ncollide](http://ncollide.org): a collision detection library.
* [kiss3d](https://kiss3d.org): a minimalistic graphics engine.
