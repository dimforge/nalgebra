/*!
# nalgebra

**nalgebra** is a linear algebra library written for Rust targeting:

* general-purpose linear algebra (still misses a lot of features…).
* real time computer graphics.
* real time computer physics.

## Using **nalgebra**
All the functionalities of **nalgebra** are grouped in one place: the `na` module.
This module re-exports everything and includes free functions for all traits methods.
Free functions are useful if you prefer doing something like `na::dot(v1, v2)` instead of
`v1.dot(v2)`.

* You can import the whole prelude, including free functions, using:

```.rust
use nalgebra::na::*;
```

* If you dont want to import everything but only every trait:

```.rust
use nalgebra::traits::*;
```

* If you dont want to import everything but only every structure:

```.rust
use nalgebra::structs::*;
```
Of course, you can still import `nalgebra::na` alone, and get anything you want using the prefix
`na`.

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
* Almost one trait per functionality: useful for generic programming.
* Operator overloading using the double trait dispatch
  [trick](http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/).
  For example, the following works:

```rust
extern mod nalgebra;
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

```.rust
make doc
```

## **nalgebra** in use
Here are some projects using **nalgebra**.
Feel free to add your project to this list if you happen to use **nalgebra**!

* [nphysics](https://github.com/sebcrozet/nphysics): a real-time physics engine.
* [ncollide](https://github.com/sebcrozet/ncollide): a collision detection library.
* [kiss3d](https://github.com/sebcrozet/kiss3d): a minimalistic graphics engine.
* [frog](https://github.com/natal/frog): a machine learning library.

*/
#[link(name = "nalgebra"
       , vers = "0.1"
       , author = "Sébastien Crozet"
       , uuid = "1e96070f-4778-4ec1-b080-bf69f7048216")];
#[crate_type = "lib"];
#[deny(non_camel_case_types)];
#[deny(non_uppercase_statics)];
#[deny(unnecessary_qualification)];
#[deny(missing_doc)];
#[feature(macro_rules)];

extern mod std;
extern mod extra;

pub mod na;
pub mod structs;
pub mod traits;

// mod lower_triangular;
// mod chol;

#[cfg(test)]
mod tests {
    mod vec;
    mod mat;
}

#[cfg(test)]
mod bench {
    mod vec;
    mod mat;
}
