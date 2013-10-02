nalgebra
========

**nalgebra** is a _n_-dimensional linear algebra library written with the rust
programming language.

## Features
**nalgebra** is meant to be a general-purpose linear algebra library (but is very far from that…),
and keeps an optimized set of tools for computational graphics and physics. Those features include:

* Vectors with static sizes: `Vec0`, `Vec1`, `Vec2`, ..., `Vec6`.
* Square matrices with static sizes: `Mat1`, `Mat2`, ..., `Mat6 `.
* Dynamically sized vector: `DVec`.
* Dynamically sized (square or rectangular) matrix: `DMat`.
* Geometry-specific matrix wrapper to ensure at compile-time some properties: `Rotmat`, `Transform`.
* Most well-known geometric functions.
* A few methods for data analysis: `Cov` (covariance), `Mean` (mean).
* Operator overloading using the double trait dispatch [trick](http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/).
This allows using operators for both matrix/matrix multiplication and matrix/vector
multiplication for example.
* Almost one trait per functionality. This is very useful for generic programming.

Since there is almost one trait per functionality, one might end up importing a lot of traits. To
lighten your `use` prelude, all trait are re-exported by the `nalgebra::vec` and `nalgebra::mat`
modules. Thus, to bring every functionalities of `nalgebra` in scope, you can do:

```rust
use nalgebra::vec::*;
use nalgebra::mat::*;
```

## Compilation
You will need the last rust compiler from the master branch.
If you encounter problems, make sure you have the last version before creating an issue.

    git clone git://github.com/sebcrozet/nalgebra.git
    cd nalgebra
    make

There is also a light, but existing, documentation for most functionalities. Use `make doc` to
generate it on the `doc` folder.

## nalgebra in use
Feel free to add your project to this list if you happen to use **nalgebra**!

* [nphysics](https://github.com/sebcrozet/nphysics): a real-time physics engine.
* [ncollide](https://github.com/sebcrozet/ncollide): a collision detection library.
* [kiss3d](https://github.com/sebcrozet/kiss3d): a minimalistic graphics engine.
* [frog](https://github.com/natal/frog): a machine learning library.

## Design note

**nalgebra** is mostly written with non-idiomatic rust code. This is mostly because of limitations
of the trait system not allowing (easy) multiple overloading. Those overloading problems ares
worked around by this
[hack](http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/)
(section _What if I want overloading_).
