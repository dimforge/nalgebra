# Frequently Asked Questions

--------

#### What does the **n** of **nalgebra** stands for?

**nalgebra** is some kind of abbreviation for _n-dimensional linear algebra
library_, where `n` is a finite strictly positive integer.

--------

#### Trivial math functions like `dot` and `add` show up on my benchmarks!

Please make sure:

1. You compiled your project and its dependencies in release mode, i.e., `cargo
   build --release`.
2. You did **not** enable [incremental
   compilation](https://internals.rust-lang.org/t/incremental-compilation-beta).
   This prevents some optimization that may cause a 30x slow-down! We observed
   this running the [nphysics](http://nphysics.org) examples in particular.

If you did all that and small `nalgebra` methods still show un in your
benchmarks, please fill an
[issue](https://github.com/sebcrozet/nalgebra/issues) or tell us about it on
the [users forum](http://users.nphysics.org/c/nalgebra).

--------

#### What is the memory layout of matrices?

All matrices of **nalgebra** are stored in column-major order. This means that
any two consecutive elements of a single matrix column will be contiguous in
memory as well. For example the matrix:

```rust
let _ = Matrix3::new(11, 12, 13,
                     21, 22, 23,
                     31, 32, 33);
```

is arranged in memory the same way as the array `[ 11, 21, 31, 12, 22, 32, 13,
23, 33 ]`.

--------

#### Some error messages are very hard to understand or are cryptic!

Because **nalgebra** relies on a lot of generics, some error messages might be
hard to understand. Please, open an
[issue](https://github.com/sebcrozet/nalgebra/issues) or create a post on the
[users forum](http://users.nphysics.org/c/nalgebra) to get help. We take the
quality of error messages seriously, so providing details about how you got
them can be useful to improve them in the future.

--------

#### How do I convert a `Vector3<f32>` to a `Vector3<f64>`?

Use the `::convert(...)` function. For example:

```rust
let a = Vector3::new(10.0f32, 0.0, 1.0);
let b: Vector3<f64> = na::convert(a);
```

Conversions with this function will work for most structures as long as it
preserves the fundamental algebraic properties objects. If some algebraic
properties may be lost during conversion, use `::try_convert(...)` or
`::try_convert_unchecked(...)` instead.

--------

#### Can I serialize/deserialize structures from **nalgebra** ?

Yes, serialization and deserialization are supported using
[serde](https://serde.rs). Just enable the **serde-serialize** feature for
**nalgebra** on your project's `Cargo.toml` file.

--------

#### Why so many types? Why not just stick with raw matrices and vectors?

It is common within the computer-graphics community to work only with 4x4
matrices for transformations and 3D vectors for translations and positions.
**nalgebra** on the other hand has different types for rotations, isometries,
points, unit quaternions, etc. This wide variety of types:

* adds extra semantics so that the user constantly knows exactly what kind of
  algebraic object are being manipulated.
* allows them to be used safely in a generic context because their
  intrinsic properties are always known.
* allows optimizations, e.g., for transformation matrices inversion.

Thus, instead of working with raw matrices, higher-level types should be
preferred and only the end-result of all operations should be transformed into
a raw matrix to be usable by, e.g., a shader.

--------

#### Do I need any permission to reuse the figures of this guide?

Some figures on the front page are licenced under CC 3.0 BY licence and their
respective authors are credited [there](../about#image-credits).  All the other
figures on the guide have been created using
[Inkscape](http://www.inkscape.org/) and may be modified, published, and
redistributed anywhere without asking or even telling anybody!
