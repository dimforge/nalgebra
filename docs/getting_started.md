# Getting started
**nalgebra** relies on the official Rust package manager
[Cargo](http://crates.io) for dependency resolution and compilation. Therefore,
making **nalgebra** ready to be used by your project is simply a matter of
adding a new dependency to your `Cargo.toml` file.

```toml
[dependencies]
nalgebra = "0.14.0"
```

Until **nalgebra** reaches 1.0, it is strongly recommended to always use its
latest version, though you might encounter breaking changes from time to time.
Once your `Cargo.toml` file is set up, the corresponding crate must be imported
by your project with the usual `extern crate` directive. We recommend using the
`na` abbreviation for referencing the crate throughout your application.

```rust
extern crate nalgebra as na;
```

## Cargo example
You may use this `Cargo.toml` file to compile the downloadable examples of this
guide. Simply replace `example.rs` by the actual example's file name.

<ul class="nav nav-tabs">
  <li class="active"><a id="tab_nav_link" data-toggle="tab" href="#cargo">Example</a></li>

  <div class="btn-primary" onclick="window.open('https://raw.githubusercontent.com/sebcrozet/nalgebra/master/examples/cargo/Cargo.toml')"></div>
</ul>

<div class="tab-content" markdown="1">
  <div id="cargo" class="tab-pane in active">
```toml
[package]
name    = "example-using-nalgebra"
version = "0.0.0"
authors = [ "You" ]

[dependencies]
nalgebra = "0.14.0"

[[bin]]
name = "example"
path = "./example.rs"
```
  </div>
</div>

## Usage and cargo features
**nalgebra** is a monolithic crate with all its functionalities exported at its
root. Data types, traits, and functions that do not take mutable inputs are
directly accessible behind the `nalgebra::` path (or `na::` if you use the
recommended alias). However, methods that perform in-place modifications
(normalization, appending a translation in-place, etc.) are not accessible as
free-functions. Instead, see the details of each data structure and the traits
they implement.


Finally, two optional features can be enabled if needed:

  * **arbitrary**: makes possible the use of [quickcheck](https://crates.io/crates/quickcheck).
  by adding implementations of the `Arbitrary` trait from quickcheck.
  * **serde-serialize**: makes possible the use of [serde](https://serde.rs)
  by adding implementations of the `Serialize` and `Deserialize` traits.
