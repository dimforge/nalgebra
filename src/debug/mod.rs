//! Various tools useful for testing, debugging, and benchmarking.
//!
//! This module provides utilities for generating special types of matrices that are
//! commonly needed when testing numerical linear algebra algorithms. These tools help
//! verify that your code works correctly with different matrix properties.
//!
//! # What's Available
//!
//! - [`RandomOrthogonal`]: Generate random orthogonal matrices (matrices where columns
//!   are perpendicular unit vectors)
//! - [`RandomSDP`]: Generate random symmetric positive-definite matrices (important for
//!   many optimization and numerical algorithms)
//!
//! # Why Use These Tools?
//!
//! When testing linear algebra code, you often need matrices with specific mathematical
//! properties. For example:
//! - Orthogonal matrices are useful for testing rotation and transformation code
//! - Symmetric positive-definite matrices are essential for testing optimization algorithms,
//!   Cholesky decomposition, and other numerical methods
//!
//! These generators ensure your test matrices have the exact properties you need, which
//! is difficult to achieve with purely random values.
//!
//! # Example: Testing an Algorithm
//!
//! ```
//! use nalgebra::{Matrix3, DMatrix};
//! use nalgebra::debug::{RandomOrthogonal, RandomSDP};
//!
//! // Simple counter-based random number generator for testing
//! let mut counter = 0.0_f64;
//! let mut rand_fn = || { counter += 0.1; counter.sin() };
//!
//! // Generate a 3x3 orthogonal matrix for testing rotations
//! let orthogonal = RandomOrthogonal::<f64, _>::new(
//!     nalgebra::Const::<3>,
//!     &mut rand_fn
//! );
//! let matrix = orthogonal.unwrap();
//!
//! // Verify it's orthogonal: Q^T * Q should equal identity
//! let identity = matrix.transpose() * &matrix;
//! assert!((identity - Matrix3::identity()).norm() < 1e-10);
//!
//! // Generate a symmetric positive-definite matrix for testing algorithms
//! let mut counter2 = 0.0_f64;
//! let sdp = RandomSDP::<f64, _>::new(
//!     nalgebra::Const::<3>,
//!     || { counter2 += 0.1; counter2.abs() }
//! );
//! let spd_matrix = sdp.unwrap();
//! ```

mod random_orthogonal;
mod random_sdp;

pub use self::random_orthogonal::*;
pub use self::random_sdp::*;
