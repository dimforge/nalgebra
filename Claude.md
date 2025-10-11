# nalgebra - Linear Algebra Library for Rust

## Overview

**nalgebra** is a comprehensive, general-purpose linear algebra library for Rust, specifically targeting:
- General-purpose linear algebra computations
- Real-time computer graphics
- Real-time computer physics

**Current Version**: 0.34.1 (released Sept 20, 2025)
**License**: Apache-2.0
**MSRV**: 1.87.0 (Rust 2024 edition)
**Author**: Sébastien Crozet

## Recent Documentation Enhancement (October 2025)

**All 1,065+ public functions** in the `src/` directory now have comprehensive, beginner-friendly documentation with:
- Detailed explanations in clear, simple language
- 3,000+ working code examples
- Practical use cases from game dev, robotics, graphics, physics, and scientific computing
- "See Also" sections linking related functions
- Complete parameter and return value documentation
- Performance notes and panic conditions

See `DOCUMENTATION_SUMMARY.md` for complete details.

## Project Structure

This is a **workspace** containing multiple crates:

### Main Crates

1. **nalgebra** (core library) - ~54,765 lines of Rust code
   - Main linear algebra functionality
   - Matrix and vector operations
   - Geometric transformations
   - Matrix decompositions

2. **nalgebra-sparse** - Sparse matrix computations
   - COO (Coordinate format)
   - CSR (Compressed Sparse Row)
   - CSC (Compressed Sparse Column)
   - Matrix market format support

3. **nalgebra-glm** - OpenGL Mathematics (GLM) API
   - Computer graphics-oriented API inspired by C++ GLM library
   - Provides familiar interface for graphics programmers

4. **nalgebra-lapack** - LAPACK bindings
   - High-performance linear algebra using LAPACK
   - Additional decompositions and solvers

5. **nalgebra-macros** - Procedural macros
   - Convenient constructors: `matrix!`, `vector!`, `point!`, `stack!`, etc.

## Core Architecture

### Module Organization

```
src/
├── base/           - Core matrix and vector types (38 files)
│   ├── matrix.rs   - Generic Matrix<T, R, C, S> type
│   ├── storage/    - Storage backends (array, vec, views)
│   ├── allocator/  - Memory allocation strategies
│   └── ...         - Operations, construction, indexing, etc.
│
├── geometry/       - Geometric types and transformations (71 files)
│   ├── point.rs
│   ├── rotation.rs
│   ├── quaternion.rs
│   ├── isometry.rs
│   ├── similarity.rs
│   ├── transform.rs
│   └── perspective.rs / orthographic.rs
│
├── linalg/         - Matrix decompositions (27 files)
│   ├── cholesky.rs
│   ├── lu.rs
│   ├── qr.rs
│   ├── svd.rs
│   ├── schur.rs
│   └── ...         - Other decompositions
│
├── sparse/         - Sparse matrix support
├── third_party/    - Integration with other libraries
└── lib.rs          - Main entry point
```

### Key Design Patterns

#### 1. Generic Dimensions
The library supports both compile-time and runtime dimensions:
- **Static dimensions**: `Matrix3x4`, `Vector3` - dimensions known at compile time
- **Dynamic dimensions**: `DMatrix`, `DVector` - dimensions known at runtime
- **Type-level integers**: `U1` to `U127` (now aliases for `Const<D>`)
- **Dynamic marker**: `Dyn` for runtime-sized dimensions

#### 2. Storage Abstraction
Matrix storage is abstracted through the `Storage` trait:
- `ArrayStorage<T, R, C>` - Stack-allocated for static sizes
- `VecStorage<T, R, C>` - Heap-allocated for dynamic sizes
- Matrix views (formerly called slices) - Borrowed references

#### 3. Type Aliases
Extensive use of type aliases for ergonomics:
```rust
type Matrix3<T> = Matrix<T, U3, U3, ArrayStorage<T, 3, 3>>;
type Vector3<T> = Matrix<T, U3, U1, ArrayStorage<T, 3, 1>>;
type DMatrix<T> = Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>;
```

## Major Features

### 1. Matrix and Vector Types

- **Statically sized**: `Matrix1x1` to `Matrix6x6`, `Vector1` to `Vector6`
- **Dynamically sized**: `DMatrix`, `DVector`
- **Mixed sizes**: `Matrix2xX` (2 rows, dynamic columns)
- **Points**: `Point1` to `Point6`

### 2. Geometric Transformations

All transformation types for 2D and 3D:

- **Translations**: `Translation2`, `Translation3`
- **Rotations**: `Rotation2`, `Rotation3`, `UnitQuaternion`, `UnitComplex`
- **Isometries**: `Isometry2`, `Isometry3` (translation + rotation)
- **Similarities**: `Similarity2`, `Similarity3` (translation + rotation + uniform scale)
- **Affine transformations**: `Affine2`, `Affine3`
- **Projective transformations**: `Projective2`, `Projective3`
- **General transformations**: `Transform2`, `Transform3`
- **Scaling**: `Scale` type for non-uniform scaling
- **Unit types**: `Unit<T>` wrapper for normalized values

### 3. Matrix Decompositions (Pure Rust)

All implemented without external dependencies:

- **Cholesky** decomposition
- **LU** decomposition with partial pivoting
- **FullPivLU** with full pivoting
- **QR** decomposition
- **ColPivQR** with column pivoting
- **SVD** (Singular Value Decomposition) - special cases for 2x2 and 3x3
- **Schur** decomposition
- **Hessenberg** reduction
- **SymmetricEigen** decomposition
- **SymmetricTridiagonal** reduction
- **Bidiagonal** decomposition
- **UDU** factorization
- Matrix **exponential** and **polar** decomposition

### 4. Computer Graphics Support

- 3D projections: `Perspective3`, `Orthographic3`
- Homogeneous coordinates
- Look-at matrices (both LH and RH)
- Transformation helpers for graphics pipelines

### 5. BLAS-like Operations

Pure Rust implementations:
- `gemv`, `gemm` - Matrix-vector and matrix-matrix multiplication
- `ger`, `gerc` - Outer products
- `axpy` - Scaled addition
- `iamax`, `icamax` - Index of max element
- Symmetric variants: `sygemv`, `hegemv`
- Hermitian operations for complex matrices

## Feature Flags

### Core Features
- `std` (default) - Standard library support
- `macros` (default) - Enable procedural macros
- `alloc` - Allocation without std
- `libm` / `libm-force` - Math functions for no_std

### Serialization
- `serde-serialize` - Full serde support
- `serde-serialize-no-std` - Serde without std
- `rkyv-serialize` - rkyv serialization support
- `rkyv-serialize-no-std` - rkyv without std

### Randomness
- `rand` - Full random number generation
- `rand-no-std` - RNG without std

### Conversions
- `convert-mint` - Mint library compatibility
- `convert-bytemuck` - Bytemuck trait implementations
- `convert-glam014` through `convert-glam030` - glam library conversions

### Other
- `sparse` - Sparse matrix support
- `compare` - Matrix comparison utilities
- `debug` - Additional debugging features
- `io` - Matrix I/O using pest parser
- `proptest-support` - Property testing
- `rayon` - Parallel iterators
- `defmt` - Embedded debugging format
- `encase` - Shader-friendly types

## No-std Support

nalgebra is designed for embedded systems and WebAssembly:
- Core functionality works in `#![no_std]` environments
- Optional `alloc` feature for heap allocation without std
- Optional `libm` feature for math functions
- Careful feature gating throughout

## Dependencies

**Core dependencies**:
- `typenum` - Type-level integers
- `num-traits`, `num-complex`, `num-rational` - Numeric traits
- `approx` - Approximate equality testing
- `simba` 0.9 - Abstract algebra and SIMD traits

**Optional**:
- `rand` 0.9 - Random number generation
- `serde` - Serialization
- `rkyv` - Zero-copy serialization
- `matrixmultiply` - Optimized matrix multiplication
- `glam` (multiple versions) - Game math library integration
- `rayon` - Data parallelism
- `bytemuck` - Type casting

## Recent Changes (v0.34.x)

### Added
- `encase` feature for shader-compatible types
- `defmt` feature for embedded debugging
- `convert-glam030` for glam v0.30
- `AsRef<[T]>` for contiguous matrices
- Bytemuck traits for isometries and similarities

### Changed
- Bumped MSRV to 1.87.0
- Moved to Rust 2024 edition
- Updated to rand 0.9.0
- Many methods now `const fn` where possible
- `DimName::USIZE` renamed to `DimName::DIM`
- Glam conversion features no longer enable default features (better no_std support)

### Fixed
- Infinite loop in Schur decomposition of zero matrices
- Build issues with `glam` in no_std environments

## Testing and Examples

- **Examples**: 18 example files demonstrating various features
  - Matrix construction
  - Point construction
  - Transformations (MVP matrices, view coordinates)
  - Linear system resolution
  - Homogeneous coordinates
  - Generic programming patterns

- **Tests**: Organized in separate test files by category
  - Core tests
  - Geometry tests
  - Linear algebra tests
  - Macro tests (using `trybuild`)

- **Benchmarks**: Comprehensive benchmark suite using Criterion

## Notable Implementation Details

### 1. Memory Safety
- Extensive use of `MaybeUninit` for uninitialized matrices
- Safe abstractions over potentially uninitialized memory
- Careful handling of allocator traits

### 2. Trait System
- `Scalar` trait for matrix components (auto-implemented for most `'static + Clone`)
- `ComplexField` and `RealField` from simba
- Separate traits for different field requirements
- Generic over storage backend

### 3. Views (formerly Slices)
Matrix views provide zero-copy access to matrix data:
- Renamed from "slices" to "views" in v0.32.0
- Support for strided views
- Mutable and immutable variants
- Range-based indexing

### 4. Const Generics
Modern Rust const generics are used throughout:
- `Const<D>` replaces old typenum approach
- Many constructors are `const fn`
- Better type inference and error messages

## Integration Points

### Graphics Libraries
- **glam**: Multiple version support (0.14-0.30) for game development
- **mint**: Generic math interchange format
- **GLM-style API**: Via nalgebra-glm subcrate

### Serialization
- **serde**: Full support for all types
- **rkyv**: Zero-copy deserialization
- **bytemuck**: Type casting for GPU buffers

### Ecosystem
- **simba**: Abstract algebra traits and SIMD support
- **alga**: Legacy algebra traits (optional, deprecated)
- **LAPACK**: Via nalgebra-lapack for high-performance computing

## Community and Resources

- **Website**: https://nalgebra.rs (with user guide)
- **Documentation**: https://docs.rs/nalgebra
- **Repository**: https://github.com/dimforge/nalgebra
- **Discord**: Active community for support
- **Maintained by**: dimforge organization

## Development Philosophy

1. **Performance**: Zero-cost abstractions where possible
2. **Genericity**: Works with different scalar types (float, complex, SIMD)
3. **Safety**: Rust's type system prevents common linear algebra bugs
4. **Ergonomics**: Convenient APIs with helpful type aliases
5. **Portability**: Works in no_std, WASM, and embedded contexts
6. **Correctness**: Comprehensive test suite and property testing

## Historical Notes

- Originally started before Rust 1.0
- Major redesign in v0.11 to use alga traits
- Transitioned to const generics in v0.26
- Views renamed from slices in v0.32 (see issue #1076)
- Moved from alga to simba in v0.21 for SIMD support
- Continuous evolution with Rust's type system improvements

## Code Statistics

- **Total source code**: ~54,765 lines of Rust
- **Test code**: ~9,815 lines
- **Benchmark code**: ~1,611 lines
- **Documentation**: ~1,597 lines across markdown files
- **Geometry module**: 71 files covering all transformation types
- **Matrix public methods**: 70+ public functions just in core Matrix type
- **Largest files**:
  - `matrix.rs` (2,380 lines) - Core matrix implementation
  - `quaternion.rs` (1,711 lines) - Quaternion operations
  - `edition.rs` (1,308 lines) - Matrix editing operations
  - `construction.rs` (1,293 lines) - Matrix construction methods

## Contributors

Led by **Sébastien Crozet** with major contributions from:
- Andreas Longva (sparse matrices)
- metric-space
- Bruce Mitchener
- Eduard Bopp
- Jack Wrenn
- Violeta Hernández
- And 100+ other contributors

Total: 1,000+ commits from the core maintainer, with strong community involvement.

## CI/CD

- GitHub Actions workflow for continuous integration
- Tests across multiple Rust versions
- Feature flag combination testing
- Documentation generation verification
- No Makefile - pure Cargo-based build system

## Active Development

The library is actively maintained with:
- Regular updates for new Rust editions
- Support for latest dependency versions
- Bug fixes and performance improvements (recent: Schur decomposition fix)
- New features based on community needs (recent: encase support, defmt)
- Excellent backward compatibility (semantic versioning)
- Active issue tracking and PR reviews on GitHub
