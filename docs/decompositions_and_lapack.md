# Matrix decompositions

Matrix decomposition is a family of methods that aim to represent a matrix as
the product of several matrices. Those factors can either allow more efficient
operations like inversion or linear system resolution, and might provide some
insight regarding intrinsic properties of some data to be analysed (e.g. by
observing singular values, eigenvectors, etc.) Some decompositions are
implemented in pure Rust or available as bindings to a Fortran Lapack
implementation (refer to the section on
[nalgebra-lapack](#lapack-integration)). In this section, we present
decompositions implemented in pure Rust for **real** matrices (complex matrices
are not supported yet):

Decomposition            | Factors                   | Rust methods
-------------------------|---------------------------|--------------
QR                       | $Q ~ R$                   | `.qr()`
LU with partial pivoting | $P^{-1} ~ L ~ U$          | `.lu()`
LU with full pivoting    | $P^{-1} ~ L ~ U ~ Q^{-1}$ | `.full_piv_lu()`
Hessenberg               | $P ~ H ~ P^*$             | `.hessenberg()`
Cholesky                 | $L ~ L^*$                 | `.cholesky()`
Real Schur decomposition | $Q ~ U ~ Q^*$             | `.real_schur()` or `.try_real_schur(eps, max_iter)`
Symmetric eigendecomposition | $U ~ \Lambda ~ U^*$   | `.symmetric_eigen()` or `.try_symmetric_eigen(eps, max_iter)`
SVD                      | $U ~ \Sigma ~ V^*$        | `.svd(compute_u, compute_v)` or `.try_svd(compute_u, compute_v, eps, max_iter)`

All those methods return a dedicated data structure representing the
corresponding decomposition. Those structures themselves often offer
interesting features. For example, the `LU` structure returned by the
`.lu(...)` method allows the efficient resolution of multiple linear systems.

Methods prefixed by `.try_` allow the customization of the error epsilon `eps`
and of a hard limit of iteration number `max_niter` for iterative algorithms.
`None` is returned if the given number of iterations is exceedeed before
convergence. By default, the relative and absolute error epsilons are equal to
the floating-point epsilon (i.e. the difference between 1 and the least value
greater than 1 that is representable).

In the following, all `.unpack` methods consume the decomposition data structure to
produce the factors with as little allocations as possible.

## Cholesky decomposition
The Cholesky decompositon of a `n × n` Symmetric Definite Positive (SDP) matrix
$M$ is composed of a `n × n` lower-triangular matrix $L$ such that $M = LL^*$.
Where $L^*$ designates the conjugate-transpose of $L$.  If the input matrix is
not SDP, such a decomposition does not exist and the matrix method
`.cholesky(...)` returns `None`. Note that the input matrix is interpreted as a
symmetric matrix. Only its lower-triangular part (including the diagonal) is
read by the Cholesky decomposition (its strictly upper-triangular is never
accessed in memory). See [the wikipedia
article](https://en.wikipedia.org/wiki/Cholesky_decomposition) for further
details about the Cholesky decomposition.

Typical uses of the Cholesky decomposition include the inversion of SDP
matrices and resolution of SDP linear systems.


<center>
![Cholesky decompositpion of a 3x3 matrix.](../img/cholesky.svg)
</center>

Method            | Effect
------------------|-----------
`.l()`            | Retrieves the lower-triangular factor $L$ of this decomposition, setting its strictly upper-triangular part to 0. |
`.l_dirty()`      | Retrieves  reference to the lower-triangular factor $L$ of this decomposition. Its strictly upper-triangular part is not set to 0 and may contain garbage. |
`.inverse()`      | Computes the inverse of the decomposed matrix.
`.solve(b)`       | Solves the system $Ax = b$ where $A$ is represented by `self` and $x$ the unkown. |
`.solve_mut(b)`   | Overwrites `b` with the solution of the system $Ax = b$ where $A$ is represented by `self` and $x$ the unkown. |
`.unpack()`       | Consumes `self` to return the lower-triangular factor $L$ of this decomposition, setting its strictly upper-triangular part to 0. |
`.unpack_dirty()` | Consumes `self` to return the lower-triangular factor $L$ of this decomposition. Its strictly upper-triangular part is not set to 0 and may contain garbage. |

## QR
The QR decomposition of a general `m × n` matrix $M$ is composed of an `m ×
min(n, m)` unitary matrix $Q$, and a `min(n, m) × m` upper triangular matrix
(or upper trapezoidal if $m < n$) $R$ such that $M = QR$. This can be used to
compute the inverse or a matrix or for solving linear systems. See also [the
wikipedia article](https://en.wikipedia.org/wiki/QR_decomposition) for further
details.

<center>
![QR decomposition of a 3x3 matrix.](../img/QR.svg)
</center>

Method             | Effect
-------------------|-----------
`.q()`             | Retrieves the unitary matrix $Q$ part of the decomposition.
`.r()`             | Retrieves the upper-triangular matrix $R$ part of the decomposition.
`.q_tr_mul(rhs)`   | Overwrites `rhs` with the result of `self.q() * rhs`. This is much more efficient than computing $Q$ explicitly. |
`.is_invertible()` | Determines if the decomposed matrix is invertible. |
`.inverse()`       | Computes the inverse of the decomposed matrix.
`.solve(b)`        | Solves the system $Ax = b$ where $A$ is represented by `self` and $x$ the unkown. |
`.solve_mut(b)`    | Overwrites `b` with the solution of the system $Ax = b$ where `A` is represented by `self` and $x$ the unkown. |
`.unpack()`        | Consumes `self` to return the two factors $(Q, R)$ of this decomposition. |


## LU with partial or full pivoting
The LU decomposition of a general `m × n` matrix is composed of a `m × min(n,
m)` lower triangular matrix $L$ with a diagonal filled with 1, and a `min(n, m)
× m` upper triangular matrix $U$ such that $M = LU$. This decomposition is
typically used for solving linear systems, compute determinants, matrix
inverse, and matrix rank. Two versions of the decomposition are implemented in
**nalgebra**:

* `LU` decomposition with partial (row) pivoting which computes additionally
  only one permutation matrix $P$ such that $PM = LU$. Implemented only for
  **square matrices**.
* `FullPivLU`: decomposition with full (row and column) pivoting which computes
  additionally two permutation matrices $P$ and $Q$ such that $PMQ = LU$.

Partial pivoting should provide good results in general. Is used internally to
compute the determinant, inversibility of a general matrix. Full pivoting is
less efficient but more numerically stable. See also [the wikipedia
article](https://en.wikipedia.org/wiki/LU_decomposition) for further details.

<center>
![LU decompositpon of a 3x3 matrix.](../img/LU.svg)
</center>


Method                   | Effect
-------------------------|-----------
`.l()`                   | Retrieves the lower-triangular matrix $L$ part of the decomposition.
`.u()`                   | Retrieves the upper-triangular matrix $U$ part of the decomposition.
`.p()`                   | Computes the explicitly permutation matrix $P$ that made the decomposition possible.
`.is_invertible()`       | Determines if the decomposed matrix is invertible. |
`.inverse()`             | Computes the inverse of the decomposed matrix.
`.determinant()`         | Computes the determinant of the decomposed matrix.
`.solve(b)`              | Solves the system $Ax = b$ where $A$ is represented by `self` and $x$ the unkown. |
`.solve_mut(b)`          | Overwrites `b` with the solution of the system $Ax = b$ where $A$ is represented by `self` and $x$ the unkown. |
`.unpack()`              | Consumes `self` to return the three factors $(P, L, U)$ of this decomposition. The four factors $(P, L, U, Q)$ are returned when using full pivoting. |


## Hessenberg decomposition
The hessenberg decomposition of a square matrix $M$ is composed of an orthogonal
matrix $Q$ and an upper-Hessenberg matrix $H$ such that $M = QHQ^*$. The matrix
$H$ being upper-Hessenberg means that all components below its first
subdiagonal are zero. See also [the wikipedia
article](https://en.wikipedia.org/wiki/Hessenberg_matrix) for further details.

The hessenberg decomposition is typically used as an intermediat representation
of a wide variety of algorithms that can benefit from its structure close to
the structure of an upper-triangular matrix.

<center>
![Hessenberg decomposition of a 3x3 matrix.](../img/hessenberg.svg)
</center>

Method                   | Effect
-------------------------|-----------
`.q()`                   | Retrieves the unitary matrix $Q$ part of the decomposition.
`.r()`                   | Retrieves the Hessenberg matrix $H$ of this decomposition.
`.unpack()`              | Consumes `self` to return the two factors $(Q, H)$ of this decomposition. |
`.unpack_h()`            | Consumes `self` to return the Hessenberg matrix $H$ of this decomposition. |

## Real Schur decomposition
The real Schur decomposition of a general `m × n` matrix $M$ is composed of an
`m × min(n, m)` unitary matrix $Q$, and a `min(n, m) × m` upper
quasi-upper-triangular matrix $T$ such that $M = QTQ^*$. A
quasi-upper-triangular matrix is a matrix which is upper-triangular except for
some 2x2 blocks on its diagonal (i.e. some of its subdiagonal elements are
sometimes non-zero and two consecutive diagonal elements cannot be zero
simultanously).

It is noteworthy that the diagonal elements of the quasi-upper-triangular
matrix are the eigenvalues of the decomposed matrix. The complex eigenvalues of
the 2x2 blocks on the diagonal corresponds to a conjugate pair of complex
eigenvalues. In the following example, the decomposed 4x4 matrix has two real
eigenvalues $\sigma_1$ and $\sigma_4$ and a conjugate pair of complex
eigenvalues $\sigma_2$ and $\sigma_3$ equal to the complex eigenvalues of the
2x2 diagonal block in the middle.

<center>
![Real Schur decomposition of a 4x4 matrix.](../img/real_schur.svg)
</center>

Method                   | Effect
-------------------------|-----------
`.eigenvalues()`         | Retrieves the real eigenvalues of the decomposed matrix. `None` if some eigenvalues should be complex.
`.complex_eigenvalues()` | Retrieves all the eigenvalues of the decomposed matrix returned as complex numbers.
`.unpack()`              | Consumes `self` to return the two factors $(Q, T)$ of this decomposition. |

## Eigendecomposition of a symmetric matrix
The eigendecomposition of a real square symmetric matrix $M$ is composed of an
unitary matrix $Q$ and a real diagonal matrix $\Lambda$ such that $M = Q\Lambda
Q^t$. The columns of $Q$ are called the _eigenvectors_ of $M$ and the diagonal
elements of $\Lambda$ its _eigenvalues_.

The matrix $Q$ and the eigenvalues of the decomposed matrix are respectively
accessible as public the fields `eigenvectors` and `eigenvalues` of the
`SymmetricEigen` structure.

<center>
![Eigendecomposition of a 3x3 symmetric matrix.](../img/symmetric_eigen.svg)
</center>

Method                   | Effect
-------------------------|-----------
`.recompose()`           | Recomputes the original matrix, i.e., $Q\Lambda{}Q^t$. Useful if some eigenvalues or eigenvectors have been manually modified.

## Singular Value Decomposition
The Singular Value Decomposition (SVD) of a real rectangular matrix if composed
of two orthogonal matrices $U$ and $V$ and a diagonal matrix $\Sigma$ with positive
(or zero) components. Typical uses of the SVD are the pseudo-inverse, rank
computation, and the resolution of least-square problems.

The singular values, left singular vectors, and right singular vectors are
respectively stored on the public fields `singular_values`, `u` and `v_t`. Note
that `v_t` represents the transpose of the matrix $V$.

<center>
![SVD decomposition of a 3x3 matrix.](../img/SVD.svg)
</center>

Method                 | Effect
-----------------------|-----------
`.recompose()`         | Reconstructs the matrix from its decomposition. Useful if some singular values or singular vectors have been manually modified.
`.pseudo_inverse(eps)` | Computes the pseudo-inverse of the decomposed matrix. All singular values below `eps` will be interpreted as equal to zero.
`.rank(eps)`           | Computes the rank of the decomposed matrix, i.e., the number of singular values strictly greater than `eps`.
`.solve(b, eps)`       | Solves the linear system $Ax = b$ where $A$ is the decomposed square matrix and $x$ the unkonwn. All singular value smaller or equal to `eps` are interpreted as zero.

# Lapack integration


The [**nalgebra-lapack**](https://crates.io/crates/nalgebra-lapack)
crate is based on bindings to C or Fortran implementation of the Linear Algebra
PACKage, aka. [LAPACK](). The factorizations supported by **nalgebra-lapack**
are the same as those supported by the pure-Rust version. They are all computed
by the constructor of a Rust structure:

Decomposition            | Factors                   | Rust constructors
-------------------------|---------------------------|--------------
QR                       | $Q ~ R$                   | `QR::new(matrix)`
LU with partial pivoting | $P^{-1} ~ L ~ U$          | `LU::new(matrix)`
Hessenberg               | $P ~ H ~ P^*$             | `Hessenberg::new(matrix)`
Cholesky                 | $L ~ L^*$                 | `Cholesky::new(matrix)`
Real Schur decomposition | $Q ~ U ~ Q^*$             | `Schur::new(matrix)` or `Schur::try_new(matrix)`
Symmetric eigendecomposition | $U ~ \Lambda ~ U^*$ | `SymmetricEigen::new(matrix)` or `SymmetricEigen::try_new(matrix)`
SVD                      | $U ~ \Sigma ~ V^*$        | `SVD::new(matrix)` or `SVD::try_new(matrix)`

The `::try_new` constructors return `None` if the decomposition fails while
`::new` constructiors panic.


The next subsections describe how to select the desired Lapack implementation,
and provide more details reagarding each decomposition.

## Setting up **nalgebra-lapack**

Several implementations of Lapack exist. The desired one should be selected on
your `Cargo.toml` file by enabling the related feature for your
**nalgebra-lapack** dependency. The currently supported implementations are:

* [OpenBLAS](www.openblas.net) enabled by the `openblas` feature.
* [netlib](www.netlib.org) enabled by the `netlib` feature.
* [Accelerate](https://developer.apple.com/reference/accelerate) enabled by the
  `accelerate` feature on Mac OS only.

The `openblas` feature is enabled by default. The following examples shows how
to enable `Accelerate` instead:

```yaml
[dependencies.nalgebra_lapack]
version = "0.5"
default-features = false
features = [ "Accelerate" ]
```

Note that enabling two such features simultaneously will lead to compilation
errors inside of the **nalgebra-lapack** crate itself. Thus, specifying
`default-features = false` is extremely important when selecting an
implementation other than the default.
