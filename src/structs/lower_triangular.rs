use std::rand::Rand;
use std::rand;
use std::num::{One, Zero};
use std::vec;

/// A structure optimized to store lower triangular matrices.
pub struct LowerTriangularMat<N> {
    priv dim: uint,
    priv mij: ~[N]
}

/// Trait to be implemented by objects which can be left-multiplied by a lower triangular array.
pub trait LowerTriangularMatMulRhs<N, Res> {
    /// Apply the muliplicitaion.
    fn binop(left: &LowerTriangularMat<N>, right: &Self) -> Res;
}

impl<N, Rhs: LowerTriangularMatMulRhs<N, Res>, Res> Mul<Rhs, Res> for LowerTriangularMat<N> {
    #[inline(always)]
    fn mul(&self, other: &Rhs) -> Res {
        LowerTriangularMatMulRhs::binop(self, other)
    }
}

impl<N> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix without initializing its arguments.
    #[inline]
    pub unsafe fn new_uninitialized(dim: uint) -> LowerTriangularMat<N> {
        let mut vec = vec::with_capacity(dim * (dim + 1) / 2);
        vec::raw::set_len(&mut vec, dim * (dim + 1) / 2);

        LowerTriangularMat {
            dim: dim,
            mij: vec
        }
    }
}

impl<N: Zero + Clone> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix filled with zeros.
    #[inline]
    pub fn new_zeros(dim: uint) -> LowerTriangularMat<N> {
        LowerTriangularMat::from_elem(dim, Zero::zero())
    }

    /// Tests if every entry of the matrix are exactly zeros.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mij.iter().all(|e| e.is_zero())
    }
}

impl<N: Rand> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix filled with random elements.
    #[inline]
    pub fn new_random(dim: uint) -> LowerTriangularMat<N> {
        LowerTriangularMat::from_fn(dim, |_, _| rand::random())
    }
}

impl<N: One + Clone> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix filled with ones.
    #[inline]
    pub fn new_ones(dim: uint) -> LowerTriangularMat<N> {
        LowerTriangularMat::from_elem(dim, One::one())
    }
}

impl<N: Clone> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix filled with a given value.
    #[inline]
    pub fn from_elem(dim: uint, val: N) -> LowerTriangularMat<N> {
        LowerTriangularMat {
            dim: dim,
            mij: vec::from_elem(dim * (dim + 1) / 2, val)
        }
    }
}

impl<N> LowerTriangularMat<N> {
    /// Creates a lower triangular matrix filled by a function.
    #[inline(always)]
    pub fn from_fn(dim: uint, f: &fn(uint, uint) -> N) -> LowerTriangularMat<N> {
        let mij = do vec::from_fn(dim * (dim + 1) / 2) |i| {
            let l = (((1.0f64 + 8.0f64 * i as f64).sqrt() - 1.) / 2.0f64).floor() as uint;
            let c = i - l * (l + 1) / 2;

            f(l, c)
        };

        LowerTriangularMat {
            dim: dim,
            mij: mij
        }
    }

    #[inline]
    fn offset(&self, i: uint, j: uint) -> uint {
        i * (i + 1) / 2 + j + 1
    }

    /// Transforms this matrix into an array. This consumes the matrix and is O(1).
    #[inline]
    pub fn to_array(self) -> ~[N] {
        self.mij
    }
}

impl<N: Zero + Clone> LowerTriangularMat<N> {
    /// Changes the value of a component of the matrix.
    /// Fails if the indices point outside of the lower-triangular part of the matrix.
    ///
    /// # Arguments
    ///   * `row` - 0-based index of the line to be changed
    ///   * `col` - 0-based index of the column to be changed
    #[inline]
    pub fn set(&mut self, row: uint, col: uint, val: N) {
        assert!(row < self.dim);
        assert!(col < self.dim);
        assert!(col <= row);
        unsafe { self.set_fast(row, col, val) }
    }

    /// Just like `set` without bounds checking.
    #[inline]
    pub unsafe fn set_fast(&mut self, row: uint, col: uint, val: N) {
        let offset = self.offset(row, col);
        *self.mij.unsafe_mut_ref(offset) = val
    }

    /// Reads the value of a component of the matrix.
    /// Fails if the indices point outside of the lower-triangular part of the matrix.
    ///
    /// # Arguments
    ///   * `row` - 0-based index of the line to be read
    ///   * `col` - 0-based index of the column to be read
    #[inline]
    pub fn at(&self, row: uint, col: uint) -> N {
        assert!(row < self.dim);
        assert!(col < self.dim);
        unsafe { self.at_fast(row, col) }
    }

    /// Just like `at` without bounds checking.
    #[inline]
    pub unsafe fn at_fast(&self, row: uint, col: uint) -> N {
        if col > row {
            Zero::zero()
        }

        vec::raw::get(self.mij, self.offset(row, col))
    }
}
