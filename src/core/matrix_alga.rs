use num::{Zero, One};

use alga::general::{AbstractMagma, AbstractGroupAbelian, AbstractGroup, AbstractLoop,
                    AbstractMonoid, AbstractQuasigroup, AbstractSemigroup, AbstractModule,
                    Module, Field, RingCommutative, Real, Inverse, Additive, Multiplicative,
                    MeetSemilattice, JoinSemilattice, Lattice, Identity,
                    ClosedAdd, ClosedNeg, ClosedMul};
use alga::linear::{VectorSpace, NormedSpace, InnerSpace, FiniteDimVectorSpace, FiniteDimInnerSpace};

use core::{Scalar, Matrix, SquareMatrix};
use core::dimension::{Dim, DimName};
use core::storage::OwnedStorage;
use core::allocator::OwnedAllocator;

/*
 *
 * Additive structures.
 *
 */
impl<N, R: DimName, C: DimName, S> Identity<Additive> for Matrix<N, R, C, S>
    where N: Scalar + Zero,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn identity() -> Self {
        Self::from_element(N::zero())
    }
}

impl<N, R: DimName, C: DimName, S> AbstractMagma<Additive> for Matrix<N, R, C, S>
    where N: Scalar + ClosedAdd,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn operate(&self, other: &Self) -> Self {
        self + other
    }
}

impl<N, R: DimName, C: DimName, S> Inverse<Additive> for Matrix<N, R, C, S>
    where N: Scalar + ClosedNeg,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn inverse(&self) -> Matrix<N, R, C, S> {
        -self
    }

    #[inline]
    fn inverse_mut(&mut self) {
        *self = -self.clone()
    }
}

macro_rules! inherit_additive_structure(
    ($($marker: ident<$operator: ident> $(+ $bounds: ident)*),* $(,)*) => {$(
        impl<N, R: DimName, C: DimName, S> $marker<$operator> for Matrix<N, R, C, S>
            where N: Scalar + $marker<$operator> $(+ $bounds)*,
                  S: OwnedStorage<N, R, C>,
                  S::Alloc: OwnedAllocator<N, R, C, S> { }
    )*}
);

inherit_additive_structure!(
    AbstractSemigroup<Additive>    + ClosedAdd,
    AbstractMonoid<Additive>       + Zero + ClosedAdd,
    AbstractQuasigroup<Additive>   + ClosedAdd + ClosedNeg,
    AbstractLoop<Additive>         + Zero + ClosedAdd + ClosedNeg,
    AbstractGroup<Additive>        + Zero + ClosedAdd + ClosedNeg,
    AbstractGroupAbelian<Additive> + Zero + ClosedAdd + ClosedNeg
);

impl<N, R: DimName, C: DimName, S> AbstractModule for Matrix<N, R, C, S>
    where N: Scalar + RingCommutative,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    type AbstractRing = N;

    #[inline]
    fn multiply_by(&self, n: N) -> Self {
        self * n
    }
}

impl<N, R: DimName, C: DimName, S> Module for Matrix<N, R, C, S>
    where N: Scalar + RingCommutative,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    type Ring = N;
}

impl<N, R: DimName, C: DimName, S> VectorSpace for Matrix<N, R, C, S>
    where N: Scalar + Field,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    type Field = N;
}

impl<N, R: DimName, C: DimName, S> FiniteDimVectorSpace for Matrix<N, R, C, S>
    where N: Scalar + Field,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn dimension() -> usize {
        R::dim() * C::dim()
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Self {
        assert!(i < Self::dimension(), "Index out of bound.");

        let mut res = Self::zero();
        unsafe { *res.data.get_unchecked_linear_mut(i) = N::one(); }

        res
    }

    #[inline]
    fn dot(&self, other: &Self) -> N {
        self.dot(other)
    }

    #[inline]
    unsafe fn component_unchecked(&self, i: usize) -> &N {
        self.data.get_unchecked_linear(i)
    }

    #[inline]
    unsafe fn component_unchecked_mut(&mut self, i: usize) -> &mut N {
        self.data.get_unchecked_linear_mut(i)
    }
}

impl<N, R: DimName, C: DimName, S> NormedSpace for Matrix<N, R, C, S>
    where N: Real,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn norm_squared(&self) -> N {
        self.norm_squared()
    }

    #[inline]
    fn norm(&self) -> N {
        self.norm()
    }

    #[inline]
    fn normalize(&self) -> Self {
        self.normalize()
    }

    #[inline]
    fn normalize_mut(&mut self) -> N {
        self.normalize_mut()
    }

    #[inline]
    fn try_normalize(&self, min_norm: N) -> Option<Self> {
        self.try_normalize(min_norm)
    }

    #[inline]
    fn try_normalize_mut(&mut self, min_norm: N) -> Option<N> {
        self.try_normalize_mut(min_norm)
    }
}

impl<N, R: DimName, C: DimName, S> InnerSpace for Matrix<N, R, C, S>
    where N: Real,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    type Real = N;

    #[inline]
    fn angle(&self, other: &Self) -> N {
        self.angle(other)
    }

    #[inline]
    fn inner_product(&self, other: &Self) -> N {
        self.dot(other)
    }
}

// FIXME: specialization will greatly simplify this implementation in the future.
// In particular:
//   − use `x()` instead of `::canonical_basis_element`
//   − use `::new(x, y, z)` instead of `::from_slice`
impl<N, R: DimName, C: DimName, S> FiniteDimInnerSpace for Matrix<N, R, C, S>
    where N: Real,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn orthonormalize(vs: &mut [Matrix<N, R, C, S>]) -> usize {
        let mut nbasis_elements = 0;

        for i in 0 .. vs.len() {
            {
                let (elt, basis) = vs[.. i + 1].split_last_mut().unwrap();

                for basis_element in &basis[.. nbasis_elements] {
                    *elt -= &*basis_element * elt.dot(basis_element)
                }
            }

            if vs[i].try_normalize_mut(N::zero()).is_some() {
                // FIXME: this will be efficient on dynamically-allocated vectors but for
                // statically-allocated ones, `.clone_from` would be better.
                vs.swap(nbasis_elements, i);
                nbasis_elements += 1;

                // All the other vectors will be dependent.
                if nbasis_elements == Self::dimension() {
                    break;
                }
            }
        }

        nbasis_elements
    }

    #[inline]
    fn orthonormal_subspace_basis<F>(vs: &[Self], mut f: F)
        where F: FnMut(&Self) -> bool {
        // FIXME: is this necessary?
        assert!(vs.len() <= Self::dimension(), "The given set of vectors has no chance of being a free family.");

        match Self::dimension() {
            1 => {
                if vs.len() == 0 {
                    f(&Self::canonical_basis_element(0));
                }
            },
            2 => {
                if vs.len() == 0 {
                    let _ = f(&Self::canonical_basis_element(0)) &&
                            f(&Self::canonical_basis_element(1));
                }
                else if vs.len() == 1 {
                    let v = &vs[0];
                    let res = Self::from_column_slice(&[-v[1], v[0]]);

                    f(&res.normalize());
                }

                // Otherwise, nothing.
            },
            3 => {
                if vs.len() == 0 {
                    let _ = f(&Self::canonical_basis_element(0)) &&
                            f(&Self::canonical_basis_element(1)) &&
                            f(&Self::canonical_basis_element(2));
                }
                else if vs.len() == 1 {
                    let v = &vs[0];
                    let mut a;

                    if v[0].abs() > v[1].abs() {
                        a = Self::from_column_slice(&[v[2], N::zero(), -v[0]]);
                    }
                    else {
                        a = Self::from_column_slice(&[N::zero(), -v[2], v[1]]);
                    };

                    let _ = a.normalize_mut();

                    if f(&a.cross(v)) {
                        f(&a);
                    }
                }
                else if vs.len() == 2 {
                    f(&vs[0].cross(&vs[1]).normalize());
                }
            },
            _ => {
                // XXX: use a GenericArray instead.
                let mut known_basis = Vec::new();

                for v in vs.iter() {
                    known_basis.push(v.normalize())
                }

                for i in 0 .. Self::dimension() - vs.len() {
                    let mut elt = Self::canonical_basis_element(i);

                    for v in &known_basis {
                        elt -= v * elt.dot(v)
                    };

                    if let Some(subsp_elt) = elt.try_normalize(N::zero()) {
                        if !f(&subsp_elt) { return };

                        known_basis.push(subsp_elt);
                    }
                }
            }
        }
    }
}


/*
 *
 *
 * Multiplicative structures.
 *
 *
 */
impl<N, D: DimName, S> Identity<Multiplicative> for SquareMatrix<N, D, S>
    where N: Scalar + Zero + One,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn identity() -> Self {
        Self::identity()
    }
}

impl<N, D: DimName, S> AbstractMagma<Multiplicative> for SquareMatrix<N, D, S>
    where N: Scalar + Zero + ClosedAdd + ClosedMul,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    #[inline]
    fn operate(&self, other: &Self) -> Self {
        self * other
    }
}

macro_rules! impl_multiplicative_structure(
    ($($marker: ident<$operator: ident> $(+ $bounds: ident)*),* $(,)*) => {$(
        impl<N, D: DimName, S> $marker<$operator> for SquareMatrix<N, D, S>
            where N: Scalar + Zero + ClosedAdd + ClosedMul + $marker<$operator> $(+ $bounds)*,
                  S: OwnedStorage<N, D, D>,
                  S::Alloc: OwnedAllocator<N, D, D, S> { }
    )*}
);

impl_multiplicative_structure!(
    AbstractSemigroup<Multiplicative>,
    AbstractMonoid<Multiplicative> + One
);

// // FIXME: Field too strong?
// impl<N, S> Matrix for Matrix<N, S>
//     where N: Scalar + Field,
//           S: Storage<N> {
//     type Field     = N;
//     type Row       = OwnedMatrix<N, Static<U1>, S::C, S::Alloc>;
//     type Column    = OwnedMatrix<N, S::R, Static<U1>, S::Alloc>;
//     type Transpose = OwnedMatrix<N, S::C, S::R, S::Alloc>;

//     #[inline]
//     fn nrows(&self) -> usize {
//         self.shape().0
//     }

//     #[inline]
//     fn ncolumns(&self) -> usize {
//         self.shape().1
//     }

//     #[inline]
//     fn row(&self, row: usize) -> Self::Row {
//         let mut res: Self::Row = ::zero();

//         for (column, e) in res.iter_mut().enumerate() {
//             *e = self[(row, column)];
//         }

//         res
//     }

//     #[inline]
//     fn column(&self, column: usize) -> Self::Column {
//         let mut res: Self::Column = ::zero();

//         for (row, e) in res.iter_mut().enumerate() {
//             *e = self[(row, column)];
//         }

//         res
//     }

//     #[inline]
//     unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Field {
//         self.get_unchecked(i, j)
//     }

//     #[inline]
//     fn transpose(&self) -> Self::Transpose {
//         self.transpose()
//     }
// }

// impl<N, S> MatrixMut for Matrix<N, S>
//     where N: Scalar + Field,
//           S: StorageMut<N> {
//     #[inline]
//     fn set_row_mut(&mut self, irow: usize, row: &Self::Row) {
//         assert!(irow < self.shape().0, "Row index out of bounds.");

//         for (icol, e) in row.iter().enumerate() {
//             unsafe { self.set_unchecked(irow, icol, *e) }
//         }
//     }

//     #[inline]
//     fn set_column_mut(&mut self, icol: usize, col: &Self::Column) {
//         assert!(icol < self.shape().1, "Column index out of bounds.");
//         for (irow, e) in col.iter().enumerate() {
//             unsafe { self.set_unchecked(irow, icol, *e) }
//         }
//     }

//     #[inline]
//     unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: Self::Field) {
//         *self.get_unchecked_mut(i, j) = val
//     }
// }

// // FIXME: Real is needed here only for invertibility...
// impl<N: Real> SquareMatrixMut for $t<N> {
//     #[inline]
//     fn from_diagonal(diag: &Self::Coordinates) -> Self {
//         let mut res: $t<N> = ::zero();
//         res.set_diagonal_mut(diag);
//         res
//     }

//     #[inline]
//     fn set_diagonal_mut(&mut self, diag: &Self::Coordinates) {
//         for (i, e) in diag.iter().enumerate() {
//             unsafe { self.set_unchecked(i, i, *e) }
//         }
//     }
// }



// Specializations depending on the dimension.
// matrix_group_approx_impl!(common: $t, 1, $vector, $($compN),+);

// // FIXME: Real is needed here only for invertibility...
// impl<N: Real> SquareMatrix for $t<N> {
//     type Vector = $vector<N>;

//     #[inline]
//     fn diagonal(&self) -> Self::Coordinates {
//         $vector::new(self.m11)
//     }

//     #[inline]
//     fn determinant(&self) -> Self::Field {
//         self.m11
//     }

//     #[inline]
//     fn try_inverse(&self) -> Option<Self> {
//         let mut res = *self;
//         if res.try_inverse_mut() {
//             Some(res)
//         }
//         else {
//             None
//         }
//     }

//     #[inline]
//     fn try_inverse_mut(&mut self) -> bool {
//         if relative_eq!(&self.m11, &::zero()) {
//             false
//         }
//         else {
//             self.m11 = ::one::<N>() / ::determinant(self);

//             true
//         }
//     }

//     #[inline]
//     fn transpose_mut(&mut self) {
//         // no-op
//     }
// }

//  ident, 2, $vector: ident, $($compN: ident),+) => {
// matrix_group_approx_impl!(common: $t, 2, $vector, $($compN),+);

// // FIXME: Real is needed only for inversion here.
// impl<N: Real> SquareMatrix for $t<N> {
//     type Vector = $vector<N>;

//     #[inline]
//     fn diagonal(&self) -> Self::Coordinates {
//         $vector::new(self.m11, self.m22)
//     }

//     #[inline]
//     fn determinant(&self) -> Self::Field {
//         self.m11 * self.m22 - self.m21 * self.m12
//     }

//     #[inline]
//     fn try_inverse(&self) -> Option<Self> {
//         let mut res = *self;
//         if res.try_inverse_mut() {
//             Some(res)
//         }
//         else {
//             None
//         }
//     }

//     #[inline]
//     fn try_inverse_mut(&mut self) -> bool {
//         let determinant = ::determinant(self);

//         if relative_eq!(&determinant, &::zero()) {
//             false
//         }
//         else {
//             *self = Matrix2::new(
//                 self.m22 / determinant , -self.m12 / determinant,
//                 -self.m21 / determinant, self.m11 / determinant);

//             true
//         }
//     }

//     #[inline]
//     fn transpose_mut(&mut self) {
//         mem::swap(&mut self.m12, &mut self.m21)
//     }
// }

//  ident, 3, $vector: ident, $($compN: ident),+) => {
// matrix_group_approx_impl!(common: $t, 3, $vector, $($compN),+);

// // FIXME: Real is needed only for inversion here.
// impl<N: Real> SquareMatrix for $t<N> {
//     type Vector = $vector<N>;

//     #[inline]
//     fn diagonal(&self) -> Self::Coordinates {
//         $vector::new(self.m11, self.m22, self.m33)
//     }

//     #[inline]
//     fn determinant(&self) -> Self::Field {
//         let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
//         let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
//         let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

//         self.m11 * minor_m12_m23 - self.m12 * minor_m11_m23 + self.m13 * minor_m11_m22
//     }

//     #[inline]
//     fn try_inverse(&self) -> Option<Self> {
//         let mut res = *self;
//         if res.try_inverse_mut() {
//             Some(res)
//         }
//         else {
//             None
//         }
//     }

//     #[inline]
//     fn try_inverse_mut(&mut self) -> bool {
//         let minor_m12_m23 = self.m22 * self.m33 - self.m32 * self.m23;
//         let minor_m11_m23 = self.m21 * self.m33 - self.m31 * self.m23;
//         let minor_m11_m22 = self.m21 * self.m32 - self.m31 * self.m22;

//         let determinant = self.m11 * minor_m12_m23 -
//                           self.m12 * minor_m11_m23 +
//                           self.m13 * minor_m11_m22;

//         if relative_eq!(&determinant, &::zero()) {
//             false
//         }
//         else {
//             *self = Matrix3::new(
//                 (minor_m12_m23 / determinant),
//                 ((self.m13 * self.m32 - self.m33 * self.m12) / determinant),
//                 ((self.m12 * self.m23 - self.m22 * self.m13) / determinant),

//                 (-minor_m11_m23 / determinant),
//                 ((self.m11 * self.m33 - self.m31 * self.m13) / determinant),
//                 ((self.m13 * self.m21 - self.m23 * self.m11) / determinant),

//                 (minor_m11_m22  / determinant),
//                 ((self.m12 * self.m31 - self.m32 * self.m11) / determinant),
//                 ((self.m11 * self.m22 - self.m21 * self.m12) / determinant)
//                 );

//             true
//         }
//     }

//     #[inline]
//     fn transpose_mut(&mut self) {
//         mem::swap(&mut self.m12, &mut self.m21);
//         mem::swap(&mut self.m13, &mut self.m31);
//         mem::swap(&mut self.m23, &mut self.m32);
//     }
// }

//  ident, $dimension: expr, $vector: ident, $($compN: ident),+) => {
// matrix_group_approx_impl!(common: $t, $dimension, $vector, $($compN),+);

// // FIXME: Real is needed only for inversion here.
// impl<N: Real> SquareMatrix for $t<N> {
//     type Vector = $vector<N>;

//     #[inline]
//     fn diagonal(&self) -> Self::Coordinates {
//         let mut diagonal: $vector<N> = ::zero();

//         for i in 0 .. $dimension {
//             unsafe { diagonal.unsafe_set(i, self.get_unchecked(i, i)) }
//         }

//         diagonal
//     }

//     #[inline]
//     fn determinant(&self) -> Self::Field {
//         // FIXME: extremely naive implementation.
//         let mut det = ::zero();

//         for icol in 0 .. $dimension {
//             let e = unsafe { self.unsafe_at((0, icol)) };

//             if e != ::zero() {
//                 let minor_mat = self.delete_row_column(0, icol);
//                 let minor     = minor_mat.determinant();

//                 if icol % 2 == 0 {
//                     det += minor;
//                 }
//                 else {
//                     det -= minor;
//                 }
//             }
//         }

//         det
//     }

//     #[inline]
//     fn try_inverse(&self) -> Option<Self> {
//         let mut res = *self;
//         if res.try_inverse_mut() {
//             Some(res)
//         }
//         else {
//             None
//         }
//     }

//     #[inline]
//     fn try_inverse_mut(&mut self) -> bool {
//         let mut res: $t<N> = ::one();

//         // Inversion using Gauss-Jordan elimination
//         for k in 0 .. $dimension {
//             // search a non-zero value on the k-th column
//             // FIXME: would it be worth it to spend some more time searching for the
//             // max instead?

//             let mut n0 = k; // index of a non-zero entry

//             while n0 != $dimension {
//                 if self[(n0, k)] != ::zero() {
//                     break;
//                 }

//                 n0 = n0 + 1;
//             }

//             if n0 == $dimension {
//                 return false
//             }

//             // swap pivot line
//             if n0 != k {
//                 for j in 0 .. $dimension {
//                     self.swap((n0, j), (k, j));
//                     res.swap((n0, j), (k, j));
//                 }
//             }

//             let pivot = self[(k, k)];

//             for j in k .. $dimension {
//                 let selfval = self[(k, j)] / pivot;
//                 self[(k, j)] = selfval;
//             }

//             for j in 0 .. $dimension {
//                 let resval = res[(k, j)] / pivot;
//                 res[(k, j)] = resval;
//             }

//             for l in 0 .. $dimension {
//                 if l != k {
//                     let normalizer = self[(l, k)];

//                     for j in k .. $dimension {
//                         let selfval = self[(l, j)] - self[(k, j)] * normalizer;
//                         self[(l, j)] = selfval;
//                     }

//                     for j in 0 .. $dimension {
//                         let resval  = res[(l, j)] - res[(k, j)] * normalizer;
//                         res[(l, j)] = resval;
//                     }
//                 }
//             }
//         }

//         *self = res;

//         true
//     }

//     #[inline]
//     fn transpose_mut(&mut self) {
//         for i in 1 .. $dimension {
//             for j in 0 .. i {
//                 self.swap((i, j), (j, i))
//             }
//         }
//     }




/*
 *
 * Ordering
 *
 */
impl<N, R: Dim, C: Dim, S> MeetSemilattice for Matrix<N, R, C, S>
    where N: Scalar + MeetSemilattice,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a.meet(&b))
    }
}

impl<N, R: Dim, C: Dim, S> JoinSemilattice for Matrix<N, R, C, S>
    where N: Scalar + JoinSemilattice,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a.join(&b))
    }
}


impl<N, R: Dim, C: Dim, S> Lattice for Matrix<N, R, C, S>
    where N: Scalar + Lattice,
          S: OwnedStorage<N, R, C>,
          S::Alloc: OwnedAllocator<N, R, C, S> {
    #[inline]
    fn meet_join(&self, other: &Self) -> (Self, Self) {
        let shape = self.data.shape();
        assert!(shape == other.data.shape(), "Matrix meet/join error: mismatched dimensions.");

        let mut mres = unsafe { Self::new_uninitialized_generic(shape.0, shape.1) };
        let mut jres = unsafe { Self::new_uninitialized_generic(shape.0, shape.1) };

        for i in 0 .. shape.0.value() * shape.1.value() {
            unsafe {
                let mj = self.data.get_unchecked_linear(i).meet_join(other.data.get_unchecked_linear(i));
                *mres.data.get_unchecked_linear_mut(i) = mj.0;
                *jres.data.get_unchecked_linear_mut(i) = mj.1;
            }
        }

        (mres, jres)
    }
}
