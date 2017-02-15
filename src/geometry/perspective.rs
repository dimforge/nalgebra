#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};
use rand::{Rand, Rng};

use alga::general::Real;

use core::{Scalar, SquareMatrix, OwnedSquareMatrix, ColumnVector, OwnedColumnVector, MatrixArray};
use core::dimension::{U1, U3, U4};
use core::storage::{OwnedStorage, Storage, StorageMut};
use core::allocator::OwnedAllocator;
use core::helper;

use geometry::{PointBase, OwnedPoint};

/// A 3D perspective projection stored as an homogeneous 4x4 matrix.
#[derive(Debug, Clone, Copy)] // FIXME: Hash
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct PerspectiveBase<N: Scalar, S: Storage<N, U4, U4>> {
    matrix: SquareMatrix<N, U4, S>
}

/// A 3D perspective projection stored as a static homogeneous 4x4 matrix.
pub type Perspective3<N> = PerspectiveBase<N, MatrixArray<N, U4, U4>>;

impl<N, S> Eq for PerspectiveBase<N, S>
    where N: Scalar + Eq,
          S: Storage<N, U4, U4> { }

impl<N, S> PartialEq for PerspectiveBase<N, S>
    where N: Scalar,
          S: Storage<N, U4, U4> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, S> PerspectiveBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U4>,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    /// Creates a new perspective matrix from the aspect ratio, y field of view, and near/far planes.
    pub fn new(aspect: N, fovy: N, znear: N, zfar: N) -> Self {
        assert!(!relative_eq!(zfar - znear, N::zero()), "The near-plane and far-plane must not be superimposed.");
        assert!(!relative_eq!(aspect, N::zero()), "The apsect ratio must not be zero.");

        let matrix = SquareMatrix::<N, U4, S>::identity();
        let mut res = PerspectiveBase::from_matrix_unchecked(matrix);

        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);

        res.matrix[(3, 3)] = N::zero();
        res.matrix[(3, 2)] = -N::one();

        res
    }


    /// Wraps the given matrix to interpret it as a 3D perspective matrix.
    ///
    /// It is not checked whether or not the given matrix actually represents an orthographic
    /// projection.
    #[inline]
    pub fn from_matrix_unchecked(matrix: SquareMatrix<N, U4, S>) -> Self {
        PerspectiveBase {
            matrix: matrix
        }
    }
}

impl<N, S> PerspectiveBase<N, S>
    where N: Real,
          S: Storage<N, U4, U4> {

    /// A reference to the underlying homogeneous transformation matrix.
    #[inline]
    pub fn as_matrix(&self) -> &SquareMatrix<N, U4, S> {
        &self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    #[inline]
    pub fn unwrap(self) -> SquareMatrix<N, U4, S> {
        self.matrix
    }

    /// Retrieves the inverse of the underlying homogeneous matrix.
    #[inline]
    pub fn inverse(&self) -> OwnedSquareMatrix<N, U4, S::Alloc> {
        let mut res = self.to_homogeneous();

        res[(0, 0)] = N::one() / self.matrix[(0, 0)];
        res[(1, 1)] = N::one() / self.matrix[(1, 1)];
        res[(2, 2)] = N::zero();

        let m23 = self.matrix[(2, 3)];
        let m32 = self.matrix[(3, 2)];

        res[(2, 3)] = N::one() / m32;
        res[(3, 2)] = N::one() / m23;
        res[(3, 3)] = -self.matrix[(2, 2)] / (m23 * m32);

        res
    }

    /// Computes the corresponding homogeneous matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, U4, S::Alloc> {
        self.matrix.clone_owned()
    }

    /// Gets the `width / height` aspect ratio of the view frustrum.
    #[inline]
    pub fn aspect(&self) -> N {
        self.matrix[(1, 1)] / self.matrix[(0, 0)]
    }

    /// Gets the y field of view of the view frustrum.
    #[inline]
    pub fn fovy(&self) -> N {
        (N::one() / self.matrix[(1, 1)]).atan() * ::convert(2.0)
    }

    /// Gets the near plane offset of the view frustrum.
    #[inline]
    pub fn znear(&self) -> N {
        let ratio = (-self.matrix[(2, 2)] + N::one()) / (-self.matrix[(2, 2)] - N::one());

        self.matrix[(2, 3)] / (ratio * ::convert(2.0)) - self.matrix[(2, 3)] / ::convert(2.0)
    }

    /// Gets the far plane offset of the view frustrum.
    #[inline]
    pub fn zfar(&self) -> N {
        let ratio = (-self.matrix[(2, 2)] + N::one()) / (-self.matrix[(2, 2)] - N::one());

        (self.matrix[(2, 3)] - ratio * self.matrix[(2, 3)]) / ::convert(2.0)
    }

    // FIXME: add a method to retrieve znear and zfar simultaneously?



    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a point. Faster than matrix multiplication.
    #[inline]
    pub fn project_point<SB>(&self, p: &PointBase<N, U3, SB>) -> OwnedPoint<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        let inverse_denom = -N::one() / p[2];
        OwnedPoint::<N, U3, SB::Alloc>::new(
             self.matrix[(0, 0)] * p[0] * inverse_denom,
             self.matrix[(1, 1)] * p[1] * inverse_denom,
            (self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)]) * inverse_denom
        )
    }

    /// Un-projects a point. Faster than multiplication by the matrix inverse.
    #[inline]
    pub fn unproject_point<SB>(&self, p: &PointBase<N, U3, SB>) -> OwnedPoint<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        let inverse_denom = self.matrix[(2, 3)] / (p[2] + self.matrix[(2, 2)]);

        OwnedPoint::<N, U3, SB::Alloc>::new(
            p[0] * inverse_denom / self.matrix[(0, 0)],
            p[1] * inverse_denom / self.matrix[(1, 1)],
            -inverse_denom
        )
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    #[inline]
    pub fn project_vector<SB>(&self, p: &ColumnVector<N, U3, SB>) -> OwnedColumnVector<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        let inverse_denom = -N::one() / p[2];
        OwnedColumnVector::<N, U3, SB::Alloc>::new(
            self.matrix[(0, 0)] * p[0] * inverse_denom,
            self.matrix[(1, 1)] * p[1] * inverse_denom,
            self.matrix[(2, 2)]
        )
    }
}


impl<N, S> PerspectiveBase<N, S>
    where N: Real,
          S: StorageMut<N, U4, U4> {
    /// Updates this perspective matrix with a new `width / height` aspect ratio of the view
    /// frustrum.
    #[inline]
    pub fn set_aspect(&mut self, aspect: N) {
        assert!(!relative_eq!(aspect, N::zero()), "The aspect ratio must not be zero.");
        self.matrix[(0, 0)] = self.matrix[(1, 1)] / aspect;
    }

    /// Updates this perspective with a new y field of view of the view frustrum.
    #[inline]
    pub fn set_fovy(&mut self, fovy: N) {
        let old_m22  = self.matrix[(1, 1)];
        self.matrix[(1, 1)] = N::one() / (fovy / ::convert(2.0)).tan();
        self.matrix[(0, 0)] = self.matrix[(0, 0)] * (self.matrix[(1, 1)] / old_m22);
    }

    /// Updates this perspective matrix with a new near plane offset of the view frustrum.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with a new far plane offset of the view frustrum.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with new near and far plane offsets of the view frustrum.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        self.matrix[(2, 2)] = (zfar + znear) / (znear - zfar);
        self.matrix[(2, 3)] = zfar * znear * ::convert(2.0) / (znear - zfar);
    }
}

impl<N, S> Rand for PerspectiveBase<N, S>
    where N: Real + Rand,
          S: OwnedStorage<N, U4, U4>,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    fn rand<R: Rng>(r: &mut R) -> Self {
        let znear  = Rand::rand(r);
        let zfar   = helper::reject_rand(r, |&x: &N| !(x - znear).is_zero());
        let aspect = helper::reject_rand(r, |&x: &N| !x.is_zero());

        Self::new(aspect, Rand::rand(r), znear, zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N, S> Arbitrary for PerspectiveBase<N, S>
    where N: Real + Arbitrary,
          S: OwnedStorage<N, U4, U4> + Send,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let znear  = Arbitrary::arbitrary(g);
        let zfar   = helper::reject(g, |&x: &N| !(x - znear).is_zero());
        let aspect = helper::reject(g, |&x: &N| !x.is_zero());

        Self::new(aspect, Arbitrary::arbitrary(g), znear, zfar)
    }
}
