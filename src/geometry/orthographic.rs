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

/// A 3D orthographic projection stored as an homogeneous 4x4 matrix.
#[derive(Debug, Clone, Copy)] // FIXME: Hash
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct OrthographicBase<N: Scalar, S: Storage<N, U4, U4>> {
    matrix: SquareMatrix<N, U4, S>
}

/// A 3D orthographic projection stored as a static homogeneous 4x4 matrix.
pub type Orthographic3<N> = OrthographicBase<N, MatrixArray<N, U4, U4>>;

impl<N, S> Eq for OrthographicBase<N, S>
    where N: Scalar + Eq,
          S: Storage<N, U4, U4> { }

impl<N: Scalar, S: Storage<N, U4, U4>> PartialEq for OrthographicBase<N, S> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

impl<N, S> OrthographicBase<N, S>
    where N: Real,
          S: OwnedStorage<N, U4, U4>,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    /// Creates a new orthographic projection matrix.
    #[inline]
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Self {
        assert!(left < right, "The left corner must be farther than the right corner.");
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        assert!(znear < zfar, "The far plane must be farther than the near plane.");

        let matrix = SquareMatrix::<N, U4, S>::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_left_and_right(left, right);
        res.set_bottom_and_top(bottom, top);
        res.set_znear_and_zfar(znear, zfar);

        res
    }

    /// Wraps the given matrix to interpret it as a 3D orthographic matrix.
    ///
    /// It is not checked whether or not the given matrix actually represents an orthographic
    /// projection.
    #[inline]
    pub fn from_matrix_unchecked(matrix: SquareMatrix<N, U4, S>) -> Self {
        OrthographicBase {
            matrix: matrix
        }
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    #[inline]
    pub fn from_fov(aspect: N, vfov: N, znear: N, zfar: N) -> Self {
        assert!(znear < zfar, "The far plane must be farther than the near plane.");
        assert!(!relative_eq!(aspect, N::zero()), "The apsect ratio must not be zero.");

        let half: N = ::convert(0.5);
        let width  = zfar * (vfov * half).tan();
        let height = width / aspect;

        Self::new(-width * half, width * half, -height * half, height * half, znear, zfar)
    }
}

impl<N: Real, S: Storage<N, U4, U4>> OrthographicBase<N, S> {
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

        let inv_m11 = N::one() / self.matrix[(0, 0)];
        let inv_m22 = N::one() / self.matrix[(1, 1)];
        let inv_m33 = N::one() / self.matrix[(2, 2)];

        res[(0, 0)] = inv_m11;
        res[(1, 1)] = inv_m22;
        res[(2, 2)] = inv_m33;

        res[(0, 3)] = -self.matrix[(0, 3)] * inv_m11; 
        res[(1, 3)] = -self.matrix[(1, 3)] * inv_m22; 
        res[(2, 3)] = -self.matrix[(2, 3)] * inv_m33; 

        res
    }

    /// Computes the corresponding homogeneous matrix.
    #[inline]
    pub fn to_homogeneous(&self) -> OwnedSquareMatrix<N, U4, S::Alloc> {
        self.matrix.clone_owned()
    }

    /// The smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn left(&self) -> N {
        (-N::one() - self.matrix[(0, 3)]) / self.matrix[(0, 0)]
    }

    /// The largest x-coordinate of the view cuboid.
    #[inline]
    pub fn right(&self) -> N {
        (N::one() - self.matrix[(0, 3)]) / self.matrix[(0, 0)]
    }

    /// The smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn bottom(&self) -> N {
        (-N::one() - self.matrix[(1, 3)]) / self.matrix[(1, 1)]
    }

    /// The largest y-coordinate of the view cuboid.
    #[inline]
    pub fn top(&self) -> N {
        (N::one() - self.matrix[(1, 3)]) / self.matrix[(1, 1)]
    }

    /// The near plane offset of the view cuboid.
    #[inline]
    pub fn znear(&self) -> N {
        (N::one() + self.matrix[(2, 3)]) / self.matrix[(2, 2)]
    }

    /// The far plane offset of the view cuboid.
    #[inline]
    pub fn zfar(&self) -> N {
        (-N::one() + self.matrix[(2, 3)]) / self.matrix[(2, 2)]
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a point. Faster than matrix multiplication.
    #[inline]
    pub fn project_point<SB>(&self, p: &PointBase<N, U3, SB>) -> OwnedPoint<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        OwnedPoint::<N, U3, SB::Alloc>::new(
            self.matrix[(0, 0)] * p[0] + self.matrix[(0, 3)],
            self.matrix[(1, 1)] * p[1] + self.matrix[(1, 3)],
            self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)]
        )
    }

    /// Un-projects a point. Faster than multiplication by the underlying matrix inverse.
    #[inline]
    pub fn unproject_point<SB>(&self, p: &PointBase<N, U3, SB>) -> OwnedPoint<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        OwnedPoint::<N, U3, SB::Alloc>::new(
            (p[0] - self.matrix[(0, 3)]) / self.matrix[(0, 0)],
            (p[1] - self.matrix[(1, 3)]) / self.matrix[(1, 1)],
            (p[2] - self.matrix[(2, 3)]) / self.matrix[(2, 2)]
        )
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    #[inline]
    pub fn project_vector<SB>(&self, p: &ColumnVector<N, U3, SB>) -> OwnedColumnVector<N, U3, SB::Alloc>
        where SB: Storage<N, U3, U1> {

        OwnedColumnVector::<N, U3, SB::Alloc>::new(
            self.matrix[(0, 0)] * p[0],
            self.matrix[(1, 1)] * p[1],
            self.matrix[(2, 2)] * p[2]
        )
    }
}

impl<N: Real, S: StorageMut<N, U4, U4>> OrthographicBase<N, S> {
    /// Sets the smallest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_left(&mut self, left: N) {
        let right = self.right();
        self.set_left_and_right(left, right);
    }

    /// Sets the largest x-coordinate of the view cuboid.
    #[inline]
    pub fn set_right(&mut self, right: N) {
        let left = self.left();
        self.set_left_and_right(left, right);
    }

    /// Sets the smallest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_bottom(&mut self, bottom: N) {
        let top = self.top();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the largest y-coordinate of the view cuboid.
    #[inline]
    pub fn set_top(&mut self, top: N) {
        let bottom = self.bottom();
        self.set_bottom_and_top(bottom, top);
    }

    /// Sets the near plane offset of the view cuboid.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the far plane offset of the view cuboid.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Sets the view cuboid coordinates along the `x` axis.
    #[inline]
    pub fn set_left_and_right(&mut self, left: N, right: N) {
        assert!(left < right, "The left corner must be farther than the right corner.");
        self.matrix[(0, 0)] = ::convert::<_, N>(2.0) / (right - left);
        self.matrix[(0, 3)] = -(right + left) / (right - left);
    }

    /// Sets the view cuboid coordinates along the `y` axis.
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: N, top: N) {
        assert!(bottom < top, "The top corner must be higher than the bottom corner.");
        self.matrix[(1, 1)] = ::convert::<_, N>(2.0) / (top - bottom);
        self.matrix[(1, 3)] = -(top + bottom) / (top - bottom);
    }

    /// Sets the near and far plane offsets of the view cuboid.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        assert!(!relative_eq!(zfar - znear, N::zero()), "The near-plane and far-plane must not be superimposed.");
        self.matrix[(2, 2)] = -::convert::<_, N>(2.0) / (zfar - znear);
        self.matrix[(2, 3)] = -(zfar + znear) / (zfar - znear);
    }
}

impl<N, S> Rand for OrthographicBase<N, S>
    where N: Real + Rand,
          S: OwnedStorage<N, U4, U4>,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    fn rand<R: Rng>(r: &mut R) -> Self {
        let left   = Rand::rand(r);
        let right  = helper::reject_rand(r, |x: &N| *x > left);
        let bottom = Rand::rand(r);
        let top    = helper::reject_rand(r, |x: &N| *x > bottom);
        let znear  = Rand::rand(r);
        let zfar   = helper::reject_rand(r, |x: &N| *x > znear);

        Self::new(left, right, bottom, top, znear, zfar)
    }
}

#[cfg(feature="arbitrary")]
impl<N, S> Arbitrary for OrthographicBase<N, S>
    where N: Real + Arbitrary,
          S: OwnedStorage<N, U4, U4> + Send,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let left   = Arbitrary::arbitrary(g);
        let right  = helper::reject(g, |x: &N| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top    = helper::reject(g, |x: &N| *x > bottom);
        let znear  = Arbitrary::arbitrary(g);
        let zfar   = helper::reject(g, |x: &N| *x > znear);

        Self::new(left, right, bottom, top, znear, zfar)
    }
}
