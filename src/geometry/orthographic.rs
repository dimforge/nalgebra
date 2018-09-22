#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use std::fmt;
use std::mem;

use alga::general::Real;

use base::dimension::U3;
use base::helper;
use base::storage::Storage;
use base::{Matrix4, Vector, Vector3};

use geometry::{Projective3, Point3};

/// A 3D orthographic projection stored as an homogeneous 4x4 matrix.
pub struct Orthographic3<N: Real> {
    matrix: Matrix4<N>,
}

impl<N: Real> Copy for Orthographic3<N> {}

impl<N: Real> Clone for Orthographic3<N> {
    #[inline]
    fn clone(&self) -> Self {
        Orthographic3::from_matrix_unchecked(self.matrix.clone())
    }
}

impl<N: Real> fmt::Debug for Orthographic3<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.matrix.fmt(f)
    }
}

impl<N: Real> PartialEq for Orthographic3<N> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: Real + Serialize> Serialize for Orthographic3<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: Real + Deserialize<'a>> Deserialize<'a> for Orthographic3<N> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = Matrix4::<N>::deserialize(deserializer)?;

        Ok(Orthographic3::from_matrix_unchecked(matrix))
    }
}

impl<N: Real> Orthographic3<N> {
    /// Creates a new orthographic projection matrix.
    #[inline]
    pub fn new(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Self {
        assert!(
            left < right,
            "The left corner must be farther than the right corner."
        );
        assert!(
            bottom < top,
            "The top corner must be higher than the bottom corner."
        );
        assert!(
            znear < zfar,
            "The far plane must be farther than the near plane."
        );

        let matrix = Matrix4::<N>::identity();
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
    pub fn from_matrix_unchecked(matrix: Matrix4<N>) -> Self {
        Orthographic3 { matrix: matrix }
    }

    /// Creates a new orthographic projection matrix from an aspect ratio and the vertical field of view.
    #[inline]
    pub fn from_fov(aspect: N, vfov: N, znear: N, zfar: N) -> Self {
        assert!(
            znear < zfar,
            "The far plane must be farther than the near plane."
        );
        assert!(
            !relative_eq!(aspect, N::zero()),
            "The apsect ratio must not be zero."
        );

        let half: N = ::convert(0.5);
        let width = zfar * (vfov * half).tan();
        let height = width / aspect;

        Self::new(
            -width * half,
            width * half,
            -height * half,
            height * half,
            znear,
            zfar,
        )
    }

    /// Retrieves the inverse of the underlying homogeneous matrix.
    #[inline]
    pub fn inverse(&self) -> Matrix4<N> {
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
    pub fn to_homogeneous(&self) -> Matrix4<N> {
        self.matrix.clone_owned()
    }

    /// A reference to the underlying homogeneous transformation matrix.
    #[inline]
    pub fn as_matrix(&self) -> &Matrix4<N> {
        &self.matrix
    }

    /// A reference to this transformation seen as a `Projective3`.
    #[inline]
    pub fn as_projective(&self) -> &Projective3<N> {
        unsafe { mem::transmute(self) }
    }

    /// This transformation seen as a `Projective3`.
    #[inline]
    pub fn to_projective(&self) -> Projective3<N> {
        Projective3::from_matrix_unchecked(self.matrix)
    }

    /// Retrieves the underlying homogeneous matrix.
    #[inline]
    pub fn unwrap(self) -> Matrix4<N> {
        self.matrix
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
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        Point3::new(
            self.matrix[(0, 0)] * p[0] + self.matrix[(0, 3)],
            self.matrix[(1, 1)] * p[1] + self.matrix[(1, 3)],
            self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)],
        )
    }

    /// Un-projects a point. Faster than multiplication by the underlying matrix inverse.
    #[inline]
    pub fn unproject_point(&self, p: &Point3<N>) -> Point3<N> {
        Point3::new(
            (p[0] - self.matrix[(0, 3)]) / self.matrix[(0, 0)],
            (p[1] - self.matrix[(1, 3)]) / self.matrix[(1, 1)],
            (p[2] - self.matrix[(2, 3)]) / self.matrix[(2, 2)],
        )
    }

    // FIXME: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    #[inline]
    pub fn project_vector<SB>(&self, p: &Vector<N, U3, SB>) -> Vector3<N>
    where
        SB: Storage<N, U3>,
    {
        Vector3::new(
            self.matrix[(0, 0)] * p[0],
            self.matrix[(1, 1)] * p[1],
            self.matrix[(2, 2)] * p[2],
        )
    }

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
        assert!(
            left < right,
            "The left corner must be farther than the right corner."
        );
        self.matrix[(0, 0)] = ::convert::<_, N>(2.0) / (right - left);
        self.matrix[(0, 3)] = -(right + left) / (right - left);
    }

    /// Sets the view cuboid coordinates along the `y` axis.
    #[inline]
    pub fn set_bottom_and_top(&mut self, bottom: N, top: N) {
        assert!(
            bottom < top,
            "The top corner must be higher than the bottom corner."
        );
        self.matrix[(1, 1)] = ::convert::<_, N>(2.0) / (top - bottom);
        self.matrix[(1, 3)] = -(top + bottom) / (top - bottom);
    }

    /// Sets the near and far plane offsets of the view cuboid.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        assert!(
            !relative_eq!(zfar - znear, N::zero()),
            "The near-plane and far-plane must not be superimposed."
        );
        self.matrix[(2, 2)] = -::convert::<_, N>(2.0) / (zfar - znear);
        self.matrix[(2, 3)] = -(zfar + znear) / (zfar - znear);
    }
}

impl<N: Real> Distribution<Orthographic3<N>> for Standard
where
    Standard: Distribution<N>,
{
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Orthographic3<N> {
        let left = r.gen();
        let right = helper::reject_rand(r, |x: &N| *x > left);
        let bottom = r.gen();
        let top = helper::reject_rand(r, |x: &N| *x > bottom);
        let znear = r.gen();
        let zfar = helper::reject_rand(r, |x: &N| *x > znear);

        Orthographic3::new(left, right, bottom, top, znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<N: Real + Arbitrary> Arbitrary for Orthographic3<N>
where
    Matrix4<N>: Send,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let left = Arbitrary::arbitrary(g);
        let right = helper::reject(g, |x: &N| *x > left);
        let bottom = Arbitrary::arbitrary(g);
        let top = helper::reject(g, |x: &N| *x > bottom);
        let znear = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |x: &N| *x > znear);

        Self::new(left, right, bottom, top, znear, zfar)
    }
}
