#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::marker::PhantomData;
use std::mem;

use simba::scalar::RealField;

use crate::base::dimension::U3;
use crate::base::helper;
use crate::base::storage::Storage;
use crate::base::{Matrix4, Scalar, Vector, Vector3};

use crate::geometry::{Point3, Projective3};

/// Normalized device coordinates systems
pub trait System {}

/// OpenGL
/// Note that we will probably want to go with more generic names that encode the handedness and depth range
/// Please consider all names as placeholders
#[derive(Default)]
pub struct OpenGL {}

/// OpenGL is a System
impl System for OpenGL {}

/// Vulkan is also a System
#[derive(Default)]
pub struct Vulkan {}

/// Vulkan is also a System
impl System for Vulkan {}

/// Note that it is possible to alias systems (OpenGL and Vulkan would be aliases of generic systems)
//pub type OpenGL = RHS_NO;
//pub type OpenGL = LHS_ZO;
pub type VulkanX = Vulkan;

/// A 3D perspective projection stored as a homogeneous 4x4 matrix.
/// Perspective3 is now generic over System
/// Note that :
/// - S was put in first place to avoid having to specify N when specifying S
/// - S defaults to OpenGL and rust requires N to have a default too (only trailing type parameters can have defaults)
/// But unfortunately default type parameters have some limitations and don't fully work as one would expect.
/// See cg.rs for the issue at hand.
/// And [RFC 0213-defaulted-type-params](@ https://github.com/rust-lang/rfcs/blob/master/text/0213-defaulted-type-params.md) for more details on the issue.
pub struct Perspective3<S: System = OpenGL, N: Scalar = f32> {
    matrix: Matrix4<N>,
    /// See [PhantomData](https://doc.rust-lang.org/std/marker/struct.PhantomData.html#unused-type-parameters)
    /// TODO add above comment to all other PhantomData uses.
    phantom: PhantomData<S>,
}

/// It is possible to avoid the breaking changes by renaming Perspective3 to, lets say, Perspective3S.
/// And then alias Perspective3<N> to Perspective3S<OpenGL, N> and, voilà, PerspectiveS<N> is still a thing and no code breaks.
/// But it is ugly and if you want to use another NDC you end up with the Perspective3S<Vulkan>.
//pub type Perspective3<N> = Perspective3S<OpenGL, N>;

// Dummy alias to demonstrate that this approach works (see cg.rs)
pub type Perspective3OpenGL<N> = Perspective3<OpenGL, N>;

impl<S: System, N: RealField> Copy for Perspective3<S, N> {}

impl<S: System, N: RealField> Clone for Perspective3<S, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix.clone())
    }
}

impl<S: System, N: RealField> fmt::Debug for Perspective3<S, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.matrix.fmt(f)
    }
}

impl<S: System, N: RealField> PartialEq for Perspective3<S, N> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "serde-serialize")]
impl<N: RealField + Serialize> Serialize for Perspective3<S, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, N: RealField + Deserialize<'a>> Deserialize<'a> for Perspective3<S, N> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = Matrix4::<N>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<S: System, N: RealField> Perspective3<S, N> {
    /// Wraps the given matrix to interpret it as a 3D perspective matrix.
    ///
    /// It is not checked whether or not the given matrix actually represents a perspective
    /// projection.
    #[inline]
    pub fn from_matrix_unchecked(matrix: Matrix4<N>) -> Self {
        Self {
            matrix,
            phantom: PhantomData,
        }
    }

    /// Retrieves the inverse of the underlying homogeneous matrix.
    #[inline]
    pub fn inverse(&self) -> Matrix4<N> {
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
    pub fn into_inner(self) -> Matrix4<N> {
        self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    /// Deprecated: Use [Perspective3::into_inner] instead.
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> Matrix4<N> {
        self.matrix
    }

    /// Gets the `width / height` aspect ratio of the view frustum.
    #[inline]
    pub fn aspect(&self) -> N {
        self.matrix[(1, 1)] / self.matrix[(0, 0)]
    }

    /// Gets the y field of view of the view frustum.
    #[inline]
    pub fn fovy(&self) -> N {
        (N::one() / self.matrix[(1, 1)]).atan() * crate::convert(2.0)
    }

    /// Gets the near plane offset of the view frustum.
    #[inline]
    pub fn znear(&self) -> N {
        self.matrix[(2, 3)] / self.matrix[(2, 2)]
    }

    /// Gets the far plane offset of the view frustum.
    #[inline]
    pub fn zfar(&self) -> N {
        self.matrix[(2, 3)] / (N::one() + self.matrix[(2, 2)])
    }

    // TODO: add a method to retrieve znear and zfar simultaneously?

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a point. Faster than matrix multiplication.
    #[inline]
    pub fn project_point(&self, p: &Point3<N>) -> Point3<N> {
        let inverse_denom = -N::one() / p[2];
        Point3::new(
            self.matrix[(0, 0)] * p[0] * inverse_denom,
            self.matrix[(1, 1)] * p[1] * inverse_denom,
            (self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)]) * inverse_denom,
        )
    }

    /// Un-projects a point. Faster than multiplication by the matrix inverse.
    #[inline]
    pub fn unproject_point(&self, p: &Point3<N>) -> Point3<N> {
        let inverse_denom = self.matrix[(2, 3)] / (p[2] + self.matrix[(2, 2)]);

        Point3::new(
            p[0] * inverse_denom / self.matrix[(0, 0)],
            p[1] * inverse_denom / self.matrix[(1, 1)],
            -inverse_denom,
        )
    }

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    #[inline]
    pub fn project_vector<SB>(&self, p: &Vector<N, U3, SB>) -> Vector3<N>
    where
        SB: Storage<N, U3>,
    {
        let inverse_denom = -N::one() / p[2];
        Vector3::new(
            self.matrix[(0, 0)] * p[0] * inverse_denom,
            self.matrix[(1, 1)] * p[1] * inverse_denom,
            self.matrix[(2, 2)],
        )
    }

    /// Updates this perspective matrix with a new `width / height` aspect ratio of the view
    /// frustum.
    #[inline]
    pub fn set_aspect(&mut self, aspect: N) {
        assert!(
            !relative_eq!(aspect, N::zero()),
            "The aspect ratio must not be zero."
        );
        self.matrix[(0, 0)] = self.matrix[(1, 1)] / aspect;
    }
}

// OpenGL specialization
// For now not all required functions are specialized for sake of illustration.
// Specializating the other functions should be trivial.

impl<N: RealField> Perspective3<OpenGL, N> {
    /// Implementation note: new() must be specialized because it calls other specialized functions.
    pub fn new(aspect: N, fovy: N, znear: N, zfar: N) -> Self {
        assert!(
            !relative_eq!(zfar - znear, N::zero()),
            "The near-plane and far-plane must not be superimposed."
        );
        assert!(
            !relative_eq!(aspect, N::zero()),
            "The aspect ratio must not be zero."
        );

        let matrix = Matrix4::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);

        res.matrix[(3, 3)] = N::zero();
        res.matrix[(3, 2)] = -N::one();

        res
    }
    /// Updates this perspective with a new y field of view of the view frustum.
    #[inline]
    pub fn set_fovy(&mut self, fovy: N) {
        let old_m22 = self.matrix[(1, 1)];
        self.matrix[(1, 1)] = N::one() / (fovy / crate::convert(2.0)).tan();
        self.matrix[(0, 0)] = self.matrix[(0, 0)] * (self.matrix[(1, 1)] / old_m22);
    }

    /// Updates this perspective matrix with a new near plane offset of the view frustum.
    /// Implementation note: set_znear() must be specialized because it calls other specialized functions.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with a new far plane offset of the view frustum.
    /// Implementation note: set_zfar() must be specialized because it calls other specialized functions.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with new near and far plane offsets of the view frustum.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        self.matrix[(2, 2)] = (zfar + znear) / (znear - zfar);
        self.matrix[(2, 3)] = zfar * znear * crate::convert(2.0) / (znear - zfar);
    }
}

// Vulkan specialization

impl<N: RealField> Perspective3<Vulkan, N> {
    /// Implementation note: new() must be specialized because it calls other specialized functions.
    pub fn new(aspect: N, fovy: N, znear: N, zfar: N) -> Self {
        assert!(
            !relative_eq!(zfar - znear, N::zero()),
            "The near-plane and far-plane must not be superimposed."
        );
        assert!(
            !relative_eq!(aspect, N::zero()),
            "The aspect ratio must not be zero."
        );

        let matrix = Matrix4::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);

        res.matrix[(3, 3)] = N::zero();
        res.matrix[(3, 2)] = -N::one();

        res
    }

    /// Updates this perspective with a new y field of view of the view frustum.
    #[inline]
    pub fn set_fovy(&mut self, fovy: N) {
        let old_m22 = self.matrix[(1, 1)];
        let f = N::one() / (fovy / crate::convert(2.0)).tan();
        self.matrix[(1, 1)] = -f;
        self.matrix[(0, 0)] *= f / old_m22;
    }

    /// Updates this perspective matrix with a new near plane offset of the view frustum.
    /// Implementation note: set_znear() must be specialized because it calls other specialized functions.
    #[inline]
    pub fn set_znear(&mut self, znear: N) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with a new far plane offset of the view frustum.
    /// Implementation note: set_zfar() must be specialized because it calls other specialized functions.
    #[inline]
    pub fn set_zfar(&mut self, zfar: N) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with new near and far plane offsets of the view frustum.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: N, zfar: N) {
        self.matrix[(2, 2)] = -zfar / (zfar - znear);
        self.matrix[(2, 3)] = -(zfar * znear) / (zfar - znear);
    }
}

impl<N: RealField> Distribution<Perspective3<OpenGL, N>> for Standard
where
    Standard: Distribution<N>,
{
    fn sample<'a, R: Rng + ?Sized>(&self, r: &'a mut R) -> Perspective3<OpenGL, N> {
        let znear = r.gen();
        let zfar = helper::reject_rand(r, |&x: &N| !(x - znear).is_zero());
        let aspect = helper::reject_rand(r, |&x: &N| !x.is_zero());

        Perspective3::<OpenGL, N>::new(aspect, r.gen(), znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<S: System, N: RealField + Arbitrary> Arbitrary for Perspective3<S, N> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let znear = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |&x: &N| !(x - znear).is_zero());
        let aspect = helper::reject(g, |&x: &N| !x.is_zero());

        Self::new(aspect, Arbitrary::arbitrary(g), znear, zfar)
    }
}

impl<S: System, N: RealField> From<Perspective3<S, N>> for Matrix4<N> {
    #[inline]
    fn from(pers: Perspective3<S, N>) -> Self {
        pers.into_inner()
    }
}
