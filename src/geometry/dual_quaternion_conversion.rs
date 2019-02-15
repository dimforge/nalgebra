// use num::Zero;

// use alga::general::{Real, SubsetOf, SupersetOf};
// use alga::linear::Rotation as AlgaRotation;

// #[cfg(feature = "mint")]
// use mint;

// use base::dimension::U3;
// use base::{Matrix3, Matrix4, Vector4};
// use geometry::{
//     Isometry, Point3, DualQuaternion, Rotation, Rotation3, Similarity, SuperTCategoryOf, TAffine,
//     Transform, Translation, UnitDualQuaternion,
// };

// /*
//  * This file provides the following conversions:
//  * =============================================
//  *
//  * DualQuaternion     -> DualQuaternion
//  * UnitDualQuaternion -> UnitDualQuaternion
//  * UnitDualQuaternion -> Rotation<U3>
//  * UnitDualQuaternion -> Isometry<U3>
//  * UnitDualQuaternion -> Similarity<U3>
//  * UnitDualQuaternion -> Transform<U3>
//  * UnitDualQuaternion -> Matrix<U4> (homogeneous)
//  *
//  * mint::DualQuaternion <-> DualQuaternion
//  * UnitDualQuaternion -> mint::DualQuaternion
//  *
//  * NOTE:
//  * UnitDualQuaternion -> DualQuaternion is already provided by: Unit<T> -> T
//  */

// impl<N1, N2> SubsetOf<DualQuaternion<N2>> for DualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
// {
//     #[inline]
//     fn to_superset(&self) -> DualQuaternion<N2> {
//         DualQuaternion::from(self.coords.to_superset())
//     }

//     #[inline]
//     fn is_in_subset(q: &DualQuaternion<N2>) -> bool {
//         ::is_convertible::<_, Vector4<N1>>(&q.coords)
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(q: &DualQuaternion<N2>) -> Self {
//         Self {
//             coords: q.coords.to_subset_unchecked(),
//         }
//     }
// }

// impl<N1, N2> SubsetOf<UnitDualQuaternion<N2>> for UnitDualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
// {
//     #[inline]
//     fn to_superset(&self) -> UnitDualQuaternion<N2> {
//         UnitDualQuaternion::new_unchecked(self.as_ref().to_superset())
//     }

//     #[inline]
//     fn is_in_subset(uq: &UnitDualQuaternion<N2>) -> bool {
//         ::is_convertible::<_, DualQuaternion<N1>>(uq.as_ref())
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(uq: &UnitDualQuaternion<N2>) -> Self {
//         Self::new_unchecked(::convert_ref_unchecked(uq.as_ref()))
//     }
// }

// impl<N1, N2> SubsetOf<Rotation<N2, U3>> for UnitDualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
// {
//     #[inline]
//     fn to_superset(&self) -> Rotation3<N2> {
//         let q: UnitDualQuaternion<N2> = self.to_superset();
//         q.to_rotation_matrix()
//     }

//     #[inline]
//     fn is_in_subset(rot: &Rotation3<N2>) -> bool {
//         ::is_convertible::<_, Rotation3<N1>>(rot)
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(rot: &Rotation3<N2>) -> Self {
//         let q = UnitDualQuaternion::<N2>::from_rotation_matrix(rot);
//         ::convert_unchecked(q)
//     }
// }

// impl<N1, N2, R> SubsetOf<Isometry<N2, U3, R>> for UnitDualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
//     R: AlgaRotation<Point3<N2>> + SupersetOf<UnitDualQuaternion<N1>>,
// {
//     #[inline]
//     fn to_superset(&self) -> Isometry<N2, U3, R> {
//         Isometry::from_parts(Translation::identity(), ::convert_ref(self))
//     }

//     #[inline]
//     fn is_in_subset(iso: &Isometry<N2, U3, R>) -> bool {
//         iso.translation.vector.is_zero()
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(iso: &Isometry<N2, U3, R>) -> Self {
//         ::convert_ref_unchecked(&iso.rotation)
//     }
// }

// impl<N1, N2, R> SubsetOf<Similarity<N2, U3, R>> for UnitDualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
//     R: AlgaRotation<Point3<N2>> + SupersetOf<UnitDualQuaternion<N1>>,
// {
//     #[inline]
//     fn to_superset(&self) -> Similarity<N2, U3, R> {
//         Similarity::from_isometry(::convert_ref(self), N2::one())
//     }

//     #[inline]
//     fn is_in_subset(sim: &Similarity<N2, U3, R>) -> bool {
//         sim.isometry.translation.vector.is_zero() && sim.scaling() == N2::one()
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(sim: &Similarity<N2, U3, R>) -> Self {
//         ::convert_ref_unchecked(&sim.isometry)
//     }
// }

// impl<N1, N2, C> SubsetOf<Transform<N2, U3, C>> for UnitDualQuaternion<N1>
// where
//     N1: Real,
//     N2: Real + SupersetOf<N1>,
//     C: SuperTCategoryOf<TAffine>,
// {
//     #[inline]
//     fn to_superset(&self) -> Transform<N2, U3, C> {
//         Transform::from_matrix_unchecked(self.to_homogeneous().to_superset())
//     }

//     #[inline]
//     fn is_in_subset(t: &Transform<N2, U3, C>) -> bool {
//         <Self as SubsetOf<_>>::is_in_subset(t.matrix())
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(t: &Transform<N2, U3, C>) -> Self {
//         Self::from_superset_unchecked(t.matrix())
//     }
// }

// impl<N1: Real, N2: Real + SupersetOf<N1>> SubsetOf<Matrix4<N2>> for UnitDualQuaternion<N1> {
//     #[inline]
//     fn to_superset(&self) -> Matrix4<N2> {
//         self.to_homogeneous().to_superset()
//     }

//     #[inline]
//     fn is_in_subset(m: &Matrix4<N2>) -> bool {
//         ::is_convertible::<_, Rotation3<N1>>(m)
//     }

//     #[inline]
//     unsafe fn from_superset_unchecked(m: &Matrix4<N2>) -> Self {
//         let rot: Rotation3<N1> = ::convert_ref_unchecked(m);
//         Self::from_rotation_matrix(&rot)
//     }
// }

// #[cfg(feature = "mint")]
// impl<N: Real> From<mint::DualQuaternion<N>> for DualQuaternion<N> {
//     fn from(q: mint::DualQuaternion<N>) -> Self {
//         DualQuaternion::new(q.s, q.v.x, q.v.y, q.v.z)
//     }
// }

// #[cfg(feature = "mint")]
// impl<N: Real> Into<mint::DualQuaternion<N>> for DualQuaternion<N> {
//     fn into(self) -> mint::DualQuaternion<N> {
//         mint::DualQuaternion {
//             v: mint::Vector3 {
//                 x: self[0],
//                 y: self[1],
//                 z: self[2],
//             },
//             s: self[3],
//         }
//     }
// }

// #[cfg(feature = "mint")]
// impl<N: Real> Into<mint::DualQuaternion<N>> for UnitDualQuaternion<N> {
//     fn into(self) -> mint::DualQuaternion<N> {
//         mint::DualQuaternion {
//             v: mint::Vector3 {
//                 x: self[0],
//                 y: self[1],
//                 z: self[2],
//             },
//             s: self[3],
//         }
//     }
// }

// impl<N: Real> From<UnitDualQuaternion<N>> for Matrix4<N> {
//     #[inline]
//     fn from(q: UnitDualQuaternion<N>) -> Matrix4<N> {
//         panic!("adam")
//     }
// }

// impl<N: Real> From<UnitDualQuaternion<N>> for Matrix3<N> {
//     #[inline]
//     fn from(q: UnitDualQuaternion<N>) -> Matrix3<N> {
//         q.to_rotation_matrix().into_inner()
//     }
// }

// impl<N: Real> From<Vector4<N>> for DualQuaternion<N> {
//     #[inline]
//     fn from(coords: Vector4<N>) -> Self {
//         DualQuaternion { coords }
//     }
// }
