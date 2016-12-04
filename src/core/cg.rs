/*
 *
 * Computer-graphics specific implementations.
 * Currently, it is mostly implemented for homogeneous matrices in 2- and 3-space.
 *
 */

use num::One;

use core::{Scalar, SquareMatrix, OwnedSquareMatrix, ColumnVector, Unit};
use core::dimension::{DimName, DimNameSub, DimNameDiff, U1, U2, U3, U4};
use core::storage::{Storage, StorageMut, OwnedStorage};
use core::allocator::{Allocator, OwnedAllocator};
use geometry::{PointBase, OrthographicBase, PerspectiveBase, IsometryBase, OwnedRotation, OwnedPoint};

use alga::general::{Real, Field};
use alga::linear::Transformation;


impl<N, D: DimName, S> SquareMatrix<N, D, S>
    where N: Scalar + Field,
          S: OwnedStorage<N, D, D>,
          S::Alloc: OwnedAllocator<N, D, D, S> {
    /// Creates a new homogeneous matrix that applies the same scaling factor on each dimension.
    #[inline]
    pub fn new_scaling(scaling: N) -> Self {
        let mut res = Self::from_diagonal_element(scaling);
        res[(D::dim(), D::dim())] = N::one();

        res
    }

    /// Creates a new homogeneous matrix that applies a distinct scaling factor for each dimension.
    #[inline]
    pub fn new_nonuniform_scaling<SB>(scaling: &ColumnVector<N, DimNameDiff<D, U1>, SB>) -> Self
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1> {
        let mut res = Self::one();
        for i in 0 .. scaling.len() {
            res[(i, i)] = scaling[i];
        }

        res
    }

    /// Creates a new homogeneous matrix that applies a pure translation.
    #[inline]
    pub fn new_translation<SB>(translation: &ColumnVector<N, DimNameDiff<D, U1>, SB>) -> Self
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, U1> {
        let mut res = Self::one();
        res.fixed_slice_mut::<DimNameDiff<D, U1>, U1>(0, D::dim()).copy_from(translation);

        res
    }
}

impl<N, S> SquareMatrix<N, U3, S>
    where N: Real,
          S: OwnedStorage<N, U3, U3>,
          S::Alloc: OwnedAllocator<N, U3, U3, S> {
    /// Builds a 2 dimensional homogeneous rotation matrix from an angle in radian.
    #[inline]
    pub fn new_rotation(angle: N) -> Self
        where S::Alloc: Allocator<N, U2, U2> {
        OwnedRotation::<N, U2, S::Alloc>::new(angle).to_homogeneous()
    }
}

impl<N, S> SquareMatrix<N, U4, S>
    where N: Real,
          S: OwnedStorage<N, U4, U4>,
          S::Alloc: OwnedAllocator<N, U4, U4, S> {

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    /// 
    /// Returns the identity matrix if the given argument is zero.
    #[inline]
    pub fn new_rotation<SB>(axisangle: ColumnVector<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U1>,
              S::Alloc: Allocator<N, U3, U3> {
        OwnedRotation::<N, U3, S::Alloc>::new(axisangle).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    /// 
    /// Returns the identity matrix if the given argument is zero.
    #[inline]
    pub fn new_rotation_wrt_point<SB>(axisangle: ColumnVector<N, U3, SB>, pt: OwnedPoint<N, U3, S::Alloc>) -> Self
        where SB: Storage<N, U3, U1>,
              S::Alloc: Allocator<N, U3, U3> +
                        Allocator<N, U3, U1> +
                        Allocator<N, U1, U3> {
        let rot = OwnedRotation::<N, U3, S::Alloc>::from_scaled_axis(axisangle);
        IsometryBase::rotation_wrt_point(rot, pt).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and an angle (multiplied together).
    /// 
    /// Returns the identity matrix if the given argument is zero.
    /// This is identical to `Self::new_rotation`.
    #[inline]
    pub fn from_scaled_axis<SB>(axisangle: ColumnVector<N, U3, SB>) -> Self
        where SB: Storage<N, U3, U1>,
              S::Alloc: Allocator<N, U3, U3> {
        OwnedRotation::<N, U3, S::Alloc>::from_scaled_axis(axisangle).to_homogeneous()
    }

    /// Creates a new rotation from Euler angles.
    ///
    /// The primitive rotations are applied in order: 1 roll − 2 pitch − 3 yaw.
    pub fn from_euler_angles(roll: N, pitch: N, yaw: N) -> Self
        where S::Alloc: Allocator<N, U3, U3> {
        OwnedRotation::<N, U3, S::Alloc>::from_euler_angles(roll, pitch, yaw).to_homogeneous()
    }

    /// Builds a 3D homogeneous rotation matrix from an axis and a rotation angle.
    pub fn from_axis_angle<SB>(axis: &Unit<ColumnVector<N, U3, SB>>, angle: N) -> Self
        where SB: Storage<N, U3, U1>,
              S::Alloc: Allocator<N, U3, U3> {
        OwnedRotation::<N, U3, S::Alloc>::from_axis_angle(axis, angle).to_homogeneous()
    }

    /// Creates a new homogeneous matrix for an orthographic projection.
    #[inline]
    pub fn new_orthographic(left: N, right: N, bottom: N, top: N, znear: N, zfar: N) -> Self {
        OrthographicBase::new(left, right, bottom, top, znear, zfar).unwrap()
    }

    /// Creates a new homogeneous matrix for a perspective projection.
    #[inline]
    pub fn new_perspective(aspect: N, fovy: N, znear: N, zfar: N) -> Self {
        PerspectiveBase::new(aspect, fovy, znear, zfar).unwrap()
    }

    /// Creates an isometry that corresponds to the local frame of an observer standing at the
    /// point `eye` and looking toward `target`.
    ///
    /// It maps the view direction `target - eye` to the positive `z` axis and the origin to the
    /// `eye`.
    #[inline]
    pub fn new_observer_frame<SB>(eye:    &PointBase<N, U3, SB>,
                                  target: &PointBase<N, U3, SB>,
                                  up:     &ColumnVector<N, U3, SB>)
                                  -> Self
        where SB: OwnedStorage<N, U3, U1, Alloc = S::Alloc>,
              SB::Alloc: OwnedAllocator<N, U3, U1, SB> +
                         Allocator<N, U1, U3> +
                         Allocator<N, U3, U3> {
        IsometryBase::<N, U3, SB, OwnedRotation<N, U3, SB::Alloc>>
                    ::new_observer_frame(eye, target, up).to_homogeneous()
    }

    /// Builds a right-handed look-at view matrix.
    #[inline]
    pub fn look_at_rh<SB>(eye:    &PointBase<N, U3, SB>,
                          target: &PointBase<N, U3, SB>,
                          up:     &ColumnVector<N, U3, SB>)
                          -> Self
        where SB: OwnedStorage<N, U3, U1, Alloc = S::Alloc>,
              SB::Alloc: OwnedAllocator<N, U3, U1, SB> +
                         Allocator<N, U1, U3> +
                         Allocator<N, U3, U3> {
        IsometryBase::<N, U3, SB, OwnedRotation<N, U3, SB::Alloc>>
                    ::look_at_rh(eye, target, up).to_homogeneous()
    }

    /// Builds a left-handed look-at view matrix.
    #[inline]
    pub fn look_at_lh<SB>(eye:    &PointBase<N, U3, SB>,
                          target: &PointBase<N, U3, SB>,
                          up:     &ColumnVector<N, U3, SB>)
                          -> Self
        where SB: OwnedStorage<N, U3, U1, Alloc = S::Alloc>,
              SB::Alloc: OwnedAllocator<N, U3, U1, SB> +
                         Allocator<N, U1, U3> +
                         Allocator<N, U3, U3> {
        IsometryBase::<N, U3, SB, OwnedRotation<N, U3, SB::Alloc>>
                    ::look_at_lh(eye, target, up).to_homogeneous()
    }
}


impl<N, D: DimName, S> SquareMatrix<N, D, S>
    where N: Scalar + Field,
          S: Storage<N, D, D> {

    /// Computes the transformation equal to `self` followed by an uniform scaling factor.
    #[inline]
    pub fn append_scaling(&self, scaling: N) -> OwnedSquareMatrix<N, D, S::Alloc>
        where D: DimNameSub<U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, D> {
        let mut res = self.clone_owned();
        res.append_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to an uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_scaling(&self, scaling: N) -> OwnedSquareMatrix<N, D, S::Alloc>
        where D: DimNameSub<U1>,
              S::Alloc: Allocator<N, D, DimNameDiff<D, U1>> {
        let mut res = self.clone_owned();
        res.prepend_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a non-uniform scaling factor.
    #[inline]
    pub fn append_nonuniform_scaling<SB>(&self, scaling: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        -> OwnedSquareMatrix<N, D, S::Alloc>
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, U1, D> {
        let mut res = self.clone_owned();
        res.append_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to a non-uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_nonuniform_scaling<SB>(&self, scaling: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        -> OwnedSquareMatrix<N, D, S::Alloc>
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, D, U1> {
        let mut res = self.clone_owned();
        res.prepend_nonuniform_scaling_mut(scaling);
        res
    }

    /// Computes the transformation equal to `self` followed by a translation.
    #[inline]
    pub fn append_translation<SB>(&self, shift: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        -> OwnedSquareMatrix<N, D, S::Alloc>
        where D: DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, U1> {
        let mut res = self.clone_owned();
        res.append_translation_mut(shift);
        res
    }

    /// Computes the transformation equal to a translation followed by `self`.
    #[inline]
    pub fn prepend_translation<SB>(&self, shift: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        -> OwnedSquareMatrix<N, D, S::Alloc>
        where D: DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, U1> +
                        Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>> +
                        Allocator<N, U1, DimNameDiff<D, U1>> {
        let mut res = self.clone_owned();
        res.prepend_translation_mut(shift);
        res
    }
}

impl<N, D: DimName, S> SquareMatrix<N, D, S>
    where N: Scalar + Field,
          S: StorageMut<N, D, D> {

    /// Computes in-place the transformation equal to `self` followed by an uniform scaling factor.
    #[inline]
    pub fn append_scaling_mut(&mut self, scaling: N)
        where D: DimNameSub<U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, D> {
        let mut to_scale = self.fixed_rows_mut::<DimNameDiff<D, U1>>(0);
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to an uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_scaling_mut(&mut self, scaling: N)
        where D: DimNameSub<U1>,
              S::Alloc: Allocator<N, D, DimNameDiff<D, U1>> {
        let mut to_scale = self.fixed_columns_mut::<DimNameDiff<D, U1>>(0);
        to_scale *= scaling;
    }

    /// Computes in-place the transformation equal to `self` followed by a non-uniform scaling factor.
    #[inline]
    pub fn append_nonuniform_scaling_mut<SB>(&mut self, scaling: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, U1, D> {
        for i in 0 .. scaling.len() {
            let mut to_scale = self.fixed_rows_mut::<U1>(i);
            to_scale *= scaling[i];
        }
    }

    /// Computes in-place the transformation equal to a non-uniform scaling factor followed by `self`.
    #[inline]
    pub fn prepend_nonuniform_scaling_mut<SB>(&mut self, scaling: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        where D:  DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, D, U1> {
        for i in 0 .. scaling.len() {
            let mut to_scale = self.fixed_columns_mut::<U1>(i);
            to_scale *= scaling[i];
        }
    }

    /// Computes the transformation equal to `self` followed by a translation.
    #[inline]
    pub fn append_translation_mut<SB>(&mut self, shift: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        where D: DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, U1> {
        for i in 0 .. D::dim() {
            for j in 0 .. D::dim() - 1 {
                self[(j, i)] += shift[i] * self[(D::dim(), j)];
            }
        }
    }

    /// Computes the transformation equal to a translation followed by `self`.
    #[inline]
    pub fn prepend_translation_mut<SB>(&mut self, shift: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        where D: DimNameSub<U1>,
              SB: Storage<N, DimNameDiff<D, U1>, U1>,
              S::Alloc: Allocator<N, DimNameDiff<D, U1>, U1> +
                        Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>> +
                        Allocator<N, U1, DimNameDiff<D, U1>> {
        let scale = self.fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim(), 0).tr_dot(&shift);
        let post_translation = self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0) * shift;

        self[(D::dim(), D::dim())] += scale;

        let mut translation = self.fixed_slice_mut::<DimNameDiff<D, U1>, U1>(0, D::dim());
        translation += post_translation;
    }
}


impl<N, D, SA, SB> Transformation<PointBase<N, DimNameDiff<D, U1>, SB>> for SquareMatrix<N, D, SA>
    where N:  Real,
          D:  DimNameSub<U1>,
          SA: OwnedStorage<N, D, D>,
          SB: OwnedStorage<N, DimNameDiff<D, U1>, U1, Alloc = SA::Alloc>,
          SA::Alloc: OwnedAllocator<N, D, D, SA> +
                     Allocator<N, DimNameDiff<D, U1>, DimNameDiff<D, U1>> +
                     Allocator<N, DimNameDiff<D, U1>, U1> +
                     Allocator<N, U1, DimNameDiff<D, U1>>,
          SB::Alloc: OwnedAllocator<N, DimNameDiff<D, U1>, U1, SB> {
    #[inline]
    fn transform_vector(&self, v: &ColumnVector<N, DimNameDiff<D, U1>, SB>)
        -> ColumnVector<N, DimNameDiff<D, U1>, SB> {
        let transform  = self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0);
        let normalizer = self.fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim(), 0);
        let n = normalizer.tr_dot(&v);

        if !n.is_zero() {
            return transform * (v / n);
        }

        transform * v
    }

    #[inline]
    fn transform_point(&self, pt: &PointBase<N, DimNameDiff<D, U1>, SB>)
        -> PointBase<N, DimNameDiff<D, U1>, SB> {
        let transform   = self.fixed_slice::<DimNameDiff<D, U1>, DimNameDiff<D, U1>>(0, 0);
        let translation = self.fixed_slice::<DimNameDiff<D, U1>, U1>(0, D::dim());
        let normalizer  = self.fixed_slice::<U1, DimNameDiff<D, U1>>(D::dim(), 0);
        let n = normalizer.tr_dot(&pt.coords) + unsafe { *self.get_unchecked(D::dim(), D::dim()) };

        if !n.is_zero() {
            return transform * (pt / n) + translation;
        }

        transform * pt + translation
    }
}
