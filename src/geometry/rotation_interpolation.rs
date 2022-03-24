use crate::{RealField, Rotation, Rotation2, Rotation3, SimdRealField, UnitComplex, UnitQuaternion};
use crate::{Const, U1, DimSub, DimDiff, Storage, ArrayStorage, Allocator, DefaultAllocator};
use crate::SMatrix;

/// # Interpolation
impl<T: SimdRealField> Rotation2<T> {
    /// Spherical linear interpolation between two rotation matrices.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::geometry::Rotation2;
    ///
    /// let rot1 = Rotation2::new(std::f32::consts::FRAC_PI_4);
    /// let rot2 = Rotation2::new(-std::f32::consts::PI);
    ///
    /// let rot = rot1.slerp(&rot2, 1.0 / 3.0);
    ///
    /// assert_relative_eq!(rot.angle(), std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self
    where
        T::Element: SimdRealField,
    {
        let c1 = UnitComplex::from(self.clone());
        let c2 = UnitComplex::from(other.clone());
        c1.slerp(&c2, t).into()
    }
}

impl<T: SimdRealField> Rotation3<T> {
    /// Spherical linear interpolation between two rotation matrices.
    ///
    /// Panics if the angle between both rotations is 180 degrees (in which case the interpolation
    /// is not well-defined). Use `.try_slerp` instead to avoid the panic.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::geometry::Rotation3;
    ///
    /// let q1 = Rotation3::from_euler_angles(std::f32::consts::FRAC_PI_4, 0.0, 0.0);
    /// let q2 = Rotation3::from_euler_angles(-std::f32::consts::PI, 0.0, 0.0);
    ///
    /// let q = q1.slerp(&q2, 1.0 / 3.0);
    ///
    /// assert_eq!(q.euler_angles(), (std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: T) -> Self
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.slerp(&q2, t).into()
    }

    /// Computes the spherical linear interpolation between two rotation matrices or returns `None`
    /// if both rotations are approximately 180 degrees apart (in which case the interpolation is
    /// not well-defined).
    ///
    /// # Arguments
    /// * `self`: the first rotation to interpolate from.
    /// * `other`: the second rotation to interpolate toward.
    /// * `t`: the interpolation parameter. Should be between 0 and 1.
    /// * `epsilon`: the value below which the sinus of the angle separating both rotations
    /// must be to return `None`.
    #[inline]
    #[must_use]
    pub fn try_slerp(&self, other: &Self, t: T, epsilon: T) -> Option<Self>
    where
        T: RealField,
    {
        let q1 = UnitQuaternion::from(self.clone());
        let q2 = UnitQuaternion::from(other.clone());
        q1.try_slerp(&q2, t, epsilon).map(|q| q.into())
    }
}

impl<T:SimdRealField, const D: usize> Rotation<T,D> {

    #[warn(missing_docs)]
    pub fn basis_rot(i:usize, j:usize, angle:T) -> Self {

        let mut m = SMatrix::identity();
        if i==j { return Self::from_matrix_unchecked(m); }

        let (s,c) = angle.simd_sin_cos();
        m[(i,i)] = c.clone();
        m[(i,j)] = -s.clone();
        m[(j,i)] = s;
        m[(j,j)] = c;

        Self::from_matrix_unchecked(m)

    }
}

impl<T:RealField, const D: usize> Rotation<T,D> where
    Const<D>: DimSub<U1>,
    ArrayStorage<T,D,D>: Storage<T,Const<D>,Const<D>>,
    DefaultAllocator: Allocator<T,Const<D>,Const<D>,Buffer=ArrayStorage<T,D,D>> + Allocator<T,Const<D>> +
        Allocator<T,Const<D>,DimDiff<Const<D>,U1>> +
        Allocator<T,DimDiff<Const<D>,U1>>
{

    #[warn(missing_docs)]
    pub fn general_slerp(&self, other: &Self, t:T) -> Self {
        self * (self/other).general_pow(t)
    }

    #[warn(missing_docs)]
    pub fn general_pow(self, t:T) -> Self {
        if D<=1 { return self; }

        println!("r:{}", self);

        //taking the (real) schur form is guaranteed to produce a block-diagonal matrix
        //where each block is either a 1 (if there's no rotation in that axis) or a 2x2
        //rotation matrix in a particular plane
        let schur = self.into_inner().schur();
        let (q, mut d) = schur.unpack();

        println!("q:{}d:{:.3}", q, d);

        //go down the diagonal and pow every block
        for i in 0..(D-1) {

            //we've found a 2x2 block!
            //NOTE: the impl of the schur decomposition always sets the inferior diagonal to 0
            if !d[(i+1,i)].is_zero() {

                println!("{}", i);

                //convert to a complex num and take the arg()
                let (c, s) = (d[(i,i)].clone(), d[(i+1,i)].clone());
                let angle = s.atan2(c);

                println!("{}", angle);

                //scale the arg and exponentiate back
                let angle2 = angle * t.clone();
                let (s2, c2) = angle2.sin_cos();

                //convert back into a rot block
                d[(i,  i  )] =  c2.clone();
                d[(i,  i+1)] = -s2.clone();
                d[(i+1,i  )] =  s2;
                d[(i+1,i+1)] =  c2;

            }

        }
        println!("d:{:.3}", d);

        let qt = q.transpose(); //avoids an extra clone

        Self::from_matrix_unchecked(q * d * qt)

    }

}

#[cfg(test)]
mod tests {

    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 2E-10;

    #[test]
    fn rot_pow() {

        let r1 = Rotation2::new(0.0);
        let r2 = Rotation2::new(PI/4.0);
        let r3 = Rotation2::new(PI/2.0);

        assert_relative_eq!(r1.general_pow(0.5), r1, epsilon=EPS);
        assert_relative_eq!(r3.general_pow(0.5), r2, epsilon=EPS);

    }

    #[test]
    fn basis_rot() {

        const D:usize = 4;

        let basis_blades = |n| (0..n).flat_map(
            move |i| (i..n).map(move |j| (i,j))
        ).filter(|(i,j)| i!=j);

        for (i1,j1) in basis_blades(D) {

            for (i2,j2) in basis_blades(D) {

                if i1==i2 || j1==j2 || i1==j2 || j1==i2 { continue; }

                let r1 = Rotation::<_,D>::basis_rot(i1,j1,PI/4.0) *
                    Rotation::<_,D>::basis_rot(i2,j2,PI/4.0);
                let r2 = Rotation::<_,D>::basis_rot(i1,j1,PI/2.0) *
                    Rotation::<_,D>::basis_rot(i2,j2,PI/2.0);

                println!("{}{}",r1,r2);

                assert_relative_eq!(r2.general_pow(0.5), r1, epsilon=EPS);

            }

        }

    }


}
