use std::ops::{Sub, Mul, Neg};
use num::{Zero, One};
use traits::structure::{Cast, Row, Basis, BaseFloat};
use traits::geometry::{Norm, Cross, CrossMatrix, RotationTo, UniformSphereSample};
use structs::vector::{Vector1, Vector2, Vector3, Vector4};
use structs::matrix::Matrix3;
use structs::rotation::{Rotation2, Rotation3};

impl<N: BaseFloat> RotationTo for Vector2<N> {
    type AngleType = N;
    type DeltaRotationType = Rotation2<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        ::cross(self, other).x.atan2(::dot(self, other))
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rotation2<N> {
        Rotation2::new(Vector1::new(self.angle_to(other)))
    }
}

impl<N: BaseFloat> RotationTo for Vector3<N> {
    type AngleType = N;
    type DeltaRotationType = Rotation3<N>;

    #[inline]
    fn angle_to(&self, other: &Self) -> N {
        ::cross(self, other).norm().atan2(::dot(self, other))
    }

    #[inline]
    fn rotation_to(&self, other: &Self) -> Rotation3<N> {
        let mut axis = ::cross(self, other);
        let norm = axis.normalize_mut();

        if ::is_zero(&norm) {
            ::one()
        }
        else {
            let axis_angle = axis * norm.atan2(::dot(self, other));

            Rotation3::new(axis_angle)
        }
    }
}

impl<N: Copy + Mul<N, Output = N> + Sub<N, Output = N>> Cross for Vector2<N> {
    type CrossProductType = Vector1<N>;

    #[inline]
    fn cross(&self, other: &Vector2<N>) -> Vector1<N> {
        Vector1::new(self.x * other.y - self.y * other.x)
    }
}

// FIXME: instead of returning a Vector2, define a Matrix2x1 matrix?
impl<N: Neg<Output = N> + Copy> CrossMatrix<Vector2<N>> for Vector2<N> {
    #[inline]
    fn cross_matrix(&self) -> Vector2<N> {
        Vector2::new(-self.y, self.x)
    }
}

impl<N: Copy + Mul<N, Output = N> + Sub<N, Output = N>> Cross for Vector3<N> {
    type CrossProductType = Vector3<N>;

    #[inline]
    fn cross(&self, other: &Vector3<N>) -> Vector3<N> {
        Vector3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }
}

impl<N: Neg<Output = N> + Zero + Copy> CrossMatrix<Matrix3<N>> for Vector3<N> {
    #[inline]
    fn cross_matrix(&self) -> Matrix3<N> {
        Matrix3::new(
            ::zero(), -self.z,  self.y,
            self.z,   ::zero(), -self.x,
            -self.y,  self.x,   ::zero()
        )
    }
}

// FIXME: implement this for all other vectors
impl<N: Copy> Row<Vector1<N>> for Vector2<N> {
    #[inline]
    fn nrows(&self) -> usize {
        2
    }

    #[inline]
    fn row(&self, i: usize) -> Vector1<N> {
        match i {
            0 => Vector1::new(self.x),
            1 => Vector1::new(self.y),
            _ => panic!(format!("Index out of range: 2d vectors do not have {} rows. ", i))
        }
    }

    #[inline]
    fn set_row(&mut self, i: usize, r: Vector1<N>) {
        match i {
            0 => self.x = r.x,
            1 => self.y = r.x,
            _ => panic!(format!("Index out of range: 2d vectors do not have {} rows.", i))

        }
    }
}

impl<N: One> Basis for Vector1<N> {
    #[inline]
    fn canonical_basis<F: FnMut(Vector1<N>) -> bool>(mut f: F) {
        f(Vector1::new(::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis<F: FnMut(Vector1<N>) -> bool>(_: &Vector1<N>, _: F) { }

    #[inline]
    fn canonical_basis_element(i: usize) -> Option<Vector1<N>> {
        if i == 0 {
            Some(Vector1::new(::one()))
        }
        else {
            None
        }
    }
}

impl<N: Copy + One + Zero + Neg<Output = N>> Basis for Vector2<N> {
    #[inline]
    fn canonical_basis<F: FnMut(Vector2<N>) -> bool>(mut f: F) {
        if !f(Vector2::new(::one(), ::zero())) { return };
        f(Vector2::new(::zero(), ::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis<F: FnMut(Vector2<N>) -> bool>(n: &Vector2<N>, mut f: F) {
        f(Vector2::new(-n.y, n.x));
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Option<Vector2<N>> {
        if i == 0 {
            Some(Vector2::new(::one(), ::zero()))
        }
        else if i == 1 {
            Some(Vector2::new(::zero(), ::one()))
        }
        else {
            None
        }
    }
}

impl<N: BaseFloat> Basis for Vector3<N> {
    #[inline]
    fn canonical_basis<F: FnMut(Vector3<N>) -> bool>(mut f: F) {
        if !f(Vector3::new(::one(), ::zero(), ::zero())) { return };
        if !f(Vector3::new(::zero(), ::one(), ::zero())) { return };
        f(Vector3::new(::zero(), ::zero(), ::one()));
    }

    #[inline]
    fn orthonormal_subspace_basis<F: FnMut(Vector3<N>) -> bool>(n: &Vector3<N>, mut f: F) {
        let a = 
            if n.x.abs() > n.y.abs() {
                ::normalize(&Vector3::new(n.z, ::zero(), -n.x))
            }
            else {
                ::normalize(&Vector3::new(::zero(), -n.z, n.y))
            };

        if !f(Cross::cross(&a, n)) { return };
        f(a);
    }

    #[inline]
    fn canonical_basis_element(i: usize) -> Option<Vector3<N>> {
        if i == 0 {
            Some(Vector3::new(::one(), ::zero(), ::zero()))
        }
        else if i == 1 {
            Some(Vector3::new(::zero(), ::one(), ::zero()))
        }
        else if i == 2 {
            Some(Vector3::new(::zero(), ::zero(), ::one()))
        }
        else {
            None
        }
    }
}

// FIXME: this bad: this fixes definitly the number of samples…
static SAMPLES_2_F64: [Vector2<f64>; 21] = [
    Vector2 { x: 1.0,         y: 0.0         },
    Vector2 { x: 0.95557281,  y: 0.29475517  },
    Vector2 { x: 0.82623877,  y: 0.56332006  },
    Vector2 { x: 0.6234898,   y: 0.78183148  },
    Vector2 { x: 0.36534102,  y: 0.93087375  },
    Vector2 { x: 0.07473009,  y: 0.9972038   },
    Vector2 { x: -0.22252093, y: 0.97492791  },
    Vector2 { x: -0.5,        y: 0.8660254   },
    Vector2 { x: -0.73305187, y: 0.68017274  },
    Vector2 { x: -0.90096887, y: 0.43388374  },
    Vector2 { x: -0.98883083, y: 0.14904227  },
    Vector2 { x: -0.98883083, y: -0.14904227 },
    Vector2 { x: -0.90096887, y: -0.43388374 },
    Vector2 { x: -0.73305187, y: -0.68017274 },
    Vector2 { x: -0.5,        y: -0.8660254  },
    Vector2 { x: -0.22252093, y: -0.97492791 },
    Vector2 { x: 0.07473009,  y: -0.9972038  },
    Vector2 { x: 0.36534102,  y: -0.93087375 },
    Vector2 { x: 0.6234898,   y: -0.78183148 },
    Vector2 { x: 0.82623877,  y: -0.56332006 },
    Vector2 { x: 0.95557281,  y: -0.29475517 },
];

// Those vectors come from bullet 3d
static SAMPLES_3_F64: [Vector3<f64>; 42] = [
    Vector3 { x: 0.000000 , y: -0.000000, z: -1.000000 },
    Vector3 { x: 0.723608 , y: -0.525725, z: -0.447219 },
    Vector3 { x: -0.276388, y: -0.850649, z: -0.447219 },
    Vector3 { x: -0.894426, y: -0.000000, z: -0.447216 },
    Vector3 { x: -0.276388, y: 0.850649 , z: -0.447220 },
    Vector3 { x: 0.723608 , y: 0.525725 , z: -0.447219 },
    Vector3 { x: 0.276388 , y: -0.850649, z: 0.447220 },
    Vector3 { x: -0.723608, y: -0.525725, z: 0.447219 },
    Vector3 { x: -0.723608, y: 0.525725 , z: 0.447219 },
    Vector3 { x: 0.276388 , y: 0.850649 , z: 0.447219 },
    Vector3 { x: 0.894426 , y: 0.000000 , z: 0.447216 },
    Vector3 { x: -0.000000, y: 0.000000 , z: 1.000000 },
    Vector3 { x: 0.425323 , y: -0.309011, z: -0.850654 },
    Vector3 { x: -0.162456, y: -0.499995, z: -0.850654 },
    Vector3 { x: 0.262869 , y: -0.809012, z: -0.525738 },
    Vector3 { x: 0.425323 , y: 0.309011 , z: -0.850654 },
    Vector3 { x: 0.850648 , y: -0.000000, z: -0.525736 },
    Vector3 { x: -0.525730, y: -0.000000, z: -0.850652 },
    Vector3 { x: -0.688190, y: -0.499997, z: -0.525736 },
    Vector3 { x: -0.162456, y: 0.499995 , z: -0.850654 },
    Vector3 { x: -0.688190, y: 0.499997 , z: -0.525736 },
    Vector3 { x: 0.262869 , y: 0.809012 , z: -0.525738 },
    Vector3 { x: 0.951058 , y: 0.309013 , z: 0.000000 },
    Vector3 { x: 0.951058 , y: -0.309013, z: 0.000000 },
    Vector3 { x: 0.587786 , y: -0.809017, z: 0.000000 },
    Vector3 { x: 0.000000 , y: -1.000000, z: 0.000000 },
    Vector3 { x: -0.587786, y: -0.809017, z: 0.000000 },
    Vector3 { x: -0.951058, y: -0.309013, z: -0.000000 },
    Vector3 { x: -0.951058, y: 0.309013 , z: -0.000000 },
    Vector3 { x: -0.587786, y: 0.809017 , z: -0.000000 },
    Vector3 { x: -0.000000, y: 1.000000 , z: -0.000000 },
    Vector3 { x: 0.587786 , y: 0.809017 , z: -0.000000 },
    Vector3 { x: 0.688190 , y: -0.499997, z: 0.525736 },
    Vector3 { x: -0.262869, y: -0.809012, z: 0.525738 },
    Vector3 { x: -0.850648, y: 0.000000 , z: 0.525736 },
    Vector3 { x: -0.262869, y: 0.809012 , z: 0.525738 },
    Vector3 { x: 0.688190 , y: 0.499997 , z: 0.525736 },
    Vector3 { x: 0.525730 , y: 0.000000 , z: 0.850652 },
    Vector3 { x: 0.162456 , y: -0.499995, z: 0.850654 },
    Vector3 { x: -0.425323, y: -0.309011, z: 0.850654 },
    Vector3 { x: -0.425323, y: 0.309011 , z: 0.850654 },
    Vector3 { x: 0.162456 , y: 0.499995 , z: 0.850654 }
];

impl<N> UniformSphereSample for Vector1<N>
    where Vector1<N>: One {
    #[inline]
    fn sample<F: FnMut(Vector1<N>)>(mut f: F) {
        f(::one())
     }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vector2<N> {
    #[inline]
    fn sample<F: FnMut(Vector2<N>)>(mut f: F) {
         for sample in SAMPLES_2_F64.iter() {
             f(Cast::from(*sample))
         }
     }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vector3<N> {
    #[inline]
    fn sample<F: FnMut(Vector3<N>)>(mut f: F) {
        for sample in SAMPLES_3_F64.iter() {
            f(Cast::from(*sample))
        }
    }
}

impl<N: Cast<f64> + Copy> UniformSphereSample for Vector4<N> {
    #[inline]
    fn sample<F: FnMut(Vector4<N>)>(_: F) {
        panic!("UniformSphereSample::<Vector4<N>>::sample : Not yet implemented.")
        // for sample in SAMPLES_3_F32.iter() {
        //     f(Cast::from(*sample))
        // }
    }
}
