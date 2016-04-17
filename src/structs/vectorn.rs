use std::slice::{Iter, IterMut};
use std::iter::{FromIterator, IntoIterator};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::mem;
use rand::{Rand, Rng};
use num::{Zero, One};
use generic_array::{GenericArray, ArrayLength};
use traits::operations::{ApproxEq, Axpy, Mean};
use traits::geometry::{Dot, Norm};
use traits::structure::{Iterable, IterableMut, Indexable, Shape, BaseFloat, BaseNum, Cast, Dimension};
#[cfg(feature="arbitrary")]
use quickcheck::{Arbitrary, Gen};

/// A static array of arbitrary dimension.
#[repr(C)]
#[derive(Eq, PartialEq, Debug)] //  FIXME: Hash, RustcEncodable, RustcDecodable
pub struct VectorN<N, D: ArrayLength<N>> {
    /// The underlying data of the vector.
    pub at: GenericArray<N, D>
}

unsafe impl<N: Send, D: ArrayLength<N>> Send for VectorN<N, D> {
}

impl<N: Clone, D: ArrayLength<N>> Clone for VectorN<N, D> {
    fn clone(&self) -> VectorN<N, D> {
        VectorN::new(self.at.clone())
    }
}

impl<N: Copy, D: ArrayLength<N>> Copy for VectorN<N, D>
    where D::ArrayType: Copy { }

impl<N, D: ArrayLength<N>> VectorN<N, D> {
    /// Creates a new vector from a given arbirtarily-sized array.
    #[inline]
    pub fn new(components: GenericArray<N, D>) -> VectorN<N, D> {
        VectorN {
            at: components
        }
    }

    /// The vector length.
    #[inline]
    pub fn len(&self) -> usize {
        self.at.len()
    }
}

impl<N, D: ArrayLength<N>> Dimension for VectorN<N, D> {
    fn dimension(_unused: Option<Self>) -> usize {
        D::to_usize()
    }
}

impl<N: Copy, D: ArrayLength<N>> FromIterator<N> for VectorN<N, D> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = N>>(param: I) -> VectorN<N, D> {
        let mut res: VectorN<N, D> = unsafe { mem::uninitialized() };

        let mut it = param.into_iter();

        for e in res.iter_mut() {
            *e = it.next().expect("Not enough data into the provided iterator to initialize this `VectorN`.");
        }

        res
    }
}

impl<N: Rand + Zero, D: ArrayLength<N>> Rand for VectorN<N, D> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> VectorN<N, D> {
        let mut res: VectorN<N, D> = unsafe { mem::uninitialized() };

        for e in res.iter_mut() {
            *e = Rand::rand(rng)
        }

        res
    }
}

impl<N: Copy + One + Zero, D: ArrayLength<N>> One for VectorN<N, D> {
    #[inline]
    fn one() -> VectorN<N, D> {
        let mut res: VectorN<N, D> = unsafe { mem::uninitialized() };

        for e in res.iter_mut() {
            *e = ::one()
        }

        res
    }
}

impl<N: Copy + Zero, D: ArrayLength<N>> Zero for VectorN<N, D> {
    #[inline]
    fn zero() -> VectorN<N, D> {
        let mut res: VectorN<N, D> = unsafe { mem::uninitialized() };

        for e in res.iter_mut() {
            *e = ::zero()
        }

        res
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(|e| e.is_zero())
    }
}

#[cfg(feature="arbitrary")]
impl<N: Arbitrary + Zero + Copy, D: 'static + ArrayLength<N>> Arbitrary for VectorN<N, D> {
    #[inline]
    fn arbitrary<G: Gen>(g: &mut G) -> VectorN<N, D> {
        (0 .. D::to_usize()).map(|_| Arbitrary::arbitrary(g)).collect()
    }
}


vecn_dvec_common_impl!(VectorN, D);
