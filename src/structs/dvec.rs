//! Vector with dimensions unknown at compile-time.

#[allow(missing_doc)]; // we hide doc to not have to document the $trhs double dispatch trait.

use std::num::{Zero, One, Float};
use std::rand::Rand;
use std::rand;
use std::vec;
use std::vec::{Items, MutItems};
use traits::operations::ApproxEq;
use std::iter::FromIterator;
use traits::geometry::{Dot, Norm};
use traits::structure::{Iterable, IterableMut};

#[doc(hidden)]
mod metal;

/// Vector with a dimension unknown at compile-time.
#[deriving(Eq, Show, Clone)]
pub struct DVec<N> {
    /// Components of the vector. Contains as much elements as the vector dimension.
    at: ~[N]
}

double_dispatch_binop_decl_trait!(DVec, DVecMulRhs)
double_dispatch_binop_decl_trait!(DVec, DVecDivRhs)
double_dispatch_binop_decl_trait!(DVec, DVecAddRhs)
double_dispatch_binop_decl_trait!(DVec, DVecSubRhs)

mul_redispatch_impl!(DVec, DVecMulRhs)
div_redispatch_impl!(DVec, DVecDivRhs)
add_redispatch_impl!(DVec, DVecAddRhs)
sub_redispatch_impl!(DVec, DVecSubRhs)

impl<N: Zero + Clone> DVec<N> {
    /// Builds a vector filled with zeros.
    /// 
    /// # Arguments
    /// * `dim` - The dimension of the vector.
    #[inline]
    pub fn new_zeros(dim: uint) -> DVec<N> {
        DVec::from_elem(dim, Zero::zero())
    }

    /// Tests if all components of the vector are zeroes.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.at.iter().all(|e| e.is_zero())
    }
}

impl<N: Clone> DVec<N> {
    /// Indexing without bounds checking.
    pub unsafe fn at_fast(&self, i: uint) -> N {
        (*self.at.unsafe_ref(i)).clone()
    }
}

impl<N: One + Clone> DVec<N> {
    /// Builds a vector filled with ones.
    /// 
    /// # Arguments
    /// * `dim` - The dimension of the vector.
    #[inline]
    pub fn new_ones(dim: uint) -> DVec<N> {
        DVec::from_elem(dim, One::one())
    }
}

impl<N: Rand> DVec<N> {
    /// Builds a vector filled with random values.
    #[inline]
    pub fn new_random(dim: uint) -> DVec<N> {
        DVec::from_fn(dim, |_| rand::random())
    }
}

impl<N> DVec<N> {
    /// Creates an uninitialized vec.
    #[inline]
    pub unsafe fn new_uninitialized(dim: uint) -> DVec<N> {
        let mut vec = vec::with_capacity(dim);
        vec.set_len(dim);

        DVec {
            at: vec
        }
    }

    #[inline]
    pub unsafe fn set_fast(&mut self, i: uint, val: N) {
        *self.at.unsafe_mut_ref(i) = val
    }

    /// Gets a reference to of this vector data.
    #[inline]
    pub fn as_vec<'r>(&'r self) -> &'r [N] {
        let data: &'r [N] = self.at;

        data
    }

    /// Gets a mutable reference to of this vector data.
    #[inline]
    pub fn as_mut_vec<'r>(&'r mut self) -> &'r mut [N] {
        let data: &'r mut [N] = self.at;

        data
    }

    /// Extracts this vector data.
    #[inline]
    pub fn to_vec(self) -> ~[N] {
        self.at
    }
}

impl<N: Clone> DVec<N> {
    /// Builds a vector filled with a constant.
    #[inline]
    pub fn from_elem(dim: uint, elem: N) -> DVec<N> {
        DVec { at: vec::from_elem(dim, elem) }
    }

    /// Builds a vector filled with the components provided by a vector.
    ///
    /// The vector must have at least `dim` elements.
    #[inline]
    pub fn from_vec(dim: uint, vec: &[N]) -> DVec<N> {
        assert!(dim <= vec.len());

        DVec {
            at: vec.slice_to(dim).to_owned()
        }
    }
}

impl<N> DVec<N> {
    /// Builds a vector filled with the result of a function.
    #[inline(always)]
    pub fn from_fn(dim: uint, f: |uint| -> N) -> DVec<N> {
        DVec { at: vec::from_fn(dim, |i| f(i)) }
    }
}

impl<N> Container for DVec<N> {
    #[inline]
    fn len(&self) -> uint {
        self.at.len()
    }
}

impl<N> Iterable<N> for DVec<N> {
    #[inline]
    fn iter<'l>(&'l self) -> Items<'l, N> {
        self.at.iter()
    }
}

impl<N> IterableMut<N> for DVec<N> {
    #[inline]
    fn mut_iter<'l>(&'l mut self) -> MutItems<'l, N> {
        self.at.mut_iter()
    }
}

impl<N> FromIterator<N> for DVec<N> {
    #[inline]
    fn from_iterator<I: Iterator<N>>(mut param: &mut I) -> DVec<N> {
        let mut res = DVec { at: ~[] };

        for e in param {
            res.at.push(e)
        }

        res
    }
}

impl<N: Clone + Num + Float + ApproxEq<N> + DVecMulRhs<N, DVec<N>>> DVec<N> {
    /// Computes the canonical basis for the given dimension. A canonical basis is a set of
    /// vectors, mutually orthogonal, with all its component equal to 0.0 except one which is equal
    /// to 1.0.
    pub fn canonical_basis_with_dim(dim: uint) -> ~[DVec<N>] {
        let mut res : ~[DVec<N>] = ~[];

        for i in range(0u, dim) {
            let mut basis_element : DVec<N> = DVec::new_zeros(dim);

            basis_element.at[i] = One::one();

            res.push(basis_element);
        }

        res
    }

    /// Computes a basis of the space orthogonal to the vector. If the input vector is of dimension
    /// `n`, this will return `n - 1` vectors.
    pub fn orthogonal_subspace_basis(&self) -> ~[DVec<N>] {
        // compute the basis of the orthogonal subspace using Gram-Schmidt
        // orthogonalization algorithm
        let     dim              = self.at.len();
        let mut res : ~[DVec<N>] = ~[];

        for i in range(0u, dim) {
            let mut basis_element : DVec<N> = DVec::new_zeros(self.at.len());

            basis_element.at[i] = One::one();

            if res.len() == dim - 1 {
                break;
            }

            let mut elt = basis_element.clone();

            elt = elt - self * Dot::dot(&basis_element, self);

            for v in res.iter() {
                elt = elt - v * Dot::dot(&elt, v)
            };

            if !ApproxEq::approx_eq(&Norm::sqnorm(&elt), &Zero::zero()) {
                res.push(Norm::normalize_cpy(&elt));
            }
        }

        assert!(res.len() == dim - 1);

        res
    }
}

impl<N: Add<N, N>> DVecAddRhs<N, DVec<N>> for DVec<N> {
    #[inline]
    fn binop(left: &DVec<N>, right: &DVec<N>) -> DVec<N> {
        assert!(left.at.len() == right.at.len());
        DVec {
            at: left.at.iter().zip(right.at.iter()).map(|(a, b)| *a + *b).collect()
        }
    }
}

impl<N: Sub<N, N>> DVecSubRhs<N, DVec<N>> for DVec<N> {
    #[inline]
    fn binop(left: &DVec<N>, right: &DVec<N>) -> DVec<N> {
        assert!(left.at.len() == right.at.len());
        DVec {
            at: left.at.iter().zip(right.at.iter()).map(|(a, b)| *a - *b).collect()
        }
    }
}

impl<N: Neg<N>> Neg<DVec<N>> for DVec<N> {
    #[inline]
    fn neg(&self) -> DVec<N> {
        DVec { at: self.at.iter().map(|a| -a).collect() }
    }
}

impl<N: Num + Clone> Dot<N> for DVec<N> {
    #[inline]
    fn dot(a: &DVec<N>, b: &DVec<N>) -> N {
        assert!(a.at.len() == b.at.len());

        let mut res: N = Zero::zero();

        for i in range(0u, a.at.len()) {
            res = res + unsafe { a.at_fast(i) * b.at_fast(i) };
        }

        res
    } 

    #[inline]
    fn sub_dot(a: &DVec<N>, b: &DVec<N>, c: &DVec<N>) -> N {
        let mut res: N = Zero::zero();

        for i in range(0u, a.at.len()) {
            res = res + unsafe { (a.at_fast(i) - b.at_fast(i)) * c.at_fast(i) };
        }

        res
    } 
}

impl<N: Num + Float + Clone> Norm<N> for DVec<N> {
    #[inline]
    fn sqnorm(v: &DVec<N>) -> N {
        Dot::dot(v, v)
    }

    #[inline]
    fn norm(v: &DVec<N>) -> N {
        Norm::sqnorm(v).sqrt()
    }

    #[inline]
    fn normalize_cpy(v: &DVec<N>) -> DVec<N> {
        let mut res : DVec<N> = v.clone();

        let _ = res.normalize();

        res
    }

    #[inline]
    fn normalize(&mut self) -> N {
        let l = Norm::norm(self);

        for i in range(0u, self.at.len()) {
            self.at[i] = self.at[i] / l;
        }

        l
    }
}

impl<N: ApproxEq<N>> ApproxEq<N> for DVec<N> {
    #[inline]
    fn approx_epsilon(_: Option<DVec<N>>) -> N {
        ApproxEq::approx_epsilon(None::<N>)
    }

    #[inline]
    fn approx_eq(a: &DVec<N>, b: &DVec<N>) -> bool {
        let mut zip = a.at.iter().zip(b.at.iter());

        zip.all(|(a, b)| ApproxEq::approx_eq(a, b))
    }

    #[inline]
    fn approx_eq_eps(a: &DVec<N>, b: &DVec<N>, epsilon: &N) -> bool {
        let mut zip = a.at.iter().zip(b.at.iter());

        zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
    }
}

macro_rules! scalar_mul_impl (
    ($n: ident) => (
        impl DVecMulRhs<$n, DVec<$n>> for $n {
            #[inline]
            fn binop(left: &DVec<$n>, right: &$n) -> DVec<$n> {
                DVec { at: left.at.iter().map(|a| a * *right).collect() }
            }
        }
    )
)

macro_rules! scalar_div_impl (
    ($n: ident) => (
        impl DVecDivRhs<$n, DVec<$n>> for $n {
            #[inline]
            fn binop(left: &DVec<$n>, right: &$n) -> DVec<$n> {
                DVec { at: left.at.iter().map(|a| a / *right).collect() }
            }
        }
    )
)

macro_rules! scalar_add_impl (
    ($n: ident) => (
        impl DVecAddRhs<$n, DVec<$n>> for $n {
            #[inline]
            fn binop(left: &DVec<$n>, right: &$n) -> DVec<$n> {
                DVec { at: left.at.iter().map(|a| a + *right).collect() }
            }
        }
    )
)

macro_rules! scalar_sub_impl (
    ($n: ident) => (
        impl DVecSubRhs<$n, DVec<$n>> for $n {
            #[inline]
            fn binop(left: &DVec<$n>, right: &$n) -> DVec<$n> {
                DVec { at: left.at.iter().map(|a| a - *right).collect() }
            }
        }
    )
)

scalar_mul_impl!(f64)
scalar_mul_impl!(f32)
scalar_mul_impl!(u64)
scalar_mul_impl!(u32)
scalar_mul_impl!(u16)
scalar_mul_impl!(u8)
scalar_mul_impl!(i64)
scalar_mul_impl!(i32)
scalar_mul_impl!(i16)
scalar_mul_impl!(i8)
scalar_mul_impl!(uint)
scalar_mul_impl!(int)

scalar_div_impl!(f64)
scalar_div_impl!(f32)
scalar_div_impl!(u64)
scalar_div_impl!(u32)
scalar_div_impl!(u16)
scalar_div_impl!(u8)
scalar_div_impl!(i64)
scalar_div_impl!(i32)
scalar_div_impl!(i16)
scalar_div_impl!(i8)
scalar_div_impl!(uint)
scalar_div_impl!(int)

scalar_add_impl!(f64)
scalar_add_impl!(f32)
scalar_add_impl!(u64)
scalar_add_impl!(u32)
scalar_add_impl!(u16)
scalar_add_impl!(u8)
scalar_add_impl!(i64)
scalar_add_impl!(i32)
scalar_add_impl!(i16)
scalar_add_impl!(i8)
scalar_add_impl!(uint)
scalar_add_impl!(int)

scalar_sub_impl!(f64)
scalar_sub_impl!(f32)
scalar_sub_impl!(u64)
scalar_sub_impl!(u32)
scalar_sub_impl!(u16)
scalar_sub_impl!(u8)
scalar_sub_impl!(i64)
scalar_sub_impl!(i32)
scalar_sub_impl!(i16)
scalar_sub_impl!(i8)
scalar_sub_impl!(uint)
scalar_sub_impl!(int)
