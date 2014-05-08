#![macro_escape]

#![doc(hidden)] // we hide doc to not have to document the $trhs double dispatch trait.

// Create the traits needed to do fancy operator oveloading.
// This is a meta version of
// http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/ 
//
// Hopefully future version of the language will make this useless.
macro_rules! double_dispatch_binop_decl_trait(
    ($t: ident, $trhs: ident) => (
        pub trait $trhs<N, Res> {
            /// Applies the binary operation represented by this trait.
            ///
            /// In infix notation, assuming the operation is noted `*`, the following is computed:
            /// `right * left`.
            fn binop(left: &$t<N>, right: &Self) -> Res;
        }
     )
)

// Macro driving the `Mul` trait to use the related trait for double redispatch.
macro_rules! mul_redispatch_impl(
    ($t: ident, $trhs: ident) => (
        impl<N, Rhs: $trhs<N, Res>, Res> Mul<Rhs, Res> for $t<N> {
            #[inline(always)]
            fn mul(&self, other: &Rhs) -> Res {
                $trhs::binop(self, other)
            }
        }
    )
)

// Macro driving the `Div` trait to use the related trait for double redispatch.
macro_rules! div_redispatch_impl(
    ($t: ident, $trhs: ident) => (
        impl<N, Rhs: $trhs<N, Res>, Res> Div<Rhs, Res> for $t<N> {
            #[inline(always)]
            fn div(&self, other: &Rhs) -> Res {
                $trhs::binop(self, other)
            }
        }
    )
)

// Macro driving the `Add` trait to use the related trait for double redispatch.
macro_rules! add_redispatch_impl(
    ($t: ident, $trhs: ident) => (
        impl<N, Rhs: $trhs<N, Res>, Res> Add<Rhs, Res> for $t<N> {
            #[inline(always)]
            fn add(&self, other: &Rhs) -> Res {
                $trhs::binop(self, other)
            }
        }
    )
)

// Macro driving the `Sub` trait to use the related trait for double redispatch.
macro_rules! sub_redispatch_impl(
    ($t: ident, $trhs: ident) => (
        impl<N, Rhs: $trhs<N, Res>, Res> Sub<Rhs, Res> for $t<N> {
            #[inline(always)]
            fn sub(&self, other: &Rhs) -> Res {
                $trhs::binop(self, other)
            }
        }
    )
)

// Double dispatch traits to drive the Cast method for structured types with one type parameter.
macro_rules! double_dispatch_cast_decl_trait(
    ($t: ident, $tcast: ident) => (
        pub trait $tcast<N> {
            fn to(Self) -> $t<N>;
        }
    )
)

macro_rules! cast_redispatch_impl(
    ($t:ident, $tcast: ident) => (
        impl<T: $tcast<N>, N> Cast<T> for $t<N> {
            #[inline(always)]
            fn from(t: T) -> $t<N> {
                $tcast::to(t)
            }
        }
    )
)
