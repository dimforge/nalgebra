#[macro_escape];

// Create the traits needed to do fancy operator oveloading.
// This is a meta version of
// http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/ 
//
// Hopefully future version of the language will make this useless.
macro_rules! double_dispatch_binop_decl_trait(
    ($t: ident, $trhs: ident) => (
        pub trait $trhs<N, Res> {
            fn $trhs(&self, other: &$t<N>) -> Res;
        }
     )
)

// Macro driving the `Mul` trait to use the related trait for double redispatch.
macro_rules! mul_redispatch_impl(
  ($t: ident, $trhs: ident) => (
    impl<N, Rhs: $trhs<N, Res>, Res> Mul<Rhs, Res> for $t<N> {
        #[inline(always)]
        fn mul(&self, other: &Rhs) -> Res {
            other.$trhs(self)
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
            other.$trhs(self)
        }
    }
  )
)

// Macro driving the `Div` trait to use the related trait for double redispatch.
macro_rules! add_redispatch_impl(
  ($t: ident, $trhs: ident) => (
    impl<N, Rhs: $trhs<N, Res>, Res> Add<Rhs, Res> for $t<N> {
        #[inline(always)]
        fn add(&self, other: &Rhs) -> Res {
            other.$trhs(self)
        }
    }
  )
)

// Macro driving the `Div` trait to use the related trait for double redispatch.
macro_rules! sub_redispatch_impl(
  ($t: ident, $trhs: ident) => (
    impl<N, Rhs: $trhs<N, Res>, Res> Sub<Rhs, Res> for $t<N> {
        #[inline(always)]
        fn sub(&self, other: &Rhs) -> Res {
            other.$trhs(self)
        }
    }
  )
)
