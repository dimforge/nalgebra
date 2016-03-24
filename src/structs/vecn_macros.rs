#![macro_use]

macro_rules! vecn_dvec_common_impl(
    ($vecn: ident $(, $param: ident)*) => (
        impl<N: Zero + Copy + Clone $(, $param: ArrayLength<N>)*> $vecn<N $(, $param)*> {
            /// Tests if all components of the vector are zeroes.
            #[inline]
            pub fn is_zero(&self) -> bool {
                self.as_ref().iter().all(|e| e.is_zero())
            }
        }

        impl<N $(, $param: ArrayLength<N>)*> AsRef<[N]> for $vecn<N $(, $param)*> {
            #[inline]
            fn as_ref(&self) -> &[N] {
                &self[.. self.len()]
            }
        }

        impl<N $(, $param: ArrayLength<N>)*> AsMut<[N]> for $vecn<N $(, $param)*> {
            #[inline]
            fn as_mut(&mut self) -> &mut [N] {
                let len = self.len();
                &mut self[.. len]
            }
        }

        impl<N $(, $param: ArrayLength<N>)*> Shape<usize> for $vecn<N $(, $param)*> {
            #[inline]
            fn shape(&self) -> usize {
                self.len()
            }
        }

        impl<N: Copy $(, $param : ArrayLength<N>)*> Indexable<usize, N> for $vecn<N $(, $param)*> {
            #[inline]
            fn swap(&mut self, i: usize, j: usize) {
                assert!(i < self.len());
                assert!(j < self.len());
                self.as_mut().swap(i, j);
            }

            #[inline]
            unsafe fn unsafe_at(&self, i: usize) -> N {
                *self[..].get_unchecked(i)
            }

            #[inline]
            unsafe fn unsafe_set(&mut self, i: usize, val: N) {
                *self[..].get_unchecked_mut(i) = val
            }

        }

        impl<N, T $(, $param: ArrayLength<N>)*> Index<T> for $vecn<N $(, $param)*> where [N]: Index<T> {
            type Output = <[N] as Index<T>>::Output;

            fn index(&self, i: T) -> &<[N] as Index<T>>::Output {
                &self.as_ref()[i]
            }
        }

        impl<N, T $(, $param: ArrayLength<N>)*> IndexMut<T> for $vecn<N $(, $param)*> where [N]: IndexMut<T> {
            fn index_mut(&mut self, i: T) -> &mut <[N] as Index<T>>::Output {
                &mut self.as_mut()[i]
            }
        }

        impl<N $(, $param : ArrayLength<N>)*> Iterable<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn iter<'l>(&'l self) -> Iter<'l, N> {
                self.as_ref().iter()
            }
        }

        impl<N $(, $param : ArrayLength<N>)*> IterableMut<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn iter_mut<'l>(&'l mut self) -> IterMut<'l, N> {
                self.as_mut().iter_mut()
            }
        }

        impl<N: Copy + Add<N, Output = N> + Mul<N, Output = N> $(, $param : ArrayLength<N>)*>
            Axpy<N> for $vecn<N $(, $param)*> {
            fn axpy(&mut self, a: &N, x: &$vecn<N $(, $param)*>) {
                assert!(self.len() == x.len());

                for i in 0 .. x.len() {
                    unsafe {
                        let self_i = self.unsafe_at(i);
                        self.unsafe_set(i, self_i + *a * x.unsafe_at(i))
                    }
                }
            }
        }

        impl<N: Copy + Mul<N, Output = N> + Zero $(, $param : ArrayLength<N>)*>
            Mul<$vecn<N $(, $param)*>> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn mul(self, right: $vecn<N $(, $param)*>) -> $vecn<N $(, $param)*> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left * *right
                }

                res
            }
        }

        impl<N: Copy + Div<N, Output = N> + Zero $(, $param : ArrayLength<N>)*>
            Div<$vecn<N $(, $param)*>> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn div(self, right: $vecn<N $(, $param)*>) -> $vecn<N $(, $param)*> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left / *right
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N> + Zero $(, $param : ArrayLength<N>)*>
            Add<$vecn<N $(, $param)*>> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn add(self, right: $vecn<N $(, $param)*>) -> $vecn<N $(, $param)*> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left + *right
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N> + Zero $(, $param : ArrayLength<N>)*>
            Sub<$vecn<N $(, $param)*>> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn sub(self, right: $vecn<N $(, $param)*>) -> $vecn<N $(, $param)*> {
                assert!(self.len() == right.len());

                let mut res = self;

                for (left, right) in res.as_mut().iter_mut().zip(right.as_ref().iter()) {
                    *left = *left - *right
                }

                res
            }
        }

        impl<N: Neg<Output = N> + Zero + Copy $(, $param : ArrayLength<N>)*> Neg for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn neg(mut self) -> $vecn<N $(, $param)*> {
                for e in self.as_mut().iter_mut() {
                    *e = -*e;
                }

                self
            }
        }

        impl<N: BaseNum $(, $param : ArrayLength<N>)*> Dot<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn dot(&self, other: &$vecn<N $(, $param)*>) -> N {
                assert!(self.len() == other.len());
                let mut res: N = ::zero();
                for i in 0 .. self.len() {
                    res = res + unsafe { self.unsafe_at(i) * other.unsafe_at(i) };
                }
                res
            }
        }

        impl<N: BaseFloat $(, $param : ArrayLength<N>)*> Norm<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn sqnorm(&self) -> N {
                Dot::dot(self, self)
            }

            #[inline]
            fn normalize(&self) -> $vecn<N $(, $param)*> {
                let mut res : $vecn<N $(, $param)*> = self.clone();
                let _ = res.normalize_mut();
                res
            }

            #[inline]
            fn normalize_mut(&mut self) -> N {
                let l = Norm::norm(self);

                for n in self.as_mut().iter_mut() {
                    *n = *n / l;
                }

                l
            }
        }

        impl<N: BaseFloat + Cast<f64> $(, $param : ArrayLength<N>)*> Mean<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn mean(&self) -> N {
                let normalizer = ::cast(1.0f64 / self.len() as f64);
                self.iter().fold(::zero(), |acc, x| acc + *x * normalizer)
            }
        }

        impl<N: ApproxEq<N> $(, $param : ArrayLength<N>)*> ApproxEq<N> for $vecn<N $(, $param)*> {
            #[inline]
            fn approx_epsilon(_: Option<$vecn<N $(, $param)*>>) -> N {
                ApproxEq::approx_epsilon(None::<N>)
            }

            #[inline]
            fn approx_ulps(_: Option<$vecn<N $(, $param)*>>) -> u32 {
                ApproxEq::approx_ulps(None::<N>)
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$vecn<N $(, $param)*>, epsilon: &N) -> bool {
                let mut zip = self.as_ref().iter().zip(other.as_ref().iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_eps(a, b, epsilon))
            }

            #[inline]
            fn approx_eq_ulps(&self, other: &$vecn<N $(, $param)*>, ulps: u32) -> bool {
                let mut zip = self.as_ref().iter().zip(other.as_ref().iter());
                zip.all(|(a, b)| ApproxEq::approx_eq_ulps(a, b, ulps))
            }
        }

        impl<N: Copy + Mul<N, Output = N> + Zero $(, $param : ArrayLength<N>)*>
            Mul<N> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn mul(self, right: N) -> $vecn<N $(, $param)*> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e * right
                }

                res
            }
        }

        impl<N: Copy + Div<N, Output = N> + Zero $(, $param : ArrayLength<N>)*> Div<N> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn div(self, right: N) -> $vecn<N $(, $param)*> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e / right
                }

                res
            }
        }

        impl<N: Copy + Add<N, Output = N> + Zero $(, $param : ArrayLength<N>)*> Add<N> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn add(self, right: N) -> $vecn<N $(, $param)*> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e + right
                }

                res
            }
        }

        impl<N: Copy + Sub<N, Output = N> + Zero $(, $param : ArrayLength<N>)*> Sub<N> for $vecn<N $(, $param)*> {
            type Output = $vecn<N $(, $param)*>;

            #[inline]
            fn sub(self, right: N) -> $vecn<N $(, $param)*> {
                let mut res = self;

                for e in res.as_mut().iter_mut() {
                    *e = *e - right
                }

                res
            }
        }
    )
);
