//! Matrix iterators.

use std::marker::PhantomData;
use std::mem;

use base::Scalar;
use base::dimension::Dim;
use base::storage::{Storage, StorageMut};

macro_rules! iterator {
    (struct $Name:ident for $Storage:ident.$ptr: ident -> $Ptr:ty, $Ref:ty, $SRef: ty) => {

        /// An iterator through a dense matrix with arbitrary strides matrix.
        pub struct $Name<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> {
            ptr:       $Ptr,
            inner_ptr: $Ptr,
            inner_end: $Ptr,
            size:      usize, // We can't use an end pointer here because a stride might be zero.
            strides:   (S::RStride, S::CStride),
            _phantoms: PhantomData<($Ref, R, C, S)>
        }

        // FIXME: we need to specialize for the case where the matrix storage is owned (in which
        // case the iterator is trivial because it does not have any stride).
        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> $Name<'a, N, R, C, S> {
            /// Creates a new iterator for the given matrix storage.
            pub fn new(storage: $SRef) -> $Name<'a, N, R, C, S> {
                let shape   = storage.shape();
                let strides = storage.strides();
                let inner_offset = shape.0.value() * strides.0.value();
                let ptr = storage.$ptr();

                $Name {
                    ptr:       ptr,
                    inner_ptr: ptr,
                    inner_end: unsafe { ptr.offset(inner_offset as isize) },
                    size:      shape.0.value() * shape.1.value(),
                    strides:   strides,
                    _phantoms: PhantomData
                }
            }
        }

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> Iterator for $Name<'a, N, R, C, S> {
            type Item = $Ref;

            #[inline]
            fn next(&mut self) -> Option<$Ref> {
                unsafe {
                    if self.size == 0 {
                        None
                    } else {
                        self.size -= 1;

                        // Jump to the next outer dimension if needed.
                        if self.ptr == self.inner_end {
                            let stride = self.strides.1.value() as isize;
                            self.inner_end = self.ptr.offset(stride);
                            self.ptr = self.inner_ptr.offset(stride);
                            self.inner_ptr = self.ptr;
                        }

                        // Go to the next element.
                        let old = self.ptr;

                        let stride = self.strides.0.value() as isize;
                        self.ptr = self.ptr.offset(stride);

                        Some(mem::transmute(old))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.size, Some(self.size))
            }

            #[inline]
            fn count(self) -> usize {
                self.size_hint().0
            }
        }

        impl<'a, N: Scalar, R: Dim, C: Dim, S: 'a + $Storage<N, R, C>> ExactSizeIterator for $Name<'a, N, R, C, S> {
            #[inline]
            fn len(&self) -> usize {
                self.size
            }
        }
    }
}

iterator!(struct MatrixIter for Storage.ptr -> *const N, &'a N, &'a S);
iterator!(struct MatrixIterMut for StorageMut.ptr_mut -> *mut N, &'a mut N, &'a mut S);
