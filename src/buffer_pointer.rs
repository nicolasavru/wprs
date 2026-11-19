// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::array;
use std::cmp;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::ptr::NonNull;

use crate::utils::AssertN;
use crate::utils::AssertN3;

/// A slice-like wrapper around a pointer and length.
///
/// The unsafe boundary is at construction: `ptr` must be valid for reads of
/// `len` elements for the lifetime of the `BufferPointer` (`new` checks
/// non-null and alignment). Assuming the pointer isn't {d,r}eallocated by
/// anything else, during the lifetime of `BufferPointer`, the implemented
/// operations on `BufferPointer` will be safe.
///
/// Useful when you have a raw pointer that cannot be wrapped in a slice (using
/// slice::from_raw_parts), for example if it points to shared memory that could
/// be mutated during the lifetime of the slice.
#[derive(Debug, Eq, PartialEq)]
pub struct BufferPointer<'a, T: 'a + Copy> {
    ptr: NonNull<T>,
    len: usize,
    lifetime: PhantomData<&'a T>,
}

impl<'a, T: 'a + Copy> Copy for BufferPointer<'a, T> {}

#[allow(clippy::non_canonical_clone_impl)]
impl<'a, T: 'a + Copy> Clone for BufferPointer<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a + Copy> BufferPointer<'a, T> {
    /// # Safety
    /// `ptr` must be aligned for `T` and valid for reads of `len` elements for `'a`.
    #[inline]
    unsafe fn new_unchecked(ptr: NonNull<T>, len: usize, lifetime: PhantomData<&'a T>) -> Self {
        Self { ptr, len, lifetime }
    }

    /// # Safety
    /// * `ptr` must be valid for reads of `len` elements for the lifetime of the
    ///   `BufferPointer`.
    ///
    /// # Panics
    /// If `*ptr` is null or unaligned for `T`.
    #[inline]
    pub unsafe fn new(ptr: &'a *const T, len: usize) -> Self {
        let ptr = NonNull::new((*ptr).cast_mut()).expect("BufferPointer must be non-null");
        assert!(ptr.is_aligned());
        Self {
            ptr,
            len,
            lifetime: PhantomData::<&'a T>,
        }
    }

    #[inline]
    pub fn ptr(self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// All possible bit patterns of T must be valid bit patterns of U, including any
    /// padding.
    /// # Panics
    /// If `self.ptr` is not aligned to `align_of::<U>()` or if
    /// `size_of::<T>() * self.len()` is not a multiple of `size_of::<U>()`.
    #[inline]
    pub unsafe fn cast<U: Copy>(self) -> BufferPointer<'a, U> {
        let new_ptr = self.ptr.cast::<U>();
        assert!(new_ptr.is_aligned());

        let old_size = mem::size_of::<T>();
        let new_size = mem::size_of::<U>();
        let new_len = (self.len() * old_size) / new_size;
        assert!(
            old_size * self.len() == new_size * new_len,
            "old_size = {old_size}, self.len() = {}, new_size = {new_size}, new_len = {new_len}",
            self.len()
        );

        // SAFETY: self.ptr is non-null, and we verified that new_ptr is aligned
        // and that length in bytes doesn't change.
        unsafe { BufferPointer::new_unchecked(new_ptr, new_len, PhantomData) }
    }

    /// Splits BufferPointer([0, len]) into BufferPointer([0, mid-1]) and
    /// BufferPointer([mid, end]).
    ///
    /// # Panics
    /// If mid > len.
    #[inline]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len());
        // SAFETY: self.ptr is non-null, aligned, and valid for self.len elements;
        // adding `mid` preserves alignment and stays within the allocation.
        unsafe {
            (
                Self::new_unchecked(self.ptr, mid, self.lifetime),
                Self::new_unchecked(self.ptr.add(mid), self.len() - mid, self.lifetime),
            )
        }
    }

    /// # Panics
    /// If chunk_size == 0.
    #[inline]
    pub fn chunks(self, chunk_size: usize) -> Chunks<'a, T> {
        assert!(chunk_size != 0);
        Chunks::new(self, chunk_size)
    }

    #[inline]
    pub fn array_chunks<const N: usize>(self) -> ArrayChunks<'a, T, N> {
        ArrayChunks::new(self)
    }

    fn as_ptr_range(&self) -> Range<*const T> {
        let start = self.ptr();
        // SAFETY: precondition for BufferPointer::new requires self.ptr to be valid
        // for reads of self.len elements.
        start..unsafe { start.add(self.len) }
    }

    /// # Panics
    /// If `dst.len() < self.len()` or if `dst` overlaps with `self`.
    pub fn copy_to_nonoverlapping(self, dst: &mut [T]) {
        assert!(dst.len() >= self.len());
        assert!(!self.as_ptr_range().contains(&dst.as_ptr()));
        assert!(!dst.as_ptr_range().contains(&self.ptr()));

        // SAFETY: precondition for BufferPointer::new requires self.ptr to be
        // valid for reads of self.len elements and we verified that dst.len()
        // >= self.len() and that self and dst are non-overlapping.
        unsafe {
            self.ptr()
                .copy_to_nonoverlapping(dst.as_mut_ptr(), self.len());
        }
    }
}

// SAFETY: `BufferPointer` provides read-only copy access to `T` values across
// threads, requiring `T: Send + Sync` in addition to `Copy`.
unsafe impl<'a, T: 'a + Copy + Send + Sync> Send for BufferPointer<'a, T> {}

impl<'a, T: 'a + Copy> IntoIterator for &'a BufferPointer<'a, T> {
    type Item = T;
    type IntoIter = BufferPointerIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        BufferPointerIter::new(self)
    }
}

#[derive(Debug)]
pub struct BufferPointerIter<'a, T: 'a + Copy> {
    ptr: &'a BufferPointer<'a, T>,
    idx: usize,
}

impl<'a, T: 'a + Copy> BufferPointerIter<'a, T> {
    fn new(ptr: &'a BufferPointer<'a, T>) -> Self {
        Self { ptr, idx: 0 }
    }
}

impl<T: Copy> Iterator for BufferPointerIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx >= self.ptr.len {
            None
        } else {
            // SAFETY: precondition for BufferPointer::new requires self.ptr to be
            // valid for reads of self.len elements.
            let ret = unsafe { self.ptr.ptr().add(self.idx).read() };
            self.idx += 1;
            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.ptr.len - self.idx;
        (rem, Some(rem))
    }
}

impl<T: Copy> ExactSizeIterator for BufferPointerIter<'_, T> {}
impl<T: Copy> FusedIterator for BufferPointerIter<'_, T> {}

#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Chunks<'a, T: 'a + Copy> {
    ptr: BufferPointer<'a, T>,
    chunk_size: usize,
}

impl<'a, T: 'a + Copy> Chunks<'a, T> {
    #[inline]
    fn new(ptr: BufferPointer<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0);
        Self { ptr, chunk_size }
    }
}

impl<'a, T: 'a + Copy> Iterator for Chunks<'a, T> {
    type Item = BufferPointer<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.len == 0 {
            None
        } else {
            let chunk_size = cmp::min(self.ptr.len, self.chunk_size);
            let (fst, snd) = self.ptr.split_at(chunk_size);
            self.ptr = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.ptr.len.div_ceil(self.chunk_size);
        (n, Some(n))
    }
}

impl<T: Copy> ExactSizeIterator for Chunks<'_, T> {}
impl<T: Copy> FusedIterator for Chunks<'_, T> {}

#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ArrayChunks<'a, T: 'a + Copy, const N: usize> {
    ptr: BufferPointer<'a, T>,
}

impl<'a, T: 'a + Copy, const N: usize> ArrayChunks<'a, T, N> {
    #[inline]
    fn new(ptr: BufferPointer<'a, T>) -> Self {
        _ = AssertN::<N>::NE_0;
        Self { ptr }
    }
}

impl<'a, T: 'a + Copy, const N: usize> Iterator for ArrayChunks<'a, T, N> {
    type Item = KnownSizeBufferPointer<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.len < N {
            None
        } else {
            let cur = self.ptr.ptr;
            // SAFETY: `self.ptr.len >= N` and `N != 0`, so `cur` is valid for `N`
            // elements and advancing by `N` stays within the allocation.
            self.ptr = unsafe {
                BufferPointer::new_unchecked(cur.add(N), self.ptr.len - N, self.ptr.lifetime)
            };
            // SAFETY: `cur` is non-null, aligned, and valid for `N` elements.
            Some(unsafe { KnownSizeBufferPointer::new_unchecked(cur, PhantomData) })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.ptr.len / N;
        (n, Some(n))
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for ArrayChunks<'_, T, N> {}
impl<T: Copy, const N: usize> FusedIterator for ArrayChunks<'_, T, N> {}

#[derive(Debug, Eq, PartialEq)]
pub struct KnownSizeBufferPointer<'a, T: 'a + Copy, const N: usize> {
    ptr: NonNull<T>,
    lifetime: PhantomData<&'a T>,
}

impl<'a, T: 'a + Copy, const N: usize> Copy for KnownSizeBufferPointer<'a, T, N> {}

#[allow(clippy::non_canonical_clone_impl)]
impl<'a, T: 'a + Copy, const N: usize> Clone for KnownSizeBufferPointer<'a, T, N> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a + Copy, const N: usize> KnownSizeBufferPointer<'a, T, N> {
    /// # Safety
    /// `ptr` must be aligned for `T` and valid for reads of `N` elements for `'a`.
    #[inline]
    unsafe fn new_unchecked(ptr: NonNull<T>, lifetime: PhantomData<&'a T>) -> Self {
        let _ = AssertN::<N>::NE_0;
        Self { ptr, lifetime }
    }

    #[inline]
    pub fn ptr(self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Reads and returns element `i` by value without creating a Rust
    /// reference into potentially shared memory.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline]
    pub fn get(&self, i: usize) -> T {
        assert!(i < N);
        // SAFETY: `self.ptr` is valid for reads of `N` elements and `i < N`.
        unsafe { self.ptr.add(i).read() }
    }

    // TODO(https://github.com/rust-lang/rust/issues/76560): derive N_PARTS from
    // SIZE.
    /// Partitions KnownSizeBufferPointer of length N into N_PARTS
    /// KnownSizeBufferPointers of length SIZE.
    ///
    /// Fails to compile if SIZE * N_PARTS != N.
    #[inline]
    pub fn as_chunks<const SIZE: usize, const N_PARTS: usize>(
        self,
    ) -> [KnownSizeBufferPointer<'a, T, SIZE>; N_PARTS] {
        let _ = AssertN::<SIZE>::NE_0;
        let _ = AssertN::<N_PARTS>::NE_0;
        let _ = AssertN3::<SIZE, N_PARTS, N>::N1_X_N2_EQ_N3;
        // SAFETY: self.ptr is non-null, aligned, and valid for N == SIZE * N_PARTS
        // elements, so each `i * SIZE` offset (`i < N_PARTS`) is aligned and valid
        // for SIZE elements.
        array::from_fn(|i| unsafe {
            KnownSizeBufferPointer::new_unchecked(self.ptr.add(i * SIZE), self.lifetime)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::slice;

    use super::*;

    fn test_arr<const LEN: usize>() -> [u32; LEN] {
        (0..(LEN as u32)).collect::<Vec<u32>>().try_into().unwrap()
    }

    #[test]
    #[should_panic]
    fn test_nullptr() {
        let a_ptr: *const u32 = ptr::null();
        let _a_bufptr = unsafe { BufferPointer::new(&a_ptr, 8) };
    }

    #[test]
    #[should_panic]
    fn test_new_unaligned() {
        let aligned = [0u32; 2];
        let unaligned_ptr = unsafe { aligned.as_ptr().cast::<u8>().add(1).cast::<u32>() };
        let _a_bufptr = unsafe { BufferPointer::new(&unaligned_ptr, 1) };
    }

    #[test]
    fn test_cast_aligned() {
        let a = [0u32; 4];
        let a_ptr = a.as_ptr();
        let bufptr_u32 = unsafe { BufferPointer::new(&a_ptr, a.len()) };
        let _bufptr_u8: BufferPointer<u8> = unsafe { bufptr_u32.cast() };
    }

    #[test]
    #[should_panic]
    fn test_cast_unsaligned() {
        let aligned = [0u32; 2];
        let a_ptr = unsafe { aligned.as_ptr().cast::<u8>().add(1) };
        let bufptr_u8 = unsafe { BufferPointer::new(&a_ptr, 4) };
        let _bufptr_u32: BufferPointer<u32> = unsafe { bufptr_u8.cast() };
    }

    #[test]
    #[should_panic]
    fn test_cast_unequal_size() {
        let a = [0u8; 5];
        let a_ptr = a.as_ptr();
        let bufptr_struct = unsafe { BufferPointer::new(&a_ptr, a.len()) };
        let _bufptr_u32: BufferPointer<u32> = unsafe { bufptr_struct.cast() };
    }

    #[test]
    fn test_split_at() {
        let a = [0u32; 5];
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let (a_bufptr_0, a_bufptr_1) = a_bufptr.split_at(2);
        assert_eq!(a_bufptr_0.len(), 2);
        assert_eq!(a_bufptr_1.len(), 3);

        let (a_bufptr_00, a_bufptr_01) = a_bufptr_0.split_at(1);
        let (a_bufptr_10, a_bufptr_11) = a_bufptr_1.split_at(1);
        assert_eq!(a_bufptr_00.len(), 1);
        assert_eq!(a_bufptr_01.len(), 1);
        assert_eq!(a_bufptr_10.len(), 1);
        assert_eq!(a_bufptr_11.len(), 2);

        let (a_bufptr_000, a_bufptr_001) = a_bufptr_00.split_at(0);
        assert_eq!(a_bufptr_000.len(), 0);
        assert_eq!(a_bufptr_001.len(), 1);

        let (a_bufptr_100, a_bufptr_101) = a_bufptr_10.split_at(1);
        assert_eq!(a_bufptr_100.len(), 1);
        assert_eq!(a_bufptr_101.len(), 0);
    }

    #[test]
    #[should_panic]
    fn test_split_at_past_end() {
        let a = [0u32; 5];
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };
        let (_a_bufptr_0, _a_bufptr_1) = a_bufptr.split_at(6);
    }

    #[test]
    fn test_chunks() {
        let a: [u32; 5] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let chunks = a_bufptr.chunks(2);

        let a_ptr0 = unsafe { a_ptr.offset(0) };
        let a_ptr2 = unsafe { a_ptr.offset(2) };
        let a_ptr4 = unsafe { a_ptr.offset(4) };

        assert_eq!(
            chunks.collect::<Vec<_>>(),
            vec![
                unsafe { BufferPointer::new(&a_ptr0, 2) },
                unsafe { BufferPointer::new(&a_ptr2, 2) },
                unsafe { BufferPointer::new(&a_ptr4, 1) },
            ]
        );
    }

    #[test]
    fn test_copy_to_nonoverlapping() {
        let a: [u32; 5] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let mut b = [0u32; 6];
        a_bufptr.copy_to_nonoverlapping(&mut b);
        assert_eq!(b, [0, 1, 2, 3, 4, 0]);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_nonoverlapping_dst_too_small() {
        let a: [u32; 5] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let mut b = [0u32; 4];
        a_bufptr.copy_to_nonoverlapping(&mut b);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_nonoverlapping_actually_overlapping() {
        let mut a: [u32; 5] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len() - 1) };

        let a_subslice = unsafe { slice::from_raw_parts_mut(a.as_mut_ptr().add(1), a.len() - 1) };
        a_bufptr.copy_to_nonoverlapping(a_subslice);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_nonoverlapping_actually_overlapping_other_direction() {
        let mut a: [u32; 5] = test_arr();
        let a_ptr = unsafe { a.as_ptr().add(1) };
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len() - 1) };

        a_bufptr.copy_to_nonoverlapping(&mut a);
    }

    #[test]
    fn test_iter() {
        let a: [u32; 5] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let mut iter = a_bufptr.into_iter();
        assert_eq!(iter.len(), 5);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_array_chunks_and_known_size() {
        let a: [u32; 10] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };

        let mut chunks = a_bufptr.array_chunks::<4>();
        assert_eq!(chunks.len(), 2);

        let c0 = chunks.next().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(array::from_fn::<_, 4, _>(|i| c0.get(i)), [0, 1, 2, 3]);

        let [p0, p1] = c0.as_chunks::<2, 2>();
        assert_eq!(array::from_fn::<_, 2, _>(|i| p0.get(i)), [0, 1]);
        assert_eq!(array::from_fn::<_, 2, _>(|i| p1.get(i)), [2, 3]);

        let c1 = chunks.next().unwrap();
        assert_eq!(chunks.len(), 0);
        assert_eq!(array::from_fn::<_, 4, _>(|i| c1.get(i)), [4, 5, 6, 7]);
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.len(), 0);

        // Exact-multiple length (`len == N -> 0`) and asymmetric `as_chunks`
        // (`SIZE < N_PARTS` and `SIZE > N_PARTS`).
        let exact: [u32; 6] = test_arr();
        let exact_ptr = exact.as_ptr();
        let exact_bufptr = unsafe { BufferPointer::new(&exact_ptr, exact.len()) };
        let mut exact_chunks = exact_bufptr.array_chunks::<6>();
        assert_eq!(exact_chunks.len(), 1);
        let c_exact = exact_chunks.next().unwrap();
        assert_eq!(exact_chunks.len(), 0);
        assert_eq!(exact_chunks.next(), None);

        let [s0, s1, s2] = c_exact.as_chunks::<2, 3>();
        assert_eq!(array::from_fn::<_, 2, _>(|i| s0.get(i)), [0, 1]);
        assert_eq!(array::from_fn::<_, 2, _>(|i| s1.get(i)), [2, 3]);
        assert_eq!(array::from_fn::<_, 2, _>(|i| s2.get(i)), [4, 5]);

        let [l0, l1] = c_exact.as_chunks::<3, 2>();
        assert_eq!(array::from_fn::<_, 3, _>(|i| l0.get(i)), [0, 1, 2]);
        assert_eq!(array::from_fn::<_, 3, _>(|i| l1.get(i)), [3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn test_known_size_get_out_of_bounds() {
        let a: [u32; 4] = test_arr();
        let a_ptr = a.as_ptr();
        let a_bufptr = unsafe { BufferPointer::new(&a_ptr, a.len()) };
        let c0 = a_bufptr.array_chunks::<4>().next().unwrap();
        let _ = c0.get(4);
    }
}
