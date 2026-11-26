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

//! x86_64 row kernels for direct 3-byte `SubGreen(BGR)` AoS filtering.
//!
//! Dispatches statically at compile time via `cfg_select!`:
//! - `avx512bw + avx512vl + avx512vbmi` (`x86-64-v4` + `+avx512vbmi`): 512-bit
//!   AVX-512 `vpermb`/`vpermt2b` + 48-byte masked load/store 3-byte prefix sum
//!   + 64-byte-aligned `_mm512_stream_si512` non-temporal stores.
//! - `avx2` (`x86-64-v3` default): 256-bit AVX2 + 128-bit SSSE3 3-byte prefix
//!   sum + 32-byte-aligned `_mm256_stream_si256` non-temporal stores.
//! - `ssse3` (`+ssse3`, `x86-64-v2`, `+avx`): 128-bit native SSSE3 + 16-byte
//!   `_mm_stream_si128` non-temporal stores.
//! - `sse2` (`x86-64` baseline) and Miri: portable [`crate::filtering::ScalarKernel`].

cfg_select! {
    all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
        not(miri),
    ) => {
        mod imp {
            use std::arch::x86_64::__m128i;
            use std::arch::x86_64::__m512i;
            use std::arch::x86_64::_mm_loadu_si128;
            use std::arch::x86_64::_mm_sfence;
            use std::arch::x86_64::_mm_storeu_si128;
            use std::arch::x86_64::_mm512_abs_epi8;
            use std::arch::x86_64::_mm512_add_epi8;
            use std::arch::x86_64::_mm512_add_epi64;
            use std::arch::x86_64::_mm512_bslli_epi128;
            use std::arch::x86_64::_mm512_cmpeq_epi8_mask;
            use std::arch::x86_64::_mm512_cmpneq_epi8_mask;
            use std::arch::x86_64::_mm512_extracti32x4_epi32;
            use std::arch::x86_64::_mm512_loadu_si512;
            use std::arch::x86_64::_mm512_mask_cmpneq_epi8_mask;
            use std::arch::x86_64::_mm512_mask_storeu_epi8;
            use std::arch::x86_64::_mm512_maskz_loadu_epi8;
            use std::arch::x86_64::_mm512_maskz_permutexvar_epi8;
            use std::arch::x86_64::_mm512_or_si512;
            use std::arch::x86_64::_mm512_permutex2var_epi8;
            use std::arch::x86_64::_mm512_permutexvar_epi8;
            use std::arch::x86_64::_mm512_reduce_add_epi64;
            use std::arch::x86_64::_mm512_sad_epu8;
            use std::arch::x86_64::_mm512_set1_epi8;
            use std::arch::x86_64::_mm512_set1_epi32;
            use std::arch::x86_64::_mm512_setzero_si512;
            use std::arch::x86_64::_mm512_shuffle_epi8;
            use std::arch::x86_64::_mm512_storeu_si512;
            use std::arch::x86_64::_mm512_stream_si512;
            use std::arch::x86_64::_mm512_sub_epi8;
            use std::arch::x86_64::_mm512_zextsi128_si512;

            use crate::buffer_pointer::BufferPointer;
            use crate::filtering::RowKernel;
            use crate::vec4u8::Vec4u8;

            /// Opmask selecting the lower 48 bytes (`16` 3-byte BGR pixels) of a ZMM register.
            const MASK_48: u64 = 0x0000_FFFF_FFFF_FFFF;

            /// Opmask selecting B, G, R bytes (`0, 1, 2`) of every 4-byte pixel while ignoring byte `3`.
            const MASK_RGB: u64 = 0x7777_7777_7777_7777;

            /// In-lane `vpshufb` mask replicating each BGRA pixel's green channel into its
            /// B and R slots while zeroing G and A (`0x80`).
            const SHUF_G: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut p = 0;
                while p < 16 {
                    let g = ((p & 3) * 4 + 1) as u8;
                    t[p * 4] = g;
                    t[p * 4 + 1] = 0x80;
                    t[p * 4 + 2] = g;
                    t[p * 4 + 3] = 0x80;
                    p += 1;
                }
                t
            };

            /// Cross-lane `vpermb` indices packing 16 `SubGreen` BGRA pixels into 48
            /// contiguous BGR bytes (`0..48`) followed by 16 alpha bytes (`48..64`).
            const PACK_BGR_ALPHA_IDX: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut p = 0;
                while p < 16 {
                    t[p * 3] = (p * 4) as u8;
                    t[p * 3 + 1] = (p * 4 + 1) as u8;
                    t[p * 3 + 2] = (p * 4 + 2) as u8;
                    t[48 + p] = (p * 4 + 3) as u8;
                    p += 1;
                }
                t
            };

            /// Cross-lane `vpermb` indices unpacking 48 BGR bytes (`0..48`) into 16 BGRA
            /// pixels, sourcing each alpha byte from lane `48` (zero-masked by load).
            const UNPACK_BGR_IDX: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut p = 0;
                while p < 16 {
                    t[p * 4] = (p * 3) as u8;
                    t[p * 4 + 1] = (p * 3 + 1) as u8;
                    t[p * 4 + 2] = (p * 3 + 2) as u8;
                    t[p * 4 + 3] = 48;
                    p += 1;
                }
                t
            };

            /// Cross-lane `vpermt2b` indices unpacking 48 BGR bytes (`0..48` from source `a`)
            /// and 16 alpha bytes (`64..80` from source `b`) into 16 BGRA pixels.
            const UNPACK_BGRA_IDX: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut p = 0;
                while p < 16 {
                    t[p * 4] = (p * 3) as u8;
                    t[p * 4 + 1] = (p * 3 + 1) as u8;
                    t[p * 4 + 2] = (p * 3 + 2) as u8;
                    t[p * 4 + 3] = (64 + p) as u8;
                    p += 1;
                }
                t
            };

            /// Builds a `vpermt2b` index table shifting the 128-byte concatenation `(prev, cur)`
            /// left by `shift` bytes (`0..64` selects `prev`, `64..128` selects `cur`).
            const fn concat_shift_left_idx(shift: usize) -> [u8; 64] {
                let mut t = [0u8; 64];
                let mut j = 0;
                while j < 64 {
                    t[j] = (64 - shift + j) as u8;
                    j += 1;
                }
                t
            }

            /// Shifts `(prev, cur)` left by 3 bytes (`prev[61..64]` followed by `cur[0..61]`).
            const SHIFT_LEFT3_IDX: [u8; 64] = concat_shift_left_idx(3);

            /// Shifts `(prev, cur)` left by 1 byte (`prev[63]` followed by `cur[0..63]`).
            const SHIFT_LEFT1_IDX: [u8; 64] = concat_shift_left_idx(1);

            /// Builds a `vpermb` index table shifting a 64-byte vector left by `shift` bytes
            /// (lanes `< shift` are zeroed via opmask `u64::MAX << shift`).
            const fn intra_shift_left_idx(shift: usize) -> [u8; 64] {
                let mut t = [0u8; 64];
                let mut j = shift;
                while j < 64 {
                    t[j] = (j - shift) as u8;
                    j += 1;
                }
                t
            }

            const PREFIX_SH3_IDX: [u8; 64] = intra_shift_left_idx(3);
            const PREFIX_SH6_IDX: [u8; 64] = intra_shift_left_idx(6);
            const PREFIX_SH12_IDX: [u8; 64] = intra_shift_left_idx(12);
            const PREFIX_SH24_IDX: [u8; 64] = intra_shift_left_idx(24);

            /// Repeats pixel 15 (`bytes 45, 46, 47`) across all 16 3-byte pixel slots (`0..48`).
            const CARRY_LAST3_IDX: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut j = 0;
                while j < 64 {
                    t[j] = (45 + (j % 3)) as u8;
                    j += 1;
                }
                t
            };

            /// Broadcasts byte `15` across lanes `16..32` and byte `47` across lanes `48..64`
            /// for the first cross-lane step of the 64-byte `Sub1` prefix sum.
            const SUB1_STEP5_IDX: [u8; 64] = {
                let mut t = [0u8; 64];
                let mut j = 16;
                while j < 32 {
                    t[j] = 15;
                    j += 1;
                }
                j = 48;
                while j < 64 {
                    t[j] = 47;
                    j += 1;
                }
                t
            };

            #[inline]
            fn load_table(table: &[u8; 64]) -> __m512i {
                // SAFETY: `table` is 64 bytes and AVX-512 is statically enabled.
                unsafe { _mm512_loadu_si512(table.as_ptr().cast()) }
            }

            #[inline]
            fn prefix_sum_sub3_48(
                mut v: __m512i,
                sh3: __m512i,
                sh6: __m512i,
                sh12: __m512i,
                sh24: __m512i,
            ) -> __m512i {
                // SAFETY: Register-to-register AVX-512BW + VBMI ops; statically enabled.
                unsafe {
                    v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 3, sh3, v));
                    v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 6, sh6, v));
                    v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 12, sh12, v));
                    _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 24, sh24, v))
                }
            }

            pub(crate) struct Kernel;

            impl RowKernel for Kernel {
                const STREAM_ALIGN: usize = 64;

                #[inline]
                fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    dst_bgr: &mut [u8],
                    dst_alpha: &mut [u8],
                ) {
                    let tw = src.len();
                    debug_assert_eq!(dst_bgr.len(), tw * 3);
                    debug_assert!(!HAS_ALPHA || dst_alpha.len() == tw);

                    // Masked 48-byte stores suppress faults and never touch bytes `48..64`,
                    // so every full 16-pixel block needs zero tail margin.
                    let lim = (tw / 16) * 16;
                    let mut x = 0usize;
                    let mut di = 0usize;

                    // SAFETY: AVX-512BW/VL/VBMI are statically enabled; `x + 16 <= tw`,
                    // `di + 48 <= dst_bgr.len()`, and `x + 16 <= dst_alpha.len()` when `HAS_ALPHA`.
                    unsafe {
                        let shuf_g = load_table(&SHUF_G);
                        let pack_idx = load_table(&PACK_BGR_ALPHA_IDX);

                        while x < lim {
                            let v = _mm512_loadu_si512(src.ptr().add(x).cast());
                            let g = _mm512_shuffle_epi8(v, shuf_g);
                            let sg = _mm512_sub_epi8(v, g);
                            let packed = _mm512_permutexvar_epi8(pack_idx, sg);
                            _mm512_mask_storeu_epi8(
                                dst_bgr.as_mut_ptr().add(di).cast(),
                                MASK_48,
                                packed,
                            );
                            if HAS_ALPHA {
                                _mm_storeu_si128(
                                    dst_alpha.as_mut_ptr().add(x).cast::<__m128i>(),
                                    _mm512_extracti32x4_epi32::<3>(packed),
                                );
                            }
                            x += 16;
                            di += 48;
                        }
                    }

                    for (off, bgr) in dst_bgr[lim * 3..]
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let i = lim + off;
                        // SAFETY: `i < tw == src.len()`.
                        let px = unsafe { src.ptr().add(i).read() };
                        bgr[0] = px.0.wrapping_sub(px.1);
                        bgr[1] = px.1;
                        bgr[2] = px.2.wrapping_sub(px.1);
                        if HAS_ALPHA {
                            dst_alpha[i] = px.3;
                        }
                    }
                }

                #[inline]
                fn sad_signed_bytes(data: &[u8]) -> u64 {
                    let lim = (data.len() / 64) * 64;
                    let mut sum = 0u64;
                    // SAFETY: AVX-512BW is statically enabled and `i + 64 <= data.len()`.
                    unsafe {
                        let zero = _mm512_setzero_si512();
                        let mut acc = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_loadu_si512(data.as_ptr().add(i).cast());
                            let ab = _mm512_abs_epi8(v);
                            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(ab, zero));
                            i += 64;
                        }
                        sum += _mm512_reduce_add_epi64(acc) as u64;
                    }
                    for &b in &data[lim..] {
                        sum += u64::from((b as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64 {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    let mut sum = 0u64;
                    // SAFETY: AVX-512BW/VBMI are statically enabled and `i + 64 <= len`.
                    unsafe {
                        let zero = _mm512_setzero_si512();
                        let shift_idx = load_table(&SHIFT_LEFT3_IDX);
                        let mut acc = _mm512_setzero_si512();
                        let mut prev = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                            let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                            let diff = _mm512_sub_epi8(v, shifted);
                            prev = v;
                            _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), diff);
                            acc =
                                _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
                            i += 64;
                        }
                        sum += _mm512_reduce_add_epi64(acc) as u64;
                    }
                    for j in lim..len {
                        let p = if j >= 3 { src[j - 3] } else { 0 };
                        let d = src[j].wrapping_sub(p);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_up_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    let mut sum = 0u64;
                    // SAFETY: AVX-512BW is statically enabled and `i + 64 <= len`.
                    unsafe {
                        let zero = _mm512_setzero_si512();
                        let mut acc = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm512_loadu_si512(cur.as_ptr().add(i).cast());
                            let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                            let diff = _mm512_sub_epi8(c, u);
                            _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), diff);
                            acc =
                                _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
                            i += 64;
                        }
                        sum += _mm512_reduce_add_epi64(acc) as u64;
                    }
                    for j in lim..len {
                        let d = cur[j].wrapping_sub(prev[j]);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_subup3_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    let mut sum = 0u64;
                    // SAFETY: AVX-512BW/VBMI are statically enabled and `i + 64 <= len`.
                    unsafe {
                        let zero = _mm512_setzero_si512();
                        let shift_idx = load_table(&SHIFT_LEFT3_IDX);
                        let mut acc = _mm512_setzero_si512();
                        let mut prev_up = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm512_loadu_si512(cur.as_ptr().add(i).cast());
                            let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                            let up = _mm512_sub_epi8(c, u);
                            let shifted = _mm512_permutex2var_epi8(prev_up, shift_idx, up);
                            let diff = _mm512_sub_epi8(up, shifted);
                            prev_up = up;
                            _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), diff);
                            acc =
                                _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
                            i += 64;
                        }
                        sum += _mm512_reduce_add_epi64(acc) as u64;
                    }
                    for j in lim..len {
                        let up = cur[j].wrapping_sub(prev[j]);
                        let prev_u = if j >= 3 {
                            cur[j - 3].wrapping_sub(prev[j - 3])
                        } else {
                            0
                        };
                        let d = up.wrapping_sub(prev_u);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 48) * 48;
                    // SAFETY: AVX-512BW/VBMI are statically enabled; masked 48-byte loads/stores
                    // strictly access `i..i + 48 <= len`.
                    unsafe {
                        let sh3 = load_table(&PREFIX_SH3_IDX);
                        let sh6 = load_table(&PREFIX_SH6_IDX);
                        let sh12 = load_table(&PREFIX_SH12_IDX);
                        let sh24 = load_table(&PREFIX_SH24_IDX);
                        let carry_idx = load_table(&CARRY_LAST3_IDX);
                        let mut carry = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_maskz_loadu_epi8(MASK_48, src.as_ptr().add(i).cast());
                            let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                            let v = _mm512_add_epi8(v, carry);
                            carry = _mm512_permutexvar_epi8(carry_idx, v);
                            _mm512_mask_storeu_epi8(dst.as_mut_ptr().add(i).cast(), MASK_48, v);
                            i += 48;
                        }
                    }
                    for j in lim..len {
                        let p = if j >= 3 { dst[j - 3] } else { 0 };
                        dst[j] = src[j].wrapping_add(p);
                    }
                }

                #[inline]
                fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    // SAFETY: AVX-512BW is statically enabled and `i + 64 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let d = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                            let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                            _mm512_storeu_si512(
                                dst.as_mut_ptr().add(i).cast(),
                                _mm512_add_epi8(d, u),
                            );
                            i += 64;
                        }
                    }
                    for j in lim..len {
                        dst[j] = src[j].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    debug_assert!(len.is_multiple_of(3));
                    let lim = (len / 48) * 48;
                    // SAFETY: AVX-512BW/VBMI are statically enabled; masked 48-byte loads/stores
                    // strictly access `i..i + 48 <= len`.
                    unsafe {
                        let sh3 = load_table(&PREFIX_SH3_IDX);
                        let sh6 = load_table(&PREFIX_SH6_IDX);
                        let sh12 = load_table(&PREFIX_SH12_IDX);
                        let sh24 = load_table(&PREFIX_SH24_IDX);
                        let carry_idx = load_table(&CARRY_LAST3_IDX);
                        let mut carry = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_maskz_loadu_epi8(MASK_48, src.as_ptr().add(i).cast());
                            let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                            let v = _mm512_add_epi8(v, carry);
                            carry = _mm512_permutexvar_epi8(carry_idx, v);
                            let u = _mm512_maskz_loadu_epi8(MASK_48, prev.as_ptr().add(i).cast());
                            _mm512_mask_storeu_epi8(
                                dst.as_mut_ptr().add(i).cast(),
                                MASK_48,
                                _mm512_add_epi8(v, u),
                            );
                            i += 48;
                        }
                    }
                    let mut acc = [0u8; 3];
                    if lim >= 3 {
                        for ch in 0..3 {
                            acc[ch] = dst[lim - 3 + ch].wrapping_sub(prev[lim - 3 + ch]);
                        }
                    }
                    for j in lim..len {
                        let ch = j % 3;
                        acc[ch] = acc[ch].wrapping_add(src[j]);
                        dst[j] = acc[ch].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
                    src_bgr: &[u8],
                    alpha_row: &[u8],
                    const_alpha: u8,
                    dst: &mut [Vec4u8],
                ) {
                    let tw = dst.len();
                    debug_assert_eq!(src_bgr.len(), tw * 3);
                    debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(64));

                    // Masked 48-byte loads never touch bytes `48..64`, so every full
                    // 16-pixel block reads strictly in bounds with zero tail over-read.
                    let lim = (tw / 16) * 16;
                    let mut x = 0usize;
                    let mut si = 0usize;

                    // SAFETY: AVX-512BW/VBMI are statically enabled; `si + 48 <= src_bgr.len()`,
                    // `x + 16 <= dst.len()`, `x + 16 <= alpha_row.len()` when `!CONST_ALPHA`,
                    // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
                    unsafe {
                        let shuf_g = load_table(&SHUF_G);
                        let unpack_bgr = load_table(&UNPACK_BGR_IDX);
                        let unpack_bgra = load_table(&UNPACK_BGRA_IDX);
                        let const_alpha_vec =
                            _mm512_set1_epi32((u32::from(const_alpha) << 24) as i32);

                        while x < lim {
                            let bgr =
                                _mm512_maskz_loadu_epi8(MASK_48, src_bgr.as_ptr().add(si).cast());
                            let bgra = if CONST_ALPHA {
                                let sg = _mm512_permutexvar_epi8(unpack_bgr, bgr);
                                let g = _mm512_shuffle_epi8(sg, shuf_g);
                                _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                            } else {
                                let a128 =
                                    _mm_loadu_si128(alpha_row.as_ptr().add(x).cast::<__m128i>());
                                let sg = _mm512_permutex2var_epi8(
                                    bgr,
                                    unpack_bgra,
                                    _mm512_zextsi128_si512(a128),
                                );
                                let g = _mm512_shuffle_epi8(sg, shuf_g);
                                _mm512_add_epi8(sg, g)
                            };
                            let p = dst.as_mut_ptr().add(x).cast::<__m512i>();
                            if STREAM {
                                _mm512_stream_si512(p, bgra);
                            } else {
                                _mm512_storeu_si512(p.cast(), bgra);
                            }
                            x += 16;
                            si += 48;
                        }
                    }

                    for i in lim..tw {
                        let s = i * 3;
                        let g = src_bgr[s + 1];
                        let a = if CONST_ALPHA {
                            const_alpha
                        } else {
                            alpha_row[i]
                        };
                        dst[i] = Vec4u8(
                            src_bgr[s].wrapping_add(g),
                            g,
                            src_bgr[s + 2].wrapping_add(g),
                            a,
                        );
                    }
                }

                #[inline]
                unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8) {
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(64));
                    let lim = (dst.len() / 16) * 16;
                    // SAFETY: AVX-512F is statically enabled; each iteration writes 16 `Vec4u8`
                    // (64 bytes), and caller upholds 64-byte alignment when `STREAM` is `true`.
                    unsafe {
                        let word = u32::from_le_bytes([pixel.0, pixel.1, pixel.2, pixel.3]) as i32;
                        let vec = _mm512_set1_epi32(word);
                        let mut x = 0usize;
                        while x < lim {
                            let p = dst.as_mut_ptr().add(x).cast::<__m512i>();
                            if STREAM {
                                _mm512_stream_si512(p, vec);
                            } else {
                                _mm512_storeu_si512(p.cast(), vec);
                            }
                            x += 16;
                        }
                    }
                    dst[lim..].fill(pixel);
                }

                #[inline]
                fn is_constant_plane(data: &[u8]) -> Option<u8> {
                    let first = *data.first()?;
                    let lim = (data.len() / 64) * 64;
                    // SAFETY: AVX-512BW is statically enabled and `i + 64 <= data.len()`.
                    unsafe {
                        let target = _mm512_set1_epi8(first as i8);
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_loadu_si512(data.as_ptr().add(i).cast());
                            if _mm512_cmpeq_epi8_mask(v, target) != u64::MAX {
                                return None;
                            }
                            i += 64;
                        }
                    }
                    data[lim..].iter().all(|&b| b == first).then_some(first)
                }

                #[inline]
                fn row_sub1_forward(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    // SAFETY: AVX-512BW/VBMI are statically enabled and `i + 64 <= len`.
                    unsafe {
                        let shift_idx = load_table(&SHIFT_LEFT1_IDX);
                        let mut prev = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                            let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                            _mm512_storeu_si512(
                                dst.as_mut_ptr().add(i).cast(),
                                _mm512_sub_epi8(v, shifted),
                            );
                            prev = v;
                            i += 64;
                        }
                    }
                    let mut p = if lim > 0 { src[lim - 1] } else { 0 };
                    for j in lim..len {
                        let cur = src[j];
                        dst[j] = cur.wrapping_sub(p);
                        p = cur;
                    }
                }

                #[inline]
                fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    // SAFETY: AVX-512BW is statically enabled and `i + 64 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm512_loadu_si512(cur.as_ptr().add(i).cast());
                            let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                            _mm512_storeu_si512(
                                dst.as_mut_ptr().add(i).cast(),
                                _mm512_sub_epi8(c, u),
                            );
                            i += 64;
                        }
                    }
                    for j in lim..len {
                        dst[j] = cur[j].wrapping_sub(prev[j]);
                    }
                }

                #[inline]
                fn row_sub1_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 64) * 64;
                    // SAFETY: AVX-512BW/VBMI are statically enabled and `i + 64 <= len`.
                    unsafe {
                        let step5_idx = load_table(&SUB1_STEP5_IDX);
                        let bcast_31 = _mm512_set1_epi8(31);
                        let bcast_63 = _mm512_set1_epi8(63);
                        let mut carry = _mm512_setzero_si512();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                            // In-lane 128-bit prefix sums (4 independent 16-byte lanes).
                            v = _mm512_add_epi8(v, _mm512_bslli_epi128::<1>(v));
                            v = _mm512_add_epi8(v, _mm512_bslli_epi128::<2>(v));
                            v = _mm512_add_epi8(v, _mm512_bslli_epi128::<4>(v));
                            v = _mm512_add_epi8(v, _mm512_bslli_epi128::<8>(v));
                            // Propagate lane 0 tail (`v[15]`) into lane 1 (`16..32`) and
                            // lane 2 tail (`v[47]`) into lane 3 (`48..64`).
                            v = _mm512_add_epi8(
                                v,
                                _mm512_maskz_permutexvar_epi8(0xFFFF_0000_FFFF_0000, step5_idx, v),
                            );
                            // Propagate lower-half tail (`v[31]`) into upper half (`32..64`).
                            v = _mm512_add_epi8(
                                v,
                                _mm512_maskz_permutexvar_epi8(0xFFFF_FFFF_0000_0000, bcast_31, v),
                            );
                            v = _mm512_add_epi8(v, carry);
                            carry = _mm512_permutexvar_epi8(bcast_63, v);
                            _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), v);
                            i += 64;
                        }
                    }
                    let mut acc = if lim > 0 { dst[lim - 1] } else { 0 };
                    for j in lim..len {
                        acc = acc.wrapping_add(src[j]);
                        dst[j] = acc;
                    }
                }

                #[inline]
                fn diff_and_update_row<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    shadow: &mut [Vec4u8],
                ) -> bool {
                    let n = src.len();
                    debug_assert_eq!(shadow.len(), n);
                    let src_p = src.ptr();
                    let dst_p = shadow.as_mut_ptr();
                    let lim = (n / 16) * 16;
                    // SAFETY: AVX-512BW is statically enabled, `src_p` and `dst_p` are
                    // valid for `n` `Vec4u8`s and non-overlapping (`src` is shm/caller,
                    // `shadow` is exclusively borrowed), and `i + 16 <= n`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let s = _mm512_loadu_si512(src_p.add(i).cast());
                            let d = _mm512_loadu_si512(dst_p.add(i).cast());
                            let neq = if HAS_ALPHA {
                                _mm512_cmpneq_epi8_mask(s, d)
                            } else {
                                _mm512_mask_cmpneq_epi8_mask(MASK_RGB, s, d)
                            };
                            if neq != 0 {
                                _mm512_storeu_si512(dst_p.add(i).cast(), s);
                                if i + 16 < n {
                                    src.split_at(i + 16)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[i + 16..]);
                                }
                                return true;
                            }
                            i += 16;
                        }
                        for j in lim..n {
                            let s = src_p.add(j).read();
                            let d = dst_p.add(j).read();
                            let differs = if HAS_ALPHA {
                                s != d
                            } else {
                                s.0 != d.0 || s.1 != d.1 || s.2 != d.2
                            };
                            if differs {
                                dst_p.add(j).write(s);
                                if j + 1 < n {
                                    src.split_at(j + 1)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[j + 1..]);
                                }
                                return true;
                            }
                        }
                    }
                    false
                }

                #[inline]
                fn fence_stream() {
                    // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
                    unsafe { _mm_sfence() };
                }
            }
        }
        pub(super) use imp::Kernel;
    },
    all(target_feature = "avx2", not(miri)) => {
        mod imp {
            use std::arch::x86_64::__m128i;
            use std::arch::x86_64::__m256i;
            use std::arch::x86_64::_mm_add_epi8;
            use std::arch::x86_64::_mm_cvtsi64_si128;
            use std::arch::x86_64::_mm_cvtsi128_si32;
            use std::arch::x86_64::_mm_loadu_si128;
            use std::arch::x86_64::_mm_set_epi8;
            use std::arch::x86_64::_mm_set1_epi8;
            use std::arch::x86_64::_mm_setzero_si128;
            use std::arch::x86_64::_mm_sfence;
            use std::arch::x86_64::_mm_shuffle_epi8;
            use std::arch::x86_64::_mm_slli_si128;
            use std::arch::x86_64::_mm_srli_si128;
            use std::arch::x86_64::_mm_storeu_si128;
            use std::arch::x86_64::_mm256_abs_epi8;
            use std::arch::x86_64::_mm256_add_epi8;
            use std::arch::x86_64::_mm256_add_epi64;
            use std::arch::x86_64::_mm256_alignr_epi8;
            use std::arch::x86_64::_mm256_and_si256;
            use std::arch::x86_64::_mm256_castsi256_si128;
            use std::arch::x86_64::_mm256_extracti128_si256;
            use std::arch::x86_64::_mm256_loadu_si256;
            use std::arch::x86_64::_mm256_or_si256;
            use std::arch::x86_64::_mm256_permute2x128_si256;
            use std::arch::x86_64::_mm256_sad_epu8;
            use std::arch::x86_64::_mm256_set_epi8;
            use std::arch::x86_64::_mm256_set_m128i;
            use std::arch::x86_64::_mm256_set1_epi8;
            use std::arch::x86_64::_mm256_set1_epi32;
            use std::arch::x86_64::_mm256_setzero_si256;
            use std::arch::x86_64::_mm256_shuffle_epi8;
            use std::arch::x86_64::_mm256_slli_si256;
            use std::arch::x86_64::_mm256_storeu_si256;
            use std::arch::x86_64::_mm256_stream_si256;
            use std::arch::x86_64::_mm256_sub_epi8;
            use std::arch::x86_64::_mm256_testz_si256;
            use std::arch::x86_64::_mm256_xor_si256;

            use crate::buffer_pointer::BufferPointer;
            use crate::filtering::RowKernel;
            use crate::vec4u8::Vec4u8;

            #[inline]
            fn hsum_epi64(acc: __m256i) -> u64 {
                let mut tmp = [0u64; 4];
                // SAFETY: `tmp` is 32 bytes and AVX2 is statically enabled.
                unsafe {
                    _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), acc);
                }
                tmp[0] + tmp[1] + tmp[2] + tmp[3]
            }

            pub(crate) struct Kernel;

            impl RowKernel for Kernel {
                const STREAM_ALIGN: usize = 32;

                #[inline]
                fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    dst_bgr: &mut [u8],
                    dst_alpha: &mut [u8],
                ) {
                    let tw = src.len();
                    debug_assert_eq!(dst_bgr.len(), tw * 3);
                    debug_assert!(!HAS_ALPHA || dst_alpha.len() == tw);

                    // Each 8-pixel iteration writes 16 bytes at `di` and 16 bytes at `di + 12`
                    // (reaching `di + 28`). Stopping at `lim <= tw - 2` ensures `di + 28 <= tw * 3`
                    // with zero out-of-row writes and no slack bytes required.
                    let lim = if tw >= 10 { ((tw - 2) / 8) * 8 } else { 0 };
                    let mut x = 0usize;
                    let mut di = 0usize;

                    // SAFETY: AVX2 is statically enabled; `x + 8 <= tw` and `di + 28 <= dst_bgr.len()`
                    // for all iterations `x < lim`.
                    unsafe {
                        let shuf_g = _mm256_set_epi8(
                            -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1, 13, -1,
                            13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
                        );
                        let pack12 = _mm256_set_epi8(
                            -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, -1, -1, -1, -1,
                            14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
                        );
                        let shuf_a = _mm256_set_epi8(
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3, -1, -1,
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3,
                        );

                        while x < lim {
                            let v = _mm256_loadu_si256(src.ptr().add(x).cast::<__m256i>());
                            let g = _mm256_shuffle_epi8(v, shuf_g);
                            let sg = _mm256_sub_epi8(v, g);
                            let pk = _mm256_shuffle_epi8(sg, pack12);
                            _mm_storeu_si128(
                                dst_bgr.as_mut_ptr().add(di).cast::<__m128i>(),
                                _mm256_castsi256_si128(pk),
                            );
                            _mm_storeu_si128(
                                dst_bgr.as_mut_ptr().add(di + 12).cast::<__m128i>(),
                                _mm256_extracti128_si256::<1>(pk),
                            );
                            if HAS_ALPHA {
                                let a_pk = _mm256_shuffle_epi8(v, shuf_a);
                                let a_lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(a_pk)) as u32;
                                let a_hi =
                                    _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(a_pk)) as u32;
                                dst_alpha[x..x + 4].copy_from_slice(&a_lo.to_le_bytes());
                                dst_alpha[x + 4..x + 8].copy_from_slice(&a_hi.to_le_bytes());
                            }
                            x += 8;
                            di += 24;
                        }
                    }

                    for (off, bgr) in dst_bgr[lim * 3..]
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let i = lim + off;
                        // SAFETY: `i < tw == src.len()`.
                        let px = unsafe { src.ptr().add(i).read() };
                        bgr[0] = px.0.wrapping_sub(px.1);
                        bgr[1] = px.1;
                        bgr[2] = px.2.wrapping_sub(px.1);
                        if HAS_ALPHA {
                            dst_alpha[i] = px.3;
                        }
                    }
                }

                #[inline]
                fn sad_signed_bytes(data: &[u8]) -> u64 {
                    let lim = (data.len() / 32) * 32;
                    let mut sum = 0u64;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= data.len()`.
                    unsafe {
                        let zero = _mm256_setzero_si256();
                        let mut acc = _mm256_setzero_si256();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm256_loadu_si256(data.as_ptr().add(i).cast::<__m256i>());
                            let ab = _mm256_abs_epi8(v);
                            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(ab, zero));
                            i += 32;
                        }
                        sum += hsum_epi64(acc);
                    }
                    for &b in &data[lim..] {
                        sum += u64::from((b as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64 {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    let mut sum = 0u64;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let zero = _mm256_setzero_si256();
                        let mut acc = _mm256_setzero_si256();
                        let mut prev = _mm256_setzero_si256();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>());
                            let prev_lane = _mm256_permute2x128_si256::<0x21>(prev, v);
                            let shifted = _mm256_alignr_epi8::<13>(v, prev_lane);
                            let diff = _mm256_sub_epi8(v, shifted);
                            prev = v;
                            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), diff);
                            acc =
                                _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
                            i += 32;
                        }
                        sum += hsum_epi64(acc);
                    }
                    for j in lim..len {
                        let p = if j >= 3 { src[j - 3] } else { 0 };
                        let d = src[j].wrapping_sub(p);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_up_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    let mut sum = 0u64;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let zero = _mm256_setzero_si256();
                        let mut acc = _mm256_setzero_si256();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm256_loadu_si256(cur.as_ptr().add(i).cast::<__m256i>());
                            let u = _mm256_loadu_si256(prev.as_ptr().add(i).cast::<__m256i>());
                            let diff = _mm256_sub_epi8(c, u);
                            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), diff);
                            acc =
                                _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
                            i += 32;
                        }
                        sum += hsum_epi64(acc);
                    }
                    for j in lim..len {
                        let d = cur[j].wrapping_sub(prev[j]);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_subup3_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    let mut sum = 0u64;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let zero = _mm256_setzero_si256();
                        let mut acc = _mm256_setzero_si256();
                        let mut prev_up = _mm256_setzero_si256();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm256_loadu_si256(cur.as_ptr().add(i).cast::<__m256i>());
                            let u = _mm256_loadu_si256(prev.as_ptr().add(i).cast::<__m256i>());
                            let up = _mm256_sub_epi8(c, u);
                            let prev_lane = _mm256_permute2x128_si256::<0x21>(prev_up, up);
                            let shifted = _mm256_alignr_epi8::<13>(up, prev_lane);
                            let diff = _mm256_sub_epi8(up, shifted);
                            prev_up = up;
                            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), diff);
                            acc =
                                _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
                            i += 32;
                        }
                        sum += hsum_epi64(acc);
                    }
                    for j in lim..len {
                        let up = cur[j].wrapping_sub(prev[j]);
                        let prev_u = if j >= 3 {
                            cur[j - 3].wrapping_sub(prev[j - 3])
                        } else {
                            0
                        };
                        let d = up.wrapping_sub(prev_u);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    // Step by 12 bytes (4 RGB pixels) while guaranteeing 16-byte loads/stores
                    // stay strictly inside `0..len` (`i + 16 <= len`).
                    let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
                    // SAFETY: AVX2 implies SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
                    unsafe {
                        // Repeats pixel 3 (`bytes 9, 10, 11`) across all four 3-byte pixel slots.
                        let shuf_last3 =
                            _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
                        let mut prev_px = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                            v = _mm_add_epi8(v, prev_px);
                            prev_px = _mm_shuffle_epi8(v, shuf_last3);
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), v);
                            i += 12;
                        }
                    }
                    for j in lim..len {
                        let p = if j >= 3 { dst[j - 3] } else { 0 };
                        dst[j] = src[j].wrapping_add(p);
                    }
                }

                #[inline]
                fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let d = _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>());
                            let u = _mm256_loadu_si256(prev.as_ptr().add(i).cast::<__m256i>());
                            _mm256_storeu_si256(
                                dst.as_mut_ptr().add(i).cast::<__m256i>(),
                                _mm256_add_epi8(d, u),
                            );
                            i += 32;
                        }
                    }
                    for j in lim..len {
                        dst[j] = src[j].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    debug_assert!(len.is_multiple_of(3));
                    let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
                    // SAFETY: AVX2 implies SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
                    unsafe {
                        let shuf_last3 =
                            _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
                        let mut prev_px = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                            v = _mm_add_epi8(v, prev_px);
                            prev_px = _mm_shuffle_epi8(v, shuf_last3);
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            _mm_storeu_si128(
                                dst.as_mut_ptr().add(i).cast::<__m128i>(),
                                _mm_add_epi8(v, u),
                            );
                            i += 12;
                        }
                    }
                    let mut acc = [0u8; 3];
                    if lim >= 3 {
                        for ch in 0..3 {
                            acc[ch] = dst[lim - 3 + ch].wrapping_sub(prev[lim - 3 + ch]);
                        }
                    }
                    for j in lim..len {
                        let ch = j % 3;
                        acc[ch] = acc[ch].wrapping_add(src[j]);
                        dst[j] = acc[ch].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
                    src_bgr: &[u8],
                    alpha_row: &[u8],
                    const_alpha: u8,
                    dst: &mut [Vec4u8],
                ) {
                    let tw = dst.len();
                    debug_assert_eq!(src_bgr.len(), tw * 3);
                    debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(32));

                    let full8 = (tw / 8) * 8;
                    let lim = if tw >= 10 { ((tw - 2) / 8) * 8 } else { 0 };
                    let mut x = 0usize;
                    let mut si = 0usize;

                    // SAFETY: AVX2 is statically enabled; `si + 28 <= src_bgr.len()` inside the
                    // main loop, the optional final 8-pixel block reads `si..si + 24 <= src_bgr.len()`,
                    // and caller upholds 32-byte alignment on `dst` when `STREAM` is `true`.
                    unsafe {
                        let shuf_g = _mm256_set_epi8(
                            -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1, 13, -1,
                            13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
                        );
                        let unpack12 = _mm256_set_epi8(
                            -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9,
                            -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0,
                        );
                        let const_alpha_vec =
                            _mm256_set1_epi32((u32::from(const_alpha) << 24) as i32);
                        let shuf_a_lo = _mm_set_epi8(
                            3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1,
                        );
                        let shuf_a_hi = _mm_set_epi8(
                            7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4, -1, -1, -1,
                        );

                        while x < lim {
                            let lo = _mm_loadu_si128(src_bgr.as_ptr().add(si).cast::<__m128i>());
                            let hi =
                                _mm_loadu_si128(src_bgr.as_ptr().add(si + 12).cast::<__m128i>());
                            let v = _mm256_set_m128i(hi, lo);
                            let sg = _mm256_shuffle_epi8(v, unpack12);
                            let g = _mm256_shuffle_epi8(sg, shuf_g);
                            let alpha_vec = if CONST_ALPHA {
                                const_alpha_vec
                            } else {
                                let a64 = _mm_cvtsi64_si128(i64::from_le_bytes(
                                    alpha_row[x..x + 8].try_into().unwrap(),
                                ));
                                _mm256_set_m128i(
                                    _mm_shuffle_epi8(a64, shuf_a_hi),
                                    _mm_shuffle_epi8(a64, shuf_a_lo),
                                )
                            };
                            let bgra = _mm256_or_si256(_mm256_add_epi8(sg, g), alpha_vec);
                            let p = dst.as_mut_ptr().add(x).cast::<__m256i>();
                            if STREAM {
                                _mm256_stream_si256(p, bgra);
                            } else {
                                _mm256_storeu_si256(p, bgra);
                            }
                            x += 8;
                            si += 24;
                        }
                        if x < full8 {
                            // Final 8-pixel block: load `si + 8..si + 24` in bounds and shift
                            // right 4 bytes so lanes `0..12` hold `si + 12..si + 24`, keeping
                            // `STREAM` rows 100% non-temporal with zero WC/cached split.
                            let lo = _mm_loadu_si128(src_bgr.as_ptr().add(si).cast::<__m128i>());
                            let hi = _mm_srli_si128::<4>(_mm_loadu_si128(
                                src_bgr.as_ptr().add(si + 8).cast::<__m128i>(),
                            ));
                            let v = _mm256_set_m128i(hi, lo);
                            let sg = _mm256_shuffle_epi8(v, unpack12);
                            let g = _mm256_shuffle_epi8(sg, shuf_g);
                            let alpha_vec = if CONST_ALPHA {
                                const_alpha_vec
                            } else {
                                let a64 = _mm_cvtsi64_si128(i64::from_le_bytes(
                                    alpha_row[x..x + 8].try_into().unwrap(),
                                ));
                                _mm256_set_m128i(
                                    _mm_shuffle_epi8(a64, shuf_a_hi),
                                    _mm_shuffle_epi8(a64, shuf_a_lo),
                                )
                            };
                            let bgra = _mm256_or_si256(_mm256_add_epi8(sg, g), alpha_vec);
                            let p = dst.as_mut_ptr().add(x).cast::<__m256i>();
                            if STREAM {
                                _mm256_stream_si256(p, bgra);
                            } else {
                                _mm256_storeu_si256(p, bgra);
                            }
                        }
                    }

                    for i in full8..tw {
                        let s = i * 3;
                        let g = src_bgr[s + 1];
                        let a = if CONST_ALPHA {
                            const_alpha
                        } else {
                            alpha_row[i]
                        };
                        dst[i] = Vec4u8(
                            src_bgr[s].wrapping_add(g),
                            g,
                            src_bgr[s + 2].wrapping_add(g),
                            a,
                        );
                    }
                }

                #[inline]
                unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8) {
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(32));
                    let lim = (dst.len() / 8) * 8;
                    // SAFETY: AVX2 is statically enabled; each iteration writes 8 `Vec4u8` (32 bytes),
                    // and caller upholds 32-byte alignment when `STREAM` is `true`.
                    unsafe {
                        let word = u32::from_le_bytes([pixel.0, pixel.1, pixel.2, pixel.3]) as i32;
                        let vec = _mm256_set1_epi32(word);
                        let mut x = 0usize;
                        while x < lim {
                            let p = dst.as_mut_ptr().add(x).cast::<__m256i>();
                            if STREAM {
                                _mm256_stream_si256(p, vec);
                            } else {
                                _mm256_storeu_si256(p, vec);
                            }
                            x += 8;
                        }
                    }
                    dst[lim..].fill(pixel);
                }

                #[inline]
                fn is_constant_plane(data: &[u8]) -> Option<u8> {
                    let first = *data.first()?;
                    let lim = (data.len() / 32) * 32;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= data.len()`.
                    unsafe {
                        let target = _mm256_set1_epi8(first as i8);
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm256_loadu_si256(data.as_ptr().add(i).cast::<__m256i>());
                            let diff = _mm256_xor_si256(v, target);
                            if _mm256_testz_si256(diff, diff) == 0 {
                                return None;
                            }
                            i += 32;
                        }
                    }
                    data[lim..].iter().all(|&b| b == first).then_some(first)
                }

                #[inline]
                fn row_sub1_forward(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let mut prev = _mm256_setzero_si256();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>());
                            let prev_lane = _mm256_permute2x128_si256::<0x21>(prev, v);
                            let shifted = _mm256_alignr_epi8::<15>(v, prev_lane);
                            _mm256_storeu_si256(
                                dst.as_mut_ptr().add(i).cast::<__m256i>(),
                                _mm256_sub_epi8(v, shifted),
                            );
                            prev = v;
                            i += 32;
                        }
                    }
                    let mut p = if lim > 0 { src[lim - 1] } else { 0 };
                    for j in lim..len {
                        let cur = src[j];
                        dst[j] = cur.wrapping_sub(p);
                        p = cur;
                    }
                }

                #[inline]
                fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm256_loadu_si256(cur.as_ptr().add(i).cast::<__m256i>());
                            let u = _mm256_loadu_si256(prev.as_ptr().add(i).cast::<__m256i>());
                            _mm256_storeu_si256(
                                dst.as_mut_ptr().add(i).cast::<__m256i>(),
                                _mm256_sub_epi8(c, u),
                            );
                            i += 32;
                        }
                    }
                    for j in lim..len {
                        dst[j] = cur[j].wrapping_sub(prev[j]);
                    }
                }

                #[inline]
                fn row_sub1_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 32) * 32;
                    // SAFETY: AVX2 is statically enabled and `i + 32 <= len`.
                    unsafe {
                        let bcast_last = _mm_set1_epi8(15);
                        let mut prev = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>());
                            v = _mm256_add_epi8(v, _mm256_slli_si256::<1>(v));
                            v = _mm256_add_epi8(v, _mm256_slli_si256::<2>(v));
                            v = _mm256_add_epi8(v, _mm256_slli_si256::<4>(v));
                            v = _mm256_add_epi8(v, _mm256_slli_si256::<8>(v));
                            let lo = _mm_add_epi8(_mm256_castsi256_si128(v), prev);
                            prev = _mm_shuffle_epi8(lo, bcast_last);
                            let hi = _mm_add_epi8(_mm256_extracti128_si256::<1>(v), prev);
                            prev = _mm_shuffle_epi8(hi, bcast_last);
                            _mm256_storeu_si256(
                                dst.as_mut_ptr().add(i).cast::<__m256i>(),
                                _mm256_set_m128i(hi, lo),
                            );
                            i += 32;
                        }
                    }
                    let mut acc = if lim > 0 { dst[lim - 1] } else { 0 };
                    for j in lim..len {
                        acc = acc.wrapping_add(src[j]);
                        dst[j] = acc;
                    }
                }

                #[inline]
                fn diff_and_update_row<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    shadow: &mut [Vec4u8],
                ) -> bool {
                    let n = src.len();
                    debug_assert_eq!(shadow.len(), n);
                    let src_p = src.ptr();
                    let dst_p = shadow.as_mut_ptr();
                    let lim = (n / 8) * 8;
                    // SAFETY: AVX2 is statically enabled, `src_p` and `dst_p` are
                    // valid for `n` `Vec4u8`s and non-overlapping (`src` is shm/caller,
                    // `shadow` is exclusively borrowed), and `i + 8 <= n`.
                    unsafe {
                        let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
                        let mut i = 0usize;
                        while i < lim {
                            let s = _mm256_loadu_si256(src_p.add(i).cast::<__m256i>());
                            let d = _mm256_loadu_si256(dst_p.add(i).cast::<__m256i>());
                            let diff = _mm256_xor_si256(s, d);
                            let masked = if HAS_ALPHA {
                                diff
                            } else {
                                _mm256_and_si256(diff, rgb_mask)
                            };
                            if _mm256_testz_si256(masked, masked) == 0 {
                                _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), s);
                                if i + 8 < n {
                                    src.split_at(i + 8)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[i + 8..]);
                                }
                                return true;
                            }
                            i += 8;
                        }
                        for j in lim..n {
                            let s = src_p.add(j).read();
                            let d = dst_p.add(j).read();
                            let differs = if HAS_ALPHA {
                                s != d
                            } else {
                                s.0 != d.0 || s.1 != d.1 || s.2 != d.2
                            };
                            if differs {
                                dst_p.add(j).write(s);
                                if j + 1 < n {
                                    src.split_at(j + 1)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[j + 1..]);
                                }
                                return true;
                            }
                        }
                    }
                    false
                }

                #[inline]
                fn fence_stream() {
                    // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
                    unsafe { _mm_sfence() };
                }
            }
        }
        pub(super) use imp::Kernel;
    },
    all(target_feature = "ssse3", not(miri)) => {
        mod imp {
            use std::arch::x86_64::__m128i;
            use std::arch::x86_64::_mm_abs_epi8;
            use std::arch::x86_64::_mm_add_epi8;
            use std::arch::x86_64::_mm_add_epi64;
            use std::arch::x86_64::_mm_alignr_epi8;
            use std::arch::x86_64::_mm_and_si128;
            use std::arch::x86_64::_mm_cmpeq_epi8;
            use std::arch::x86_64::_mm_cvtsi32_si128;
            use std::arch::x86_64::_mm_cvtsi128_si32;
            use std::arch::x86_64::_mm_loadu_si128;
            use std::arch::x86_64::_mm_movemask_epi8;
            use std::arch::x86_64::_mm_or_si128;
            use std::arch::x86_64::_mm_sad_epu8;
            use std::arch::x86_64::_mm_set_epi8;
            use std::arch::x86_64::_mm_set1_epi8;
            use std::arch::x86_64::_mm_set1_epi32;
            use std::arch::x86_64::_mm_setzero_si128;
            use std::arch::x86_64::_mm_sfence;
            use std::arch::x86_64::_mm_shuffle_epi8;
            use std::arch::x86_64::_mm_slli_si128;
            use std::arch::x86_64::_mm_srli_si128;
            use std::arch::x86_64::_mm_storeu_si128;
            use std::arch::x86_64::_mm_stream_si128;
            use std::arch::x86_64::_mm_sub_epi8;
            use std::arch::x86_64::_mm_xor_si128;

            use crate::buffer_pointer::BufferPointer;
            use crate::filtering::RowKernel;
            use crate::vec4u8::Vec4u8;

            #[inline]
            fn hsum_epi64_128(acc: __m128i) -> u64 {
                let mut tmp = [0u64; 2];
                // SAFETY: `tmp` is 16 bytes and SSE2 is statically enabled.
                unsafe {
                    _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), acc);
                }
                tmp[0] + tmp[1]
            }

            pub(crate) struct Kernel;

            impl RowKernel for Kernel {
                const STREAM_ALIGN: usize = 16;

                #[inline]
                fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    dst_bgr: &mut [u8],
                    dst_alpha: &mut [u8],
                ) {
                    let tw = src.len();
                    debug_assert_eq!(dst_bgr.len(), tw * 3);
                    debug_assert!(!HAS_ALPHA || dst_alpha.len() == tw);

                    // Each 4-pixel iteration stores 16 bytes at `di = x * 3`, reaching `di + 16`.
                    // Stopping at `lim <= tw - 2` ensures `di + 16 <= tw * 3`.
                    let lim = if tw >= 6 { ((tw - 2) / 4) * 4 } else { 0 };
                    let mut x = 0usize;
                    let mut di = 0usize;

                    // SAFETY: SSSE3 is statically enabled; `x + 4 <= tw` and `di + 16 <= dst_bgr.len()`.
                    unsafe {
                        let shuf_g =
                            _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
                        let pack12 =
                            _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);
                        let shuf_a = _mm_set_epi8(
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3,
                        );

                        while x < lim {
                            let v = _mm_loadu_si128(src.ptr().add(x).cast::<__m128i>());
                            let g = _mm_shuffle_epi8(v, shuf_g);
                            let sg = _mm_sub_epi8(v, g);
                            let pk = _mm_shuffle_epi8(sg, pack12);
                            _mm_storeu_si128(dst_bgr.as_mut_ptr().add(di).cast::<__m128i>(), pk);
                            if HAS_ALPHA {
                                let a = _mm_cvtsi128_si32(_mm_shuffle_epi8(v, shuf_a)) as u32;
                                dst_alpha[x..x + 4].copy_from_slice(&a.to_le_bytes());
                            }
                            x += 4;
                            di += 12;
                        }
                    }

                    for (off, bgr) in dst_bgr[lim * 3..]
                        .as_chunks_mut::<3>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let i = lim + off;
                        // SAFETY: `i < tw == src.len()`.
                        let px = unsafe { src.ptr().add(i).read() };
                        bgr[0] = px.0.wrapping_sub(px.1);
                        bgr[1] = px.1;
                        bgr[2] = px.2.wrapping_sub(px.1);
                        if HAS_ALPHA {
                            dst_alpha[i] = px.3;
                        }
                    }
                }

                #[inline]
                fn sad_signed_bytes(data: &[u8]) -> u64 {
                    let lim = (data.len() / 16) * 16;
                    let mut sum = 0u64;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= data.len()`.
                    unsafe {
                        let zero = _mm_setzero_si128();
                        let mut acc = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm_loadu_si128(data.as_ptr().add(i).cast::<__m128i>());
                            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_abs_epi8(v), zero));
                            i += 16;
                        }
                        sum += hsum_epi64_128(acc);
                    }
                    for &b in &data[lim..] {
                        sum += u64::from((b as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64 {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    let mut sum = 0u64;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let zero = _mm_setzero_si128();
                        let mut acc = _mm_setzero_si128();
                        let mut prev = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            let shifted = _mm_alignr_epi8::<13>(v, prev);
                            let diff = _mm_sub_epi8(v, shifted);
                            prev = v;
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), diff);
                            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_abs_epi8(diff), zero));
                            i += 16;
                        }
                        sum += hsum_epi64_128(acc);
                    }
                    for j in lim..len {
                        let p = if j >= 3 { src[j - 3] } else { 0 };
                        let d = src[j].wrapping_sub(p);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_up_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    let mut sum = 0u64;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let zero = _mm_setzero_si128();
                        let mut acc = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm_loadu_si128(cur.as_ptr().add(i).cast::<__m128i>());
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            let diff = _mm_sub_epi8(c, u);
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), diff);
                            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_abs_epi8(diff), zero));
                            i += 16;
                        }
                        sum += hsum_epi64_128(acc);
                    }
                    for j in lim..len {
                        let d = cur[j].wrapping_sub(prev[j]);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_subup3_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    let mut sum = 0u64;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let zero = _mm_setzero_si128();
                        let mut acc = _mm_setzero_si128();
                        let mut prev_up = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm_loadu_si128(cur.as_ptr().add(i).cast::<__m128i>());
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            let up = _mm_sub_epi8(c, u);
                            let shifted = _mm_alignr_epi8::<13>(up, prev_up);
                            let diff = _mm_sub_epi8(up, shifted);
                            prev_up = up;
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), diff);
                            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_abs_epi8(diff), zero));
                            i += 16;
                        }
                        sum += hsum_epi64_128(acc);
                    }
                    for j in lim..len {
                        let up = cur[j].wrapping_sub(prev[j]);
                        let prev_u = if j >= 3 {
                            cur[j - 3].wrapping_sub(prev[j - 3])
                        } else {
                            0
                        };
                        let d = up.wrapping_sub(prev_u);
                        dst[j] = d;
                        sum += u64::from((d as i8).unsigned_abs());
                    }
                    sum
                }

                #[inline]
                fn row_sub3_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
                    // SAFETY: SSSE3 is statically enabled; for `i < lim`, `i + 16 <= lim + 4 <= len`.
                    unsafe {
                        let shuf_last3 =
                            _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
                        let mut prev_px = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                            v = _mm_add_epi8(v, prev_px);
                            prev_px = _mm_shuffle_epi8(v, shuf_last3);
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), v);
                            i += 12;
                        }
                    }
                    for j in lim..len {
                        let p = if j >= 3 { dst[j - 3] } else { 0 };
                        dst[j] = src[j].wrapping_add(p);
                    }
                }

                #[inline]
                fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    // SAFETY: SSE2 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let d = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            _mm_storeu_si128(
                                dst.as_mut_ptr().add(i).cast::<__m128i>(),
                                _mm_add_epi8(d, u),
                            );
                            i += 16;
                        }
                    }
                    for j in lim..len {
                        dst[j] = src[j].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    debug_assert!(len.is_multiple_of(3));
                    let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
                    // SAFETY: SSSE3 is statically enabled; for `i < lim`, `i + 16 <= lim + 4 <= len`.
                    unsafe {
                        let shuf_last3 =
                            _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
                        let mut prev_px = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                            v = _mm_add_epi8(v, prev_px);
                            prev_px = _mm_shuffle_epi8(v, shuf_last3);
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            _mm_storeu_si128(
                                dst.as_mut_ptr().add(i).cast::<__m128i>(),
                                _mm_add_epi8(v, u),
                            );
                            i += 12;
                        }
                    }
                    let mut acc = [0u8; 3];
                    if lim >= 3 {
                        for ch in 0..3 {
                            acc[ch] = dst[lim - 3 + ch].wrapping_sub(prev[lim - 3 + ch]);
                        }
                    }
                    for j in lim..len {
                        let ch = j % 3;
                        acc[ch] = acc[ch].wrapping_add(src[j]);
                        dst[j] = acc[ch].wrapping_add(prev[j]);
                    }
                }

                #[inline]
                unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
                    src_bgr: &[u8],
                    alpha_row: &[u8],
                    const_alpha: u8,
                    dst: &mut [Vec4u8],
                ) {
                    let tw = dst.len();
                    debug_assert_eq!(src_bgr.len(), tw * 3);
                    debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(16));

                    let full4 = if tw >= 8 { (tw / 4) * 4 } else { 0 };
                    let lim = if tw >= 6 { ((tw - 2) / 4) * 4 } else { 0 };
                    let end = full4.max(lim);
                    let mut x = 0usize;
                    let mut si = 0usize;

                    // SAFETY: SSSE3 is statically enabled; `si + 16 <= src_bgr.len()` inside `x < lim`,
                    // and for the final 4-pixel block (`x >= 4`, `si >= 12`), `si - 4..si + 12` stays
                    // in bounds and shifts right 4 bytes to align lanes `0..12`.
                    unsafe {
                        let shuf_g =
                            _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
                        let unpack12 =
                            _mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);
                        let const_alpha_vec = _mm_set1_epi32((u32::from(const_alpha) << 24) as i32);
                        let shuf_a = _mm_set_epi8(
                            3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1,
                        );

                        while x < lim {
                            let v = _mm_loadu_si128(src_bgr.as_ptr().add(si).cast::<__m128i>());
                            let sg = _mm_shuffle_epi8(v, unpack12);
                            let g = _mm_shuffle_epi8(sg, shuf_g);
                            let alpha_vec = if CONST_ALPHA {
                                const_alpha_vec
                            } else {
                                let a32 = _mm_cvtsi32_si128(i32::from_le_bytes(
                                    alpha_row[x..x + 4].try_into().unwrap(),
                                ));
                                _mm_shuffle_epi8(a32, shuf_a)
                            };
                            let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                            let p = dst.as_mut_ptr().add(x).cast::<__m128i>();
                            if STREAM {
                                _mm_stream_si128(p, bgra);
                            } else {
                                _mm_storeu_si128(p, bgra);
                            }
                            x += 4;
                            si += 12;
                        }
                        if x < end {
                            let v = _mm_srli_si128::<4>(_mm_loadu_si128(
                                src_bgr.as_ptr().add(si - 4).cast::<__m128i>(),
                            ));
                            let sg = _mm_shuffle_epi8(v, unpack12);
                            let g = _mm_shuffle_epi8(sg, shuf_g);
                            let alpha_vec = if CONST_ALPHA {
                                const_alpha_vec
                            } else {
                                let a32 = _mm_cvtsi32_si128(i32::from_le_bytes(
                                    alpha_row[x..x + 4].try_into().unwrap(),
                                ));
                                _mm_shuffle_epi8(a32, shuf_a)
                            };
                            let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                            let p = dst.as_mut_ptr().add(x).cast::<__m128i>();
                            if STREAM {
                                _mm_stream_si128(p, bgra);
                            } else {
                                _mm_storeu_si128(p, bgra);
                            }
                        }
                    }

                    for i in end..tw {
                        let s = i * 3;
                        let g = src_bgr[s + 1];
                        let a = if CONST_ALPHA {
                            const_alpha
                        } else {
                            alpha_row[i]
                        };
                        dst[i] = Vec4u8(
                            src_bgr[s].wrapping_add(g),
                            g,
                            src_bgr[s + 2].wrapping_add(g),
                            a,
                        );
                    }
                }

                #[inline]
                unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8) {
                    debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(16));
                    let lim = (dst.len() / 4) * 4;
                    // SAFETY: SSE2 is statically enabled; each iteration writes 4 `Vec4u8` (16 bytes),
                    // and caller upholds 16-byte alignment when `STREAM` is `true`.
                    unsafe {
                        let word = u32::from_le_bytes([pixel.0, pixel.1, pixel.2, pixel.3]) as i32;
                        let vec = _mm_set1_epi32(word);
                        let mut x = 0usize;
                        while x < lim {
                            let p = dst.as_mut_ptr().add(x).cast::<__m128i>();
                            if STREAM {
                                _mm_stream_si128(p, vec);
                            } else {
                                _mm_storeu_si128(p, vec);
                            }
                            x += 4;
                        }
                    }
                    dst[lim..].fill(pixel);
                }

                #[inline]
                fn is_constant_plane(data: &[u8]) -> Option<u8> {
                    let first = *data.first()?;
                    let lim = (data.len() / 16) * 16;
                    // SAFETY: SSE2 is statically enabled and `i + 16 <= data.len()`.
                    unsafe {
                        let target = _mm_set1_epi8(first as i8);
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm_loadu_si128(data.as_ptr().add(i).cast::<__m128i>());
                            if _mm_movemask_epi8(_mm_cmpeq_epi8(v, target)) != 0xFFFF {
                                return None;
                            }
                            i += 16;
                        }
                    }
                    data[lim..].iter().all(|&b| b == first).then_some(first)
                }

                #[inline]
                fn row_sub1_forward(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let mut prev = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            let shifted = _mm_alignr_epi8::<15>(v, prev);
                            _mm_storeu_si128(
                                dst.as_mut_ptr().add(i).cast::<__m128i>(),
                                _mm_sub_epi8(v, shifted),
                            );
                            prev = v;
                            i += 16;
                        }
                    }
                    let mut p = if lim > 0 { src[lim - 1] } else { 0 };
                    for j in lim..len {
                        let cur = src[j];
                        dst[j] = cur.wrapping_sub(p);
                        p = cur;
                    }
                }

                #[inline]
                fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
                    let len = cur.len();
                    debug_assert_eq!(prev.len(), len);
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    // SAFETY: SSE2 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let mut i = 0usize;
                        while i < lim {
                            let c = _mm_loadu_si128(cur.as_ptr().add(i).cast::<__m128i>());
                            let u = _mm_loadu_si128(prev.as_ptr().add(i).cast::<__m128i>());
                            _mm_storeu_si128(
                                dst.as_mut_ptr().add(i).cast::<__m128i>(),
                                _mm_sub_epi8(c, u),
                            );
                            i += 16;
                        }
                    }
                    for j in lim..len {
                        dst[j] = cur[j].wrapping_sub(prev[j]);
                    }
                }

                #[inline]
                fn row_sub1_inverse(src: &[u8], dst: &mut [u8]) {
                    let len = src.len();
                    debug_assert_eq!(dst.len(), len);
                    let lim = (len / 16) * 16;
                    // SAFETY: SSSE3 is statically enabled and `i + 16 <= len`.
                    unsafe {
                        let bcast_last = _mm_set1_epi8(15);
                        let mut prev = _mm_setzero_si128();
                        let mut i = 0usize;
                        while i < lim {
                            let mut v = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
                            v = _mm_add_epi8(v, _mm_slli_si128::<1>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<2>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<4>(v));
                            v = _mm_add_epi8(v, _mm_slli_si128::<8>(v));
                            v = _mm_add_epi8(v, prev);
                            prev = _mm_shuffle_epi8(v, bcast_last);
                            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), v);
                            i += 16;
                        }
                    }
                    let mut acc = if lim > 0 { dst[lim - 1] } else { 0 };
                    for j in lim..len {
                        acc = acc.wrapping_add(src[j]);
                        dst[j] = acc;
                    }
                }

                #[inline]
                fn diff_and_update_row<const HAS_ALPHA: bool>(
                    src: BufferPointer<Vec4u8>,
                    shadow: &mut [Vec4u8],
                ) -> bool {
                    let n = src.len();
                    debug_assert_eq!(shadow.len(), n);
                    let src_p = src.ptr();
                    let dst_p = shadow.as_mut_ptr();
                    let lim = (n / 4) * 4;
                    // SAFETY: SSE2/SSSE3 are statically enabled, `src_p` and `dst_p` are
                    // valid for `n` `Vec4u8`s and non-overlapping, and `i + 4 <= n`.
                    unsafe {
                        let zero = _mm_setzero_si128();
                        let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
                        let mut i = 0usize;
                        while i < lim {
                            let s = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                            let d = _mm_loadu_si128(dst_p.add(i).cast::<__m128i>());
                            let diff = _mm_xor_si128(s, d);
                            let masked = if HAS_ALPHA {
                                diff
                            } else {
                                _mm_and_si128(diff, rgb_mask)
                            };
                            if _mm_movemask_epi8(_mm_cmpeq_epi8(masked, zero)) != 0xFFFF {
                                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), s);
                                if i + 4 < n {
                                    src.split_at(i + 4)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[i + 4..]);
                                }
                                return true;
                            }
                            i += 4;
                        }
                        for j in lim..n {
                            let s = src_p.add(j).read();
                            let d = dst_p.add(j).read();
                            let differs = if HAS_ALPHA {
                                s != d
                            } else {
                                s.0 != d.0 || s.1 != d.1 || s.2 != d.2
                            };
                            if differs {
                                dst_p.add(j).write(s);
                                if j + 1 < n {
                                    src.split_at(j + 1)
                                        .1
                                        .copy_to_nonoverlapping(&mut shadow[j + 1..]);
                                }
                                return true;
                            }
                        }
                    }
                    false
                }

                #[inline]
                fn fence_stream() {
                    // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
                    unsafe { _mm_sfence() };
                }
            }
        }
        pub(super) use imp::Kernel;
    },
    _ => {
        pub(super) use crate::filtering::ScalarKernel as Kernel;
    },
}
