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

//! `cfg`-pruned runtime-dispatched x86_64 SIMD backends for direct 3-byte
//! `SubGreen(BGR)` AoS filtering, tile unfiltering, and shadow diffing.
//!
//! Only tiers at or above the compile-time target-feature baseline are compiled:
//! - `avx512` (`avx512bw + avx512vl + avx512vbmi`): always compiled on non-Miri
//!   `x86_64`; 512-bit `vpermb`/`vpermt2b`, 48-byte masked load/store 3-byte
//!   prefix sum, and 64-byte-aligned `_mm512_stream_si512` non-temporal stores.
//! - `avx2`: compiled unless static AVX-512 (`avx512bw + avx512vl + avx512vbmi`)
//!   is enabled; 256-bit AVX2 + 128-bit SSSE3 3-byte prefix sum + 32-byte
//!   `_mm256_stream_si256` stores.
//! - `ssse3`: compiled unless `avx2` (or static AVX-512) is enabled; 128-bit
//!   SSSE3 + 16-byte `_mm_stream_si128` stores.
//! - `ScalarKernel`: compiled into runtime dispatch only on baseline SSE2
//!   (`not(target_feature = "ssse3")`).
//!
//! Runtime feature detection caches CPUID in a relaxed [`AtomicU8`] after the
//! first call and folds away completely to a zero-sized static `X86Tier::Avx512`
//! when compiled with static AVX-512. Each SIMD kernel is a private zero-sized
//! capability token constructed exclusively inside runtime-verified
//! `#[target_feature]` outer tile/strip/worker wrappers so every [`RowKernel`]
//! method inlines directly into the loop with zero per-row calls and zero ISA
//! leakage into baseline code.

use std::borrow::Cow;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::ClientTileCache;
use crate::filtering::DecodeCacheCtx;
use crate::filtering::FrameRows;
#[cfg(all(
    not(target_feature = "ssse3"),
    not(target_feature = "avx2"),
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
    )),
))]
use crate::filtering::ScalarKernel;
use crate::filtering::ScanSubBand;
use crate::filtering::ShadowRows;
use crate::filtering::tile_cache::SlotScans;
use crate::filtering::tile_cache::ValidatedCacheHit;
use crate::filtering::tile_hash::RectScanMeta;
use crate::prelude::*;
use crate::serialization::framing::AlphaChannel;
use crate::serialization::framing::CompressedTile;
use crate::serialization::framing::LeafPayload;
use crate::serialization::framing::ScrollRect;
use crate::serialization::framing::TileRect;
use crate::sharding_compression::CompressorWorkerState;
use crate::sharding_compression::DecompressorWorkerState;
use crate::vec4u8::Vec4u8;

/// Compiled x86_64 SIMD execution tier, pruned to tiers at or above the
/// compile-time baseline target features.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) enum X86Tier {
    #[cfg(all(
        not(target_feature = "ssse3"),
        not(target_feature = "avx2"),
        not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )),
    ))]
    Scalar,
    #[cfg(all(
        not(target_feature = "avx2"),
        not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )),
    ))]
    Ssse3,
    #[cfg(not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
    )))]
    Avx2,
    Avx512,
}

pub(super) use X86Tier as Tier;

#[cfg(not(all(
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi",
)))]
mod detect {
    use std::arch::is_x86_feature_detected;
    use std::sync::atomic::AtomicU8;
    use std::sync::atomic::Ordering;

    use super::X86Tier;

    #[cfg(all(not(target_feature = "ssse3"), not(target_feature = "avx2")))]
    const TIER_SCALAR: u8 = 1;
    #[cfg(not(target_feature = "avx2"))]
    const TIER_SSSE3: u8 = 2;
    const TIER_AVX2: u8 = 3;
    const TIER_AVX512: u8 = 4;

    static CACHED_TIER: AtomicU8 = AtomicU8::new(0);

    #[cold]
    #[inline(never)]
    fn detect_uncached() -> X86Tier {
        let (tier, code) = if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vbmi")
        {
            (X86Tier::Avx512, TIER_AVX512)
        } else {
            cfg_select! {
                target_feature = "avx2" => (X86Tier::Avx2, TIER_AVX2),
                target_feature = "ssse3" => {
                    if is_x86_feature_detected!("avx2") {
                        (X86Tier::Avx2, TIER_AVX2)
                    } else {
                        (X86Tier::Ssse3, TIER_SSSE3)
                    }
                },
                _ => {
                    if is_x86_feature_detected!("avx2") {
                        (X86Tier::Avx2, TIER_AVX2)
                    } else if is_x86_feature_detected!("ssse3") {
                        (X86Tier::Ssse3, TIER_SSSE3)
                    } else {
                        (X86Tier::Scalar, TIER_SCALAR)
                    }
                },
            }
        };
        CACHED_TIER.store(code, Ordering::Relaxed);
        tier
    }

    #[inline(always)]
    pub(super) fn detect_tier() -> X86Tier {
        match CACHED_TIER.load(Ordering::Relaxed) {
            #[cfg(all(not(target_feature = "ssse3"), not(target_feature = "avx2")))]
            TIER_SCALAR => X86Tier::Scalar,
            #[cfg(not(target_feature = "avx2"))]
            TIER_SSSE3 => X86Tier::Ssse3,
            TIER_AVX2 => X86Tier::Avx2,
            TIER_AVX512 => X86Tier::Avx512,
            _ => detect_uncached(),
        }
    }
}

mod avx512 {
    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_sfence;
    use std::arch::x86_64::_mm_storeu_si128;
    use std::arch::x86_64::_mm512_abs_epi8;
    use std::arch::x86_64::_mm512_add_epi8;
    use std::arch::x86_64::_mm512_add_epi64;
    use std::arch::x86_64::_mm512_and_si512;
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
    use std::arch::x86_64::_mm512_xor_si512;
    use std::arch::x86_64::_mm512_zextsi128_si512;
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::RowKernel;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    fn load_table(table: &[u8; 64]) -> __m512i {
        // SAFETY: `table` is 64 bytes and caller guarantees AVX-512BW/VL/VBMI.
        unsafe { _mm512_loadu_si512(table.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    fn prefix_sum_sub3_48(
        mut v: __m512i,
        sh3: __m512i,
        sh6: __m512i,
        sh12: __m512i,
        sh24: __m512i,
    ) -> __m512i {
        v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 3, sh3, v));
        v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 6, sh6, v));
        v = _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 12, sh12, v));
        _mm512_add_epi8(v, _mm512_maskz_permutexvar_epi8(u64::MAX << 24, sh24, v))
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn bgra_to_subgreen_bgr3_impl<const HAS_ALPHA: bool>(
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

        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; `x + 16 <= tw`,
        // `di + 48 <= dst_bgr.len()`, and `x + 16 <= dst_alpha.len()` when `HAS_ALPHA`.
        unsafe {
            let shuf_g = load_table(&SHUF_G);
            let pack_idx = load_table(&PACK_BGR_ALPHA_IDX);

            while x < lim {
                let v = _mm512_loadu_si512(src.ptr().add(x).cast());
                let g = _mm512_shuffle_epi8(v, shuf_g);
                let sg = _mm512_sub_epi8(v, g);
                let packed = _mm512_permutexvar_epi8(pack_idx, sg);
                _mm512_mask_storeu_epi8(dst_bgr.as_mut_ptr().add(di).cast(), MASK_48, packed);
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn sad_signed_bytes_impl(data: &[u8]) -> u64 {
        let lim = (data.len() / 64) * 64;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX-512BW and `i + 64 <= data.len()`.
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_forward_sad_impl(src: &[u8], dst: &mut [u8]) -> u64 {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX-512BW/VBMI and `i + 64 <= len`.
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
                acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_up_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX-512BW and `i + 64 <= len`.
        unsafe {
            let zero = _mm512_setzero_si512();
            let mut acc = _mm512_setzero_si512();
            let mut i = 0usize;
            while i < lim {
                let c = _mm512_loadu_si512(cur.as_ptr().add(i).cast());
                let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                let diff = _mm512_sub_epi8(c, u);
                _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), diff);
                acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_subup3_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX-512BW/VBMI and `i + 64 <= len`.
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
                acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 48) * 48;
        // SAFETY: Caller guarantees AVX-512BW/VBMI; masked 48-byte loads/stores
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_up_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW and `i + 64 <= len`.
        unsafe {
            let mut i = 0usize;
            while i < lim {
                let d = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_add_epi8(d, u));
                i += 64;
            }
        }
        for j in lim..len {
            dst[j] = src[j].wrapping_add(prev[j]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_subup3_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len().min(prev.len()).min(dst.len());
        debug_assert_eq!(src.len(), len);
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        debug_assert!(len.is_multiple_of(3));
        let lim = (len / 48) * 48;
        // SAFETY: Caller guarantees AVX-512BW/VBMI; masked 48-byte loads/stores
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
        let src = &src[..len];
        let prev = &prev[..len];
        let dst = &mut dst[..len];
        for j in lim..len {
            let prev_up = if j >= 3 {
                dst[j - 3].wrapping_sub(prev[j - 3])
            } else {
                0
            };
            dst[j] = prev_up.wrapping_add(src[j]).wrapping_add(prev[j]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn subgreen_bgr3_to_bgra_impl<const STREAM: bool, const CONST_ALPHA: bool>(
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
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

        // SAFETY: Caller guarantees AVX-512BW/VBMI; `si + 48 <= src_bgr.len()`,
        // `x + 16 <= dst.len()`, `x + 16 <= alpha_row.len()` when `!CONST_ALPHA`,
        // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let shuf_g = load_table(&SHUF_G);
            let unpack_bgr = load_table(&UNPACK_BGR_IDX);
            let unpack_bgra = load_table(&UNPACK_BGRA_IDX);
            let const_alpha_vec = _mm512_set1_epi32((u32::from(const_alpha) << 24) as i32);

            while x < lim {
                let bgr = _mm512_maskz_loadu_epi8(MASK_48, src_bgr.as_ptr().add(si).cast());
                let bgra = if CONST_ALPHA {
                    let sg = _mm512_permutexvar_epi8(unpack_bgr, bgr);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                } else {
                    let a128 = _mm_loadu_si128(alpha_row.as_ptr().add(x).cast::<__m128i>());
                    let sg =
                        _mm512_permutex2var_epi8(bgr, unpack_bgra, _mm512_zextsi128_si512(a128));
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
            dst[i].write(Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            ));
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn splat_row_impl<const STREAM: bool>(dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8) {
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(64));
        let lim = (dst.len() / 16) * 16;
        // SAFETY: Caller guarantees AVX-512F; each iteration writes 16 `Vec4u8`
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
        dst[lim..].fill(MaybeUninit::new(pixel));
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn copy_row_stream_impl(src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
        let n = dst.len();
        debug_assert_eq!(src.len(), n);
        debug_assert!(n.is_multiple_of(16));
        debug_assert!((dst.as_mut_ptr() as usize).is_multiple_of(64));
        // SAFETY: Caller guarantees AVX-512F, 64-byte alignment on `dst`, and `n % 16 == 0`.
        unsafe {
            let mut x = 0usize;
            while x < n {
                let v = _mm512_loadu_si512(src.as_ptr().add(x).cast());
                _mm512_stream_si512(dst.as_mut_ptr().add(x).cast::<__m512i>(), v);
                x += 16;
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn is_constant_plane_impl(data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        let lim = (data.len() / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW and `i + 64 <= data.len()`.
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub1_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VBMI and `i + 64 <= len`.
        unsafe {
            let shift_idx = load_table(&SHIFT_LEFT1_IDX);
            let mut prev = _mm512_setzero_si512();
            let mut i = 0usize;
            while i < lim {
                let v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_sub_epi8(v, shifted));
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_up_forward_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW and `i + 64 <= len`.
        unsafe {
            let mut i = 0usize;
            while i < lim {
                let c = _mm512_loadu_si512(cur.as_ptr().add(i).cast());
                let u = _mm512_loadu_si512(prev.as_ptr().add(i).cast());
                _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_sub_epi8(c, u));
                i += 64;
            }
        }
        for j in lim..len {
            dst[j] = cur[j].wrapping_sub(prev[j]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub1_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VBMI and `i + 64 <= len`.
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn diff_and_update_row_impl<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> Option<(usize, usize)> {
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        let dst_p = shadow.as_mut_ptr();
        let lim = (n / 16) * 16;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees AVX-512BW, `src_p` and `dst_p` are valid
        // for `n` `Vec4u8`s and non-overlapping (`src` is shm/caller, `shadow`
        // is exclusively borrowed), and `i + 16 <= n`.
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
                    if first == usize::MAX {
                        first = i + (neq.trailing_zeros() as usize >> 2);
                    }
                    last_end = i + ((63 - neq.leading_zeros() as usize) >> 2) + 1;
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
                    if first == usize::MAX {
                        first = j;
                    }
                    last_end = j + 1;
                }
            }
        }
        (first != usize::MAX).then_some((first, last_end))
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn copy_and_scan_row_impl<const HAS_ALPHA: bool>(
        src: *const u8,
        dst: *mut u8,
        n_px: usize,
        first_word: u32,
    ) -> (bool, bool) {
        let lim = (n_px / 16) * 16;
        // SAFETY: Caller guarantees AVX-512BW, `src` readable and `dst` writable
        // for `n_px * 4` bytes and non-overlapping.
        unsafe {
            let alpha_mask = _mm512_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
            let first_vec = _mm512_set1_epi32(first_word as i32);
            let mut dacc = _mm512_setzero_si512();
            let mut aand = _mm512_set1_epi8(-1);
            let mut i = 0usize;
            while i < lim {
                let off = i * 4;
                let r = _mm512_loadu_si512(src.add(off).cast());
                _mm512_storeu_si512(dst.add(off).cast(), r);
                let m = if HAS_ALPHA {
                    r
                } else {
                    _mm512_or_si512(r, alpha_mask)
                };
                dacc = _mm512_or_si512(dacc, _mm512_xor_si512(m, first_vec));
                if HAS_ALPHA {
                    aand = _mm512_and_si512(aand, m);
                }
                i += 16;
            }
            let mut tail_diff = 0u32;
            let mut tail_alpha = u32::MAX;
            for j in lim..n_px {
                let off = j * 4;
                let raw = src.add(off).cast::<u32>().read_unaligned();
                dst.add(off).cast::<u32>().write_unaligned(raw);
                let canon = if HAS_ALPHA {
                    raw
                } else {
                    raw | XRGB_ALPHA_MASK_U32
                };
                tail_diff |= canon ^ first_word;
                if HAS_ALPHA {
                    tail_alpha &= canon;
                }
            }
            let solid =
                tail_diff == 0 && _mm512_cmpneq_epi8_mask(dacc, _mm512_setzero_si512()) == 0;
            let opaque = !HAS_ALPHA
                || ((tail_alpha & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32
                    && (_mm512_cmpeq_epi8_mask(aand, _mm512_set1_epi8(-1))
                        & 0x8888_8888_8888_8888)
                        == 0x8888_8888_8888_8888);
            (solid, opaque)
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct Kernel(());

    impl Kernel {
        /// # Safety
        /// Caller must have verified CPU support for `avx512bw + avx512vl + avx512vbmi`.
        #[inline(always)]
        pub(super) unsafe fn new_unchecked() -> Self {
            Self(())
        }
    }

    impl RowKernel for Kernel {
        const STREAM_ALIGN: usize = 64;

        #[inline(always)]
        fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            dst_bgr: &mut [u8],
            dst_alpha: &mut [u8],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { bgra_to_subgreen_bgr3_impl::<HAS_ALPHA>(src, dst_bgr, dst_alpha) }
        }

        #[inline(always)]
        fn sad_signed_bytes(self, data: &[u8]) -> u64 {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { sad_signed_bytes_impl(data) }
        }

        #[inline(always)]
        fn row_sub3_forward_sad(self, src: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub3_forward_sad_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_up_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_subup3_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub3_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_up_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_subup3_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src_bgr: &[u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA>(
                    src_bgr,
                    alpha_row,
                    const_alpha,
                    dst,
                )
            }
        }

        #[inline(always)]
        unsafe fn splat_row<const STREAM: bool>(
            self,
            dst: &mut [MaybeUninit<Vec4u8>],
            pixel: Vec4u8,
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe { splat_row_impl::<STREAM>(dst, pixel) }
        }

        #[inline(always)]
        unsafe fn copy_row_stream(self, src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds 64-byte alignment and 16-pixel multiple length.
            unsafe { copy_row_stream_impl(src, dst) }
        }

        #[inline(always)]
        fn is_constant_plane(self, data: &[u8]) -> Option<u8> {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { is_constant_plane_impl(data) }
        }

        #[inline(always)]
        fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub1_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_up_forward_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub1_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn diff_and_update_row<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            shadow: &mut [Vec4u8],
        ) -> Option<(usize, usize)> {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { diff_and_update_row_impl::<HAS_ALPHA>(src, shadow) }
        }

        #[inline(always)]
        unsafe fn copy_and_scan_row<const HAS_ALPHA: bool>(
            self,
            src: *const u8,
            dst: *mut u8,
            n_px: usize,
            first_word: u32,
        ) -> (bool, bool) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds pointer bounds and non-overlapping invariants.
            unsafe { copy_and_scan_row_impl::<HAS_ALPHA>(src, dst, n_px, first_word) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }
    }
}

#[cfg(not(all(
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi",
)))]
mod avx2 {
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
    use std::arch::x86_64::_mm256_cmpeq_epi8;
    use std::arch::x86_64::_mm256_cmpeq_epi32;
    use std::arch::x86_64::_mm256_extracti128_si256;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_movemask_epi8;
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
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::RowKernel;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
    use crate::vec4u8::Vec4u8;

    #[target_feature(enable = "avx2")]
    #[inline]
    fn hsum_epi64(acc: __m256i) -> u64 {
        let mut tmp = [0u64; 4];
        // SAFETY: `tmp` is 32 bytes and caller guarantees AVX2.
        unsafe {
            _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), acc);
        }
        tmp[0] + tmp[1] + tmp[2] + tmp[3]
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bgra_to_subgreen_bgr3_impl<const HAS_ALPHA: bool>(
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

        // SAFETY: Caller guarantees AVX2; `x + 8 <= tw` and `di + 28 <= dst_bgr.len()`
        // for all iterations `x < lim`.
        unsafe {
            let shuf_g = _mm256_set_epi8(
                -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1, 13, -1, 13, -1, 9,
                -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
            );
            let pack12 = _mm256_set_epi8(
                -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, -1, -1, -1, -1, 14, 13, 12,
                10, 9, 8, 6, 5, 4, 2, 1, 0,
            );
            let shuf_a = _mm256_set_epi8(
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3,
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
                    let a_hi = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(a_pk)) as u32;
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sad_signed_bytes_impl(data: &[u8]) -> u64 {
        let lim = (data.len() / 32) * 32;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= data.len()`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_forward_sad_impl(src: &[u8], dst: &mut [u8]) -> u64 {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_up_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut acc = _mm256_setzero_si256();
            let mut i = 0usize;
            while i < lim {
                let c = _mm256_loadu_si256(cur.as_ptr().add(i).cast::<__m256i>());
                let u = _mm256_loadu_si256(prev.as_ptr().add(i).cast::<__m256i>());
                let diff = _mm256_sub_epi8(c, u);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), diff);
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_subup3_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_abs_epi8(diff), zero));
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        // Step by 12 bytes (4 RGB pixels) while guaranteeing 16-byte loads/stores
        // stay strictly inside `0..len` (`i + 16 <= len`).
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: AVX2 implies SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            // Repeats pixel 3 (`bytes 9, 10, 11`) across all four 3-byte pixel slots.
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_up_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_subup3_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len().min(prev.len()).min(dst.len());
        debug_assert_eq!(src.len(), len);
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        debug_assert!(len.is_multiple_of(3));
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: AVX2 implies SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
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
        let src = &src[..len];
        let prev = &prev[..len];
        let dst = &mut dst[..len];
        for j in lim..len {
            let prev_up = if j >= 3 {
                dst[j - 3].wrapping_sub(prev[j - 3])
            } else {
                0
            };
            dst[j] = prev_up.wrapping_add(src[j]).wrapping_add(prev[j]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn subgreen_bgr3_to_bgra_impl<const STREAM: bool, const CONST_ALPHA: bool>(
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(src_bgr.len(), tw * 3);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(32));

        let full8 = (tw / 8) * 8;
        let lim = if tw >= 10 { ((tw - 2) / 8) * 8 } else { 0 };
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees AVX2; `si + 28 <= src_bgr.len()` inside the
        // main loop, the optional final 8-pixel block reads `si..si + 24 <= src_bgr.len()`,
        // and caller upholds 32-byte alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let shuf_g = _mm256_set_epi8(
                -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1, 13, -1, 13, -1, 9,
                -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
            );
            let unpack12 = _mm256_set_epi8(
                -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9, -1, 8, 7, 6,
                -1, 5, 4, 3, -1, 2, 1, 0,
            );
            let const_alpha_vec = _mm256_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a_lo =
                _mm_set_epi8(3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1);
            let shuf_a_hi =
                _mm_set_epi8(7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4, -1, -1, -1);

            while x < lim {
                let lo = _mm_loadu_si128(src_bgr.as_ptr().add(si).cast::<__m128i>());
                let hi = _mm_loadu_si128(src_bgr.as_ptr().add(si + 12).cast::<__m128i>());
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
            dst[i].write(Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            ));
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat_row_impl<const STREAM: bool>(dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8) {
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(32));
        let lim = (dst.len() / 8) * 8;
        // SAFETY: Caller guarantees AVX2; each iteration writes 8 `Vec4u8` (32 bytes),
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
        dst[lim..].fill(MaybeUninit::new(pixel));
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn copy_row_stream_impl(src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
        let n = dst.len();
        debug_assert_eq!(src.len(), n);
        debug_assert!(n.is_multiple_of(16));
        debug_assert!((dst.as_mut_ptr() as usize).is_multiple_of(32));
        // SAFETY: Caller guarantees AVX2, 32-byte alignment on `dst`, and `n % 16 == 0`.
        unsafe {
            let mut x = 0usize;
            while x < n {
                let v0 = _mm256_loadu_si256(src.as_ptr().add(x).cast::<__m256i>());
                let v1 = _mm256_loadu_si256(src.as_ptr().add(x + 8).cast::<__m256i>());
                _mm256_stream_si256(dst.as_mut_ptr().add(x).cast::<__m256i>(), v0);
                _mm256_stream_si256(dst.as_mut_ptr().add(x + 8).cast::<__m256i>(), v1);
                x += 16;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn is_constant_plane_impl(data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        let lim = (data.len() / 32) * 32;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= data.len()`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub1_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_up_forward_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub1_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 32) * 32;
        // SAFETY: Caller guarantees AVX2 and `i + 32 <= len`.
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn diff_and_update_row_impl<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> Option<(usize, usize)> {
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        let dst_p = shadow.as_mut_ptr();
        let lim = (n / 8) * 8;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees AVX2, `src_p` and `dst_p` are valid for `n`
        // `Vec4u8`s and non-overlapping (`src` is shm/caller, `shadow` is
        // exclusively borrowed), and `i + 8 <= n`.
        unsafe {
            let zero = _mm256_setzero_si256();
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
                    let eq = _mm256_movemask_epi8(_mm256_cmpeq_epi32(masked, zero)) as u32;
                    let neq = !eq;
                    debug_assert_ne!(neq, 0);
                    if first == usize::MAX {
                        first = i + (neq.trailing_zeros() as usize >> 2);
                    }
                    last_end = i + ((31 - neq.leading_zeros() as usize) >> 2) + 1;
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
                    if first == usize::MAX {
                        first = j;
                    }
                    last_end = j + 1;
                }
            }
        }
        (first != usize::MAX).then_some((first, last_end))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn copy_and_scan_row_impl<const HAS_ALPHA: bool>(
        src: *const u8,
        dst: *mut u8,
        n_px: usize,
        first_word: u32,
    ) -> (bool, bool) {
        let lim = (n_px / 8) * 8;
        // SAFETY: Caller guarantees AVX2, `src` readable and `dst` writable
        // for `n_px * 4` bytes and non-overlapping.
        unsafe {
            let alpha_mask = _mm256_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
            let first_vec = _mm256_set1_epi32(first_word as i32);
            let mut dacc = _mm256_setzero_si256();
            let mut aand = _mm256_set1_epi8(-1);
            let mut i = 0usize;
            while i < lim {
                let off = i * 4;
                let r = _mm256_loadu_si256(src.add(off).cast());
                _mm256_storeu_si256(dst.add(off).cast(), r);
                let m = if HAS_ALPHA {
                    r
                } else {
                    _mm256_or_si256(r, alpha_mask)
                };
                dacc = _mm256_or_si256(dacc, _mm256_xor_si256(m, first_vec));
                if HAS_ALPHA {
                    aand = _mm256_and_si256(aand, m);
                }
                i += 8;
            }
            let mut tail_diff = 0u32;
            let mut tail_alpha = u32::MAX;
            for j in lim..n_px {
                let off = j * 4;
                let raw = src.add(off).cast::<u32>().read_unaligned();
                dst.add(off).cast::<u32>().write_unaligned(raw);
                let canon = if HAS_ALPHA {
                    raw
                } else {
                    raw | XRGB_ALPHA_MASK_U32
                };
                tail_diff |= canon ^ first_word;
                if HAS_ALPHA {
                    tail_alpha &= canon;
                }
            }
            let solid = tail_diff == 0 && _mm256_testz_si256(dacc, dacc) != 0;
            let opaque = !HAS_ALPHA
                || ((tail_alpha & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32
                    && (_mm256_movemask_epi8(_mm256_cmpeq_epi8(aand, _mm256_set1_epi8(-1)))
                        as u32
                        & 0x8888_8888)
                        == 0x8888_8888);
            (solid, opaque)
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct Kernel(());

    impl Kernel {
        /// # Safety
        /// Caller must have verified CPU support for `avx2`.
        #[inline(always)]
        pub(super) unsafe fn new_unchecked() -> Self {
            Self(())
        }
    }

    impl RowKernel for Kernel {
        const STREAM_ALIGN: usize = 32;

        #[inline(always)]
        fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            dst_bgr: &mut [u8],
            dst_alpha: &mut [u8],
        ) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { bgra_to_subgreen_bgr3_impl::<HAS_ALPHA>(src, dst_bgr, dst_alpha) }
        }

        #[inline(always)]
        fn sad_signed_bytes(self, data: &[u8]) -> u64 {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { sad_signed_bytes_impl(data) }
        }

        #[inline(always)]
        fn row_sub3_forward_sad(self, src: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub3_forward_sad_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_up_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_subup3_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub3_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_up_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_subup3_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src_bgr: &[u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // 32-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA>(
                    src_bgr,
                    alpha_row,
                    const_alpha,
                    dst,
                )
            }
        }

        #[inline(always)]
        unsafe fn splat_row<const STREAM: bool>(
            self,
            dst: &mut [MaybeUninit<Vec4u8>],
            pixel: Vec4u8,
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // 32-byte alignment on `dst` when `STREAM` is `true`.
            unsafe { splat_row_impl::<STREAM>(dst, pixel) }
        }

        #[inline(always)]
        unsafe fn copy_row_stream(self, src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // 32-byte alignment and 16-pixel multiple length.
            unsafe { copy_row_stream_impl(src, dst) }
        }

        #[inline(always)]
        fn is_constant_plane(self, data: &[u8]) -> Option<u8> {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { is_constant_plane_impl(data) }
        }

        #[inline(always)]
        fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub1_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_up_forward_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub1_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn diff_and_update_row<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            shadow: &mut [Vec4u8],
        ) -> Option<(usize, usize)> {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { diff_and_update_row_impl::<HAS_ALPHA>(src, shadow) }
        }

        #[inline(always)]
        unsafe fn copy_and_scan_row<const HAS_ALPHA: bool>(
            self,
            src: *const u8,
            dst: *mut u8,
            n_px: usize,
            first_word: u32,
        ) -> (bool, bool) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // pointer bounds and non-overlapping invariants.
            unsafe { copy_and_scan_row_impl::<HAS_ALPHA>(src, dst, n_px, first_word) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }
    }
}

#[cfg(all(
    not(target_feature = "avx2"),
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
    )),
))]
mod ssse3 {
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
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::RowKernel;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
    use crate::vec4u8::Vec4u8;

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn hsum_epi64_128(acc: __m128i) -> u64 {
        let mut tmp = [0u64; 2];
        // SAFETY: `tmp` is 16 bytes and SSE2 is mandatory on x86_64.
        unsafe {
            _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), acc);
        }
        tmp[0] + tmp[1]
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn bgra_to_subgreen_bgr3_impl<const HAS_ALPHA: bool>(
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

        // SAFETY: Caller guarantees SSSE3; `x + 4 <= tw` and `di + 16 <= dst_bgr.len()`.
        unsafe {
            let shuf_g = _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
            let pack12 = _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);
            let shuf_a = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3);

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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn sad_signed_bytes_impl(data: &[u8]) -> u64 {
        let lim = (data.len() / 16) * 16;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= data.len()`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_forward_sad_impl(src: &[u8], dst: &mut [u8]) -> u64 {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_up_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_subup3_forward_sad_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        let mut sum = 0u64;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: Caller guarantees SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_up_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: SSE2 is mandatory on x86_64 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_subup3_inverse_impl(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len().min(prev.len()).min(dst.len());
        debug_assert_eq!(src.len(), len);
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        debug_assert!(len.is_multiple_of(3));
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: Caller guarantees SSSE3; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
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
        let src = &src[..len];
        let prev = &prev[..len];
        let dst = &mut dst[..len];
        for j in lim..len {
            let prev_up = if j >= 3 {
                dst[j - 3].wrapping_sub(prev[j - 3])
            } else {
                0
            };
            dst[j] = prev_up.wrapping_add(src[j]).wrapping_add(prev[j]);
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn subgreen_bgr3_to_bgra_impl<const STREAM: bool, const CONST_ALPHA: bool>(
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
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

        // SAFETY: Caller guarantees SSSE3; `si + 16 <= src_bgr.len()` inside `x < lim`,
        // and for the final 4-pixel block (`x >= 4`, `si >= 12`), `si - 4..si + 12` stays
        // in bounds and shifts right 4 bytes to align lanes `0..12`.
        unsafe {
            let shuf_g = _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
            let unpack12 = _mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);
            let const_alpha_vec = _mm_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a = _mm_set_epi8(3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1);

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
            dst[i].write(Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            ));
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn splat_row_impl<const STREAM: bool>(dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8) {
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(16));
        let lim = (dst.len() / 4) * 4;
        // SAFETY: SSE2 is mandatory on x86_64; each iteration writes 4 `Vec4u8` (16 bytes),
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
        dst[lim..].fill(MaybeUninit::new(pixel));
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn copy_row_stream_impl(src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
        let n = dst.len();
        debug_assert_eq!(src.len(), n);
        debug_assert!(n.is_multiple_of(16));
        debug_assert!((dst.as_mut_ptr() as usize).is_multiple_of(16));
        // SAFETY: SSE2 is mandatory on x86_64, caller guarantees 16-byte
        // alignment on `dst` and `n % 16 == 0`.
        unsafe {
            let mut x = 0usize;
            while x < n {
                let v0 = _mm_loadu_si128(src.as_ptr().add(x).cast::<__m128i>());
                let v1 = _mm_loadu_si128(src.as_ptr().add(x + 4).cast::<__m128i>());
                let v2 = _mm_loadu_si128(src.as_ptr().add(x + 8).cast::<__m128i>());
                let v3 = _mm_loadu_si128(src.as_ptr().add(x + 12).cast::<__m128i>());
                _mm_stream_si128(dst.as_mut_ptr().add(x).cast::<__m128i>(), v0);
                _mm_stream_si128(dst.as_mut_ptr().add(x + 4).cast::<__m128i>(), v1);
                _mm_stream_si128(dst.as_mut_ptr().add(x + 8).cast::<__m128i>(), v2);
                _mm_stream_si128(dst.as_mut_ptr().add(x + 12).cast::<__m128i>(), v3);
                x += 16;
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn is_constant_plane_impl(data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        let lim = (data.len() / 16) * 16;
        // SAFETY: SSE2 is mandatory on x86_64 and `i + 16 <= data.len()`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub1_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_up_forward_impl(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: SSE2 is mandatory on x86_64 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub1_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn diff_and_update_row_impl<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> Option<(usize, usize)> {
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        let dst_p = shadow.as_mut_ptr();
        let lim = (n / 4) * 4;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees SSSE3, `src_p` and `dst_p` are valid for
        // `n` `Vec4u8`s and non-overlapping, and `i + 4 <= n`.
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
                let eq_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(masked, zero)) as u32;
                if eq_mask != 0xFFFF {
                    _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), s);
                    let neq = (!eq_mask) & 0xFFFF;
                    debug_assert_ne!(neq, 0);
                    if first == usize::MAX {
                        first = i + (neq.trailing_zeros() as usize >> 2);
                    }
                    last_end = i + ((31 - neq.leading_zeros() as usize) >> 2) + 1;
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
                    if first == usize::MAX {
                        first = j;
                    }
                    last_end = j + 1;
                }
            }
        }
        (first != usize::MAX).then_some((first, last_end))
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn copy_and_scan_row_impl<const HAS_ALPHA: bool>(
        src: *const u8,
        dst: *mut u8,
        n_px: usize,
        first_word: u32,
    ) -> (bool, bool) {
        let lim = (n_px / 4) * 4;
        // SAFETY: Caller guarantees SSSE3, `src` readable and `dst` writable
        // for `n_px * 4` bytes and non-overlapping.
        unsafe {
            let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
            let first_vec = _mm_set1_epi32(first_word as i32);
            let mut dacc = _mm_setzero_si128();
            let mut aand = _mm_set1_epi8(-1);
            let mut i = 0usize;
            while i < lim {
                let off = i * 4;
                let r = _mm_loadu_si128(src.add(off).cast());
                _mm_storeu_si128(dst.add(off).cast(), r);
                let m = if HAS_ALPHA {
                    r
                } else {
                    _mm_or_si128(r, alpha_mask)
                };
                dacc = _mm_or_si128(dacc, _mm_xor_si128(m, first_vec));
                if HAS_ALPHA {
                    aand = _mm_and_si128(aand, m);
                }
                i += 4;
            }
            let mut tail_diff = 0u32;
            let mut tail_alpha = u32::MAX;
            for j in lim..n_px {
                let off = j * 4;
                let raw = src.add(off).cast::<u32>().read_unaligned();
                dst.add(off).cast::<u32>().write_unaligned(raw);
                let canon = if HAS_ALPHA {
                    raw
                } else {
                    raw | XRGB_ALPHA_MASK_U32
                };
                tail_diff |= canon ^ first_word;
                if HAS_ALPHA {
                    tail_alpha &= canon;
                }
            }
            let solid = tail_diff == 0
                && _mm_movemask_epi8(_mm_cmpeq_epi8(dacc, _mm_setzero_si128())) == 0xFFFF;
            let opaque = !HAS_ALPHA
                || ((tail_alpha & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32
                    && (_mm_movemask_epi8(_mm_cmpeq_epi8(aand, _mm_set1_epi8(-1))) & 0x8888)
                        == 0x8888);
            (solid, opaque)
        }
    }

    #[derive(Copy, Clone)]
    pub(super) struct Kernel(());

    impl Kernel {
        /// # Safety
        /// Caller must have verified CPU support for `ssse3`.
        #[inline(always)]
        pub(super) unsafe fn new_unchecked() -> Self {
            Self(())
        }
    }

    impl RowKernel for Kernel {
        const STREAM_ALIGN: usize = 16;

        #[inline(always)]
        fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            dst_bgr: &mut [u8],
            dst_alpha: &mut [u8],
        ) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { bgra_to_subgreen_bgr3_impl::<HAS_ALPHA>(src, dst_bgr, dst_alpha) }
        }

        #[inline(always)]
        fn sad_signed_bytes(self, data: &[u8]) -> u64 {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { sad_signed_bytes_impl(data) }
        }

        #[inline(always)]
        fn row_sub3_forward_sad(self, src: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub3_forward_sad_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_up_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_subup3_forward_sad_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub3_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_up_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_subup3_inverse_impl(src, prev, dst) }
        }

        #[inline(always)]
        unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src_bgr: &[u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // 16-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA>(
                    src_bgr,
                    alpha_row,
                    const_alpha,
                    dst,
                )
            }
        }

        #[inline(always)]
        unsafe fn splat_row<const STREAM: bool>(
            self,
            dst: &mut [MaybeUninit<Vec4u8>],
            pixel: Vec4u8,
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // 16-byte alignment on `dst` when `STREAM` is `true`.
            unsafe { splat_row_impl::<STREAM>(dst, pixel) }
        }

        #[inline(always)]
        unsafe fn copy_row_stream(self, src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // 16-byte alignment and 16-pixel multiple length.
            unsafe { copy_row_stream_impl(src, dst) }
        }

        #[inline(always)]
        fn is_constant_plane(self, data: &[u8]) -> Option<u8> {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { is_constant_plane_impl(data) }
        }

        #[inline(always)]
        fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub1_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_up_forward_impl(cur, prev, dst) }
        }

        #[inline(always)]
        fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub1_inverse_impl(src, dst) }
        }

        #[inline(always)]
        fn diff_and_update_row<const HAS_ALPHA: bool>(
            self,
            src: BufferPointer<Vec4u8>,
            shadow: &mut [Vec4u8],
        ) -> Option<(usize, usize)> {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { diff_and_update_row_impl::<HAS_ALPHA>(src, shadow) }
        }

        #[inline(always)]
        unsafe fn copy_and_scan_row<const HAS_ALPHA: bool>(
            self,
            src: *const u8,
            dst: *mut u8,
            n_px: usize,
            first_word: u32,
        ) -> (bool, bool) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // pointer bounds and non-overlapping invariants.
            unsafe { copy_and_scan_row_impl::<HAS_ALPHA>(src, dst, n_px, first_word) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }
    }
}

macro_rules! define_tier_wrappers {
    (
        mod_name: $wrapper_mod:ident,
        tier_mod: $tier_mod:ident,
        feature: $feat:literal,
        tier: $tier_variant:expr,
        label: $label:literal,
    ) => {
        mod $wrapper_mod {
            use std::borrow::Cow;
            use std::sync::Mutex;
            use std::sync::atomic::AtomicBool;
            use std::sync::atomic::AtomicU32;
            use std::sync::atomic::AtomicUsize;

            use super::X86Tier;
            use crate::buffer_pointer::BufferPointer;
            use crate::filtering::ClientTileCache;
            use crate::filtering::DecodeCacheCtx;
            use crate::filtering::FrameRows;
            use crate::filtering::ScanSubBand;
            use crate::filtering::ShadowRows;
            use crate::filtering::tile_cache::SlotScans;
            use crate::filtering::tile_cache::ValidatedCacheHit;
            use crate::filtering::tile_hash::RectScanMeta;
            use crate::prelude::*;
            use crate::serialization::framing::AlphaChannel;
            use crate::serialization::framing::CompressedTile;
            use crate::serialization::framing::LeafPayload;
            use crate::serialization::framing::ScrollRect;
            use crate::serialization::framing::TileRect;
            use crate::sharding_compression::CompressorWorkerState;
            use crate::sharding_compression::DecompressorWorkerState;
            use crate::vec4u8::Vec4u8;

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn filter_aos_zstd_leaf<const HAS_ALPHA: bool>(
                aos: BufferPointer<Vec4u8>,
                stride_px: usize,
                rect: TileRect,
                state: &mut CompressorWorkerState,
            ) -> LeafPayload {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`.
                let kernel = unsafe { super::$tier_mod::Kernel::new_unchecked() };
                crate::filtering::filter_aos_zstd_leaf_with::<super::$tier_mod::Kernel, HAS_ALPHA>(
                    kernel, aos, stride_px, rect, state,
                )
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn encode_alpha_plane(
                state: &mut CompressorWorkerState,
                tw: usize,
                th: usize,
            ) -> AlphaChannel {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`.
                let kernel = unsafe { super::$tier_mod::Kernel::new_unchecked() };
                crate::filtering::encode_alpha_plane_with(kernel, state, tw, th)
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn unfilter_tiles_worker<'a, const HAS_ALPHA: bool, F>(
                tile_rects: &[TileRect],
                frame: &FrameRows,
                claimed: &[AtomicBool],
                completed: &AtomicUsize,
                first_err: &Mutex<Option<anyhow::Error>>,
                cache_ctx: Option<&DecodeCacheCtx>,
                next_tile: &F,
                state: &mut DecompressorWorkerState,
            ) where
                F: Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
            {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld validated-frame/disjoint-tile invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::unfilter_tiles_worker_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                        F,
                    >(
                        kernel,
                        $tier_variant,
                        tile_rects,
                        frame,
                        claimed,
                        completed,
                        first_err,
                        cache_ctx,
                        next_tile,
                        state,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn blit_hits_worker(
                cache: &ClientTileCache,
                hits: &[ValidatedCacheHit],
                tile_rects: &[TileRect],
                frame: &FrameRows,
                next_hit: &AtomicUsize,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld validated-hit and disjoint-tile invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::blit_hits_worker_with(
                        kernel, cache, hits, tile_rects, frame, next_hit,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn decode_jpeg_alpha_rows(
                a_filt: &[u8],
                jpeg_rows: &[Vec4u8],
                frame: &FrameRows,
                x0: usize,
                y0: usize,
                tw: usize,
                th: usize,
                alpha_row_a: &mut Vec<u8>,
                alpha_row_b: &mut Vec<u8>,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld validated-frame/disjoint-rect invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::decode_jpeg_alpha_rows_with(
                        kernel,
                        a_filt,
                        jpeg_rows,
                        frame,
                        x0,
                        y0,
                        tw,
                        th,
                        alpha_row_a,
                        alpha_row_b,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn copy_and_scan_rect_to_shadow<const HAS_ALPHA: bool>(
                src: BufferPointer<u8>,
                shadow: &ShadowRows,
                stride: usize,
                rect: TileRect,
            ) -> RectScanMeta {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld disjoint-rect shadow invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::copy_and_scan_rect_to_shadow_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(kernel, src, shadow, stride, rect)
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn diff_and_update_worker<const HAS_ALPHA: bool>(
                data: BufferPointer<u8>,
                aos: BufferPointer<Vec4u8>,
                shadow: &ShadowRows,
                width: usize,
                stride: usize,
                stride_px: usize,
                n_qx: usize,
                sub_bands: &[ScanSubBand],
                dirty_slots: &[AtomicU32],
                slot_scans: Option<&SlotScans>,
                cache_non_empty: bool,
                next_sub: &AtomicUsize,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld disjoint-band shadow invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::diff_and_update_worker_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(
                        kernel,
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        stride_px,
                        n_qx,
                        sub_bands,
                        dirty_slots,
                        slot_scans,
                        cache_non_empty,
                        next_sub,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            pub(super) unsafe fn diff_and_update_scroll_worker<const HAS_ALPHA: bool>(
                data: BufferPointer<u8>,
                aos: BufferPointer<Vec4u8>,
                shadow: &ShadowRows,
                width: usize,
                stride: usize,
                stride_px: usize,
                n_qx: usize,
                sub_bands: &[ScanSubBand],
                sr: ScrollRect,
                dirty_slots: &[AtomicU32],
                slot_scans: Option<&SlotScans>,
                cache_non_empty: bool,
                next_sub: &AtomicUsize,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld disjoint-band and scroll-band shadow invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::diff_and_update_scroll_worker_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(
                        kernel,
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        stride_px,
                        n_qx,
                        sub_bands,
                        sr,
                        dirty_slots,
                        slot_scans,
                        cache_non_empty,
                        next_sub,
                    );
                }
            }

            #[cfg(test)]
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn diff_and_update_strip<const HAS_ALPHA: bool>(
                aos: BufferPointer<Vec4u8>,
                shadow: &ShadowRows,
                stride_px: usize,
                rect: TileRect,
            ) -> bool {
                // SAFETY: Caller verified CPU feature support and disjoint-rect invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::diff_and_update_strip_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(kernel, aos, shadow, stride_px, rect)
                }
            }

            #[cfg(test)]
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
                data: BufferPointer<u8>,
                aos: BufferPointer<Vec4u8>,
                shadow: &ShadowRows,
                width: usize,
                stride: usize,
                y0: usize,
                y1: usize,
                dirty_slots: &[AtomicU32],
            ) {
                // SAFETY: Caller verified CPU feature support and disjoint-row invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::diff_and_update_row_band_2d_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(
                        kernel,
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        y0,
                        y1,
                        0,
                        dirty_slots,
                        None,
                        false,
                    );
                }
            }

            #[cfg(test)]
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn verify_kernel() {
                // SAFETY: Caller verified CPU feature support before invoking test.
                let kernel = unsafe { super::$tier_mod::Kernel::new_unchecked() };
                crate::filtering::tests::verify_kernel_matches_scalar(kernel, $label);
            }
        }
    };
}

define_tier_wrappers! {
    mod_name: dispatch_avx512,
    tier_mod: avx512,
    feature: "avx512bw,avx512vl,avx512vbmi",
    tier: X86Tier::Avx512,
    label: "avx512",
}

#[cfg(not(all(
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi",
)))]
define_tier_wrappers! {
    mod_name: dispatch_avx2,
    tier_mod: avx2,
    feature: "avx2",
    tier: X86Tier::Avx2,
    label: "avx2",
}

#[cfg(all(
    not(target_feature = "avx2"),
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
    )),
))]
define_tier_wrappers! {
    mod_name: dispatch_ssse3,
    tier_mod: ssse3,
    feature: "ssse3",
    tier: X86Tier::Ssse3,
    label: "ssse3",
}

impl X86Tier {
    /// Returns the highest compiled x86_64 SIMD tier supported by the current
    /// host CPU, caching CPUID in a relaxed atomic after first detection (or
    /// folding to static [`Self::Avx512`] when compiled with AVX-512).
    #[inline(always)]
    pub(super) fn detect() -> Self {
        cfg_select! {
            all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            ) => Self::Avx512,
            _ => detect::detect_tier(),
        }
    }

    #[inline(always)]
    pub(super) fn filter_aos_zstd_leaf<const HAS_ALPHA: bool>(
        self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
        state: &mut CompressorWorkerState,
    ) -> LeafPayload {
        match self {
            #[cfg(all(
                not(target_feature = "ssse3"),
                not(target_feature = "avx2"),
                not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )),
            ))]
            Self::Scalar => super::filter_aos_zstd_leaf_with::<ScalarKernel, HAS_ALPHA>(
                ScalarKernel,
                aos,
                stride_px,
                rect,
                state,
            ),
            #[cfg(all(
                not(target_feature = "avx2"),
                not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )),
            ))]
            // SAFETY: `X86Tier::Ssse3` is produced only after verifying `ssse3`.
            Self::Ssse3 => unsafe {
                dispatch_ssse3::filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state)
            },
            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )))]
            // SAFETY: `X86Tier::Avx2` is produced only after verifying `avx2`.
            Self::Avx2 => unsafe {
                dispatch_avx2::filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state)
            },
            // SAFETY: `X86Tier::Avx512` is produced only after verifying
            // `avx512bw + avx512vl + avx512vbmi`.
            Self::Avx512 => unsafe {
                dispatch_avx512::filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state)
            },
        }
    }

    #[inline(always)]
    pub(super) fn encode_alpha_plane(
        self,
        state: &mut CompressorWorkerState,
        tw: usize,
        th: usize,
    ) -> AlphaChannel {
        match self {
            #[cfg(all(
                not(target_feature = "ssse3"),
                not(target_feature = "avx2"),
                not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )),
            ))]
            Self::Scalar => super::encode_alpha_plane_with(ScalarKernel, state, tw, th),
            #[cfg(all(
                not(target_feature = "avx2"),
                not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )),
            ))]
            // SAFETY: `X86Tier::Ssse3` is produced only after verifying `ssse3`.
            Self::Ssse3 => unsafe { dispatch_ssse3::encode_alpha_plane(state, tw, th) },
            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )))]
            // SAFETY: `X86Tier::Avx2` is produced only after verifying `avx2`.
            Self::Avx2 => unsafe { dispatch_avx2::encode_alpha_plane(state, tw, th) },
            // SAFETY: `X86Tier::Avx512` is produced only after verifying
            // `avx512bw + avx512vl + avx512vbmi`.
            Self::Avx512 => unsafe { dispatch_avx512::encode_alpha_plane(state, tw, th) },
        }
    }

    /// # Safety
    /// Caller must uphold [`super::unfilter_tiles_worker_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn unfilter_tiles_worker<'a, F>(
        self,
        has_alpha: bool,
        tile_rects: &[TileRect],
        frame: &FrameRows,
        claimed: &[AtomicBool],
        completed: &AtomicUsize,
        first_err: &Mutex<Option<anyhow::Error>>,
        cache_ctx: Option<&DecodeCacheCtx>,
        next_tile: &F,
        state: &mut DecompressorWorkerState,
    ) where
        F: Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
    {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds validated-frame/disjoint-tile invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => {
                    if has_alpha {
                        super::unfilter_tiles_worker_with::<ScalarKernel, true, F>(
                            ScalarKernel,
                            self,
                            tile_rects,
                            frame,
                            claimed,
                            completed,
                            first_err,
                            cache_ctx,
                            next_tile,
                            state,
                        );
                    } else {
                        super::unfilter_tiles_worker_with::<ScalarKernel, false, F>(
                            ScalarKernel,
                            self,
                            tile_rects,
                            frame,
                            claimed,
                            completed,
                            first_err,
                            cache_ctx,
                            next_tile,
                            state,
                        );
                    }
                },
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => {
                    if has_alpha {
                        dispatch_ssse3::unfilter_tiles_worker::<true, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    } else {
                        dispatch_ssse3::unfilter_tiles_worker::<false, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    }
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    if has_alpha {
                        dispatch_avx2::unfilter_tiles_worker::<true, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    } else {
                        dispatch_avx2::unfilter_tiles_worker::<false, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    }
                },
                Self::Avx512 => {
                    if has_alpha {
                        dispatch_avx512::unfilter_tiles_worker::<true, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    } else {
                        dispatch_avx512::unfilter_tiles_worker::<false, F>(
                            tile_rects, frame, claimed, completed, first_err, cache_ctx, next_tile,
                            state,
                        );
                    }
                },
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::blit_hits_worker_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn blit_hits_worker(
        self,
        cache: &ClientTileCache,
        hits: &[ValidatedCacheHit],
        tile_rects: &[TileRect],
        frame: &FrameRows,
        next_hit: &AtomicUsize,
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds validated-hit and disjoint-tile invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => super::blit_hits_worker_with(
                    ScalarKernel,
                    cache,
                    hits,
                    tile_rects,
                    frame,
                    next_hit,
                ),
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => {
                    dispatch_ssse3::blit_hits_worker(cache, hits, tile_rects, frame, next_hit);
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    dispatch_avx2::blit_hits_worker(cache, hits, tile_rects, frame, next_hit);
                },
                Self::Avx512 => {
                    dispatch_avx512::blit_hits_worker(cache, hits, tile_rects, frame, next_hit);
                },
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::decode_jpeg_alpha_rows_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn decode_jpeg_alpha_rows(
        self,
        a_filt: &[u8],
        jpeg_rows: &[Vec4u8],
        frame: &FrameRows,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
        alpha_row_a: &mut Vec<u8>,
        alpha_row_b: &mut Vec<u8>,
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds validated-frame/disjoint-rect invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => super::decode_jpeg_alpha_rows_with(
                    ScalarKernel,
                    a_filt,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => dispatch_ssse3::decode_jpeg_alpha_rows(
                    a_filt,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => dispatch_avx2::decode_jpeg_alpha_rows(
                    a_filt,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
                Self::Avx512 => dispatch_avx512::decode_jpeg_alpha_rows(
                    a_filt,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::copy_and_scan_rect_to_shadow_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn copy_and_scan_rect_to_shadow<const HAS_ALPHA: bool>(
        self,
        src: BufferPointer<u8>,
        shadow: &ShadowRows,
        stride: usize,
        rect: TileRect,
    ) -> RectScanMeta {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds disjoint-rect shadow invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => {
                    super::copy_and_scan_rect_to_shadow_with::<ScalarKernel, HAS_ALPHA>(
                        ScalarKernel,
                        src,
                        shadow,
                        stride,
                        rect,
                    )
                },
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => dispatch_ssse3::copy_and_scan_rect_to_shadow::<HAS_ALPHA>(
                    src, shadow, stride, rect,
                ),
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => dispatch_avx2::copy_and_scan_rect_to_shadow::<HAS_ALPHA>(
                    src, shadow, stride, rect,
                ),
                Self::Avx512 => dispatch_avx512::copy_and_scan_rect_to_shadow::<HAS_ALPHA>(
                    src, shadow, stride, rect,
                ),
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::diff_and_update_worker_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn diff_and_update_worker(
        self,
        has_alpha: bool,
        data: BufferPointer<u8>,
        aos: BufferPointer<Vec4u8>,
        shadow: &ShadowRows,
        width: usize,
        stride: usize,
        stride_px: usize,
        n_qx: usize,
        sub_bands: &[ScanSubBand],
        dirty_slots: &[AtomicU32],
        slot_scans: Option<&SlotScans>,
        cache_non_empty: bool,
        next_sub: &AtomicUsize,
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds disjoint-band shadow invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => {
                    if has_alpha {
                        super::diff_and_update_worker_with::<ScalarKernel, true>(
                            ScalarKernel,
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        super::diff_and_update_worker_with::<ScalarKernel, false>(
                            ScalarKernel,
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => {
                    if has_alpha {
                        dispatch_ssse3::diff_and_update_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_ssse3::diff_and_update_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    if has_alpha {
                        dispatch_avx2::diff_and_update_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_avx2::diff_and_update_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                Self::Avx512 => {
                    if has_alpha {
                        dispatch_avx512::diff_and_update_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_avx512::diff_and_update_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::diff_and_update_scroll_worker_with`] invariants.
    #[inline(never)]
    pub(super) unsafe fn diff_and_update_scroll_worker(
        self,
        has_alpha: bool,
        data: BufferPointer<u8>,
        aos: BufferPointer<Vec4u8>,
        shadow: &ShadowRows,
        width: usize,
        stride: usize,
        stride_px: usize,
        n_qx: usize,
        sub_bands: &[ScanSubBand],
        sr: ScrollRect,
        dirty_slots: &[AtomicU32],
        slot_scans: Option<&SlotScans>,
        cache_non_empty: bool,
        next_sub: &AtomicUsize,
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds disjoint-band and scroll-band shadow invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => {
                    if has_alpha {
                        super::diff_and_update_scroll_worker_with::<ScalarKernel, true>(
                            ScalarKernel,
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        super::diff_and_update_scroll_worker_with::<ScalarKernel, false>(
                            ScalarKernel,
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => {
                    if has_alpha {
                        dispatch_ssse3::diff_and_update_scroll_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_ssse3::diff_and_update_scroll_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    if has_alpha {
                        dispatch_avx2::diff_and_update_scroll_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_avx2::diff_and_update_scroll_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
                Self::Avx512 => {
                    if has_alpha {
                        dispatch_avx512::diff_and_update_scroll_worker::<true>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    } else {
                        dispatch_avx512::diff_and_update_scroll_worker::<false>(
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            sub_bands,
                            sr,
                            dirty_slots,
                            slot_scans,
                            cache_non_empty,
                            next_sub,
                        );
                    }
                },
            }
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn diff_and_update_strip<const HAS_ALPHA: bool>(
        self,
        aos: BufferPointer<Vec4u8>,
        shadow: &ShadowRows,
        stride_px: usize,
        rect: TileRect,
    ) -> bool {
        // SAFETY: Caller verified `self` is host-supported and upheld disjoint-rect invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => super::diff_and_update_strip_with::<ScalarKernel, HAS_ALPHA>(
                    ScalarKernel,
                    aos,
                    shadow,
                    stride_px,
                    rect,
                ),
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => {
                    dispatch_ssse3::diff_and_update_strip::<HAS_ALPHA>(aos, shadow, stride_px, rect)
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    dispatch_avx2::diff_and_update_strip::<HAS_ALPHA>(aos, shadow, stride_px, rect)
                },
                Self::Avx512 => dispatch_avx512::diff_and_update_strip::<HAS_ALPHA>(
                    aos, shadow, stride_px, rect,
                ),
            }
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
        self,
        data: BufferPointer<u8>,
        aos: BufferPointer<Vec4u8>,
        shadow: &ShadowRows,
        width: usize,
        stride: usize,
        y0: usize,
        y1: usize,
        dirty_slots: &[AtomicU32],
    ) {
        // SAFETY: Caller verified `self` is host-supported and upheld disjoint-row invariants.
        unsafe {
            match self {
                #[cfg(all(
                    not(target_feature = "ssse3"),
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Scalar => {
                    super::diff_and_update_row_band_2d_with::<ScalarKernel, HAS_ALPHA>(
                        ScalarKernel,
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        y0,
                        y1,
                        0,
                        dirty_slots,
                        None,
                        false,
                    );
                },
                #[cfg(all(
                    not(target_feature = "avx2"),
                    not(all(
                        target_feature = "avx512bw",
                        target_feature = "avx512vl",
                        target_feature = "avx512vbmi",
                    )),
                ))]
                Self::Ssse3 => dispatch_ssse3::diff_and_update_row_band_2d::<HAS_ALPHA>(
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    y0,
                    y1,
                    dirty_slots,
                ),
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => dispatch_avx2::diff_and_update_row_band_2d::<HAS_ALPHA>(
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    y0,
                    y1,
                    dirty_slots,
                ),
                Self::Avx512 => dispatch_avx512::diff_and_update_row_band_2d::<HAS_ALPHA>(
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    y0,
                    y1,
                    dirty_slots,
                ),
            }
        }
    }

    /// Returns every compiled x86_64 tier supported by the current host CPU in
    /// ascending order of ISA width.
    #[cfg(test)]
    pub(super) fn supported_tiers() -> Vec<Self> {
        let mut tiers = Vec::new();
        #[cfg(all(
            not(target_feature = "ssse3"),
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ))]
        tiers.push(Self::Scalar);

        #[cfg(all(
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ))]
        if std::arch::is_x86_feature_detected!("ssse3") {
            tiers.push(Self::Ssse3);
        }

        #[cfg(not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )))]
        if std::arch::is_x86_feature_detected!("avx2") {
            tiers.push(Self::Avx2);
        }

        if std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
        {
            tiers.push(Self::Avx512);
        }

        tiers
    }
}

#[cfg(test)]
pub(super) fn verify_all_supported_kernels() -> Vec<X86Tier> {
    let mut tested = Vec::new();
    #[cfg(all(
        not(target_feature = "ssse3"),
        not(target_feature = "avx2"),
        not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )),
    ))]
    {
        super::tests::verify_kernel_matches_scalar(ScalarKernel, "scalar");
        tested.push(X86Tier::Scalar);
    }

    #[cfg(all(
        not(target_feature = "avx2"),
        not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )),
    ))]
    if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: `is_x86_feature_detected!("ssse3")` verified above.
        unsafe { dispatch_ssse3::verify_kernel() };
        tested.push(X86Tier::Ssse3);
    }

    #[cfg(not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
    )))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: `is_x86_feature_detected!("avx2")` verified above.
        unsafe { dispatch_avx2::verify_kernel() };
        tested.push(X86Tier::Avx2);
    }

    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
    {
        // SAFETY: `avx512bw + avx512vl + avx512vbmi` verified above.
        unsafe { dispatch_avx512::verify_kernel() };
        tested.push(X86Tier::Avx512);
    }

    tested
}
