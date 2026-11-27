// Copyright 2026 Google LLC
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

//! AArch64 NEON row kernels for direct 3-byte `SubGreen(BGR)` AoS filtering.
//!
//! Uses hardware structure load/store (`vld4q_u8`/`vst3q_u8` on encode and
//! `vld3q_u8`/`vst4q_u8` on decode), `vextq_u8` for 3-byte and 1-byte shifts,
//! `vqtbl1q_u8` for 3-byte prefix-sum carry broadcast, and `vabsq_s8` +
//! `vpaddlq_u8`/`vpadalq_u16` for fused signed-byte SAD.
//!
//! # `unsafe` and `#[target_feature]`
//!
//! NEON is architecturally mandatory on AArch64 (ARMv8-A and later), but
//! `std::arch::aarch64` intrinsics carry `#[target_feature(enable = "neon")]`.
//! Per RFC 2396 (stabilized in Rust 1.86), calling such functions requires
//! `unsafe` unless the *caller* also carries a matching `#[target_feature]`
//! annotation — the compilation target's default features alone do not suffice.

use std::arch::aarch64::uint8x8x3_t;
use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::uint8x16x3_t;
use std::arch::aarch64::uint8x16x4_t;
use std::arch::aarch64::vabsq_s8;
use std::arch::aarch64::vaddq_u8;
use std::arch::aarch64::vaddv_u8;
use std::arch::aarch64::vaddvq_u32;
use std::arch::aarch64::vand_u8;
use std::arch::aarch64::vandq_u8;
use std::arch::aarch64::vandq_u32;
use std::arch::aarch64::vbslq_u32;
use std::arch::aarch64::vcombine_u8;
use std::arch::aarch64::vdup_n_u8;
use std::arch::aarch64::vdupq_laneq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vdupq_n_u32;
use std::arch::aarch64::veorq_u8;
use std::arch::aarch64::vextq_u8;
use std::arch::aarch64::vget_high_u8;
use std::arch::aarch64::vget_low_u8;
use std::arch::aarch64::vld1_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vld1q_u32;
use std::arch::aarch64::vld2q_u8;
use std::arch::aarch64::vld3q_u8;
use std::arch::aarch64::vld4_u8;
use std::arch::aarch64::vld4q_u8;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vorrq_u8;
use std::arch::aarch64::vpadalq_u16;
use std::arch::aarch64::vpaddlq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_s8_u8;
use std::arch::aarch64::vreinterpretq_u8_s8;
use std::arch::aarch64::vreinterpretq_u8_u16;
use std::arch::aarch64::vreinterpretq_u8_u32;
use std::arch::aarch64::vreinterpretq_u16_u8;
use std::arch::aarch64::vreinterpretq_u32_u8;
use std::arch::aarch64::vshlq_n_u8;
use std::arch::aarch64::vshr_n_u8;
use std::arch::aarch64::vshrq_n_u8;
use std::arch::aarch64::vst1_u8;
use std::arch::aarch64::vst1q_u8;
use std::arch::aarch64::vst3_u8;
use std::arch::aarch64::vst3q_u8;
use std::arch::aarch64::vst4q_u8;
use std::arch::aarch64::vsub_u8;
use std::arch::aarch64::vsubq_u8;
use std::arch::aarch64::vtstq_u8;
use std::arch::aarch64::vtstq_u32;
use std::arch::aarch64::vzip1q_u8;
use std::arch::aarch64::vzip1q_u16;
use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::ClientTileCache;
use crate::filtering::DecodeCacheCtx;
use crate::filtering::FrameRows;
use crate::filtering::RowKernel;
use crate::filtering::ScanSubBand;
use crate::filtering::ShadowRows;
use crate::filtering::palette::MAX_PALETTE_COLORS;
use crate::filtering::palette::PaletteLut4;
use crate::filtering::palette::decode_palette_1bpp_scalar;
use crate::filtering::palette::decode_palette_2bpp_scalar;
use crate::filtering::palette::decode_palette_4bpp_scalar;
use crate::filtering::palette::decode_palette_8bpp_scalar;
use crate::filtering::palette::pack_palette_indices_scalar;
use crate::filtering::palette::validate_palette_indices_scalar;
use crate::filtering::tile_cache::SlotScans;
use crate::filtering::tile_cache::ValidatedCacheHit;
use crate::filtering::tile_hash::RectScanMeta;
use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
use crate::prelude::*;
use crate::serialization::framing::AlphaChannel;
use crate::serialization::framing::CompressedTile;
use crate::serialization::framing::LeafPayload;
use crate::serialization::framing::ScrollRect;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::packed_palette_indices_len;
use crate::sharding_compression::CompressorWorkerState;
use crate::sharding_compression::DecompressorWorkerState;
use crate::vec4u8::Vec4u8;

/// Table indices repeating pixel 3 (`bytes 9, 10, 11`) across all four 3-byte
/// pixel slots inside a 16-byte NEON vector.
const LAST3_TBL: [u8; 16] = [9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9];

/// Max bytes accumulated into a `uint32x4_t` SAD accumulator before folding
/// into `u64` (`65,536 * 128 = 8,388,608 << u32::MAX`).
const SAD_CHUNK_BYTES: usize = 65_536;

#[inline(always)]
fn abs_signed_u8(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: register-to-register NEON; `vabsq_s8(-128)` wraps to `-128`
    // (`0x80`), which reinterpreted as `u8` is `128 == (-128i8).unsigned_abs()`.
    unsafe { vreinterpretq_u8_s8(vabsq_s8(vreinterpretq_s8_u8(v))) }
}

#[derive(Copy, Clone)]
struct Kernel;

impl RowKernel for Kernel {
    const STREAM_ALIGN: usize = 1;

    #[inline(always)]
    fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
        self,
        src: BufferPointer<Vec4u8>,
        dst_bgr: &mut [u8],
        dst_alpha: &mut [u8],
    ) {
        let tw = src.len();
        debug_assert_eq!(dst_bgr.len(), tw * 3);
        debug_assert!(!HAS_ALPHA || dst_alpha.len() == tw);

        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        // SAFETY: `x + 16 <= tw` in the loop and `x + 8 <= tw` in the 8-pixel
        // tail, so all structure loads/stores stay strictly in bounds.
        unsafe {
            let src_base = src.ptr().cast::<u8>();
            while x < lim {
                let px = vld4q_u8(src_base.add(x * 4));
                let b = vsubq_u8(px.0, px.1);
                let g = px.1;
                let r = vsubq_u8(px.2, px.1);
                vst3q_u8(dst_bgr.as_mut_ptr().add(x * 3), uint8x16x3_t(b, g, r));
                if HAS_ALPHA {
                    vst1q_u8(dst_alpha.as_mut_ptr().add(x), px.3);
                }
                x += 16;
            }
            if x + 8 <= tw {
                let px = vld4_u8(src_base.add(x * 4));
                let b = vsub_u8(px.0, px.1);
                let g = px.1;
                let r = vsub_u8(px.2, px.1);
                vst3_u8(dst_bgr.as_mut_ptr().add(x * 3), uint8x8x3_t(b, g, r));
                if HAS_ALPHA {
                    vst1_u8(dst_alpha.as_mut_ptr().add(x), px.3);
                }
                x += 8;
            }
        }
        for (off, bgr) in dst_bgr[x * 3..]
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .enumerate()
        {
            let i = x + off;
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

    #[inline(always)]
    fn row_sub3_forward_and_sads(self, cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
        let len = cur.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        let mut s_none = 0u64;
        let mut s_sub = 0u64;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let chunk_end = (i + SAD_CHUNK_BYTES).min(lim);
                let mut acc_none = vdupq_n_u32(0);
                let mut acc_sub = vdupq_n_u32(0);
                while i < chunk_end {
                    let c = vld1q_u8(cur.as_ptr().add(i));
                    let shifted = vextq_u8::<13>(prev, c);
                    let sub = vsubq_u8(c, shifted);
                    prev = c;
                    vst1q_u8(dst.as_mut_ptr().add(i), sub);
                    acc_none = vpadalq_u16(acc_none, vpaddlq_u8(abs_signed_u8(c)));
                    acc_sub = vpadalq_u16(acc_sub, vpaddlq_u8(abs_signed_u8(sub)));
                    i += 16;
                }
                s_none += u64::from(vaddvq_u32(acc_none));
                s_sub += u64::from(vaddvq_u32(acc_sub));
            }
        }
        for j in lim..len {
            let c = cur[j];
            let p = if j >= 3 { cur[j - 3] } else { 0 };
            let sub = c.wrapping_sub(p);
            dst[j] = sub;
            s_none += u64::from((c as i8).unsigned_abs());
            s_sub += u64::from((sub as i8).unsigned_abs());
        }
        (s_none, s_sub)
    }

    #[inline(always)]
    fn row_up_forward_and_sads_all4(self, cur: &[u8], prev: &[u8], dst_up: &mut [u8]) -> [u64; 4] {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst_up.len(), len);
        let lim = (len / 16) * 16;
        let mut s_none = 0u64;
        let mut s_sub = 0u64;
        let mut s_up = 0u64;
        let mut s_su = 0u64;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev_c = vdupq_n_u8(0);
            let mut prev_up = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let chunk_end = (i + SAD_CHUNK_BYTES).min(lim);
                let mut acc0 = vdupq_n_u32(0);
                let mut acc1 = vdupq_n_u32(0);
                let mut acc2 = vdupq_n_u32(0);
                let mut acc3 = vdupq_n_u32(0);
                while i < chunk_end {
                    let c = vld1q_u8(cur.as_ptr().add(i));
                    let u = vld1q_u8(prev.as_ptr().add(i));
                    let up = vsubq_u8(c, u);
                    vst1q_u8(dst_up.as_mut_ptr().add(i), up);
                    let sub = vsubq_u8(c, vextq_u8::<13>(prev_c, c));
                    let su = vsubq_u8(up, vextq_u8::<13>(prev_up, up));
                    prev_c = c;
                    prev_up = up;
                    acc0 = vpadalq_u16(acc0, vpaddlq_u8(abs_signed_u8(c)));
                    acc1 = vpadalq_u16(acc1, vpaddlq_u8(abs_signed_u8(sub)));
                    acc2 = vpadalq_u16(acc2, vpaddlq_u8(abs_signed_u8(up)));
                    acc3 = vpadalq_u16(acc3, vpaddlq_u8(abs_signed_u8(su)));
                    i += 16;
                }
                s_none += u64::from(vaddvq_u32(acc0));
                s_sub += u64::from(vaddvq_u32(acc1));
                s_up += u64::from(vaddvq_u32(acc2));
                s_su += u64::from(vaddvq_u32(acc3));
            }
        }
        for j in lim..len {
            let c = cur[j];
            let up = c.wrapping_sub(prev[j]);
            dst_up[j] = up;
            let (pc, pu) = if j >= 3 {
                (cur[j - 3], cur[j - 3].wrapping_sub(prev[j - 3]))
            } else {
                (0, 0)
            };
            let sub = c.wrapping_sub(pc);
            let su = up.wrapping_sub(pu);
            s_none += u64::from((c as i8).unsigned_abs());
            s_sub += u64::from((sub as i8).unsigned_abs());
            s_up += u64::from((up as i8).unsigned_abs());
            s_su += u64::from((su as i8).unsigned_abs());
        }
        [s_none, s_sub, s_up, s_su]
    }

    #[inline(always)]
    fn row_sub3_forward(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let v = vld1q_u8(src.as_ptr().add(i));
                let shifted = vextq_u8::<13>(prev, v);
                vst1q_u8(dst.as_mut_ptr().add(i), vsubq_u8(v, shifted));
                prev = v;
                i += 16;
            }
        }
        for j in lim..len {
            let p = if j >= 3 { src[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_sub(p);
        }
    }

    #[inline(always)]
    fn row_sub3_forward_in_place(self, buf: &mut [u8]) {
        let len = buf.len();
        let mut i = len;
        // SAFETY: NEON is mandatory on AArch64; backwards 16-byte steps write
        // `i..i + 16` strictly above all subsequent reads `..i`.
        unsafe {
            let p = buf.as_mut_ptr();
            while i >= 19 {
                i -= 16;
                let v = vld1q_u8(p.add(i));
                let sh = vld1q_u8(p.add(i - 3));
                vst1q_u8(p.add(i), vsubq_u8(v, sh));
            }
            if i >= 16 {
                if i == 18 {
                    *p.add(17) = (*p.add(17)).wrapping_sub(*p.add(14));
                }
                if i >= 17 {
                    *p.add(16) = (*p.add(16)).wrapping_sub(*p.add(13));
                }
                let v0 = vld1q_u8(p);
                let sh0 = vextq_u8::<13>(vdupq_n_u8(0), v0);
                vst1q_u8(p, vsubq_u8(v0, sh0));
                return;
            }
            for j in (3..i).rev() {
                *p.add(j) = (*p.add(j)).wrapping_sub(*p.add(j - 3));
            }
        }
    }

    #[inline(always)]
    fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: NEON is mandatory on AArch64; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            let zero = vdupq_n_u8(0);
            let tbl = vld1q_u8(LAST3_TBL.as_ptr());
            let mut prev_px = zero;
            let mut i = 0usize;
            while i < lim {
                let mut v = vld1q_u8(src.as_ptr().add(i));
                v = vaddq_u8(v, vextq_u8::<13>(zero, v));
                v = vaddq_u8(v, vextq_u8::<10>(zero, v));
                v = vaddq_u8(v, prev_px);
                prev_px = vqtbl1q_u8(v, tbl);
                vst1q_u8(dst.as_mut_ptr().add(i), v);
                i += 12;
            }
        }
        for j in lim..len {
            let p = if j >= 3 { dst[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_add(p);
        }
    }

    #[inline(always)]
    fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut i = 0usize;
            while i < lim {
                let d = vld1q_u8(src.as_ptr().add(i));
                let u = vld1q_u8(prev.as_ptr().add(i));
                vst1q_u8(dst.as_mut_ptr().add(i), vaddq_u8(d, u));
                i += 16;
            }
        }
        for j in lim..len {
            dst[j] = src[j].wrapping_add(prev[j]);
        }
    }

    #[inline(always)]
    fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        debug_assert!(len.is_multiple_of(3));
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: NEON is mandatory on AArch64; for `i < lim`, `i + 16 <= lim + 4 <= len`.
        unsafe {
            let zero = vdupq_n_u8(0);
            let tbl = vld1q_u8(LAST3_TBL.as_ptr());
            let mut prev_px = zero;
            let mut i = 0usize;
            while i < lim {
                let mut v = vld1q_u8(src.as_ptr().add(i));
                v = vaddq_u8(v, vextq_u8::<13>(zero, v));
                v = vaddq_u8(v, vextq_u8::<10>(zero, v));
                v = vaddq_u8(v, prev_px);
                prev_px = vqtbl1q_u8(v, tbl);
                let u = vld1q_u8(prev.as_ptr().add(i));
                vst1q_u8(dst.as_mut_ptr().add(i), vaddq_u8(v, u));
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

    #[inline(always)]
    unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
        self,
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(src_bgr.len(), tw * 3);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);

        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        // SAFETY: `x + 16 <= tw`, so `vld3q_u8` reads 48 bytes in bounds,
        // `vld1q_u8` reads 16 bytes in bounds, and `vst4q_u8` writes 64 bytes in bounds.
        unsafe {
            let const_a = vdupq_n_u8(const_alpha);
            let dst_base = dst.as_mut_ptr().cast::<u8>();
            while x < lim {
                let bgr = vld3q_u8(src_bgr.as_ptr().add(x * 3));
                let g = bgr.1;
                let b = vaddq_u8(bgr.0, g);
                let r = vaddq_u8(bgr.2, g);
                let a = if CONST_ALPHA {
                    const_a
                } else {
                    vld1q_u8(alpha_row.as_ptr().add(x))
                };
                vst4q_u8(dst_base.add(x * 4), uint8x16x4_t(b, g, r, a));
                x += 16;
            }
        }
        for (off, out) in dst[lim..].iter_mut().enumerate() {
            let i = lim + off;
            let s = i * 3;
            let g = src_bgr[s + 1];
            let a = if CONST_ALPHA {
                const_alpha
            } else {
                alpha_row[i]
            };
            out.write(Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            ));
        }
    }

    #[inline(always)]
    unsafe fn splat_row<const STREAM: bool>(self, dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8) {
        let lim = (dst.len() / 4) * 4;
        // SAFETY: `x + 4 <= dst.len()`, so `vst1q_u8` writes 16 bytes in bounds.
        unsafe {
            let word = u32::from_ne_bytes([pixel.0, pixel.1, pixel.2, pixel.3]);
            let vec = vreinterpretq_u8_u32(vdupq_n_u32(word));
            let base = dst.as_mut_ptr().cast::<u8>();
            let mut x = 0usize;
            while x < lim {
                vst1q_u8(base.add(x * 4), vec);
                x += 4;
            }
        }
        dst[lim..].fill(MaybeUninit::new(pixel));
    }

    #[inline(always)]
    unsafe fn copy_row_stream(self, src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]) {
        debug_assert_eq!(src.len(), dst.len());
        // SAFETY: `src` and `dst` have equal lengths and cannot overlap.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<Vec4u8>(), src.len());
        }
    }

    #[inline(always)]
    fn is_constant_plane(self, data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        let lim = (data.len() / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= data.len()`.
        unsafe {
            let target = vdupq_n_u8(first);
            let mut i = 0usize;
            while i < lim {
                let v = vld1q_u8(data.as_ptr().add(i));
                if vmaxvq_u8(veorq_u8(v, target)) != 0 {
                    return None;
                }
                i += 16;
            }
        }
        data[lim..].iter().all(|&b| b == first).then_some(first)
    }

    #[inline(always)]
    fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let v = vld1q_u8(src.as_ptr().add(i));
                let shifted = vextq_u8::<15>(prev, v);
                vst1q_u8(dst.as_mut_ptr().add(i), vsubq_u8(v, shifted));
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

    #[inline(always)]
    fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut i = 0usize;
            while i < lim {
                let c = vld1q_u8(cur.as_ptr().add(i));
                let u = vld1q_u8(prev.as_ptr().add(i));
                vst1q_u8(dst.as_mut_ptr().add(i), vsubq_u8(c, u));
                i += 16;
            }
        }
        for j in lim..len {
            dst[j] = cur[j].wrapping_sub(prev[j]);
        }
    }

    #[inline(always)]
    fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let zero = vdupq_n_u8(0);
            let mut carry = zero;
            let mut i = 0usize;
            while i < lim {
                let mut v = vld1q_u8(src.as_ptr().add(i));
                v = vaddq_u8(v, vextq_u8::<15>(zero, v));
                v = vaddq_u8(v, vextq_u8::<14>(zero, v));
                v = vaddq_u8(v, vextq_u8::<12>(zero, v));
                v = vaddq_u8(v, vextq_u8::<8>(zero, v));
                v = vaddq_u8(v, carry);
                carry = vdupq_laneq_u8::<15>(v);
                vst1q_u8(dst.as_mut_ptr().add(i), v);
                i += 16;
            }
        }
        let mut acc = if lim > 0 { dst[lim - 1] } else { 0 };
        for j in lim..len {
            acc = acc.wrapping_add(src[j]);
            dst[j] = acc;
        }
    }

    #[inline(always)]
    fn diff_and_update_row<const HAS_ALPHA: bool>(
        self,
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> Option<(usize, usize)> {
        const LANE_WEIGHTS: [u32; 4] = [1, 2, 4, 8];
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        let dst_p = shadow.as_mut_ptr();
        let lim = (n / 4) * 4;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: NEON is mandatory on AArch64, `src_p` and `dst_p` are valid
        // for `n` `Vec4u8`s and non-overlapping, and `i + 4 <= n`.
        unsafe {
            let rgb_mask = vreinterpretq_u8_u32(vdupq_n_u32(0x00FF_FFFF));
            let weights = vld1q_u32(LANE_WEIGHTS.as_ptr());
            let mut i = 0usize;
            while i < lim {
                let s = vld1q_u8(src_p.add(i).cast::<u8>());
                let d = vld1q_u8(dst_p.add(i).cast::<u8>());
                let diff = veorq_u8(s, d);
                let masked = if HAS_ALPHA {
                    diff
                } else {
                    vandq_u8(diff, rgb_mask)
                };
                if vmaxvq_u8(masked) != 0 {
                    vst1q_u8(dst_p.add(i).cast::<u8>(), s);
                    let m32 = vreinterpretq_u32_u8(masked);
                    let lane_mask = vaddvq_u32(vandq_u32(vtstq_u32(m32, m32), weights));
                    debug_assert_ne!(lane_mask, 0);
                    if first == usize::MAX {
                        first = i + lane_mask.trailing_zeros() as usize;
                    }
                    last_end = i + (32 - lane_mask.leading_zeros() as usize);
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

    #[inline(always)]
    unsafe fn copy_and_scan_row<const HAS_ALPHA: bool>(
        self,
        src: *const u8,
        dst: *mut u8,
        n_px: usize,
        first_word: u32,
    ) -> (bool, bool) {
        let lim = (n_px / 4) * 4;
        // SAFETY: NEON is mandatory on AArch64; caller guarantees `src` readable
        // and `dst` writable for `n_px * 4` bytes and non-overlapping.
        unsafe {
            let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
            let first_vec = vreinterpretq_u8_u32(vdupq_n_u32(first_word));
            let mut dacc = vdupq_n_u8(0);
            let mut aand = vdupq_n_u8(0xFF);
            let mut i = 0usize;
            while i < lim {
                let off = i * 4;
                let r = vld1q_u8(src.add(off));
                vst1q_u8(dst.add(off), r);
                let m = if HAS_ALPHA {
                    r
                } else {
                    vorrq_u8(r, alpha_mask)
                };
                dacc = vorrq_u8(dacc, veorq_u8(m, first_vec));
                if HAS_ALPHA {
                    aand = vandq_u8(aand, m);
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
            let solid = tail_diff == 0 && vmaxvq_u8(dacc) == 0;
            let opaque = !HAS_ALPHA
                || ((tail_alpha & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32
                    && vmaxvq_u8(veorq_u8(vandq_u8(aand, alpha_mask), alpha_mask)) == 0);
            (solid, opaque)
        }
    }

    #[inline(always)]
    fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool {
        let len = packed.len();
        // SAFETY: NEON is mandatory on AArch64 and all loads stay in `0..len`.
        unsafe {
            if n_colors == 3 {
                let lim = (len / 16) * 16;
                let mut acc = vdupq_n_u8(0);
                let mut i = 0usize;
                while i < lim {
                    let v = vld1q_u8(packed.as_ptr().add(i));
                    acc = vorrq_u8(acc, vandq_u8(v, vshrq_n_u8::<1>(v)));
                    i += 16;
                }
                if vmaxvq_u8(vandq_u8(acc, vdupq_n_u8(0x55))) != 0 {
                    return false;
                }
                validate_palette_indices_scalar(3, &packed[lim..])
            } else if (5..16).contains(&n_colors) {
                let lim = (len / 16) * 16;
                let mask_0f = vdupq_n_u8(0x0F);
                let mut max_lo = vdupq_n_u8(0);
                let mut max_hi = vdupq_n_u8(0);
                let mut i = 0usize;
                while i < lim {
                    let v = vld1q_u8(packed.as_ptr().add(i));
                    max_lo = vmaxq_u8(max_lo, vandq_u8(v, mask_0f));
                    max_hi = vmaxq_u8(max_hi, vshrq_n_u8::<4>(v));
                    i += 16;
                }
                if vmaxvq_u8(vmaxq_u8(max_lo, max_hi)) >= n_colors as u8 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else if (17..MAX_PALETTE_COLORS).contains(&n_colors) {
                let lim = (len / 16) * 16;
                let mut max_v = vdupq_n_u8(0);
                let mut i = 0usize;
                while i < lim {
                    max_v = vmaxq_u8(max_v, vld1q_u8(packed.as_ptr().add(i)));
                    i += 16;
                }
                if vmaxvq_u8(max_v) >= n_colors as u8 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else {
                true
            }
        }
    }

    #[inline(always)]
    fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
        const BIT_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let n = indices.len();
        let out_len = packed_palette_indices_len(n, n_colors);
        debug_assert!(packed.len() >= out_len);
        if n_colors <= 1 {
            return 0;
        }
        // SAFETY: NEON is mandatory on AArch64 and `packed.len() >= out_len`.
        unsafe {
            let src = indices.as_ptr();
            let dst = packed.as_mut_ptr();
            if n_colors <= 2 {
                let lim = (n / 16) * 16;
                let ones = vdupq_n_u8(1);
                let weights = vld1q_u8(BIT_WEIGHTS.as_ptr());
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = vld1q_u8(src.add(i));
                    let w = vandq_u8(vtstq_u8(v, ones), weights);
                    *dst.add(oi) = vaddv_u8(vget_low_u8(w));
                    *dst.add(oi + 1) = vaddv_u8(vget_high_u8(w));
                    i += 16;
                    oi += 2;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 2, &mut packed[oi..]);
                }
            } else if n_colors <= 4 {
                let lim = (n / 64) * 64;
                let m3 = vdupq_n_u8(3);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let q = vld4q_u8(src.add(i));
                    let lo = vorrq_u8(vandq_u8(q.0, m3), vshlq_n_u8::<2>(vandq_u8(q.1, m3)));
                    let hi = vorrq_u8(vshlq_n_u8::<4>(vandq_u8(q.2, m3)), vshlq_n_u8::<6>(q.3));
                    vst1q_u8(dst.add(oi), vorrq_u8(lo, hi));
                    i += 64;
                    oi += 16;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 4, &mut packed[oi..]);
                }
            } else if n_colors <= 16 {
                let lim = (n / 32) * 32;
                let m0f = vdupq_n_u8(0x0F);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let q = vld2q_u8(src.add(i));
                    let pk = vorrq_u8(vandq_u8(q.0, m0f), vshlq_n_u8::<4>(q.1));
                    vst1q_u8(dst.add(oi), pk);
                    i += 32;
                    oi += 16;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 16, &mut packed[oi..]);
                }
            } else {
                packed[..n].copy_from_slice(indices);
            }
        }
        out_len
    }

    #[inline(always)]
    unsafe fn decode_palette_1bpp_row<const STREAM: bool>(
        self,
        c0: Vec4u8,
        c1: Vec4u8,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        const BITS_LO: [u32; 4] = [1, 2, 4, 8];
        const BITS_HI: [u32; 4] = [16, 32, 64, 128];
        let tw = dst.len();
        let head = ((8 - (idx_pos & 7)) & 7).min(tw);
        if head != 0 {
            decode_palette_1bpp_scalar(c0, c1, packed, idx_pos, &mut dst[..head]);
        }
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 8 <= tw {
            debug_assert_eq!(pos & 7, 0);
            let mut bi = pos >> 3;
            let end = x + ((tw - x) / 8) * 8;
            // SAFETY: NEON is mandatory on AArch64 and caller upholds `packed` bounds.
            unsafe {
                let vc0 = vdupq_n_u32(u32::from_ne_bytes([c0.0, c0.1, c0.2, c0.3]));
                let vc1 = vdupq_n_u32(u32::from_ne_bytes([c1.0, c1.1, c1.2, c1.3]));
                let bits_lo = vld1q_u32(BITS_LO.as_ptr());
                let bits_hi = vld1q_u32(BITS_HI.as_ptr());
                let src = packed.as_ptr();
                let base = dst.as_mut_ptr().cast::<u8>();
                while x < end {
                    let v = vdupq_n_u32(u32::from(*src.add(bi)));
                    let p0 = vbslq_u32(vtstq_u32(v, bits_lo), vc1, vc0);
                    let p1 = vbslq_u32(vtstq_u32(v, bits_hi), vc1, vc0);
                    vst1q_u8(base.add(x * 4), vreinterpretq_u8_u32(p0));
                    vst1q_u8(base.add((x + 4) * 4), vreinterpretq_u8_u32(p1));
                    x += 8;
                    bi += 1;
                }
            }
            pos = bi << 3;
        }
        if x < tw {
            decode_palette_1bpp_scalar(c0, c1, packed, pos, &mut dst[x..]);
        }
    }

    #[inline(always)]
    unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
        self,
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        let head = ((4 - (idx_pos & 3)) & 3).min(tw);
        if head != 0 {
            decode_palette_2bpp_scalar(lut, packed, idx_pos, &mut dst[..head]);
        }
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 3, 0);
            let mut bi = pos >> 2;
            let end = x + ((tw - x) / 16) * 16;
            let b_arr = [
                lut[0].0, lut[1].0, lut[2].0, lut[3].0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            let g_arr = [
                lut[0].1, lut[1].1, lut[2].1, lut[3].1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            let r_arr = [
                lut[0].2, lut[1].2, lut[2].2, lut[3].2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            let a_arr = [
                lut[0].3, lut[1].3, lut[2].3, lut[3].3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            // SAFETY: NEON is mandatory on AArch64 and caller upholds `packed` bounds.
            unsafe {
                let b_lut = vld1q_u8(b_arr.as_ptr());
                let g_lut = vld1q_u8(g_arr.as_ptr());
                let r_lut = vld1q_u8(r_arr.as_ptr());
                let a_lut = vld1q_u8(a_arr.as_ptr());
                let m3 = vdupq_n_u8(3);
                let src = packed.as_ptr();
                let base = dst.as_mut_ptr().cast::<u8>();
                while x < end {
                    let w = u32::from_le(src.add(bi).cast::<u32>().read_unaligned());
                    let v = vreinterpretq_u8_u32(vdupq_n_u32(w));
                    let p0 = vandq_u8(v, m3);
                    let p1 = vandq_u8(vshrq_n_u8::<2>(v), m3);
                    let p2 = vandq_u8(vshrq_n_u8::<4>(v), m3);
                    let p3 = vshrq_n_u8::<6>(v);
                    let lo01 = vzip1q_u8(p0, p1);
                    let lo23 = vzip1q_u8(p2, p3);
                    let idx = vreinterpretq_u8_u16(vzip1q_u16(
                        vreinterpretq_u16_u8(lo01),
                        vreinterpretq_u16_u8(lo23),
                    ));
                    let b = vqtbl1q_u8(b_lut, idx);
                    let g = vqtbl1q_u8(g_lut, idx);
                    let r = vqtbl1q_u8(r_lut, idx);
                    let a = vqtbl1q_u8(a_lut, idx);
                    vst4q_u8(base.add(x * 4), uint8x16x4_t(b, g, r, a));
                    x += 16;
                    bi += 4;
                }
            }
            pos = bi << 2;
        }
        if x < tw {
            decode_palette_2bpp_scalar(lut, packed, pos, &mut dst[x..]);
        }
    }

    #[inline(always)]
    unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
        self,
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        let head = (idx_pos & 1).min(tw);
        if head != 0 {
            decode_palette_4bpp_scalar(&lut.colors, packed, idx_pos, &mut dst[..head]);
        }
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 1, 0);
            let mut bi = pos >> 1;
            let end = x + ((tw - x) / 16) * 16;
            // SAFETY: NEON is mandatory on AArch64 and caller upholds `packed` bounds.
            unsafe {
                let b_lut = vld1q_u8(lut.b.as_ptr());
                let g_lut = vld1q_u8(lut.g.as_ptr());
                let r_lut = vld1q_u8(lut.r.as_ptr());
                let a_lut = vld1q_u8(lut.a.as_ptr());
                let mask_0f = vdup_n_u8(0x0F);
                let src = packed.as_ptr();
                let base = dst.as_mut_ptr().cast::<u8>();
                while x < end {
                    let raw = vld1_u8(src.add(bi));
                    let lo = vand_u8(raw, mask_0f);
                    let hi = vshr_n_u8::<4>(raw);
                    let idx = vzip1q_u8(vcombine_u8(lo, lo), vcombine_u8(hi, hi));
                    let b = vqtbl1q_u8(b_lut, idx);
                    let g = vqtbl1q_u8(g_lut, idx);
                    let r = vqtbl1q_u8(r_lut, idx);
                    let a = vqtbl1q_u8(a_lut, idx);
                    vst4q_u8(base.add(x * 4), uint8x16x4_t(b, g, r, a));
                    x += 16;
                    bi += 8;
                }
            }
            pos = bi << 1;
        }
        if x < tw {
            decode_palette_4bpp_scalar(&lut.colors, packed, pos, &mut dst[x..]);
        }
    }

    #[inline(always)]
    unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
        self,
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(packed_row.len(), tw);
        let lim = (tw / 8) * 8;
        // SAFETY: Caller guarantees `packed_row.len() == tw`.
        unsafe {
            let src = packed_row.as_ptr();
            let out = dst.as_mut_ptr().cast::<u32>();
            let mut x = 0usize;
            while x < lim {
                let w = u64::from_le(src.add(x).cast::<u64>().read_unaligned());
                let p = out.add(x);
                p.write_unaligned(lut[(w & 0xFF) as usize]);
                p.add(1).write_unaligned(lut[((w >> 8) & 0xFF) as usize]);
                p.add(2).write_unaligned(lut[((w >> 16) & 0xFF) as usize]);
                p.add(3).write_unaligned(lut[((w >> 24) & 0xFF) as usize]);
                p.add(4).write_unaligned(lut[((w >> 32) & 0xFF) as usize]);
                p.add(5).write_unaligned(lut[((w >> 40) & 0xFF) as usize]);
                p.add(6).write_unaligned(lut[((w >> 48) & 0xFF) as usize]);
                p.add(7).write_unaligned(lut[(w >> 56) as usize]);
                x += 8;
            }
        }
        if lim < tw {
            decode_palette_8bpp_scalar(lut, &packed_row[lim..], &mut dst[lim..]);
        }
    }

    #[inline(always)]
    fn fence_stream(self) {}
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) struct Tier;

impl Tier {
    #[inline(always)]
    pub(super) const fn detect() -> Self {
        Self
    }

    #[inline(always)]
    pub(super) fn pack_palette_indices(
        self,
        indices: &[u8],
        n_colors: usize,
        packed: &mut [u8],
    ) -> usize {
        Kernel.pack_palette_indices(indices, n_colors, packed)
    }

    #[inline(always)]
    pub(super) fn filter_aos_zstd_leaf<const HAS_ALPHA: bool>(
        self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
        state: &mut CompressorWorkerState,
    ) -> LeafPayload {
        super::filter_aos_zstd_leaf_with::<Kernel, HAS_ALPHA>(Kernel, aos, stride_px, rect, state)
    }

    #[inline(always)]
    pub(super) fn encode_alpha_plane(
        self,
        state: &mut CompressorWorkerState,
        tw: usize,
        th: usize,
    ) -> AlphaChannel {
        super::encode_alpha_plane_with(Kernel, state, tw, th)
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
        // SAFETY: Caller upholds validated-frame and disjoint-tile invariants.
        unsafe {
            if has_alpha {
                super::unfilter_tiles_worker_with::<Kernel, true, F>(
                    Kernel, self, tile_rects, frame, claimed, completed, first_err, cache_ctx,
                    next_tile, state,
                );
            } else {
                super::unfilter_tiles_worker_with::<Kernel, false, F>(
                    Kernel, self, tile_rects, frame, claimed, completed, first_err, cache_ctx,
                    next_tile, state,
                );
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
        // SAFETY: Caller upholds validated-hit and disjoint-tile invariants.
        unsafe {
            super::blit_hits_worker_with(Kernel, cache, hits, tile_rects, frame, next_hit);
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
        // SAFETY: Caller upholds validated-frame and disjoint-rect invariants.
        unsafe {
            super::decode_jpeg_alpha_rows_with(
                Kernel,
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
        // SAFETY: Caller upholds disjoint-rect shadow invariants.
        unsafe {
            super::copy_and_scan_rect_to_shadow_with::<Kernel, HAS_ALPHA>(
                Kernel, src, shadow, stride, rect,
            )
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
        // SAFETY: Caller upholds disjoint-band shadow invariants.
        unsafe {
            if has_alpha {
                super::diff_and_update_worker_with::<Kernel, true>(
                    Kernel,
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
                super::diff_and_update_worker_with::<Kernel, false>(
                    Kernel,
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
    }

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
        // SAFETY: Caller upholds disjoint-band and scroll-band shadow invariants.
        unsafe {
            if has_alpha {
                super::diff_and_update_scroll_worker_with::<Kernel, true>(
                    Kernel,
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
                super::diff_and_update_scroll_worker_with::<Kernel, false>(
                    Kernel,
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
    }

    #[cfg(test)]
    pub(super) unsafe fn diff_and_update_strip<const HAS_ALPHA: bool>(
        self,
        aos: BufferPointer<Vec4u8>,
        shadow: &ShadowRows,
        stride_px: usize,
        rect: TileRect,
    ) -> bool {
        // SAFETY: Caller upholds disjoint-rect shadow invariants.
        unsafe {
            super::diff_and_update_strip_with::<Kernel, HAS_ALPHA>(
                Kernel, aos, shadow, stride_px, rect,
            )
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
        // SAFETY: Caller upholds disjoint-row shadow invariants.
        unsafe {
            super::diff_and_update_row_band_2d_with::<Kernel, HAS_ALPHA>(
                Kernel,
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
    pub(super) fn supported_tiers() -> Vec<Self> {
        vec![Self]
    }
}

#[cfg(test)]
pub(super) fn verify_all_supported_kernels() -> Vec<Tier> {
    super::tests::verify_kernel_matches_scalar(Kernel, "neon");
    vec![Tier]
}
