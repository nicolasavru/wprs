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

use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::uint8x16x3_t;
use std::arch::aarch64::uint8x16x4_t;
use std::arch::aarch64::vabsq_s8;
use std::arch::aarch64::vaddq_u8;
use std::arch::aarch64::vaddvq_u32;
use std::arch::aarch64::vdupq_laneq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vdupq_n_u32;
use std::arch::aarch64::veorq_u8;
use std::arch::aarch64::vextq_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vld3q_u8;
use std::arch::aarch64::vld4q_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vpadalq_u16;
use std::arch::aarch64::vpaddlq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_s8_u8;
use std::arch::aarch64::vreinterpretq_u8_s8;
use std::arch::aarch64::vreinterpretq_u8_u32;
use std::arch::aarch64::vst1q_u8;
use std::arch::aarch64::vst3q_u8;
use std::arch::aarch64::vst4q_u8;
use std::arch::aarch64::vsubq_u8;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::RowKernel;
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

pub(super) struct Kernel;

impl RowKernel for Kernel {
    const STREAM_ALIGN: usize = 1;

    #[inline]
    fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        dst_bgr: &mut [u8],
        dst_alpha: &mut [u8],
    ) {
        let tw = src.len();
        debug_assert_eq!(dst_bgr.len(), tw * 3);
        debug_assert!(!HAS_ALPHA || dst_alpha.len() == tw);

        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        // SAFETY: `x + 16 <= tw`, so `vld4q_u8` reads 64 bytes in bounds,
        // `vst3q_u8` writes 48 bytes in bounds, and `vst1q_u8` writes 16 bytes in bounds.
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
        let mut sum = 0u64;
        for block in data.chunks(SAD_CHUNK_BYTES) {
            let lim = (block.len() / 16) * 16;
            // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= block.len()`.
            unsafe {
                let mut acc = vdupq_n_u32(0);
                let mut i = 0usize;
                while i < lim {
                    let v = vld1q_u8(block.as_ptr().add(i));
                    acc = vpadalq_u16(acc, vpaddlq_u8(abs_signed_u8(v)));
                    i += 16;
                }
                sum += u64::from(vaddvq_u32(acc));
            }
            for &b in &block[lim..] {
                sum += u64::from((b as i8).unsigned_abs());
            }
        }
        sum
    }

    #[inline]
    fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64 {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        let mut sum = 0u64;
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let chunk_end = (i + SAD_CHUNK_BYTES).min(lim);
                let mut acc = vdupq_n_u32(0);
                while i < chunk_end {
                    let v = vld1q_u8(src.as_ptr().add(i));
                    let shifted = vextq_u8::<13>(prev, v);
                    let diff = vsubq_u8(v, shifted);
                    prev = v;
                    vst1q_u8(dst.as_mut_ptr().add(i), diff);
                    acc = vpadalq_u16(acc, vpaddlq_u8(abs_signed_u8(diff)));
                    i += 16;
                }
                sum += u64::from(vaddvq_u32(acc));
            }
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
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut i = 0usize;
            while i < lim {
                let chunk_end = (i + SAD_CHUNK_BYTES).min(lim);
                let mut acc = vdupq_n_u32(0);
                while i < chunk_end {
                    let c = vld1q_u8(cur.as_ptr().add(i));
                    let u = vld1q_u8(prev.as_ptr().add(i));
                    let diff = vsubq_u8(c, u);
                    vst1q_u8(dst.as_mut_ptr().add(i), diff);
                    acc = vpadalq_u16(acc, vpaddlq_u8(abs_signed_u8(diff)));
                    i += 16;
                }
                sum += u64::from(vaddvq_u32(acc));
            }
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
        // SAFETY: NEON is mandatory on AArch64 and `i + 16 <= len`.
        unsafe {
            let mut prev_up = vdupq_n_u8(0);
            let mut i = 0usize;
            while i < lim {
                let chunk_end = (i + SAD_CHUNK_BYTES).min(lim);
                let mut acc = vdupq_n_u32(0);
                while i < chunk_end {
                    let c = vld1q_u8(cur.as_ptr().add(i));
                    let u = vld1q_u8(prev.as_ptr().add(i));
                    let up = vsubq_u8(c, u);
                    let shifted = vextq_u8::<13>(prev_up, up);
                    let diff = vsubq_u8(up, shifted);
                    prev_up = up;
                    vst1q_u8(dst.as_mut_ptr().add(i), diff);
                    acc = vpadalq_u16(acc, vpaddlq_u8(abs_signed_u8(diff)));
                    i += 16;
                }
                sum += u64::from(vaddvq_u32(acc));
            }
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

    #[inline]
    fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
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

    #[inline]
    fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
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
            *out = Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            );
        }
    }

    #[inline]
    unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8) {
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
        dst[lim..].fill(pixel);
    }

    #[inline]
    fn is_constant_plane(data: &[u8]) -> Option<u8> {
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

    #[inline]
    fn row_sub1_forward(src: &[u8], dst: &mut [u8]) {
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

    #[inline]
    fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
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

    #[inline]
    fn row_sub1_inverse(src: &[u8], dst: &mut [u8]) {
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

    #[inline]
    fn fence_stream() {}
}
