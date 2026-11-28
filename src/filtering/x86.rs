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
//!   is enabled; 256-bit AVX2 + parallel two-128b-lane (24-byte) 3-byte prefix
//!   sum + fused `Up+SubGreen->BGRA` + 32-byte `_mm256_stream_si256` stores.
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
use crate::filtering::RowKernel;
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

/// Defines a single-pass `SIMD + aes` active `RowBand2D` finisher inside an
/// x86 SIMD tier module (`avx512`, `avx2`, or `ssse3`). Walks each remaining
/// row strictly contiguously from `x = 0..width` (zero 512 B prefetcher holes),
/// executing `CopyAndAbsorb` in-place on `bbox.full` quadrants and absorbing
/// `!bbox.full` active quadrants directly from the L1d-hot 512 B `shadow` chunk
/// just scanned by `diff_and_update_row_impl`, with VEX `vaesenc` (zero
/// `vzeroupper` transitions on AVX2/AVX-512) and a `copy_and_absorb` fast exit
/// once `full_count == n_qx`.
macro_rules! impl_finish_active_row_band_aes {
    ($base_feat:literal, $aes_feat:literal) => {
        #[inline(never)]
        #[target_feature(enable = $base_feat)]
        unsafe fn diff_and_update_row_band_2d_impl<const HAS_ALPHA: bool>(
            kernel: Kernel,
            ctx: &crate::filtering::ActiveRowBandCtx<'_>,
        ) {
            // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
            unsafe {
                crate::filtering::diff_and_update_row_band_2d_with::<Kernel, HAS_ALPHA>(
                    kernel, ctx,
                );
            }
        }

        #[cold]
        #[inline(never)]
        #[target_feature(enable = $base_feat)]
        unsafe fn diff_and_update_active_band_fallback<const HAS_ALPHA: bool>(
            kernel: Kernel,
            ctx: &crate::filtering::ActiveRowBandCtx<'_>,
            qx_bboxes: &mut [crate::filtering::QuadBbox],
            trigger_y: usize,
            full_count: usize,
        ) {
            // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
            unsafe {
                crate::filtering::finish_active_row_band_2d_with::<Kernel, HAS_ALPHA>(
                    kernel, ctx, qx_bboxes, trigger_y, full_count,
                );
            }
        }

        #[inline(never)]
        #[target_feature(enable = $aes_feat)]
        unsafe fn diff_and_update_active_band_aes<const HAS_ALPHA: bool>(
            ctx: &crate::filtering::ActiveRowBandCtx<'_>,
            scans: &crate::filtering::SlotScans,
            qx_bboxes: &mut [crate::filtering::QuadBbox],
            trigger_y: usize,
            mut full_count: usize,
        ) {
            use std::arch::x86_64::_mm_set1_epi32;
            use std::mem::MaybeUninit;
            use std::ptr;
            use std::slice;
            use std::sync::atomic::Ordering;

            use crate::filtering::CROP_CACHE_LOOKUP_MIN_PIXELS;
            use crate::filtering::CROP_MAX_AREA_PCT;
            use crate::filtering::STACK_HASH_CAP;
            use crate::filtering::SUBTILE_W;
            use crate::filtering::SlotDirty;
            use crate::filtering::tile_hash::BandRowAction;
            use crate::filtering::tile_hash::RowTileHasher;
            use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
            use crate::filtering::tile_hash::x86_aes;
            use crate::vec4u8::Vec4u8;

            let width = ctx.width;
            let stride = ctx.stride;
            let stride_px = stride / 4;
            let y0 = ctx.y0;
            let y1 = ctx.y1;
            let ch = y1 - y0;
            let n_qx = ctx.dirty_slots.len();
            let key = scans.key();

            let mut stack_actions = [BandRowAction::Skip; STACK_HASH_CAP];
            let mut heap_actions;
            let qx_actions: &mut [BandRowAction] = if n_qx <= STACK_HASH_CAP {
                &mut stack_actions[..n_qx]
            } else {
                heap_actions = vec![BandRowAction::Skip; n_qx];
                &mut heap_actions
            };
            let mut stack_hashers = [MaybeUninit::<RowTileHasher>::uninit(); STACK_HASH_CAP];
            let mut heap_hashers;
            let qx_hashers: &mut [MaybeUninit<RowTileHasher>] = if n_qx <= STACK_HASH_CAP {
                &mut stack_hashers[..n_qx]
            } else {
                heap_hashers = vec![MaybeUninit::<RowTileHasher>::uninit(); n_qx];
                &mut heap_hashers
            };

            let absorbed_init = (trigger_y - y0) + 1;
            let mut x = 0usize;
            for ((bbox, act), slot_h) in qx_bboxes
                .iter()
                .zip(qx_actions.iter_mut())
                .zip(qx_hashers.iter_mut())
            {
                let cw = (width - x).min(SUBTILE_W);
                let start_next = if bbox.full {
                    Some(BandRowAction::CopyAndAbsorb)
                } else if ctx.cache_non_empty && bbox.is_dirty() {
                    let area =
                        usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
                    (area >= CROP_CACHE_LOOKUP_MIN_PIXELS).then_some(BandRowAction::AbsorbShadow)
                } else {
                    None
                };
                if let Some(next) = start_next {
                    // SAFETY: Rows `[y0, trigger_y]` are finalized in `shadow`.
                    let h = unsafe {
                        x86_aes::start_row_hasher_from_shadow::<HAS_ALPHA>(
                            key,
                            ctx.shadow.base.cast_const(),
                            stride,
                            x * 4,
                            y0,
                            cw,
                            ch,
                            absorbed_init,
                        )
                    };
                    slot_h.write(h);
                    *act = next;
                }
                x += cw;
            }

            if trigger_y + 1 < y1 && width > 0 {
                let last_end = (y1 - 1) * stride + width * 4;
                assert!(last_end <= ctx.data.len());
            }

            let rk = x86_aes::load_key_block(&key.round_key);
            let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
            let src_base = ctx.data.ptr();

            for y in (trigger_y + 1)..y1 {
                if full_count == n_qx {
                    // SAFETY: Bounds validated above; every hasher slot is initialized.
                    unsafe {
                        x86_aes::copy_and_absorb_band_rows::<HAS_ALPHA>(
                            key,
                            src_base,
                            ctx.shadow.base,
                            stride,
                            width,
                            y,
                            y1,
                            qx_hashers,
                        );
                    }
                    break;
                }

                let ry = y - y0;
                let row_off = y * stride;
                // SAFETY: `[y0, y1)` bounds validated above; `shadow` is exclusively owned.
                let src_row_base = unsafe { src_base.add(row_off) };
                let sh_row_base = unsafe { ctx.shadow.base.add(row_off) };
                let (_, row_after) = ctx.aos.split_at(y * stride_px);
                let (mut rem_aos, _) = row_after.split_at(width);
                let dst_row =
                    unsafe { slice::from_raw_parts_mut(sh_row_base.cast::<Vec4u8>(), width) };

                let mut x = 0usize;
                for ((bbox, act), slot_h) in qx_bboxes
                    .iter_mut()
                    .zip(qx_actions.iter_mut())
                    .zip(qx_hashers.iter_mut())
                {
                    let cw = (width - x).min(SUBTILE_W);
                    let (src_chunk, rest_aos) = rem_aos.split_at(cw);
                    rem_aos = rest_aos;
                    let x_bytes = x * 4;
                    if bbox.full {
                        // SAFETY: `bbox.full` guarantees `slot_h` was initialized and
                        // transitioned to `CopyAndAbsorb` on an earlier row.
                        let h = unsafe { slot_h.assume_init_mut() };
                        unsafe {
                            x86_aes::x86_absorb_one_hasher_row!(
                                HAS_ALPHA,
                                true,
                                h,
                                src_row_base.add(x_bytes),
                                sh_row_base.add(x_bytes),
                                rk,
                                alpha_mask
                            );
                        }
                    } else {
                        let dst_chunk = &mut dst_row[x..x + cw];
                        let mut became_full = false;
                        let mut dirty = false;
                        // SAFETY: Caller guarantees this tier's SIMD feature is enabled.
                        if let Some((fx, lx)) =
                            unsafe { diff_and_update_row_impl::<HAS_ALPHA>(src_chunk, dst_chunk) }
                        {
                            dirty = true;
                            let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
                            became_full = bbox.record_row(ry, fx, lx, max_crop_area);
                            if became_full {
                                full_count += 1;
                            }
                        }
                        if *act != BandRowAction::Skip {
                            // SAFETY: Non-`Skip` guarantees `slot_h` is initialized;
                            // `sh_row_base + x_bytes` is L1d-hot from the diff above.
                            let h = unsafe { slot_h.assume_init_mut() };
                            unsafe {
                                x86_aes::x86_absorb_one_hasher_row!(
                                    HAS_ALPHA,
                                    false,
                                    h,
                                    sh_row_base.cast_const().add(x_bytes),
                                    ptr::null_mut(),
                                    rk,
                                    alpha_mask
                                );
                            }
                            if became_full {
                                *act = BandRowAction::CopyAndAbsorb;
                            }
                        } else if dirty {
                            let area = usize::from(bbox.max_x - bbox.min_x)
                                * usize::from(bbox.max_y - bbox.min_y);
                            if became_full
                                || (ctx.cache_non_empty && area >= CROP_CACHE_LOOKUP_MIN_PIXELS)
                            {
                                // SAFETY: Rows `[y0, y]` are finalized in `shadow`.
                                let h = unsafe {
                                    x86_aes::start_row_hasher_from_shadow::<HAS_ALPHA>(
                                        key,
                                        ctx.shadow.base.cast_const(),
                                        stride,
                                        x_bytes,
                                        y0,
                                        cw,
                                        ch,
                                        ry + 1,
                                    )
                                };
                                slot_h.write(h);
                                *act = if became_full {
                                    BandRowAction::CopyAndAbsorb
                                } else {
                                    BandRowAction::AbsorbShadow
                                };
                            }
                        }
                    }
                    x += cw;
                }
            }

            // SAFETY: Every non-`Skip` slot was initialized and absorbed all `ch` rows.
            unsafe {
                x86_aes::finish_band_hashers(key, qx_actions, qx_hashers, ctx.slot_base, scans);
            }

            for (slot, bbox) in ctx.dirty_slots.iter().zip(qx_bboxes.iter()) {
                if bbox.full {
                    slot.store(SlotDirty::FULL_RAW, Ordering::Relaxed);
                } else if bbox.is_dirty() {
                    let crop_area =
                        usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
                    let has_quad_hash =
                        ctx.cache_non_empty && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
                    slot.store(
                        SlotDirty::encode_cropped(
                            bbox.min_x,
                            bbox.max_x,
                            bbox.min_y,
                            bbox.max_y,
                            has_quad_hash,
                        ),
                        Ordering::Relaxed,
                    );
                }
            }
        }

        #[inline(never)]
        #[target_feature(enable = $base_feat)]
        unsafe fn detect_scroll_phase1b_impl<const HAS_ALPHA: bool>(
            kernel: Kernel,
            aos: crate::buffer_pointer::BufferPointer<crate::vec4u8::Vec4u8>,
            prev: &[crate::vec4u8::Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            n_strips: usize,
            tier0_step_y: usize,
            changed_rows: &[u16; crate::filtering::scroll::MAX_STRIPS],
            changed_non_uniform: &[u16; crate::filtering::scroll::MAX_STRIPS],
            min_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
            max_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: Caller upholds band and buffer bounds invariants.
            unsafe {
                crate::filtering::scroll::detect_scroll_phase1b_with::<Kernel, HAS_ALPHA>(
                    kernel,
                    aos,
                    prev,
                    width,
                    stride_px,
                    band_y0,
                    band_y1,
                    n_strips,
                    tier0_step_y,
                    changed_rows,
                    changed_non_uniform,
                    min_y,
                    max_y,
                )
            }
        }

        #[inline(never)]
        #[target_feature(enable = $base_feat)]
        unsafe fn expand_and_verify_scroll_candidate_impl<const HAS_ALPHA: bool>(
            kernel: Kernel,
            aos: crate::buffer_pointer::BufferPointer<crate::vec4u8::Vec4u8>,
            prev: &[crate::vec4u8::Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            seed_sx: usize,
            dy: i32,
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: Caller upholds band and buffer bounds invariants.
            unsafe {
                crate::filtering::scroll::expand_and_verify_candidate_with::<Kernel, HAS_ALPHA>(
                    kernel, aos, prev, width, stride_px, band_y0, band_y1, seed_sx, dy,
                )
            }
        }
    };
}

mod avx512 {
    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm_cvtsi128_si64;
    use std::arch::x86_64::_mm_extract_epi64;
    use std::arch::x86_64::_mm_loadl_epi64;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_sfence;
    use std::arch::x86_64::_mm_srli_epi16;
    use std::arch::x86_64::_mm_storeu_si128;
    use std::arch::x86_64::_mm_unpacklo_epi8;
    use std::arch::x86_64::_mm_xor_si128;
    use std::arch::x86_64::_mm256_add_epi64;
    use std::arch::x86_64::_mm256_castsi256_si128;
    use std::arch::x86_64::_mm256_extracti128_si256;
    use std::arch::x86_64::_mm256_setr_epi32;
    use std::arch::x86_64::_mm256_storeu_si256;
    use std::arch::x86_64::_mm256_xor_si256;
    use std::arch::x86_64::_mm512_abs_epi8;
    use std::arch::x86_64::_mm512_add_epi8;
    use std::arch::x86_64::_mm512_add_epi64;
    use std::arch::x86_64::_mm512_and_si512;
    use std::arch::x86_64::_mm512_andnot_si512;
    use std::arch::x86_64::_mm512_broadcast_i32x4;
    use std::arch::x86_64::_mm512_broadcastd_epi32;
    use std::arch::x86_64::_mm512_bslli_epi128;
    use std::arch::x86_64::_mm512_castsi256_si512;
    use std::arch::x86_64::_mm512_castsi512_si128;
    use std::arch::x86_64::_mm512_castsi512_si256;
    use std::arch::x86_64::_mm512_cmpeq_epi8_mask;
    use std::arch::x86_64::_mm512_cmpeq_epi32_mask;
    use std::arch::x86_64::_mm512_cmpneq_epi8_mask;
    use std::arch::x86_64::_mm512_cvtepi16_epi8;
    use std::arch::x86_64::_mm512_cvtepi32_epi8;
    use std::arch::x86_64::_mm512_cvtepu8_epi32;
    use std::arch::x86_64::_mm512_extracti32x4_epi32;
    use std::arch::x86_64::_mm512_extracti64x4_epi64;
    use std::arch::x86_64::_mm512_inserti64x4;
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_madd_epi16;
    use std::arch::x86_64::_mm512_maddubs_epi16;
    use std::arch::x86_64::_mm512_mask_blend_epi8;
    use std::arch::x86_64::_mm512_mask_blend_epi32;
    use std::arch::x86_64::_mm512_mask_cmpneq_epi32_mask;
    use std::arch::x86_64::_mm512_mask_storeu_epi8;
    use std::arch::x86_64::_mm512_mask_test_epi8_mask;
    use std::arch::x86_64::_mm512_maskz_loadu_epi8;
    use std::arch::x86_64::_mm512_maskz_permutexvar_epi8;
    use std::arch::x86_64::_mm512_maskz_sub_epi8;
    use std::arch::x86_64::_mm512_max_epu8;
    use std::arch::x86_64::_mm512_mul_epu32;
    use std::arch::x86_64::_mm512_or_si512;
    use std::arch::x86_64::_mm512_permutex2var_epi8;
    use std::arch::x86_64::_mm512_permutexvar_epi8;
    use std::arch::x86_64::_mm512_permutexvar_epi32;
    use std::arch::x86_64::_mm512_reduce_add_epi64;
    use std::arch::x86_64::_mm512_sad_epu8;
    use std::arch::x86_64::_mm512_set1_epi8;
    use std::arch::x86_64::_mm512_set1_epi16;
    use std::arch::x86_64::_mm512_set1_epi32;
    use std::arch::x86_64::_mm512_setzero_si512;
    use std::arch::x86_64::_mm512_shuffle_epi8;
    use std::arch::x86_64::_mm512_srli_epi16;
    use std::arch::x86_64::_mm512_srli_epi64;
    use std::arch::x86_64::_mm512_srlv_epi32;
    use std::arch::x86_64::_mm512_storeu_si512;
    use std::arch::x86_64::_mm512_stream_si512;
    use std::arch::x86_64::_mm512_sub_epi8;
    use std::arch::x86_64::_mm512_subs_epu8;
    use std::arch::x86_64::_mm512_test_epi8_mask;
    use std::arch::x86_64::_mm512_xor_si512;
    use std::arch::x86_64::_mm512_zextsi128_si512;
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::ActiveRowBandCtx;
    use crate::filtering::QuadBbox;
    use crate::filtering::RowKernel;
    use crate::filtering::palette::MAX_PALETTE_COLORS;
    use crate::filtering::palette::PaletteLut4;
    use crate::filtering::palette::decode_palette_1bpp_scalar;
    use crate::filtering::palette::decode_palette_2bpp_scalar;
    use crate::filtering::palette::decode_palette_4bpp_scalar;
    use crate::filtering::palette::decode_palette_8bpp_scalar;
    use crate::filtering::palette::pack_palette_indices_scalar;
    use crate::filtering::palette::validate_palette_indices_scalar;
    use crate::filtering::tile_hash;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
    use crate::serialization::framing::packed_palette_indices_len;
    use crate::vec4u8::Vec4u8;

    const SHIFTS_2BPP_512: [u32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];

    /// Opmask selecting the lower 48 bytes (`16` 3-byte BGR pixels) of a ZMM register.
    const MASK_48: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Opmask selecting B, G, R bytes (`0, 1, 2`) of every 4-byte pixel while ignoring byte `3`.
    const MASK_RGB: u64 = 0x7777_7777_7777_7777;

    /// Opmask selecting byte `3` (alpha) of every 4-byte pixel in a 64-byte ZMM register.
    const MASK_ALPHA_BYTE: u64 = 0x8888_8888_8888_8888;

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

    /// Cross-lane `vpermt2b` indices keeping B, G, R bytes (`4*p..4*p+3` from
    /// source `a`) and replacing byte `3` of each pixel with `alpha[p]`
    /// (`64 + p` from source `b`).
    const MERGE_BGRX_ALPHA_IDX: [u8; 64] = {
        let mut t = [0u8; 64];
        let mut p = 0;
        while p < 16 {
            t[p * 4] = (p * 4) as u8;
            t[p * 4 + 1] = (p * 4 + 1) as u8;
            t[p * 4 + 2] = (p * 4 + 2) as u8;
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
        // and the masked tail handles `1..=15` trailing pixels with zero scalar loop.
        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        let mut di = 0usize;

        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; `x + 16 <= tw`,
        // `di + 48 <= dst_bgr.len()`, `x + 16 <= dst_alpha.len()` when `HAS_ALPHA`,
        // and the masked tail accesses strictly `lim..tw` (`rem <= 15`).
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
            let rem = tw - lim;
            if rem != 0 {
                let src_mask = (1u64 << (rem * 4)) - 1;
                let bgr_mask = (1u64 << (rem * 3)) - 1;
                let v = _mm512_maskz_loadu_epi8(src_mask, src.ptr().add(lim).cast());
                let g = _mm512_shuffle_epi8(v, shuf_g);
                let sg = _mm512_sub_epi8(v, g);
                let packed = _mm512_permutexvar_epi8(pack_idx, sg);
                _mm512_mask_storeu_epi8(dst_bgr.as_mut_ptr().add(di).cast(), bgr_mask, packed);
                if HAS_ALPHA {
                    let alpha_mask = (1u64 << rem) - 1;
                    let a128 = _mm512_extracti32x4_epi32::<3>(packed);
                    _mm512_mask_storeu_epi8(
                        dst_alpha.as_mut_ptr().add(lim).cast(),
                        alpha_mask,
                        _mm512_zextsi128_si512(a128),
                    );
                }
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_forward_and_sads_impl(cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
        let len = cur.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; full iterations access
        // `i..i + 64 <= len` and the masked tail accesses strictly `lim..len`.
        unsafe {
            let zero = _mm512_setzero_si512();
            let shift_idx = load_table(&SHIFT_LEFT3_IDX);
            let mut acc_none = zero;
            let mut acc_sub = zero;
            let mut prev = zero;
            let cur_p = cur.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let c = _mm512_loadu_si512(cur_p.add(i).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, c);
                let sub = _mm512_sub_epi8(c, shifted);
                prev = c;
                _mm512_storeu_si512(dst_p.add(i).cast(), sub);
                acc_none = _mm512_add_epi64(acc_none, _mm512_sad_epu8(_mm512_abs_epi8(c), zero));
                acc_sub = _mm512_add_epi64(acc_sub, _mm512_sad_epu8(_mm512_abs_epi8(sub), zero));
                i += 64;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let c = _mm512_maskz_loadu_epi8(mask, cur_p.add(lim).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, c);
                let sub = _mm512_maskz_sub_epi8(mask, c, shifted);
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), mask, sub);
                acc_none = _mm512_add_epi64(acc_none, _mm512_sad_epu8(_mm512_abs_epi8(c), zero));
                acc_sub = _mm512_add_epi64(acc_sub, _mm512_sad_epu8(_mm512_abs_epi8(sub), zero));
            }
            (
                _mm512_reduce_add_epi64(acc_none) as u64,
                _mm512_reduce_add_epi64(acc_sub) as u64,
            )
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_up_forward_and_sads_all4_impl(
        cur: &[u8],
        prev: &[u8],
        dst_up: &mut [u8],
    ) -> [u64; 4] {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst_up.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; full iterations access
        // `i..i + 64 <= len` and the masked tail accesses strictly `lim..len`.
        unsafe {
            let zero = _mm512_setzero_si512();
            let shift_idx = load_table(&SHIFT_LEFT3_IDX);
            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            let mut prev_c = zero;
            let mut prev_up = zero;
            let cur_p = cur.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst_up.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let c = _mm512_loadu_si512(cur_p.add(i).cast());
                let u = _mm512_loadu_si512(prev_p.add(i).cast());
                let up = _mm512_sub_epi8(c, u);
                _mm512_storeu_si512(dst_p.add(i).cast(), up);
                let sub = _mm512_sub_epi8(c, _mm512_permutex2var_epi8(prev_c, shift_idx, c));
                let su = _mm512_sub_epi8(up, _mm512_permutex2var_epi8(prev_up, shift_idx, up));
                prev_c = c;
                prev_up = up;
                acc0 = _mm512_add_epi64(acc0, _mm512_sad_epu8(_mm512_abs_epi8(c), zero));
                acc1 = _mm512_add_epi64(acc1, _mm512_sad_epu8(_mm512_abs_epi8(sub), zero));
                acc2 = _mm512_add_epi64(acc2, _mm512_sad_epu8(_mm512_abs_epi8(up), zero));
                acc3 = _mm512_add_epi64(acc3, _mm512_sad_epu8(_mm512_abs_epi8(su), zero));
                i += 64;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let c = _mm512_maskz_loadu_epi8(mask, cur_p.add(lim).cast());
                let u = _mm512_maskz_loadu_epi8(mask, prev_p.add(lim).cast());
                let up = _mm512_sub_epi8(c, u);
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), mask, up);
                let sub =
                    _mm512_maskz_sub_epi8(mask, c, _mm512_permutex2var_epi8(prev_c, shift_idx, c));
                let su = _mm512_maskz_sub_epi8(
                    mask,
                    up,
                    _mm512_permutex2var_epi8(prev_up, shift_idx, up),
                );
                acc0 = _mm512_add_epi64(acc0, _mm512_sad_epu8(_mm512_abs_epi8(c), zero));
                acc1 = _mm512_add_epi64(acc1, _mm512_sad_epu8(_mm512_abs_epi8(sub), zero));
                acc2 = _mm512_add_epi64(acc2, _mm512_sad_epu8(_mm512_abs_epi8(up), zero));
                acc3 = _mm512_add_epi64(acc3, _mm512_sad_epu8(_mm512_abs_epi8(su), zero));
            }
            [
                _mm512_reduce_add_epi64(acc0) as u64,
                _mm512_reduce_add_epi64(acc1) as u64,
                _mm512_reduce_add_epi64(acc2) as u64,
                _mm512_reduce_add_epi64(acc3) as u64,
            ]
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; full iterations access
        // `i..i + 64 <= len` and the masked tail accesses strictly `lim..len`.
        unsafe {
            let shift_idx = load_table(&SHIFT_LEFT3_IDX);
            let mut prev = _mm512_setzero_si512();
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let v = _mm512_loadu_si512(src_p.add(i).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                _mm512_storeu_si512(dst_p.add(i).cast(), _mm512_sub_epi8(v, shifted));
                prev = v;
                i += 64;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, src_p.add(lim).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), mask, _mm512_sub_epi8(v, shifted));
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_forward_in_place_impl(buf: &mut [u8]) {
        let len = buf.len();
        let lim = (len / 64) * 64;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; full iterations load `v`
        // before overwriting `i..i + 64 <= len`, and the masked tail uses the
        // pre-overwrite `prev` register while accessing strictly `lim..len`.
        unsafe {
            let shift_idx = load_table(&SHIFT_LEFT3_IDX);
            let mut prev = _mm512_setzero_si512();
            let p = buf.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let v = _mm512_loadu_si512(p.add(i).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                _mm512_storeu_si512(p.add(i).cast(), _mm512_sub_epi8(v, shifted));
                prev = v;
                i += 64;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, p.add(lim).cast());
                let shifted = _mm512_permutex2var_epi8(prev, shift_idx, v);
                _mm512_mask_storeu_epi8(p.add(lim).cast(), mask, _mm512_sub_epi8(v, shifted));
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 48) * 48;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; masked 48-byte loads/stores
        // strictly access `i..i + 48 <= len` and the masked tail accesses strictly
        // `lim..len` (`rem <= 47`).
        unsafe {
            let sh3 = load_table(&PREFIX_SH3_IDX);
            let sh6 = load_table(&PREFIX_SH6_IDX);
            let sh12 = load_table(&PREFIX_SH12_IDX);
            let sh24 = load_table(&PREFIX_SH24_IDX);
            let carry_idx = load_table(&CARRY_LAST3_IDX);
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut carry = _mm512_setzero_si512();
            let mut i = 0usize;
            while i < lim {
                let v = _mm512_maskz_loadu_epi8(MASK_48, src_p.add(i).cast());
                let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                let out = _mm512_add_epi8(v, carry);
                carry = _mm512_add_epi8(carry, _mm512_permutexvar_epi8(carry_idx, v));
                _mm512_mask_storeu_epi8(dst_p.add(i).cast(), MASK_48, out);
                i += 48;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, src_p.add(lim).cast());
                let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                let v = _mm512_add_epi8(v, carry);
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), mask, v);
            }
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
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; masked 48-byte loads/stores
        // strictly access `i..i + 48 <= len` and the masked tail accesses strictly
        // `lim..len` (`rem <= 47`).
        unsafe {
            let sh3 = load_table(&PREFIX_SH3_IDX);
            let sh6 = load_table(&PREFIX_SH6_IDX);
            let sh12 = load_table(&PREFIX_SH12_IDX);
            let sh24 = load_table(&PREFIX_SH24_IDX);
            let carry_idx = load_table(&CARRY_LAST3_IDX);
            let src_p = src.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut carry = _mm512_setzero_si512();
            let mut i = 0usize;
            while i < lim {
                let v = _mm512_maskz_loadu_epi8(MASK_48, src_p.add(i).cast());
                let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                let out = _mm512_add_epi8(v, carry);
                carry = _mm512_add_epi8(carry, _mm512_permutexvar_epi8(carry_idx, v));
                let u = _mm512_maskz_loadu_epi8(MASK_48, prev_p.add(i).cast());
                _mm512_mask_storeu_epi8(dst_p.add(i).cast(), MASK_48, _mm512_add_epi8(out, u));
                i += 48;
            }
            let rem = len - lim;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, src_p.add(lim).cast());
                let v = prefix_sum_sub3_48(v, sh3, sh6, sh12, sh24);
                let v = _mm512_add_epi8(v, carry);
                let u = _mm512_maskz_loadu_epi8(mask, prev_p.add(lim).cast());
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), mask, _mm512_add_epi8(v, u));
            }
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

        // Masked 48-byte loads never touch bytes `48..64`, and the masked tail
        // handles `1..=15` trailing pixels with zero scalar loop.
        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; `si + 48 <= src_bgr.len()`,
        // `x + 16 <= dst.len()`, `x + 16 <= alpha_row.len()` when `!CONST_ALPHA`,
        // caller upholds 64-byte alignment on `dst` when `STREAM` is `true`, and
        // the masked tail accesses strictly `lim..tw` (`rem <= 15`).
        unsafe {
            let shuf_g = load_table(&SHUF_G);
            let unpack_bgr = load_table(&UNPACK_BGR_IDX);
            let unpack_bgra = load_table(&UNPACK_BGRA_IDX);
            let const_alpha_vec = _mm512_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let src_p = src_bgr.as_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim {
                let bgr = _mm512_maskz_loadu_epi8(MASK_48, src_p.add(si).cast());
                let bgra = if CONST_ALPHA {
                    let sg = _mm512_permutexvar_epi8(unpack_bgr, bgr);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                } else {
                    let a128 = _mm_loadu_si128(alpha_p.add(x).cast::<__m128i>());
                    let sg =
                        _mm512_permutex2var_epi8(bgr, unpack_bgra, _mm512_zextsi128_si512(a128));
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_add_epi8(sg, g)
                };
                let p = dst_p.add(x).cast::<__m512i>();
                if STREAM {
                    _mm512_stream_si512(p, bgra);
                } else {
                    _mm512_storeu_si512(p.cast(), bgra);
                }
                x += 16;
                si += 48;
            }
            let rem = tw - lim;
            if rem != 0 {
                let bgr_mask = (1u64 << (rem * 3)) - 1;
                let dst_mask = (1u64 << (rem * 4)) - 1;
                let bgr = _mm512_maskz_loadu_epi8(bgr_mask, src_p.add(si).cast());
                let bgra = if CONST_ALPHA {
                    let sg = _mm512_permutexvar_epi8(unpack_bgr, bgr);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                } else {
                    let alpha_mask = (1u64 << rem) - 1;
                    let a = _mm512_maskz_loadu_epi8(alpha_mask, alpha_p.add(lim).cast());
                    let sg = _mm512_permutex2var_epi8(bgr, unpack_bgra, a);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_add_epi8(sg, g)
                };
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), dst_mask, bgra);
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn row_up_subgreen_bgr3_to_bgra_impl<
        const STREAM: bool,
        const CONST_ALPHA: bool,
        const WRITE_DEC: bool,
    >(
        filt_bgr: &[u8],
        prev_bgr: &[u8],
        cur_dec: &mut [u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        let row_b = tw * 3;
        debug_assert_eq!(filt_bgr.len(), row_b);
        debug_assert_eq!(prev_bgr.len(), row_b);
        debug_assert!(!WRITE_DEC || cur_dec.len() == row_b);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(64));

        let lim = (tw / 16) * 16;
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI; full iterations access
        // `si..si + 48 <= row_b` and `x..x + 16 <= tw`, `cur_dec` has `row_b`
        // bytes when `WRITE_DEC`, caller upholds 64-byte alignment on `dst`
        // when `STREAM` is `true`, and the masked tail accesses strictly `lim..tw`.
        unsafe {
            let shuf_g = load_table(&SHUF_G);
            let unpack_bgr = load_table(&UNPACK_BGR_IDX);
            let unpack_bgra = load_table(&UNPACK_BGRA_IDX);
            let const_alpha_vec = _mm512_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let filt_p = filt_bgr.as_ptr();
            let prev_p = prev_bgr.as_ptr();
            let dec_p = cur_dec.as_mut_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim {
                let f = _mm512_maskz_loadu_epi8(MASK_48, filt_p.add(si).cast());
                let u = _mm512_maskz_loadu_epi8(MASK_48, prev_p.add(si).cast());
                // Zero-masked loads guarantee lane 48 is `0 + 0 = 0` for `UNPACK_BGR_IDX`.
                let dec = _mm512_add_epi8(f, u);
                if WRITE_DEC {
                    _mm512_mask_storeu_epi8(dec_p.add(si).cast(), MASK_48, dec);
                }
                let bgra = if CONST_ALPHA {
                    let sg = _mm512_permutexvar_epi8(unpack_bgr, dec);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                } else {
                    let a128 = _mm_loadu_si128(alpha_p.add(x).cast::<__m128i>());
                    let sg =
                        _mm512_permutex2var_epi8(dec, unpack_bgra, _mm512_zextsi128_si512(a128));
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_add_epi8(sg, g)
                };
                let p = dst_p.add(x).cast::<__m512i>();
                if STREAM {
                    _mm512_stream_si512(p, bgra);
                } else {
                    _mm512_storeu_si512(p.cast(), bgra);
                }
                x += 16;
                si += 48;
            }
            let rem = tw - lim;
            if rem != 0 {
                let bgr_mask = (1u64 << (rem * 3)) - 1;
                let dst_mask = (1u64 << (rem * 4)) - 1;
                let f = _mm512_maskz_loadu_epi8(bgr_mask, filt_p.add(si).cast());
                let u = _mm512_maskz_loadu_epi8(bgr_mask, prev_p.add(si).cast());
                let dec = _mm512_add_epi8(f, u);
                if WRITE_DEC {
                    _mm512_mask_storeu_epi8(dec_p.add(si).cast(), bgr_mask, dec);
                }
                let bgra = if CONST_ALPHA {
                    let sg = _mm512_permutexvar_epi8(unpack_bgr, dec);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_or_si512(_mm512_add_epi8(sg, g), const_alpha_vec)
                } else {
                    let alpha_mask = (1u64 << rem) - 1;
                    let a = _mm512_maskz_loadu_epi8(alpha_mask, alpha_p.add(lim).cast());
                    let sg = _mm512_permutex2var_epi8(dec, unpack_bgra, a);
                    let g = _mm512_shuffle_epi8(sg, shuf_g);
                    _mm512_add_epi8(sg, g)
                };
                _mm512_mask_storeu_epi8(dst_p.add(lim).cast(), dst_mask, bgra);
            }
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
    unsafe fn merge_bgrx_alpha_row_ptr<const STREAM: bool, const CONST_ALPHA: bool>(
        src: *const Vec4u8,
        alpha_p: *const u8,
        const_alpha: u8,
        dst: *mut MaybeUninit<Vec4u8>,
        tw: usize,
    ) {
        debug_assert!(!STREAM || (dst as usize).is_multiple_of(64));
        let lim = (tw / 16) * 16;
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI, `src` readable and `dst`
        // writable for `tw` pixels (identical pointers permitted when `!STREAM`),
        // `alpha_p` readable for `tw` bytes when `!CONST_ALPHA`, and 64-byte
        // alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let const_alpha_vec = _mm512_set1_epi8(const_alpha as i8);
            let merge_idx = load_table(&MERGE_BGRX_ALPHA_IDX);
            let mut x = 0usize;
            while x < lim {
                let v = _mm512_loadu_si512(src.add(x).cast());
                let out = if CONST_ALPHA {
                    _mm512_mask_blend_epi8(MASK_ALPHA_BYTE, v, const_alpha_vec)
                } else {
                    let a128 = _mm_loadu_si128(alpha_p.add(x).cast::<__m128i>());
                    _mm512_permutex2var_epi8(v, merge_idx, _mm512_zextsi128_si512(a128))
                };
                let p = dst.add(x).cast::<__m512i>();
                if STREAM {
                    _mm512_stream_si512(p, out);
                } else {
                    _mm512_storeu_si512(p.cast(), out);
                }
                x += 16;
            }
            let rem = tw - lim;
            if rem != 0 {
                let px_mask = (1u64 << (rem * 4)) - 1;
                let v = _mm512_maskz_loadu_epi8(px_mask, src.add(lim).cast());
                let out = if CONST_ALPHA {
                    _mm512_mask_blend_epi8(MASK_ALPHA_BYTE, v, const_alpha_vec)
                } else {
                    let alpha_mask = (1u64 << rem) - 1;
                    let a = _mm512_maskz_loadu_epi8(alpha_mask, alpha_p.add(lim).cast());
                    _mm512_permutex2var_epi8(v, merge_idx, a)
                };
                _mm512_mask_storeu_epi8(dst.add(lim).cast(), px_mask, out);
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn hash_strip_avx512<const HAS_ALPHA: bool>(c: __m512i) -> u64 {
        const M: [u32; 16] = crate::filtering::scroll::STRIP_HASH_MULS;
        const MUL_EVEN: [u64; 8] = [
            M[0] as u64,
            M[2] as u64,
            M[4] as u64,
            M[6] as u64,
            M[8] as u64,
            M[10] as u64,
            M[12] as u64,
            M[14] as u64,
        ];
        const MUL_ODD: [u64; 8] = [
            M[1] as u64,
            M[3] as u64,
            M[5] as u64,
            M[7] as u64,
            M[9] as u64,
            M[11] as u64,
            M[13] as u64,
            M[15] as u64,
        ];
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI.
        unsafe {
            let cm = if HAS_ALPHA {
                c
            } else {
                _mm512_and_si512(c, _mm512_set1_epi32(0x00FF_FFFFu32 as i32))
            };
            let me = _mm512_loadu_si512(MUL_EVEN.as_ptr().cast());
            let mo = _mm512_loadu_si512(MUL_ODD.as_ptr().cast());
            let even = _mm512_mul_epu32(cm, me);
            let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(cm), mo);
            let a0 = _mm512_castsi512_si256(even);
            let b0 = _mm512_extracti64x4_epi64::<1>(even);
            let a1 = _mm512_castsi512_si256(odd);
            let b1 = _mm512_extracti64x4_epi64::<1>(odd);
            let t = _mm256_add_epi64(_mm256_xor_si256(a0, b0), _mm256_xor_si256(a1, b1));
            let v128 = _mm_xor_si128(_mm256_castsi256_si128(t), _mm256_extracti128_si256::<1>(t));
            let x0 = _mm_cvtsi128_si64(v128) as u64;
            let x1 = _mm_extract_epi64::<1>(v128) as u64;
            crate::filtering::scroll::finalize_strip_hash(x0, x1)
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn probe_strip_activity_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> Option<bool> {
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and 16 readable pixels.
        unsafe {
            let c = _mm512_loadu_si512(curr.cast());
            let p = _mm512_loadu_si512(prev.cast());
            let diff = _mm512_xor_si512(c, p);
            if avx512_diff_mask::<HAS_ALPHA>(diff) == 0 {
                return None;
            }
            let first = _mm512_broadcastd_epi32(_mm512_castsi512_si128(c));
            let uniform = _mm512_xor_si512(c, first);
            Some(avx512_diff_mask::<HAS_ALPHA>(uniform) != 0)
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn probe_curr_strip_hash_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> (bool, Option<u64>) {
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and 16 readable pixels.
        unsafe {
            let c = _mm512_loadu_si512(curr.cast());
            let p = _mm512_loadu_si512(prev.cast());
            let diff = _mm512_xor_si512(c, p);
            if avx512_diff_mask::<HAS_ALPHA>(diff) == 0 {
                return (false, None);
            }
            let first = _mm512_broadcastd_epi32(_mm512_castsi512_si128(c));
            let uniform = _mm512_xor_si512(c, first);
            if avx512_diff_mask::<HAS_ALPHA>(uniform) == 0 {
                return (true, None);
            }
            (true, Some(hash_strip_avx512::<HAS_ALPHA>(c)))
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn hash_prev_strip_non_uniform_impl<const HAS_ALPHA: bool>(
        prev: *const Vec4u8,
    ) -> Option<u64> {
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and 16 readable pixels.
        unsafe {
            let p = _mm512_loadu_si512(prev.cast());
            let first = _mm512_broadcastd_epi32(_mm512_castsi512_si128(p));
            let uniform = _mm512_xor_si512(p, first);
            if avx512_diff_mask::<HAS_ALPHA>(uniform) == 0 {
                return None;
            }
            Some(hash_strip_avx512::<HAS_ALPHA>(p))
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn score_strip_row_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev_src: *const Vec4u8,
        prev_dst: *const Vec4u8,
    ) -> i32 {
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm512_set1_epi32(0x00FF_FFFFu32 as i32);
            let c_raw = _mm512_loadu_si512(curr.cast());
            let ps_raw = _mm512_loadu_si512(prev_src.cast());
            let (cm, psm) = if HAS_ALPHA {
                (c_raw, ps_raw)
            } else {
                (
                    _mm512_and_si512(c_raw, rgb_mask),
                    _mm512_and_si512(ps_raw, rgb_mask),
                )
            };
            let eq_ps = _mm512_cmpeq_epi32_mask(cm, psm);
            if eq_ps.count_ones() < 12 {
                return -4;
            }
            let p0_raw = _mm512_loadu_si512(prev_dst.cast());
            let p0m = if HAS_ALPHA {
                p0_raw
            } else {
                _mm512_and_si512(p0_raw, rgb_mask)
            };
            let moved = _mm512_mask_cmpneq_epi32_mask(eq_ps, cm, p0m);
            if moved != 0 { 3 } else { 0 }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn copy_bytes_stream_impl(src: *const u8, dst: *mut u8, len: usize) {
        if len == 0 {
            return;
        }
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and valid non-overlapping ranges.
        unsafe {
            if len < 128 {
                if len <= 64 {
                    let mask = if len == 64 {
                        u64::MAX
                    } else {
                        (1u64 << len) - 1
                    };
                    let v = _mm512_maskz_loadu_epi8(mask, src.cast());
                    _mm512_mask_storeu_epi8(dst.cast(), mask, v);
                } else {
                    let v0 = _mm512_loadu_si512(src.cast());
                    let rem = len - 64;
                    let mask = (1u64 << rem) - 1;
                    let v1 = _mm512_maskz_loadu_epi8(mask, src.add(64).cast());
                    _mm512_storeu_si512(dst.cast(), v0);
                    _mm512_mask_storeu_epi8(dst.add(64).cast(), mask, v1);
                }
                return;
            }
            let head = (64 - (dst as usize & 63)) & 63;
            if head != 0 {
                let mask = (1u64 << head) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, src.cast());
                _mm512_mask_storeu_epi8(dst.cast(), mask, v);
            }
            let s = src.add(head);
            let d = dst.add(head);
            let rem = len - head;
            let lim256 = rem & !255;
            let lim64 = rem & !63;
            let mut i = 0usize;
            while i < lim256 {
                let v0 = _mm512_loadu_si512(s.add(i).cast());
                let v1 = _mm512_loadu_si512(s.add(i + 64).cast());
                let v2 = _mm512_loadu_si512(s.add(i + 128).cast());
                let v3 = _mm512_loadu_si512(s.add(i + 192).cast());
                _mm512_stream_si512(d.add(i).cast::<__m512i>(), v0);
                _mm512_stream_si512(d.add(i + 64).cast::<__m512i>(), v1);
                _mm512_stream_si512(d.add(i + 128).cast::<__m512i>(), v2);
                _mm512_stream_si512(d.add(i + 192).cast::<__m512i>(), v3);
                i += 256;
            }
            while i < lim64 {
                let v = _mm512_loadu_si512(s.add(i).cast());
                _mm512_stream_si512(d.add(i).cast::<__m512i>(), v);
                i += 64;
            }
            let tail = rem - lim64;
            if tail != 0 {
                let mask = (1u64 << tail) - 1;
                let v = _mm512_maskz_loadu_epi8(mask, s.add(i).cast());
                _mm512_mask_storeu_epi8(d.add(i).cast(), mask, v);
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
    fn avx512_diff_mask<const HAS_ALPHA: bool>(diff: __m512i) -> u64 {
        if HAS_ALPHA {
            _mm512_test_epi8_mask(diff, diff)
        } else {
            _mm512_mask_test_epi8_mask(MASK_RGB, diff, diff)
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
        let lim64 = n & !63;
        let lim16 = n & !15;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees AVX-512BW, `src_p` and `dst_p` are valid
        // for `n` `Vec4u8`s and non-overlapping (`src` is shm/caller, `shadow`
        // is exclusively borrowed), and all offsets remain strictly `< n`.
        unsafe {
            let mut i = 0usize;

            // 4-vector OR-screened scan (256 B = 64 px/iter). Keeps `s0..s3` in
            // ZMM registers so a mismatch commits those exact loads without
            // ever re-reading `src_p` (TOCTOU-safe under concurrent `wl_shm`
            // mutation).
            while i < lim64 {
                let s0 = _mm512_loadu_si512(src_p.add(i).cast());
                let x0 = _mm512_xor_si512(s0, _mm512_loadu_si512(dst_p.add(i).cast()));
                let s1 = _mm512_loadu_si512(src_p.add(i + 16).cast());
                let x1 = _mm512_xor_si512(s1, _mm512_loadu_si512(dst_p.add(i + 16).cast()));
                let s2 = _mm512_loadu_si512(src_p.add(i + 32).cast());
                let x2 = _mm512_xor_si512(s2, _mm512_loadu_si512(dst_p.add(i + 32).cast()));
                let s3 = _mm512_loadu_si512(src_p.add(i + 48).cast());
                let x3 = _mm512_xor_si512(s3, _mm512_loadu_si512(dst_p.add(i + 48).cast()));
                let any = _mm512_or_si512(_mm512_or_si512(x0, x1), _mm512_or_si512(x2, x3));
                if avx512_diff_mask::<HAS_ALPHA>(any) != 0 {
                    let m0 = avx512_diff_mask::<HAS_ALPHA>(x0);
                    let m1 = avx512_diff_mask::<HAS_ALPHA>(x1);
                    let m2 = avx512_diff_mask::<HAS_ALPHA>(x2);
                    let m3 = avx512_diff_mask::<HAS_ALPHA>(x3);
                    if m0 != 0 {
                        _mm512_storeu_si512(dst_p.add(i).cast(), s0);
                        if first == usize::MAX {
                            first = i + (m0.trailing_zeros() as usize >> 2);
                        }
                        last_end = i + ((63 - m0.leading_zeros() as usize) >> 2) + 1;
                    }
                    if m1 != 0 {
                        _mm512_storeu_si512(dst_p.add(i + 16).cast(), s1);
                        if first == usize::MAX {
                            first = i + 16 + (m1.trailing_zeros() as usize >> 2);
                        }
                        last_end = i + 16 + ((63 - m1.leading_zeros() as usize) >> 2) + 1;
                    }
                    if m2 != 0 {
                        _mm512_storeu_si512(dst_p.add(i + 32).cast(), s2);
                        if first == usize::MAX {
                            first = i + 32 + (m2.trailing_zeros() as usize >> 2);
                        }
                        last_end = i + 32 + ((63 - m2.leading_zeros() as usize) >> 2) + 1;
                    }
                    if m3 != 0 {
                        _mm512_storeu_si512(dst_p.add(i + 48).cast(), s3);
                        if first == usize::MAX {
                            first = i + 48 + (m3.trailing_zeros() as usize >> 2);
                        }
                        last_end = i + 48 + ((63 - m3.leading_zeros() as usize) >> 2) + 1;
                    }
                }
                i += 64;
            }

            while i < lim16 {
                let s = _mm512_loadu_si512(src_p.add(i).cast());
                let x = _mm512_xor_si512(s, _mm512_loadu_si512(dst_p.add(i).cast()));
                let neq = avx512_diff_mask::<HAS_ALPHA>(x);
                if neq != 0 {
                    _mm512_storeu_si512(dst_p.add(i).cast(), s);
                    if first == usize::MAX {
                        first = i + (neq.trailing_zeros() as usize >> 2);
                    }
                    last_end = i + ((63 - neq.leading_zeros() as usize) >> 2) + 1;
                }
                i += 16;
            }
            for j in lim16..n {
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn validate_palette_indices_impl(n_colors: usize, packed: &[u8]) -> bool {
        let len = packed.len();
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and all loads stay in `0..len`.
        unsafe {
            if n_colors == 3 {
                let lim = (len / 64) * 64;
                let mut acc = _mm512_setzero_si512();
                let mut i = 0usize;
                while i < lim {
                    let v = _mm512_loadu_si512(packed.as_ptr().add(i).cast());
                    acc = _mm512_or_si512(acc, _mm512_and_si512(v, _mm512_srli_epi16::<1>(v)));
                    i += 64;
                }
                if _mm512_test_epi8_mask(acc, _mm512_set1_epi8(0x55)) != 0 {
                    return false;
                }
                validate_palette_indices_scalar(3, &packed[lim..])
            } else if (5..16).contains(&n_colors) {
                let lim = (len / 64) * 64;
                let mask_0f = _mm512_set1_epi8(0x0F);
                let mut max_lo = _mm512_setzero_si512();
                let mut max_hi = _mm512_setzero_si512();
                let mut i = 0usize;
                while i < lim {
                    let v = _mm512_loadu_si512(packed.as_ptr().add(i).cast());
                    max_lo = _mm512_max_epu8(max_lo, _mm512_and_si512(v, mask_0f));
                    max_hi = _mm512_max_epu8(max_hi, _mm512_andnot_si512(mask_0f, v));
                    i += 64;
                }
                let combined = _mm512_max_epu8(max_lo, _mm512_srli_epi16::<4>(max_hi));
                let excess = _mm512_subs_epu8(combined, _mm512_set1_epi8((n_colors - 1) as i8));
                if _mm512_cmpneq_epi8_mask(excess, _mm512_setzero_si512()) != 0 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else if (17..MAX_PALETTE_COLORS).contains(&n_colors) {
                let lim128 = (len / 128) * 128;
                let mut max0 = _mm512_setzero_si512();
                let mut max1 = _mm512_setzero_si512();
                let mut i = 0usize;
                while i < lim128 {
                    max0 = _mm512_max_epu8(max0, _mm512_loadu_si512(packed.as_ptr().add(i).cast()));
                    max1 = _mm512_max_epu8(
                        max1,
                        _mm512_loadu_si512(packed.as_ptr().add(i + 64).cast()),
                    );
                    i += 128;
                }
                if i + 64 <= len {
                    max0 = _mm512_max_epu8(max0, _mm512_loadu_si512(packed.as_ptr().add(i).cast()));
                    i += 64;
                }
                let combined = _mm512_max_epu8(max0, max1);
                let excess = _mm512_subs_epu8(combined, _mm512_set1_epi8((n_colors - 1) as i8));
                if _mm512_cmpneq_epi8_mask(excess, _mm512_setzero_si512()) != 0 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[i..])
            } else {
                true
            }
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn pack_palette_indices_impl(
        indices: &[u8],
        n_colors: usize,
        packed: &mut [u8],
    ) -> usize {
        let n = indices.len();
        let out_len = packed_palette_indices_len(n, n_colors);
        debug_assert!(packed.len() >= out_len);
        if n_colors <= 1 {
            return 0;
        }
        // SAFETY: Caller guarantees AVX-512BW/VL/VBMI and `packed.len() >= out_len`.
        unsafe {
            let src = indices.as_ptr();
            let dst = packed.as_mut_ptr();
            if n_colors <= 2 {
                let lim = (n / 64) * 64;
                let ones = _mm512_set1_epi8(1);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm512_loadu_si512(src.add(i).cast());
                    let mask = _mm512_test_epi8_mask(v, ones);
                    dst.add(oi).cast::<u64>().write_unaligned(mask.to_le());
                    i += 64;
                    oi += 8;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 2, &mut packed[oi..]);
                }
            } else if n_colors <= 4 {
                let lim = (n / 64) * 64;
                let mask3 = _mm512_set1_epi8(3);
                let mul1 = _mm512_set1_epi16(0x0401);
                let mul2 = _mm512_set1_epi32(0x0010_0001);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm512_and_si512(_mm512_loadu_si512(src.add(i).cast()), mask3);
                    let t1 = _mm512_maddubs_epi16(v, mul1);
                    let t2 = _mm512_madd_epi16(t1, mul2);
                    _mm_storeu_si128(dst.add(oi).cast(), _mm512_cvtepi32_epi8(t2));
                    i += 64;
                    oi += 16;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 4, &mut packed[oi..]);
                }
            } else if n_colors <= 16 {
                let lim = (n / 64) * 64;
                let mask_0f = _mm512_set1_epi8(0x0F);
                let mul16 = _mm512_set1_epi16(0x1001);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm512_and_si512(_mm512_loadu_si512(src.add(i).cast()), mask_0f);
                    let t = _mm512_maddubs_epi16(v, mul16);
                    _mm256_storeu_si256(dst.add(oi).cast::<__m256i>(), _mm512_cvtepi16_epi8(t));
                    i += 64;
                    oi += 32;
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

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn decode_palette_1bpp_row_impl<const STREAM: bool>(
        c0: Vec4u8,
        c1: Vec4u8,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(64)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 7) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((8 - (idx_pos & 7)) & 7).min(tw);
            if h != 0 {
                decode_palette_1bpp_scalar(c0, c1, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 7, 0);
            let mut bi = pos >> 3;
            let end = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees AVX-512BW/VL/VBMI, `packed` bounds, and
            // 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let vc0 = _mm512_set1_epi32(u32::from_le_bytes([c0.0, c0.1, c0.2, c0.3]) as i32);
                let vc1 = _mm512_set1_epi32(u32::from_le_bytes([c1.0, c1.1, c1.2, c1.3]) as i32);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    let mask = u16::from_le(src.add(bi).cast::<u16>().read_unaligned());
                    let px = _mm512_mask_blend_epi32(mask, vc0, vc1);
                    let p = out.add(x).cast::<__m512i>();
                    if STREAM {
                        _mm512_stream_si512(p, px);
                    } else {
                        _mm512_storeu_si512(p.cast(), px);
                    }
                    x += 16;
                    bi += 2;
                }
            }
            pos = bi << 3;
        }
        if !STREAM && x < tw {
            decode_palette_1bpp_scalar(c0, c1, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn decode_palette_2bpp_row_impl<const STREAM: bool>(
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(64)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 3) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((4 - (idx_pos & 3)) & 3).min(tw);
            if h != 0 {
                decode_palette_2bpp_scalar(lut, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 3, 0);
            let mut bi = pos >> 2;
            let end = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees AVX-512BW/VL/VBMI, `packed` bounds, and
            // 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                // Replicating `[c0, c1, c2, c3]` 4x across all 16 lanes makes
                // `vpermd` (`idx[3:0]`) wrap modulo 4 with zero `vpand` mask.
                let lut_512 = _mm512_broadcast_i32x4(_mm_loadu_si128(lut.as_ptr().cast()));
                let shifts = _mm512_loadu_si512(SHIFTS_2BPP_512.as_ptr().cast());
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    let w = u32::from_le(src.add(bi).cast::<u32>().read_unaligned());
                    let idx = _mm512_srlv_epi32(_mm512_set1_epi32(w as i32), shifts);
                    let px = _mm512_permutexvar_epi32(idx, lut_512);
                    let p = out.add(x).cast::<__m512i>();
                    if STREAM {
                        _mm512_stream_si512(p, px);
                    } else {
                        _mm512_storeu_si512(p.cast(), px);
                    }
                    x += 16;
                    bi += 4;
                }
            }
            pos = bi << 2;
        }
        if !STREAM && x < tw {
            decode_palette_2bpp_scalar(lut, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn decode_palette_4bpp_row_impl<const STREAM: bool>(
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(64)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 1) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = (idx_pos & 1).min(tw);
            if h != 0 {
                decode_palette_4bpp_scalar(&lut.colors, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 1, 0);
            let mut bi = pos >> 1;
            let end = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees AVX-512BW/VL/VBMI, `packed` bounds, and
            // 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let lut_512 = _mm512_loadu_si512(lut.colors.as_ptr().cast());
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    // `vpermd` uses only bits `[3:0]` of each 32-bit index lane,
                    // so `raw64` and `raw64 >> 4` need no `0x0F` masking.
                    let raw64 = _mm_loadl_epi64(src.add(bi).cast());
                    let hi = _mm_srli_epi16::<4>(raw64);
                    let idx16 = _mm_unpacklo_epi8(raw64, hi);
                    let idx32 = _mm512_cvtepu8_epi32(idx16);
                    let px = _mm512_permutexvar_epi32(idx32, lut_512);
                    let p = out.add(x).cast::<__m512i>();
                    if STREAM {
                        _mm512_stream_si512(p, px);
                    } else {
                        _mm512_storeu_si512(p.cast(), px);
                    }
                    x += 16;
                    bi += 8;
                }
            }
            pos = bi << 1;
        }
        if !STREAM && x < tw {
            decode_palette_4bpp_scalar(&lut.colors, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    unsafe fn decode_palette_8bpp_row_impl<const STREAM: bool>(
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(packed_row.len(), tw);
        debug_assert!(
            !STREAM || ((dst.as_mut_ptr() as usize).is_multiple_of(64) && tw.is_multiple_of(16))
        );
        let lim = if STREAM { tw } else { (tw / 16) * 16 };
        // SAFETY: Caller guarantees AVX-512F, `packed_row.len() == tw`, and
        // 64-byte alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let src = packed_row.as_ptr();
            let out = dst.as_mut_ptr();
            let mut x = 0usize;
            while x < lim {
                let w0 = u32::from_le(src.add(x).cast::<u32>().read_unaligned());
                let w1 = u32::from_le(src.add(x + 4).cast::<u32>().read_unaligned());
                let w2 = u32::from_le(src.add(x + 8).cast::<u32>().read_unaligned());
                let w3 = u32::from_le(src.add(x + 12).cast::<u32>().read_unaligned());
                let v0 = _mm256_setr_epi32(
                    lut[(w0 & 0xFF) as usize] as i32,
                    lut[((w0 >> 8) & 0xFF) as usize] as i32,
                    lut[((w0 >> 16) & 0xFF) as usize] as i32,
                    lut[(w0 >> 24) as usize] as i32,
                    lut[(w1 & 0xFF) as usize] as i32,
                    lut[((w1 >> 8) & 0xFF) as usize] as i32,
                    lut[((w1 >> 16) & 0xFF) as usize] as i32,
                    lut[(w1 >> 24) as usize] as i32,
                );
                let v1 = _mm256_setr_epi32(
                    lut[(w2 & 0xFF) as usize] as i32,
                    lut[((w2 >> 8) & 0xFF) as usize] as i32,
                    lut[((w2 >> 16) & 0xFF) as usize] as i32,
                    lut[(w2 >> 24) as usize] as i32,
                    lut[(w3 & 0xFF) as usize] as i32,
                    lut[((w3 >> 8) & 0xFF) as usize] as i32,
                    lut[((w3 >> 16) & 0xFF) as usize] as i32,
                    lut[(w3 >> 24) as usize] as i32,
                );
                if STREAM {
                    let v = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
                    _mm512_stream_si512(out.add(x).cast::<__m512i>(), v);
                } else {
                    let p = out.add(x).cast::<__m256i>();
                    _mm256_storeu_si256(p, v0);
                    _mm256_storeu_si256(p.add(1), v1);
                }
                x += 16;
            }
        }
        if !STREAM && lim < tw {
            decode_palette_8bpp_scalar(lut, &packed_row[lim..], &mut dst[lim..]);
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
        fn row_sub3_forward_and_sads(self, cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub3_forward_and_sads_impl(cur, dst) }
        }

        #[inline(always)]
        fn row_up_forward_and_sads_all4(
            self,
            cur: &[u8],
            prev: &[u8],
            dst_up: &mut [u8],
        ) -> [u64; 4] {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_up_forward_and_sads_all4_impl(cur, prev, dst_up) }
        }

        #[inline(always)]
        fn row_sub3_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub3_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_sub3_forward_in_place(self, buf: &mut [u8]) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { row_sub3_forward_in_place_impl(buf) }
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
        unsafe fn row_up_subgreen_bgr3_to_bgra<
            const STREAM: bool,
            const CONST_ALPHA: bool,
            const WRITE_DEC: bool,
        >(
            self,
            filt_bgr: &[u8],
            prev_bgr: &[u8],
            cur_dec: &mut [u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                row_up_subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA, WRITE_DEC>(
                    filt_bgr,
                    prev_bgr,
                    cur_dec,
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
        unsafe fn merge_bgrx_alpha_row<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src: &[Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            let tw = dst.len();
            debug_assert_eq!(src.len(), tw);
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds 64-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                merge_bgrx_alpha_row_ptr::<STREAM, CONST_ALPHA>(
                    src.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    dst.as_mut_ptr(),
                    tw,
                );
            }
        }

        #[inline(always)]
        fn merge_bgrx_alpha_row_in_place<const CONST_ALPHA: bool>(
            self,
            pixels: &mut [Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
        ) {
            let tw = pixels.len();
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and `merge_bgrx_alpha_row_ptr` loads each vector before storing back.
            unsafe {
                merge_bgrx_alpha_row_ptr::<false, CONST_ALPHA>(
                    pixels.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    pixels.as_mut_ptr().cast::<MaybeUninit<Vec4u8>>(),
                    tw,
                );
            }
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
        fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { validate_palette_indices_impl(n_colors, packed) }
        }

        #[inline(always)]
        fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support.
            unsafe { pack_palette_indices_impl(indices, n_colors, packed) }
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
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_1bpp_row_impl::<STREAM>(c0, c1, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
            self,
            lut: &[Vec4u8; 4],
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_2bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
            self,
            lut: &PaletteLut4,
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_4bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
            self,
            lut: &[u32; MAX_PALETTE_COLORS],
            packed_row: &[u8],
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx512bw + avx512vl + avx512vbmi` support,
            // and caller upholds `packed_row` and `STREAM` alignment invariants.
            unsafe { decode_palette_8bpp_row_impl::<STREAM>(lut, packed_row, dst) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }

        #[inline(always)]
        unsafe fn probe_strip_activity<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> Option<bool> {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller guarantees 16 readable pixels.
            unsafe { probe_strip_activity_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn probe_curr_strip_hash<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> (bool, Option<u64>) {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller guarantees 16 readable pixels.
            unsafe { probe_curr_strip_hash_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn hash_prev_strip_non_uniform<const HAS_ALPHA: bool>(
            self,
            prev: *const Vec4u8,
        ) -> Option<u64> {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller guarantees 16 readable pixels.
            unsafe { hash_prev_strip_non_uniform_impl::<HAS_ALPHA>(prev) }
        }

        #[inline(always)]
        unsafe fn score_strip_row<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev_src: *const Vec4u8,
            prev_dst: *const Vec4u8,
        ) -> i32 {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller guarantees 16 readable pixels.
            unsafe { score_strip_row_impl::<HAS_ALPHA>(curr, prev_src, prev_dst) }
        }

        #[inline(always)]
        unsafe fn copy_bytes_stream(self, src: *const u8, dst: *mut u8, len: usize) {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller guarantees non-overlapping ranges.
            unsafe { copy_bytes_stream_impl(src, dst, len) }
        }

        #[inline(always)]
        unsafe fn detect_scroll_phase1b<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            n_strips: usize,
            tier0_step_y: usize,
            changed_rows: &[u16; crate::filtering::scroll::MAX_STRIPS],
            changed_non_uniform: &[u16; crate::filtering::scroll::MAX_STRIPS],
            min_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
            max_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller upholds band bounds invariants.
            unsafe {
                detect_scroll_phase1b_impl::<HAS_ALPHA>(
                    self,
                    aos,
                    prev,
                    width,
                    stride_px,
                    band_y0,
                    band_y1,
                    n_strips,
                    tier0_step_y,
                    changed_rows,
                    changed_non_uniform,
                    min_y,
                    max_y,
                )
            }
        }

        #[inline(always)]
        unsafe fn expand_and_verify_scroll_candidate<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            seed_sx: usize,
            dy: i32,
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses AVX-512BW/VL/VBMI; caller upholds band bounds invariants.
            unsafe {
                expand_and_verify_scroll_candidate_impl::<HAS_ALPHA>(
                    self, aos, prev, width, stride_px, band_y0, band_y1, seed_sx, dy,
                )
            }
        }

        #[inline(always)]
        unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
        ) {
            // SAFETY: `self` witnesses this tier's SIMD support, and caller upholds
            // disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_row_band_2d_impl::<HAS_ALPHA>(self, ctx);
            }
        }

        #[inline(always)]
        unsafe fn finish_active_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
            qx_bboxes: &mut [QuadBbox],
            trigger_y: usize,
            full_count: usize,
        ) {
            if let Some(scans) = ctx.slot_scans
                && tile_hash::has_hw_aes()
            {
                // SAFETY: `self` witnesses this tier's SIMD support, `has_hw_aes()`
                // verified `aes`, and caller upholds disjoint-row shadow bounds.
                unsafe {
                    diff_and_update_active_band_aes::<HAS_ALPHA>(
                        ctx, scans, qx_bboxes, trigger_y, full_count,
                    );
                }
                return;
            }
            // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_active_band_fallback::<HAS_ALPHA>(
                    self, ctx, qx_bboxes, trigger_y, full_count,
                );
            }
        }
    }

    impl_finish_active_row_band_aes!(
        "avx512bw,avx512vl,avx512vbmi",
        "avx512bw,avx512vl,avx512vbmi,aes"
    );
}

#[cfg(not(all(
    target_feature = "avx512bw",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi",
)))]
mod avx2 {
    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_mm_abs_epi8;
    use std::arch::x86_64::_mm_add_epi8;
    use std::arch::x86_64::_mm_add_epi64;
    use std::arch::x86_64::_mm_alignr_epi8;
    use std::arch::x86_64::_mm_blend_epi32;
    use std::arch::x86_64::_mm_cvtsi32_si128;
    use std::arch::x86_64::_mm_cvtsi128_si32;
    use std::arch::x86_64::_mm_cvtsi128_si64;
    use std::arch::x86_64::_mm_extract_epi64;
    use std::arch::x86_64::_mm_loadl_epi64;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_or_si128;
    use std::arch::x86_64::_mm_sad_epu8;
    use std::arch::x86_64::_mm_set_epi8;
    use std::arch::x86_64::_mm_set1_epi8;
    use std::arch::x86_64::_mm_setr_epi8;
    use std::arch::x86_64::_mm_setzero_si128;
    use std::arch::x86_64::_mm_sfence;
    use std::arch::x86_64::_mm_shuffle_epi8;
    use std::arch::x86_64::_mm_slli_si128;
    use std::arch::x86_64::_mm_srli_si128;
    use std::arch::x86_64::_mm_storel_epi64;
    use std::arch::x86_64::_mm_storeu_si128;
    use std::arch::x86_64::_mm_stream_si128;
    use std::arch::x86_64::_mm_sub_epi8;
    use std::arch::x86_64::_mm_unpacklo_epi32;
    use std::arch::x86_64::_mm_unpacklo_epi64;
    use std::arch::x86_64::_mm_xor_si128;
    use std::arch::x86_64::_mm256_abs_epi8;
    use std::arch::x86_64::_mm256_add_epi8;
    use std::arch::x86_64::_mm256_add_epi64;
    use std::arch::x86_64::_mm256_alignr_epi8;
    use std::arch::x86_64::_mm256_and_si256;
    use std::arch::x86_64::_mm256_andnot_si256;
    use std::arch::x86_64::_mm256_blendv_ps;
    use std::arch::x86_64::_mm256_broadcastd_epi32;
    use std::arch::x86_64::_mm256_broadcastsi128_si256;
    use std::arch::x86_64::_mm256_castps_si256;
    use std::arch::x86_64::_mm256_castsi256_ps;
    use std::arch::x86_64::_mm256_castsi256_si128;
    use std::arch::x86_64::_mm256_cmpeq_epi8;
    use std::arch::x86_64::_mm256_cmpeq_epi32;
    use std::arch::x86_64::_mm256_cvtepu8_epi32;
    use std::arch::x86_64::_mm256_extracti128_si256;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_madd_epi16;
    use std::arch::x86_64::_mm256_maddubs_epi16;
    use std::arch::x86_64::_mm256_max_epu8;
    use std::arch::x86_64::_mm256_movemask_epi8;
    use std::arch::x86_64::_mm256_movemask_ps;
    use std::arch::x86_64::_mm256_mul_epu32;
    use std::arch::x86_64::_mm256_or_si256;
    use std::arch::x86_64::_mm256_packus_epi16;
    use std::arch::x86_64::_mm256_permute2x128_si256;
    use std::arch::x86_64::_mm256_permute4x64_epi64;
    use std::arch::x86_64::_mm256_permutevar8x32_epi32;
    use std::arch::x86_64::_mm256_sad_epu8;
    use std::arch::x86_64::_mm256_set_epi8;
    use std::arch::x86_64::_mm256_set_m128i;
    use std::arch::x86_64::_mm256_set1_epi8;
    use std::arch::x86_64::_mm256_set1_epi16;
    use std::arch::x86_64::_mm256_set1_epi32;
    use std::arch::x86_64::_mm256_set1_epi64x;
    use std::arch::x86_64::_mm256_setr_epi8;
    use std::arch::x86_64::_mm256_setr_epi32;
    use std::arch::x86_64::_mm256_setzero_si256;
    use std::arch::x86_64::_mm256_shuffle_epi8;
    use std::arch::x86_64::_mm256_slli_epi16;
    use std::arch::x86_64::_mm256_slli_epi32;
    use std::arch::x86_64::_mm256_slli_si256;
    use std::arch::x86_64::_mm256_sllv_epi32;
    use std::arch::x86_64::_mm256_srli_epi16;
    use std::arch::x86_64::_mm256_srli_epi64;
    use std::arch::x86_64::_mm256_srlv_epi32;
    use std::arch::x86_64::_mm256_storeu_si256;
    use std::arch::x86_64::_mm256_stream_si256;
    use std::arch::x86_64::_mm256_sub_epi8;
    use std::arch::x86_64::_mm256_subs_epu8;
    use std::arch::x86_64::_mm256_testz_si256;
    use std::arch::x86_64::_mm256_unpackhi_epi8;
    use std::arch::x86_64::_mm256_unpackhi_epi16;
    use std::arch::x86_64::_mm256_unpackhi_epi64;
    use std::arch::x86_64::_mm256_unpacklo_epi8;
    use std::arch::x86_64::_mm256_unpacklo_epi16;
    use std::arch::x86_64::_mm256_unpacklo_epi64;
    use std::arch::x86_64::_mm256_xor_si256;
    use std::arch::x86_64::_mm256_zextsi128_si256;
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::ActiveRowBandCtx;
    use crate::filtering::QuadBbox;
    use crate::filtering::RowKernel;
    use crate::filtering::palette::MAX_PALETTE_COLORS;
    use crate::filtering::palette::PaletteLut4;
    use crate::filtering::palette::decode_palette_1bpp_scalar;
    use crate::filtering::palette::decode_palette_2bpp_scalar;
    use crate::filtering::palette::decode_palette_4bpp_scalar;
    use crate::filtering::palette::decode_palette_8bpp_scalar;
    use crate::filtering::palette::pack_palette_indices_scalar;
    use crate::filtering::palette::validate_palette_indices_scalar;
    use crate::filtering::tile_hash;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
    use crate::serialization::framing::packed_palette_indices_len;
    use crate::vec4u8::Vec4u8;

    #[target_feature(enable = "avx2")]
    #[inline]
    fn hsum2_epi64(a0: __m256i, a1: __m256i) -> (u64, u64) {
        let s01 = _mm256_add_epi64(_mm256_unpacklo_epi64(a0, a1), _mm256_unpackhi_epi64(a0, a1));
        let v01 = _mm_add_epi64(
            _mm256_castsi256_si128(s01),
            _mm256_extracti128_si256::<1>(s01),
        );
        (
            _mm_cvtsi128_si64(v01) as u64,
            _mm_extract_epi64::<1>(v01) as u64,
        )
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn hsum4_epi64(a0: __m256i, a1: __m256i, a2: __m256i, a3: __m256i) -> [u64; 4] {
        let (s0, s1) = hsum2_epi64(a0, a1);
        let (s2, s3) = hsum2_epi64(a2, a3);
        [s0, s1, s2, s3]
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

        // Main 8-pixel loop writes overlapping 16 B at `di` and `di + 12` (fast
        // single-instruction `vextracti128` memory store, safe for `di + 28 <= tw * 3`,
        // i.e. `x < lim`). Any trailing full 8-pixel block (`x < full8`, e.g. the
        // final 8 px of `tw = 8, 24, 40, 72, 120, 256`) uses strictly in-bounds
        // non-overlapping 16 B + 8 B stores, followed by an optional 4-pixel SSE
        // block (`full8 < full4`) so at most `0..=3` scalar tail pixels remain.
        let full8 = tw & !7;
        let full4 = tw & !3;
        let lim = if tw >= 10 { ((tw - 2) / 8) * 8 } else { 0 };
        let mut x = 0usize;
        let mut di = 0usize;

        // SAFETY: Caller guarantees AVX2; `di + 28 <= dst_bgr.len()` for `x < lim`,
        // `di + 24 <= dst_bgr.len()` for `x < full8`, `di + 12 <= dst_bgr.len()`
        // for `full8 < full4`, `di + 3 <= dst_bgr.len()` in the `tw & 3` tail,
        // and `dst_alpha.len() == tw` when `HAS_ALPHA`.
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
            let dst_p = dst_bgr.as_mut_ptr();
            let alpha_p = dst_alpha.as_mut_ptr();

            while x < lim {
                let v = _mm256_loadu_si256(src.ptr().add(x).cast::<__m256i>());
                let g = _mm256_shuffle_epi8(v, shuf_g);
                let sg = _mm256_sub_epi8(v, g);
                let pk = _mm256_shuffle_epi8(sg, pack12);
                _mm_storeu_si128(dst_p.add(di).cast::<__m128i>(), _mm256_castsi256_si128(pk));
                _mm_storeu_si128(
                    dst_p.add(di + 12).cast::<__m128i>(),
                    _mm256_extracti128_si256::<1>(pk),
                );
                if HAS_ALPHA {
                    let a_pk = _mm256_shuffle_epi8(v, shuf_a);
                    let a8 = _mm_unpacklo_epi32(
                        _mm256_castsi256_si128(a_pk),
                        _mm256_extracti128_si256::<1>(a_pk),
                    );
                    _mm_storel_epi64(alpha_p.add(x).cast::<__m128i>(), a8);
                }
                x += 8;
                di += 24;
            }
            if x < full8 {
                let v = _mm256_loadu_si256(src.ptr().add(x).cast::<__m256i>());
                let g = _mm256_shuffle_epi8(v, shuf_g);
                let sg = _mm256_sub_epi8(v, g);
                let pk = _mm256_shuffle_epi8(sg, pack12);
                let lo = _mm256_castsi256_si128(pk);
                let hi = _mm256_extracti128_si256::<1>(pk);
                let b0_15 = _mm_or_si128(lo, _mm_slli_si128::<12>(hi));
                let b16_23 = _mm_srli_si128::<4>(hi);
                _mm_storeu_si128(dst_p.add(di).cast::<__m128i>(), b0_15);
                _mm_storel_epi64(dst_p.add(di + 16).cast::<__m128i>(), b16_23);
                if HAS_ALPHA {
                    let a_pk = _mm256_shuffle_epi8(v, shuf_a);
                    let a8 = _mm_unpacklo_epi32(
                        _mm256_castsi256_si128(a_pk),
                        _mm256_extracti128_si256::<1>(a_pk),
                    );
                    _mm_storel_epi64(alpha_p.add(x).cast::<__m128i>(), a8);
                }
                x += 8;
                di += 24;
            }
            if full8 < full4 {
                let v = _mm_loadu_si128(src.ptr().add(x).cast::<__m128i>());
                let g = _mm_shuffle_epi8(v, _mm256_castsi256_si128(shuf_g));
                let sg = _mm_sub_epi8(v, g);
                let pk = _mm_shuffle_epi8(sg, _mm256_castsi256_si128(pack12));
                _mm_storel_epi64(dst_p.add(di).cast::<__m128i>(), pk);
                let last4 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(pk)) as u32;
                dst_p
                    .add(di + 8)
                    .cast::<u32>()
                    .write_unaligned(last4.to_le());
                if HAS_ALPHA {
                    let a4 = _mm_cvtsi128_si32(_mm_shuffle_epi8(v, _mm256_castsi256_si128(shuf_a)))
                        as u32;
                    alpha_p.add(x).cast::<u32>().write_unaligned(a4.to_le());
                }
                x += 4;
                di += 12;
            }
            for _ in 0..(tw & 3) {
                let px = src.ptr().add(x).read();
                let out = dst_p.add(di);
                out.write(px.0.wrapping_sub(px.1));
                out.add(1).write(px.1);
                out.add(2).write(px.2.wrapping_sub(px.1));
                if HAS_ALPHA {
                    alpha_p.add(x).write(px.3);
                }
                x += 1;
                di += 3;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_forward_and_sads_impl(cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
        let len = cur.len();
        debug_assert_eq!(dst.len(), len);
        let lim32 = (len / 32) * 32;
        let mut i = 0usize;
        // SAFETY: Caller guarantees AVX2, `i + 32 <= len` in main loop,
        // `i + 16 <= len` in 16-byte tail, and `j < len` in scalar tail.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut acc_none = zero;
            let mut acc_sub = zero;
            let mut prev = zero;
            let cur_p = cur.as_ptr();
            let dst_p = dst.as_mut_ptr();
            while i < lim32 {
                let c = _mm256_loadu_si256(cur_p.add(i).cast::<__m256i>());
                let sub = _mm256_sub_epi8(
                    c,
                    _mm256_alignr_epi8::<13>(c, _mm256_permute2x128_si256::<0x21>(prev, c)),
                );
                prev = c;
                _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), sub);
                acc_none = _mm256_add_epi64(acc_none, _mm256_sad_epu8(_mm256_abs_epi8(c), zero));
                acc_sub = _mm256_add_epi64(acc_sub, _mm256_sad_epu8(_mm256_abs_epi8(sub), zero));
                i += 32;
            }
            if i + 16 <= len {
                let z128 = _mm_setzero_si128();
                let c128 = _mm_loadu_si128(cur_p.add(i).cast::<__m128i>());
                let prev128 = _mm256_extracti128_si256::<1>(prev);
                let sub128 = _mm_sub_epi8(c128, _mm_alignr_epi8::<13>(c128, prev128));
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), sub128);
                acc_none = _mm256_add_epi64(
                    acc_none,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(c128), z128)),
                );
                acc_sub = _mm256_add_epi64(
                    acc_sub,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(sub128), z128)),
                );
                i += 16;
            }
            let (mut s_none, mut s_sub) = hsum2_epi64(acc_none, acc_sub);
            for j in i..len {
                let c = *cur_p.add(j);
                let p = if j >= 3 { *cur_p.add(j - 3) } else { 0 };
                let sub = c.wrapping_sub(p);
                *dst_p.add(j) = sub;
                s_none += u64::from((c as i8).unsigned_abs());
                s_sub += u64::from((sub as i8).unsigned_abs());
            }
            (s_none, s_sub)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_up_forward_and_sads_all4_impl(
        cur: &[u8],
        prev: &[u8],
        dst_up: &mut [u8],
    ) -> [u64; 4] {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst_up.len(), len);
        let lim32 = (len / 32) * 32;
        let mut i = 0usize;
        // SAFETY: Caller guarantees AVX2, `i + 32 <= len` in main loop,
        // `i + 16 <= len` in 16-byte tail, and `j < len` in scalar tail.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            let mut prev_c = zero;
            let mut prev_up = zero;
            let cur_p = cur.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst_up.as_mut_ptr();
            while i < lim32 {
                let c = _mm256_loadu_si256(cur_p.add(i).cast::<__m256i>());
                let u = _mm256_loadu_si256(prev_p.add(i).cast::<__m256i>());
                let up = _mm256_sub_epi8(c, u);
                _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), up);
                let sub = _mm256_sub_epi8(
                    c,
                    _mm256_alignr_epi8::<13>(c, _mm256_permute2x128_si256::<0x21>(prev_c, c)),
                );
                let su = _mm256_sub_epi8(
                    up,
                    _mm256_alignr_epi8::<13>(up, _mm256_permute2x128_si256::<0x21>(prev_up, up)),
                );
                prev_c = c;
                prev_up = up;

                acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(_mm256_abs_epi8(c), zero));
                acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(_mm256_abs_epi8(sub), zero));
                acc2 = _mm256_add_epi64(acc2, _mm256_sad_epu8(_mm256_abs_epi8(up), zero));
                acc3 = _mm256_add_epi64(acc3, _mm256_sad_epu8(_mm256_abs_epi8(su), zero));
                i += 32;
            }
            if i + 16 <= len {
                let z128 = _mm_setzero_si128();
                let c128 = _mm_loadu_si128(cur_p.add(i).cast::<__m128i>());
                let u128 = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                let up128 = _mm_sub_epi8(c128, u128);
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), up128);
                let sub128 = _mm_sub_epi8(
                    c128,
                    _mm_alignr_epi8::<13>(c128, _mm256_extracti128_si256::<1>(prev_c)),
                );
                let su128 = _mm_sub_epi8(
                    up128,
                    _mm_alignr_epi8::<13>(up128, _mm256_extracti128_si256::<1>(prev_up)),
                );
                acc0 = _mm256_add_epi64(
                    acc0,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(c128), z128)),
                );
                acc1 = _mm256_add_epi64(
                    acc1,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(sub128), z128)),
                );
                acc2 = _mm256_add_epi64(
                    acc2,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(up128), z128)),
                );
                acc3 = _mm256_add_epi64(
                    acc3,
                    _mm256_zextsi128_si256(_mm_sad_epu8(_mm_abs_epi8(su128), z128)),
                );
                i += 16;
            }
            let mut sums = hsum4_epi64(acc0, acc1, acc2, acc3);
            for j in i..len {
                let c = *cur_p.add(j);
                let up = c.wrapping_sub(*prev_p.add(j));
                *dst_p.add(j) = up;
                let (pc, pu) = if j >= 3 {
                    let cp = *cur_p.add(j - 3);
                    (cp, cp.wrapping_sub(*prev_p.add(j - 3)))
                } else {
                    (0, 0)
                };
                let sub = c.wrapping_sub(pc);
                let su = up.wrapping_sub(pu);
                sums[0] += u64::from((c as i8).unsigned_abs());
                sums[1] += u64::from((sub as i8).unsigned_abs());
                sums[2] += u64::from((up as i8).unsigned_abs());
                sums[3] += u64::from((su as i8).unsigned_abs());
            }
            sums
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim32 = (len / 32) * 32;
        let mut i = 0usize;
        // SAFETY: Caller guarantees AVX2; `i + 32 <= len` in main loop,
        // `i - 3 >= 29` for `i >= 32`, `i + 16 <= len` in 16-byte tail, and
        // `j < len` in scalar tail.
        unsafe {
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            if lim32 >= 32 {
                let v0 = _mm256_loadu_si256(src_p.cast::<__m256i>());
                let prev_lane = _mm256_permute2x128_si256::<0x21>(_mm256_setzero_si256(), v0);
                let sh0 = _mm256_alignr_epi8::<13>(v0, prev_lane);
                _mm256_storeu_si256(dst_p.cast::<__m256i>(), _mm256_sub_epi8(v0, sh0));
                i = 32;
                while i < lim32 {
                    let v = _mm256_loadu_si256(src_p.add(i).cast::<__m256i>());
                    let sh = _mm256_loadu_si256(src_p.add(i - 3).cast::<__m256i>());
                    _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), _mm256_sub_epi8(v, sh));
                    i += 32;
                }
            }
            if i + 16 <= len {
                let v = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let sh = if i >= 3 {
                    _mm_loadu_si128(src_p.add(i - 3).cast::<__m128i>())
                } else {
                    _mm_slli_si128::<3>(v)
                };
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_sub_epi8(v, sh));
                i += 16;
            }
            for j in i..len {
                let p = if j >= 3 { *src_p.add(j - 3) } else { 0 };
                *dst_p.add(j) = (*src_p.add(j)).wrapping_sub(p);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_forward_in_place_impl(buf: &mut [u8]) {
        let len = buf.len();
        let mut i = len;
        // SAFETY: Caller guarantees AVX2; backwards 32-byte and 16-byte steps
        // write `i..i + step` strictly above all subsequent reads `..i` with
        // `i >= 3`, and `0..32` stays in bounds when `i >= 32` (`i <= 34`).
        unsafe {
            let p = buf.as_mut_ptr();
            while i >= 35 {
                i -= 32;
                let v = _mm256_loadu_si256(p.add(i).cast::<__m256i>());
                let sh = _mm256_loadu_si256(p.add(i - 3).cast::<__m256i>());
                _mm256_storeu_si256(p.add(i).cast::<__m256i>(), _mm256_sub_epi8(v, sh));
            }
            if i >= 32 {
                // `i` is in `32..=34`; handle the `0..=2` bytes above `32` first.
                if i == 34 {
                    *p.add(33) = (*p.add(33)).wrapping_sub(*p.add(30));
                }
                if i >= 33 {
                    *p.add(32) = (*p.add(32)).wrapping_sub(*p.add(29));
                }
                let v0 = _mm256_loadu_si256(p.cast::<__m256i>());
                let prev_lane = _mm256_permute2x128_si256::<0x21>(_mm256_setzero_si256(), v0);
                let sh0 = _mm256_alignr_epi8::<13>(v0, prev_lane);
                _mm256_storeu_si256(p.cast::<__m256i>(), _mm256_sub_epi8(v0, sh0));
                return;
            }
            if i >= 19 {
                i -= 16;
                let v = _mm_loadu_si128(p.add(i).cast::<__m128i>());
                let sh = _mm_loadu_si128(p.add(i - 3).cast::<__m128i>());
                _mm_storeu_si128(p.add(i).cast::<__m128i>(), _mm_sub_epi8(v, sh));
            }
            for j in (3..i).rev() {
                *p.add(j) = (*p.add(j)).wrapping_sub(*p.add(j - 3));
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_12(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees AVX2 and `p` is readable for at least 12 bytes.
        unsafe {
            let lo8 = _mm_loadl_epi64(p.cast::<__m128i>());
            let hi4 = _mm_cvtsi32_si128(p.add(8).cast::<u32>().read_unaligned() as i32);
            _mm_unpacklo_epi64(lo8, hi4)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_12(p: *mut u8, v: __m128i) {
        // SAFETY: Caller guarantees AVX2 and `p` is writable for at least 12 bytes.
        unsafe {
            _mm_storel_epi64(p.cast::<__m128i>(), v);
            let hi4 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(v)) as u32;
            p.add(8).cast::<u32>().write_unaligned(hi4.to_le());
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_24_lanes(p: *const u8) -> (__m128i, __m128i) {
        // Non-overlapping 16 B + 8 B loads matching `store_24_exact` give
        // exact store-to-load forwarding across consecutive rows (no Zen 3
        // split-STLF replays) and never read past `p + 24`.
        // SAFETY: Caller guarantees AVX2 and `p` is readable for at least 24 bytes.
        unsafe {
            let b0_15 = _mm_loadu_si128(p.cast::<__m128i>());
            let b16_23 = _mm_loadl_epi64(p.add(16).cast::<__m128i>());
            (b0_15, _mm_alignr_epi8::<12>(b16_23, b0_15))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_24_exact(p: *mut u8, lo: __m128i, hi: __m128i) {
        // Non-overlapping 16 B (`lo[0..12] || hi[0..4]`) + 8 B (`hi[4..12]`)
        // stores match `load_24_lanes` for exact store-to-load forwarding and
        // never write past `p + 24`.
        // SAFETY: Caller guarantees AVX2 and `p` is writable for at least 24 bytes.
        unsafe {
            let b0_15 = _mm_blend_epi32::<0x8>(lo, _mm_slli_si128::<12>(hi));
            let b16_23 = _mm_srli_si128::<4>(hi);
            _mm_storeu_si128(p.cast::<__m128i>(), b0_15);
            _mm_storel_epi64(p.add(16).cast::<__m128i>(), b16_23);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        // Four parallel 128-bit lanes (48 B = 16 px) compute independent
        // 4-pixel prefix sums with a single 1-cycle carry recurrence per 48 B
        // and zero cross-lane `vinserti128`/`vextracti128` latency, while
        // keeping overlapping 16 B loads/stores strictly in-bounds
        // (`i + 52 <= len`), matching `subgreen_bgr3_to_bgra`'s 16-px / 8-px
        // load geometry for clean Zen 3 store-to-load forwarding.
        let lim48 = if len >= 52 { ((len - 4) / 48) * 48 } else { 0 };
        // SAFETY: Caller guarantees AVX2; `i + 52 <= len` in main loop,
        // `i + 28 <= len` in 24 B overlapping step, `i + 24 <= len` in exact
        // 24 B tail, `i + 12 <= len` in 12 B tail, and `i..len` in the
        // `0..11` B remainder.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut carry = _mm_setzero_si128();
            let mut i = 0usize;
            while i < lim48 {
                let mut p0 = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let mut p1 = _mm_loadu_si128(src_p.add(i + 12).cast::<__m128i>());
                let mut p2 = _mm_loadu_si128(src_p.add(i + 24).cast::<__m128i>());
                let mut p3 = _mm_loadu_si128(src_p.add(i + 36).cast::<__m128i>());
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p2 = _mm_add_epi8(p2, _mm_slli_si128::<3>(p2));
                p3 = _mm_add_epi8(p3, _mm_slli_si128::<3>(p3));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                p2 = _mm_add_epi8(p2, _mm_slli_si128::<6>(p2));
                p3 = _mm_add_epi8(p3, _mm_slli_si128::<6>(p3));
                let s01 = _mm_add_epi8(p0, p1);
                let s23 = _mm_add_epi8(p2, p3);
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let b01 = _mm_shuffle_epi8(s01, shuf_last3);
                let b012 = _mm_shuffle_epi8(_mm_add_epi8(s01, p2), shuf_last3);
                let b0123 = _mm_shuffle_epi8(_mm_add_epi8(s01, s23), shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                let c2 = _mm_add_epi8(carry, b01);
                let c3 = _mm_add_epi8(carry, b012);
                carry = _mm_add_epi8(carry, b0123);
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_add_epi8(p0, c0));
                _mm_storeu_si128(dst_p.add(i + 12).cast::<__m128i>(), _mm_add_epi8(p1, c1));
                _mm_storeu_si128(dst_p.add(i + 24).cast::<__m128i>(), _mm_add_epi8(p2, c2));
                _mm_storeu_si128(dst_p.add(i + 36).cast::<__m128i>(), _mm_add_epi8(p3, c3));
                i += 48;
            }
            if i + 28 <= len {
                let mut p0 = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let mut p1 = _mm_loadu_si128(src_p.add(i + 12).cast::<__m128i>());
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                carry = _mm_add_epi8(carry, _mm_shuffle_epi8(_mm_add_epi8(p0, p1), shuf_last3));
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_add_epi8(p0, c0));
                _mm_storeu_si128(dst_p.add(i + 12).cast::<__m128i>(), _mm_add_epi8(p1, c1));
                i += 24;
            }
            if i + 24 <= len {
                let (mut p0, mut p1) = load_24_lanes(src_p.add(i));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                carry = _mm_add_epi8(carry, _mm_shuffle_epi8(_mm_add_epi8(p0, p1), shuf_last3));
                store_24_exact(dst_p.add(i), _mm_add_epi8(p0, c0), _mm_add_epi8(p1, c1));
                i += 24;
            }
            if i + 12 <= len {
                let mut v = load_12(src_p.add(i));
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, carry);
                carry = _mm_shuffle_epi8(v, shuf_last3);
                store_12(dst_p.add(i), v);
                i += 12;
            }
            if i < len {
                let c32 = _mm_cvtsi128_si32(carry) as u32;
                let mut acc = [c32 as u8, (c32 >> 8) as u8, (c32 >> 16) as u8];
                while i + 3 <= len {
                    acc[0] = acc[0].wrapping_add(*src_p.add(i));
                    *dst_p.add(i) = acc[0];
                    acc[1] = acc[1].wrapping_add(*src_p.add(i + 1));
                    *dst_p.add(i + 1) = acc[1];
                    acc[2] = acc[2].wrapping_add(*src_p.add(i + 2));
                    *dst_p.add(i + 2) = acc[2];
                    i += 3;
                }
                for (ch, slot) in acc.iter_mut().enumerate().take(len - i) {
                    *slot = slot.wrapping_add(*src_p.add(i + ch));
                    *dst_p.add(i + ch) = *slot;
                }
            }
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
        let lim48 = if len >= 52 { ((len - 4) / 48) * 48 } else { 0 };
        // SAFETY: Caller guarantees AVX2; `i + 52 <= len` in main loop,
        // `i + 28 <= len` in 24 B overlapping step, `i + 24 <= len` in exact
        // 24 B tail, `i + 12 <= len` in 12 B tail, and `i + 3 <= len` in the
        // `0..3`-pixel scalar remainder.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
            let src_p = src.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut carry = _mm_setzero_si128();
            let mut i = 0usize;
            while i < lim48 {
                let mut p0 = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let mut p1 = _mm_loadu_si128(src_p.add(i + 12).cast::<__m128i>());
                let mut p2 = _mm_loadu_si128(src_p.add(i + 24).cast::<__m128i>());
                let mut p3 = _mm_loadu_si128(src_p.add(i + 36).cast::<__m128i>());
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p2 = _mm_add_epi8(p2, _mm_slli_si128::<3>(p2));
                p3 = _mm_add_epi8(p3, _mm_slli_si128::<3>(p3));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                p2 = _mm_add_epi8(p2, _mm_slli_si128::<6>(p2));
                p3 = _mm_add_epi8(p3, _mm_slli_si128::<6>(p3));
                let s01 = _mm_add_epi8(p0, p1);
                let s23 = _mm_add_epi8(p2, p3);
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let b01 = _mm_shuffle_epi8(s01, shuf_last3);
                let b012 = _mm_shuffle_epi8(_mm_add_epi8(s01, p2), shuf_last3);
                let b0123 = _mm_shuffle_epi8(_mm_add_epi8(s01, s23), shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                let c2 = _mm_add_epi8(carry, b01);
                let c3 = _mm_add_epi8(carry, b012);
                carry = _mm_add_epi8(carry, b0123);
                let u0 = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                let u1 = _mm_loadu_si128(prev_p.add(i + 12).cast::<__m128i>());
                let u2 = _mm_loadu_si128(prev_p.add(i + 24).cast::<__m128i>());
                let u3 = _mm_loadu_si128(prev_p.add(i + 36).cast::<__m128i>());
                _mm_storeu_si128(
                    dst_p.add(i).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p0, u0), c0),
                );
                _mm_storeu_si128(
                    dst_p.add(i + 12).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p1, u1), c1),
                );
                _mm_storeu_si128(
                    dst_p.add(i + 24).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p2, u2), c2),
                );
                _mm_storeu_si128(
                    dst_p.add(i + 36).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p3, u3), c3),
                );
                i += 48;
            }
            if i + 28 <= len {
                let mut p0 = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let mut p1 = _mm_loadu_si128(src_p.add(i + 12).cast::<__m128i>());
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                carry = _mm_add_epi8(carry, _mm_shuffle_epi8(_mm_add_epi8(p0, p1), shuf_last3));
                let u0 = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                let u1 = _mm_loadu_si128(prev_p.add(i + 12).cast::<__m128i>());
                _mm_storeu_si128(
                    dst_p.add(i).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p0, u0), c0),
                );
                _mm_storeu_si128(
                    dst_p.add(i + 12).cast::<__m128i>(),
                    _mm_add_epi8(_mm_add_epi8(p1, u1), c1),
                );
                i += 24;
            }
            if i + 24 <= len {
                let (mut p0, mut p1) = load_24_lanes(src_p.add(i));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<3>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<3>(p1));
                p0 = _mm_add_epi8(p0, _mm_slli_si128::<6>(p0));
                p1 = _mm_add_epi8(p1, _mm_slli_si128::<6>(p1));
                let b0 = _mm_shuffle_epi8(p0, shuf_last3);
                let c0 = carry;
                let c1 = _mm_add_epi8(carry, b0);
                carry = _mm_add_epi8(carry, _mm_shuffle_epi8(_mm_add_epi8(p0, p1), shuf_last3));
                let (u0, u1) = load_24_lanes(prev_p.add(i));
                store_24_exact(
                    dst_p.add(i),
                    _mm_add_epi8(_mm_add_epi8(p0, u0), c0),
                    _mm_add_epi8(_mm_add_epi8(p1, u1), c1),
                );
                i += 24;
            }
            if i + 12 <= len {
                let mut v = load_12(src_p.add(i));
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, carry);
                carry = _mm_shuffle_epi8(v, shuf_last3);
                let u = load_12(prev_p.add(i));
                store_12(dst_p.add(i), _mm_add_epi8(v, u));
                i += 12;
            }
            if i < len {
                let c32 = _mm_cvtsi128_si32(carry) as u32;
                let mut acc = [c32 as u8, (c32 >> 8) as u8, (c32 >> 16) as u8];
                while i + 3 <= len {
                    acc[0] = acc[0].wrapping_add(*src_p.add(i));
                    *dst_p.add(i) = acc[0].wrapping_add(*prev_p.add(i));
                    acc[1] = acc[1].wrapping_add(*src_p.add(i + 1));
                    *dst_p.add(i + 1) = acc[1].wrapping_add(*prev_p.add(i + 1));
                    acc[2] = acc[2].wrapping_add(*src_p.add(i + 2));
                    *dst_p.add(i + 2) = acc[2].wrapping_add(*prev_p.add(i + 2));
                    i += 3;
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn unpack_8px_bgr_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
        v: __m256i,
        unpack12: __m256i,
        unpack_g: __m256i,
        const_alpha_vec: __m256i,
        shuf_a256: __m256i,
        alpha_p: *const u8,
        dst_p: *mut MaybeUninit<Vec4u8>,
        x: usize,
    ) {
        // SAFETY: Caller guarantees AVX2, `x + 8 <= dst.len()`, `x + 8 <= alpha.len()`
        // when `!CONST_ALPHA`, and 32-byte alignment on `dst_p.add(x)` when `STREAM`.
        unsafe {
            let sg = _mm256_shuffle_epi8(v, unpack12);
            let g = _mm256_shuffle_epi8(v, unpack_g);
            let alpha_vec = if CONST_ALPHA {
                const_alpha_vec
            } else {
                let a256 = _mm256_set1_epi64x(alpha_p.add(x).cast::<i64>().read_unaligned());
                _mm256_shuffle_epi8(a256, shuf_a256)
            };
            let bgra = _mm256_or_si256(_mm256_add_epi8(sg, g), alpha_vec);
            let p = dst_p.add(x).cast::<__m256i>();
            if STREAM {
                _mm256_stream_si256(p, bgra);
            } else {
                _mm256_storeu_si256(p, bgra);
            }
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

        // 16-pixel (48 B -> 64 B full cache line) main loop uses 12-byte-strided
        // 16 B loads matching `row_sub3_inverse` / `row_subup3_inverse` (zero
        // `vpermd`/`vpermq` shuffles and exact Zen 3 store-to-load forwarding),
        // followed by optional 8-px overlapping, 8-px exact (`load_24_lanes`),
        // 4-px exact (`load_12`), and `0..=3`-px scalar tails.
        let lim16 = if tw >= 18 { ((tw - 2) / 16) * 16 } else { 0 };
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees AVX2; `si + 52 <= src_bgr.len()` inside
        // `x < lim16`, `si + 28 <= src_bgr.len()` in 8-px overlapping step,
        // `si + 24 <= src_bgr.len()` in 8-px exact tail, `si + 12 <= src_bgr.len()`
        // in 4-px tail, `si + 3 <= src_bgr.len()` in `tw & 3`, `alpha_row.len() == tw`
        // when `!CONST_ALPHA`, and caller upholds 32-byte alignment on `dst` when
        // `STREAM` is `true`.
        unsafe {
            let unpack12 = _mm256_set_epi8(
                -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9, -1, 8, 7, 6,
                -1, 5, 4, 3, -1, 2, 1, 0,
            );
            let unpack_g = _mm256_set_epi8(
                -1, 10, -1, 10, -1, 7, -1, 7, -1, 4, -1, 4, -1, 1, -1, 1, -1, 10, -1, 10, -1, 7,
                -1, 7, -1, 4, -1, 4, -1, 1, -1, 1,
            );
            let const_alpha_vec = _mm256_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a256 = _mm256_set_epi8(
                7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4, -1, -1, -1, 3, -1, -1, -1, 2, -1,
                -1, -1, 1, -1, -1, -1, 0, -1, -1, -1,
            );
            let src_p = src_bgr.as_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim16 {
                let a0 = _mm_loadu_si128(src_p.add(si).cast::<__m128i>());
                let a1 = _mm_loadu_si128(src_p.add(si + 12).cast::<__m128i>());
                let b0 = _mm_loadu_si128(src_p.add(si + 24).cast::<__m128i>());
                let b1 = _mm_loadu_si128(src_p.add(si + 36).cast::<__m128i>());
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(a1, a0),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x,
                );
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(b1, b0),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x + 8,
                );
                x += 16;
                si += 48;
            }
            if x + 10 <= tw {
                let a0 = _mm_loadu_si128(src_p.add(si).cast::<__m128i>());
                let a1 = _mm_loadu_si128(src_p.add(si + 12).cast::<__m128i>());
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(a1, a0),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x,
                );
                x += 8;
                si += 24;
            }
            if x + 8 <= tw {
                let (lo, hi) = load_24_lanes(src_p.add(si));
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(hi, lo),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x,
                );
                x += 8;
                si += 24;
            }
            if x < tw {
                if x + 4 <= tw {
                    let v = load_12(src_p.add(si));
                    let sg = _mm_shuffle_epi8(v, _mm256_castsi256_si128(unpack12));
                    let g = _mm_shuffle_epi8(v, _mm256_castsi256_si128(unpack_g));
                    let alpha_vec = if CONST_ALPHA {
                        _mm256_castsi256_si128(const_alpha_vec)
                    } else {
                        let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                        _mm_shuffle_epi8(a32, _mm256_castsi256_si128(shuf_a256))
                    };
                    let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                    let p = dst_p.add(x).cast::<__m128i>();
                    if STREAM {
                        _mm_stream_si128(p, bgra);
                    } else {
                        _mm_storeu_si128(p, bgra);
                    }
                    x += 4;
                    si += 12;
                }
                for _ in 0..(tw & 3) {
                    let g = *src_p.add(si + 1);
                    let a = if CONST_ALPHA {
                        const_alpha
                    } else {
                        *alpha_p.add(x)
                    };
                    dst_p.add(x).write(MaybeUninit::new(Vec4u8(
                        (*src_p.add(si)).wrapping_add(g),
                        g,
                        (*src_p.add(si + 2)).wrapping_add(g),
                        a,
                    )));
                    x += 1;
                    si += 3;
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_up_subgreen_bgr3_to_bgra_impl<
        const STREAM: bool,
        const CONST_ALPHA: bool,
        const WRITE_DEC: bool,
    >(
        filt_bgr: &[u8],
        prev_bgr: &[u8],
        cur_dec: &mut [u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        let row_b = tw * 3;
        debug_assert_eq!(filt_bgr.len(), row_b);
        debug_assert_eq!(prev_bgr.len(), row_b);
        debug_assert!(!WRITE_DEC || cur_dec.len() == row_b);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(32));

        // 16-pixel (48 B = 3 x 16 B non-overlapping -> 64 B full cache line)
        // loop computes `dec = filt + prev` in three 16 B slices, optionally
        // writes them non-overlapping to `cur_dec` (`WRITE_DEC`, exact STLF
        // match across consecutive `Up` rows), and extracts four 12 B pixel
        // quads using three 1-cycle in-lane ops (`vpalignr $12`, `vpalignr $8`,
        // `vpsrldq $4`) with zero `vpermd`/`vpermq`/`vextracti128` stalls.
        let lim16 = tw & !15;
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees AVX2; `si + 48 <= row_b` inside `x < lim16`,
        // `si + 24 <= row_b` in 8-px tail, `si + 12 <= row_b` in 4-px tail,
        // `si + 3 <= row_b` in `tw & 3`, `cur_dec.len() == row_b` when
        // `WRITE_DEC`, `alpha_row.len() == tw` when `!CONST_ALPHA`, and caller
        // upholds 32-byte alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let unpack12 = _mm256_set_epi8(
                -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9, -1, 8, 7, 6,
                -1, 5, 4, 3, -1, 2, 1, 0,
            );
            let unpack_g = _mm256_set_epi8(
                -1, 10, -1, 10, -1, 7, -1, 7, -1, 4, -1, 4, -1, 1, -1, 1, -1, 10, -1, 10, -1, 7,
                -1, 7, -1, 4, -1, 4, -1, 1, -1, 1,
            );
            let const_alpha_vec = _mm256_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a256 = _mm256_set_epi8(
                7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4, -1, -1, -1, 3, -1, -1, -1, 2, -1,
                -1, -1, 1, -1, -1, -1, 0, -1, -1, -1,
            );
            let filt_p = filt_bgr.as_ptr();
            let prev_p = prev_bgr.as_ptr();
            let dec_p = cur_dec.as_mut_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim16 {
                let d0 = _mm_add_epi8(
                    _mm_loadu_si128(filt_p.add(si).cast::<__m128i>()),
                    _mm_loadu_si128(prev_p.add(si).cast::<__m128i>()),
                );
                let d1 = _mm_add_epi8(
                    _mm_loadu_si128(filt_p.add(si + 16).cast::<__m128i>()),
                    _mm_loadu_si128(prev_p.add(si + 16).cast::<__m128i>()),
                );
                let d2 = _mm_add_epi8(
                    _mm_loadu_si128(filt_p.add(si + 32).cast::<__m128i>()),
                    _mm_loadu_si128(prev_p.add(si + 32).cast::<__m128i>()),
                );
                if WRITE_DEC {
                    _mm_storeu_si128(dec_p.add(si).cast::<__m128i>(), d0);
                    _mm_storeu_si128(dec_p.add(si + 16).cast::<__m128i>(), d1);
                    _mm_storeu_si128(dec_p.add(si + 32).cast::<__m128i>(), d2);
                }
                let q1 = _mm_alignr_epi8::<12>(d1, d0);
                let q2 = _mm_alignr_epi8::<8>(d2, d1);
                let q3 = _mm_srli_si128::<4>(d2);
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(q1, d0),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x,
                );
                unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                    _mm256_set_m128i(q3, q2),
                    unpack12,
                    unpack_g,
                    const_alpha_vec,
                    shuf_a256,
                    alpha_p,
                    dst_p,
                    x + 8,
                );
                x += 16;
                si += 48;
            }
            if x < tw {
                if x + 8 <= tw {
                    let d0_15 = _mm_add_epi8(
                        _mm_loadu_si128(filt_p.add(si).cast::<__m128i>()),
                        _mm_loadu_si128(prev_p.add(si).cast::<__m128i>()),
                    );
                    let d16_23 = _mm_add_epi8(
                        _mm_loadl_epi64(filt_p.add(si + 16).cast::<__m128i>()),
                        _mm_loadl_epi64(prev_p.add(si + 16).cast::<__m128i>()),
                    );
                    if WRITE_DEC {
                        _mm_storeu_si128(dec_p.add(si).cast::<__m128i>(), d0_15);
                        _mm_storel_epi64(dec_p.add(si + 16).cast::<__m128i>(), d16_23);
                    }
                    let dec_hi = _mm_alignr_epi8::<12>(d16_23, d0_15);
                    unpack_8px_bgr_to_bgra::<STREAM, CONST_ALPHA>(
                        _mm256_set_m128i(dec_hi, d0_15),
                        unpack12,
                        unpack_g,
                        const_alpha_vec,
                        shuf_a256,
                        alpha_p,
                        dst_p,
                        x,
                    );
                    x += 8;
                    si += 24;
                }
                if x + 4 <= tw {
                    let dec = _mm_add_epi8(load_12(filt_p.add(si)), load_12(prev_p.add(si)));
                    if WRITE_DEC {
                        store_12(dec_p.add(si), dec);
                    }
                    let sg = _mm_shuffle_epi8(dec, _mm256_castsi256_si128(unpack12));
                    let g = _mm_shuffle_epi8(dec, _mm256_castsi256_si128(unpack_g));
                    let alpha_vec = if CONST_ALPHA {
                        _mm256_castsi256_si128(const_alpha_vec)
                    } else {
                        let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                        _mm_shuffle_epi8(a32, _mm256_castsi256_si128(shuf_a256))
                    };
                    let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                    let p = dst_p.add(x).cast::<__m128i>();
                    if STREAM {
                        _mm_stream_si128(p, bgra);
                    } else {
                        _mm_storeu_si128(p, bgra);
                    }
                    x += 4;
                    si += 12;
                }
                for _ in 0..(tw & 3) {
                    let db = (*filt_p.add(si)).wrapping_add(*prev_p.add(si));
                    let dg = (*filt_p.add(si + 1)).wrapping_add(*prev_p.add(si + 1));
                    let dr = (*filt_p.add(si + 2)).wrapping_add(*prev_p.add(si + 2));
                    if WRITE_DEC {
                        *dec_p.add(si) = db;
                        *dec_p.add(si + 1) = dg;
                        *dec_p.add(si + 2) = dr;
                    }
                    let a = if CONST_ALPHA {
                        const_alpha
                    } else {
                        *alpha_p.add(x)
                    };
                    dst_p.add(x).write(MaybeUninit::new(Vec4u8(
                        db.wrapping_add(dg),
                        dg,
                        dr.wrapping_add(dg),
                        a,
                    )));
                    x += 1;
                    si += 3;
                }
            }
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
    unsafe fn merge_bgrx_alpha_row_ptr<const STREAM: bool, const CONST_ALPHA: bool>(
        src: *const Vec4u8,
        alpha_p: *const u8,
        const_alpha: u8,
        dst: *mut MaybeUninit<Vec4u8>,
        tw: usize,
    ) {
        debug_assert!(!STREAM || (dst as usize).is_multiple_of(32));
        let lim = (tw / 8) * 8;
        // SAFETY: Caller guarantees AVX2, `src` readable and `dst` writable for
        // `tw` pixels (identical pointers permitted when `!STREAM`), `alpha_p`
        // readable for `tw` bytes when `!CONST_ALPHA`, and 32-byte alignment on
        // `dst` when `STREAM` is `true`.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let const_alpha_vec = _mm256_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let mut x = 0usize;
            while x < lim {
                let v = _mm256_loadu_si256(src.add(x).cast::<__m256i>());
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a64 = _mm_loadl_epi64(alpha_p.add(x).cast::<__m128i>());
                    _mm256_slli_epi32::<24>(_mm256_cvtepu8_epi32(a64))
                };
                let out = _mm256_or_si256(_mm256_and_si256(v, rgb_mask), alpha_vec);
                let p = dst.add(x).cast::<__m256i>();
                if STREAM {
                    _mm256_stream_si256(p, out);
                } else {
                    _mm256_storeu_si256(p, out);
                }
                x += 8;
            }
            while x < tw {
                let px = src.add(x).read();
                let a = if CONST_ALPHA {
                    const_alpha
                } else {
                    *alpha_p.add(x)
                };
                dst.add(x)
                    .write(MaybeUninit::new(Vec4u8(px.0, px.1, px.2, a)));
                x += 1;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_strip_avx2<const HAS_ALPHA: bool>(
        c0: __m256i,
        c1: __m256i,
        rgb_mask: __m256i,
    ) -> u64 {
        const M: [u32; 16] = crate::filtering::scroll::STRIP_HASH_MULS;
        const MUL_EVEN_LO: [u64; 4] = [M[0] as u64, M[2] as u64, M[4] as u64, M[6] as u64];
        const MUL_ODD_LO: [u64; 4] = [M[1] as u64, M[3] as u64, M[5] as u64, M[7] as u64];
        const MUL_EVEN_HI: [u64; 4] = [M[8] as u64, M[10] as u64, M[12] as u64, M[14] as u64];
        const MUL_ODD_HI: [u64; 4] = [M[9] as u64, M[11] as u64, M[13] as u64, M[15] as u64];
        // SAFETY: Caller guarantees AVX2.
        unsafe {
            let (cm0, cm1) = if HAS_ALPHA {
                (c0, c1)
            } else {
                (
                    _mm256_and_si256(c0, rgb_mask),
                    _mm256_and_si256(c1, rgb_mask),
                )
            };
            let me_lo = _mm256_loadu_si256(MUL_EVEN_LO.as_ptr().cast::<__m256i>());
            let mo_lo = _mm256_loadu_si256(MUL_ODD_LO.as_ptr().cast::<__m256i>());
            let me_hi = _mm256_loadu_si256(MUL_EVEN_HI.as_ptr().cast::<__m256i>());
            let mo_hi = _mm256_loadu_si256(MUL_ODD_HI.as_ptr().cast::<__m256i>());
            let a0 = _mm256_mul_epu32(cm0, me_lo);
            let a1 = _mm256_mul_epu32(_mm256_srli_epi64::<32>(cm0), mo_lo);
            let b0 = _mm256_mul_epu32(cm1, me_hi);
            let b1 = _mm256_mul_epu32(_mm256_srli_epi64::<32>(cm1), mo_hi);
            let t = _mm256_add_epi64(_mm256_xor_si256(a0, b0), _mm256_xor_si256(a1, b1));
            let v128 = _mm_xor_si128(_mm256_castsi256_si128(t), _mm256_extracti128_si256::<1>(t));
            let x0 = _mm_cvtsi128_si64(v128) as u64;
            let x1 = _mm_extract_epi64::<1>(v128) as u64;
            crate::filtering::scroll::finalize_strip_hash(x0, x1)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn probe_strip_activity_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> Option<bool> {
        // SAFETY: Caller guarantees AVX2 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let c0 = _mm256_loadu_si256(curr.cast::<__m256i>());
            let c1 = _mm256_loadu_si256(curr.add(8).cast::<__m256i>());
            let d0 = _mm256_xor_si256(c0, _mm256_loadu_si256(prev.cast::<__m256i>()));
            let d1 = _mm256_xor_si256(c1, _mm256_loadu_si256(prev.add(8).cast::<__m256i>()));
            let diff = _mm256_or_si256(d0, d1);
            let changed = if HAS_ALPHA {
                _mm256_testz_si256(diff, diff) == 0
            } else {
                _mm256_testz_si256(diff, rgb_mask) == 0
            };
            if !changed {
                return None;
            }
            let first = _mm256_broadcastd_epi32(_mm256_castsi256_si128(c0));
            let u = _mm256_or_si256(_mm256_xor_si256(c0, first), _mm256_xor_si256(c1, first));
            let non_uniform = if HAS_ALPHA {
                _mm256_testz_si256(u, u) == 0
            } else {
                _mm256_testz_si256(u, rgb_mask) == 0
            };
            Some(non_uniform)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn probe_curr_strip_hash_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> (bool, Option<u64>) {
        // SAFETY: Caller guarantees AVX2 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let c0 = _mm256_loadu_si256(curr.cast::<__m256i>());
            let c1 = _mm256_loadu_si256(curr.add(8).cast::<__m256i>());
            let d0 = _mm256_xor_si256(c0, _mm256_loadu_si256(prev.cast::<__m256i>()));
            let d1 = _mm256_xor_si256(c1, _mm256_loadu_si256(prev.add(8).cast::<__m256i>()));
            let diff = _mm256_or_si256(d0, d1);
            let changed = if HAS_ALPHA {
                _mm256_testz_si256(diff, diff) == 0
            } else {
                _mm256_testz_si256(diff, rgb_mask) == 0
            };
            if !changed {
                return (false, None);
            }
            let first = _mm256_broadcastd_epi32(_mm256_castsi256_si128(c0));
            let u = _mm256_or_si256(_mm256_xor_si256(c0, first), _mm256_xor_si256(c1, first));
            let non_uniform = if HAS_ALPHA {
                _mm256_testz_si256(u, u) == 0
            } else {
                _mm256_testz_si256(u, rgb_mask) == 0
            };
            if !non_uniform {
                return (true, None);
            }
            (true, Some(hash_strip_avx2::<HAS_ALPHA>(c0, c1, rgb_mask)))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_prev_strip_non_uniform_impl<const HAS_ALPHA: bool>(
        prev: *const Vec4u8,
    ) -> Option<u64> {
        // SAFETY: Caller guarantees AVX2 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let p0 = _mm256_loadu_si256(prev.cast::<__m256i>());
            let p1 = _mm256_loadu_si256(prev.add(8).cast::<__m256i>());
            let first = _mm256_broadcastd_epi32(_mm256_castsi256_si128(p0));
            let u = _mm256_or_si256(_mm256_xor_si256(p0, first), _mm256_xor_si256(p1, first));
            let non_uniform = if HAS_ALPHA {
                _mm256_testz_si256(u, u) == 0
            } else {
                _mm256_testz_si256(u, rgb_mask) == 0
            };
            if !non_uniform {
                return None;
            }
            Some(hash_strip_avx2::<HAS_ALPHA>(p0, p1, rgb_mask))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn score_strip_row_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev_src: *const Vec4u8,
        prev_dst: *const Vec4u8,
    ) -> i32 {
        // SAFETY: Caller guarantees AVX2 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm256_setzero_si256();
            let c0 = _mm256_loadu_si256(curr.cast::<__m256i>());
            let c1 = _mm256_loadu_si256(curr.add(8).cast::<__m256i>());
            let d0 = _mm256_xor_si256(c0, _mm256_loadu_si256(prev_src.cast::<__m256i>()));
            let d1 = _mm256_xor_si256(c1, _mm256_loadu_si256(prev_src.add(8).cast::<__m256i>()));
            let (dm0, dm1) = if HAS_ALPHA {
                (d0, d1)
            } else {
                (
                    _mm256_and_si256(d0, rgb_mask),
                    _mm256_and_si256(d1, rgb_mask),
                )
            };
            let eq0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(dm0, zero))) as u32;
            let eq1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(dm1, zero))) as u32;
            let eq_ps = eq0 | (eq1 << 8);
            if eq_ps.count_ones() < 12 {
                return -4;
            }
            let e0 = _mm256_xor_si256(c0, _mm256_loadu_si256(prev_dst.cast::<__m256i>()));
            let e1 = _mm256_xor_si256(c1, _mm256_loadu_si256(prev_dst.add(8).cast::<__m256i>()));
            let (em0, em1) = if HAS_ALPHA {
                (e0, e1)
            } else {
                (
                    _mm256_and_si256(e0, rgb_mask),
                    _mm256_and_si256(e1, rgb_mask),
                )
            };
            let eq_p0_0 =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(em0, zero))) as u32;
            let eq_p0_1 =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(em1, zero))) as u32;
            let eq_p0 = eq_p0_0 | (eq_p0_1 << 8);
            if (eq_ps & !eq_p0) != 0 { 3 } else { 0 }
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn copy_lt64_avx2(src: *const u8, dst: *mut u8, len: usize) {
        debug_assert!(len > 0 && len < 64);
        // SAFETY: Caller guarantees AVX2, `1 <= len < 64`, and valid non-overlapping ranges.
        unsafe {
            if len >= 32 {
                let a = _mm256_loadu_si256(src.cast::<__m256i>());
                let b = _mm256_loadu_si256(src.add(len - 32).cast::<__m256i>());
                _mm256_storeu_si256(dst.cast::<__m256i>(), a);
                _mm256_storeu_si256(dst.add(len - 32).cast::<__m256i>(), b);
            } else if len >= 16 {
                let a = _mm_loadu_si128(src.cast::<__m128i>());
                let b = _mm_loadu_si128(src.add(len - 16).cast::<__m128i>());
                _mm_storeu_si128(dst.cast::<__m128i>(), a);
                _mm_storeu_si128(dst.add(len - 16).cast::<__m128i>(), b);
            } else if len >= 8 {
                let a = src.cast::<u64>().read_unaligned();
                let b = src.add(len - 8).cast::<u64>().read_unaligned();
                dst.cast::<u64>().write_unaligned(a);
                dst.add(len - 8).cast::<u64>().write_unaligned(b);
            } else if len >= 4 {
                let a = src.cast::<u32>().read_unaligned();
                let b = src.add(len - 4).cast::<u32>().read_unaligned();
                dst.cast::<u32>().write_unaligned(a);
                dst.add(len - 4).cast::<u32>().write_unaligned(b);
            } else if len >= 2 {
                let a = src.cast::<u16>().read_unaligned();
                let b = src.add(len - 2).cast::<u16>().read_unaligned();
                dst.cast::<u16>().write_unaligned(a);
                dst.add(len - 2).cast::<u16>().write_unaligned(b);
            } else {
                *dst = *src;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn copy_bytes_stream_impl(src: *const u8, dst: *mut u8, len: usize) {
        if len == 0 {
            return;
        }
        // SAFETY: Caller guarantees AVX2 and valid non-overlapping ranges.
        unsafe {
            if len < 128 {
                if len < 64 {
                    copy_lt64_avx2(src, dst, len);
                } else {
                    let a = _mm256_loadu_si256(src.cast::<__m256i>());
                    let b = _mm256_loadu_si256(src.add(32).cast::<__m256i>());
                    let c = _mm256_loadu_si256(src.add(len - 64).cast::<__m256i>());
                    let d = _mm256_loadu_si256(src.add(len - 32).cast::<__m256i>());
                    _mm256_storeu_si256(dst.cast::<__m256i>(), a);
                    _mm256_storeu_si256(dst.add(32).cast::<__m256i>(), b);
                    _mm256_storeu_si256(dst.add(len - 64).cast::<__m256i>(), c);
                    _mm256_storeu_si256(dst.add(len - 32).cast::<__m256i>(), d);
                }
                return;
            }
            let head = (64 - (dst as usize & 63)) & 63;
            if head != 0 {
                copy_lt64_avx2(src, dst, head);
            }
            let s = src.add(head);
            let d = dst.add(head);
            let rem = len - head;
            let lim128 = rem & !127;
            let lim64 = rem & !63;
            let mut i = 0usize;
            while i < lim128 {
                let v0 = _mm256_loadu_si256(s.add(i).cast::<__m256i>());
                let v1 = _mm256_loadu_si256(s.add(i + 32).cast::<__m256i>());
                let v2 = _mm256_loadu_si256(s.add(i + 64).cast::<__m256i>());
                let v3 = _mm256_loadu_si256(s.add(i + 96).cast::<__m256i>());
                _mm256_stream_si256(d.add(i).cast::<__m256i>(), v0);
                _mm256_stream_si256(d.add(i + 32).cast::<__m256i>(), v1);
                _mm256_stream_si256(d.add(i + 64).cast::<__m256i>(), v2);
                _mm256_stream_si256(d.add(i + 96).cast::<__m256i>(), v3);
                i += 128;
            }
            if i < lim64 {
                let v0 = _mm256_loadu_si256(s.add(i).cast::<__m256i>());
                let v1 = _mm256_loadu_si256(s.add(i + 32).cast::<__m256i>());
                _mm256_stream_si256(d.add(i).cast::<__m256i>(), v0);
                _mm256_stream_si256(d.add(i + 32).cast::<__m256i>(), v1);
                i += 64;
            }
            let tail = rem - lim64;
            if tail != 0 {
                copy_lt64_avx2(s.add(i), d.add(i), tail);
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
        let mut i = 0usize;
        // SAFETY: Caller guarantees AVX2, `i + 32 <= len` in main loop, and
        // `i + 16 <= len` in optional 16-byte tail.
        unsafe {
            let cur_p = cur.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst.as_mut_ptr();
            while i < lim {
                let c = _mm256_loadu_si256(cur_p.add(i).cast::<__m256i>());
                let u = _mm256_loadu_si256(prev_p.add(i).cast::<__m256i>());
                _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), _mm256_sub_epi8(c, u));
                i += 32;
            }
            if i + 16 <= len {
                let c = _mm_loadu_si128(cur_p.add(i).cast::<__m128i>());
                let u = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_sub_epi8(c, u));
                i += 16;
            }
        }
        for j in i..len {
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
    fn avx2_px_neq_mask8<const HAS_ALPHA: bool>(
        diff: __m256i,
        rgb_mask: __m256i,
        zero: __m256i,
    ) -> u32 {
        let masked = if HAS_ALPHA {
            diff
        } else {
            _mm256_and_si256(diff, rgb_mask)
        };
        let eq = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(masked, zero))) as u32;
        (!eq) & 0xFF
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
        let lim32 = n & !31;
        let lim8 = n & !7;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees AVX2, `src_p` and `dst_p` are valid for `n`
        // `Vec4u8`s and non-overlapping (`src` is shm/caller, `shadow` is
        // exclusively borrowed), and all offsets remain strictly `< n`.
        unsafe {
            let rgb_mask = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm256_setzero_si256();
            let mut i = 0usize;

            // 4-vector OR-screened scan (128 B = 32 px/iter). Keeps `s0..s3` in
            // YMM registers so a mismatch commits those exact loads without
            // ever re-reading `src_p` (TOCTOU-safe under concurrent `wl_shm`
            // mutation), and packs all 32 pixel-diff bits into a single `u32`
            // via `vmovmskps` for branchless `first`/`last_end` extraction.
            while i < lim32 {
                let s0 = _mm256_loadu_si256(src_p.add(i).cast::<__m256i>());
                let x0 = _mm256_xor_si256(s0, _mm256_loadu_si256(dst_p.add(i).cast::<__m256i>()));
                let s1 = _mm256_loadu_si256(src_p.add(i + 8).cast::<__m256i>());
                let x1 =
                    _mm256_xor_si256(s1, _mm256_loadu_si256(dst_p.add(i + 8).cast::<__m256i>()));
                let s2 = _mm256_loadu_si256(src_p.add(i + 16).cast::<__m256i>());
                let x2 =
                    _mm256_xor_si256(s2, _mm256_loadu_si256(dst_p.add(i + 16).cast::<__m256i>()));
                let s3 = _mm256_loadu_si256(src_p.add(i + 24).cast::<__m256i>());
                let x3 =
                    _mm256_xor_si256(s3, _mm256_loadu_si256(dst_p.add(i + 24).cast::<__m256i>()));
                let any = _mm256_or_si256(_mm256_or_si256(x0, x1), _mm256_or_si256(x2, x3));
                let is_dirty = if HAS_ALPHA {
                    _mm256_testz_si256(any, any) == 0
                } else {
                    _mm256_testz_si256(any, rgb_mask) == 0
                };
                if is_dirty {
                    let m0 = avx2_px_neq_mask8::<HAS_ALPHA>(x0, rgb_mask, zero);
                    let m1 = avx2_px_neq_mask8::<HAS_ALPHA>(x1, rgb_mask, zero);
                    let m2 = avx2_px_neq_mask8::<HAS_ALPHA>(x2, rgb_mask, zero);
                    let m3 = avx2_px_neq_mask8::<HAS_ALPHA>(x3, rgb_mask, zero);
                    if m0 != 0 {
                        _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), s0);
                    }
                    if m1 != 0 {
                        _mm256_storeu_si256(dst_p.add(i + 8).cast::<__m256i>(), s1);
                    }
                    if m2 != 0 {
                        _mm256_storeu_si256(dst_p.add(i + 16).cast::<__m256i>(), s2);
                    }
                    if m3 != 0 {
                        _mm256_storeu_si256(dst_p.add(i + 24).cast::<__m256i>(), s3);
                    }
                    let mask32 = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
                    debug_assert_ne!(mask32, 0);
                    if first == usize::MAX {
                        first = i + mask32.trailing_zeros() as usize;
                    }
                    last_end = i + (32 - mask32.leading_zeros() as usize);
                }
                i += 32;
            }

            while i < lim8 {
                let s = _mm256_loadu_si256(src_p.add(i).cast::<__m256i>());
                let diff = _mm256_xor_si256(s, _mm256_loadu_si256(dst_p.add(i).cast::<__m256i>()));
                let is_dirty = if HAS_ALPHA {
                    _mm256_testz_si256(diff, diff) == 0
                } else {
                    _mm256_testz_si256(diff, rgb_mask) == 0
                };
                if is_dirty {
                    _mm256_storeu_si256(dst_p.add(i).cast::<__m256i>(), s);
                    let neq = avx2_px_neq_mask8::<HAS_ALPHA>(diff, rgb_mask, zero);
                    debug_assert_ne!(neq, 0);
                    if first == usize::MAX {
                        first = i + neq.trailing_zeros() as usize;
                    }
                    last_end = i + (32 - neq.leading_zeros() as usize);
                }
                i += 8;
            }
            for j in lim8..n {
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

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn validate_palette_indices_impl(n_colors: usize, packed: &[u8]) -> bool {
        let len = packed.len();
        // SAFETY: Caller guarantees AVX2 and all loads stay in `0..len`.
        unsafe {
            if n_colors == 3 {
                let lim64 = (len / 64) * 64;
                let mut acc0 = _mm256_setzero_si256();
                let mut acc1 = _mm256_setzero_si256();
                let mut i = 0usize;
                while i < lim64 {
                    let v0 = _mm256_loadu_si256(packed.as_ptr().add(i).cast());
                    let v1 = _mm256_loadu_si256(packed.as_ptr().add(i + 32).cast());
                    acc0 = _mm256_or_si256(acc0, _mm256_and_si256(v0, _mm256_srli_epi16::<1>(v0)));
                    acc1 = _mm256_or_si256(acc1, _mm256_and_si256(v1, _mm256_srli_epi16::<1>(v1)));
                    i += 64;
                }
                if i + 32 <= len {
                    let v = _mm256_loadu_si256(packed.as_ptr().add(i).cast());
                    acc0 = _mm256_or_si256(acc0, _mm256_and_si256(v, _mm256_srli_epi16::<1>(v)));
                    i += 32;
                }
                let acc = _mm256_or_si256(acc0, acc1);
                if _mm256_testz_si256(acc, _mm256_set1_epi8(0x55)) == 0 {
                    return false;
                }
                validate_palette_indices_scalar(3, &packed[i..])
            } else if (5..16).contains(&n_colors) {
                let lim = (len / 32) * 32;
                let mask_0f = _mm256_set1_epi8(0x0F);
                let mut max_lo = _mm256_setzero_si256();
                let mut max_hi = _mm256_setzero_si256();
                let mut i = 0usize;
                while i < lim {
                    let v = _mm256_loadu_si256(packed.as_ptr().add(i).cast());
                    max_lo = _mm256_max_epu8(max_lo, _mm256_and_si256(v, mask_0f));
                    max_hi = _mm256_max_epu8(max_hi, _mm256_andnot_si256(mask_0f, v));
                    i += 32;
                }
                let combined = _mm256_max_epu8(max_lo, _mm256_srli_epi16::<4>(max_hi));
                let excess = _mm256_subs_epu8(combined, _mm256_set1_epi8((n_colors - 1) as i8));
                if _mm256_testz_si256(excess, excess) == 0 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else if (17..MAX_PALETTE_COLORS).contains(&n_colors) {
                let lim64 = (len / 64) * 64;
                let mut max0 = _mm256_setzero_si256();
                let mut max1 = _mm256_setzero_si256();
                let mut i = 0usize;
                while i < lim64 {
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(packed.as_ptr().add(i).cast()));
                    max1 = _mm256_max_epu8(
                        max1,
                        _mm256_loadu_si256(packed.as_ptr().add(i + 32).cast()),
                    );
                    i += 64;
                }
                if i + 32 <= len {
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(packed.as_ptr().add(i).cast()));
                    i += 32;
                }
                let combined = _mm256_max_epu8(max0, max1);
                let excess = _mm256_subs_epu8(combined, _mm256_set1_epi8((n_colors - 1) as i8));
                if _mm256_testz_si256(excess, excess) == 0 {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[i..])
            } else {
                true
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn pack_palette_indices_impl(
        indices: &[u8],
        n_colors: usize,
        packed: &mut [u8],
    ) -> usize {
        let n = indices.len();
        let out_len = packed_palette_indices_len(n, n_colors);
        debug_assert!(packed.len() >= out_len);
        if n_colors <= 1 {
            return 0;
        }
        // SAFETY: Caller guarantees AVX2 and `packed.len() >= out_len`.
        unsafe {
            let src = indices.as_ptr();
            let dst = packed.as_mut_ptr();
            if n_colors <= 2 {
                let lim = (n / 32) * 32;
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm256_loadu_si256(src.add(i).cast());
                    let mask = _mm256_movemask_epi8(_mm256_slli_epi16::<7>(v)) as u32;
                    dst.add(oi).cast::<u32>().write_unaligned(mask.to_le());
                    i += 32;
                    oi += 4;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 2, &mut packed[oi..]);
                }
            } else if n_colors <= 4 {
                let lim = (n / 32) * 32;
                let mask3 = _mm256_set1_epi8(3);
                let mul1 = _mm256_set1_epi16(0x0401);
                let mul2 = _mm256_set1_epi32(0x0010_0001);
                let shuf = _mm256_setr_epi8(
                    0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                );
                let gather_d = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm256_and_si256(_mm256_loadu_si256(src.add(i).cast()), mask3);
                    let t1 = _mm256_maddubs_epi16(v, mul1);
                    let t2 = _mm256_madd_epi16(t1, mul2);
                    let pk = _mm256_shuffle_epi8(t2, shuf);
                    let pk64 = _mm256_permutevar8x32_epi32(pk, gather_d);
                    _mm_storel_epi64(dst.add(oi).cast(), _mm256_castsi256_si128(pk64));
                    i += 32;
                    oi += 8;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 4, &mut packed[oi..]);
                }
            } else if n_colors <= 16 {
                let lim64 = (n / 64) * 64;
                let mask_0f = _mm256_set1_epi8(0x0F);
                let mul16 = _mm256_set1_epi16(0x1001);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim64 {
                    let v0 = _mm256_and_si256(_mm256_loadu_si256(src.add(i).cast()), mask_0f);
                    let v1 = _mm256_and_si256(_mm256_loadu_si256(src.add(i + 32).cast()), mask_0f);
                    let t0 = _mm256_maddubs_epi16(v0, mul16);
                    let t1 = _mm256_maddubs_epi16(v1, mul16);
                    let pk = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(t0, t1));
                    _mm256_storeu_si256(dst.add(oi).cast(), pk);
                    i += 64;
                    oi += 32;
                }
                if i + 32 <= n {
                    let v = _mm256_and_si256(_mm256_loadu_si256(src.add(i).cast()), mask_0f);
                    let t = _mm256_maddubs_epi16(v, mul16);
                    let pk = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(t, t));
                    _mm_storeu_si128(dst.add(oi).cast(), _mm256_castsi256_si128(pk));
                    i += 32;
                    oi += 16;
                }
                if i < n {
                    pack_palette_indices_scalar(&indices[i..], 16, &mut packed[oi..]);
                }
            } else {
                packed[..n].copy_from_slice(indices);
            }
        }
        out_len
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn decode_palette_1bpp_row_impl<const STREAM: bool>(
        c0: Vec4u8,
        c1: Vec4u8,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(32)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 7) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((8 - (idx_pos & 7)) & 7).min(tw);
            if h != 0 {
                decode_palette_1bpp_scalar(c0, c1, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 8 <= tw {
            debug_assert_eq!(pos & 7, 0);
            let mut bi = pos >> 3;
            let end32 = x + ((tw - x) / 32) * 32;
            let end8 = if STREAM { tw } else { x + ((tw - x) / 8) * 8 };
            // SAFETY: Caller guarantees AVX2, `packed` bounds, and 32-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let vc0 = _mm256_castsi256_ps(_mm256_set1_epi32(u32::from_le_bytes([
                    c0.0, c0.1, c0.2, c0.3,
                ]) as i32));
                let vc1 = _mm256_castsi256_ps(_mm256_set1_epi32(u32::from_le_bytes([
                    c1.0, c1.1, c1.2, c1.3,
                ]) as i32));
                let sh0 = _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24);
                let sh1 = _mm256_setr_epi32(23, 22, 21, 20, 19, 18, 17, 16);
                let sh2 = _mm256_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8);
                let sh3 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end32 {
                    let w = _mm256_set1_epi32(u32::from_le(
                        src.add(bi).cast::<u32>().read_unaligned(),
                    ) as i32);
                    let p0 = _mm256_castps_si256(_mm256_blendv_ps(
                        vc0,
                        vc1,
                        _mm256_castsi256_ps(_mm256_sllv_epi32(w, sh0)),
                    ));
                    let p1 = _mm256_castps_si256(_mm256_blendv_ps(
                        vc0,
                        vc1,
                        _mm256_castsi256_ps(_mm256_sllv_epi32(w, sh1)),
                    ));
                    let p2 = _mm256_castps_si256(_mm256_blendv_ps(
                        vc0,
                        vc1,
                        _mm256_castsi256_ps(_mm256_sllv_epi32(w, sh2)),
                    ));
                    let p3 = _mm256_castps_si256(_mm256_blendv_ps(
                        vc0,
                        vc1,
                        _mm256_castsi256_ps(_mm256_sllv_epi32(w, sh3)),
                    ));
                    let p = out.add(x).cast::<__m256i>();
                    if STREAM {
                        _mm256_stream_si256(p, p0);
                        _mm256_stream_si256(p.add(1), p1);
                        _mm256_stream_si256(p.add(2), p2);
                        _mm256_stream_si256(p.add(3), p3);
                    } else {
                        _mm256_storeu_si256(p, p0);
                        _mm256_storeu_si256(p.add(1), p1);
                        _mm256_storeu_si256(p.add(2), p2);
                        _mm256_storeu_si256(p.add(3), p3);
                    }
                    x += 32;
                    bi += 4;
                }
                while x < end8 {
                    let b = _mm256_set1_epi8(*src.add(bi) as i8);
                    let px = _mm256_castps_si256(_mm256_blendv_ps(
                        vc0,
                        vc1,
                        _mm256_castsi256_ps(_mm256_sllv_epi32(b, sh0)),
                    ));
                    let p = out.add(x).cast::<__m256i>();
                    if STREAM {
                        _mm256_stream_si256(p, px);
                    } else {
                        _mm256_storeu_si256(p, px);
                    }
                    x += 8;
                    bi += 1;
                }
            }
            pos = bi << 3;
        }
        if !STREAM && x < tw {
            decode_palette_1bpp_scalar(c0, c1, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn decode_palette_2bpp_row_impl<const STREAM: bool>(
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(32)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 3) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((4 - (idx_pos & 3)) & 3).min(tw);
            if h != 0 {
                decode_palette_2bpp_scalar(lut, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 8 <= tw {
            debug_assert_eq!(pos & 3, 0);
            let mut bi = pos >> 2;
            let end16 = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            let end8 = x + ((tw - x) / 8) * 8;
            // SAFETY: Caller guarantees AVX2, `packed` bounds, and 32-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                // Replicating `[c0, c1, c2, c3]` into both 128-bit halves makes
                // `vpermd` (`idx[2:0]`) wrap modulo 4 with zero `vpand` mask.
                let lut_256 = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.as_ptr().cast()));
                let sh0 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
                let sh1 = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end16 {
                    let w = _mm256_set1_epi32(u32::from_le(
                        src.add(bi).cast::<u32>().read_unaligned(),
                    ) as i32);
                    let p0 = _mm256_permutevar8x32_epi32(lut_256, _mm256_srlv_epi32(w, sh0));
                    let p1 = _mm256_permutevar8x32_epi32(lut_256, _mm256_srlv_epi32(w, sh1));
                    let p = out.add(x).cast::<__m256i>();
                    if STREAM {
                        _mm256_stream_si256(p, p0);
                        _mm256_stream_si256(p.add(1), p1);
                    } else {
                        _mm256_storeu_si256(p, p0);
                        _mm256_storeu_si256(p.add(1), p1);
                    }
                    x += 16;
                    bi += 4;
                }
                if !STREAM && x < end8 {
                    let w = _mm256_set1_epi32(i32::from(u16::from_le(
                        src.add(bi).cast::<u16>().read_unaligned(),
                    )));
                    let p0 = _mm256_permutevar8x32_epi32(lut_256, _mm256_srlv_epi32(w, sh0));
                    let p = out.add(x).cast::<__m256i>();
                    _mm256_storeu_si256(p, p0);
                    x += 8;
                    bi += 2;
                }
            }
            pos = bi << 2;
        }
        if !STREAM && x < tw {
            decode_palette_2bpp_scalar(lut, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn decode_palette_4bpp_row_impl<const STREAM: bool>(
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(32)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 1) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = (idx_pos & 1).min(tw);
            if h != 0 {
                decode_palette_4bpp_scalar(&lut.colors, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 1, 0);
            let mut bi = pos >> 1;
            let end32 = x + ((tw - x) / 32) * 32;
            let end16 = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees AVX2, `packed` bounds, and 32-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let b_lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.b.as_ptr().cast()));
                let g_lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.g.as_ptr().cast()));
                let r_lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.r.as_ptr().cast()));
                let a_lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.a.as_ptr().cast()));
                // De-interleave 16-bit pairs into even/odd halves so in-lane
                // AVX2 `unpacklo/hi` yields `[P0..7, P8..15, P16..23, P24..31]`
                // with zero cross-lane output shuffles.
                let deint_w = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                let mask_0f = _mm256_set1_epi8(0x0F);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end32 {
                    let raw = _mm_loadu_si128(src.add(bi).cast());
                    let grouped = _mm_shuffle_epi8(raw, deint_w);
                    let g256 = _mm256_permute4x64_epi64::<0x10>(_mm256_zextsi128_si256(grouped));
                    let lo = _mm256_and_si256(g256, mask_0f);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(g256), mask_0f);
                    let idx = _mm256_unpacklo_epi8(lo, hi);

                    let b = _mm256_shuffle_epi8(b_lut, idx);
                    let g = _mm256_shuffle_epi8(g_lut, idx);
                    let r = _mm256_shuffle_epi8(r_lut, idx);
                    let a = _mm256_shuffle_epi8(a_lut, idx);

                    let bg_lo = _mm256_unpacklo_epi8(b, g);
                    let bg_hi = _mm256_unpackhi_epi8(b, g);
                    let ra_lo = _mm256_unpacklo_epi8(r, a);
                    let ra_hi = _mm256_unpackhi_epi8(r, a);

                    let out0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                    let out1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);
                    let out2 = _mm256_unpacklo_epi16(bg_hi, ra_hi);
                    let out3 = _mm256_unpackhi_epi16(bg_hi, ra_hi);

                    let p = out.add(x).cast::<__m256i>();
                    if STREAM {
                        _mm256_stream_si256(p, out0);
                        _mm256_stream_si256(p.add(1), out1);
                        _mm256_stream_si256(p.add(2), out2);
                        _mm256_stream_si256(p.add(3), out3);
                    } else {
                        _mm256_storeu_si256(p, out0);
                        _mm256_storeu_si256(p.add(1), out1);
                        _mm256_storeu_si256(p.add(2), out2);
                        _mm256_storeu_si256(p.add(3), out3);
                    }
                    x += 32;
                    bi += 16;
                }
                if x < end16 {
                    let raw = _mm_loadl_epi64(src.add(bi).cast());
                    let grouped = _mm_shuffle_epi8(raw, deint_w);
                    let g256 = _mm256_permute4x64_epi64::<0x10>(_mm256_zextsi128_si256(grouped));
                    let lo = _mm256_and_si256(g256, mask_0f);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(g256), mask_0f);
                    let idx = _mm256_unpacklo_epi8(lo, hi);

                    let b = _mm256_shuffle_epi8(b_lut, idx);
                    let g = _mm256_shuffle_epi8(g_lut, idx);
                    let r = _mm256_shuffle_epi8(r_lut, idx);
                    let a = _mm256_shuffle_epi8(a_lut, idx);

                    let bg_lo = _mm256_unpacklo_epi8(b, g);
                    let ra_lo = _mm256_unpacklo_epi8(r, a);
                    let out0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                    let out1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);

                    let p = out.add(x).cast::<__m256i>();
                    if STREAM {
                        _mm256_stream_si256(p, out0);
                        _mm256_stream_si256(p.add(1), out1);
                    } else {
                        _mm256_storeu_si256(p, out0);
                        _mm256_storeu_si256(p.add(1), out1);
                    }
                    x += 16;
                    bi += 8;
                }
            }
            pos = bi << 1;
        }
        if !STREAM && x < tw {
            decode_palette_4bpp_scalar(&lut.colors, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn decode_palette_8bpp_row_impl<const STREAM: bool>(
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(packed_row.len(), tw);
        debug_assert!(
            !STREAM || ((dst.as_mut_ptr() as usize).is_multiple_of(32) && tw.is_multiple_of(16))
        );
        let lim = if STREAM { tw } else { (tw / 8) * 8 };
        // SAFETY: Caller guarantees AVX2, `packed_row.len() == tw`, and 32-byte
        // alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let src = packed_row.as_ptr();
            let out = dst.as_mut_ptr();
            let mut x = 0usize;
            while x < lim {
                let w0 = u32::from_le(src.add(x).cast::<u32>().read_unaligned());
                let w1 = u32::from_le(src.add(x + 4).cast::<u32>().read_unaligned());
                let v = _mm256_setr_epi32(
                    lut[(w0 & 0xFF) as usize] as i32,
                    lut[((w0 >> 8) & 0xFF) as usize] as i32,
                    lut[((w0 >> 16) & 0xFF) as usize] as i32,
                    lut[(w0 >> 24) as usize] as i32,
                    lut[(w1 & 0xFF) as usize] as i32,
                    lut[((w1 >> 8) & 0xFF) as usize] as i32,
                    lut[((w1 >> 16) & 0xFF) as usize] as i32,
                    lut[(w1 >> 24) as usize] as i32,
                );
                let p = out.add(x).cast::<__m256i>();
                if STREAM {
                    _mm256_stream_si256(p, v);
                } else {
                    _mm256_storeu_si256(p, v);
                }
                x += 8;
            }
        }
        if !STREAM && lim < tw {
            decode_palette_8bpp_scalar(lut, &packed_row[lim..], &mut dst[lim..]);
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
        fn row_sub3_forward_and_sads(self, cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub3_forward_and_sads_impl(cur, dst) }
        }

        #[inline(always)]
        fn row_up_forward_and_sads_all4(
            self,
            cur: &[u8],
            prev: &[u8],
            dst_up: &mut [u8],
        ) -> [u64; 4] {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_up_forward_and_sads_all4_impl(cur, prev, dst_up) }
        }

        #[inline(always)]
        fn row_sub3_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub3_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_sub3_forward_in_place(self, buf: &mut [u8]) {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { row_sub3_forward_in_place_impl(buf) }
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
        unsafe fn row_up_subgreen_bgr3_to_bgra<
            const STREAM: bool,
            const CONST_ALPHA: bool,
            const WRITE_DEC: bool,
        >(
            self,
            filt_bgr: &[u8],
            prev_bgr: &[u8],
            cur_dec: &mut [u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // 32-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                row_up_subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA, WRITE_DEC>(
                    filt_bgr,
                    prev_bgr,
                    cur_dec,
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
        unsafe fn merge_bgrx_alpha_row<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src: &[Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            let tw = dst.len();
            debug_assert_eq!(src.len(), tw);
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // 32-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                merge_bgrx_alpha_row_ptr::<STREAM, CONST_ALPHA>(
                    src.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    dst.as_mut_ptr(),
                    tw,
                );
            }
        }

        #[inline(always)]
        fn merge_bgrx_alpha_row_in_place<const CONST_ALPHA: bool>(
            self,
            pixels: &mut [Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
        ) {
            let tw = pixels.len();
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `avx2` support, and
            // `merge_bgrx_alpha_row_ptr` loads each vector before storing back.
            unsafe {
                merge_bgrx_alpha_row_ptr::<false, CONST_ALPHA>(
                    pixels.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    pixels.as_mut_ptr().cast::<MaybeUninit<Vec4u8>>(),
                    tw,
                );
            }
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
        fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { validate_palette_indices_impl(n_colors, packed) }
        }

        #[inline(always)]
        fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
            // SAFETY: `self` witnesses `avx2` support.
            unsafe { pack_palette_indices_impl(indices, n_colors, packed) }
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
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_1bpp_row_impl::<STREAM>(c0, c1, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
            self,
            lut: &[Vec4u8; 4],
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_2bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
            self,
            lut: &PaletteLut4,
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_4bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
            self,
            lut: &[u32; MAX_PALETTE_COLORS],
            packed_row: &[u8],
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `avx2` support, and caller upholds
            // `packed_row` and `STREAM` alignment invariants.
            unsafe { decode_palette_8bpp_row_impl::<STREAM>(lut, packed_row, dst) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }

        #[inline(always)]
        unsafe fn probe_strip_activity<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> Option<bool> {
            // SAFETY: `self` witnesses AVX2; caller guarantees 16 readable pixels.
            unsafe { probe_strip_activity_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn probe_curr_strip_hash<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> (bool, Option<u64>) {
            // SAFETY: `self` witnesses AVX2; caller guarantees 16 readable pixels.
            unsafe { probe_curr_strip_hash_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn hash_prev_strip_non_uniform<const HAS_ALPHA: bool>(
            self,
            prev: *const Vec4u8,
        ) -> Option<u64> {
            // SAFETY: `self` witnesses AVX2; caller guarantees 16 readable pixels.
            unsafe { hash_prev_strip_non_uniform_impl::<HAS_ALPHA>(prev) }
        }

        #[inline(always)]
        unsafe fn score_strip_row<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev_src: *const Vec4u8,
            prev_dst: *const Vec4u8,
        ) -> i32 {
            // SAFETY: `self` witnesses AVX2; caller guarantees 16 readable pixels.
            unsafe { score_strip_row_impl::<HAS_ALPHA>(curr, prev_src, prev_dst) }
        }

        #[inline(always)]
        unsafe fn copy_bytes_stream(self, src: *const u8, dst: *mut u8, len: usize) {
            // SAFETY: `self` witnesses AVX2; caller guarantees non-overlapping ranges.
            unsafe { copy_bytes_stream_impl(src, dst, len) }
        }

        #[inline(always)]
        unsafe fn detect_scroll_phase1b<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            n_strips: usize,
            tier0_step_y: usize,
            changed_rows: &[u16; crate::filtering::scroll::MAX_STRIPS],
            changed_non_uniform: &[u16; crate::filtering::scroll::MAX_STRIPS],
            min_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
            max_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses AVX2; caller upholds band bounds invariants.
            unsafe {
                detect_scroll_phase1b_impl::<HAS_ALPHA>(
                    self,
                    aos,
                    prev,
                    width,
                    stride_px,
                    band_y0,
                    band_y1,
                    n_strips,
                    tier0_step_y,
                    changed_rows,
                    changed_non_uniform,
                    min_y,
                    max_y,
                )
            }
        }

        #[inline(always)]
        unsafe fn expand_and_verify_scroll_candidate<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            seed_sx: usize,
            dy: i32,
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses AVX2; caller upholds band bounds invariants.
            unsafe {
                expand_and_verify_scroll_candidate_impl::<HAS_ALPHA>(
                    self, aos, prev, width, stride_px, band_y0, band_y1, seed_sx, dy,
                )
            }
        }

        #[inline(always)]
        unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
        ) {
            // SAFETY: `self` witnesses this tier's SIMD support, and caller upholds
            // disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_row_band_2d_impl::<HAS_ALPHA>(self, ctx);
            }
        }

        #[inline(always)]
        unsafe fn finish_active_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
            qx_bboxes: &mut [QuadBbox],
            trigger_y: usize,
            full_count: usize,
        ) {
            if let Some(scans) = ctx.slot_scans
                && tile_hash::has_hw_aes()
            {
                // SAFETY: `self` witnesses this tier's SIMD support, `has_hw_aes()`
                // verified `aes`, and caller upholds disjoint-row shadow bounds.
                unsafe {
                    diff_and_update_active_band_aes::<HAS_ALPHA>(
                        ctx, scans, qx_bboxes, trigger_y, full_count,
                    );
                }
                return;
            }
            // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_active_band_fallback::<HAS_ALPHA>(
                    self, ctx, qx_bboxes, trigger_y, full_count,
                );
            }
        }
    }

    impl_finish_active_row_band_aes!("avx2", "avx2,aes");
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
    use std::arch::x86_64::_mm_andnot_si128;
    use std::arch::x86_64::_mm_castsi128_ps;
    use std::arch::x86_64::_mm_cmpeq_epi8;
    use std::arch::x86_64::_mm_cmpeq_epi32;
    use std::arch::x86_64::_mm_cvtsi32_si128;
    use std::arch::x86_64::_mm_cvtsi128_si32;
    use std::arch::x86_64::_mm_cvtsi128_si64;
    use std::arch::x86_64::_mm_loadl_epi64;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_madd_epi16;
    use std::arch::x86_64::_mm_maddubs_epi16;
    use std::arch::x86_64::_mm_max_epu8;
    use std::arch::x86_64::_mm_movemask_epi8;
    use std::arch::x86_64::_mm_movemask_ps;
    use std::arch::x86_64::_mm_mul_epu32;
    use std::arch::x86_64::_mm_or_si128;
    use std::arch::x86_64::_mm_packus_epi16;
    use std::arch::x86_64::_mm_sad_epu8;
    use std::arch::x86_64::_mm_set_epi8;
    use std::arch::x86_64::_mm_set_epi64x;
    use std::arch::x86_64::_mm_set1_epi8;
    use std::arch::x86_64::_mm_set1_epi16;
    use std::arch::x86_64::_mm_set1_epi32;
    use std::arch::x86_64::_mm_setr_epi8;
    use std::arch::x86_64::_mm_setr_epi32;
    use std::arch::x86_64::_mm_setzero_si128;
    use std::arch::x86_64::_mm_sfence;
    use std::arch::x86_64::_mm_shuffle_epi8;
    use std::arch::x86_64::_mm_shuffle_epi32;
    use std::arch::x86_64::_mm_slli_epi16;
    use std::arch::x86_64::_mm_slli_si128;
    use std::arch::x86_64::_mm_srli_epi16;
    use std::arch::x86_64::_mm_srli_epi64;
    use std::arch::x86_64::_mm_srli_si128;
    use std::arch::x86_64::_mm_storel_epi64;
    use std::arch::x86_64::_mm_storeu_si128;
    use std::arch::x86_64::_mm_stream_si128;
    use std::arch::x86_64::_mm_sub_epi8;
    use std::arch::x86_64::_mm_subs_epu8;
    use std::arch::x86_64::_mm_unpackhi_epi8;
    use std::arch::x86_64::_mm_unpackhi_epi16;
    use std::arch::x86_64::_mm_unpackhi_epi64;
    use std::arch::x86_64::_mm_unpacklo_epi8;
    use std::arch::x86_64::_mm_unpacklo_epi16;
    use std::arch::x86_64::_mm_unpacklo_epi64;
    use std::arch::x86_64::_mm_xor_si128;
    use std::mem::MaybeUninit;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::ActiveRowBandCtx;
    use crate::filtering::QuadBbox;
    use crate::filtering::RowKernel;
    use crate::filtering::palette::MAX_PALETTE_COLORS;
    use crate::filtering::palette::PaletteLut4;
    use crate::filtering::palette::decode_palette_1bpp_scalar;
    use crate::filtering::palette::decode_palette_2bpp_scalar;
    use crate::filtering::palette::decode_palette_4bpp_scalar;
    use crate::filtering::palette::decode_palette_8bpp_scalar;
    use crate::filtering::palette::pack_palette_indices_scalar;
    use crate::filtering::palette::validate_palette_indices_scalar;
    use crate::filtering::tile_hash;
    use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
    use crate::serialization::framing::packed_palette_indices_len;
    use crate::vec4u8::Vec4u8;

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn hsum2_epi64_128(a0: __m128i, a1: __m128i) -> (u64, u64) {
        let s01 = _mm_add_epi64(_mm_unpacklo_epi64(a0, a1), _mm_unpackhi_epi64(a0, a1));
        (
            _mm_cvtsi128_si64(s01) as u64,
            _mm_cvtsi128_si64(_mm_srli_si128::<8>(s01)) as u64,
        )
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn hsum4_epi64_128(a0: __m128i, a1: __m128i, a2: __m128i, a3: __m128i) -> [u64; 4] {
        let (s0, s1) = hsum2_epi64_128(a0, a1);
        let (s2, s3) = hsum2_epi64_128(a2, a3);
        [s0, s1, s2, s3]
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

        // Main 4-pixel loop stores 16 B (`di + 16 <= tw * 3` for `x < lim`), and
        // any trailing full 4-pixel block (`x < full4`) writes strictly in-bounds
        // 8 B + 4 B, leaving at most `0..=3` scalar tail pixels.
        let full4 = tw & !3;
        let lim = if tw >= 6 { ((tw - 2) / 4) * 4 } else { 0 };
        let mut x = 0usize;
        let mut di = 0usize;

        // SAFETY: Caller guarantees SSSE3; `di + 16 <= dst_bgr.len()` for `x < lim`,
        // `di + 12 <= dst_bgr.len()` for `x < full4`, `di + 3 <= dst_bgr.len()`
        // in the `tw & 3` tail, and `dst_alpha.len() == tw` when `HAS_ALPHA`.
        unsafe {
            let shuf_g = _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
            let pack12 = _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);
            let shuf_a = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 11, 7, 3);
            let dst_p = dst_bgr.as_mut_ptr();
            let alpha_p = dst_alpha.as_mut_ptr();

            while x < lim {
                let v = _mm_loadu_si128(src.ptr().add(x).cast::<__m128i>());
                let g = _mm_shuffle_epi8(v, shuf_g);
                let sg = _mm_sub_epi8(v, g);
                let pk = _mm_shuffle_epi8(sg, pack12);
                _mm_storeu_si128(dst_p.add(di).cast::<__m128i>(), pk);
                if HAS_ALPHA {
                    let a = _mm_cvtsi128_si32(_mm_shuffle_epi8(v, shuf_a)) as u32;
                    alpha_p.add(x).cast::<u32>().write_unaligned(a.to_le());
                }
                x += 4;
                di += 12;
            }
            if x < full4 {
                let v = _mm_loadu_si128(src.ptr().add(x).cast::<__m128i>());
                let g = _mm_shuffle_epi8(v, shuf_g);
                let sg = _mm_sub_epi8(v, g);
                let pk = _mm_shuffle_epi8(sg, pack12);
                _mm_storel_epi64(dst_p.add(di).cast::<__m128i>(), pk);
                let last4 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(pk)) as u32;
                dst_p
                    .add(di + 8)
                    .cast::<u32>()
                    .write_unaligned(last4.to_le());
                if HAS_ALPHA {
                    let a = _mm_cvtsi128_si32(_mm_shuffle_epi8(v, shuf_a)) as u32;
                    alpha_p.add(x).cast::<u32>().write_unaligned(a.to_le());
                }
                x += 4;
                di += 12;
            }
            for _ in 0..(tw & 3) {
                let px = src.ptr().add(x).read();
                let out = dst_p.add(di);
                out.write(px.0.wrapping_sub(px.1));
                out.add(1).write(px.1);
                out.add(2).write(px.2.wrapping_sub(px.1));
                if HAS_ALPHA {
                    alpha_p.add(x).write(px.3);
                }
                x += 1;
                di += 3;
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_forward_and_sads_impl(cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
        let len = cur.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: Caller guarantees SSSE3, `i + 16 <= len` in loop, and `j < len` in tail.
        unsafe {
            let zero = _mm_setzero_si128();
            let mut acc_none = zero;
            let mut acc_sub = zero;
            let mut prev = zero;
            let cur_p = cur.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let c = _mm_loadu_si128(cur_p.add(i).cast::<__m128i>());
                let sub = _mm_sub_epi8(c, _mm_alignr_epi8::<13>(c, prev));
                prev = c;
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), sub);
                acc_none = _mm_add_epi64(acc_none, _mm_sad_epu8(_mm_abs_epi8(c), zero));
                acc_sub = _mm_add_epi64(acc_sub, _mm_sad_epu8(_mm_abs_epi8(sub), zero));
                i += 16;
            }
            let (mut s_none, mut s_sub) = hsum2_epi64_128(acc_none, acc_sub);
            for j in lim..len {
                let c = *cur_p.add(j);
                let p = if j >= 3 { *cur_p.add(j - 3) } else { 0 };
                let sub = c.wrapping_sub(p);
                *dst_p.add(j) = sub;
                s_none += u64::from((c as i8).unsigned_abs());
                s_sub += u64::from((sub as i8).unsigned_abs());
            }
            (s_none, s_sub)
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_up_forward_and_sads_all4_impl(
        cur: &[u8],
        prev: &[u8],
        dst_up: &mut [u8],
    ) -> [u64; 4] {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst_up.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: Caller guarantees SSSE3, `i + 16 <= len` in loop, and `j < len` in tail.
        unsafe {
            let zero = _mm_setzero_si128();
            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            let mut prev_c = zero;
            let mut prev_up = zero;
            let cur_p = cur.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst_up.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let c = _mm_loadu_si128(cur_p.add(i).cast::<__m128i>());
                let u = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                let up = _mm_sub_epi8(c, u);
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), up);
                let sub = _mm_sub_epi8(c, _mm_alignr_epi8::<13>(c, prev_c));
                let su = _mm_sub_epi8(up, _mm_alignr_epi8::<13>(up, prev_up));
                prev_c = c;
                prev_up = up;
                acc0 = _mm_add_epi64(acc0, _mm_sad_epu8(_mm_abs_epi8(c), zero));
                acc1 = _mm_add_epi64(acc1, _mm_sad_epu8(_mm_abs_epi8(sub), zero));
                acc2 = _mm_add_epi64(acc2, _mm_sad_epu8(_mm_abs_epi8(up), zero));
                acc3 = _mm_add_epi64(acc3, _mm_sad_epu8(_mm_abs_epi8(su), zero));
                i += 16;
            }
            let mut sums = hsum4_epi64_128(acc0, acc1, acc2, acc3);
            for j in lim..len {
                let c = *cur_p.add(j);
                let up = c.wrapping_sub(*prev_p.add(j));
                *dst_p.add(j) = up;
                let (pc, pu) = if j >= 3 {
                    let cp = *cur_p.add(j - 3);
                    (cp, cp.wrapping_sub(*prev_p.add(j - 3)))
                } else {
                    (0, 0)
                };
                let sub = c.wrapping_sub(pc);
                let su = up.wrapping_sub(pu);
                sums[0] += u64::from((c as i8).unsigned_abs());
                sums[1] += u64::from((sub as i8).unsigned_abs());
                sums[2] += u64::from((up as i8).unsigned_abs());
                sums[3] += u64::from((su as i8).unsigned_abs());
            }
            sums
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_forward_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = (len / 16) * 16;
        // SAFETY: Caller guarantees SSSE3 and `i + 16 <= len`.
        unsafe {
            let mut prev = _mm_setzero_si128();
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut i = 0usize;
            while i < lim {
                let v = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let shifted = _mm_alignr_epi8::<13>(v, prev);
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_sub_epi8(v, shifted));
                prev = v;
                i += 16;
            }
        }
        for j in lim..len {
            let p = if j >= 3 { src[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_sub(p);
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_forward_in_place_impl(buf: &mut [u8]) {
        let len = buf.len();
        let mut i = len;
        // SAFETY: Caller guarantees SSSE3; backwards 16-byte steps write
        // `i..i + 16` strictly above all subsequent reads `..i` with `i >= 3`,
        // and `0..16` stays in bounds when `i >= 16`.
        unsafe {
            let p = buf.as_mut_ptr();
            while i >= 19 {
                i -= 16;
                let v = _mm_loadu_si128(p.add(i).cast::<__m128i>());
                let sh = _mm_loadu_si128(p.add(i - 3).cast::<__m128i>());
                _mm_storeu_si128(p.add(i).cast::<__m128i>(), _mm_sub_epi8(v, sh));
            }
            if i >= 16 {
                if i == 18 {
                    *p.add(17) = (*p.add(17)).wrapping_sub(*p.add(14));
                }
                if i >= 17 {
                    *p.add(16) = (*p.add(16)).wrapping_sub(*p.add(13));
                }
                let v0 = _mm_loadu_si128(p.cast::<__m128i>());
                let sh0 = _mm_slli_si128::<3>(v0);
                _mm_storeu_si128(p.cast::<__m128i>(), _mm_sub_epi8(v0, sh0));
                return;
            }
            for j in (3..i).rev() {
                *p.add(j) = (*p.add(j)).wrapping_sub(*p.add(j - 3));
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn load_12(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees SSSE3 and `p` is readable for at least 12 bytes.
        unsafe {
            let lo8 = _mm_loadl_epi64(p.cast::<__m128i>());
            let hi4 = _mm_cvtsi32_si128(p.add(8).cast::<u32>().read_unaligned() as i32);
            _mm_unpacklo_epi64(lo8, hi4)
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn store_12(p: *mut u8, v: __m128i) {
        // SAFETY: Caller guarantees SSSE3 and `p` is writable for at least 12 bytes.
        unsafe {
            _mm_storel_epi64(p.cast::<__m128i>(), v);
            let hi4 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(v)) as u32;
            p.add(8).cast::<u32>().write_unaligned(hi4.to_le());
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_sub3_inverse_impl(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let lim = if len >= 16 { ((len - 4) / 12) * 12 } else { 0 };
        // SAFETY: Caller guarantees SSSE3; `i + 16 <= len` in main loop,
        // `i + 12 <= len` in 12 B tail, and `i..len` in remainder.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
            let src_p = src.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut prev_px = _mm_setzero_si128();
            let mut i = 0usize;
            while i < lim {
                let mut v = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, prev_px);
                prev_px = _mm_shuffle_epi8(v, shuf_last3);
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), v);
                i += 12;
            }
            if i + 12 <= len {
                let mut v = load_12(src_p.add(i));
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, prev_px);
                prev_px = _mm_shuffle_epi8(v, shuf_last3);
                store_12(dst_p.add(i), v);
                i += 12;
            }
            if i < len {
                let c32 = _mm_cvtsi128_si32(prev_px) as u32;
                let mut acc = [c32 as u8, (c32 >> 8) as u8, (c32 >> 16) as u8];
                while i + 3 <= len {
                    acc[0] = acc[0].wrapping_add(*src_p.add(i));
                    *dst_p.add(i) = acc[0];
                    acc[1] = acc[1].wrapping_add(*src_p.add(i + 1));
                    *dst_p.add(i + 1) = acc[1];
                    acc[2] = acc[2].wrapping_add(*src_p.add(i + 2));
                    *dst_p.add(i + 2) = acc[2];
                    i += 3;
                }
                for (ch, slot) in acc.iter_mut().enumerate().take(len - i) {
                    *slot = slot.wrapping_add(*src_p.add(i + ch));
                    *dst_p.add(i + ch) = *slot;
                }
            }
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
        // SAFETY: Caller guarantees SSSE3; `i + 16 <= len` in main loop,
        // `i + 12 <= len` in 12 B tail, and `i + 3 <= len` in remainder.
        unsafe {
            let shuf_last3 = _mm_set_epi8(9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9);
            let src_p = src.as_ptr();
            let prev_p = prev.as_ptr();
            let dst_p = dst.as_mut_ptr();
            let mut prev_px = _mm_setzero_si128();
            let mut i = 0usize;
            while i < lim {
                let mut v = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, prev_px);
                prev_px = _mm_shuffle_epi8(v, shuf_last3);
                let u = _mm_loadu_si128(prev_p.add(i).cast::<__m128i>());
                _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), _mm_add_epi8(v, u));
                i += 12;
            }
            if i + 12 <= len {
                let mut v = load_12(src_p.add(i));
                v = _mm_add_epi8(v, _mm_slli_si128::<3>(v));
                v = _mm_add_epi8(v, _mm_slli_si128::<6>(v));
                v = _mm_add_epi8(v, prev_px);
                prev_px = _mm_shuffle_epi8(v, shuf_last3);
                let u = load_12(prev_p.add(i));
                store_12(dst_p.add(i), _mm_add_epi8(v, u));
                i += 12;
            }
            if i < len {
                let c32 = _mm_cvtsi128_si32(prev_px) as u32;
                let mut acc = [c32 as u8, (c32 >> 8) as u8, (c32 >> 16) as u8];
                while i + 3 <= len {
                    acc[0] = acc[0].wrapping_add(*src_p.add(i));
                    *dst_p.add(i) = acc[0].wrapping_add(*prev_p.add(i));
                    acc[1] = acc[1].wrapping_add(*src_p.add(i + 1));
                    *dst_p.add(i + 1) = acc[1].wrapping_add(*prev_p.add(i + 1));
                    acc[2] = acc[2].wrapping_add(*src_p.add(i + 2));
                    *dst_p.add(i + 2) = acc[2].wrapping_add(*prev_p.add(i + 2));
                    i += 3;
                }
            }
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

        let full4 = tw & !3;
        let lim = if tw >= 6 { ((tw - 2) / 4) * 4 } else { 0 };
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees SSSE3; `si + 16 <= src_bgr.len()` inside
        // `x < lim`, `si + 12 <= src_bgr.len()` in `x < full4`, `si + 3 <= src_bgr.len()`
        // in `tw & 3`, and caller upholds 16-byte alignment on `dst` when `STREAM`.
        unsafe {
            let unpack12 = _mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);
            let unpack_g = _mm_set_epi8(-1, 10, -1, 10, -1, 7, -1, 7, -1, 4, -1, 4, -1, 1, -1, 1);
            let const_alpha_vec = _mm_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a = _mm_set_epi8(3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1);
            let src_p = src_bgr.as_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim {
                let v = _mm_loadu_si128(src_p.add(si).cast::<__m128i>());
                let sg = _mm_shuffle_epi8(v, unpack12);
                let g = _mm_shuffle_epi8(v, unpack_g);
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                    _mm_shuffle_epi8(a32, shuf_a)
                };
                let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                let p = dst_p.add(x).cast::<__m128i>();
                if STREAM {
                    _mm_stream_si128(p, bgra);
                } else {
                    _mm_storeu_si128(p, bgra);
                }
                x += 4;
                si += 12;
            }
            if x < full4 {
                let v = load_12(src_p.add(si));
                let sg = _mm_shuffle_epi8(v, unpack12);
                let g = _mm_shuffle_epi8(v, unpack_g);
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                    _mm_shuffle_epi8(a32, shuf_a)
                };
                let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                let p = dst_p.add(x).cast::<__m128i>();
                if STREAM {
                    _mm_stream_si128(p, bgra);
                } else {
                    _mm_storeu_si128(p, bgra);
                }
                x += 4;
                si += 12;
            }
            for _ in 0..(tw & 3) {
                let g = *src_p.add(si + 1);
                let a = if CONST_ALPHA {
                    const_alpha
                } else {
                    *alpha_p.add(x)
                };
                dst_p.add(x).write(MaybeUninit::new(Vec4u8(
                    (*src_p.add(si)).wrapping_add(g),
                    g,
                    (*src_p.add(si + 2)).wrapping_add(g),
                    a,
                )));
                x += 1;
                si += 3;
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn row_up_subgreen_bgr3_to_bgra_impl<
        const STREAM: bool,
        const CONST_ALPHA: bool,
        const WRITE_DEC: bool,
    >(
        filt_bgr: &[u8],
        prev_bgr: &[u8],
        cur_dec: &mut [u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        let row_b = tw * 3;
        debug_assert_eq!(filt_bgr.len(), row_b);
        debug_assert_eq!(prev_bgr.len(), row_b);
        debug_assert!(!WRITE_DEC || cur_dec.len() == row_b);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
        debug_assert!(!STREAM || (dst.as_mut_ptr() as usize).is_multiple_of(16));

        let full4 = tw & !3;
        let lim = if tw >= 6 { ((tw - 2) / 4) * 4 } else { 0 };
        let mut x = 0usize;
        let mut si = 0usize;

        // SAFETY: Caller guarantees SSSE3; `si + 16 <= row_b` inside `x < lim`,
        // `si + 12 <= row_b` in `x < full4`, `si + 3 <= row_b` in `tw & 3`,
        // `cur_dec.len() == row_b` when `WRITE_DEC`, and caller upholds 16-byte
        // alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let unpack12 = _mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);
            let unpack_g = _mm_set_epi8(-1, 10, -1, 10, -1, 7, -1, 7, -1, 4, -1, 4, -1, 1, -1, 1);
            let const_alpha_vec = _mm_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a = _mm_set_epi8(3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1);
            let filt_p = filt_bgr.as_ptr();
            let prev_p = prev_bgr.as_ptr();
            let dec_p = cur_dec.as_mut_ptr();
            let alpha_p = alpha_row.as_ptr();
            let dst_p = dst.as_mut_ptr();

            while x < lim {
                let dec = _mm_add_epi8(
                    _mm_loadu_si128(filt_p.add(si).cast::<__m128i>()),
                    _mm_loadu_si128(prev_p.add(si).cast::<__m128i>()),
                );
                if WRITE_DEC {
                    _mm_storeu_si128(dec_p.add(si).cast::<__m128i>(), dec);
                }
                let sg = _mm_shuffle_epi8(dec, unpack12);
                let g = _mm_shuffle_epi8(dec, unpack_g);
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                    _mm_shuffle_epi8(a32, shuf_a)
                };
                let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                let p = dst_p.add(x).cast::<__m128i>();
                if STREAM {
                    _mm_stream_si128(p, bgra);
                } else {
                    _mm_storeu_si128(p, bgra);
                }
                x += 4;
                si += 12;
            }
            if x < full4 {
                let dec = _mm_add_epi8(load_12(filt_p.add(si)), load_12(prev_p.add(si)));
                if WRITE_DEC {
                    store_12(dec_p.add(si), dec);
                }
                let sg = _mm_shuffle_epi8(dec, unpack12);
                let g = _mm_shuffle_epi8(dec, unpack_g);
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                    _mm_shuffle_epi8(a32, shuf_a)
                };
                let bgra = _mm_or_si128(_mm_add_epi8(sg, g), alpha_vec);
                let p = dst_p.add(x).cast::<__m128i>();
                if STREAM {
                    _mm_stream_si128(p, bgra);
                } else {
                    _mm_storeu_si128(p, bgra);
                }
                x += 4;
                si += 12;
            }
            for _ in 0..(tw & 3) {
                let db = (*filt_p.add(si)).wrapping_add(*prev_p.add(si));
                let dg = (*filt_p.add(si + 1)).wrapping_add(*prev_p.add(si + 1));
                let dr = (*filt_p.add(si + 2)).wrapping_add(*prev_p.add(si + 2));
                if WRITE_DEC {
                    *dec_p.add(si) = db;
                    *dec_p.add(si + 1) = dg;
                    *dec_p.add(si + 2) = dr;
                }
                let a = if CONST_ALPHA {
                    const_alpha
                } else {
                    *alpha_p.add(x)
                };
                dst_p.add(x).write(MaybeUninit::new(Vec4u8(
                    db.wrapping_add(dg),
                    dg,
                    dr.wrapping_add(dg),
                    a,
                )));
                x += 1;
                si += 3;
            }
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
    unsafe fn merge_bgrx_alpha_row_ptr<const STREAM: bool, const CONST_ALPHA: bool>(
        src: *const Vec4u8,
        alpha_p: *const u8,
        const_alpha: u8,
        dst: *mut MaybeUninit<Vec4u8>,
        tw: usize,
    ) {
        debug_assert!(!STREAM || (dst as usize).is_multiple_of(16));
        let lim = (tw / 4) * 4;
        // SAFETY: Caller guarantees SSSE3, `src` readable and `dst` writable for
        // `tw` pixels (identical pointers permitted when `!STREAM`), `alpha_p`
        // readable for `tw` bytes when `!CONST_ALPHA`, and 16-byte alignment on
        // `dst` when `STREAM` is `true`.
        unsafe {
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let const_alpha_vec = _mm_set1_epi32((u32::from(const_alpha) << 24) as i32);
            let shuf_a = _mm_set_epi8(3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1);
            let mut x = 0usize;
            while x < lim {
                let v = _mm_loadu_si128(src.add(x).cast::<__m128i>());
                let alpha_vec = if CONST_ALPHA {
                    const_alpha_vec
                } else {
                    let a32 = _mm_cvtsi32_si128(alpha_p.add(x).cast::<i32>().read_unaligned());
                    _mm_shuffle_epi8(a32, shuf_a)
                };
                let out = _mm_or_si128(_mm_and_si128(v, rgb_mask), alpha_vec);
                let p = dst.add(x).cast::<__m128i>();
                if STREAM {
                    _mm_stream_si128(p, out);
                } else {
                    _mm_storeu_si128(p, out);
                }
                x += 4;
            }
            while x < tw {
                let px = src.add(x).read();
                let a = if CONST_ALPHA {
                    const_alpha
                } else {
                    *alpha_p.add(x)
                };
                dst.add(x)
                    .write(MaybeUninit::new(Vec4u8(px.0, px.1, px.2, a)));
                x += 1;
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn ssse3_any_nonzero<const HAS_ALPHA: bool>(
        v: __m128i,
        rgb_mask: __m128i,
        zero: __m128i,
    ) -> bool {
        let masked = if HAS_ALPHA {
            v
        } else {
            _mm_and_si128(v, rgb_mask)
        };
        _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(masked, zero))) != 0xF
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn hash_strip_ssse3<const HAS_ALPHA: bool>(
        v0: __m128i,
        v1: __m128i,
        v2: __m128i,
        v3: __m128i,
        rgb_mask: __m128i,
    ) -> u64 {
        const M: [u32; 16] = crate::filtering::scroll::STRIP_HASH_MULS;
        const ME0: [u64; 2] = [M[0] as u64, M[2] as u64];
        const MO0: [u64; 2] = [M[1] as u64, M[3] as u64];
        const ME1: [u64; 2] = [M[4] as u64, M[6] as u64];
        const MO1: [u64; 2] = [M[5] as u64, M[7] as u64];
        const ME2: [u64; 2] = [M[8] as u64, M[10] as u64];
        const MO2: [u64; 2] = [M[9] as u64, M[11] as u64];
        const ME3: [u64; 2] = [M[12] as u64, M[14] as u64];
        const MO3: [u64; 2] = [M[13] as u64, M[15] as u64];
        // SAFETY: SSE2 is mandatory on x86_64.
        unsafe {
            let (c0, c1, c2, c3) = if HAS_ALPHA {
                (v0, v1, v2, v3)
            } else {
                (
                    _mm_and_si128(v0, rgb_mask),
                    _mm_and_si128(v1, rgb_mask),
                    _mm_and_si128(v2, rgb_mask),
                    _mm_and_si128(v3, rgb_mask),
                )
            };
            let a0_lo = _mm_mul_epu32(c0, _mm_loadu_si128(ME0.as_ptr().cast::<__m128i>()));
            let a1_lo = _mm_mul_epu32(
                _mm_srli_epi64::<32>(c0),
                _mm_loadu_si128(MO0.as_ptr().cast::<__m128i>()),
            );
            let b0_lo = _mm_mul_epu32(c2, _mm_loadu_si128(ME2.as_ptr().cast::<__m128i>()));
            let b1_lo = _mm_mul_epu32(
                _mm_srli_epi64::<32>(c2),
                _mm_loadu_si128(MO2.as_ptr().cast::<__m128i>()),
            );
            let t_lo = _mm_add_epi64(_mm_xor_si128(a0_lo, b0_lo), _mm_xor_si128(a1_lo, b1_lo));

            let a0_hi = _mm_mul_epu32(c1, _mm_loadu_si128(ME1.as_ptr().cast::<__m128i>()));
            let a1_hi = _mm_mul_epu32(
                _mm_srli_epi64::<32>(c1),
                _mm_loadu_si128(MO1.as_ptr().cast::<__m128i>()),
            );
            let b0_hi = _mm_mul_epu32(c3, _mm_loadu_si128(ME3.as_ptr().cast::<__m128i>()));
            let b1_hi = _mm_mul_epu32(
                _mm_srli_epi64::<32>(c3),
                _mm_loadu_si128(MO3.as_ptr().cast::<__m128i>()),
            );
            let t_hi = _mm_add_epi64(_mm_xor_si128(a0_hi, b0_hi), _mm_xor_si128(a1_hi, b1_hi));

            let v128 = _mm_xor_si128(t_lo, t_hi);
            let x0 = _mm_cvtsi128_si64(v128) as u64;
            let x1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v128, v128)) as u64;
            crate::filtering::scroll::finalize_strip_hash(x0, x1)
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn probe_strip_activity_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> Option<bool> {
        // SAFETY: Caller guarantees SSSE3 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm_setzero_si128();
            let c0 = _mm_loadu_si128(curr.cast::<__m128i>());
            let c1 = _mm_loadu_si128(curr.add(4).cast::<__m128i>());
            let c2 = _mm_loadu_si128(curr.add(8).cast::<__m128i>());
            let c3 = _mm_loadu_si128(curr.add(12).cast::<__m128i>());
            let d0 = _mm_xor_si128(c0, _mm_loadu_si128(prev.cast::<__m128i>()));
            let d1 = _mm_xor_si128(c1, _mm_loadu_si128(prev.add(4).cast::<__m128i>()));
            let d2 = _mm_xor_si128(c2, _mm_loadu_si128(prev.add(8).cast::<__m128i>()));
            let d3 = _mm_xor_si128(c3, _mm_loadu_si128(prev.add(12).cast::<__m128i>()));
            let diff = _mm_or_si128(_mm_or_si128(d0, d1), _mm_or_si128(d2, d3));
            if !ssse3_any_nonzero::<HAS_ALPHA>(diff, rgb_mask, zero) {
                return None;
            }
            let first = _mm_shuffle_epi32::<0x00>(c0);
            let u = _mm_or_si128(
                _mm_or_si128(_mm_xor_si128(c0, first), _mm_xor_si128(c1, first)),
                _mm_or_si128(_mm_xor_si128(c2, first), _mm_xor_si128(c3, first)),
            );
            Some(ssse3_any_nonzero::<HAS_ALPHA>(u, rgb_mask, zero))
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn probe_curr_strip_hash_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> (bool, Option<u64>) {
        // SAFETY: Caller guarantees SSSE3 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm_setzero_si128();
            let c0 = _mm_loadu_si128(curr.cast::<__m128i>());
            let c1 = _mm_loadu_si128(curr.add(4).cast::<__m128i>());
            let c2 = _mm_loadu_si128(curr.add(8).cast::<__m128i>());
            let c3 = _mm_loadu_si128(curr.add(12).cast::<__m128i>());
            let d0 = _mm_xor_si128(c0, _mm_loadu_si128(prev.cast::<__m128i>()));
            let d1 = _mm_xor_si128(c1, _mm_loadu_si128(prev.add(4).cast::<__m128i>()));
            let d2 = _mm_xor_si128(c2, _mm_loadu_si128(prev.add(8).cast::<__m128i>()));
            let d3 = _mm_xor_si128(c3, _mm_loadu_si128(prev.add(12).cast::<__m128i>()));
            let diff = _mm_or_si128(_mm_or_si128(d0, d1), _mm_or_si128(d2, d3));
            if !ssse3_any_nonzero::<HAS_ALPHA>(diff, rgb_mask, zero) {
                return (false, None);
            }
            let first = _mm_shuffle_epi32::<0x00>(c0);
            let u = _mm_or_si128(
                _mm_or_si128(_mm_xor_si128(c0, first), _mm_xor_si128(c1, first)),
                _mm_or_si128(_mm_xor_si128(c2, first), _mm_xor_si128(c3, first)),
            );
            if !ssse3_any_nonzero::<HAS_ALPHA>(u, rgb_mask, zero) {
                return (true, None);
            }
            (
                true,
                Some(hash_strip_ssse3::<HAS_ALPHA>(c0, c1, c2, c3, rgb_mask)),
            )
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn hash_prev_strip_non_uniform_impl<const HAS_ALPHA: bool>(
        prev: *const Vec4u8,
    ) -> Option<u64> {
        // SAFETY: Caller guarantees SSSE3 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm_setzero_si128();
            let p0 = _mm_loadu_si128(prev.cast::<__m128i>());
            let p1 = _mm_loadu_si128(prev.add(4).cast::<__m128i>());
            let p2 = _mm_loadu_si128(prev.add(8).cast::<__m128i>());
            let p3 = _mm_loadu_si128(prev.add(12).cast::<__m128i>());
            let first = _mm_shuffle_epi32::<0x00>(p0);
            let u = _mm_or_si128(
                _mm_or_si128(_mm_xor_si128(p0, first), _mm_xor_si128(p1, first)),
                _mm_or_si128(_mm_xor_si128(p2, first), _mm_xor_si128(p3, first)),
            );
            if !ssse3_any_nonzero::<HAS_ALPHA>(u, rgb_mask, zero) {
                return None;
            }
            Some(hash_strip_ssse3::<HAS_ALPHA>(p0, p1, p2, p3, rgb_mask))
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn score_strip_row_impl<const HAS_ALPHA: bool>(
        curr: *const Vec4u8,
        prev_src: *const Vec4u8,
        prev_dst: *const Vec4u8,
    ) -> i32 {
        // SAFETY: Caller guarantees SSSE3 and 16 readable pixels.
        unsafe {
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let zero = _mm_setzero_si128();
            let c0 = _mm_loadu_si128(curr.cast::<__m128i>());
            let c1 = _mm_loadu_si128(curr.add(4).cast::<__m128i>());
            let c2 = _mm_loadu_si128(curr.add(8).cast::<__m128i>());
            let c3 = _mm_loadu_si128(curr.add(12).cast::<__m128i>());
            let eq4 = |a: __m128i, b: __m128i| -> u32 {
                let d = _mm_xor_si128(a, b);
                let dm = if HAS_ALPHA {
                    d
                } else {
                    _mm_and_si128(d, rgb_mask)
                };
                _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(dm, zero))) as u32
            };
            let eq_ps = eq4(c0, _mm_loadu_si128(prev_src.cast::<__m128i>()))
                | (eq4(c1, _mm_loadu_si128(prev_src.add(4).cast::<__m128i>())) << 4)
                | (eq4(c2, _mm_loadu_si128(prev_src.add(8).cast::<__m128i>())) << 8)
                | (eq4(c3, _mm_loadu_si128(prev_src.add(12).cast::<__m128i>())) << 12);
            if eq_ps.count_ones() < 12 {
                return -4;
            }
            let eq_p0 = eq4(c0, _mm_loadu_si128(prev_dst.cast::<__m128i>()))
                | (eq4(c1, _mm_loadu_si128(prev_dst.add(4).cast::<__m128i>())) << 4)
                | (eq4(c2, _mm_loadu_si128(prev_dst.add(8).cast::<__m128i>())) << 8)
                | (eq4(c3, _mm_loadu_si128(prev_dst.add(12).cast::<__m128i>())) << 12);
            if (eq_ps & !eq_p0) != 0 { 3 } else { 0 }
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "ssse3")]
    unsafe fn copy_lt64_ssse3(src: *const u8, dst: *mut u8, len: usize) {
        debug_assert!(len > 0 && len < 64);
        // SAFETY: SSE2 is mandatory on x86_64; `1 <= len < 64` and non-overlapping.
        unsafe {
            let mut off = 0usize;
            while off + 16 <= len {
                let v = _mm_loadu_si128(src.add(off).cast::<__m128i>());
                _mm_storeu_si128(dst.add(off).cast::<__m128i>(), v);
                off += 16;
            }
            let rem = len - off;
            if rem != 0 {
                if len >= 16 {
                    let v = _mm_loadu_si128(src.add(len - 16).cast::<__m128i>());
                    _mm_storeu_si128(dst.add(len - 16).cast::<__m128i>(), v);
                } else if rem >= 8 {
                    let a = src.cast::<u64>().read_unaligned();
                    let b = src.add(rem - 8).cast::<u64>().read_unaligned();
                    dst.cast::<u64>().write_unaligned(a);
                    dst.add(rem - 8).cast::<u64>().write_unaligned(b);
                } else if rem >= 4 {
                    let a = src.cast::<u32>().read_unaligned();
                    let b = src.add(rem - 4).cast::<u32>().read_unaligned();
                    dst.cast::<u32>().write_unaligned(a);
                    dst.add(rem - 4).cast::<u32>().write_unaligned(b);
                } else if rem >= 2 {
                    let a = src.cast::<u16>().read_unaligned();
                    let b = src.add(rem - 2).cast::<u16>().read_unaligned();
                    dst.cast::<u16>().write_unaligned(a);
                    dst.add(rem - 2).cast::<u16>().write_unaligned(b);
                } else {
                    *dst = *src;
                }
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn copy_bytes_stream_impl(src: *const u8, dst: *mut u8, len: usize) {
        if len == 0 {
            return;
        }
        // SAFETY: SSE2 is mandatory on x86_64; caller guarantees valid non-overlapping ranges.
        unsafe {
            if len < 128 {
                if len < 64 {
                    copy_lt64_ssse3(src, dst, len);
                } else {
                    let a = _mm_loadu_si128(src.cast::<__m128i>());
                    let b = _mm_loadu_si128(src.add(16).cast::<__m128i>());
                    let c = _mm_loadu_si128(src.add(32).cast::<__m128i>());
                    let d = _mm_loadu_si128(src.add(48).cast::<__m128i>());
                    let e = _mm_loadu_si128(src.add(len - 64).cast::<__m128i>());
                    let f = _mm_loadu_si128(src.add(len - 48).cast::<__m128i>());
                    let g = _mm_loadu_si128(src.add(len - 32).cast::<__m128i>());
                    let h = _mm_loadu_si128(src.add(len - 16).cast::<__m128i>());
                    _mm_storeu_si128(dst.cast::<__m128i>(), a);
                    _mm_storeu_si128(dst.add(16).cast::<__m128i>(), b);
                    _mm_storeu_si128(dst.add(32).cast::<__m128i>(), c);
                    _mm_storeu_si128(dst.add(48).cast::<__m128i>(), d);
                    _mm_storeu_si128(dst.add(len - 64).cast::<__m128i>(), e);
                    _mm_storeu_si128(dst.add(len - 48).cast::<__m128i>(), f);
                    _mm_storeu_si128(dst.add(len - 32).cast::<__m128i>(), g);
                    _mm_storeu_si128(dst.add(len - 16).cast::<__m128i>(), h);
                }
                return;
            }
            let head = (64 - (dst as usize & 63)) & 63;
            if head != 0 {
                copy_lt64_ssse3(src, dst, head);
            }
            let s = src.add(head);
            let d = dst.add(head);
            let rem = len - head;
            let lim64 = rem & !63;
            let mut i = 0usize;
            while i < lim64 {
                let v0 = _mm_loadu_si128(s.add(i).cast::<__m128i>());
                let v1 = _mm_loadu_si128(s.add(i + 16).cast::<__m128i>());
                let v2 = _mm_loadu_si128(s.add(i + 32).cast::<__m128i>());
                let v3 = _mm_loadu_si128(s.add(i + 48).cast::<__m128i>());
                _mm_stream_si128(d.add(i).cast::<__m128i>(), v0);
                _mm_stream_si128(d.add(i + 16).cast::<__m128i>(), v1);
                _mm_stream_si128(d.add(i + 32).cast::<__m128i>(), v2);
                _mm_stream_si128(d.add(i + 48).cast::<__m128i>(), v3);
                i += 64;
            }
            let tail = rem - lim64;
            if tail != 0 {
                copy_lt64_ssse3(s.add(i), d.add(i), tail);
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
    fn ssse3_px_neq_mask4<const HAS_ALPHA: bool>(
        diff: __m128i,
        rgb_mask: __m128i,
        zero: __m128i,
    ) -> u32 {
        let masked = if HAS_ALPHA {
            diff
        } else {
            _mm_and_si128(diff, rgb_mask)
        };
        let eq = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(masked, zero))) as u32;
        (!eq) & 0xF
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
        let lim16 = n & !15;
        let lim4 = n & !3;
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        // SAFETY: Caller guarantees SSSE3, `src_p` and `dst_p` are valid for
        // `n` `Vec4u8`s and non-overlapping, and all offsets remain `< n`.
        unsafe {
            let zero = _mm_setzero_si128();
            let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
            let mut i = 0usize;

            // 4-vector OR-screened scan (64 B = 16 px/iter). Keeps `s0..s3` in
            // XMM registers so a mismatch commits those exact loads without
            // ever re-reading `src_p` (TOCTOU-safe).
            while i < lim16 {
                let s0 = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let x0 = _mm_xor_si128(s0, _mm_loadu_si128(dst_p.add(i).cast::<__m128i>()));
                let s1 = _mm_loadu_si128(src_p.add(i + 4).cast::<__m128i>());
                let x1 = _mm_xor_si128(s1, _mm_loadu_si128(dst_p.add(i + 4).cast::<__m128i>()));
                let s2 = _mm_loadu_si128(src_p.add(i + 8).cast::<__m128i>());
                let x2 = _mm_xor_si128(s2, _mm_loadu_si128(dst_p.add(i + 8).cast::<__m128i>()));
                let s3 = _mm_loadu_si128(src_p.add(i + 12).cast::<__m128i>());
                let x3 = _mm_xor_si128(s3, _mm_loadu_si128(dst_p.add(i + 12).cast::<__m128i>()));
                let any = _mm_or_si128(_mm_or_si128(x0, x1), _mm_or_si128(x2, x3));
                if ssse3_px_neq_mask4::<HAS_ALPHA>(any, rgb_mask, zero) != 0 {
                    let m0 = ssse3_px_neq_mask4::<HAS_ALPHA>(x0, rgb_mask, zero);
                    let m1 = ssse3_px_neq_mask4::<HAS_ALPHA>(x1, rgb_mask, zero);
                    let m2 = ssse3_px_neq_mask4::<HAS_ALPHA>(x2, rgb_mask, zero);
                    let m3 = ssse3_px_neq_mask4::<HAS_ALPHA>(x3, rgb_mask, zero);
                    if m0 != 0 {
                        _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), s0);
                    }
                    if m1 != 0 {
                        _mm_storeu_si128(dst_p.add(i + 4).cast::<__m128i>(), s1);
                    }
                    if m2 != 0 {
                        _mm_storeu_si128(dst_p.add(i + 8).cast::<__m128i>(), s2);
                    }
                    if m3 != 0 {
                        _mm_storeu_si128(dst_p.add(i + 12).cast::<__m128i>(), s3);
                    }
                    let mask16 = m0 | (m1 << 4) | (m2 << 8) | (m3 << 12);
                    debug_assert_ne!(mask16, 0);
                    if first == usize::MAX {
                        first = i + mask16.trailing_zeros() as usize;
                    }
                    last_end = i + (32 - mask16.leading_zeros() as usize);
                }
                i += 16;
            }

            while i < lim4 {
                let s = _mm_loadu_si128(src_p.add(i).cast::<__m128i>());
                let diff = _mm_xor_si128(s, _mm_loadu_si128(dst_p.add(i).cast::<__m128i>()));
                let neq = ssse3_px_neq_mask4::<HAS_ALPHA>(diff, rgb_mask, zero);
                if neq != 0 {
                    _mm_storeu_si128(dst_p.add(i).cast::<__m128i>(), s);
                    if first == usize::MAX {
                        first = i + neq.trailing_zeros() as usize;
                    }
                    last_end = i + (32 - neq.leading_zeros() as usize);
                }
                i += 4;
            }
            for j in lim4..n {
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn validate_palette_indices_impl(n_colors: usize, packed: &[u8]) -> bool {
        let len = packed.len();
        // SAFETY: Caller guarantees SSSE3 and all loads stay in `0..len`.
        unsafe {
            let zero = _mm_setzero_si128();
            if n_colors == 3 {
                let lim = (len / 16) * 16;
                let mut acc = zero;
                let mut i = 0usize;
                while i < lim {
                    let v = _mm_loadu_si128(packed.as_ptr().add(i).cast());
                    acc = _mm_or_si128(acc, _mm_and_si128(v, _mm_srli_epi16::<1>(v)));
                    i += 16;
                }
                let masked = _mm_and_si128(acc, _mm_set1_epi8(0x55));
                if _mm_movemask_epi8(_mm_cmpeq_epi8(masked, zero)) != 0xFFFF {
                    return false;
                }
                validate_palette_indices_scalar(3, &packed[lim..])
            } else if (5..16).contains(&n_colors) {
                let lim = (len / 16) * 16;
                let mask_0f = _mm_set1_epi8(0x0F);
                let mut max_lo = zero;
                let mut max_hi = zero;
                let mut i = 0usize;
                while i < lim {
                    let v = _mm_loadu_si128(packed.as_ptr().add(i).cast());
                    max_lo = _mm_max_epu8(max_lo, _mm_and_si128(v, mask_0f));
                    max_hi = _mm_max_epu8(max_hi, _mm_andnot_si128(mask_0f, v));
                    i += 16;
                }
                let combined = _mm_max_epu8(max_lo, _mm_srli_epi16::<4>(max_hi));
                let excess = _mm_subs_epu8(combined, _mm_set1_epi8((n_colors - 1) as i8));
                if _mm_movemask_epi8(_mm_cmpeq_epi8(excess, zero)) != 0xFFFF {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else if (17..MAX_PALETTE_COLORS).contains(&n_colors) {
                let lim = (len / 16) * 16;
                let mut max_v = zero;
                let mut i = 0usize;
                while i < lim {
                    max_v = _mm_max_epu8(max_v, _mm_loadu_si128(packed.as_ptr().add(i).cast()));
                    i += 16;
                }
                let excess = _mm_subs_epu8(max_v, _mm_set1_epi8((n_colors - 1) as i8));
                if _mm_movemask_epi8(_mm_cmpeq_epi8(excess, zero)) != 0xFFFF {
                    return false;
                }
                validate_palette_indices_scalar(n_colors, &packed[lim..])
            } else {
                true
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn pack_palette_indices_impl(
        indices: &[u8],
        n_colors: usize,
        packed: &mut [u8],
    ) -> usize {
        let n = indices.len();
        let out_len = packed_palette_indices_len(n, n_colors);
        debug_assert!(packed.len() >= out_len);
        if n_colors <= 1 {
            return 0;
        }
        // SAFETY: Caller guarantees SSSE3 and `packed.len() >= out_len`.
        unsafe {
            let src = indices.as_ptr();
            let dst = packed.as_mut_ptr();
            if n_colors <= 2 {
                let lim = (n / 16) * 16;
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm_loadu_si128(src.add(i).cast());
                    let mask = _mm_movemask_epi8(_mm_slli_epi16::<7>(v)) as u16;
                    dst.add(oi).cast::<u16>().write_unaligned(mask.to_le());
                    i += 16;
                    oi += 2;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 2, &mut packed[oi..]);
                }
            } else if n_colors <= 4 {
                let lim = (n / 16) * 16;
                let mask3 = _mm_set1_epi8(3);
                let mul1 = _mm_set1_epi16(0x0401);
                let mul2 = _mm_set1_epi32(0x0010_0001);
                let shuf =
                    _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v = _mm_and_si128(_mm_loadu_si128(src.add(i).cast()), mask3);
                    let t1 = _mm_maddubs_epi16(v, mul1);
                    let t2 = _mm_madd_epi16(t1, mul2);
                    let pk = _mm_shuffle_epi8(t2, shuf);
                    dst.add(oi)
                        .cast::<u32>()
                        .write_unaligned((_mm_cvtsi128_si32(pk) as u32).to_le());
                    i += 16;
                    oi += 4;
                }
                if lim < n {
                    pack_palette_indices_scalar(&indices[lim..], 4, &mut packed[oi..]);
                }
            } else if n_colors <= 16 {
                let lim = (n / 32) * 32;
                let mask_0f = _mm_set1_epi8(0x0F);
                let mul16 = _mm_set1_epi16(0x1001);
                let mut i = 0usize;
                let mut oi = 0usize;
                while i < lim {
                    let v0 = _mm_and_si128(_mm_loadu_si128(src.add(i).cast()), mask_0f);
                    let v1 = _mm_and_si128(_mm_loadu_si128(src.add(i + 16).cast()), mask_0f);
                    let t0 = _mm_maddubs_epi16(v0, mul16);
                    let t1 = _mm_maddubs_epi16(v1, mul16);
                    _mm_storeu_si128(dst.add(oi).cast(), _mm_packus_epi16(t0, t1));
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

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn decode_palette_1bpp_row_impl<const STREAM: bool>(
        c0: Vec4u8,
        c1: Vec4u8,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(16)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 7) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((8 - (idx_pos & 7)) & 7).min(tw);
            if h != 0 {
                decode_palette_1bpp_scalar(c0, c1, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 8 <= tw {
            debug_assert_eq!(pos & 7, 0);
            let mut bi = pos >> 3;
            let end = if STREAM { tw } else { x + ((tw - x) / 8) * 8 };
            // SAFETY: Caller guarantees SSSE3, `packed` bounds, and 16-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let vc0 = _mm_set1_epi32(u32::from_le_bytes([c0.0, c0.1, c0.2, c0.3]) as i32);
                let vc1 = _mm_set1_epi32(u32::from_le_bytes([c1.0, c1.1, c1.2, c1.3]) as i32);
                let xdiff = _mm_xor_si128(vc0, vc1);
                let bits_lo = _mm_setr_epi32(1, 2, 4, 8);
                let bits_hi = _mm_setr_epi32(16, 32, 64, 128);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    let b = _mm_set1_epi32(i32::from(*src.add(bi)));
                    let m0 = _mm_cmpeq_epi32(_mm_and_si128(b, bits_lo), bits_lo);
                    let m1 = _mm_cmpeq_epi32(_mm_and_si128(b, bits_hi), bits_hi);
                    let p0 = _mm_xor_si128(vc0, _mm_and_si128(m0, xdiff));
                    let p1 = _mm_xor_si128(vc0, _mm_and_si128(m1, xdiff));
                    let p = out.add(x).cast::<__m128i>();
                    if STREAM {
                        _mm_stream_si128(p, p0);
                        _mm_stream_si128(p.add(1), p1);
                    } else {
                        _mm_storeu_si128(p, p0);
                        _mm_storeu_si128(p.add(1), p1);
                    }
                    x += 8;
                    bi += 1;
                }
            }
            pos = bi << 3;
        }
        if !STREAM && x < tw {
            decode_palette_1bpp_scalar(c0, c1, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn store_16_indexed_bgra<const STREAM: bool>(
        b_lut: __m128i,
        g_lut: __m128i,
        r_lut: __m128i,
        a_lut: __m128i,
        idx: __m128i,
        out_p: *mut __m128i,
    ) {
        // SAFETY: Caller guarantees SSSE3, `out_p` writable for 64 bytes, and
        // 16-byte alignment when `STREAM` is `true`.
        unsafe {
            let b = _mm_shuffle_epi8(b_lut, idx);
            let g = _mm_shuffle_epi8(g_lut, idx);
            let r = _mm_shuffle_epi8(r_lut, idx);
            let a = _mm_shuffle_epi8(a_lut, idx);
            let bg_lo = _mm_unpacklo_epi8(b, g);
            let bg_hi = _mm_unpackhi_epi8(b, g);
            let ra_lo = _mm_unpacklo_epi8(r, a);
            let ra_hi = _mm_unpackhi_epi8(r, a);
            let p0 = _mm_unpacklo_epi16(bg_lo, ra_lo);
            let p1 = _mm_unpackhi_epi16(bg_lo, ra_lo);
            let p2 = _mm_unpacklo_epi16(bg_hi, ra_hi);
            let p3 = _mm_unpackhi_epi16(bg_hi, ra_hi);
            if STREAM {
                _mm_stream_si128(out_p, p0);
                _mm_stream_si128(out_p.add(1), p1);
                _mm_stream_si128(out_p.add(2), p2);
                _mm_stream_si128(out_p.add(3), p3);
            } else {
                _mm_storeu_si128(out_p, p0);
                _mm_storeu_si128(out_p.add(1), p1);
                _mm_storeu_si128(out_p.add(2), p2);
                _mm_storeu_si128(out_p.add(3), p3);
            }
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn decode_palette_2bpp_row_impl<const STREAM: bool>(
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(16)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 3) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = ((4 - (idx_pos & 3)) & 3).min(tw);
            if h != 0 {
                decode_palette_2bpp_scalar(lut, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 3, 0);
            let mut bi = pos >> 2;
            let end = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees SSSE3, `packed` bounds, and 16-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let b_lut = _mm_setr_epi8(
                    lut[0].0 as i8,
                    lut[1].0 as i8,
                    lut[2].0 as i8,
                    lut[3].0 as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let g_lut = _mm_setr_epi8(
                    lut[0].1 as i8,
                    lut[1].1 as i8,
                    lut[2].1 as i8,
                    lut[3].1 as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let r_lut = _mm_setr_epi8(
                    lut[0].2 as i8,
                    lut[1].2 as i8,
                    lut[2].2 as i8,
                    lut[3].2 as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let a_lut = _mm_setr_epi8(
                    lut[0].3 as i8,
                    lut[1].3 as i8,
                    lut[2].3 as i8,
                    lut[3].3 as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let mask3 = _mm_set1_epi8(3);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    let w = u32::from_le(src.add(bi).cast::<u32>().read_unaligned());
                    let v = _mm_cvtsi32_si128(w as i32);
                    let p0 = _mm_and_si128(v, mask3);
                    let p1 = _mm_and_si128(_mm_srli_epi16::<2>(v), mask3);
                    let p2 = _mm_and_si128(_mm_srli_epi16::<4>(v), mask3);
                    let p3 = _mm_and_si128(_mm_srli_epi16::<6>(v), mask3);
                    let idx =
                        _mm_unpacklo_epi16(_mm_unpacklo_epi8(p0, p1), _mm_unpacklo_epi8(p2, p3));
                    store_16_indexed_bgra::<STREAM>(
                        b_lut,
                        g_lut,
                        r_lut,
                        a_lut,
                        idx,
                        out.add(x).cast::<__m128i>(),
                    );
                    x += 16;
                    bi += 4;
                }
            }
            pos = bi << 2;
        }
        if !STREAM && x < tw {
            decode_palette_2bpp_scalar(lut, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn decode_palette_4bpp_row_impl<const STREAM: bool>(
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert!(
            !STREAM
                || ((dst.as_mut_ptr() as usize).is_multiple_of(16)
                    && tw.is_multiple_of(16)
                    && (idx_pos & 1) == 0)
        );
        let head = if STREAM {
            0
        } else {
            let h = (idx_pos & 1).min(tw);
            if h != 0 {
                decode_palette_4bpp_scalar(&lut.colors, packed, idx_pos, &mut dst[..h]);
            }
            h
        };
        let mut x = head;
        let mut pos = idx_pos + head;
        if x + 16 <= tw {
            debug_assert_eq!(pos & 1, 0);
            let mut bi = pos >> 1;
            let end = if STREAM { tw } else { x + ((tw - x) / 16) * 16 };
            // SAFETY: Caller guarantees SSSE3, `packed` bounds, and 16-byte
            // alignment on `dst` when `STREAM` is `true`.
            unsafe {
                let b_lut = _mm_loadu_si128(lut.b.as_ptr().cast());
                let g_lut = _mm_loadu_si128(lut.g.as_ptr().cast());
                let r_lut = _mm_loadu_si128(lut.r.as_ptr().cast());
                let a_lut = _mm_loadu_si128(lut.a.as_ptr().cast());
                let mask_0f = _mm_set1_epi8(0x0F);
                let src = packed.as_ptr();
                let out = dst.as_mut_ptr();
                while x < end {
                    let raw64 = _mm_loadl_epi64(src.add(bi).cast());
                    let lo = _mm_and_si128(raw64, mask_0f);
                    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw64), mask_0f);
                    let idx = _mm_unpacklo_epi8(lo, hi);
                    store_16_indexed_bgra::<STREAM>(
                        b_lut,
                        g_lut,
                        r_lut,
                        a_lut,
                        idx,
                        out.add(x).cast::<__m128i>(),
                    );
                    x += 16;
                    bi += 8;
                }
            }
            pos = bi << 1;
        }
        if !STREAM && x < tw {
            decode_palette_4bpp_scalar(&lut.colors, packed, pos, &mut dst[x..]);
        }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn decode_palette_8bpp_row_impl<const STREAM: bool>(
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        let tw = dst.len();
        debug_assert_eq!(packed_row.len(), tw);
        debug_assert!(
            !STREAM || ((dst.as_mut_ptr() as usize).is_multiple_of(16) && tw.is_multiple_of(16))
        );
        let lim = if STREAM { tw } else { (tw / 4) * 4 };
        // SAFETY: Caller guarantees SSSE3, `packed_row.len() == tw`, and 16-byte
        // alignment on `dst` when `STREAM` is `true`.
        unsafe {
            let src = packed_row.as_ptr();
            let out = dst.as_mut_ptr();
            let mut x = 0usize;
            while x < lim {
                let w = u32::from_le(src.add(x).cast::<u32>().read_unaligned());
                let q0 = u64::from(lut[(w & 0xFF) as usize])
                    | (u64::from(lut[((w >> 8) & 0xFF) as usize]) << 32);
                let q1 = u64::from(lut[((w >> 16) & 0xFF) as usize])
                    | (u64::from(lut[(w >> 24) as usize]) << 32);
                if STREAM {
                    _mm_stream_si128(
                        out.add(x).cast::<__m128i>(),
                        _mm_set_epi64x(q1 as i64, q0 as i64),
                    );
                } else {
                    let p = out.add(x).cast::<u64>();
                    p.write_unaligned(q0);
                    p.add(1).write_unaligned(q1);
                }
                x += 4;
            }
        }
        if !STREAM && lim < tw {
            decode_palette_8bpp_scalar(lut, &packed_row[lim..], &mut dst[lim..]);
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
        fn row_sub3_forward_and_sads(self, cur: &[u8], dst: &mut [u8]) -> (u64, u64) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub3_forward_and_sads_impl(cur, dst) }
        }

        #[inline(always)]
        fn row_up_forward_and_sads_all4(
            self,
            cur: &[u8],
            prev: &[u8],
            dst_up: &mut [u8],
        ) -> [u64; 4] {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_up_forward_and_sads_all4_impl(cur, prev, dst_up) }
        }

        #[inline(always)]
        fn row_sub3_forward(self, src: &[u8], dst: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub3_forward_impl(src, dst) }
        }

        #[inline(always)]
        fn row_sub3_forward_in_place(self, buf: &mut [u8]) {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { row_sub3_forward_in_place_impl(buf) }
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
        unsafe fn row_up_subgreen_bgr3_to_bgra<
            const STREAM: bool,
            const CONST_ALPHA: bool,
            const WRITE_DEC: bool,
        >(
            self,
            filt_bgr: &[u8],
            prev_bgr: &[u8],
            cur_dec: &mut [u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // 16-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                row_up_subgreen_bgr3_to_bgra_impl::<STREAM, CONST_ALPHA, WRITE_DEC>(
                    filt_bgr,
                    prev_bgr,
                    cur_dec,
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
        unsafe fn merge_bgrx_alpha_row<const STREAM: bool, const CONST_ALPHA: bool>(
            self,
            src: &[Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            let tw = dst.len();
            debug_assert_eq!(src.len(), tw);
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // 16-byte alignment on `dst` when `STREAM` is `true`.
            unsafe {
                merge_bgrx_alpha_row_ptr::<STREAM, CONST_ALPHA>(
                    src.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    dst.as_mut_ptr(),
                    tw,
                );
            }
        }

        #[inline(always)]
        fn merge_bgrx_alpha_row_in_place<const CONST_ALPHA: bool>(
            self,
            pixels: &mut [Vec4u8],
            alpha_row: &[u8],
            const_alpha: u8,
        ) {
            let tw = pixels.len();
            debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
            // SAFETY: `self` witnesses `ssse3` support, and
            // `merge_bgrx_alpha_row_ptr` loads each vector before storing back.
            unsafe {
                merge_bgrx_alpha_row_ptr::<false, CONST_ALPHA>(
                    pixels.as_ptr(),
                    alpha_row.as_ptr(),
                    const_alpha,
                    pixels.as_mut_ptr().cast::<MaybeUninit<Vec4u8>>(),
                    tw,
                );
            }
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
        fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { validate_palette_indices_impl(n_colors, packed) }
        }

        #[inline(always)]
        fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
            // SAFETY: `self` witnesses `ssse3` support.
            unsafe { pack_palette_indices_impl(indices, n_colors, packed) }
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
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_1bpp_row_impl::<STREAM>(c0, c1, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
            self,
            lut: &[Vec4u8; 4],
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_2bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
            self,
            lut: &PaletteLut4,
            packed: &[u8],
            idx_pos: usize,
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // `packed` and `STREAM` alignment invariants.
            unsafe { decode_palette_4bpp_row_impl::<STREAM>(lut, packed, idx_pos, dst) }
        }

        #[inline(always)]
        unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
            self,
            lut: &[u32; MAX_PALETTE_COLORS],
            packed_row: &[u8],
            dst: &mut [MaybeUninit<Vec4u8>],
        ) {
            // SAFETY: `self` witnesses `ssse3` support, and caller upholds
            // `packed_row` and `STREAM` alignment invariants.
            unsafe { decode_palette_8bpp_row_impl::<STREAM>(lut, packed_row, dst) }
        }

        #[inline(always)]
        fn fence_stream(self) {
            // SAFETY: `_mm_sfence` is baseline SSE (mandatory on all x86_64 CPUs).
            unsafe { _mm_sfence() };
        }

        #[inline(always)]
        unsafe fn probe_strip_activity<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> Option<bool> {
            // SAFETY: `self` witnesses SSSE3; caller guarantees 16 readable pixels.
            unsafe { probe_strip_activity_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn probe_curr_strip_hash<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev: *const Vec4u8,
        ) -> (bool, Option<u64>) {
            // SAFETY: `self` witnesses SSSE3; caller guarantees 16 readable pixels.
            unsafe { probe_curr_strip_hash_impl::<HAS_ALPHA>(curr, prev) }
        }

        #[inline(always)]
        unsafe fn hash_prev_strip_non_uniform<const HAS_ALPHA: bool>(
            self,
            prev: *const Vec4u8,
        ) -> Option<u64> {
            // SAFETY: `self` witnesses SSSE3; caller guarantees 16 readable pixels.
            unsafe { hash_prev_strip_non_uniform_impl::<HAS_ALPHA>(prev) }
        }

        #[inline(always)]
        unsafe fn score_strip_row<const HAS_ALPHA: bool>(
            self,
            curr: *const Vec4u8,
            prev_src: *const Vec4u8,
            prev_dst: *const Vec4u8,
        ) -> i32 {
            // SAFETY: `self` witnesses SSSE3; caller guarantees 16 readable pixels.
            unsafe { score_strip_row_impl::<HAS_ALPHA>(curr, prev_src, prev_dst) }
        }

        #[inline(always)]
        unsafe fn copy_bytes_stream(self, src: *const u8, dst: *mut u8, len: usize) {
            // SAFETY: `self` witnesses SSSE3; caller guarantees non-overlapping ranges.
            unsafe { copy_bytes_stream_impl(src, dst, len) }
        }

        #[inline(always)]
        unsafe fn detect_scroll_phase1b<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            n_strips: usize,
            tier0_step_y: usize,
            changed_rows: &[u16; crate::filtering::scroll::MAX_STRIPS],
            changed_non_uniform: &[u16; crate::filtering::scroll::MAX_STRIPS],
            min_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
            max_y: &[u32; crate::filtering::scroll::MAX_STRIPS],
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses SSSE3; caller upholds band bounds invariants.
            unsafe {
                detect_scroll_phase1b_impl::<HAS_ALPHA>(
                    self,
                    aos,
                    prev,
                    width,
                    stride_px,
                    band_y0,
                    band_y1,
                    n_strips,
                    tier0_step_y,
                    changed_rows,
                    changed_non_uniform,
                    min_y,
                    max_y,
                )
            }
        }

        #[inline(always)]
        unsafe fn expand_and_verify_scroll_candidate<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
            seed_sx: usize,
            dy: i32,
        ) -> Option<(crate::serialization::framing::ScrollRect, u64)> {
            // SAFETY: `self` witnesses SSSE3; caller upholds band bounds invariants.
            unsafe {
                expand_and_verify_scroll_candidate_impl::<HAS_ALPHA>(
                    self, aos, prev, width, stride_px, band_y0, band_y1, seed_sx, dy,
                )
            }
        }

        #[inline(always)]
        unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
        ) {
            // SAFETY: `self` witnesses this tier's SIMD support, and caller upholds
            // disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_row_band_2d_impl::<HAS_ALPHA>(self, ctx);
            }
        }

        #[inline(always)]
        unsafe fn finish_active_row_band_2d<const HAS_ALPHA: bool>(
            self,
            ctx: &ActiveRowBandCtx<'_>,
            qx_bboxes: &mut [QuadBbox],
            trigger_y: usize,
            full_count: usize,
        ) {
            if let Some(scans) = ctx.slot_scans
                && tile_hash::has_hw_aes()
            {
                // SAFETY: `self` witnesses this tier's SIMD support, `has_hw_aes()`
                // verified `aes`, and caller upholds disjoint-row shadow bounds.
                unsafe {
                    diff_and_update_active_band_aes::<HAS_ALPHA>(
                        ctx, scans, qx_bboxes, trigger_y, full_count,
                    );
                }
                return;
            }
            // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
            unsafe {
                diff_and_update_active_band_fallback::<HAS_ALPHA>(
                    self, ctx, qx_bboxes, trigger_y, full_count,
                );
            }
        }
    }

    impl_finish_active_row_band_aes!("ssse3", "ssse3,aes");
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
            use crate::filtering::RowKernel;
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
            pub(super) unsafe fn pack_palette_indices(
                indices: &[u8],
                n_colors: usize,
                packed: &mut [u8],
            ) -> usize {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`.
                let kernel = unsafe { super::$tier_mod::Kernel::new_unchecked() };
                kernel.pack_palette_indices(indices, n_colors, packed)
            }

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
            #[inline(never)]
            pub(super) unsafe fn unfilter_aos_zstd_rows<
                const STREAM: bool,
                const CONST_ALPHA: bool,
            >(
                predictors: &[u8],
                filt_bgr: &[u8],
                alpha_filt: &[u8],
                const_alpha: u8,
                frame: &FrameRows,
                x0: usize,
                y0: usize,
                tw: usize,
                th: usize,
                dec_row_a: &mut [u8],
                dec_row_b: &mut [u8],
                alpha_row_a: &mut [u8],
                alpha_row_b: &mut [u8],
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld validated-frame/disjoint-rect/alignment invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::unfilter_aos_zstd_rows_with::<
                        super::$tier_mod::Kernel,
                        STREAM,
                        CONST_ALPHA,
                    >(
                        kernel,
                        predictors,
                        filt_bgr,
                        alpha_filt,
                        const_alpha,
                        frame,
                        x0,
                        y0,
                        tw,
                        th,
                        dec_row_a,
                        dec_row_b,
                        alpha_row_a,
                        alpha_row_b,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline(never)]
            pub(super) unsafe fn decode_jpeg_alpha_rows<
                const STREAM: bool,
                const CONST_ALPHA: bool,
            >(
                a_filt: &[u8],
                const_alpha: u8,
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
                // and upheld validated-frame/disjoint-rect/alignment invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::decode_jpeg_alpha_rows_with::<
                        super::$tier_mod::Kernel,
                        STREAM,
                        CONST_ALPHA,
                    >(
                        kernel,
                        a_filt,
                        const_alpha,
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
            #[inline(never)]
            pub(super) unsafe fn decode_jpeg_alpha_rows_in_place<const CONST_ALPHA: bool>(
                a_filt: &[u8],
                const_alpha: u8,
                frame: &FrameRows,
                x0: usize,
                y0: usize,
                tw: usize,
                th: usize,
                alpha_row_a: &mut Vec<u8>,
                alpha_row_b: &mut Vec<u8>,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`
                // and upheld validated-frame/initialized-BGRX/disjoint-rect invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::decode_jpeg_alpha_rows_in_place_with::<
                        super::$tier_mod::Kernel,
                        CONST_ALPHA,
                    >(
                        kernel,
                        a_filt,
                        const_alpha,
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
                let ctx = crate::filtering::ActiveRowBandCtx {
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    y0,
                    y1,
                    slot_base: 0,
                    dirty_slots,
                    slot_scans: None,
                    cache_non_empty: false,
                };
                // SAFETY: Caller verified CPU feature support and disjoint-row invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::RowKernel::diff_and_update_row_band_2d::<HAS_ALPHA>(
                        kernel, &ctx,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline(never)]
            pub(super) unsafe fn detect_scroll_in_band<const HAS_ALPHA: bool>(
                aos: BufferPointer<Vec4u8>,
                prev: &[Vec4u8],
                width: usize,
                stride_px: usize,
                band_y0: usize,
                band_y1: usize,
            ) -> Option<(ScrollRect, u64)> {
                // SAFETY: Caller verified CPU feature support and band bounds invariants.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::scroll::detect_scroll_in_band_with::<
                        super::$tier_mod::Kernel,
                        HAS_ALPHA,
                    >(kernel, aos, prev, width, stride_px, band_y0, band_y1)
                }
            }

            #[target_feature(enable = $feat)]
            #[inline(never)]
            pub(super) unsafe fn apply_scroll_rows_stream<const PREFETCH_SRC: bool>(
                base: *mut u8,
                stride: usize,
                sx: usize,
                len: usize,
                src_y0: usize,
                dst_y0: usize,
                copy_h: usize,
                rev: bool,
                disjoint: bool,
            ) {
                // SAFETY: Caller verified CPU feature support and disjoint-stripe bounds.
                unsafe {
                    let kernel = super::$tier_mod::Kernel::new_unchecked();
                    crate::filtering::scroll::apply_scroll_rows_stream_with::<
                        super::$tier_mod::Kernel,
                        PREFETCH_SRC,
                    >(
                        kernel, base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            #[inline(never)]
            pub(super) unsafe fn copy_packed_rows_stream(
                src_packed: &[u8],
                dst: &mut [u8],
                dst_stride: usize,
                x_bytes: usize,
                dst_y0: usize,
                row_bytes: usize,
                rows: usize,
            ) {
                // SAFETY: Caller verified CPU feature support via `X86Tier::detect()`.
                let kernel = unsafe { super::$tier_mod::Kernel::new_unchecked() };
                crate::filtering::scroll::copy_packed_rows_stream_with(
                    kernel, src_packed, dst, dst_stride, x_bytes, dst_y0, row_bytes, rows,
                );
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
    pub(super) fn pack_palette_indices(
        self,
        indices: &[u8],
        n_colors: usize,
        packed: &mut [u8],
    ) -> usize {
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
            Self::Scalar => ScalarKernel.pack_palette_indices(indices, n_colors, packed),
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
                dispatch_ssse3::pack_palette_indices(indices, n_colors, packed)
            },
            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )))]
            // SAFETY: `X86Tier::Avx2` is produced only after verifying `avx2`.
            Self::Avx2 => unsafe { dispatch_avx2::pack_palette_indices(indices, n_colors, packed) },
            // SAFETY: `X86Tier::Avx512` is produced only after verifying
            // `avx512bw + avx512vl + avx512vbmi`.
            Self::Avx512 => unsafe {
                dispatch_avx512::pack_palette_indices(indices, n_colors, packed)
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

    #[inline(always)]
    pub(super) const fn stream_align(self) -> usize {
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
            Self::Scalar => 1,
            #[cfg(all(
                not(target_feature = "avx2"),
                not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )),
            ))]
            Self::Ssse3 => 16,
            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )))]
            Self::Avx2 => 32,
            Self::Avx512 => 64,
        }
    }

    /// # Safety
    /// Caller must uphold [`super::unfilter_aos_zstd_rows_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn unfilter_aos_zstd_rows<const STREAM: bool, const CONST_ALPHA: bool>(
        self,
        predictors: &[u8],
        filt_bgr: &[u8],
        alpha_filt: &[u8],
        const_alpha: u8,
        frame: &FrameRows,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
        dec_row_a: &mut [u8],
        dec_row_b: &mut [u8],
        alpha_row_a: &mut [u8],
        alpha_row_b: &mut [u8],
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds validated-frame/disjoint-rect/alignment invariants.
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
                    super::unfilter_aos_zstd_rows_with::<ScalarKernel, false, CONST_ALPHA>(
                        ScalarKernel,
                        predictors,
                        filt_bgr,
                        alpha_filt,
                        const_alpha,
                        frame,
                        x0,
                        y0,
                        tw,
                        th,
                        dec_row_a,
                        dec_row_b,
                        alpha_row_a,
                        alpha_row_b,
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
                Self::Ssse3 => dispatch_ssse3::unfilter_aos_zstd_rows::<STREAM, CONST_ALPHA>(
                    predictors,
                    filt_bgr,
                    alpha_filt,
                    const_alpha,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    dec_row_a,
                    dec_row_b,
                    alpha_row_a,
                    alpha_row_b,
                ),
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => dispatch_avx2::unfilter_aos_zstd_rows::<STREAM, CONST_ALPHA>(
                    predictors,
                    filt_bgr,
                    alpha_filt,
                    const_alpha,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    dec_row_a,
                    dec_row_b,
                    alpha_row_a,
                    alpha_row_b,
                ),
                Self::Avx512 => dispatch_avx512::unfilter_aos_zstd_rows::<STREAM, CONST_ALPHA>(
                    predictors,
                    filt_bgr,
                    alpha_filt,
                    const_alpha,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    dec_row_a,
                    dec_row_b,
                    alpha_row_a,
                    alpha_row_b,
                ),
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::decode_jpeg_alpha_rows_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn decode_jpeg_alpha_rows<const STREAM: bool, const CONST_ALPHA: bool>(
        self,
        a_filt: &[u8],
        const_alpha: u8,
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
        // caller upholds validated-frame/disjoint-rect/alignment invariants.
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
                    super::decode_jpeg_alpha_rows_with::<ScalarKernel, false, CONST_ALPHA>(
                        ScalarKernel,
                        a_filt,
                        const_alpha,
                        jpeg_rows,
                        frame,
                        x0,
                        y0,
                        tw,
                        th,
                        alpha_row_a,
                        alpha_row_b,
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
                Self::Ssse3 => dispatch_ssse3::decode_jpeg_alpha_rows::<STREAM, CONST_ALPHA>(
                    a_filt,
                    const_alpha,
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
                Self::Avx2 => dispatch_avx2::decode_jpeg_alpha_rows::<STREAM, CONST_ALPHA>(
                    a_filt,
                    const_alpha,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
                Self::Avx512 => dispatch_avx512::decode_jpeg_alpha_rows::<STREAM, CONST_ALPHA>(
                    a_filt,
                    const_alpha,
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
    /// Caller must uphold [`super::decode_jpeg_alpha_rows_in_place_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn decode_jpeg_alpha_rows_in_place<const CONST_ALPHA: bool>(
        self,
        a_filt: &[u8],
        const_alpha: u8,
        frame: &FrameRows,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
        alpha_row_a: &mut Vec<u8>,
        alpha_row_b: &mut Vec<u8>,
    ) {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds validated-frame/initialized-BGRX/disjoint-rect invariants.
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
                    super::decode_jpeg_alpha_rows_in_place_with::<ScalarKernel, CONST_ALPHA>(
                        ScalarKernel,
                        a_filt,
                        const_alpha,
                        frame,
                        x0,
                        y0,
                        tw,
                        th,
                        alpha_row_a,
                        alpha_row_b,
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
                Self::Ssse3 => dispatch_ssse3::decode_jpeg_alpha_rows_in_place::<CONST_ALPHA>(
                    a_filt,
                    const_alpha,
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
                Self::Avx2 => dispatch_avx2::decode_jpeg_alpha_rows_in_place::<CONST_ALPHA>(
                    a_filt,
                    const_alpha,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    alpha_row_a,
                    alpha_row_b,
                ),
                Self::Avx512 => dispatch_avx512::decode_jpeg_alpha_rows_in_place::<CONST_ALPHA>(
                    a_filt,
                    const_alpha,
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
                    let ctx = super::ActiveRowBandCtx {
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        y0,
                        y1,
                        slot_base: 0,
                        dirty_slots,
                        slot_scans: None,
                        cache_non_empty: false,
                    };
                    super::diff_and_update_row_band_2d_with::<ScalarKernel, HAS_ALPHA>(
                        ScalarKernel,
                        &ctx,
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

    /// # Safety
    /// Caller must uphold [`super::scroll::detect_scroll_in_band_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn detect_scroll_in_band<const HAS_ALPHA: bool>(
        self,
        aos: BufferPointer<Vec4u8>,
        prev: &[Vec4u8],
        width: usize,
        stride_px: usize,
        band_y0: usize,
        band_y1: usize,
    ) -> Option<(ScrollRect, u64)> {
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds band and buffer bounds invariants.
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
                    super::scroll::detect_scroll_in_band_with::<ScalarKernel, HAS_ALPHA>(
                        ScalarKernel,
                        aos,
                        prev,
                        width,
                        stride_px,
                        band_y0,
                        band_y1,
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
                Self::Ssse3 => dispatch_ssse3::detect_scroll_in_band::<HAS_ALPHA>(
                    aos, prev, width, stride_px, band_y0, band_y1,
                ),
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => dispatch_avx2::detect_scroll_in_band::<HAS_ALPHA>(
                    aos, prev, width, stride_px, band_y0, band_y1,
                ),
                Self::Avx512 => dispatch_avx512::detect_scroll_in_band::<HAS_ALPHA>(
                    aos, prev, width, stride_px, band_y0, band_y1,
                ),
            }
        }
    }

    /// # Safety
    /// Caller must uphold [`super::scroll::apply_scroll_rows_stream_with`] invariants.
    #[inline(always)]
    pub(super) unsafe fn apply_scroll_rows_stream(
        self,
        base: *mut u8,
        stride: usize,
        sx: usize,
        len: usize,
        src_y0: usize,
        dst_y0: usize,
        copy_h: usize,
        rev: bool,
        disjoint: bool,
    ) {
        let pf_src = super::scroll::should_prefetch_scroll_stripe(stride, len);
        // SAFETY: Each arm's tier witness verifies required CPU features, and
        // caller upholds disjoint-stripe and buffer bounds invariants.
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
                Self::Scalar => super::scroll::apply_scroll_rows_temporal(
                    base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
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
                    if pf_src {
                        dispatch_ssse3::apply_scroll_rows_stream::<true>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    } else {
                        dispatch_ssse3::apply_scroll_rows_stream::<false>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    }
                },
                #[cfg(not(all(
                    target_feature = "avx512bw",
                    target_feature = "avx512vl",
                    target_feature = "avx512vbmi",
                )))]
                Self::Avx2 => {
                    if pf_src {
                        dispatch_avx2::apply_scroll_rows_stream::<true>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    } else {
                        dispatch_avx2::apply_scroll_rows_stream::<false>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    }
                },
                Self::Avx512 => {
                    if pf_src {
                        dispatch_avx512::apply_scroll_rows_stream::<true>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    } else {
                        dispatch_avx512::apply_scroll_rows_stream::<false>(
                            base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                        );
                    }
                },
            }
        }
    }

    #[inline(always)]
    pub(super) fn copy_packed_rows_stream(
        self,
        src_packed: &[u8],
        dst: &mut [u8],
        dst_stride: usize,
        x_bytes: usize,
        dst_y0: usize,
        row_bytes: usize,
        rows: usize,
    ) {
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
            Self::Scalar => super::scroll::copy_packed_rows_stream_with(
                ScalarKernel,
                src_packed,
                dst,
                dst_stride,
                x_bytes,
                dst_y0,
                row_bytes,
                rows,
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
                dispatch_ssse3::copy_packed_rows_stream(
                    src_packed, dst, dst_stride, x_bytes, dst_y0, row_bytes, rows,
                );
            },
            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )))]
            // SAFETY: `X86Tier::Avx2` is produced only after verifying `avx2`.
            Self::Avx2 => unsafe {
                dispatch_avx2::copy_packed_rows_stream(
                    src_packed, dst, dst_stride, x_bytes, dst_y0, row_bytes, rows,
                );
            },
            // SAFETY: `X86Tier::Avx512` is produced only after verifying
            // `avx512bw + avx512vl + avx512vbmi`.
            Self::Avx512 => unsafe {
                dispatch_avx512::copy_packed_rows_stream(
                    src_packed, dst, dst_stride, x_bytes, dst_y0, row_bytes, rows,
                );
            },
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
