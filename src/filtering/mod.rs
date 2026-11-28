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

//! 2D tiled pixel filtering, incremental sub-tile diffing, and lossless /
//! opt-in hybrid `TextSafe` TurboJPEG compression/decompression.
//!
//! Pixel buffers are partitioned into `256x256` tiles (with a full-width 1D
//! strip fallback for thin `< 8`-row partial bands so 1-row updates avoid
//! tile-header fragmentation). On incremental commits against a per-surface
//! shadow buffer, a parallel linear horizontal-band SIMD pass skips unchanged
//! `128x128` quadrants, crops small dirty changes (`<= 45%` of quadrant area)
//! to their tight pixel bounding box (with safe intra-tile seam merging and
//! without polluting the tile cache), and coalesces all-full-dirty quadrants
//! back into a `256x256` tile while omitting empty commits completely. Each
//! dirty tile is encoded on
//! a worker thread inside L1/L2 scratch memory as either a single
//! [`TilePayload::Leaf`] or (in [`CompressionMode::Hybrid`] when a splittable
//! tile mixes text and photographic quadrants) a non-recursive
//! [`TilePayload::Split`] of `2..=4` [`SubTile`] leaves:
//! - [`LeafPayload::Solid`] for 1-color tiles,
//! - [`LeafPayload::Palette`] (1/2/4/8 bpp + Zstd) for `<= 256` unique colors,
//! - [`LeafPayload::AosZstd`] using fused 3-byte `SubGreen(B-G, G, R-G)` AoS,
//!   per-row adaptive `{None, Sub, Up, SubUp}` prediction (2 bits/row), a
//!   single Zstd level-1 (`MinMatch(7)`) stream, and an optional separate alpha
//!   plane omitted completely on `Xrgb8888` and opaque `Argb8888` tiles,
//! - [`LeafPayload::Jpeg`] (opt-in hybrid mode only, for non-`TextSafe`
//!   photographic regions `>= 16x16`), with lossless separate alpha on
//!   translucent `Argb8888` tiles.
//!
//! Architecture-specific backends:
//! - `x86`: `cfg`-pruned runtime dispatch across AVX-512 (512-bit), AVX2
//!   (256-bit), SSSE3 (128-bit), and scalar fallback (pruning tiers strictly
//!   below the compile-time target baseline, and folding away runtime dispatch
//!   completely when compiled statically with AVX-512 or under Miri)
//! - `neon`: AArch64 NEON structure load/store (`vld4q`/`vst3q`, `vld3q`/`vst4q`)

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "x86.rs"]
mod arch;

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[path = "neon.rs"]
mod arch;

#[cfg(any(miri, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod arch {
    use std::borrow::Cow;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::AtomicUsize;

    use crate::buffer_pointer::BufferPointer;
    use crate::filtering::ClientTileCache;
    use crate::filtering::DecodeCacheCtx;
    use crate::filtering::FrameRows;
    use crate::filtering::PhaseAQuadHashes;
    use crate::filtering::PhaseASchedule;
    use crate::filtering::RowKernel;
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

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub(super) struct Tier;

    impl Tier {
        #[inline(always)]
        pub(super) const fn detect() -> Self {
            Self
        }

        #[inline(always)]
        pub(super) const fn stream_align(self) -> usize {
            1
        }

        #[inline(always)]
        pub(super) fn filter_aos_zstd_leaf<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            stride_px: usize,
            rect: TileRect,
            state: &mut CompressorWorkerState,
        ) -> LeafPayload {
            super::filter_aos_zstd_leaf_with::<ScalarKernel, HAS_ALPHA>(
                ScalarKernel,
                aos,
                stride_px,
                rect,
                state,
            )
        }

        #[inline(always)]
        pub(super) fn encode_alpha_plane(
            self,
            state: &mut CompressorWorkerState,
            tw: usize,
            th: usize,
        ) -> AlphaChannel {
            super::encode_alpha_plane_with(ScalarKernel, state, tw, th)
        }

        #[inline(always)]
        pub(super) fn pack_palette_indices(
            self,
            indices: &[u8],
            n_colors: usize,
            packed: &mut [u8],
        ) -> usize {
            ScalarKernel.pack_palette_indices(indices, n_colors, packed)
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
                super::blit_hits_worker_with(
                    ScalarKernel,
                    cache,
                    hits,
                    tile_rects,
                    frame,
                    next_hit,
                );
            }
        }

        /// # Safety
        /// Caller must uphold [`super::unfilter_aos_zstd_rows_with`] invariants.
        #[inline(never)]
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
            // SAFETY: Caller upholds validated-frame and disjoint-rect invariants;
            // `ScalarKernel::STREAM_ALIGN == 1` never issues streaming stores.
            unsafe {
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
                );
            }
        }

        /// # Safety
        /// Caller must uphold [`super::decode_jpeg_alpha_rows_with`] invariants.
        #[inline(never)]
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
            // SAFETY: Caller upholds validated-frame and disjoint-rect invariants;
            // `ScalarKernel::STREAM_ALIGN == 1` never issues streaming stores.
            unsafe {
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
                );
            }
        }

        /// # Safety
        /// Caller must uphold [`super::decode_jpeg_alpha_rows_in_place_with`] invariants.
        #[inline(never)]
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
            // SAFETY: Caller upholds validated-frame, initialized-BGRX, and disjoint-rect invariants.
            unsafe {
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
                super::copy_and_scan_rect_to_shadow_with::<ScalarKernel, HAS_ALPHA>(
                    ScalarKernel,
                    src,
                    shadow,
                    stride,
                    rect,
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
            quad_hashes: Option<&PhaseAQuadHashes>,
            schedule: &PhaseASchedule<'_>,
            worker_idx: usize,
        ) {
            // SAFETY: Caller upholds disjoint-band shadow invariants.
            unsafe {
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
                        quad_hashes,
                        schedule,
                        worker_idx,
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
                        quad_hashes,
                        schedule,
                        worker_idx,
                    );
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
            // SAFETY: Caller upholds disjoint-band and scroll-band invariants.
            unsafe {
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
                super::diff_and_update_strip_with::<ScalarKernel, HAS_ALPHA>(
                    ScalarKernel,
                    aos,
                    shadow,
                    stride_px,
                    rect,
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
            // SAFETY: Caller upholds disjoint-row shadow invariants.
            unsafe {
                super::diff_and_update_row_band_2d_with::<ScalarKernel, HAS_ALPHA>(
                    ScalarKernel,
                    &ctx,
                );
            }
        }

        /// # Safety
        /// Caller must uphold [`super::scroll::detect_scroll_in_band_with`] invariants.
        #[inline(never)]
        pub(super) unsafe fn detect_scroll_in_band<const HAS_ALPHA: bool>(
            self,
            aos: BufferPointer<Vec4u8>,
            prev: &[Vec4u8],
            width: usize,
            stride_px: usize,
            band_y0: usize,
            band_y1: usize,
        ) -> (Option<(ScrollRect, u64)>, u64) {
            // SAFETY: Caller upholds band and buffer bounds invariants.
            unsafe {
                super::scroll::detect_scroll_in_band_with::<ScalarKernel, HAS_ALPHA>(
                    ScalarKernel,
                    aos,
                    prev,
                    width,
                    stride_px,
                    band_y0,
                    band_y1,
                )
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
            // SAFETY: Caller upholds disjoint-stripe and buffer bounds invariants.
            unsafe {
                super::scroll::apply_scroll_rows_temporal(
                    base, stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint,
                );
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
            super::scroll::copy_packed_rows_stream_with(
                ScalarKernel,
                src_packed,
                dst,
                dst_stride,
                x_bytes,
                dst_y0,
                row_bytes,
                rows,
            );
        }

        #[cfg(test)]
        pub(super) fn supported_tiers() -> Vec<Self> {
            vec![Self]
        }
    }

    #[cfg(test)]
    pub(super) fn verify_all_supported_kernels() -> Vec<Tier> {
        super::tests::verify_kernel_matches_scalar(ScalarKernel, "scalar");
        vec![Tier]
    }
}

pub(crate) mod palette;
pub(crate) mod scroll;
pub(crate) mod tile_cache;
pub(crate) mod tile_hash;

/// Portable software prefetch hints for strided 2D row traversal across 4 KiB
/// page boundaries.
pub(crate) mod prefetch {
    /// Cache line stride (bytes) for per-row software prefetches.
    const CACHE_LINE_BYTES: usize = 64;

    /// Minimum row stride (bytes) before emitting row-ahead prefetches. Rows
    /// with a smaller stride share 4 KiB pages across adjacent rows where the
    /// hardware stride prefetcher tracks them without software hints.
    pub(crate) const MIN_PAGE_STRIDE_BYTES: usize = 4096;

    /// Row lookahead for Phase B strided shadow-buffer reads (`PREFETCHT0`).
    pub(crate) const PHASE_B_ROW_LOOKAHEAD: usize = 3;

    /// Row lookahead for cold-fill strided source reads (`PREFETCHNTA`).
    pub(crate) const COLD_FILL_ROW_LOOKAHEAD: usize = 2;

    /// Row lookahead for narrow vertical-stripe scroll decode reads (`PREFETCHT0`).
    pub(crate) const SCROLL_DECODE_ROW_LOOKAHEAD: usize = 2;

    #[inline(always)]
    fn read_t0(ptr: *const u8) {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        // SAFETY: SSE (`prefetcht0`) is baseline on `x86_64`; prefetch never faults or mutates memory.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr.cast::<i8>());
        }
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        // SAFETY: `prfm pldl1keep` is baseline on AArch64; prefetch never faults, writes memory, or modifies flags.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) ptr,
                options(readonly, nostack, preserves_flags),
            );
        }
        #[cfg(not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri))))]
        let _ = ptr;
    }

    #[inline(always)]
    fn read_nta(ptr: *const u8) {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        // SAFETY: SSE (`prefetchnta`) is baseline on `x86_64`; prefetch never faults or mutates memory.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_NTA }>(
                ptr.cast::<i8>(),
            );
        }
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        // SAFETY: `prfm pldl1strm` is baseline on AArch64; prefetch never faults, writes memory, or modifies flags.
        unsafe {
            std::arch::asm!(
                "prfm pldl1strm, [{p}]",
                p = in(reg) ptr,
                options(readonly, nostack, preserves_flags),
            );
        }
        #[cfg(not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri))))]
        let _ = ptr;
    }

    /// Prefetches every cache line of `len_bytes` starting at `ptr` into L1d (`T0`).
    ///
    /// # Safety
    /// `ptr` must be valid for reads of `len_bytes` bytes.
    #[inline(always)]
    pub(crate) unsafe fn row_t0(ptr: *const u8, len_bytes: usize) {
        let mut off = 0usize;
        while off < len_bytes {
            // SAFETY: `off < len_bytes`, caller guarantees `ptr` is valid for `len_bytes`.
            read_t0(unsafe { ptr.add(off) });
            off += CACHE_LINE_BYTES;
        }
    }

    /// Prefetches every cache line of `len_bytes` starting at `ptr` with a
    /// non-temporal read hint (`NTA`) so single-use source rows avoid polluting L2/L3.
    ///
    /// # Safety
    /// `ptr` must be valid for reads of `len_bytes` bytes.
    #[inline(always)]
    pub(crate) unsafe fn row_nta(ptr: *const u8, len_bytes: usize) {
        let mut off = 0usize;
        while off < len_bytes {
            // SAFETY: `off < len_bytes`, caller guarantees `ptr` is valid for `len_bytes`.
            read_nta(unsafe { ptr.add(off) });
            off += CACHE_LINE_BYTES;
        }
    }
}

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

pub use scroll::apply_scroll_rect_in_place;
pub use scroll::apply_scroll_rect_in_place_parallel;
pub use scroll::detect_scroll_rect;
pub use tile_cache::ClientTileCache;
pub use tile_cache::ServerTileCache;
use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::palette::HybridScan;
use crate::filtering::palette::MAX_PALETTE_COLORS;
use crate::filtering::palette::PaletteLut4;
use crate::filtering::palette::SUBTILE_PIXELS;
use crate::filtering::palette::decode_palette_1bpp_scalar;
use crate::filtering::palette::decode_palette_2bpp_scalar;
use crate::filtering::palette::decode_palette_4bpp_scalar;
use crate::filtering::palette::decode_palette_8bpp_scalar;
use crate::filtering::palette::decode_palette_tile;
use crate::filtering::palette::pack_palette_indices_scalar;
use crate::filtering::palette::parse_palette_header;
use crate::filtering::palette::validate_palette_indices_scalar;
use crate::filtering::tile_cache::ColdSubQuads;
use crate::filtering::tile_cache::SlotScans;
use crate::filtering::tile_cache::SyncClientSlots;
use crate::filtering::tile_cache::TileCacheAction;
use crate::filtering::tile_cache::TileEncodeMeta;
use crate::filtering::tile_cache::TileQuality;
use crate::filtering::tile_cache::ValidatedCacheHit;
use crate::filtering::tile_cache::payload_lossless_promotion_mask;
use crate::filtering::tile_hash::RectScanMeta;
use crate::filtering::tile_hash::XRGB_ALPHA_MASK_U32;
use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::framing::AlphaChannel;
use crate::serialization::framing::CacheHitRef;
use crate::serialization::framing::CompressedTile;
use crate::serialization::framing::CompressedTileStream;
use crate::serialization::framing::CompressedTiles;
use crate::serialization::framing::CompressionMode;
use crate::serialization::framing::EncodedBytes;
use crate::serialization::framing::JpegSubsamp;
use crate::serialization::framing::LeafPayload;
use crate::serialization::framing::MAX_FRAME_TILES;
use crate::serialization::framing::MAX_TILE_CACHE_BYTES;
use crate::serialization::framing::MAX_TILE_CACHE_SLOTS;
use crate::serialization::framing::RawBufferHeader;
use crate::serialization::framing::RowBand;
use crate::serialization::framing::SUBTILE_H;
use crate::serialization::framing::SUBTILE_W;
use crate::serialization::framing::ScrollRect;
use crate::serialization::framing::StreamingCompressedTiles;
use crate::serialization::framing::SubTile;
use crate::serialization::framing::TILE_H;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TILE_W;
use crate::serialization::framing::TilePayload;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::validate_split_rects;
use crate::serialization::geometry::Rectangle;
use crate::serialization::wayland::BufferFormat;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::WlSurfaceId;
use crate::sharding_compression::CompressorWorkerState;
use crate::sharding_compression::DecompressorWorkerState;
use crate::sharding_compression::ShardingCompressor;
use crate::sharding_compression::ShardingDecompressor;
use crate::sharding_compression::compress_bytes;
use crate::vec4u8::Vec4u8;

/// Minimum band height (rows) to partition into `256x256` 2D tiles. Bands
/// thinner than 8 rows fall back to full-width 1D strips (capped at
/// `TILE_PIXELS`) to avoid shattering 1-row updates into many tiny tile headers.
const MIN_2D_BAND_HEIGHT: u32 = 8;

/// Minimum width and height (pixels) for a tile or sub-tile to use TurboJPEG
/// in [`CompressionMode::Hybrid`]; smaller regions stay lossless where JPEG
/// headers would bloat the payload.
const MIN_JPEG_DIM: usize = 16;

/// Maximum tight dirty bounding-box area share (`%` of a `<= 128x128`
/// quadrant's pixel area) that emits a cropped [`TileRect`] instead of the
/// full grid-aligned quadrant.
///
/// At `45%` (`7,372` px on `128x128`, ~57 full-width rows), two-line editor
/// highlights (`128x40..52`), wheel-scroll exposed strips (`128x36..48`),
/// smooth scrolls (`128x6..12`), overlay scrollbars (`12x128`), popup borders,
/// and cursors (`20x20`) all crop tightly, while half/full-quadrant redraws,
/// tab switches, and popup bodies stay grid-aligned `128x128` / coalesced
/// `256x256` for the inter-frame tile cache.
const CROP_MAX_AREA_PCT: usize = 45;

/// Minimum cropped pixel area inside a quadrant before computing a lookup-only
/// full-quadrant keyed 128-bit hash against a non-empty [`ServerTileCache`]. Tiny
/// transient edits (`< 1,024` px: cursors, status digits, single-character
/// typing) skip hashing completely, while larger cropped strips (such as a
/// popup border returning on close) still hit cached full quadrants without
/// polluting the cache on misses.
const CROP_CACHE_LOOKUP_MIN_PIXELS: usize = 1024;

/// Maximum horizontal quadrants tracked on the stack inside a single
/// `RowBand2D` (`64 * 128 = 8192` px 8K width without heap allocation).
const STACK_QX_CAP: usize = 64;

/// Maximum per-quadrant incremental [`tile_hash::RowTileHasher`] slots stored
/// on the stack inside an active `RowBand2D` (`32 * 128 = 4096` px covers up to
/// 4K with zero heap allocation while keeping the stack frame under 4 KiB).
const STACK_HASH_CAP: usize = 32;

/// Maximum mismatched quadrants tolerated inside a hash-screened `RowBand2D`
/// before bailing out to row-major `diff_and_update_row_band_2d` for the
/// remaining columns.
pub(crate) const SCREEN_MAX_MISMATCHES: usize = 2;

/// Minimum frame byte size to enable single-threaded (`p == 1`) hash screening.
/// Below `16 MiB` (e.g. 1080p `8.3 MiB`), `shadow` stays L3-resident while
/// single-core strided reads on rotating `wl_shm` buffers lose to contiguous
/// row-major `diff_and_update`; at `>= 16 MiB` (4K `33.2 MiB`), `wl_shm + shadow`
/// spills 32 MiB L3 and screening wins even on 1 thread. Multi-threaded frames
/// (`p > 1`) benefit at all 2D sizes.
const SCREEN_1T_MIN_BYTES: usize = 16 * 1024 * 1024;

/// Maximum tiny (`< CROP_CACHE_LOOKUP_MIN_PIXELS`) cropped quadrants inside a
/// `diff_and_update_row_band_2d` band that [`sync_band_quad_hashes`] will
/// re-hash from L2-hot `shadow` to keep the band's cached hashes valid.
const MAX_RESEED_CROPPED_QUADS: usize = 4;

/// Architecture-specific row kernels operating on exact-length slices (no slack
/// bytes required).
///
/// Implementors are zero-sized capability tokens (`Copy`) whose construction
/// witnesses that the target CPU supports the corresponding ISA extension;
/// x86 SIMD kernels are private to [`arch`] and constructed only inside
/// runtime-verified `#[target_feature]` wrappers.
pub(crate) trait RowKernel: Copy {
    const STREAM_ALIGN: usize;

    fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
        self,
        src: BufferPointer<Vec4u8>,
        dst_bgr: &mut [u8],
        dst_alpha: &mut [u8],
    );

    /// Computes `(sad(cur), sad(Sub3(cur)))` in one read pass while writing
    /// `Sub3(cur)` into `dst` (for row 0, where only `None` and `Sub` are valid).
    fn row_sub3_forward_and_sads(self, cur: &[u8], dst: &mut [u8]) -> (u64, u64);

    /// Computes `[sad(None), sad(Sub), sad(Up), sad(SubUp)]` in a single read
    /// pass over `(cur, prev)` while speculatively writing `Up(cur, prev)` into
    /// `dst_up` (skipping a second pass whenever `Up` wins).
    fn row_up_forward_and_sads_all4(self, cur: &[u8], prev: &[u8], dst_up: &mut [u8]) -> [u64; 4];

    fn row_sub3_forward(self, src: &[u8], dst: &mut [u8]);

    /// Applies `Sub3` (`buf[j] -= buf[j - 3]`) in place; applied to `dst_up`
    /// from [`Self::row_up_forward_and_sads_all4`] to materialize `SubUp`
    /// without re-reading `cur` or `prev`.
    fn row_sub3_forward_in_place(self, buf: &mut [u8]);

    fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]);

    fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]);

    fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]);

    /// # Safety
    /// When `STREAM` is `true`, `dst.as_mut_ptr()` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
        self,
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    /// Fuses `Up` inverse prediction (`dec = filt_bgr + prev_bgr`) with
    /// `SubGreen` BGR3 -> BGRA unpacking in a single register pass. When
    /// `WRITE_DEC` is `true`, stores `dec` into `cur_dec` for subsequent rows;
    /// when `false`, skips writing `cur_dec` completely.
    ///
    /// # Safety
    /// When `STREAM` is `true`, `dst.as_mut_ptr()` must be aligned to `Self::STREAM_ALIGN`.
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
    );

    /// # Safety
    /// When `STREAM` is `true`, `dst.as_mut_ptr()` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn splat_row<const STREAM: bool>(self, dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8);

    /// Copies `src` into `dst` using non-temporal streaming stores bypassing
    /// L2/L3 cache (preventing RFO traffic on `wl_shm` output buffers).
    ///
    /// # Safety
    /// Caller must guarantee `src.len() == dst.len()`. When `Self::STREAM_ALIGN > 1`,
    /// `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)` and
    /// `dst.len().is_multiple_of(16)` must hold.
    unsafe fn copy_row_stream(self, src: &[Vec4u8], dst: &mut [MaybeUninit<Vec4u8>]);

    /// Replaces byte 3 of each `BGRX` pixel in `src` with either `const_alpha`
    /// (when `CONST_ALPHA` is `true`) or `alpha_row[i]` (when `CONST_ALPHA` is
    /// `false`), writing the resulting `BGRA` pixels into `dst`.
    ///
    /// # Safety
    /// Caller must guarantee `src.len() == dst.len()` and
    /// `CONST_ALPHA || alpha_row.len() == dst.len()`. When `STREAM` is `true`,
    /// `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)` must hold.
    unsafe fn merge_bgrx_alpha_row<const STREAM: bool, const CONST_ALPHA: bool>(
        self,
        src: &[Vec4u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    /// Replaces byte 3 of each initialized `BGRX` pixel in `pixels` in place
    /// with either `const_alpha` (when `CONST_ALPHA` is `true`) or
    /// `alpha_row[i]` (when `CONST_ALPHA` is `false`).
    fn merge_bgrx_alpha_row_in_place<const CONST_ALPHA: bool>(
        self,
        pixels: &mut [Vec4u8],
        alpha_row: &[u8],
        const_alpha: u8,
    );

    fn is_constant_plane(self, data: &[u8]) -> Option<u8>;

    fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]);

    fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]);

    fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]);

    /// Compares `src` (via [`BufferPointer`], never creating `&[u8]` over
    /// `wl_shm`) against `shadow` pixel-by-pixel (`BGRA` when `HAS_ALPHA`, or
    /// ignoring byte 3 when `!HAS_ALPHA`). Returns `None` without writing
    /// `shadow` when all pixels match; otherwise updates every differing pixel
    /// in `shadow` from `src` and returns `Some((first_diff_x, last_diff_x_exclusive))`
    /// spanning all differing pixel indices in `0..src.len()`.
    fn diff_and_update_row<const HAS_ALPHA: bool>(
        self,
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> Option<(usize, usize)>;

    /// Copies `n_px` 4-byte pixels from `src` into (possibly uninitialized)
    /// `dst`, checking in-register whether every canonical pixel equals
    /// `first_word` (with byte 3 normalized to `0xFF` when `!HAS_ALPHA`) and
    /// (when `HAS_ALPHA`) whether every pixel has alpha byte `0xFF`.
    ///
    /// Returns `(row_matches_first, row_all_opaque)`.
    ///
    /// # Safety
    /// `src` must be readable for `n_px * 4` bytes, `dst` must be writable for
    /// `n_px * 4` bytes, and `src` and `dst` must not overlap.
    unsafe fn copy_and_scan_row<const HAS_ALPHA: bool>(
        self,
        src: *const u8,
        dst: *mut u8,
        n_px: usize,
        first_word: u32,
    ) -> (bool, bool);

    /// Validates that every active palette index in `packed` is `< n_colors`
    /// (`2..=256`). Trailing padding bits in any partial final byte must
    /// already have been verified zero by the caller.
    fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool;

    /// Bitpacks `indices` (each element in `0..n_colors`) into `packed` for
    /// `n_colors` (`1..=256`), zero-filling any trailing padding bits and
    /// returning the number of packed bytes written.
    fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize;

    /// Decodes one 1-bpp (`2`-color) palette row (`dst.len()` pixels) starting
    /// at pixel offset `idx_pos` in `packed`.
    ///
    /// # Safety
    /// `packed` must contain at least `(idx_pos + dst.len()).div_ceil(8)` bytes.
    /// When `STREAM` is `true`, `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)`,
    /// `dst.len().is_multiple_of(16)`, and `idx_pos.is_multiple_of(8)` must hold.
    unsafe fn decode_palette_1bpp_row<const STREAM: bool>(
        self,
        c0: Vec4u8,
        c1: Vec4u8,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    /// Decodes one 2-bpp (`3..=4`-color) palette row (`dst.len()` pixels)
    /// starting at pixel offset `idx_pos` in `packed`.
    ///
    /// # Safety
    /// `packed` must contain at least `(idx_pos + dst.len()).div_ceil(4)` bytes.
    /// When `STREAM` is `true`, `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)`,
    /// `dst.len().is_multiple_of(16)`, and `idx_pos.is_multiple_of(4)` must hold.
    unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
        self,
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    /// Decodes one 4-bpp (`5..=16`-color) palette row (`dst.len()` pixels)
    /// starting at pixel offset `idx_pos` in `packed`.
    ///
    /// # Safety
    /// `packed` must contain at least `(idx_pos + dst.len()).div_ceil(2)` bytes.
    /// When `STREAM` is `true`, `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)`,
    /// `dst.len().is_multiple_of(16)`, and `idx_pos.is_multiple_of(2)` must hold.
    unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
        self,
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    /// Decodes one 8-bpp (`17..=256`-color) palette row (`dst.len()` pixels)
    /// from `packed_row` (`packed_row.len() == dst.len()`), where each `lut[i]`
    /// holds `[b, g, r, a]` in native-endian `u32` byte order.
    ///
    /// # Safety
    /// `packed_row.len() == dst.len()` must hold. When `STREAM` is `true`,
    /// `(dst.as_mut_ptr() as usize).is_multiple_of(Self::STREAM_ALIGN)` and
    /// `dst.len().is_multiple_of(16)` must hold.
    unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
        self,
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    );

    fn fence_stream(self);

    /// Zero-hash Tier-0 16-pixel (`64 B`) scroll strip activity probe.
    ///
    /// Reads `curr` strictly once through raw loads (TOCTOU-safe over `wl_shm`)
    /// and compares against `prev` (`BGRA` when `HAS_ALPHA`, ignoring byte 3
    /// when `!HAS_ALPHA`). Returns `None` when all 16 pixels match `prev`, or
    /// `Some(non_uniform)` when at least one pixel differs (`non_uniform` is
    /// `true` iff the 16 canonical `curr` pixels are not all identical).
    ///
    /// # Safety
    /// `curr` and `prev` must each be readable for at least 16 `Vec4u8` pixels.
    unsafe fn probe_strip_activity<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> Option<bool>;

    /// Phase-1b 16-pixel (`64 B`) `curr` scroll strip probe returning
    /// `(changed, Option<hash>)`.
    ///
    /// Reads `curr` strictly once (TOCTOU-safe over `wl_shm`). Returns
    /// `(false, None)` when all 16 pixels match `prev`, `(true, None)` when
    /// changed but uniform, or `(true, Some(hash))` with the canonical 64-bit
    /// strip hash matching [`Self::hash_prev_strip_non_uniform`].
    ///
    /// # Safety
    /// `curr` and `prev` must each be readable for at least 16 `Vec4u8` pixels.
    unsafe fn probe_curr_strip_hash<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> (bool, Option<u64>);

    /// Phase-1b 16-pixel (`64 B`) `prev` scroll strip probe returning `None` on
    /// uniform solid-color strips and `Some(hash)` otherwise, producing the
    /// exact same 64-bit hash as [`Self::probe_curr_strip_hash`] across all tiers.
    ///
    /// # Safety
    /// `prev` must be readable for at least 16 `Vec4u8` pixels.
    unsafe fn hash_prev_strip_non_uniform<const HAS_ALPHA: bool>(
        self,
        prev: *const Vec4u8,
    ) -> Option<u64>;

    /// Phase-2a 16-pixel (`64 B`) scroll strip match score comparing `curr`
    /// (`dst_y`) against `prev_src` (`src_y`) and `prev_dst` (`dst_y`).
    ///
    /// Reads `curr` strictly once (TOCTOU-safe). Returns `3` when `>= 12/16`
    /// pixels match `prev_src` and at least one matching pixel differs from
    /// `prev_dst`, `0` when `>= 12/16` match `prev_src` with zero moved pixels,
    /// and `-4` otherwise.
    ///
    /// # Safety
    /// `curr`, `prev_src`, and `prev_dst` must each be readable for at least 16
    /// `Vec4u8` pixels.
    unsafe fn score_strip_row<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev_src: *const Vec4u8,
        prev_dst: *const Vec4u8,
    ) -> i32;

    /// Copies `len` non-overlapping bytes from `src` to `dst` using
    /// cache-line-aligned non-temporal streaming stores on the interior (and
    /// regular stores on unaligned head/tail bytes). Caller must execute
    /// [`Self::fence_stream`] once after all streaming copies complete.
    ///
    /// # Safety
    /// `src` must be readable for `len` bytes, `dst` must be writable for `len`
    /// bytes, and `[src, src + len)` must not overlap `[dst, dst + len)`.
    unsafe fn copy_bytes_stream(self, src: *const u8, dst: *mut u8, len: usize);

    /// Runs Phase-1b scroll ruler sampling out-of-line from Tier 0 so unchanged
    /// frames never allocate the 4 KiB hash table or trigger stack probes.
    ///
    /// # Safety
    /// Caller must uphold [`scroll::detect_scroll_phase1b_with`] invariants.
    #[inline(never)]
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
        changed_rows: &[u16; scroll::MAX_STRIPS],
        changed_non_uniform: &[u16; scroll::MAX_STRIPS],
        min_y: &[u32; scroll::MAX_STRIPS],
        max_y: &[u32; scroll::MAX_STRIPS],
    ) -> Option<(ScrollRect, u64)> {
        // SAFETY: Caller upholds band and buffer bounds invariants.
        unsafe {
            scroll::detect_scroll_phase1b_with::<Self, HAS_ALPHA>(
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

    /// Runs Phase-2 scroll candidate extent expansion and verification.
    ///
    /// # Safety
    /// Caller must uphold [`scroll::expand_and_verify_candidate_with`] invariants.
    #[inline(never)]
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
    ) -> Option<(ScrollRect, u64)> {
        // SAFETY: Caller upholds band and buffer bounds invariants.
        unsafe {
            scroll::expand_and_verify_candidate_with::<Self, HAS_ALPHA>(
                self, aos, prev, width, stride_px, band_y0, band_y1, seed_sx, dy,
            )
        }
    }

    /// Scans a horizontal `<= 128`-row 2D sub-band linearly row-by-row.
    ///
    /// # Safety
    /// Rows `[y0, y1)` must be strictly in-bounds within `shadow` and disjoint
    /// from all rows concurrently accessed by other threads.
    #[inline(always)]
    unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(self, ctx: &ActiveRowBandCtx<'_>) {
        // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
        unsafe {
            diff_and_update_row_band_2d_with::<Self, HAS_ALPHA>(self, ctx);
        }
    }

    /// Finishes a `RowBand2D` from trigger row `trigger_y` through `ctx.y1`
    /// once at least one quadrant has either escalated to `bbox.full` or
    /// crossed `CROP_CACHE_LOOKUP_MIN_PIXELS`.
    ///
    /// # Safety
    /// Rows `[ctx.y0, ctx.y1)` must be strictly in-bounds within `ctx.shadow`
    /// and exclusively owned by this worker, with rows `[ctx.y0, trigger_y]`
    /// already updated in `ctx.shadow`.
    #[inline(always)]
    unsafe fn finish_active_row_band_2d<const HAS_ALPHA: bool>(
        self,
        ctx: &ActiveRowBandCtx<'_>,
        qx_bboxes: &mut [QuadBbox],
        trigger_y: usize,
        full_count: usize,
    ) {
        // SAFETY: Caller upholds disjoint-row shadow bounds invariants.
        unsafe {
            finish_active_row_band_2d_with::<Self, HAS_ALPHA>(
                self, ctx, qx_bboxes, trigger_y, full_count,
            );
        }
    }

    /// Screens a grid-aligned `RowBand2D` against `prev_hashes` by hashing only
    /// `ctx.data` (`wl_shm`), diffing mismatched quadrants immediately while
    /// L2-hot and bailing out to [`Self::diff_and_update_row_band_2d`] once
    /// more than [`SCREEN_MAX_MISMATCHES`] quadrants differ. Returns `true` iff
    /// every entry in `prev_hashes` matches finalized `ctx.shadow` upon return.
    ///
    /// # Safety
    /// Rows `[ctx.y0, ctx.y1)` must be strictly in-bounds within `ctx.shadow`
    /// and exclusively owned by this worker, `ctx.slot_scans` must match
    /// `scans`, and `prev_hashes.len() == ctx.dirty_slots.len()`.
    #[inline(always)]
    unsafe fn screen_and_diff_row_band_2d<const HAS_ALPHA: bool>(
        self,
        ctx: &ActiveRowBandCtx<'_>,
        scans: &SlotScans,
        prev_hashes: &mut [tile_hash::TileHash],
    ) -> bool {
        // SAFETY: Caller upholds disjoint-row shadow bounds and hash-slice invariants.
        unsafe {
            screen_and_diff_row_band_2d_with::<Self, HAS_ALPHA>(self, ctx, scans, prev_hashes)
        }
    }
}

/// Portable scalar reference implementation of [`RowKernel`], also used as the
/// baseline `x86-64` (SSE2-only) and Miri backend.
#[cfg_attr(
    not(any(
        test,
        miri,
        all(
            target_arch = "x86_64",
            not(target_feature = "ssse3"),
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
    )),
    allow(dead_code)
)]
#[derive(Copy, Clone)]
pub(crate) struct ScalarKernel;

impl RowKernel for ScalarKernel {
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
        for (i, bgr) in dst_bgr.as_chunks_mut::<3>().0.iter_mut().enumerate() {
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
        let mut s_none = 0u64;
        let mut s_sub = 0u64;
        for j in 0..len {
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
        let mut s_none = 0u64;
        let mut s_sub = 0u64;
        let mut s_up = 0u64;
        let mut s_su = 0u64;
        for j in 0..len {
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
        for j in 0..len {
            let p = if j >= 3 { src[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_sub(p);
        }
    }

    #[inline(always)]
    fn row_sub3_forward_in_place(self, buf: &mut [u8]) {
        for j in (3..buf.len()).rev() {
            buf[j] = buf[j].wrapping_sub(buf[j - 3]);
        }
    }

    #[inline(always)]
    fn row_sub3_inverse(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            let p = if j >= 3 { dst[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_add(p);
        }
    }

    #[inline(always)]
    fn row_up_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            dst[j] = src[j].wrapping_add(prev[j]);
        }
    }

    #[inline(always)]
    fn row_subup3_inverse(self, src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        debug_assert!(len.is_multiple_of(3));
        let mut acc = [0u8; 3];
        for j in 0..len {
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
        for (i, out) in dst.iter_mut().enumerate() {
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
        let tw = dst.len();
        let row_b = tw * 3;
        debug_assert_eq!(filt_bgr.len(), row_b);
        debug_assert_eq!(prev_bgr.len(), row_b);
        debug_assert!(!WRITE_DEC || cur_dec.len() == row_b);
        debug_assert!(CONST_ALPHA || alpha_row.len() == tw);
        for (i, out) in dst.iter_mut().enumerate() {
            let s = i * 3;
            let db = filt_bgr[s].wrapping_add(prev_bgr[s]);
            let dg = filt_bgr[s + 1].wrapping_add(prev_bgr[s + 1]);
            let dr = filt_bgr[s + 2].wrapping_add(prev_bgr[s + 2]);
            if WRITE_DEC {
                cur_dec[s] = db;
                cur_dec[s + 1] = dg;
                cur_dec[s + 2] = dr;
            }
            let a = if CONST_ALPHA {
                const_alpha
            } else {
                alpha_row[i]
            };
            out.write(Vec4u8(db.wrapping_add(dg), dg, dr.wrapping_add(dg), a));
        }
    }

    #[inline(always)]
    unsafe fn splat_row<const STREAM: bool>(self, dst: &mut [MaybeUninit<Vec4u8>], pixel: Vec4u8) {
        dst.fill(MaybeUninit::new(pixel));
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
        for (i, (out, &px)) in dst.iter_mut().zip(src).enumerate() {
            let a = if CONST_ALPHA {
                const_alpha
            } else {
                alpha_row[i]
            };
            out.write(Vec4u8(px.0, px.1, px.2, a));
        }
    }

    #[inline(always)]
    fn merge_bgrx_alpha_row_in_place<const CONST_ALPHA: bool>(
        self,
        pixels: &mut [Vec4u8],
        alpha_row: &[u8],
        const_alpha: u8,
    ) {
        debug_assert!(CONST_ALPHA || alpha_row.len() == pixels.len());
        if CONST_ALPHA {
            for px in pixels {
                px.3 = const_alpha;
            }
        } else {
            for (px, &a) in pixels.iter_mut().zip(alpha_row) {
                px.3 = a;
            }
        }
    }

    #[inline(always)]
    fn is_constant_plane(self, data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        data[1..].iter().all(|&b| b == first).then_some(first)
    }

    #[inline(always)]
    fn row_sub1_forward(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let mut prev = 0u8;
        for j in 0..len {
            let cur = src[j];
            dst[j] = cur.wrapping_sub(prev);
            prev = cur;
        }
    }

    #[inline(always)]
    fn row_up_forward(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            dst[j] = cur[j].wrapping_sub(prev[j]);
        }
    }

    #[inline(always)]
    fn row_sub1_inverse(self, src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let mut acc = 0u8;
        for j in 0..len {
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
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        let mut first = usize::MAX;
        let mut last_end = 0usize;
        for (j, dst) in shadow.iter_mut().enumerate() {
            // SAFETY: `j < shadow.len() == src.len()`.
            let s = unsafe { src_p.add(j).read() };
            let d = *dst;
            let differs = if HAS_ALPHA {
                s != d
            } else {
                s.0 != d.0 || s.1 != d.1 || s.2 != d.2
            };
            if differs {
                *dst = s;
                if first == usize::MAX {
                    first = j;
                }
                last_end = j + 1;
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
        let mut diff = 0u32;
        let mut alpha_and = u32::MAX;
        for j in 0..n_px {
            let off = j * 4;
            // SAFETY: Caller guarantees `src` readable and `dst` writable for
            // `n_px * 4` bytes and non-overlapping.
            let raw = unsafe {
                let w = src.add(off).cast::<u32>().read_unaligned();
                dst.add(off).cast::<u32>().write_unaligned(w);
                w
            };
            let canon = if HAS_ALPHA {
                raw
            } else {
                raw | XRGB_ALPHA_MASK_U32
            };
            diff |= canon ^ first_word;
            if HAS_ALPHA {
                alpha_and &= canon;
            }
        }
        (
            diff == 0,
            !HAS_ALPHA || (alpha_and & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32,
        )
    }

    #[inline(always)]
    fn validate_palette_indices(self, n_colors: usize, packed: &[u8]) -> bool {
        validate_palette_indices_scalar(n_colors, packed)
    }

    #[inline(always)]
    fn pack_palette_indices(self, indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
        pack_palette_indices_scalar(indices, n_colors, packed)
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
        decode_palette_1bpp_scalar(c0, c1, packed, idx_pos, dst);
    }

    #[inline(always)]
    unsafe fn decode_palette_2bpp_row<const STREAM: bool>(
        self,
        lut: &[Vec4u8; 4],
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        decode_palette_2bpp_scalar(lut, packed, idx_pos, dst);
    }

    #[inline(always)]
    unsafe fn decode_palette_4bpp_row<const STREAM: bool>(
        self,
        lut: &PaletteLut4,
        packed: &[u8],
        idx_pos: usize,
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        decode_palette_4bpp_scalar(&lut.colors, packed, idx_pos, dst);
    }

    #[inline(always)]
    unsafe fn decode_palette_8bpp_row<const STREAM: bool>(
        self,
        lut: &[u32; MAX_PALETTE_COLORS],
        packed_row: &[u8],
        dst: &mut [MaybeUninit<Vec4u8>],
    ) {
        decode_palette_8bpp_scalar(lut, packed_row, dst);
    }

    #[inline(always)]
    fn fence_stream(self) {}

    #[inline(always)]
    unsafe fn probe_strip_activity<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> Option<bool> {
        let mask = if HAS_ALPHA { u32::MAX } else { 0x00FF_FFFF };
        let mut c_words = [0u32; scroll::STRIP_W];
        let mut changed_diff = 0u32;
        for (i, slot) in c_words.iter_mut().enumerate() {
            // SAFETY: Caller guarantees `curr` and `prev` are readable for 16 pixels.
            let cpx = unsafe { curr.add(i).read() };
            let ppx = unsafe { prev.add(i).read() };
            let cw = u32::from_le_bytes([cpx.0, cpx.1, cpx.2, cpx.3]) & mask;
            let pw = u32::from_le_bytes([ppx.0, ppx.1, ppx.2, ppx.3]) & mask;
            *slot = cw;
            changed_diff |= cw ^ pw;
        }
        if changed_diff == 0 {
            return None;
        }
        let c0 = c_words[0];
        let mut uniform_diff = 0u32;
        for &cw in &c_words[1..] {
            uniform_diff |= cw ^ c0;
        }
        Some(uniform_diff != 0)
    }

    #[inline(always)]
    unsafe fn probe_curr_strip_hash<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev: *const Vec4u8,
    ) -> (bool, Option<u64>) {
        let mask = if HAS_ALPHA { u32::MAX } else { 0x00FF_FFFF };
        let mut c_words = [0u32; scroll::STRIP_W];
        let mut changed_diff = 0u32;
        for (i, slot) in c_words.iter_mut().enumerate() {
            // SAFETY: Caller guarantees `curr` and `prev` are readable for 16 pixels.
            let cpx = unsafe { curr.add(i).read() };
            let ppx = unsafe { prev.add(i).read() };
            let cw = u32::from_le_bytes([cpx.0, cpx.1, cpx.2, cpx.3]) & mask;
            let pw = u32::from_le_bytes([ppx.0, ppx.1, ppx.2, ppx.3]) & mask;
            *slot = cw;
            changed_diff |= cw ^ pw;
        }
        if changed_diff == 0 {
            return (false, None);
        }
        let c0 = c_words[0];
        let mut uniform_diff = 0u32;
        for &cw in &c_words[1..] {
            uniform_diff |= cw ^ c0;
        }
        if uniform_diff == 0 {
            return (true, None);
        }
        (true, Some(hash_strip_words_scalar(&c_words)))
    }

    #[inline(always)]
    unsafe fn hash_prev_strip_non_uniform<const HAS_ALPHA: bool>(
        self,
        prev: *const Vec4u8,
    ) -> Option<u64> {
        let mask = if HAS_ALPHA { u32::MAX } else { 0x00FF_FFFF };
        let mut p_words = [0u32; scroll::STRIP_W];
        for (i, slot) in p_words.iter_mut().enumerate() {
            // SAFETY: Caller guarantees `prev` is readable for 16 pixels.
            let ppx = unsafe { prev.add(i).read() };
            *slot = u32::from_le_bytes([ppx.0, ppx.1, ppx.2, ppx.3]) & mask;
        }
        let p0 = p_words[0];
        let mut uniform_diff = 0u32;
        for &pw in &p_words[1..] {
            uniform_diff |= pw ^ p0;
        }
        (uniform_diff != 0).then(|| hash_strip_words_scalar(&p_words))
    }

    #[inline(always)]
    unsafe fn score_strip_row<const HAS_ALPHA: bool>(
        self,
        curr: *const Vec4u8,
        prev_src: *const Vec4u8,
        prev_dst: *const Vec4u8,
    ) -> i32 {
        let mask = if HAS_ALPHA { u32::MAX } else { 0x00FF_FFFF };
        let mut ok = 0usize;
        let mut moved = 0usize;
        for i in 0..scroll::STRIP_W {
            // SAFETY: Caller guarantees `curr`, `prev_src`, `prev_dst` are readable for 16 pixels.
            let cpx = unsafe { curr.add(i).read() };
            let pspx = unsafe { prev_src.add(i).read() };
            let c = u32::from_le_bytes([cpx.0, cpx.1, cpx.2, cpx.3]) & mask;
            let ps = u32::from_le_bytes([pspx.0, pspx.1, pspx.2, pspx.3]) & mask;
            if c == ps {
                ok += 1;
                let p0px = unsafe { prev_dst.add(i).read() };
                let p0 = u32::from_le_bytes([p0px.0, p0px.1, p0px.2, p0px.3]) & mask;
                if c != p0 {
                    moved += 1;
                }
            }
        }
        if ok * 4 >= scroll::STRIP_W * 3 {
            if moved > 0 { 3 } else { 0 }
        } else {
            -4
        }
    }

    #[inline(always)]
    unsafe fn copy_bytes_stream(self, src: *const u8, dst: *mut u8, len: usize) {
        // SAFETY: Caller guarantees non-overlapping readable `src` and writable `dst`.
        unsafe { ptr::copy_nonoverlapping(src, dst, len) };
    }
}

#[cfg_attr(
    not(any(
        test,
        miri,
        all(
            target_arch = "x86_64",
            not(target_feature = "ssse3"),
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
    )),
    allow(dead_code)
)]
#[inline(always)]
fn hash_strip_words_scalar(p: &[u32; scroll::STRIP_W]) -> u64 {
    let m = &scroll::STRIP_HASH_MULS;
    let mut t = [0u64; 4];
    for (l, slot) in t.iter_mut().enumerate() {
        let a0 = u64::from(p[2 * l]).wrapping_mul(u64::from(m[2 * l]));
        let a1 = u64::from(p[2 * l + 1]).wrapping_mul(u64::from(m[2 * l + 1]));
        let b0 = u64::from(p[8 + 2 * l]).wrapping_mul(u64::from(m[8 + 2 * l]));
        let b1 = u64::from(p[8 + 2 * l + 1]).wrapping_mul(u64::from(m[8 + 2 * l + 1]));
        *slot = (a0 ^ b0).wrapping_add(a1 ^ b1);
    }
    scroll::finalize_strip_hash(t[0] ^ t[2], t[1] ^ t[3])
}

#[inline]
fn predictor_at(predictors: &[u8], ry: usize) -> u8 {
    (predictors[ry >> 2] >> ((ry & 3) * 2)) & 3
}

/// Ensures `buf` has length at least `needed` without shrinking existing capacity.
#[inline]
fn ensure_len(buf: &mut Vec<u8>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0);
    }
}

// --- fused 2D tile forward + compress ---

/// Selects the minimum-SAD row predictor (`0=None, 1=Sub, 2=Up, 3=SubUp`) for
/// `cur_raw` (using `prev_raw` when `ry > 0`), writes the winning filtered
/// bytes into `filt_row`, and returns the chosen 2-bit predictor tag.
#[inline(always)]
fn select_and_filter_row<K: RowKernel>(
    kernel: K,
    ry: usize,
    cur_raw: &[u8],
    prev_raw: &[u8],
    filt_row: &mut [u8],
) -> u8 {
    if ry == 0 {
        let (cost_none, cost_sub) = kernel.row_sub3_forward_and_sads(cur_raw, filt_row);
        if cost_sub < cost_none {
            return 1;
        }
        filt_row.copy_from_slice(cur_raw);
        return 0;
    }

    let [cost_none, cost_sub, cost_up, cost_su] =
        kernel.row_up_forward_and_sads_all4(cur_raw, prev_raw, filt_row);
    let mut best_p = 0u8;
    let mut best_cost = cost_none;
    if cost_sub < best_cost {
        best_cost = cost_sub;
        best_p = 1;
    }
    if cost_up < best_cost {
        best_cost = cost_up;
        best_p = 2;
    }
    if cost_su < best_cost {
        best_p = 3;
    }
    match best_p {
        0 => filt_row.copy_from_slice(cur_raw),
        1 => kernel.row_sub3_forward(cur_raw, filt_row),
        2 => {},
        _ => kernel.row_sub3_forward_in_place(filt_row),
    }
    best_p
}

/// Splits a 2D base tile (`w <= TILE_W`, `h <= TILE_H`) into 1..=4 clipped
/// `128x128` quadrants in raster order.
#[inline]
pub(crate) fn tile_quadrants(rect: TileRect) -> ([TileRect; 4], usize) {
    debug_assert!(rect.w <= TILE_W as u32 && rect.h <= TILE_H as u32);
    let mut out = [rect; 4];
    let mut n = 0;
    let sub_w = SUBTILE_W as u32;
    let sub_h = SUBTILE_H as u32;
    let mut dy = 0u32;
    while dy < rect.h {
        let h = (rect.h - dy).min(sub_h);
        let mut dx = 0u32;
        while dx < rect.w {
            let w = (rect.w - dx).min(sub_w);
            out[n] = TileRect {
                x: rect.x + dx,
                y: rect.y + dy,
                w,
                h,
            };
            n += 1;
            dx += w;
        }
        dy += h;
    }
    (out, n)
}

fn encode_palette_from_parts<const HAS_ALPHA: bool>(
    palette: &[u32],
    packed: &[u8],
    all_opaque: bool,
    compressor: &mut Compressor<'static>,
    zstd_scratch: &mut Vec<u8>,
) -> LeafPayload {
    let n_colors = palette.len();
    debug_assert!((2..=256).contains(&n_colors));
    let pal_has_alpha = HAS_ALPHA && !all_opaque && palette.iter().any(|&c| (c >> 24) != 0xFF);
    let entry_bytes = if pal_has_alpha { 4 } else { 3 };
    let mut palette_bytes = Vec::with_capacity(n_colors * entry_bytes);
    if pal_has_alpha {
        for &c in palette {
            palette_bytes.extend_from_slice(&c.to_le_bytes());
        }
    } else {
        for &c in palette {
            palette_bytes.extend_from_slice(&c.to_le_bytes()[..3]);
        }
    }
    let data = compress_bytes(compressor, zstd_scratch, packed);
    LeafPayload::Palette {
        has_alpha: pal_has_alpha,
        palette: palette_bytes,
        data,
    }
}

fn try_encode_palette_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
    all_opaque: bool,
    state: &mut CompressorWorkerState,
) -> Option<LeafPayload> {
    let Ok(n_colors) = state
        .palette
        .try_encode_2d::<HAS_ALPHA>(aos, stride_px, rect)
    else {
        return None;
    };
    if n_colors == 1 {
        return Some(LeafPayload::Solid(state.palette.palette[0].to_le_bytes()));
    }
    let n_px = rect.w as usize * rect.h as usize;
    state.palette.pack_indices_range(tier, 0, n_px, n_colors);
    Some(encode_palette_from_parts::<HAS_ALPHA>(
        &state.palette.palette[..n_colors],
        state.palette.packed_indices(0, n_px, n_colors),
        all_opaque,
        &mut state.compressor,
        &mut state.zstd_scratch,
    ))
}

#[inline(always)]
fn encode_alpha_plane_with<K: RowKernel>(
    kernel: K,
    state: &mut CompressorWorkerState,
    tw: usize,
    th: usize,
) -> AlphaChannel {
    let n_px = tw * th;
    let raw_alpha = &state.raw_alpha[..n_px];
    match kernel.is_constant_plane(raw_alpha) {
        Some(0xFF) => AlphaChannel::Omitted,
        Some(a) => AlphaChannel::Constant(a),
        None => {
            ensure_len(&mut state.filt_alpha, TILE_PIXELS);
            kernel.row_sub1_forward(&raw_alpha[..tw], &mut state.filt_alpha[..tw]);
            for ry in 1..th {
                kernel.row_up_forward(
                    &raw_alpha[ry * tw..(ry + 1) * tw],
                    &raw_alpha[(ry - 1) * tw..ry * tw],
                    &mut state.filt_alpha[ry * tw..(ry + 1) * tw],
                );
            }
            match compress_bytes(
                &mut state.compressor,
                &mut state.zstd_scratch,
                &state.filt_alpha[..n_px],
            ) {
                EncodedBytes::Uncompressed(v) => AlphaChannel::Uncompressed(v),
                EncodedBytes::Zstd(v) => AlphaChannel::Zstd(v),
            }
        },
    }
}

#[inline(always)]
fn filter_aos_zstd_leaf_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
    state: &mut CompressorWorkerState,
) -> LeafPayload {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    debug_assert!(tw * th > 0 && tw * th <= TILE_PIXELS);

    // Direct 3-byte `SubGreen(BGR)` AoS + per-row adaptive `{None, Sub, Up, SubUp}`
    // streamed through two L1d-resident row buffers (`raw_row_a`, `raw_row_b`).
    let row_b = tw * 3;
    let total_b = row_b * th;
    ensure_len(&mut state.raw_row_a, row_b);
    if th > 1 {
        ensure_len(&mut state.raw_row_b, row_b);
    }
    if HAS_ALPHA {
        ensure_len(&mut state.raw_alpha, TILE_PIXELS);
    }

    let raw_row_a = &mut state.raw_row_a[..row_b];
    let raw_row_b = &mut state.raw_row_b[..if th > 1 { row_b } else { 0 }];
    let filt_bgr = &mut state.filt_bgr[..total_b];
    let raw_alpha = &mut state.raw_alpha[..if HAS_ALPHA { tw * th } else { 0 }];

    let mut predictors = vec![0u8; th.div_ceil(4)];
    let mut use_a = true;
    let pf_rows = stride_px > tw
        && stride_px * 4 >= prefetch::MIN_PAGE_STRIDE_BYTES
        && th > prefetch::PHASE_B_ROW_LOOKAHEAD;
    if pf_rows {
        assert!((y0 + th - 1) * stride_px + x0 + tw <= aos.len());
    }
    for (ry, filt_row) in filt_bgr.chunks_exact_mut(row_b).enumerate() {
        let row_start = (y0 + ry) * stride_px + x0;
        let (_, after_start) = aos.split_at(row_start);
        let (row_ptr, _) = after_start.split_at(tw);
        if pf_rows && ry + prefetch::PHASE_B_ROW_LOOKAHEAD < th {
            // SAFETY: Asserted `(y0 + th - 1) * stride_px + x0 + tw <= aos.len()` above
            // and `ry + PHASE_B_ROW_LOOKAHEAD < th`.
            unsafe {
                prefetch::row_t0(
                    row_ptr
                        .ptr()
                        .add(prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                        .cast::<u8>(),
                    tw * 4,
                );
            }
        }

        let (cur_raw, prev_raw): (&mut [u8], &[u8]) = if use_a {
            (raw_row_a, raw_row_b)
        } else {
            (raw_row_b, raw_row_a)
        };

        if HAS_ALPHA {
            kernel.bgra_to_subgreen_bgr3::<true>(
                row_ptr,
                cur_raw,
                &mut raw_alpha[ry * tw..(ry + 1) * tw],
            );
        } else {
            kernel.bgra_to_subgreen_bgr3::<false>(row_ptr, cur_raw, &mut []);
        }

        let best_p = select_and_filter_row::<K>(kernel, ry, cur_raw, prev_raw, filt_row);
        predictors[ry >> 2] |= best_p << ((ry & 3) * 2);
        use_a = !use_a;
    }

    let color = compress_bytes(
        &mut state.compressor,
        &mut state.zstd_scratch,
        &state.filt_bgr[..total_b],
    );
    let alpha = if HAS_ALPHA {
        encode_alpha_plane_with(kernel, state, tw, th)
    } else {
        AlphaChannel::Omitted
    };

    LeafPayload::AosZstd {
        predictors,
        color,
        alpha,
    }
}

fn filter_lossless_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
    meta: RectScanMeta,
    state: &mut CompressorWorkerState,
) -> LeafPayload {
    if let Some(px) = meta.solid_px {
        return LeafPayload::Solid(px);
    }
    if let Some(leaf) =
        try_encode_palette_leaf::<HAS_ALPHA>(tier, aos, stride_px, rect, meta.all_opaque, state)
    {
        return leaf;
    }
    if HAS_ALPHA && !meta.all_opaque {
        tier.filter_aos_zstd_leaf::<true>(aos, stride_px, rect, state)
    } else {
        tier.filter_aos_zstd_leaf::<false>(aos, stride_px, rect, state)
    }
}

fn encode_jpeg_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
    aos_is_owned_shadow: bool,
    stride_px: usize,
    rect: TileRect,
    quality: u8,
    subsamp: JpegSubsamp,
    state: &mut CompressorWorkerState,
) -> LeafPayload {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;
    debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

    if HAS_ALPHA {
        ensure_len(&mut state.raw_alpha, TILE_PIXELS);
    }

    let data = if aos_is_owned_shadow {
        let last_end = (y0 + th - 1) * stride_px + x0 + tw;
        assert!(last_end <= aos.len());
        if HAS_ALPHA {
            for ry in 0..th {
                let row_start = (y0 + ry) * stride_px + x0;
                // SAFETY: `aos` points to owned `shadow` memory (`last_end <= aos.len()`),
                // and `[x0..x0 + tw)` on each row of `rect` is initialized and never
                // concurrently mutated by sibling workers.
                let src_row = unsafe { slice::from_raw_parts(aos.ptr().add(row_start), tw) };
                let alpha_row = &mut state.raw_alpha[ry * tw..(ry + 1) * tw];
                for (dst_a, px) in alpha_row.iter_mut().zip(src_row.iter()) {
                    *dst_a = px.3;
                }
            }
        }
        // SAFETY: `aos` points to owned `shadow` memory with `rect` initialized
        // and exclusively owned by this worker (or globally immutable post-Phase-A),
        // `last_end <= aos.len()`, and `compress_jpeg_strided_raw` passes `*const u8`
        // directly to `tj3Compress8` (no spanning `&[u8]`), which strictly reads
        // `[0..tw * 4]` per row.
        unsafe {
            let src = aos.ptr().add(y0 * stride_px + x0).cast::<u8>();
            state
                .compress_jpeg_strided_raw(src, tw, stride_px * 4, th, quality, subsamp)
                .expect("turbojpeg compression failed for valid tile")
        }
    } else {
        // Stateless / raw `wl_shm` caller: bounce out of shared memory via
        // `BufferPointer` into lazily allocated owned worker scratch before
        // calling TurboJPEG (preserves SIGBUS / TOCTOU safety).
        ensure_len(&mut state.jpeg_pixels, n_px * 4);
        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            let (_, after_start) = aos.split_at(row_start);
            let (row_ptr, _) = after_start.split_at(tw);
            let dst_row: &mut [Vec4u8] =
                bytemuck::cast_slice_mut(&mut state.jpeg_pixels[ry * tw * 4..(ry + 1) * tw * 4]);
            row_ptr.copy_to_nonoverlapping(dst_row);
            if HAS_ALPHA {
                let alpha_row = &mut state.raw_alpha[ry * tw..(ry + 1) * tw];
                for (dst_a, px) in alpha_row.iter_mut().zip(dst_row.iter()) {
                    *dst_a = px.3;
                }
            }
        }
        state
            .compress_jpeg_from_scratch(tw, th, quality, subsamp)
            .expect("turbojpeg compression failed for valid tile")
    };

    let alpha = if HAS_ALPHA {
        tier.encode_alpha_plane(state, tw, th)
    } else {
        AlphaChannel::Omitted
    };
    LeafPayload::Jpeg { data, alpha }
}

fn filter_hybrid_tile<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
    aos_is_owned_shadow: bool,
    stride_px: usize,
    rect: TileRect,
    quality: u8,
    subsamp: JpegSubsamp,
    meta: &TileEncodeMeta,
    state: &mut CompressorWorkerState,
) -> TilePayload {
    if let Some(px) = meta.tile.solid_px {
        return TilePayload::Leaf(LeafPayload::Solid(px));
    }

    let tw = rect.w as usize;
    let th = rect.h as usize;

    // Small/thin tiles (`< 16` px in either dimension, including `< 8`-row 1D
    // strips) always stay lossless where JPEG headers would bloat the payload.
    if tw < MIN_JPEG_DIM || th < MIN_JPEG_DIM {
        return TilePayload::Leaf(filter_lossless_leaf::<HAS_ALPHA>(
            tier, aos, stride_px, rect, meta.tile, state,
        ));
    }

    // Non-splittable (`<= 128x128`): single-pass fused palette + text-safe
    // classification — never scans pixels twice on overflow.
    if tw <= SUBTILE_W && th <= SUBTILE_H {
        let scan = if HAS_ALPHA && !meta.tile.all_opaque {
            state.palette.scan_hybrid_leaf::<true>(aos, stride_px, rect)
        } else {
            state
                .palette
                .scan_hybrid_leaf::<false>(aos, stride_px, rect)
        };
        return TilePayload::Leaf(match scan {
            HybridScan::Solid(px) => LeafPayload::Solid(px),
            HybridScan::Palette { n_colors } => {
                let n_px = tw * th;
                state.palette.pack_indices_range(tier, 0, n_px, n_colors);
                encode_palette_from_parts::<HAS_ALPHA>(
                    &state.palette.palette[..n_colors],
                    state.palette.packed_indices(0, n_px, n_colors),
                    meta.tile.all_opaque,
                    &mut state.compressor,
                    &mut state.zstd_scratch,
                )
            },
            HybridScan::TextSafeNoPalette => {
                if HAS_ALPHA && !meta.tile.all_opaque {
                    tier.filter_aos_zstd_leaf::<true>(aos, stride_px, rect, state)
                } else {
                    tier.filter_aos_zstd_leaf::<false>(aos, stride_px, rect, state)
                }
            },
            HybridScan::Photo => {
                if HAS_ALPHA && !meta.tile.all_opaque {
                    encode_jpeg_leaf::<true>(
                        tier,
                        aos,
                        aos_is_owned_shadow,
                        stride_px,
                        rect,
                        quality,
                        subsamp,
                        state,
                    )
                } else {
                    encode_jpeg_leaf::<false>(
                        tier,
                        aos,
                        aos_is_owned_shadow,
                        stride_px,
                        rect,
                        quality,
                        subsamp,
                        state,
                    )
                }
            },
        });
    }

    // Splittable (`> 128` wide or tall): fast path if the whole tile fits in
    // `<= 256` unique colors — every quadrant is guaranteed `TextSafe`, so emit
    // the encoded `Solid`/`Palette` leaf directly without re-scanning.
    let ovf_ry = match state
        .palette
        .try_encode_2d::<HAS_ALPHA>(aos, stride_px, rect)
    {
        Ok(n_colors) => {
            if n_colors == 1 {
                return TilePayload::Leaf(LeafPayload::Solid(
                    state.palette.palette[0].to_le_bytes(),
                ));
            }
            let n_px = tw * th;
            state.palette.pack_indices_range(tier, 0, n_px, n_colors);
            return TilePayload::Leaf(encode_palette_from_parts::<HAS_ALPHA>(
                &state.palette.palette[..n_colors],
                state.palette.packed_indices(0, n_px, n_colors),
                meta.tile.all_opaque,
                &mut state.compressor,
                &mut state.zstd_scratch,
            ));
        },
        Err(ry) => ry,
    };

    // Classify each `128x128` quadrant in a single pass while retaining built
    // per-quadrant palettes/indices in worker scratch so mixed `Split` tiles
    // never re-scan palette quadrants.
    //
    // If the whole-tile scan already cleared all `SUBTILE_H` top rows before
    // hitting color #257 (`ovf_ry >= SUBTILE_H`), every top-row quadrant is
    // guaranteed `<= 256` unique colors (`TextSafe`); skip classifying them so
    // all-`TextSafe` tiles emit `AosZstd` directly without re-scanning the top
    // half, and build their quadrant palettes only if a split occurs.
    let (quads, n_quads) = tile_quadrants(rect);
    let top_row_known_palette = ovf_ry >= SUBTILE_H;
    let mut scans = [HybridScan::Photo; 4];
    let mut top_deferred = [false; 4];
    let mut text_safe_count = 0usize;
    for q in 0..n_quads {
        let qmeta = meta.quads[q];
        if let Some(px) = qmeta.solid_px {
            scans[q] = HybridScan::Solid(px);
            text_safe_count += 1;
            continue;
        }
        if top_row_known_palette && quads[q].y == rect.y {
            top_deferred[q] = true;
            text_safe_count += 1;
            continue;
        }
        let scan = if HAS_ALPHA && !qmeta.all_opaque {
            state
                .palette
                .scan_hybrid_quadrant::<true>(q, aos, stride_px, quads[q])
        } else {
            state
                .palette
                .scan_hybrid_quadrant::<false>(q, aos, stride_px, quads[q])
        };
        scans[q] = scan;
        text_safe_count += usize::from(scan.is_text_safe());
    }

    if text_safe_count == n_quads {
        TilePayload::Leaf(if HAS_ALPHA && !meta.tile.all_opaque {
            tier.filter_aos_zstd_leaf::<true>(aos, stride_px, rect, state)
        } else {
            tier.filter_aos_zstd_leaf::<false>(aos, stride_px, rect, state)
        })
    } else if text_safe_count == 0 {
        TilePayload::Leaf(if HAS_ALPHA && !meta.tile.all_opaque {
            encode_jpeg_leaf::<true>(
                tier,
                aos,
                aos_is_owned_shadow,
                stride_px,
                rect,
                quality,
                subsamp,
                state,
            )
        } else {
            encode_jpeg_leaf::<false>(
                tier,
                aos,
                aos_is_owned_shadow,
                stride_px,
                rect,
                quality,
                subsamp,
                state,
            )
        })
    } else {
        let mut sub_tiles = Vec::with_capacity(n_quads);
        for q in 0..n_quads {
            let qrect = quads[q];
            let qmeta = meta.quads[q];
            let payload = if top_deferred[q] {
                // Top-row quadrants (`q < 2`, `q_px <= SUBTILE_PIXELS`) were
                // deferred; `try_encode_2d` writes only `indices[..q_px]` and
                // `self.palette`, leaving bottom-row quadrant indices
                // (`>= SUBTILE_PIXELS`) and `quad_palettes` untouched.
                debug_assert!((qrect.w as usize) * (qrect.h as usize) <= SUBTILE_PIXELS);
                filter_lossless_leaf::<HAS_ALPHA>(tier, aos, stride_px, qrect, qmeta, state)
            } else {
                match scans[q] {
                    HybridScan::Solid(px) => LeafPayload::Solid(px),
                    HybridScan::Palette { n_colors } => {
                        let start = q * SUBTILE_PIXELS;
                        let q_px = qrect.w as usize * qrect.h as usize;
                        state
                            .palette
                            .pack_indices_range(tier, start, q_px, n_colors);
                        encode_palette_from_parts::<HAS_ALPHA>(
                            &state.palette.quad_palette(q)[..n_colors],
                            state.palette.packed_indices(start, q_px, n_colors),
                            qmeta.all_opaque,
                            &mut state.compressor,
                            &mut state.zstd_scratch,
                        )
                    },
                    HybridScan::TextSafeNoPalette => {
                        if HAS_ALPHA && !qmeta.all_opaque {
                            tier.filter_aos_zstd_leaf::<true>(aos, stride_px, qrect, state)
                        } else {
                            tier.filter_aos_zstd_leaf::<false>(aos, stride_px, qrect, state)
                        }
                    },
                    HybridScan::Photo => {
                        if (qrect.w as usize) < MIN_JPEG_DIM || (qrect.h as usize) < MIN_JPEG_DIM {
                            if HAS_ALPHA && !qmeta.all_opaque {
                                tier.filter_aos_zstd_leaf::<true>(aos, stride_px, qrect, state)
                            } else {
                                tier.filter_aos_zstd_leaf::<false>(aos, stride_px, qrect, state)
                            }
                        } else if HAS_ALPHA && !qmeta.all_opaque {
                            encode_jpeg_leaf::<true>(
                                tier,
                                aos,
                                aos_is_owned_shadow,
                                stride_px,
                                qrect,
                                quality,
                                subsamp,
                                state,
                            )
                        } else {
                            encode_jpeg_leaf::<false>(
                                tier,
                                aos,
                                aos_is_owned_shadow,
                                stride_px,
                                qrect,
                                quality,
                                subsamp,
                                state,
                            )
                        }
                    },
                }
            };
            sub_tiles.push(SubTile {
                rect: qrect,
                payload,
            });
        }
        TilePayload::Split(sub_tiles)
    }
}

fn filter_tile<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
    aos_is_owned_shadow: bool,
    stride_px: usize,
    rect: TileRect,
    mode: CompressionMode,
    meta: &TileEncodeMeta,
    state: &mut CompressorWorkerState,
) -> TilePayload {
    match mode {
        CompressionMode::Lossless => TilePayload::Leaf(filter_lossless_leaf::<HAS_ALPHA>(
            tier, aos, stride_px, rect, meta.tile, state,
        )),
        CompressionMode::Hybrid { quality, subsamp } => filter_hybrid_tile::<HAS_ALPHA>(
            tier,
            aos,
            aos_is_owned_shadow,
            stride_px,
            rect,
            quality,
            subsamp,
            meta,
            state,
        ),
    }
}

// --- fused 2D tile decompress + unfilter ---

/// Raw pointer handle to the destination frame buffer (or uninitialized
/// [`ClientTileCache`] slot spare capacity) preserving pointer provenance for
/// disjoint per-row mutable writes across parallel workers.
pub(crate) struct FrameRows {
    pub(crate) base: *mut MaybeUninit<Vec4u8>,
    pub(crate) stride_px: usize,
}

// SAFETY: `unfilter_tiles_parallel` validates every accepted `TileRect` is
// strictly in-bounds and pairwise disjoint before any worker calls `row_ptr`,
// so concurrent workers never alias overlapping pixels.
unsafe impl Send for FrameRows {}
unsafe impl Sync for FrameRows {}

impl FrameRows {
    /// # Safety
    /// Caller must guarantee `(y, x..x + w)` lies within the validated frame
    /// bounds.
    #[inline]
    pub(crate) unsafe fn row_ptr(&self, y: usize, x: usize) -> *mut MaybeUninit<Vec4u8> {
        // SAFETY: Caller upholds in-bounds invariants.
        unsafe { self.base.add(y * self.stride_px + x) }
    }
}

#[inline]
fn decode_encoded_bytes<'a>(
    encoded: &'a EncodedBytes,
    expected_len: usize,
    scratch: &'a mut [u8],
    decompressor: &mut Decompressor<'static>,
) -> Result<&'a [u8]> {
    match encoded {
        EncodedBytes::Uncompressed(data) => {
            if data.len() != expected_len {
                bail!(
                    "uncompressed payload len {} != expected {expected_len}",
                    data.len()
                );
            }
            Ok(data.as_slice())
        },
        EncodedBytes::Zstd(data) => {
            let out = &mut scratch[..expected_len];
            let written = decompressor
                .decompress_to_buffer(data, out)
                .location(loc!())?;
            if written != expected_len {
                bail!("decompressed payload len {written} != expected {expected_len}");
            }
            Ok(out)
        },
    }
}

fn prepare_alpha_decode<'a, const HAS_ALPHA: bool>(
    alpha: &'a AlphaChannel,
    n_px: usize,
    alpha_filt: &'a mut Vec<u8>,
    decompressor: &mut Decompressor<'static>,
) -> Result<(Option<u8>, Option<&'a [u8]>)> {
    match alpha {
        AlphaChannel::Omitted => Ok((Some(0xFF), None)),
        AlphaChannel::Constant(a) => {
            if !HAS_ALPHA {
                bail!("AlphaChannel::Constant on Xrgb8888 buffer");
            }
            Ok((Some(*a), None))
        },
        AlphaChannel::Uncompressed(bytes) => {
            if !HAS_ALPHA {
                bail!("AlphaChannel::Uncompressed on Xrgb8888 buffer");
            }
            if bytes.len() != n_px {
                bail!("uncompressed alpha len {} != expected {n_px}", bytes.len());
            }
            Ok((None, Some(bytes.as_slice())))
        },
        AlphaChannel::Zstd(bytes) => {
            if !HAS_ALPHA {
                bail!("AlphaChannel::Zstd on Xrgb8888 buffer");
            }
            ensure_len(alpha_filt, TILE_PIXELS);
            let written = decompressor
                .decompress_to_buffer(bytes, &mut alpha_filt[..n_px])
                .location(loc!())?;
            if written != n_px {
                bail!("decompressed alpha len {written} != expected {n_px}");
            }
            Ok((None, Some(&alpha_filt[..n_px])))
        },
    }
}

/// # Safety
/// `(x0, y0, tw, th)` must be validated in-bounds within `frame` and disjoint
/// from all other tiles being decoded concurrently; when `STREAM` is `true`,
/// every row pointer `frame.row_ptr(y0 + ry, x0)` must be aligned to
/// `K::STREAM_ALIGN`.
#[inline(always)]
unsafe fn decode_jpeg_alpha_rows_with<K: RowKernel, const STREAM: bool, const CONST_ALPHA: bool>(
    kernel: K,
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
    if !CONST_ALPHA {
        ensure_len(alpha_row_a, tw);
        if th > 1 {
            ensure_len(alpha_row_b, tw);
        }
    }
    let mut use_a = true;
    for ry in 0..th {
        let alpha_row: &[u8] = if CONST_ALPHA {
            &[]
        } else {
            let (a_cur, a_prev): (&mut [u8], &[u8]) = if use_a {
                (
                    &mut alpha_row_a[..tw],
                    &alpha_row_b[..if ry > 0 { tw } else { 0 }],
                )
            } else {
                (&mut alpha_row_b[..tw], &alpha_row_a[..tw])
            };
            if ry == 0 {
                kernel.row_sub1_inverse(&a_filt[..tw], a_cur);
            } else {
                kernel.row_up_inverse(&a_filt[ry * tw..(ry + 1) * tw], a_prev, a_cur);
            }
            a_cur
        };
        let src_row = &jpeg_rows[ry * tw..(ry + 1) * tw];
        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint,
        // and aligned to `K::STREAM_ALIGN` when `STREAM` is `true`.
        let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
        unsafe {
            kernel.merge_bgrx_alpha_row::<STREAM, CONST_ALPHA>(
                src_row,
                alpha_row,
                const_alpha,
                dst_row,
            );
        }
        use_a = !use_a;
    }
}

/// # Safety
/// `(x0, y0, tw, th)` must be validated in-bounds within `frame`, already
/// initialized with decoded `BGRX` pixels, and disjoint from all other tiles
/// being decoded concurrently.
#[inline(always)]
unsafe fn decode_jpeg_alpha_rows_in_place_with<K: RowKernel, const CONST_ALPHA: bool>(
    kernel: K,
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
    if !CONST_ALPHA {
        ensure_len(alpha_row_a, tw);
        if th > 1 {
            ensure_len(alpha_row_b, tw);
        }
    }
    let mut use_a = true;
    for ry in 0..th {
        let alpha_row: &[u8] = if CONST_ALPHA {
            &[]
        } else {
            let (a_cur, a_prev): (&mut [u8], &[u8]) = if use_a {
                (
                    &mut alpha_row_a[..tw],
                    &alpha_row_b[..if ry > 0 { tw } else { 0 }],
                )
            } else {
                (&mut alpha_row_b[..tw], &alpha_row_a[..tw])
            };
            if ry == 0 {
                kernel.row_sub1_inverse(&a_filt[..tw], a_cur);
            } else {
                kernel.row_up_inverse(&a_filt[ry * tw..(ry + 1) * tw], a_prev, a_cur);
            }
            a_cur
        };
        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds, disjoint,
        // and already initialized with `Vec4u8` BGRX pixels by TurboJPEG.
        let dst_row =
            unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0).cast::<Vec4u8>(), tw) };
        kernel.merge_bgrx_alpha_row_in_place::<CONST_ALPHA>(dst_row, alpha_row, const_alpha);
        use_a = !use_a;
    }
}

/// # Safety
/// `rect` shifted by `dst_origin` must already be validated in-bounds within
/// `frame` and disjoint from all other tiles being decoded concurrently.
#[inline(never)]
unsafe fn unfilter_jpeg_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    rect: TileRect,
    dst_origin: (usize, usize),
    allow_stream: bool,
    data: &[u8],
    alpha: &AlphaChannel,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let x0 = rect.x as usize - dst_origin.0;
    let y0 = rect.y as usize - dst_origin.1;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;
    if data.is_empty() {
        bail!("empty JPEG payload");
    }
    if !HAS_ALPHA && !matches!(alpha, AlphaChannel::Omitted) {
        bail!("non-omitted AlphaChannel on Xrgb8888 JPEG tile");
    }
    let is_opaque = matches!(alpha, AlphaChannel::Omitted)
        || (HAS_ALPHA && matches!(alpha, AlphaChannel::Constant(0xFF)));
    if is_opaque {
        // Zero-copy strided decode directly into `frame` (owned `ClientTileCache`
        // slot or disjoint rect of shared `wl_shm` output) using `TJPF_BGRA`,
        // which unconditionally sets byte 3 to `0xFF` and issues its own `sfence`
        // after any internal streaming stores.
        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint
        // within `frame`; `decompress_jpeg_into_raw` uses `*mut u8` (no spanning
        // `&mut [u8]`) and libjpeg-turbo strictly writes `[0..tw * 4]` per row.
        unsafe {
            let dst = frame.row_ptr(y0, x0).cast::<u8>();
            state
                .decompress_jpeg_into_raw(data, tw, th, dst, frame.stride_px * 4)
                .location(loc!())?;
        }
        return Ok(false);
    }

    if !allow_stream {
        // Cached `ClientTileCache` slot target with non-0xFF or varying alpha:
        // decode straight into the exclusively owned slot and merge alpha
        // in-place with SIMD, avoiding `state.jpeg_pixels` allocation/copy and
        // leaving the slot L2-hot for the subsequent `frame` streaming blit.
        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and
        // exclusively owned inside `frame` (cache slot); `decompress_jpeg_into_raw`
        // initializes all `tw * th` pixels before in-place alpha merge reads them.
        unsafe {
            let dst = frame.row_ptr(y0, x0).cast::<u8>();
            state
                .decompress_jpeg_into_raw(data, tw, th, dst, frame.stride_px * 4)
                .location(loc!())?;
        }
        let (const_alpha, alpha_filt_slice) = prepare_alpha_decode::<HAS_ALPHA>(
            alpha,
            n_px,
            &mut state.alpha_filt,
            &mut state.decompressor,
        )
        .location(loc!())?;
        // SAFETY: `decompress_jpeg_into_raw` initialized all `tw * th` pixels
        // in `(x0, y0, tw, th)` inside exclusively owned `frame`.
        unsafe {
            if let Some(a) = const_alpha {
                tier.decode_jpeg_alpha_rows_in_place::<true>(
                    &[],
                    a,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            } else if HAS_ALPHA {
                let a_filt = alpha_filt_slice.unwrap();
                tier.decode_jpeg_alpha_rows_in_place::<false>(
                    a_filt,
                    0,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            }
        }
        return Ok(false);
    }

    // Uncached direct `wl_shm` target with non-0xFF or varying alpha: decode
    // BGRX into worker scratch and merge alpha in a single SIMD pass using
    // non-temporal streaming stores (`vmovntdq`) when aligned, avoiding RFO
    // read-modify-write traffic against `wl_shm`.
    state
        .decompress_jpeg_to_scratch(data, tw, th)
        .location(loc!())?;
    let (const_alpha, alpha_filt_slice) = prepare_alpha_decode::<HAS_ALPHA>(
        alpha,
        n_px,
        &mut state.alpha_filt,
        &mut state.decompressor,
    )
    .location(loc!())?;
    let jpeg_rows: &[Vec4u8] = bytemuck::cast_slice(&state.jpeg_pixels[..n_px * 4]);
    let stream_align = tier.stream_align();
    let stream_rows = stream_align > 1
        && tw.is_multiple_of(16)
        // SAFETY: Caller guarantees `(x0, y0)` is in-bounds within `frame`.
        && (unsafe { frame.row_ptr(y0, x0) } as usize).is_multiple_of(stream_align)
        && (frame.stride_px * 4).is_multiple_of(stream_align);

    // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint
    // within `frame`, and `stream_rows` verified alignment of every row.
    unsafe {
        if let Some(a) = const_alpha {
            if stream_rows {
                tier.decode_jpeg_alpha_rows::<true, true>(
                    &[],
                    a,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            } else {
                tier.decode_jpeg_alpha_rows::<false, true>(
                    &[],
                    a,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            }
        } else if HAS_ALPHA {
            let a_filt = alpha_filt_slice.unwrap();
            if stream_rows {
                tier.decode_jpeg_alpha_rows::<true, false>(
                    a_filt,
                    0,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            } else {
                tier.decode_jpeg_alpha_rows::<false, false>(
                    a_filt,
                    0,
                    jpeg_rows,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    &mut state.alpha_row_a,
                    &mut state.alpha_row_b,
                );
            }
        }
    }
    Ok(stream_rows)
}

/// # Safety
/// `(x0, y0, tw, th)` must be validated in-bounds within `frame` and disjoint
/// from all other tiles being decoded concurrently; when `STREAM` is `true`,
/// every row pointer `frame.row_ptr(y0 + ry, x0)` must be aligned to
/// `K::STREAM_ALIGN` and `tw.is_multiple_of(16)` must hold.
#[inline(always)]
unsafe fn unfilter_aos_zstd_rows_with<K: RowKernel, const STREAM: bool, const CONST_ALPHA: bool>(
    kernel: K,
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
    let row_b = tw * 3;
    let mut use_a = true;
    for ry in 0..th {
        let c = ry * row_b;
        let filt_row = &filt_bgr[c..c + row_b];
        let p_mode = predictor_at(predictors, ry);
        let next_needs_prev = ry + 1 < th && predictor_at(predictors, ry + 1) >= 2;

        let (cur_dec, prev_dec): (&mut [u8], &[u8]) = if use_a {
            (dec_row_a, if ry > 0 { dec_row_b } else { &[] })
        } else {
            (dec_row_b, dec_row_a)
        };

        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
        let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };

        let alpha_row: &[u8] = if CONST_ALPHA {
            &[]
        } else {
            let (a_cur, a_prev): (&mut [u8], &[u8]) = if use_a {
                (alpha_row_a, if ry > 0 { alpha_row_b } else { &[] })
            } else {
                (alpha_row_b, alpha_row_a)
            };
            if ry == 0 {
                kernel.row_sub1_inverse(&alpha_filt[..tw], a_cur);
            } else {
                kernel.row_up_inverse(&alpha_filt[ry * tw..(ry + 1) * tw], a_prev, a_cur);
            }
            a_cur
        };

        // SAFETY: Caller verified `dst_row` alignment matches `K::STREAM_ALIGN`
        // when `STREAM` is `true`, and row 0 rejects `p_mode >= 2` so
        // `prev_dec.len() == row_b` whenever `p_mode >= 2`.
        if p_mode == 2 {
            unsafe {
                if next_needs_prev {
                    kernel.row_up_subgreen_bgr3_to_bgra::<STREAM, CONST_ALPHA, true>(
                        filt_row,
                        prev_dec,
                        cur_dec,
                        alpha_row,
                        const_alpha,
                        dst_row,
                    );
                } else {
                    kernel.row_up_subgreen_bgr3_to_bgra::<STREAM, CONST_ALPHA, false>(
                        filt_row,
                        prev_dec,
                        &mut [],
                        alpha_row,
                        const_alpha,
                        dst_row,
                    );
                }
            }
        } else {
            let decoded_bgr: &[u8] = match p_mode {
                0 => {
                    if next_needs_prev {
                        cur_dec.copy_from_slice(filt_row);
                    }
                    filt_row
                },
                1 => {
                    kernel.row_sub3_inverse(filt_row, cur_dec);
                    cur_dec
                },
                _ => {
                    kernel.row_subup3_inverse(filt_row, prev_dec, cur_dec);
                    cur_dec
                },
            };
            // SAFETY: Caller verified `dst_row` alignment matches `K::STREAM_ALIGN`
            // when `STREAM` is `true`.
            unsafe {
                kernel.subgreen_bgr3_to_bgra::<STREAM, CONST_ALPHA>(
                    decoded_bgr,
                    alpha_row,
                    const_alpha,
                    dst_row,
                );
            }
        }
        use_a = !use_a;
    }
}

/// # Safety
/// `rect` shifted by `dst_origin` must already be validated in-bounds within
/// `frame` and disjoint from all other tiles being decoded concurrently.
#[inline(never)]
unsafe fn unfilter_aos_zstd_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    rect: TileRect,
    dst_origin: (usize, usize),
    allow_stream: bool,
    predictors: &[u8],
    color: &EncodedBytes,
    alpha: &AlphaChannel,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let x0 = rect.x as usize - dst_origin.0;
    let y0 = rect.y as usize - dst_origin.1;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;

    let expected_pred_len = th.div_ceil(4);
    if predictors.len() != expected_pred_len {
        bail!(
            "predictors len {} != expected {expected_pred_len}",
            predictors.len()
        );
    }
    let rem_bits = th & 3;
    if rem_bits != 0 && (predictors[th >> 2] >> (rem_bits * 2)) != 0 {
        bail!("non-zero padding bits in row predictors");
    }
    if predictor_at(predictors, 0) >= 2 {
        bail!("invalid row-0 predictor {}", predictor_at(predictors, 0));
    }

    let row_b = tw * 3;
    let total_b = row_b * th;
    let filt_bgr =
        decode_encoded_bytes(color, total_b, &mut state.filt_bgr, &mut state.decompressor)
            .location(loc!())?;

    let (const_alpha, alpha_filt_slice) = prepare_alpha_decode::<HAS_ALPHA>(
        alpha,
        n_px,
        &mut state.alpha_filt,
        &mut state.decompressor,
    )
    .location(loc!())?;

    ensure_len(&mut state.dec_row_a, row_b);
    if th > 1 {
        ensure_len(&mut state.dec_row_b, row_b);
    }
    if alpha_filt_slice.is_some() {
        ensure_len(&mut state.alpha_row_a, tw);
        if th > 1 {
            ensure_len(&mut state.alpha_row_b, tw);
        }
    }

    let dec_row_a = &mut state.dec_row_a[..row_b];
    let dec_row_b = &mut state.dec_row_b[..if th > 1 { row_b } else { 0 }];

    let stream_align = tier.stream_align();
    let stream_rows = allow_stream
        && stream_align > 1
        && tw.is_multiple_of(16)
        // SAFETY: Caller guarantees `(x0, y0)` is in-bounds within `frame`.
        && (unsafe { frame.row_ptr(y0, x0) } as usize).is_multiple_of(stream_align)
        && (frame.stride_px * 4).is_multiple_of(stream_align);

    // SAFETY: Caller guarantees `rect` is in-bounds and disjoint within `frame`,
    // and `stream_rows` verified alignment of every row and `tw % 16 == 0`.
    unsafe {
        if let Some(a) = const_alpha {
            if stream_rows {
                tier.unfilter_aos_zstd_rows::<true, true>(
                    predictors,
                    filt_bgr,
                    &[],
                    a,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    dec_row_a,
                    dec_row_b,
                    &mut [],
                    &mut [],
                );
            } else {
                tier.unfilter_aos_zstd_rows::<false, true>(
                    predictors,
                    filt_bgr,
                    &[],
                    a,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    dec_row_a,
                    dec_row_b,
                    &mut [],
                    &mut [],
                );
            }
        } else if HAS_ALPHA {
            let a_filt = alpha_filt_slice.unwrap();
            let alpha_row_a = &mut state.alpha_row_a[..tw];
            let alpha_row_b = &mut state.alpha_row_b[..if th > 1 { tw } else { 0 }];
            if stream_rows {
                tier.unfilter_aos_zstd_rows::<true, false>(
                    predictors,
                    filt_bgr,
                    a_filt,
                    0,
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
            } else {
                tier.unfilter_aos_zstd_rows::<false, false>(
                    predictors,
                    filt_bgr,
                    a_filt,
                    0,
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
        } else {
            unreachable!("prepare_alpha_decode::<false> always returns Some(0xFF)");
        }
    }

    Ok(stream_rows)
}

/// # Safety
/// `rect` shifted by `dst_origin` must already be validated in-bounds within
/// `frame` and disjoint from all other tiles being decoded concurrently.
#[inline(always)]
unsafe fn unfilter_leaf<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    tier: arch::Tier,
    rect: TileRect,
    dst_origin: (usize, usize),
    allow_stream: bool,
    payload: &LeafPayload,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let x0 = rect.x as usize - dst_origin.0;
    let y0 = rect.y as usize - dst_origin.1;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;
    debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

    match payload {
        LeafPayload::Solid(px) => {
            if !HAS_ALPHA && px[3] != 0xFF {
                bail!("translucent Solid tile on Xrgb8888 buffer");
            }
            let pixel = Vec4u8(px[0], px[1], px[2], px[3]);
            let mut streamed = false;
            for ry in 0..th {
                // SAFETY: Caller guarantees `rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                if allow_stream
                    && K::STREAM_ALIGN > 1
                    && tw.is_multiple_of(16)
                    && (dst_row.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN)
                {
                    // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                    unsafe { kernel.splat_row::<true>(dst_row, pixel) };
                    streamed = true;
                } else {
                    // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                    unsafe { kernel.splat_row::<false>(dst_row, pixel) };
                }
            }
            Ok(streamed)
        },
        LeafPayload::Palette {
            has_alpha: pal_alpha,
            palette,
            data,
        } => {
            if *pal_alpha && !HAS_ALPHA {
                bail!("translucent Palette tile on Xrgb8888 buffer");
            }
            let (n_colors, expected_len) =
                parse_palette_header(*pal_alpha, palette, n_px).location(loc!())?;
            let packed = decode_encoded_bytes(
                data,
                expected_len,
                &mut state.filt_bgr,
                &mut state.decompressor,
            )
            .location(loc!())?;
            // SAFETY: Caller guarantees `rect` shifted by `dst_origin` is
            // in-bounds and disjoint within `frame`, and `parse_palette_header`
            // validated `palette` and `n_colors`.
            unsafe {
                decode_palette_tile(
                    kernel,
                    *pal_alpha,
                    palette,
                    n_colors,
                    packed,
                    frame,
                    x0,
                    y0,
                    tw,
                    th,
                    allow_stream,
                )
                .location(loc!())
            }
        },
        LeafPayload::AosZstd {
            predictors,
            color,
            alpha,
        } => {
            // SAFETY: Caller guarantees `rect` shifted by `dst_origin` is in-bounds and disjoint.
            unsafe {
                unfilter_aos_zstd_leaf::<HAS_ALPHA>(
                    tier,
                    rect,
                    dst_origin,
                    allow_stream,
                    predictors,
                    color,
                    alpha,
                    frame,
                    state,
                )
            }
        },
        LeafPayload::Jpeg { data, alpha } => {
            // SAFETY: Caller guarantees `rect` shifted by `dst_origin` is in-bounds and disjoint.
            unsafe {
                unfilter_jpeg_leaf::<HAS_ALPHA>(
                    tier,
                    rect,
                    dst_origin,
                    allow_stream,
                    data,
                    alpha,
                    frame,
                    state,
                )
            }
        },
    }
}

/// # Safety
/// `rect` shifted by `dst_origin` must already be validated in-bounds within
/// `frame` and disjoint from all other tiles being decoded concurrently.
#[inline(always)]
unsafe fn unfilter_tile<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    tier: arch::Tier,
    rect: TileRect,
    dst_origin: (usize, usize),
    allow_stream: bool,
    payload: &TilePayload,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let mut idx = 0usize;
    let mut streamed = false;
    loop {
        let (leaf_rect, leaf_payload) = match payload {
            TilePayload::Leaf(leaf) => {
                if idx > 0 {
                    break;
                }
                (rect, leaf)
            },
            TilePayload::Split(sub_tiles) => {
                if idx == 0 {
                    validate_split_rects(rect, sub_tiles).location(loc!())?;
                }
                let Some(sub) = sub_tiles.get(idx) else {
                    break;
                };
                (sub.rect, &sub.payload)
            },
            TilePayload::CacheHit(_) => {
                bail!("unexpected CacheHit inside unfilter_tile");
            },
        };
        // SAFETY: Caller guarantees `rect` shifted by `dst_origin` is in-bounds
        // and disjoint; for `Split`, `validate_split_rects` verified every
        // `sub.rect` is inside `rect` (so `sub.rect.x >= dst_origin.0` and
        // `sub.rect.y >= dst_origin.1`) and pairwise disjoint across siblings.
        streamed |= unsafe {
            unfilter_leaf::<K, HAS_ALPHA>(
                kernel,
                tier,
                leaf_rect,
                dst_origin,
                allow_stream,
                leaf_payload,
                frame,
                state,
            )
            .location(loc!())?
        };
        idx += 1;
    }
    Ok(streamed)
}

/// Copies `rect.w x rect.h` pixels from `src` (row stride `src_stride_px`)
/// into `frame` at `rect`, using non-temporal streaming stores on aligned rows
/// (`K::STREAM_ALIGN > 1 && tw % 16 == 0`). Returns `true` if any non-temporal
/// store was issued (caller must execute `kernel.fence_stream()`).
///
/// # Safety
/// Caller must guarantee `src` contains at least
/// `(rect.h - 1) * src_stride_px + rect.w` pixels and `rect` is in-bounds and
/// disjoint in `frame`.
#[inline(always)]
unsafe fn copy_pixels_to_frame_with<K: RowKernel>(
    kernel: K,
    src: &[Vec4u8],
    src_stride_px: usize,
    rect: TileRect,
    frame: &FrameRows,
) -> bool {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let can_stream_width = K::STREAM_ALIGN > 1 && tw.is_multiple_of(16);
    let mut streamed = false;
    for ry in 0..th {
        let src_start = ry * src_stride_px;
        let src_row = &src[src_start..src_start + tw];
        // SAFETY: Caller guarantees `rect` is in-bounds and disjoint in `frame`.
        let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
        if can_stream_width && (dst_row.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN) {
            // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN` and `tw % 16 == 0`.
            unsafe { kernel.copy_row_stream(src_row, dst_row) };
            streamed = true;
        } else {
            // SAFETY: `src_row` and `dst_row` have equal length `tw` and do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(
                    src_row.as_ptr(),
                    dst_row.as_mut_ptr().cast::<Vec4u8>(),
                    tw,
                );
            }
        }
    }
    streamed
}

/// Worker loop for Step-2 parallel `CacheHit` blits from immutable
/// [`ClientTileCache`] slots into disjoint rects of `frame`, using non-temporal
/// streaming stores on aligned rows with a single fence per worker.
///
/// # Safety
/// Caller must have validated every entry in `hits` via
/// [`ClientTileCache::validate_hit`] and verified `tile_rects` are in-bounds
/// and pairwise disjoint in `frame`.
#[inline(always)]
unsafe fn blit_hits_worker_with<K: RowKernel>(
    kernel: K,
    cache: &ClientTileCache,
    hits: &[ValidatedCacheHit],
    tile_rects: &[TileRect],
    frame: &FrameRows,
    next_hit: &AtomicUsize,
) {
    let n_hits = hits.len();
    let mut streamed_any = false;
    loop {
        let i = next_hit.fetch_add(1, Ordering::Relaxed);
        if i >= n_hits {
            break;
        }
        let hit = hits[i];
        let (src, src_stride_px) = cache.slot_hit_src(hit);
        // SAFETY: `hit` was validated via `validate_hit` and `tile_rects` are
        // in-bounds and pairwise disjoint.
        streamed_any |= unsafe {
            copy_pixels_to_frame_with(kernel, src, src_stride_px, tile_rects[hit.tile_idx], frame)
        };
    }
    if streamed_any {
        kernel.fence_stream();
    }
}

/// Shared Step-1 state for parallel `CachePut` writes and `CacheHit` recording.
pub(crate) struct DecodeCacheCtx {
    slots: SyncClientSlots,
    put_claimed: Box<[AtomicBool; MAX_TILE_CACHE_SLOTS]>,
    frame_put_bytes: AtomicUsize,
    hit_refs: Vec<AtomicU32>,
    hit_count: AtomicUsize,
    puts_only: bool,
}

/// # Safety
/// Caller must have validated `tile_rects` in-bounds within `frame` (unless
/// `cache_ctx.puts_only` is `true`) and pairwise disjoint.
#[inline(always)]
unsafe fn decode_one_tile<'a, K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    tier: arch::Tier,
    item: Result<Cow<'a, CompressedTile>>,
    tile_rects: &[TileRect],
    frame: &FrameRows,
    claimed: &[AtomicBool],
    cache_ctx: Option<&DecodeCacheCtx>,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let tile = item?;
    let idx = tile.tile_idx as usize;
    let Some(slot) = claimed.get(idx) else {
        bail!("tile_idx {idx} out of range (n_tiles={})", tile_rects.len());
    };
    if slot.swap(true, Ordering::AcqRel) {
        bail!("duplicate tile_idx {idx}");
    }
    let rect = tile_rects[idx];

    match &tile.payload {
        TilePayload::CacheHit(hit_ref) => {
            if tile.cache_put.is_some() {
                bail!("CacheHit tile {idx} cannot also set cache_put");
            }
            let Some(ctx) = cache_ctx else {
                bail!("received CacheHit tile {idx} without active ClientTileCache");
            };
            hit_ref.validate_for_rect(rect).location(loc!())?;
            if !ctx.puts_only {
                let raw = hit_ref.to_u16().location(loc!())?;
                ctx.hit_refs[idx].store(u32::from(raw) + 1, Ordering::Relaxed);
                ctx.hit_count.fetch_add(1, Ordering::Relaxed);
            }
            Ok(false)
        },
        TilePayload::Leaf(_) | TilePayload::Split(_) => {
            if let Some(slot_id) = tile.cache_put {
                let Some(ctx) = cache_ctx else {
                    bail!("received cache_put tile {idx} without active ClientTileCache");
                };
                let slot_idx = slot_id as usize;
                if slot_idx >= MAX_TILE_CACHE_SLOTS {
                    bail!("cache_put slot_id {slot_id} out of range (max {MAX_TILE_CACHE_SLOTS})");
                }
                if ctx.put_claimed[slot_idx].swap(true, Ordering::AcqRel) {
                    bail!("duplicate cache_put slot_id {slot_id} in frame");
                }
                let tile_bytes = rect.w as usize * rect.h as usize * 4;
                let prev_bytes = ctx.frame_put_bytes.fetch_add(tile_bytes, Ordering::Relaxed);
                if prev_bytes.saturating_add(tile_bytes) > MAX_TILE_CACHE_BYTES {
                    bail!(
                        "frame cache_put byte budget exceeds MAX_TILE_CACHE_BYTES ({MAX_TILE_CACHE_BYTES})"
                    );
                }

                // Decode directly into L2-hot `ClientTileCache` slot capacity
                // (without zero-filling) and, unless `puts_only`, stream the
                // committed slot into `frame` via non-temporal stores, avoiding
                // `wl_shm` RFO traffic and cache pollution.
                // SAFETY: `put_claimed[slot_idx]` grants exclusive ownership of `slot_id`.
                let slot_frame = unsafe {
                    ClientTileCache::prepare_slot_target_uninit(
                        &ctx.slots,
                        state.worker_idx,
                        slot_id,
                        rect,
                    )
                };
                // SAFETY: `slot_frame` has capacity for `rect.w * rect.h` pixels and is
                // exclusively owned; shifting by `dst_origin = (rect.x, rect.y)` maps
                // `rect` (and any validated `Split` partition) strictly over all
                // `0..rect.w * rect.h` pixels.
                let res = unsafe {
                    unfilter_tile::<K, HAS_ALPHA>(
                        kernel,
                        tier,
                        rect,
                        (rect.x as usize, rect.y as usize),
                        false,
                        &tile.payload,
                        &slot_frame,
                        state,
                    )
                };
                if let Err(err) = res {
                    // SAFETY: `put_claimed[slot_idx]` grants exclusive ownership of `slot_id`.
                    unsafe { ClientTileCache::invalidate_slot(&ctx.slots, slot_id) };
                    return Err(err);
                }
                // SAFETY: `unfilter_tile` returned `Ok`, guaranteeing every pixel in
                // `0..rect.w * rect.h` was written into `slot_frame`.
                let slot_pixels =
                    unsafe { ClientTileCache::commit_slot(&ctx.slots, slot_id, rect, HAS_ALPHA) };
                if ctx.puts_only {
                    return Ok(false);
                }
                // SAFETY: Caller validated `tile_rects` in-bounds and pairwise disjoint, and
                // `claimed[idx]` guarantees exclusive ownership of `rect` for this worker.
                let streamed = unsafe {
                    copy_pixels_to_frame_with(kernel, slot_pixels, rect.w as usize, rect, frame)
                };
                return Ok(streamed);
            }

            if cache_ctx.is_some_and(|ctx| ctx.puts_only) {
                return Ok(false);
            }
            // SAFETY: Caller validated `tile_rects` in-bounds and pairwise disjoint, and
            // `claimed[idx]` guarantees exclusive ownership of `rect` for this worker.
            unsafe {
                unfilter_tile::<K, HAS_ALPHA>(
                    kernel,
                    tier,
                    rect,
                    (0, 0),
                    true,
                    &tile.payload,
                    frame,
                    state,
                )
            }
        },
    }
}

/// # Safety
/// Caller must have validated `tile_rects` in-bounds within `frame` (unless
/// `cache_ctx.puts_only` is `true`) and pairwise disjoint.
#[inline(always)]
unsafe fn unfilter_tiles_worker_with<'a, K: RowKernel, const HAS_ALPHA: bool, F>(
    kernel: K,
    tier: arch::Tier,
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
    let mut streamed_any = false;
    while let Some(item) = next_tile() {
        // SAFETY: Caller validated `tile_rects` in-bounds and pairwise disjoint.
        let res = unsafe {
            decode_one_tile::<K, HAS_ALPHA>(
                kernel, tier, item, tile_rects, frame, claimed, cache_ctx, state,
            )
        };
        match res {
            Ok(used_stream) => {
                streamed_any |= used_stream;
                completed.fetch_add(1, Ordering::Relaxed);
            },
            Err(err) => {
                let mut guard = first_err.lock().unwrap();
                if guard.is_none() {
                    *guard = Some(err);
                }
                break;
            },
        }
    }
    if streamed_any {
        kernel.fence_stream();
    }
}

// --- tile partitioning & row-band merging ---

/// Sorts, merges overlapping/adjacent `[y0, y1)` intervals, and returns `None`
/// if merged damaged rows cover `>= 75%` of `height`.
pub(crate) fn merge_row_bands(mut bands: Vec<RowBand>, height: i32) -> Option<Vec<RowBand>> {
    if bands.is_empty() {
        return Some(bands);
    }

    let height_u32 = height as u32;
    bands.sort_unstable_by_key(|b| b.y0);
    let mut write_idx = 0;
    for read_idx in 1..bands.len() {
        let next = bands[read_idx];
        if next.y0 <= bands[write_idx].y1 {
            bands[write_idx].y1 = bands[write_idx].y1.max(next.y1);
        } else {
            write_idx += 1;
            bands[write_idx] = next;
        }
    }
    bands.truncate(write_idx + 1);

    let damaged_rows: u64 = bands.iter().map(|b| u64::from(b.y1 - b.y0)).sum();
    if damaged_rows * 4 >= u64::from(height_u32) * 3 {
        return None;
    }

    Some(bands)
}

/// Computes sorted, merged, non-overlapping damaged row-bands `[y0, y1)` when
/// updating an existing buffer of identical dimensions/format and total damaged
/// rows cover strictly less than 75% of `height`. Returns `None` when the
/// caller should take the zero-overhead full-frame path.
pub fn compute_partial_row_bands(
    damage: &[Rectangle<i32>],
    width: i32,
    height: i32,
) -> Option<Vec<RowBand>> {
    if width <= 0 || height <= 0 || damage.is_empty() {
        return None;
    }

    let height_u32 = height as u32;
    let mut bands = Vec::with_capacity(damage.len());
    for rect in damage {
        let x0 = rect.loc.x.clamp(0, width);
        let x1 = rect.loc.x.saturating_add(rect.size.w).clamp(0, width);
        let y0 = rect.loc.y.clamp(0, height) as u32;
        let y1 = rect.loc.y.saturating_add(rect.size.h).clamp(0, height) as u32;
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        if y0 == 0 && y1 == height_u32 {
            // Short-circuit common full-surface damage in O(1) with zero extra
            // work.
            return None;
        }
        bands.push(RowBand { y0, y1 });
    }

    merge_row_bands(bands, height)
}

/// Per-surface cached `<= 128x128` quadrant keyed 128-bit hashes synchronized
/// with [`ShadowBuffer`] pixels.
///
/// Each `128`-row grid band is padded to a multiple of 4 hashes (`64 B`
/// cache-line boundary) so parallel Phase-A workers updating disjoint bands
/// never share cache lines.
#[derive(Clone, Default)]
struct ShadowQuadHashes {
    width: usize,
    height: usize,
    stride: usize,
    has_alpha: bool,
    key_fp: ([u64; 2], [u64; 2]),
    n_qx: usize,
    n_qy: usize,
    stride_quads: usize,
    hashes: Vec<tile_hash::TileHash>,
    band_valid: Vec<u8>,
}

impl ShadowQuadHashes {
    #[inline]
    fn invalidate_all(&mut self) {
        self.band_valid.fill(0);
        self.n_qy = 0;
    }

    #[inline]
    fn invalidate_y_range(&mut self, y0: usize, y1: usize) {
        if self.n_qy == 0 || y0 >= y1 {
            return;
        }
        let qy0 = y0 / SUBTILE_H;
        let qy1 = y1.div_ceil(SUBTILE_H).min(self.n_qy);
        if qy0 < qy1 {
            self.band_valid[qy0..qy1].fill(0);
        }
    }

    #[inline]
    fn matches(
        &self,
        width: usize,
        height: usize,
        stride: usize,
        has_alpha: bool,
        key: &tile_hash::TileHashKey,
    ) -> bool {
        self.n_qy > 0
            && self.width == width
            && self.height == height
            && self.stride == stride
            && self.has_alpha == has_alpha
            && self.key_fp == key.fingerprint()
    }

    fn prepare_for(
        &mut self,
        width: usize,
        height: usize,
        stride: usize,
        has_alpha: bool,
        key: &tile_hash::TileHashKey,
    ) {
        let n_qx = width.div_ceil(SUBTILE_W);
        let n_qy = height.div_ceil(SUBTILE_H);
        let stride_quads = (n_qx + 3) & !3;
        self.width = width;
        self.height = height;
        self.stride = stride;
        self.has_alpha = has_alpha;
        self.key_fp = key.fingerprint();
        self.n_qx = n_qx;
        self.n_qy = n_qy;
        self.stride_quads = stride_quads;
        let needed = n_qy * stride_quads;
        if self.hashes.len() < needed {
            self.hashes.resize(needed, [0u8; 16]);
        }
        self.band_valid.clear();
        self.band_valid.resize(n_qy, 0);
    }

    #[inline]
    fn mark_all_valid(&mut self) {
        self.band_valid.fill(1);
    }

    #[inline]
    fn sync_view(&mut self) -> SyncQuadHashes {
        SyncQuadHashes {
            hashes_base: self.hashes.as_mut_ptr(),
            band_valid_base: self.band_valid.as_mut_ptr(),
            n_qx: self.n_qx,
            n_qy: self.n_qy,
            stride_quads: self.stride_quads,
        }
    }
}

/// Raw pointer handle into [`ShadowQuadHashes`] for parallel workers writing
/// strictly disjoint `(qy, qx)` quadrant slots or disjoint `qy` bands.
#[derive(Copy, Clone)]
pub(crate) struct SyncQuadHashes {
    hashes_base: *mut tile_hash::TileHash,
    band_valid_base: *mut u8,
    n_qx: usize,
    n_qy: usize,
    stride_quads: usize,
}

// SAFETY: Callers guarantee every parallel worker owns strictly disjoint
// `(qy, qx)` slots (during full-frame seeding) or strictly disjoint `qy` bands
// (during Phase-A incremental scanning).
unsafe impl Send for SyncQuadHashes {}
unsafe impl Sync for SyncQuadHashes {}

impl SyncQuadHashes {
    /// # Safety
    /// `qy < self.n_qy`, `qx < self.n_qx`, and no other thread may access
    /// `(qy, qx)` concurrently.
    #[inline]
    unsafe fn write_hash(&self, qy: usize, qx: usize, hash: tile_hash::TileHash) {
        debug_assert!(qy < self.n_qy && qx < self.n_qx);
        // SAFETY: Caller guarantees in-bounds exclusive access to `(qy, qx)`.
        unsafe {
            self.hashes_base
                .add(qy * self.stride_quads + qx)
                .write(hash);
        }
    }

    /// # Safety
    /// `qy < self.n_qy`, and no other thread may access band `qy` concurrently.
    #[inline]
    unsafe fn band_valid(&self, qy: usize) -> bool {
        debug_assert!(qy < self.n_qy);
        // SAFETY: Caller guarantees exclusive access to band `qy`.
        unsafe { self.band_valid_base.add(qy).read() != 0 }
    }

    /// # Safety
    /// `qy < self.n_qy`, and no other thread may access band `qy` concurrently.
    #[inline]
    unsafe fn set_band_valid(&self, qy: usize, valid: bool) {
        debug_assert!(qy < self.n_qy);
        // SAFETY: Caller guarantees exclusive access to band `qy`.
        unsafe {
            self.band_valid_base.add(qy).write(u8::from(valid));
        }
    }

    /// # Safety
    /// `qy < self.n_qy`, and no other thread may access band `qy` concurrently.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn band_hashes_mut(&self, qy: usize) -> &mut [tile_hash::TileHash] {
        debug_assert!(qy < self.n_qy);
        // SAFETY: Caller guarantees exclusive access to disjoint band `qy` of
        // length `self.n_qx <= self.stride_quads`.
        unsafe {
            slice::from_raw_parts_mut(self.hashes_base.add(qy * self.stride_quads), self.n_qx)
        }
    }
}

/// Borrowed Phase-A quadrant-hash screening context passed to non-scroll
/// workers.
#[derive(Copy, Clone)]
pub(crate) struct PhaseAQuadHashes {
    sync: SyncQuadHashes,
    height: usize,
    dense_dirty_bands: u64,
}

/// Per-surface server shadow pixel buffer paired with synchronized per-quadrant
/// keyed 128-bit hashes for half-DRAM Phase-A change screening.
///
/// Mutating raw pixels via [`Self::pixels_mut`] automatically invalidates all
/// cached quadrant hashes so hashes can never desync from pixel contents.
#[derive(Clone, Default)]
pub struct ShadowBuffer {
    pixels: Vec<u8>,
    quad_hashes: ShadowQuadHashes,
}

impl ShadowBuffer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn from_pixels(pixels: Vec<u8>) -> Self {
        Self {
            pixels,
            quad_hashes: ShadowQuadHashes::default(),
        }
    }

    #[inline]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.pixels
    }

    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        self.quad_hashes.invalidate_all();
        &mut self.pixels
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.pixels.clear();
        self.quad_hashes.invalidate_all();
    }

    #[inline]
    pub fn copy_from(&mut self, other: &Self) {
        self.pixels.clear();
        self.pixels.extend_from_slice(&other.pixels);
        self.quad_hashes.clone_from(&other.quad_hashes);
    }

    #[inline]
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.pixels.capacity()
    }

    #[cfg(test)]
    pub(crate) fn valid_band_count(&self) -> usize {
        if self.quad_hashes.n_qy == 0 {
            return 0;
        }
        self.quad_hashes
            .band_valid
            .iter()
            .filter(|&&v| v != 0)
            .count()
    }
}

/// Raw pointer handle to the server's per-surface shadow pixel buffer
/// preserving provenance for disjoint per-tile mutable writes across workers.
pub(crate) struct ShadowRows {
    base: *mut u8,
    stride: usize,
}

// SAFETY: Every candidate base tile is strictly in-bounds and pairwise disjoint
// and claimed at most once via `AtomicUsize`, so concurrent workers write to
// strictly disjoint pixel rectangles in `base`.
unsafe impl Send for ShadowRows {}
unsafe impl Sync for ShadowRows {}

impl ShadowRows {
    /// # Safety
    /// Caller must guarantee `(y, x_bytes..x_bytes + row_bytes)` lies within
    /// the shadow buffer.
    #[inline]
    unsafe fn row_ptr(&self, y: usize, x_bytes: usize) -> *mut u8 {
        // SAFETY: Caller upholds in-bounds invariants.
        unsafe { self.base.add(y * self.stride + x_bytes) }
    }
}

/// Copies `rect` from `src` (`wl_shm` via [`BufferPointer`]) into `shadow`
/// without reading destination bytes (safe when `shadow` points to
/// uninitialized spare capacity).
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow`, non-overlapping with
/// `src`, and disjoint from all regions concurrently accessed by other
/// threads.
#[inline]
unsafe fn copy_rect_to_shadow(
    src: BufferPointer<u8>,
    shadow: &ShadowRows,
    stride: usize,
    rect: TileRect,
) {
    let x_bytes = rect.x as usize * 4;
    let row_bytes = rect.w as usize * 4;
    let y0 = rect.y as usize;
    let h = rect.h as usize;
    if h > 0 && row_bytes > 0 {
        let last_end = (y0 + h - 1) * stride + x_bytes + row_bytes;
        assert!(last_end <= src.len());
    }
    for ry in 0..h {
        let row_off = (y0 + ry) * stride + x_bytes;
        // SAFETY: Validated `last_end <= src.len()` above, and caller
        // guarantees `rect` is in-bounds, non-overlapping with `src`, and
        // disjoint across workers. Raw pointer copy avoids forming `&mut [u8]`
        // over uninitialized spare capacity.
        unsafe {
            ptr::copy_nonoverlapping(
                src.ptr().add(row_off),
                shadow.row_ptr(y0 + ry, x_bytes),
                row_bytes,
            );
        }
    }
}

/// Copies `rect` from `src` (`wl_shm` via [`BufferPointer`]) into (possibly
/// uninitialized) `shadow` while computing [`RectScanMeta`] (`solid_px` and
/// `all_opaque`) in-register during the single pixel load pass.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow`, non-overlapping with
/// `src`, and disjoint from all regions concurrently accessed by other
/// threads.
#[inline(always)]
unsafe fn copy_and_scan_rect_to_shadow_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    src: BufferPointer<u8>,
    shadow: &ShadowRows,
    stride: usize,
    rect: TileRect,
) -> RectScanMeta {
    let x_bytes = rect.x as usize * 4;
    let w = rect.w as usize;
    let row_bytes = w * 4;
    let y0 = rect.y as usize;
    let h = rect.h as usize;
    if h == 0 || w == 0 {
        return RectScanMeta::EMPTY;
    }
    let last_end = (y0 + h - 1) * stride + x_bytes + row_bytes;
    assert!(last_end <= src.len());

    let base_off = y0 * stride + x_bytes;
    // SAFETY: Validated `last_end <= src.len()` and `h > 0 && w > 0`, so
    // `(0, 0)`, `(w / 2, h / 2)`, and `(w - 1, h - 1)` are all strictly in-bounds.
    let base = unsafe { src.ptr().add(base_off) };
    let (first_raw, mid_raw) = unsafe {
        (
            base.cast::<u32>().read_unaligned(),
            base.add((h >> 1) * stride + (w >> 1) * 4)
                .cast::<u32>()
                .read_unaligned(),
        )
    };
    let (first_word, mut solid, mut all_opaque) =
        tile_hash::probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
            base.add((h - 1) * stride + (w - 1) * 4)
                .cast::<u32>()
                .read_unaligned()
        });
    let pf_rows = stride > row_bytes && stride >= prefetch::MIN_PAGE_STRIDE_BYTES;
    if !solid && (!HAS_ALPHA || !all_opaque) {
        for ry in 0..h {
            if pf_rows && ry + prefetch::COLD_FILL_ROW_LOOKAHEAD < h {
                let pf_off = (y0 + ry + prefetch::COLD_FILL_ROW_LOOKAHEAD) * stride + x_bytes;
                // SAFETY: Validated `last_end <= src.len()` above and
                // `ry + COLD_FILL_ROW_LOOKAHEAD < h`.
                unsafe {
                    prefetch::row_nta(src.ptr().add(pf_off), row_bytes);
                }
            }
            let row_off = (y0 + ry) * stride + x_bytes;
            // SAFETY: Validated `last_end <= src.len()` above, and caller
            // guarantees `rect` is in-bounds, non-overlapping with `src`, and
            // disjoint across workers.
            unsafe {
                ptr::copy_nonoverlapping(
                    src.ptr().add(row_off),
                    shadow.row_ptr(y0 + ry, x_bytes),
                    row_bytes,
                );
            }
        }
        return RectScanMeta {
            solid_px: None,
            all_opaque: !HAS_ALPHA,
        };
    }

    for ry in 0..h {
        if pf_rows && ry + prefetch::COLD_FILL_ROW_LOOKAHEAD < h {
            let pf_off = (y0 + ry + prefetch::COLD_FILL_ROW_LOOKAHEAD) * stride + x_bytes;
            // SAFETY: Validated `last_end <= src.len()` above and
            // `ry + COLD_FILL_ROW_LOOKAHEAD < h`.
            unsafe {
                prefetch::row_nta(src.ptr().add(pf_off), row_bytes);
            }
        }
        let row_off = (y0 + ry) * stride + x_bytes;
        if !solid && (!HAS_ALPHA || !all_opaque) {
            // SAFETY: Validated `last_end <= src.len()` above, and caller
            // guarantees `rect` is in-bounds, non-overlapping with `src`, and
            // disjoint across workers.
            unsafe {
                ptr::copy_nonoverlapping(
                    src.ptr().add(row_off),
                    shadow.row_ptr(y0 + ry, x_bytes),
                    row_bytes,
                );
            }
            continue;
        }
        // SAFETY: Validated `last_end <= src.len()` above, and caller
        // guarantees `rect` is in-bounds, non-overlapping with `src`, and
        // disjoint across workers.
        let (row_solid, row_opaque) = unsafe {
            kernel.copy_and_scan_row::<HAS_ALPHA>(
                src.ptr().add(row_off),
                shadow.row_ptr(y0 + ry, x_bytes),
                w,
                first_word,
            )
        };
        solid &= row_solid;
        all_opaque &= row_opaque;
    }
    RectScanMeta {
        solid_px: solid.then(|| first_word.to_ne_bytes()),
        all_opaque: !HAS_ALPHA || all_opaque,
    }
}

/// Zero-initializes trailing row-stride padding (`width_bytes..shadow.stride`)
/// for rows `rect.y..rect.y + rect.h` when `rect` touches the right edge of the
/// frame.
///
/// # Safety
/// `(rect.y + rect.h) * shadow.stride` must lie within `shadow`'s allocated
/// capacity, `width_bytes <= shadow.stride`, and no other thread may
/// concurrently access trailing padding bytes of rows `rect.y..rect.y + rect.h`.
#[inline]
unsafe fn zero_shadow_row_padding(shadow: &ShadowRows, width_bytes: usize, rect: TileRect) {
    let pad_bytes = shadow.stride - width_bytes;
    if pad_bytes != 0 && (rect.x + rect.w) as usize * 4 == width_bytes {
        let y0 = rect.y as usize;
        for ry in 0..rect.h as usize {
            // SAFETY: Caller guarantees `(y0 + ry, width_bytes..shadow.stride)`
            // lies within `shadow` and is exclusively owned by this right-edge
            // rect.
            unsafe {
                ptr::write_bytes(shadow.row_ptr(y0 + ry, width_bytes), 0, pad_bytes);
            }
        }
    }
}

/// Disjoint horizontal scan unit executed in parallel during Phase A so each
/// worker streams linearly across contiguous rows instead of striding across
/// `128x128` quadrants.
enum ScanSubBand {
    RowBand2D { y0: u32, y1: u32, slot_base: usize },
    Strip1D { rect: TileRect, slot_idx: usize },
}

/// Maps each `256x256` base tile (or `< 8`-row 1D strip) to its dirty-flag
/// slots populated by Phase A so the main thread coalesces all-dirty quadrants
/// back into a single `256x256` tile in deterministic raster order.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum BaseScanEntry {
    Strip1D {
        rect: TileRect,
        slot_idx: usize,
    },
    Tile2D {
        rect: TileRect,
        quad_slots: [usize; 4],
        n_quads: usize,
    },
}

impl BaseScanEntry {
    #[inline]
    fn rect(self) -> TileRect {
        match self {
            Self::Strip1D { rect, .. } | Self::Tile2D { rect, .. } => rect,
        }
    }
}

fn plan_incremental_scan(
    metadata: &BufferMetadata,
    bands: Option<&[RowBand]>,
) -> Result<(Vec<ScanSubBand>, Vec<BaseScanEntry>, usize)> {
    let width = metadata.width as u32;
    let height = metadata.height as u32;
    let regions: Vec<RowBand> = match bands {
        None => vec![RowBand { y0: 0, y1: height }],
        Some(bands) => {
            let mut prev_y1 = 0u32;
            for band in bands {
                if band.y0 < prev_y1 || band.y0 >= band.y1 || band.y1 > height {
                    bail!(
                        "invalid or overlapping RowBand [{}..{}) for height {height}",
                        band.y0,
                        band.y1
                    );
                }
                prev_y1 = band.y1;
            }
            bands.to_vec()
        },
    };

    let n_qx = (width as usize).div_ceil(SUBTILE_W);
    let mut sub_bands = Vec::new();
    let mut base_entries = Vec::new();
    let mut total_slots = 0usize;

    for region in regions {
        let band_h = region.y1 - region.y0;
        if band_h < MIN_2D_BAND_HEIGHT {
            let max_strip_w = (TILE_PIXELS as u32 / band_h).max(1);
            let mut x = 0u32;
            while x < width {
                let w = (width - x).min(max_strip_w);
                let rect = TileRect {
                    x,
                    y: region.y0,
                    w,
                    h: band_h,
                };
                let slot_idx = total_slots;
                total_slots += 1;
                sub_bands.push(ScanSubBand::Strip1D { rect, slot_idx });
                base_entries.push(BaseScanEntry::Strip1D { rect, slot_idx });
                x += w;
            }
            continue;
        }

        let mut y = region.y0;
        while y < region.y1 {
            let tile_h = (region.y1 - y).min(TILE_H as u32);
            let top_h = tile_h.min(SUBTILE_H as u32);
            let top_base = total_slots;
            total_slots += n_qx;
            sub_bands.push(ScanSubBand::RowBand2D {
                y0: y,
                y1: y + top_h,
                slot_base: top_base,
            });
            let bot_base = if tile_h > top_h {
                let base = total_slots;
                total_slots += n_qx;
                sub_bands.push(ScanSubBand::RowBand2D {
                    y0: y + top_h,
                    y1: y + tile_h,
                    slot_base: base,
                });
                Some(base)
            } else {
                None
            };

            let mut x = 0u32;
            while x < width {
                let tile_w = (width - x).min(TILE_W as u32);
                let qx0 = x as usize / SUBTILE_W;
                let mut quad_slots = [0usize; 4];
                let mut n_quads = 0usize;
                quad_slots[n_quads] = top_base + qx0;
                n_quads += 1;
                if tile_w > SUBTILE_W as u32 {
                    quad_slots[n_quads] = top_base + qx0 + 1;
                    n_quads += 1;
                }
                if let Some(bot) = bot_base {
                    quad_slots[n_quads] = bot + qx0;
                    n_quads += 1;
                    if tile_w > SUBTILE_W as u32 {
                        quad_slots[n_quads] = bot + qx0 + 1;
                        n_quads += 1;
                    }
                }
                base_entries.push(BaseScanEntry::Tile2D {
                    rect: TileRect {
                        x,
                        y,
                        w: tile_w,
                        h: tile_h,
                    },
                    quad_slots,
                    n_quads,
                });
                x += tile_w;
            }
            y += tile_h;
        }
    }

    Ok((sub_bands, base_entries, total_slots))
}

/// Packed per-quadrant / per-strip dirty state written by Phase A into
/// `dirty_slots: &[AtomicU32]`.
///
/// Encoding:
/// - `0`: clean ([`SlotDirty::Clean`])
/// - `u32::MAX`: full quadrant or 1D strip dirty ([`SlotDirty::Full`])
/// - Otherwise: cropped tight bounding box relative to the quadrant origin:
///   - bits `0..7`: `min_x` (`0..128`)
///   - bit `7`: `has_quad_hash` (lookup-only full-quadrant keyed 128-bit hash
///     written to [`SlotScans`])
///   - bits `8..16`: `max_x` (`1..=128`, exclusive)
///   - bits `16..24`: `min_y` (`0..128`)
///   - bits `24..32`: `max_y` (`1..=128`, exclusive)
///
/// Because `1 <= max_x <= 128` and `1 <= max_y <= 128`, a cropped value can
/// never equal `0` or `u32::MAX`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum SlotDirty {
    Clean,
    Full,
    Cropped {
        min_x: u8,
        max_x: u8,
        min_y: u8,
        max_y: u8,
        has_quad_hash: bool,
    },
}

impl SlotDirty {
    pub(crate) const CLEAN_RAW: u32 = 0;
    pub(crate) const FULL_RAW: u32 = u32::MAX;

    #[inline]
    pub(crate) fn encode_cropped(
        min_x: u8,
        max_x: u8,
        min_y: u8,
        max_y: u8,
        has_quad_hash: bool,
    ) -> u32 {
        debug_assert!(min_x < max_x && max_x as usize <= SUBTILE_W);
        debug_assert!(min_y < max_y && max_y as usize <= SUBTILE_H);
        let raw = u32::from(min_x)
            | (u32::from(has_quad_hash) << 7)
            | (u32::from(max_x) << 8)
            | (u32::from(min_y) << 16)
            | (u32::from(max_y) << 24);
        debug_assert!(raw != Self::CLEAN_RAW && raw != Self::FULL_RAW);
        raw
    }

    #[inline]
    pub(crate) fn decode(raw: u32) -> Self {
        match raw {
            Self::CLEAN_RAW => Self::Clean,
            Self::FULL_RAW => Self::Full,
            v => Self::Cropped {
                min_x: (v & 0x7F) as u8,
                max_x: ((v >> 8) & 0xFF) as u8,
                min_y: ((v >> 16) & 0xFF) as u8,
                max_y: ((v >> 24) & 0xFF) as u8,
                has_quad_hash: (v & 0x80) != 0,
            },
        }
    }

    #[inline]
    pub(crate) fn cropped_rect(
        quad: TileRect,
        min_x: u8,
        max_x: u8,
        min_y: u8,
        max_y: u8,
    ) -> TileRect {
        TileRect {
            x: quad.x + u32::from(min_x),
            y: quad.y + u32::from(min_y),
            w: u32::from(max_x - min_x),
            h: u32::from(max_y - min_y),
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct QuadBbox {
    min_x: u8,
    max_x: u8,
    min_y: u8,
    max_y: u8,
    full: bool,
}

impl QuadBbox {
    const CLEAN: Self = Self {
        min_x: u8::MAX,
        max_x: 0,
        min_y: u8::MAX,
        max_y: 0,
        full: false,
    };

    #[inline(always)]
    fn is_dirty(self) -> bool {
        self.full || self.max_x != 0
    }

    /// Expands the tight bounding box with a dirty row span `[first_x, last_x_end)`
    /// at quadrant-local row `ry`, escalating to `full = true` (and returning
    /// `true`) if the expanded area exceeds `max_crop_area`.
    #[inline(always)]
    fn record_row(
        &mut self,
        ry: usize,
        first_x: usize,
        last_x_end: usize,
        max_crop_area: usize,
    ) -> bool {
        let fx = first_x as u8;
        let lx = last_x_end as u8;
        let ry_u8 = ry as u8;
        if self.max_x == 0 {
            self.min_x = fx;
            self.max_x = lx;
            self.min_y = ry_u8;
            self.max_y = ry_u8 + 1;
        } else {
            self.min_x = self.min_x.min(fx);
            self.max_x = self.max_x.max(lx);
            self.min_y = self.min_y.min(ry_u8);
            self.max_y = self.max_y.max(ry_u8 + 1);
        }
        let area = usize::from(self.max_x - self.min_x) * usize::from(self.max_y - self.min_y);
        if area > max_crop_area {
            self.full = true;
            return true;
        }
        false
    }
}

/// Maps a quadrant `quad` inside `parent` (`<= 256x256`) to its 2x2 grid slot:
/// `0 = TL, 1 = TR, 2 = BL, 3 = BR`.
#[inline]
pub(crate) fn quad_grid_slot(parent: TileRect, quad: TileRect) -> usize {
    let col = usize::from(quad.x > parent.x);
    let row = usize::from(quad.y > parent.y);
    row * 2 + col
}

#[inline]
fn try_merge_adjacent(a: TileRect, b: TileRect, horizontal: bool) -> Option<TileRect> {
    let touches = if horizontal {
        a.x + a.w == b.x && a.y < b.y + b.h && b.y < a.y + a.h
    } else {
        a.y + a.h == b.y && a.x < b.x + b.w && b.x < a.x + a.w
    };
    if !touches {
        return None;
    }
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    let merged = TileRect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    };
    let merged_area = u64::from(merged.w) * u64::from(merged.h);
    let sum_area = u64::from(a.w) * u64::from(a.h) + u64::from(b.w) * u64::from(b.h);
    if merged_area <= TILE_PIXELS as u64
        && (merged_area <= sum_area + 256 || merged_area * 8 <= sum_area * 9)
    {
        Some(merged)
    } else {
        None
    }
}

/// Merges touching cropped-miss bounding boxes across internal quadrant seams
/// of a single `<= 256x256` tile (`[TL, TR, BL, BR]`) without ever overlapping
/// any `None` (clean, full, or cache-hit) sibling quadrant, combining their
/// [`RectScanMeta`] conservatively (`RectScanMeta::combine`).
#[inline]
pub(crate) fn merge_cropped_quad_slots(grid: &mut [Option<(TileRect, RectScanMeta)>; 4]) {
    let mut merged_top = false;
    if let (Some((a, ma)), Some((b, mb))) = (grid[0], grid[1])
        && let Some(m) = try_merge_adjacent(a, b, true)
    {
        grid[0] = Some((m, RectScanMeta::combine(&[ma, mb])));
        grid[1] = None;
        merged_top = true;
    }

    let mut merged_bot = false;
    if let (Some((a, ma)), Some((b, mb))) = (grid[2], grid[3])
        && let Some(m) = try_merge_adjacent(a, b, true)
    {
        grid[2] = Some((m, RectScanMeta::combine(&[ma, mb])));
        grid[3] = None;
        merged_bot = true;
    }

    // Only attempt vertical merges when both rows merged horizontally (all 4
    // quadrants were cropped misses) or neither did (each rect remains inside
    // its own column), preventing a wide `TL+TR` merge from overlapping `BR`.
    if merged_top == merged_bot {
        if let (Some((a, ma)), Some((b, mb))) = (grid[0], grid[2])
            && let Some(m) = try_merge_adjacent(a, b, false)
        {
            grid[0] = Some((m, RectScanMeta::combine(&[ma, mb])));
            grid[2] = None;
        }
        if let (Some((a, ma)), Some((b, mb))) = (grid[1], grid[3])
            && let Some(m) = try_merge_adjacent(a, b, false)
        {
            grid[1] = Some((m, RectScanMeta::combine(&[ma, mb])));
            grid[3] = None;
        }
    }
}

/// Finalizes a quadrant that just escalated to `bbox.full` at quadrant-local
/// row `ry` (so rows `0..=ry` are already updated in `shadow`), fusing the
/// remaining copy (`ry + 1..ch`) and keyed 128-bit hashing in one register pass
/// when `slot_scans` is present, or copying remaining rows directly into
/// `shadow` otherwise.
///
/// # Safety
/// `qrect` must be strictly in-bounds within `shadow`, exclusively owned by
/// this worker, with rows `[0..=ry]` already initialized in `shadow`.
#[inline(never)]
unsafe fn finish_full_quadrant<const HAS_ALPHA: bool>(
    data: BufferPointer<u8>,
    shadow: &ShadowRows,
    stride: usize,
    qrect: TileRect,
    copy_from_ry: usize,
    slot_idx: usize,
    slot: &AtomicU32,
    slot_scans: Option<&SlotScans>,
) {
    if let Some(scans) = slot_scans {
        // SAFETY: Caller guarantees `qrect` is in-bounds and exclusively owned,
        // and rows `0..copy_from_ry` are already initialized in `shadow`.
        let scan = unsafe {
            tile_hash::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                scans.key(),
                data,
                shadow.base,
                stride,
                qrect,
                copy_from_ry,
            )
        };
        // SAFETY: `slot_idx` is exclusively owned by this worker.
        unsafe { scans.write(slot_idx, scan) };
    } else if copy_from_ry < qrect.h as usize {
        let rem_rect = TileRect {
            x: qrect.x,
            y: qrect.y + copy_from_ry as u32,
            w: qrect.w,
            h: qrect.h - copy_from_ry as u32,
        };
        // SAFETY: Caller guarantees `qrect` is in-bounds and exclusively owned.
        unsafe { copy_rect_to_shadow(data, shadow, stride, rem_rect) };
    }
    slot.store(SlotDirty::FULL_RAW, Ordering::Relaxed);
}

/// Scans a `< 8`-row 1D strip linearly against `shadow`, updating `shadow`
/// through the first differing row and returning `Some(ry + 1)` (the first
/// remaining uncopied row index), or `None` when the strip is clean.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
#[inline(always)]
unsafe fn diff_and_update_strip_prefix_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    stride_px: usize,
    rect: TileRect,
) -> Option<usize> {
    let x0 = rect.x as usize;
    let x_bytes = x0 * 4;
    let w = rect.w as usize;
    let y0 = rect.y as usize;
    let h = rect.h as usize;

    for ry in 0..h {
        let (_, after) = aos.split_at((y0 + ry) * stride_px + x0);
        let (src_row, _) = after.split_at(w);
        // SAFETY: Caller guarantees `rect` is in-bounds and disjoint across workers,
        // and `Vec4u8` is `repr(C, packed)` (align 1).
        let dst_row = unsafe {
            slice::from_raw_parts_mut(shadow.row_ptr(y0 + ry, x_bytes).cast::<Vec4u8>(), w)
        };
        if kernel
            .diff_and_update_row::<HAS_ALPHA>(src_row, dst_row)
            .is_some()
        {
            return Some(ry + 1);
        }
    }
    None
}

/// Scans a `< 8`-row 1D strip linearly against `shadow`, updating `shadow`
/// from the first differing pixel onward.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
#[cfg(test)]
#[inline(always)]
unsafe fn diff_and_update_strip_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    stride_px: usize,
    rect: TileRect,
) -> bool {
    let Some(copy_from_ry) = (unsafe {
        diff_and_update_strip_prefix_with::<K, HAS_ALPHA>(kernel, aos, shadow, stride_px, rect)
    }) else {
        return false;
    };
    let x0 = rect.x as usize;
    let x_bytes = x0 * 4;
    let w = rect.w as usize;
    let y0 = rect.y as usize;
    let h = rect.h as usize;
    for rem_ry in copy_from_ry..h {
        let (_, rem_after) = aos.split_at((y0 + rem_ry) * stride_px + x0);
        let (rem_src, _) = rem_after.split_at(w);
        // SAFETY: Caller guarantees `rect` is in-bounds and disjoint across workers.
        let rem_dst = unsafe {
            slice::from_raw_parts_mut(shadow.row_ptr(y0 + rem_ry, x_bytes).cast::<Vec4u8>(), w)
        };
        rem_src.copy_to_nonoverlapping(rem_dst);
    }
    true
}

/// Scans a `< 8`-row 1D strip linearly against `shadow`, updating `shadow`
/// from the first differing pixel onward and fusing the remaining row copy with
/// keyed 128-bit hashing when `slot_scans` is present.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
#[inline(always)]
unsafe fn diff_and_update_strip_slot_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    data: BufferPointer<u8>,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    stride: usize,
    stride_px: usize,
    rect: TileRect,
    slot_idx: usize,
    slot: &AtomicU32,
    slot_scans: Option<&SlotScans>,
) {
    // SAFETY: Caller guarantees `rect` is in-bounds and disjoint across workers.
    if let Some(copy_from_ry) = unsafe {
        diff_and_update_strip_prefix_with::<K, HAS_ALPHA>(kernel, aos, shadow, stride_px, rect)
    } {
        // SAFETY: Rows `0..copy_from_ry` are updated in `shadow`; `rect` is
        // exclusively owned by this worker.
        unsafe {
            finish_full_quadrant::<HAS_ALPHA>(
                data,
                shadow,
                stride,
                rect,
                copy_from_ry,
                slot_idx,
                slot,
                slot_scans,
            );
        }
    }
}

/// Borrowed parameters for completing an active `RowBand2D` once at least one
/// quadrant has escalated to `bbox.full` or crossed `CROP_CACHE_LOOKUP_MIN_PIXELS`.
#[derive(Copy, Clone)]
pub(crate) struct ActiveRowBandCtx<'a> {
    pub(crate) data: BufferPointer<'a, u8>,
    pub(crate) aos: BufferPointer<'a, Vec4u8>,
    pub(crate) shadow: &'a ShadowRows,
    pub(crate) width: usize,
    pub(crate) stride: usize,
    pub(crate) y0: usize,
    pub(crate) y1: usize,
    pub(crate) slot_base: usize,
    pub(crate) dirty_slots: &'a [AtomicU32],
    pub(crate) slot_scans: Option<&'a SlotScans>,
    pub(crate) cache_non_empty: bool,
}

/// Portable fallback for [`RowKernel::finish_active_row_band_2d`], streaming
/// remaining rows `[trigger_y + 1, ctx.y1)` linearly with incremental
/// [`tile_hash::RowTileHasher`] updates (and `copy_and_absorb_band_rows` once
/// all quadrants reach `bbox.full`).
///
/// # Safety
/// Caller must uphold [`RowKernel::finish_active_row_band_2d`] invariants.
#[inline(always)]
pub(crate) unsafe fn finish_active_row_band_2d_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    ctx: &ActiveRowBandCtx<'_>,
    qx_bboxes: &mut [QuadBbox],
    trigger_y: usize,
    mut full_count: usize,
) {
    use tile_hash::BandRowAction;
    use tile_hash::RowTileHasher;

    let width = ctx.width;
    let stride = ctx.stride;
    let stride_px = stride / 4;
    let y0 = ctx.y0;
    let y1 = ctx.y1;
    let ch = y1 - y0;
    let n_qx = ctx.dirty_slots.len();

    if let Some(scans) = ctx.slot_scans {
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

        for (bbox, act) in qx_bboxes.iter().zip(qx_actions.iter_mut()) {
            if bbox.full {
                *act = BandRowAction::StartFull;
            } else if ctx.cache_non_empty && bbox.is_dirty() {
                let area =
                    usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
                if area >= CROP_CACHE_LOOKUP_MIN_PIXELS {
                    *act = BandRowAction::StartAbsorb;
                }
            }
        }
        // SAFETY: Rows `[y0, trigger_y]` are finalized in `shadow`.
        unsafe {
            tile_hash::process_band_row::<HAS_ALPHA>(
                scans.key(),
                ctx.data,
                ctx.shadow.base,
                stride,
                width,
                y0,
                ch,
                trigger_y,
                qx_actions,
                qx_hashers,
            );
        }

        for y in (trigger_y + 1)..y1 {
            if full_count == n_qx {
                // SAFETY: Every quadrant reached `bbox.full` on an earlier row
                // and transitioned its hasher to `CopyAndAbsorb`.
                unsafe {
                    tile_hash::copy_and_absorb_band_rows::<HAS_ALPHA>(
                        scans.key(),
                        ctx.data,
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
            let (_, row_after) = ctx.aos.split_at(y * stride_px);
            let (mut rem_aos, _) = row_after.split_at(width);
            // SAFETY: `[y0, y1)` is strictly disjoint across workers and `Vec4u8` has alignment 1.
            let dst_row = unsafe {
                slice::from_raw_parts_mut(ctx.shadow.row_ptr(y, 0).cast::<Vec4u8>(), width)
            };

            let mut x = 0usize;
            for (bbox, act) in qx_bboxes.iter_mut().zip(qx_actions.iter_mut()) {
                let cw = (width - x).min(SUBTILE_W);
                let (src_chunk, rest_aos) = rem_aos.split_at(cw);
                rem_aos = rest_aos;
                if !bbox.full {
                    let dst_chunk = &mut dst_row[x..x + cw];
                    if let Some((fx, lx)) =
                        kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
                    {
                        let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
                        let became_full = bbox.record_row(ry, fx, lx, max_crop_area);
                        if became_full {
                            full_count += 1;
                        }
                        if *act == BandRowAction::Skip {
                            let area = usize::from(bbox.max_x - bbox.min_x)
                                * usize::from(bbox.max_y - bbox.min_y);
                            if became_full
                                || (ctx.cache_non_empty && area >= CROP_CACHE_LOOKUP_MIN_PIXELS)
                            {
                                *act = if became_full {
                                    BandRowAction::StartFull
                                } else {
                                    BandRowAction::StartAbsorb
                                };
                            }
                        } else if became_full {
                            *act = BandRowAction::AbsorbShadowThenFull;
                        }
                    }
                }
                x += cw;
            }

            // SAFETY: `[y0, y1)` is exclusively owned and `dst_row` has updated
            // all `!bbox.full` quadrants for row `y`.
            unsafe {
                tile_hash::process_band_row::<HAS_ALPHA>(
                    scans.key(),
                    ctx.data,
                    ctx.shadow.base,
                    stride,
                    width,
                    y0,
                    ch,
                    y,
                    qx_actions,
                    qx_hashers,
                );
            }
        }

        // SAFETY: Every non-`Skip` slot was initialized and absorbed all `ch` rows.
        unsafe {
            tile_hash::finish_band_hashers(
                scans.key(),
                qx_actions,
                qx_hashers,
                ctx.slot_base,
                scans,
            );
        }
    } else {
        for y in (trigger_y + 1)..y1 {
            if full_count == n_qx {
                let vis_bytes = width * 4;
                if stride == vis_bytes {
                    let off = y * stride;
                    let bytes = (y1 - y) * stride;
                    let (_, after) = ctx.data.split_at(off);
                    let (chunk, _) = after.split_at(bytes);
                    // SAFETY: `[y0, y1)` is strictly disjoint across workers.
                    let dst = unsafe { slice::from_raw_parts_mut(ctx.shadow.row_ptr(y, 0), bytes) };
                    chunk.copy_to_nonoverlapping(dst);
                } else {
                    for ry in y..y1 {
                        let off = ry * stride;
                        let (_, after) = ctx.data.split_at(off);
                        let (chunk, _) = after.split_at(vis_bytes);
                        // SAFETY: `[y0, y1)` is strictly disjoint across workers.
                        let dst = unsafe {
                            slice::from_raw_parts_mut(ctx.shadow.row_ptr(ry, 0), vis_bytes)
                        };
                        chunk.copy_to_nonoverlapping(dst);
                    }
                }
                break;
            }

            let ry = y - y0;
            let (_, row_after) = ctx.aos.split_at(y * stride_px);
            let (mut rem_aos, _) = row_after.split_at(width);
            // SAFETY: `[y0, y1)` is strictly disjoint across workers and `Vec4u8` has alignment 1.
            let dst_row = unsafe {
                slice::from_raw_parts_mut(ctx.shadow.row_ptr(y, 0).cast::<Vec4u8>(), width)
            };

            let mut x = 0usize;
            for bbox in qx_bboxes.iter_mut() {
                let cw = (width - x).min(SUBTILE_W);
                let (src_chunk, rest_aos) = rem_aos.split_at(cw);
                rem_aos = rest_aos;
                let dst_chunk = &mut dst_row[x..x + cw];
                if bbox.full {
                    src_chunk.copy_to_nonoverlapping(dst_chunk);
                } else if let Some((fx, lx)) =
                    kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
                {
                    let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
                    if bbox.record_row(ry, fx, lx, max_crop_area) {
                        full_count += 1;
                    }
                }
                x += cw;
            }
        }
    }

    let has_scans = ctx.slot_scans.is_some();
    for (slot, bbox) in ctx.dirty_slots.iter().zip(qx_bboxes.iter()) {
        if bbox.full {
            slot.store(SlotDirty::FULL_RAW, Ordering::Relaxed);
        } else if bbox.is_dirty() {
            let crop_area =
                usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
            let has_quad_hash =
                ctx.cache_non_empty && has_scans && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
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

/// Scans a horizontal `<= 128`-row 2D sub-band linearly row-by-row across all
/// `n_qx` horizontal quadrants, keeping clean / small-crop bands completely
/// free of hasher state, `bbox.full` branches, and per-row function calls (so
/// SIMD mask constants remain live in registers with zero stack spills), and
/// handing off to [`RowKernel::finish_active_row_band_2d`] on the first row
/// where any quadrant escalates to `bbox.full` or reaches
/// `CROP_CACHE_LOOKUP_MIN_PIXELS`.
///
/// # Safety
/// Rows `[y0, y1)` must be strictly in-bounds within `shadow` and disjoint from
/// all rows concurrently accessed by other threads.
#[inline(always)]
pub(crate) unsafe fn diff_and_update_row_band_2d_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    ctx: &ActiveRowBandCtx<'_>,
) {
    let width = ctx.width;
    let stride = ctx.stride;
    let stride_px = stride / 4;
    let y0 = ctx.y0;
    let y1 = ctx.y1;
    let ch = y1 - y0;
    let n_qx = ctx.dirty_slots.len();
    let mut stack_bboxes = [QuadBbox::CLEAN; STACK_QX_CAP];
    let mut heap_bboxes;
    let qx_bboxes: &mut [QuadBbox] = if n_qx <= STACK_QX_CAP {
        &mut stack_bboxes[..n_qx]
    } else {
        heap_bboxes = vec![QuadBbox::CLEAN; n_qx];
        &mut heap_bboxes
    };

    let has_scans = ctx.slot_scans.is_some();
    let mut full_count = 0usize;

    for y in y0..y1 {
        let ry = y - y0;
        let (_, row_after) = ctx.aos.split_at(y * stride_px);
        let (mut rem_aos, _) = row_after.split_at(width);
        // SAFETY: `[y0, y1)` is strictly disjoint across workers and `Vec4u8` has alignment 1.
        let dst_row =
            unsafe { slice::from_raw_parts_mut(ctx.shadow.row_ptr(y, 0).cast::<Vec4u8>(), width) };

        let mut row_triggered = false;
        let mut x = 0usize;
        for bbox in qx_bboxes.iter_mut() {
            let cw = (width - x).min(SUBTILE_W);
            let (src_chunk, rest_aos) = rem_aos.split_at(cw);
            rem_aos = rest_aos;
            let dst_chunk = &mut dst_row[x..x + cw];
            if let Some((fx, lx)) = kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk) {
                let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
                let became_full = bbox.record_row(ry, fx, lx, max_crop_area);
                if became_full {
                    full_count += 1;
                    row_triggered = true;
                } else if has_scans && ctx.cache_non_empty {
                    let area =
                        usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
                    if area >= CROP_CACHE_LOOKUP_MIN_PIXELS {
                        row_triggered = true;
                    }
                }
            }
            x += cw;
        }

        if row_triggered {
            // SAFETY: Rows `[y0, y]` are updated in `shadow` and `[y0, y1)` is exclusively owned.
            unsafe {
                kernel.finish_active_row_band_2d::<HAS_ALPHA>(ctx, qx_bboxes, y, full_count);
            }
            return;
        }
    }

    for (slot, bbox) in ctx.dirty_slots.iter().zip(qx_bboxes.iter()) {
        if bbox.is_dirty() {
            slot.store(
                SlotDirty::encode_cropped(bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, false),
                Ordering::Relaxed,
            );
        }
    }
}

/// Diffs a single `<= 128x128` quadrant `qx` inside `ctx` against `shadow`
/// (preserving exact tight dirty bbox cropping and `finish_full_quadrant`
/// escalation), updates `ctx.dirty_slots[qx]` and `scans` (when full or
/// `>= CROP_CACHE_LOOKUP_MIN_PIXELS`), and refreshes `prev_hash` strictly from
/// finalized `shadow` (TOCTOU-safe). Returns `true` iff any pixel differed.
///
/// # Safety
/// Quadrant `qx` over `[ctx.y0, ctx.y1)` must be strictly in-bounds within
/// `ctx.shadow` and exclusively owned by this worker.
#[inline(always)]
pub(crate) unsafe fn diff_and_update_single_quad_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    ctx: &ActiveRowBandCtx<'_>,
    scans: &SlotScans,
    qx: usize,
    prev_hash: &mut tile_hash::TileHash,
) -> bool {
    let x0 = qx * SUBTILE_W;
    let cw = (ctx.width - x0).min(SUBTILE_W);
    let y0 = ctx.y0;
    let y1 = ctx.y1;
    let ch = y1 - y0;
    let stride = ctx.stride;
    let stride_px = stride / 4;
    let x_bytes = x0 * 4;
    let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
    let qrect = TileRect {
        x: x0 as u32,
        y: y0 as u32,
        w: cw as u32,
        h: ch as u32,
    };
    let slot_idx = ctx.slot_base + qx;
    let mut bbox = QuadBbox::CLEAN;

    for y in y0..y1 {
        let (_, row_after) = ctx.aos.split_at(y * stride_px + x0);
        let (src_chunk, _) = row_after.split_at(cw);
        // SAFETY: Quadrant `qx` over `[y0, y1)` is exclusively owned by this worker.
        let dst_chunk = unsafe {
            slice::from_raw_parts_mut(ctx.shadow.row_ptr(y, x_bytes).cast::<Vec4u8>(), cw)
        };
        if let Some((fx, lx)) = kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
            && bbox.record_row(y - y0, fx, lx, max_crop_area)
        {
            // SAFETY: Rows `0..=(y - y0)` are finalized in `shadow`, and `slot_idx`
            // is exclusively owned by this worker.
            unsafe {
                finish_full_quadrant::<HAS_ALPHA>(
                    ctx.data,
                    ctx.shadow,
                    stride,
                    qrect,
                    (y - y0) + 1,
                    slot_idx,
                    &ctx.dirty_slots[qx],
                    Some(scans),
                );
                *prev_hash = scans.read(slot_idx).hash;
            }
            return true;
        }
    }

    if !bbox.is_dirty() {
        return false;
    }

    let crop_area = usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
    let has_quad_hash = ctx.cache_non_empty && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
    if has_quad_hash {
        // SAFETY: `qrect` is finalized in `shadow` and exclusively owned.
        let scan = unsafe {
            tile_hash::hash_rect_from_shadow::<HAS_ALPHA>(
                scans.key(),
                ctx.shadow.base.cast_const(),
                stride,
                qrect,
            )
        };
        // SAFETY: `slot_idx` is exclusively owned by this worker.
        unsafe { scans.write(slot_idx, scan) };
        *prev_hash = scan.hash;
    } else {
        // SAFETY: `qrect` is finalized in `shadow` (L2-hot from the diff above)
        // and exclusively owned.
        *prev_hash = unsafe {
            tile_hash::hash_rect_only_from_shadow::<HAS_ALPHA>(
                scans.key(),
                ctx.shadow.base.cast_const(),
                stride,
                qrect,
            )
        };
    }
    ctx.dirty_slots[qx].store(
        SlotDirty::encode_cropped(
            bbox.min_x,
            bbox.max_x,
            bbox.min_y,
            bbox.max_y,
            has_quad_hash,
        ),
        Ordering::Relaxed,
    );
    true
}

/// Synchronizes `prev_hashes[start_qx..]` after `diff_and_update_row_band_2d`
/// finished columns `start_qx..n_qx` of `ctx`.
///
/// Copies already-computed hashes from `ctx.slot_scans` for `Full` and
/// `Cropped { has_quad_hash: true }` slots and, when `was_valid && allow_reseed`,
/// re-hashes at most [`MAX_RESEED_CROPPED_QUADS`] tiny cropped quadrants from
/// L2-hot `shadow` (pre-counting first so bands exceeding the cap return
/// `false` with zero wasted hashing). Dense bands pass `allow_reseed = false`
/// so full-redraw / tab-return frames never pay extra re-hashing on tiny
/// border crops. Returns `true` iff every entry in `prev_hashes` matches
/// finalized `ctx.shadow`.
///
/// # Safety
/// Rows `[ctx.y0, ctx.y1)` must be finalized in `ctx.shadow` and exclusively
/// owned by this worker, and `prev_hashes.len() == ctx.dirty_slots.len()`.
#[inline(never)]
pub(crate) unsafe fn sync_band_quad_hashes<const HAS_ALPHA: bool>(
    ctx: &ActiveRowBandCtx<'_>,
    scans: &SlotScans,
    start_qx: usize,
    prev_hashes: &mut [tile_hash::TileHash],
    was_valid: bool,
    allow_reseed: bool,
) -> bool {
    let n_qx = ctx.dirty_slots.len();
    debug_assert_eq!(prev_hashes.len(), n_qx);
    if !was_valid {
        // Only recover band validity for free when every quadrant in the band
        // already has a fresh hash in `scans` (e.g. a full redraw after scroll).
        debug_assert_eq!(start_qx, 0);
        for slot in ctx.dirty_slots {
            match SlotDirty::decode(slot.load(Ordering::Relaxed)) {
                SlotDirty::Full
                | SlotDirty::Cropped {
                    has_quad_hash: true,
                    ..
                } => {},
                SlotDirty::Clean
                | SlotDirty::Cropped {
                    has_quad_hash: false,
                    ..
                } => return false,
            }
        }
        for (qx, dst_h) in prev_hashes.iter_mut().enumerate() {
            // SAFETY: Checked above that `ctx.slot_base + qx` was written to `scans`.
            *dst_h = unsafe { scans.read(ctx.slot_base + qx).hash };
        }
        return true;
    }

    // Pre-check tiny cropped (`has_quad_hash: false`) count before doing any
    // re-hashing so dense bands (`!allow_reseed`) or bands above the cap bail
    // out immediately with zero wasted hashing.
    let max_reseed = if allow_reseed {
        MAX_RESEED_CROPPED_QUADS
    } else {
        0
    };
    let mut tiny_count = 0usize;
    for slot in &ctx.dirty_slots[start_qx..n_qx] {
        if matches!(
            SlotDirty::decode(slot.load(Ordering::Relaxed)),
            SlotDirty::Cropped {
                has_quad_hash: false,
                ..
            }
        ) {
            tiny_count += 1;
            if tiny_count > max_reseed {
                return false;
            }
        }
    }

    let ch = (ctx.y1 - ctx.y0) as u32;
    for (offset, (slot, prev_hash)) in ctx.dirty_slots[start_qx..n_qx]
        .iter()
        .zip(&mut prev_hashes[start_qx..n_qx])
        .enumerate()
    {
        let qx = start_qx + offset;
        match SlotDirty::decode(slot.load(Ordering::Relaxed)) {
            SlotDirty::Clean => {},
            SlotDirty::Full
            | SlotDirty::Cropped {
                has_quad_hash: true,
                ..
            } => {
                // SAFETY: `ctx.slot_base + qx` was written to `scans` by Phase A.
                *prev_hash = unsafe { scans.read(ctx.slot_base + qx).hash };
            },
            SlotDirty::Cropped {
                has_quad_hash: false,
                ..
            } => {
                let x0 = qx * SUBTILE_W;
                let cw = (ctx.width - x0).min(SUBTILE_W);
                let qrect = TileRect {
                    x: x0 as u32,
                    y: ctx.y0 as u32,
                    w: cw as u32,
                    h: ch,
                };
                // SAFETY: `qrect` is finalized in `shadow` and exclusively owned.
                *prev_hash = unsafe {
                    tile_hash::hash_rect_only_from_shadow::<HAS_ALPHA>(
                        scans.key(),
                        ctx.shadow.base.cast_const(),
                        ctx.stride,
                        qrect,
                    )
                };
            },
        }
    }
    true
}

/// Portable implementation of [`RowKernel::screen_and_diff_row_band_2d`] (also
/// used on AArch64 with hardware ARMv8 AES and under Miri tests).
///
/// # Safety
/// Caller must uphold [`RowKernel::screen_and_diff_row_band_2d`] invariants.
#[inline(always)]
pub(crate) unsafe fn screen_and_diff_row_band_2d_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    ctx: &ActiveRowBandCtx<'_>,
    scans: &SlotScans,
    prev_hashes: &mut [tile_hash::TileHash],
) -> bool {
    let width = ctx.width;
    let y0 = ctx.y0;
    let y1 = ctx.y1;
    let ch = (y1 - y0) as u32;
    let n_qx = ctx.dirty_slots.len();
    debug_assert_eq!(prev_hashes.len(), n_qx);
    let key = scans.key();
    let mut mismatches = 0usize;

    for qx in 0..n_qx {
        let x0 = qx * SUBTILE_W;
        let cw = (width - x0).min(SUBTILE_W);
        let qrect = TileRect {
            x: x0 as u32,
            y: y0 as u32,
            w: cw as u32,
            h: ch,
        };
        let q_hash =
            tile_hash::hash_rect_only_from_ptr::<HAS_ALPHA>(key, ctx.data, ctx.stride, qrect);
        if q_hash != prev_hashes[qx] {
            // SAFETY: Quadrant `qx` over `[y0, y1)` is exclusively owned.
            let dirty = unsafe {
                diff_and_update_single_quad_with::<K, HAS_ALPHA>(
                    kernel,
                    ctx,
                    scans,
                    qx,
                    &mut prev_hashes[qx],
                )
            };
            if dirty {
                mismatches += 1;
                if mismatches > SCREEN_MAX_MISMATCHES && qx + 1 < n_qx {
                    let start_qx = qx + 1;
                    let x_start = start_qx * SUBTILE_W;
                    let (_, shifted_data) = ctx.data.split_at(x_start * 4);
                    let (_, shifted_aos) = ctx.aos.split_at(x_start);
                    // SAFETY: `x_start < width`, so `x_start * 4` lies within `shadow`.
                    let shifted_shadow = ShadowRows {
                        base: unsafe { ctx.shadow.base.add(x_start * 4) },
                        stride: ctx.stride,
                    };
                    let rem_ctx = ActiveRowBandCtx {
                        data: shifted_data,
                        aos: shifted_aos,
                        shadow: &shifted_shadow,
                        width: width - x_start,
                        stride: ctx.stride,
                        y0,
                        y1,
                        slot_base: ctx.slot_base + start_qx,
                        dirty_slots: &ctx.dirty_slots[start_qx..],
                        slot_scans: ctx.slot_scans,
                        cache_non_empty: ctx.cache_non_empty,
                    };
                    // SAFETY: Shifted `rem_ctx` covers `[x_start, width) x [y0, y1)`
                    // exclusively owned by this worker.
                    unsafe {
                        kernel.diff_and_update_row_band_2d::<HAS_ALPHA>(&rem_ctx);
                        return sync_band_quad_hashes::<HAS_ALPHA>(
                            ctx,
                            scans,
                            start_qx,
                            prev_hashes,
                            true,
                            true,
                        );
                    }
                }
            }
        }
    }
    true
}

/// Processes a single `RowBand2D` inside one `SUBTILE_W`-wide vertical scroll
/// column stripe without nested closure indirection, fusing in-place overlap-safe
/// shadow scrolling, tight dirty bbox tracking, and (for `move_up`) immediate
/// per-quadrant `copy_and_hash_rect_to_shadow` on `bbox.full` escalation.
///
/// # Safety
/// Column stripe `[cx0, cx0 + cw)` over `[y0, y1)` must be in-bounds,
/// exclusively owned by this worker, and traversed in overlap-safe order.
#[inline(always)]
unsafe fn diff_and_update_scroll_band_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    data: BufferPointer<u8>,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    stride: usize,
    stride_px: usize,
    sr_dy: isize,
    sr_dst_y0: usize,
    sr_dst_y1: usize,
    move_up: bool,
    cx0: usize,
    cw: usize,
    ov_x_bytes: usize,
    ov_bytes: usize,
    qx: usize,
    band: &ScanSubBand,
    dirty_slots: &[AtomicU32],
    slot_scans: Option<&SlotScans>,
    cache_non_empty: bool,
) {
    let ScanSubBand::RowBand2D { y0, y1, slot_base } = *band else {
        unreachable!("scroll bands must be RowBand2D");
    };
    let y0 = y0 as usize;
    let y1 = y1 as usize;
    let ch = y1 - y0;
    let cx0_bytes = cx0 * 4;
    let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
    let mut bbox = QuadBbox::CLEAN;
    let band_overlaps_dst = ov_bytes > 0 && y1 > sr_dst_y0 && y0 < sr_dst_y1;
    let qrect = TileRect {
        x: cx0 as u32,
        y: y0 as u32,
        w: cw as u32,
        h: ch as u32,
    };

    if move_up {
        for y in y0..y1 {
            if band_overlaps_dst && y >= sr_dst_y0 && y < sr_dst_y1 {
                let src_y = (y as isize - sr_dy) as usize;
                // SAFETY: `src_y > y`, `[ov_start, ov_end)` lies inside this
                // worker's exclusive column stripe, and top-to-bottom traversal
                // guarantees `shadow[src_y]` has not yet been overwritten.
                unsafe {
                    ptr::copy_nonoverlapping(
                        shadow.row_ptr(src_y, ov_x_bytes),
                        shadow.row_ptr(y, ov_x_bytes),
                        ov_bytes,
                    );
                }
            }
            let (_, row_after) = aos.split_at(y * stride_px + cx0);
            let (src_chunk, _) = row_after.split_at(cw);
            // SAFETY: Created after `ptr::copy_nonoverlapping`; exclusive column stripe.
            let dst_chunk = unsafe {
                slice::from_raw_parts_mut(shadow.row_ptr(y, cx0_bytes).cast::<Vec4u8>(), cw)
            };
            if let Some((fx, lx)) = kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
                && bbox.record_row(y - y0, fx, lx, max_crop_area)
            {
                // Under `move_up` (`sr.dy < 0`), all later bands read `src_y >=
                // y1 + |dy| > y1`, so rows `(y + 1)..y1` of this band are never
                // read by later bands and can be overwritten & hashed directly.
                // SAFETY: Rows `0..=(y - y0)` are finalized in `shadow`.
                unsafe {
                    finish_full_quadrant::<HAS_ALPHA>(
                        data,
                        shadow,
                        stride,
                        qrect,
                        (y - y0) + 1,
                        slot_base + qx,
                        &dirty_slots[slot_base + qx],
                        slot_scans,
                    );
                }
                return;
            }
        }
    } else {
        for y in (y0..y1).rev() {
            if band_overlaps_dst && y >= sr_dst_y0 && y < sr_dst_y1 {
                let src_y = (y as isize - sr_dy) as usize;
                // SAFETY: `src_y < y`, `[ov_start, ov_end)` lies inside this
                // worker's exclusive column stripe, and bottom-to-top traversal
                // guarantees `shadow[src_y]` has not yet been overwritten.
                unsafe {
                    ptr::copy_nonoverlapping(
                        shadow.row_ptr(src_y, ov_x_bytes),
                        shadow.row_ptr(y, ov_x_bytes),
                        ov_bytes,
                    );
                }
            }
            let (_, row_after) = aos.split_at(y * stride_px + cx0);
            let (src_chunk, _) = row_after.split_at(cw);
            // SAFETY: Created after `ptr::copy_nonoverlapping`; exclusive column stripe.
            let dst_chunk = unsafe {
                slice::from_raw_parts_mut(shadow.row_ptr(y, cx0_bytes).cast::<Vec4u8>(), cw)
            };
            if let Some((fx, lx)) = kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
                && bbox.record_row(y - y0, fx, lx, max_crop_area)
            {
                // Under `!move_up` (`sr.dy > 0`), all later bands read `src_y <
                // y0`, so remaining rows `y0..y` of this band can be copied
                // directly from `data` without scroll replay.
                if y > y0 {
                    let rem_rect = TileRect {
                        x: cx0 as u32,
                        y: y0 as u32,
                        w: cw as u32,
                        h: (y - y0) as u32,
                    };
                    // SAFETY: Exclusive column stripe in `shadow`.
                    unsafe { copy_rect_to_shadow(data, shadow, stride, rem_rect) };
                }
                if let Some(scans) = slot_scans {
                    // SAFETY: `qrect` is finalized and exclusively owned.
                    let scan = unsafe {
                        tile_hash::hash_rect_from_shadow::<HAS_ALPHA>(
                            scans.key(),
                            shadow.base.cast_const(),
                            stride,
                            qrect,
                        )
                    };
                    // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                    unsafe { scans.write(slot_base + qx, scan) };
                }
                dirty_slots[slot_base + qx].store(SlotDirty::FULL_RAW, Ordering::Relaxed);
                return;
            }
        }
    }

    if bbox.is_dirty() {
        let crop_area = usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
        let has_quad_hash =
            cache_non_empty && slot_scans.is_some() && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
        if has_quad_hash && let Some(scans) = slot_scans {
            // SAFETY: Column stripe `[cx0, cx1)` over `[y0, y1)` is in-bounds,
            // finalized, and exclusively owned by this worker.
            let scan = unsafe {
                tile_hash::hash_rect_from_shadow::<HAS_ALPHA>(
                    scans.key(),
                    shadow.base.cast_const(),
                    stride,
                    qrect,
                )
            };
            // SAFETY: `slot_base + qx` is exclusively owned by this worker.
            unsafe { scans.write(slot_base + qx, scan) };
        }
        dirty_slots[slot_base + qx].store(
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

/// Scans a single `SUBTILE_W`-wide vertical column stripe `qx` across all
/// `RowBand2D`s overlapping `scroll_rect`, fusing overlap-safe in-place shadow
/// scrolling with SIMD sub-tile tight dirty bounding-box tracking and shadow
/// update in a single parallel pass.
///
/// # Safety
/// Every entry in `scroll_bands` must be [`ScanSubBand::RowBand2D`] in
/// ascending `y` order covering `[sr.y, sr.y + sr.height)`, and column stripe
/// `qx` must be exclusively owned by this worker across `scroll_bands`.
#[inline(always)]
unsafe fn diff_and_update_scroll_column_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    data: BufferPointer<u8>,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    width: usize,
    stride: usize,
    stride_px: usize,
    scroll_bands: &[ScanSubBand],
    sr: ScrollRect,
    qx: usize,
    dirty_slots: &[AtomicU32],
    slot_scans: Option<&SlotScans>,
    cache_non_empty: bool,
) {
    let cx0 = qx * SUBTILE_W;
    let cx1 = (cx0 + SUBTILE_W).min(width);
    let cw = cx1 - cx0;
    let sr_x0 = sr.x as usize;
    let sr_x1 = (sr.x + sr.width) as usize;
    let ov_start = cx0.max(sr_x0).min(cx1);
    let ov_end = cx1.min(sr_x1).max(ov_start);
    let ov_bytes = (ov_end - ov_start) * 4;
    let ov_x_bytes = ov_start * 4;

    let shift = sr.dy.unsigned_abs() as usize;
    let copy_h = sr.height as usize - shift;
    let (sr_dst_y0, sr_dst_y1) = if sr.dy < 0 {
        (sr.y as usize, sr.y as usize + copy_h)
    } else {
        (sr.y as usize + shift, sr.y as usize + sr.height as usize)
    };
    let move_up = sr.dy < 0;
    let sr_dy = sr.dy as isize;

    if move_up {
        for band in scroll_bands {
            // SAFETY: Caller guarantees exclusive column ownership and ascending band order.
            unsafe {
                diff_and_update_scroll_band_with::<K, HAS_ALPHA>(
                    kernel,
                    data,
                    aos,
                    shadow,
                    stride,
                    stride_px,
                    sr_dy,
                    sr_dst_y0,
                    sr_dst_y1,
                    true,
                    cx0,
                    cw,
                    ov_x_bytes,
                    ov_bytes,
                    qx,
                    band,
                    dirty_slots,
                    slot_scans,
                    cache_non_empty,
                );
            }
        }
    } else {
        for band in scroll_bands.iter().rev() {
            // SAFETY: Caller guarantees exclusive column ownership and descending band order.
            unsafe {
                diff_and_update_scroll_band_with::<K, HAS_ALPHA>(
                    kernel,
                    data,
                    aos,
                    shadow,
                    stride,
                    stride_px,
                    sr_dy,
                    sr_dst_y0,
                    sr_dst_y1,
                    false,
                    cx0,
                    cw,
                    ov_x_bytes,
                    ov_bytes,
                    qx,
                    band,
                    dirty_slots,
                    slot_scans,
                    cache_non_empty,
                );
            }
        }
    }
}

/// Finds the contiguous `sub_bands` index range `[sb_start, sb_end)` that
/// overlaps `[sr.y, sr.y + sr.height)`.
fn scroll_sub_band_range(sub_bands: &[ScanSubBand], sr: ScrollRect) -> (usize, usize) {
    let sr_y0 = sr.y;
    let sr_y1 = sr.y + sr.height;
    let mut start = sub_bands.len();
    let mut end = 0usize;
    for (idx, sb) in sub_bands.iter().enumerate() {
        let (y0, y1) = match *sb {
            ScanSubBand::RowBand2D { y0, y1, .. } => (y0, y1),
            ScanSubBand::Strip1D { rect, .. } => (rect.y, rect.y + rect.h),
        };
        if y1 > sr_y0 && y0 < sr_y1 {
            debug_assert!(matches!(sb, ScanSubBand::RowBand2D { .. }));
            start = start.min(idx);
            end = idx + 1;
        }
    }
    if start > end { (0, 0) } else { (start, end) }
}

/// Maximum worker count eligible for stack-allocated per-worker Phase A -> B
/// cache-affinity buckets (larger pools fall back to a single shared counter).
const MAX_AFFINITY_WORKERS: usize = 64;

/// Maximum Phase-A sub-bands tracked in a stack-allocated `AtomicU8` owner
/// array (`64` covers up to 8192-px-tall frames with zero heap allocation).
const MAX_STACK_OWNERS: usize = 64;

/// Maximum dirty tiles bucketed on the stack for Phase B home queues before
/// spilling to heap (`512` covers 4K full-quadrant cropped frames with zero
/// heap allocation).
const STACK_HOME_TILES_CAP: usize = 512;

/// Minimum number of shadow-reading tiles (`Put` or `Uncached`) required to
/// enable Phase B exact-owner bucketing (tiny edits with `< 8` encoded tiles
/// wake only a few workers and run faster on a single shared counter).
const PHASE_B_AFFINITY_MIN_ENCODE: usize = 8;

/// Cache-line-isolated work-stealing bucket covering task range `[next, end)`.
#[repr(align(64))]
pub(crate) struct WorkBucket {
    next: AtomicUsize,
    end: usize,
}

impl WorkBucket {
    #[inline]
    const fn new(start: usize, end: usize) -> Self {
        Self {
            next: AtomicUsize::new(start),
            end,
        }
    }
}

/// Lock-free work-stealing cursor over a slice of per-worker [`WorkBucket`]s:
/// worker `worker_idx` drains its own bucket first and then steals circularly
/// from `(worker_idx + 1) % p`, `(worker_idx + 2) % p`, ..., guaranteeing every
/// task in all buckets is claimed exactly once even when fewer than `p` workers
/// wake.
struct WorkStealIter<'a> {
    buckets: &'a [WorkBucket],
    start_w: usize,
    step: usize,
}

impl<'a> WorkStealIter<'a> {
    #[inline(always)]
    fn new(buckets: &'a [WorkBucket], worker_idx: usize) -> Self {
        let p = buckets.len();
        debug_assert!(p <= 1 || worker_idx < p);
        let start_w = if p <= 1 { 0 } else { worker_idx };
        Self {
            buckets,
            start_w,
            step: 0,
        }
    }

    #[inline(never)]
    fn next_idx(&mut self) -> Option<usize> {
        let p = self.buckets.len();
        while self.step < p {
            let b = self.start_w + self.step;
            let b = if b < p { b } else { b - p };
            let Some(bucket) = self.buckets.get(b) else {
                break;
            };
            if bucket.next.load(Ordering::Relaxed) < bucket.end {
                let idx = bucket.next.fetch_add(1, Ordering::Relaxed);
                if idx < bucket.end {
                    return Some(idx);
                }
            }
            self.step += 1;
        }
        None
    }
}

/// Borrowed Phase-A task scheduler passed across SIMD dispatch boundaries.
#[derive(Copy, Clone)]
pub(crate) struct PhaseASchedule<'a> {
    pub(crate) buckets: &'a [WorkBucket],
    pub(crate) task_owners: Option<&'a [AtomicU8]>,
}

impl<'a> PhaseASchedule<'a> {
    #[inline]
    fn single(bucket: &'a WorkBucket, task_owners: Option<&'a [AtomicU8]>) -> Self {
        Self {
            buckets: slice::from_ref(bucket),
            task_owners,
        }
    }

    #[inline(always)]
    fn iter(&self, worker_idx: usize) -> WorkStealIter<'a> {
        WorkStealIter::new(self.buckets, worker_idx)
    }
}

/// Computes worker `w`'s contiguous slice `[start, end)` of `0..total` across
/// `p` buckets, placing even-sized slices first so `256`-px tile pairs
/// `(2k, 2k+1)` stay on the same worker whenever `total >= 2` permits.
#[inline]
fn contiguous_slice(total: usize, p: usize, w: usize) -> (usize, usize) {
    let base = total / p;
    let rem = total % p;
    if base.is_multiple_of(2) {
        let even_workers = p - rem;
        if w < even_workers {
            let start = w * base;
            (start, start + base)
        } else {
            let extra = w - even_workers;
            let start = even_workers * base + extra * (base + 1);
            (start, start + base + 1)
        }
    } else if w < rem {
        let start = w * (base + 1);
        (start, start + base + 1)
    } else {
        let start = rem * (base + 1) + (w - rem) * base;
        (start, start + base)
    }
}

/// Maps `rect` to its owning `sub_bands` index (`0..sub_bands.len()`).
#[inline]
fn sub_band_idx_for_rect(sub_bands: &[ScanSubBand], rect: TileRect, full_frame_2d: bool) -> usize {
    let n_sub = sub_bands.len();
    if full_frame_2d {
        ((rect.y as usize) / SUBTILE_H).min(n_sub - 1)
    } else {
        sub_bands
            .partition_point(|sb| match *sb {
                ScanSubBand::RowBand2D { y1, .. } => y1 <= rect.y,
                ScanSubBand::Strip1D { rect: r, .. } => (r.y + r.h, r.x + r.w) <= (rect.y, rect.x),
            })
            .min(n_sub - 1)
    }
}

/// Returns the Phase-A worker index (`0..p`) that wrote `rect` on a non-scroll
/// frame, alternating coalesced `> 128`-tall tiles evenly between their top and
/// bottom `128`-row sub-bands so both even- and odd-band workers receive
/// L2/L3-hot home tiles.
#[inline]
fn phase_b_owner_non_scroll(
    sub_bands: &[ScanSubBand],
    owners: &[AtomicU8],
    rect: TileRect,
    full_frame_2d: bool,
    p: usize,
) -> usize {
    let mut sb = sub_band_idx_for_rect(sub_bands, rect, full_frame_2d);
    if rect.h > SUBTILE_H as u32 && sb + 1 < owners.len() && ((rect.x as usize / TILE_W) & 1) == 1 {
        sb += 1;
    }
    (owners[sb].load(Ordering::Relaxed) as usize).min(p - 1)
}

/// Worker loop for Phase A incremental shadow diffing on non-scroll frames.
///
/// # Safety
/// Every entry in `sub_bands` must cover strictly in-bounds, pairwise disjoint
/// regions in `shadow`.
#[inline(always)]
unsafe fn diff_and_update_worker_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
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
    quad_hashes: Option<&PhaseAQuadHashes>,
    schedule: &PhaseASchedule<'_>,
    worker_idx: usize,
) {
    let mut iter = schedule.iter(worker_idx);
    while let Some(idx) = iter.next_idx() {
        match sub_bands[idx] {
            ScanSubBand::Strip1D { rect, slot_idx } => {
                // SAFETY: `sub_bands` cover disjoint regions in `shadow`.
                unsafe {
                    diff_and_update_strip_slot_with::<K, HAS_ALPHA>(
                        kernel,
                        data,
                        aos,
                        shadow,
                        stride,
                        stride_px,
                        rect,
                        slot_idx,
                        &dirty_slots[slot_idx],
                        slot_scans,
                    );
                }
            },
            ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                let y0 = y0 as usize;
                let y1 = y1 as usize;
                let ctx = ActiveRowBandCtx {
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    y0,
                    y1,
                    slot_base,
                    dirty_slots: &dirty_slots[slot_base..slot_base + n_qx],
                    slot_scans,
                    cache_non_empty,
                };
                if let Some(qh) = quad_hashes
                    && let Some(scans) = slot_scans
                    && y0.is_multiple_of(SUBTILE_H)
                    && y1 == (y0 + SUBTILE_H).min(qh.height)
                {
                    let qy = y0 / SUBTILE_H;
                    // SAFETY: Grid-aligned `RowBand2D`s are strictly y-disjoint, so
                    // band `qy` is exclusively owned by this worker.
                    let (was_valid, band_hashes) =
                        unsafe { (qh.sync.band_valid(qy), qh.sync.band_hashes_mut(qy)) };
                    let dense_hint = qy < 64 && ((qh.dense_dirty_bands >> qy) & 1) != 0;
                    let now_valid = if was_valid && !dense_hint {
                        // SAFETY: Disjoint band `qy` is exclusively owned by this worker.
                        unsafe {
                            kernel.screen_and_diff_row_band_2d::<HAS_ALPHA>(
                                &ctx,
                                scans,
                                band_hashes,
                            )
                        }
                    } else {
                        // SAFETY: Disjoint band `qy` is exclusively owned by this worker.
                        unsafe {
                            kernel.diff_and_update_row_band_2d::<HAS_ALPHA>(&ctx);
                            sync_band_quad_hashes::<HAS_ALPHA>(
                                &ctx,
                                scans,
                                0,
                                band_hashes,
                                was_valid,
                                !dense_hint,
                            )
                        }
                    };
                    if now_valid != was_valid {
                        // SAFETY: Band `qy` is exclusively owned by this worker;
                        // guarding writes keeps `band_valid` read-only on steady frames.
                        unsafe { qh.sync.set_band_valid(qy, now_valid) };
                    }
                } else {
                    // SAFETY: `sub_bands` cover disjoint row ranges in `shadow`.
                    unsafe {
                        kernel.diff_and_update_row_band_2d::<HAS_ALPHA>(&ctx);
                    }
                }
            },
        }
        if let Some(owners) = schedule.task_owners {
            owners[idx].store(worker_idx as u8, Ordering::Relaxed);
        }
    }
}

/// Worker loop for Phase A incremental shadow diffing fused with overlap-safe
/// vertical scroll replay.
///
/// # Safety
/// Every entry in `sub_bands` must cover strictly in-bounds, pairwise disjoint
/// regions in `shadow`, and `sr` must lie within a single contiguous
/// `RowBand2D` range of `sub_bands`.
#[inline(always)]
unsafe fn diff_and_update_scroll_worker_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
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
    let (sb_start, sb_end) = scroll_sub_band_range(sub_bands, sr);
    let scroll_bands = &sub_bands[sb_start..sb_end];
    let n_before = sb_start;
    let n_after = sub_bands.len() - sb_end;
    let n_outside = n_before + n_after;
    let total_tasks = n_outside + n_qx;

    loop {
        let task = next_sub.fetch_add(1, Ordering::Relaxed);
        if task >= total_tasks {
            break;
        }
        if task < n_qx {
            // SAFETY: `scroll_bands` are all `RowBand2D` covering `sr`, and
            // each column `task` is claimed at most once.
            unsafe {
                diff_and_update_scroll_column_with::<K, HAS_ALPHA>(
                    kernel,
                    data,
                    aos,
                    shadow,
                    width,
                    stride,
                    stride_px,
                    scroll_bands,
                    sr,
                    task,
                    dirty_slots,
                    slot_scans,
                    cache_non_empty,
                );
            }
        } else {
            let out_idx = task - n_qx;
            let sb_idx = if out_idx < n_before {
                out_idx
            } else {
                sb_end + (out_idx - n_before)
            };
            match sub_bands[sb_idx] {
                ScanSubBand::Strip1D { rect, slot_idx } => {
                    // SAFETY: Outside `sb_start..sb_end`, disjoint from `sr`.
                    unsafe {
                        diff_and_update_strip_slot_with::<K, HAS_ALPHA>(
                            kernel,
                            data,
                            aos,
                            shadow,
                            stride,
                            stride_px,
                            rect,
                            slot_idx,
                            &dirty_slots[slot_idx],
                            slot_scans,
                        );
                    }
                },
                ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                    let ctx = ActiveRowBandCtx {
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        y0: y0 as usize,
                        y1: y1 as usize,
                        slot_base,
                        dirty_slots: &dirty_slots[slot_base..slot_base + n_qx],
                        slot_scans,
                        cache_non_empty,
                    };
                    // SAFETY: Outside `sb_start..sb_end`, disjoint from `sr`.
                    unsafe {
                        kernel.diff_and_update_row_band_2d::<HAS_ALPHA>(&ctx);
                    }
                },
            }
        }
    }
}

/// Validates that every rect in `tile_rects` is non-empty, within
/// [`TILE_PIXELS`], strictly inside `[0, width) x [0, height)`, and pairwise
/// disjoint in `O(N log N)` time via a horizontal scanline sweep. Returns the
/// total number of pixels covered.
pub(crate) fn validate_tile_rects(tile_rects: &[TileRect], width: u32, height: u32) -> Result<u64> {
    let mut total_pixels = 0u64;
    for &rect in tile_rects {
        let px = rect.validate_in_bounds(width, height).location(loc!())?;
        total_pixels += px as u64;
    }
    if tile_rects.len() <= 1 {
        return Ok(total_pixels);
    }

    let mut sorted = tile_rects.to_vec();
    sorted.sort_unstable_by_key(|r| (r.y, r.x));

    // Invariant: `active_x` maps `x0 -> x1` for every earlier rect with
    // `y1 > cur.y`. Because all active rects overlap at `y = cur.y` and have
    // already been proven pairwise disjoint, their `[x0, x1)` intervals are
    // strictly disjoint and uniquely keyed by `x0`.
    let mut active_x: BTreeMap<u32, u32> = BTreeMap::new();
    let mut expire: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(sorted.len());

    for rect in sorted {
        let rx0 = rect.x;
        let rx1 = rect.x + rect.w;
        let ry1 = rect.y + rect.h;

        while let Some(&Reverse((y1, x0))) = expire.peek()
            && y1 <= rect.y
        {
            expire.pop();
            active_x.remove(&x0);
        }

        if let Some((&prev_x0, &prev_x1)) = active_x.range(..=rx0).next_back()
            && prev_x1 > rx0
        {
            bail!(
                "overlapping TileRect {rect:?} conflicts with active x-interval [{prev_x0}, {prev_x1})"
            );
        }
        if let Some((&next_x0, &next_x1)) = active_x.range(rx0..rx1).next() {
            bail!(
                "overlapping TileRect {rect:?} conflicts with active x-interval [{next_x0}, {next_x1})"
            );
        }

        active_x.insert(rx0, rx1);
        expire.push(Reverse((ry1, rx0)));
    }

    Ok(total_pixels)
}

// --- public API ---

fn filter_and_compress_full_streaming_with_tier(
    tier: arch::Tier,
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    shadow_out: Option<&mut ShadowBuffer>,
    tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) {
    assert!(data.len().is_multiple_of(4));
    let total_pixels = metadata
        .validate()
        .expect("server passed invalid BufferMetadata");
    mode.validate()
        .expect("server passed invalid CompressionMode");
    let byte_len = total_pixels * 4;
    assert!(data.len() >= byte_len);

    let width = metadata.width as usize;
    let height = metadata.height as usize;
    let width_bytes = width * 4;
    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let seed_quad_hashes = height >= MIN_2D_BAND_HEIGHT as usize;
    let (_, base_entries, total_slots) =
        plan_incremental_scan(&metadata, None).expect("full-frame tile plan cannot fail");

    let mut shadow_opt = shadow_out;
    if let Some(shadow) = shadow_opt.as_deref_mut() {
        shadow.quad_hashes.invalidate_all();
        shadow.pixels.clear();
        if shadow.pixels.capacity() < byte_len {
            shadow.pixels.reserve_exact(byte_len);
        } else if shadow.pixels.capacity() > byte_len.saturating_mul(2) {
            shadow.pixels.shrink_to(byte_len);
        }
    }

    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };

    if let Some(cache) = tile_cache {
        if cache.occupied_slots() == 0 {
            // Fast single-pass direct cold-fill: when the cache starts empty,
            // no inter-frame hits or partial quadrant un-coalescing can occur,
            // so `tile_rects` is known upfront and each worker fuses
            // copy-to-shadow + keyed 128-bit hashing + direct `cache.index`
            // claim + immediate encoding while the tile's `<= 256 KiB` pixels
            // stay hot in private L2 cache, with zero head-of-line stalling and
            // zero lock acquisition for solid tiles.
            let hash_key = *cache.hash_key();
            let tile_rects: Vec<TileRect> = base_entries
                .iter()
                .copied()
                .map(BaseScanEntry::rect)
                .collect();
            let n_tiles = tile_rects.len();
            let reset_tile_cache = cache.take_reset_pending();
            cache.begin_cold_fill();
            let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
            let header = RawBufferHeader {
                frame_seq: 0,
                client,
                surface,
                metadata,
                incremental: false,
                scroll_rect: None,
                reset_tile_cache,
                evict_slots: Vec::new(),
                tile_rects,
            };
            on_stream_ready(StreamingCompressedTiles {
                header,
                tiles: tiles_rx,
            });

            let (shadow_rows, sync_qh) = if let Some(shadow) = shadow_opt.as_deref_mut() {
                let rows = ShadowRows {
                    base: shadow.pixels.as_mut_ptr(),
                    stride,
                };
                let qh = if seed_quad_hashes {
                    shadow
                        .quad_hashes
                        .prepare_for(width, height, stride, has_alpha, &hash_key);
                    Some(shadow.quad_hashes.sync_view())
                } else {
                    None
                };
                (Some(rows), qh)
            } else {
                (None, None)
            };
            let initial_quality = TileQuality::initial_for_mode(mode);
            let is_hybrid = matches!(mode, CompressionMode::Hybrid { .. });
            let promotions: Vec<AtomicU8> = if is_hybrid {
                (0..MAX_TILE_CACHE_SLOTS)
                    .map(|_| AtomicU8::new(0))
                    .collect()
            } else {
                Vec::new()
            };
            let cache_lock = Mutex::new(&mut *cache);
            let next_tile = AtomicUsize::new(0);
            let disconnected = AtomicBool::new(false);
            compressor.pool.execute(n_tiles, |state| {
                let shadow_ptr;
                let encode_aos = if let Some(shadow) = &shadow_rows {
                    shadow_ptr = shadow.base.cast_const().cast::<Vec4u8>();
                    // SAFETY: `shadow.base` points to `total_pixels * 4` bytes of
                    // allocated capacity; this worker exclusively initializes
                    // `rect` in `shadow` immediately before reading `rect` from
                    // `encode_aos`.
                    unsafe { BufferPointer::new(&shadow_ptr, total_pixels) }
                } else {
                    aos
                };

                loop {
                    let h = next_tile.fetch_add(1, Ordering::Relaxed);
                    if h >= n_tiles {
                        break;
                    }
                    let entry = base_entries[h];
                    if disconnected.load(Ordering::Relaxed) {
                        if let Some(shadow) = &shadow_rows {
                            let rect = entry.rect();
                            // SAFETY: `base_entries` are pairwise disjoint and in-bounds,
                            // and each `h` is claimed once.
                            unsafe {
                                copy_rect_to_shadow(data, shadow, stride, rect);
                                zero_shadow_row_padding(shadow, width_bytes, rect);
                            }
                            continue;
                        }
                        break;
                    }

                    let (hash, meta, sub_quads) = match entry {
                        BaseScanEntry::Strip1D { rect, .. } => {
                            let scan = if let Some(shadow) = &shadow_rows {
                                // SAFETY: `base_entries` are pairwise disjoint and in-bounds.
                                unsafe {
                                    let scan = if has_alpha {
                                        tile_hash::copy_and_hash_rect_to_shadow::<true>(
                                            &hash_key,
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                        )
                                    } else {
                                        tile_hash::copy_and_hash_rect_to_shadow::<false>(
                                            &hash_key,
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                        )
                                    };
                                    zero_shadow_row_padding(shadow, width_bytes, rect);
                                    scan
                                }
                            } else if has_alpha {
                                tile_hash::hash_rect_from_ptr::<true>(
                                    &hash_key, aos, stride_px, rect,
                                )
                            } else {
                                tile_hash::hash_rect_from_ptr::<false>(
                                    &hash_key, aos, stride_px, rect,
                                )
                            };
                            (scan.hash, scan.meta, None)
                        },
                        BaseScanEntry::Tile2D { rect, n_quads, .. } => {
                            let (quads, _) = tile_quadrants(rect);
                            let mut quad_hashes = [[0u8; 16]; 4];
                            let mut quad_metas = [RectScanMeta::UNKNOWN; 4];
                            for q in 0..n_quads {
                                let scan = if let Some(shadow) = &shadow_rows {
                                    // SAFETY: `quads[q]` lies inside exclusively owned `rect`.
                                    unsafe {
                                        if has_alpha {
                                            tile_hash::copy_and_hash_rect_to_shadow::<true>(
                                                &hash_key,
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                            )
                                        } else {
                                            tile_hash::copy_and_hash_rect_to_shadow::<false>(
                                                &hash_key,
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                            )
                                        }
                                    }
                                } else if has_alpha {
                                    tile_hash::hash_rect_from_ptr::<true>(
                                        &hash_key, aos, stride_px, quads[q],
                                    )
                                } else {
                                    tile_hash::hash_rect_from_ptr::<false>(
                                        &hash_key, aos, stride_px, quads[q],
                                    )
                                };
                                quad_hashes[q] = scan.hash;
                                quad_metas[q] = scan.meta;
                                if let Some(qh) = &sync_qh {
                                    // SAFETY: `Tile2D` entries own strictly disjoint `(qy, qx)` quadrants.
                                    unsafe {
                                        qh.write_hash(
                                            quads[q].y as usize / SUBTILE_H,
                                            quads[q].x as usize / SUBTILE_W,
                                            scan.hash,
                                        );
                                    }
                                }
                            }
                            if let Some(shadow) = &shadow_rows {
                                // SAFETY: `rect` is exclusively owned by this worker and
                                // disjoint from all other `base_entries`.
                                unsafe { zero_shadow_row_padding(shadow, width_bytes, rect) };
                            }
                            if n_quads == 1 {
                                (quad_hashes[0], quad_metas[0], None)
                            } else {
                                let parent_meta = RectScanMeta::combine(&quad_metas[..n_quads]);
                                if parent_meta.solid_px.is_some() {
                                    ([0u8; 16], parent_meta, None)
                                } else {
                                    let parent_hash = tile_hash::combine_quad_hashes(
                                        &hash_key,
                                        rect.w,
                                        rect.h,
                                        &quad_hashes[..n_quads],
                                    );
                                    (
                                        parent_hash,
                                        parent_meta,
                                        Some(ColdSubQuads {
                                            hashes: quad_hashes,
                                            metas: quad_metas,
                                            n_quads: n_quads as u8,
                                        }),
                                    )
                                }
                            }
                        },
                    };

                    let rect = entry.rect();
                    let tile = if let Some(px) = meta.solid_px {
                        CompressedTile {
                            tile_idx: h as u32,
                            cache_put: None,
                            payload: TilePayload::Leaf(LeafPayload::Solid(px)),
                        }
                    } else {
                        let action = cache_lock.lock().unwrap().claim_cold_tile(
                            rect,
                            hash,
                            sub_quads,
                            has_alpha,
                            initial_quality,
                        );
                        match action {
                            TileCacheAction::Hit(hit_ref) => CompressedTile {
                                tile_idx: h as u32,
                                cache_put: None,
                                payload: TilePayload::CacheHit(hit_ref),
                            },
                            TileCacheAction::Solid(px) => CompressedTile {
                                tile_idx: h as u32,
                                cache_put: None,
                                payload: TilePayload::Leaf(LeafPayload::Solid(px)),
                            },
                            TileCacheAction::Put(slot_id) => {
                                let enc_meta = TileEncodeMeta {
                                    tile: meta,
                                    quads: sub_quads
                                        .map_or([RectScanMeta::UNKNOWN; 4], |sq| sq.metas),
                                };
                                let payload = if has_alpha {
                                    filter_tile::<true>(
                                        tier,
                                        encode_aos,
                                        shadow_rows.is_some(),
                                        stride_px,
                                        rect,
                                        mode,
                                        &enc_meta,
                                        state,
                                    )
                                } else {
                                    filter_tile::<false>(
                                        tier,
                                        encode_aos,
                                        shadow_rows.is_some(),
                                        stride_px,
                                        rect,
                                        mode,
                                        &enc_meta,
                                        state,
                                    )
                                };
                                if is_hybrid {
                                    promotions[slot_id as usize].store(
                                        payload_lossless_promotion_mask(rect, &payload),
                                        Ordering::Relaxed,
                                    );
                                }
                                CompressedTile {
                                    tile_idx: h as u32,
                                    cache_put: Some(slot_id),
                                    payload,
                                }
                            },
                            TileCacheAction::Uncached => {
                                let enc_meta = TileEncodeMeta {
                                    tile: meta,
                                    quads: sub_quads
                                        .map_or([RectScanMeta::UNKNOWN; 4], |sq| sq.metas),
                                };
                                let payload = if has_alpha {
                                    filter_tile::<true>(
                                        tier,
                                        encode_aos,
                                        shadow_rows.is_some(),
                                        stride_px,
                                        rect,
                                        mode,
                                        &enc_meta,
                                        state,
                                    )
                                } else {
                                    filter_tile::<false>(
                                        tier,
                                        encode_aos,
                                        shadow_rows.is_some(),
                                        stride_px,
                                        rect,
                                        mode,
                                        &enc_meta,
                                        state,
                                    )
                                };
                                CompressedTile {
                                    tile_idx: h as u32,
                                    cache_put: None,
                                    payload,
                                }
                            },
                        }
                    };
                    if tiles_tx.send(tile).is_err() {
                        disconnected.store(true, Ordering::Relaxed);
                    }
                }
            });

            let cache = cache_lock.into_inner().unwrap();
            let was_disconnected = disconnected.load(Ordering::Relaxed);
            if let Some(shadow) = shadow_opt {
                // SAFETY: `base_entries` partitions `[0, width) x [0, height)`,
                // every tile was written (even on early stream disconnect), and
                // right-edge tiles zeroed trailing row-stride padding, so all
                // `byte_len` bytes are initialized.
                unsafe { shadow.pixels.set_len(byte_len) };
                if seed_quad_hashes && !was_disconnected {
                    shadow.quad_hashes.mark_all_valid();
                } else {
                    shadow.quad_hashes.invalidate_all();
                }
            }

            if was_disconnected {
                cache.reset_for_peer();
            } else {
                cache.finish_cold_fill();
                if is_hybrid {
                    for (slot_id, cell) in promotions.into_iter().enumerate() {
                        let mask = cell.into_inner();
                        if mask != 0 {
                            cache.promote_slot_lossless(slot_id as u16, mask);
                        }
                    }
                }
            }
            return;
        }

        // Phase A: copy all base tiles to `shadow` (when requested) and compute
        // keyed 128-bit digests and scan metadata for every quadrant directly in
        // registers.
        let mut slot_scans = SlotScans::new(total_slots, *cache.hash_key());
        {
            let _hash_span = debug_span!("tile_cache_hash_full").entered();
            let (shadow_rows, sync_qh) = if let Some(shadow) = shadow_opt.as_deref_mut() {
                let rows = ShadowRows {
                    base: shadow.pixels.as_mut_ptr(),
                    stride,
                };
                let qh = if seed_quad_hashes {
                    shadow.quad_hashes.prepare_for(
                        width,
                        height,
                        stride,
                        has_alpha,
                        slot_scans.key(),
                    );
                    Some(shadow.quad_hashes.sync_view())
                } else {
                    None
                };
                (Some(rows), qh)
            } else {
                (None, None)
            };
            let n_entries = base_entries.len();
            let next_entry = AtomicUsize::new(0);
            compressor.pool.execute(n_entries, |_state| {
                loop {
                    let entry_idx = next_entry.fetch_add(1, Ordering::Relaxed);
                    if entry_idx >= n_entries {
                        break;
                    }
                    let entry = base_entries[entry_idx];
                    if let Some(shadow) = &shadow_rows {
                        match entry {
                            BaseScanEntry::Strip1D { rect, slot_idx } => {
                                // SAFETY: `base_entries` are pairwise disjoint and in-bounds,
                                // and each `entry_idx` is claimed once.
                                let scan = unsafe {
                                    let scan = if has_alpha {
                                        tile_hash::copy_and_hash_rect_to_shadow::<true>(
                                            slot_scans.key(),
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                        )
                                    } else {
                                        tile_hash::copy_and_hash_rect_to_shadow::<false>(
                                            slot_scans.key(),
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                        )
                                    };
                                    zero_shadow_row_padding(shadow, width_bytes, rect);
                                    scan
                                };
                                // SAFETY: `slot_idx` is exclusively owned by this entry.
                                unsafe { slot_scans.write(slot_idx, scan) };
                            },
                            BaseScanEntry::Tile2D {
                                rect,
                                quad_slots,
                                n_quads,
                            } => {
                                let (quads, _) = tile_quadrants(rect);
                                for q in 0..n_quads {
                                    // SAFETY: `quads[q]` lies inside exclusively owned `rect`.
                                    let scan = unsafe {
                                        if has_alpha {
                                            tile_hash::copy_and_hash_rect_to_shadow::<true>(
                                                slot_scans.key(),
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                            )
                                        } else {
                                            tile_hash::copy_and_hash_rect_to_shadow::<false>(
                                                slot_scans.key(),
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                            )
                                        }
                                    };
                                    // SAFETY: `quad_slots[q]` is exclusively owned by this entry.
                                    unsafe { slot_scans.write(quad_slots[q], scan) };
                                    if let Some(qh) = &sync_qh {
                                        // SAFETY: `Tile2D` entries own strictly disjoint `(qy, qx)` quadrants.
                                        unsafe {
                                            qh.write_hash(
                                                quads[q].y as usize / SUBTILE_H,
                                                quads[q].x as usize / SUBTILE_W,
                                                scan.hash,
                                            );
                                        }
                                    }
                                }
                                // SAFETY: `rect` is exclusively owned by this worker and
                                // disjoint from all other `base_entries`.
                                unsafe { zero_shadow_row_padding(shadow, width_bytes, rect) };
                            },
                        }
                    } else {
                        match entry {
                            BaseScanEntry::Strip1D { rect, slot_idx } => {
                                let scan = if has_alpha {
                                    tile_hash::hash_rect_from_ptr::<true>(
                                        slot_scans.key(),
                                        aos,
                                        stride_px,
                                        rect,
                                    )
                                } else {
                                    tile_hash::hash_rect_from_ptr::<false>(
                                        slot_scans.key(),
                                        aos,
                                        stride_px,
                                        rect,
                                    )
                                };
                                // SAFETY: `slot_idx` is exclusively owned by this entry.
                                unsafe { slot_scans.write(slot_idx, scan) };
                            },
                            BaseScanEntry::Tile2D {
                                rect,
                                quad_slots,
                                n_quads,
                            } => {
                                let (quads, _) = tile_quadrants(rect);
                                for q in 0..n_quads {
                                    let scan = if has_alpha {
                                        tile_hash::hash_rect_from_ptr::<true>(
                                            slot_scans.key(),
                                            aos,
                                            stride_px,
                                            quads[q],
                                        )
                                    } else {
                                        tile_hash::hash_rect_from_ptr::<false>(
                                            slot_scans.key(),
                                            aos,
                                            stride_px,
                                            quads[q],
                                        )
                                    };
                                    // SAFETY: `quad_slots[q]` is exclusively owned by this entry.
                                    unsafe { slot_scans.write(quad_slots[q], scan) };
                                }
                            },
                        }
                    }
                }
            });
        }
        if let Some(shadow) = shadow_opt.as_deref_mut() {
            // SAFETY: Phase A workers have joined after writing every base tile
            // and zeroing trailing row-stride padding, initializing all
            // `byte_len` bytes.
            unsafe { shadow.pixels.set_len(byte_len) };
            if seed_quad_hashes {
                shadow.quad_hashes.mark_all_valid();
            }
        }

        let plan = cache.plan_frame(&base_entries, None, slot_scans.as_slice(), has_alpha, mode);
        let n_tiles = plan.tile_rects.len();
        let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
        let header = RawBufferHeader {
            frame_seq: 0,
            client,
            surface,
            metadata,
            incremental: false,
            scroll_rect: None,
            reset_tile_cache: plan.reset_tile_cache,
            evict_slots: plan.evict_slots,
            tile_rects: plan.tile_rects.clone(),
        };
        on_stream_ready(StreamingCompressedTiles {
            header,
            tiles: tiles_rx,
        });

        let shadow_ptr_storage;
        let encode_aos = if let Some(shadow) = shadow_opt.as_deref() {
            shadow_ptr_storage = shadow.pixels.as_ptr().cast::<Vec4u8>();
            // SAFETY: Phase A workers have joined, `shadow.pixels.len() == total_pixels * 4`,
            // and `Vec4u8` has alignment 1.
            unsafe { BufferPointer::new(&shadow_ptr_storage, total_pixels) }
        } else {
            aos
        };

        let encode_aos_is_shadow = shadow_opt.is_some();
        let is_hybrid = matches!(mode, CompressionMode::Hybrid { .. });
        let promotions: Vec<AtomicU8> = if is_hybrid {
            (0..n_tiles).map(|_| AtomicU8::new(0)).collect()
        } else {
            Vec::new()
        };
        let next_tile = AtomicUsize::new(0);
        compressor.pool.execute(n_tiles, |state| {
            loop {
                let tile_idx = next_tile.fetch_add(1, Ordering::Relaxed);
                if tile_idx >= n_tiles {
                    break;
                }
                let rect = plan.tile_rects[tile_idx];
                let meta = &plan.metas[tile_idx];
                let tile = match plan.actions[tile_idx] {
                    TileCacheAction::Hit(hit_ref) => CompressedTile {
                        tile_idx: tile_idx as u32,
                        cache_put: None,
                        payload: TilePayload::CacheHit(hit_ref),
                    },
                    TileCacheAction::Solid(px) => CompressedTile {
                        tile_idx: tile_idx as u32,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid(px)),
                    },
                    TileCacheAction::Put(slot_id) => {
                        let payload = if has_alpha {
                            filter_tile::<true>(
                                tier,
                                encode_aos,
                                encode_aos_is_shadow,
                                stride_px,
                                rect,
                                mode,
                                meta,
                                state,
                            )
                        } else {
                            filter_tile::<false>(
                                tier,
                                encode_aos,
                                encode_aos_is_shadow,
                                stride_px,
                                rect,
                                mode,
                                meta,
                                state,
                            )
                        };
                        if is_hybrid {
                            promotions[tile_idx].store(
                                payload_lossless_promotion_mask(rect, &payload),
                                Ordering::Relaxed,
                            );
                        }
                        CompressedTile {
                            tile_idx: tile_idx as u32,
                            cache_put: Some(slot_id),
                            payload,
                        }
                    },
                    TileCacheAction::Uncached => {
                        let payload = if has_alpha {
                            filter_tile::<true>(
                                tier,
                                encode_aos,
                                encode_aos_is_shadow,
                                stride_px,
                                rect,
                                mode,
                                meta,
                                state,
                            )
                        } else {
                            filter_tile::<false>(
                                tier,
                                encode_aos,
                                encode_aos_is_shadow,
                                stride_px,
                                rect,
                                mode,
                                meta,
                                state,
                            )
                        };
                        CompressedTile {
                            tile_idx: tile_idx as u32,
                            cache_put: None,
                            payload,
                        }
                    },
                };
                if tiles_tx.send(tile).is_err() {
                    break;
                }
            }
        });

        if is_hybrid {
            for (tile_idx, &action) in plan.actions.iter().enumerate() {
                if let TileCacheAction::Put(slot_id) = action {
                    let mask = promotions[tile_idx].load(Ordering::Relaxed);
                    if mask != 0 {
                        cache.promote_slot_lossless(slot_id, mask);
                    }
                }
            }
        }
        return;
    }

    let tile_rects: Vec<TileRect> = base_entries.into_iter().map(BaseScanEntry::rect).collect();
    let n_tiles = tile_rects.len();
    let rects_for_workers = tile_rects.clone();
    let shadow_rows = shadow_opt.as_deref_mut().map(|shadow| ShadowRows {
        base: shadow.pixels.as_mut_ptr(),
        stride,
    });

    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
    let header = RawBufferHeader {
        frame_seq: 0,
        client,
        surface,
        metadata,
        incremental: false,
        scroll_rect: None,
        reset_tile_cache: false,
        evict_slots: Vec::new(),
        tile_rects,
    };
    on_stream_ready(StreamingCompressedTiles {
        header,
        tiles: tiles_rx,
    });

    let next_tile = AtomicUsize::new(0);
    compressor.pool.execute(n_tiles, |state| {
        let mut disconnected = false;
        loop {
            let tile_idx = next_tile.fetch_add(1, Ordering::Relaxed);
            if tile_idx >= n_tiles {
                break;
            }
            let rect = rects_for_workers[tile_idx];
            let shadow_ptr;
            let (encode_aos, meta) = if let Some(shadow) = &shadow_rows {
                if disconnected {
                    // SAFETY: `rects_for_workers` are pairwise disjoint and in-bounds,
                    // and each `tile_idx` is claimed once.
                    unsafe {
                        copy_rect_to_shadow(data, shadow, stride, rect);
                        zero_shadow_row_padding(shadow, width_bytes, rect);
                    }
                    continue;
                }
                // SAFETY: `rects_for_workers` are pairwise disjoint and in-bounds,
                // and each `tile_idx` is claimed once.
                let scan_meta = unsafe {
                    let meta = if has_alpha {
                        tier.copy_and_scan_rect_to_shadow::<true>(data, shadow, stride, rect)
                    } else {
                        tier.copy_and_scan_rect_to_shadow::<false>(data, shadow, stride, rect)
                    };
                    zero_shadow_row_padding(shadow, width_bytes, rect);
                    meta
                };
                shadow_ptr = shadow.base.cast_const().cast::<Vec4u8>();
                // SAFETY: `shadow.base` points to `total_pixels * 4` bytes of
                // allocated capacity and this worker holds exclusive access to
                // `rect`'s initialized pixels after `copy_and_scan_rect_to_shadow`.
                (
                    unsafe { BufferPointer::new(&shadow_ptr, total_pixels) },
                    TileEncodeMeta::from_rect(scan_meta),
                )
            } else {
                (aos, TileEncodeMeta::UNKNOWN)
            };
            let payload = if has_alpha {
                filter_tile::<true>(
                    tier,
                    encode_aos,
                    shadow_rows.is_some(),
                    stride_px,
                    rect,
                    mode,
                    &meta,
                    state,
                )
            } else {
                filter_tile::<false>(
                    tier,
                    encode_aos,
                    shadow_rows.is_some(),
                    stride_px,
                    rect,
                    mode,
                    &meta,
                    state,
                )
            };

            if tiles_tx
                .send(CompressedTile {
                    tile_idx: tile_idx as u32,
                    cache_put: None,
                    payload,
                })
                .is_err()
            {
                if shadow_rows.is_some() {
                    disconnected = true;
                } else {
                    break;
                }
            }
        }
    });

    if let Some(shadow) = shadow_opt {
        // SAFETY: Every tile in `rects_for_workers` was copied into `shadow`
        // (even on early stream disconnect) and right-edge tiles zeroed
        // trailing row-stride padding, initializing all `byte_len` bytes.
        unsafe { shadow.pixels.set_len(byte_len) };
    }
}

/// Streams a full (`incremental = false`) frame covering the entire buffer.
/// When `shadow_out` is `Some`, each worker copies its tile from `data` into
/// `shadow_out` and encodes from that L2-hot copy so `shadow_out` always
/// matches the transmitted pixels even if shared memory mutates concurrently.
#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress_full_streaming(
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    shadow_out: Option<&mut ShadowBuffer>,
    tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) {
    filter_and_compress_full_streaming_with_tier(
        arch::Tier::detect(),
        data,
        client,
        surface,
        metadata,
        mode,
        shadow_out,
        tile_cache,
        compressor,
        on_stream_ready,
    );
}

/// Outcome of an incremental shadow diff, optional vertical scroll blit, and
/// dirty-tile encode.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IncrementalFrameResult {
    pub scroll_rect: Option<ScrollRect>,
    pub dirty_tiles: Vec<TileRect>,
}

impl IncrementalFrameResult {
    /// Returns `true` when neither a scroll blit nor any dirty tile was emitted.
    #[inline]
    pub fn is_unchanged(&self) -> bool {
        self.scroll_rect.is_none() && self.dirty_tiles.is_empty()
    }
}

/// Encodes and returns one Phase-B incremental tile from `shadow_aos`.
#[inline(always)]
fn encode_incremental_tile(
    tier: arch::Tier,
    shadow_aos: BufferPointer<Vec4u8>,
    has_alpha: bool,
    stride_px: usize,
    mode: CompressionMode,
    is_hybrid: bool,
    tile_idx: usize,
    rect: TileRect,
    action: TileCacheAction,
    meta: &TileEncodeMeta,
    promotions: &[AtomicU8],
    state: &mut CompressorWorkerState,
) -> CompressedTile {
    match action {
        TileCacheAction::Hit(hit_ref) => CompressedTile {
            tile_idx: tile_idx as u32,
            cache_put: None,
            payload: TilePayload::CacheHit(hit_ref),
        },
        TileCacheAction::Solid(px) => CompressedTile {
            tile_idx: tile_idx as u32,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Solid(px)),
        },
        TileCacheAction::Put(slot_id) => {
            let payload = if has_alpha {
                filter_tile::<true>(tier, shadow_aos, true, stride_px, rect, mode, meta, state)
            } else {
                filter_tile::<false>(tier, shadow_aos, true, stride_px, rect, mode, meta, state)
            };
            if is_hybrid {
                promotions[tile_idx].store(
                    payload_lossless_promotion_mask(rect, &payload),
                    Ordering::Relaxed,
                );
            }
            CompressedTile {
                tile_idx: tile_idx as u32,
                cache_put: Some(slot_id),
                payload,
            }
        },
        TileCacheAction::Uncached => {
            let payload = if has_alpha {
                filter_tile::<true>(tier, shadow_aos, true, stride_px, rect, mode, meta, state)
            } else {
                filter_tile::<false>(tier, shadow_aos, true, stride_px, rect, mode, meta, state)
            };
            CompressedTile {
                tile_idx: tile_idx as u32,
                cache_put: None,
                payload,
            }
        },
    }
}

/// Out-of-line Phase B exact-owner bucketing + work-stealing dispatch so the
/// stack bucket arrays (`~6.5 KiB`) never trigger stack probes on unchanged,
/// tiny, scroll, or single-threaded frames.
#[inline(never)]
fn run_phase_b_exact_affinity(
    compressor: &mut ShardingCompressor,
    tier: arch::Tier,
    shadow_aos: BufferPointer<Vec4u8>,
    has_alpha: bool,
    stride_px: usize,
    mode: CompressionMode,
    is_hybrid: bool,
    sub_bands: &[ScanSubBand],
    owners: &[AtomicU8],
    full_frame_2d: bool,
    p: usize,
    dirty_tiles: &[TileRect],
    actions: Option<&[TileCacheAction]>,
    metas: Option<&[TileEncodeMeta]>,
    promotions: &[AtomicU8],
    tiles_tx: &crossbeam_channel::Sender<CompressedTile>,
) {
    let n_tiles = dirty_tiles.len();
    let mut stack_tile_owners = [0u8; STACK_HOME_TILES_CAP];
    let mut heap_tile_owners;
    let tile_owners: &mut [u8] = if n_tiles <= STACK_HOME_TILES_CAP {
        &mut stack_tile_owners[..n_tiles]
    } else {
        heap_tile_owners = vec![0u8; n_tiles];
        &mut heap_tile_owners
    };
    let mut counts = [0u32; MAX_AFFINITY_WORKERS];
    for (rect, slot) in dirty_tiles.iter().zip(tile_owners.iter_mut()) {
        let w = phase_b_owner_non_scroll(sub_bands, owners, *rect, full_frame_2d, p);
        *slot = w as u8;
        counts[w] += 1;
    }
    let mut bucket_storage = [const { MaybeUninit::<WorkBucket>::uninit() }; MAX_AFFINITY_WORKERS];
    let mut write_pos = [0usize; MAX_AFFINITY_WORKERS];
    let mut cursor = 0usize;
    for w in 0..p {
        let start = cursor;
        cursor += counts[w] as usize;
        write_pos[w] = start;
        bucket_storage[w].write(WorkBucket::new(start, cursor));
    }
    let mut stack_home_tiles = [0u32; STACK_HOME_TILES_CAP];
    let mut heap_home_tiles;
    let home: &mut [u32] = if n_tiles <= STACK_HOME_TILES_CAP {
        &mut stack_home_tiles[..n_tiles]
    } else {
        heap_home_tiles = vec![0u32; n_tiles];
        &mut heap_home_tiles
    };
    for (tile_idx, &w) in tile_owners.iter().enumerate() {
        let w = usize::from(w);
        home[write_pos[w]] = tile_idx as u32;
        write_pos[w] += 1;
    }
    // SAFETY: `bucket_storage[..p]` were all initialized above, `WorkBucket`
    // needs no `Drop`, and `MaybeUninit<WorkBucket>` has identical layout.
    let buckets = unsafe { slice::from_raw_parts(bucket_storage.as_ptr().cast::<WorkBucket>(), p) };
    compressor.pool.execute(n_tiles, |state| {
        let mut iter = WorkStealIter::new(buckets, state.worker_idx);
        while let Some(pos) = iter.next_idx() {
            let tile_idx = home[pos] as usize;
            let rect = dirty_tiles[tile_idx];
            let action = actions.map_or(TileCacheAction::Uncached, |acts| acts[tile_idx]);
            let meta = metas.map_or(&TileEncodeMeta::UNKNOWN, |m| &m[tile_idx]);
            let tile = encode_incremental_tile(
                tier, shadow_aos, has_alpha, stride_px, mode, is_hybrid, tile_idx, rect, action,
                meta, promotions, state,
            );
            if tiles_tx.send(tile).is_err() {
                break;
            }
        }
    });
}

/// Out-of-line Phase A spatial-bucket wrapper so the 4 KiB bucket array never
/// lives on `filter_and_compress_incremental_streaming_with_tier`'s stack frame.
#[inline(never)]
fn execute_phase_a_spatial<F>(total_tasks: usize, p: usize, owners: Option<&[AtomicU8]>, run: F)
where
    F: FnOnce(&PhaseASchedule<'_>),
{
    let mut bucket_storage = [const { MaybeUninit::<WorkBucket>::uninit() }; MAX_AFFINITY_WORKERS];
    for (w, slot) in bucket_storage[..p].iter_mut().enumerate() {
        let (start, end) = contiguous_slice(total_tasks, p, w);
        slot.write(WorkBucket::new(start, end));
    }
    // SAFETY: `bucket_storage[..p]` were all initialized above.
    let buckets = unsafe { slice::from_raw_parts(bucket_storage.as_ptr().cast::<WorkBucket>(), p) };
    let schedule = PhaseASchedule {
        buckets,
        task_owners: owners,
    };
    run(&schedule);
}

fn filter_and_compress_incremental_streaming_impl(
    tier: arch::Tier,
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    bands: Option<&[RowBand]>,
    shadow: &mut ShadowBuffer,
    mut tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
    enable_screening: bool,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) -> IncrementalFrameResult {
    assert!(data.len().is_multiple_of(4));
    let total_pixels = metadata
        .validate()
        .expect("server passed invalid BufferMetadata");
    mode.validate()
        .expect("server passed invalid CompressionMode");
    let byte_len = total_pixels * 4;
    assert!(data.len() >= byte_len);
    assert!(shadow.pixels.len() >= byte_len);

    let (sub_bands, base_entries, total_slots) =
        plan_incremental_scan(&metadata, bands).expect("server generated invalid row bands");
    if sub_bands.is_empty() {
        return IncrementalFrameResult {
            scroll_rect: None,
            dirty_tiles: Vec::new(),
        };
    }

    let width = metadata.width as usize;
    let height = metadata.height as usize;
    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let n_qx = width.div_ceil(SUBTILE_W);
    let n_sub = sub_bands.len();
    let dirty_slots: Vec<AtomicU32> = (0..total_slots)
        .map(|_| AtomicU32::new(SlotDirty::CLEAN_RAW))
        .collect();
    let mut slot_scans = tile_cache
        .as_deref()
        .map(|c| SlotScans::new(total_slots, *c.hash_key()));
    let cache_non_empty = tile_cache
        .as_deref()
        .is_some_and(|c| c.occupied_slots() > 0);
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };

    // Detect vertical scroll blit (plus coarse Tier-0 dense-dirty band mask
    // with zero extra memory reads) and fuse overlap-safe parallel
    // column-stripe shadow scrolling into Phase A when present, leaving the
    // non-scroll Phase A worker loop untouched for zero codegen overhead on
    // non-scroll frames.
    let scroll_outcome =
        scroll::detect_scroll_and_activity_with_tier(data, &shadow.pixels, metadata, bands, tier);
    let scroll_rect = scroll_outcome.scroll_rect;

    // Phase A: parallel dirty scan + tight bbox tracking + shadow update (+
    // dirty quadrant hashing and pixel-property scan when `tile_cache` is
    // enabled, with optional fused scroll). On non-scroll multi-worker frames,
    // each worker scans a deterministic contiguous spatial slice of horizontal
    // sub-bands (keeping the `wl_shm` + `shadow` working set resident in its
    // local L2/CCX L3 across consecutive frames) and records which worker
    // scanned each sub-band so Phase B can route dirty tiles back to the same
    // worker. Scroll frames keep dynamic column stealing unchanged.
    let p = compressor.pool.parallelism();
    let use_affinity = scroll_rect.is_none() && p > 1 && p <= MAX_AFFINITY_WORKERS;
    let stack_owners = [const { AtomicU8::new(0) }; MAX_STACK_OWNERS];
    let heap_owners;
    let phase_a_owners: Option<&[AtomicU8]> = if use_affinity {
        if n_sub <= MAX_STACK_OWNERS {
            Some(&stack_owners[..n_sub])
        } else {
            heap_owners = (0..n_sub).map(|_| AtomicU8::new(0)).collect::<Vec<_>>();
            Some(&heap_owners)
        }
    } else {
        None
    };
    {
        if let Some(sr) = scroll_rect {
            let shadow_rows = ShadowRows {
                base: shadow.pixels.as_mut_ptr(),
                stride,
            };
            let (sb_start, sb_end) = scroll_sub_band_range(&sub_bands, sr);
            let total_tasks = (n_sub - (sb_end - sb_start)) + n_qx;
            let next_task = AtomicUsize::new(0);
            compressor.pool.execute(total_tasks, |_state| {
                // SAFETY: `sub_bands` cover disjoint regions in `shadow`, and
                // `sr` lies inside a contiguous `RowBand2D` slice.
                unsafe {
                    tier.diff_and_update_scroll_worker(
                        has_alpha,
                        data,
                        aos,
                        &shadow_rows,
                        width,
                        stride,
                        stride_px,
                        n_qx,
                        &sub_bands,
                        sr,
                        &dirty_slots,
                        slot_scans.as_ref(),
                        cache_non_empty,
                        &next_task,
                    );
                }
            });
            if tile_cache.as_deref().is_some_and(|c| {
                shadow
                    .quad_hashes
                    .matches(width, height, stride, has_alpha, c.hash_key())
            }) {
                for sb in &sub_bands[sb_start..sb_end] {
                    match *sb {
                        ScanSubBand::RowBand2D { y0, y1, .. } => {
                            shadow
                                .quad_hashes
                                .invalidate_y_range(y0 as usize, y1 as usize);
                        },
                        ScanSubBand::Strip1D { rect, .. } => {
                            shadow
                                .quad_hashes
                                .invalidate_y_range(rect.y as usize, (rect.y + rect.h) as usize);
                        },
                    }
                }
                for sb in sub_bands[..sb_start].iter().chain(&sub_bands[sb_end..]) {
                    match *sb {
                        ScanSubBand::Strip1D { rect, slot_idx } => {
                            if dirty_slots[slot_idx].load(Ordering::Relaxed) != SlotDirty::CLEAN_RAW
                            {
                                shadow.quad_hashes.invalidate_y_range(
                                    rect.y as usize,
                                    (rect.y + rect.h) as usize,
                                );
                            }
                        },
                        ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                            if dirty_slots[slot_base..slot_base + n_qx]
                                .iter()
                                .any(|s| s.load(Ordering::Relaxed) != SlotDirty::CLEAN_RAW)
                            {
                                shadow
                                    .quad_hashes
                                    .invalidate_y_range(y0 as usize, y1 as usize);
                            }
                        },
                    }
                }
            } else {
                shadow.quad_hashes.invalidate_all();
            }
        } else {
            let phase_a_qh = if enable_screening
                && let Some(cache) = tile_cache.as_deref()
                && shadow
                    .quad_hashes
                    .matches(width, height, stride, has_alpha, cache.hash_key())
            {
                // Invalidate any grid band overlapping a non-grid-aligned
                // sub-band before launching workers so workers touch strictly
                // disjoint grid-aligned `qy` bands.
                for sb in &sub_bands {
                    match *sb {
                        ScanSubBand::Strip1D { rect, .. } => {
                            shadow
                                .quad_hashes
                                .invalidate_y_range(rect.y as usize, (rect.y + rect.h) as usize);
                        },
                        ScanSubBand::RowBand2D { y0, y1, .. } => {
                            let y0 = y0 as usize;
                            let y1 = y1 as usize;
                            if !y0.is_multiple_of(SUBTILE_H) || y1 != (y0 + SUBTILE_H).min(height) {
                                shadow.quad_hashes.invalidate_y_range(y0, y1);
                            }
                        },
                    }
                }
                Some(PhaseAQuadHashes {
                    sync: shadow.quad_hashes.sync_view(),
                    height,
                    dense_dirty_bands: scroll_outcome.dense_dirty_bands,
                })
            } else {
                shadow.quad_hashes.invalidate_all();
                None
            };
            let shadow_rows = ShadowRows {
                base: shadow.pixels.as_mut_ptr(),
                stride,
            };
            let mut run_non_scroll = |schedule: &PhaseASchedule<'_>| {
                compressor.pool.execute(n_sub, |state| {
                    // SAFETY: `sub_bands` cover disjoint regions in `shadow`.
                    unsafe {
                        tier.diff_and_update_worker(
                            has_alpha,
                            data,
                            aos,
                            &shadow_rows,
                            width,
                            stride,
                            stride_px,
                            n_qx,
                            &sub_bands,
                            &dirty_slots,
                            slot_scans.as_ref(),
                            cache_non_empty,
                            phase_a_qh.as_ref(),
                            schedule,
                            state.worker_idx,
                        );
                    }
                });
            };
            if use_affinity {
                execute_phase_a_spatial(n_sub, p, phase_a_owners, run_non_scroll);
            } else {
                let single_bucket = WorkBucket::new(0, n_sub);
                let schedule = PhaseASchedule::single(&single_bucket, None);
                run_non_scroll(&schedule);
            }
        }
    }

    let any_dirty = dirty_slots
        .iter()
        .any(|f| f.load(Ordering::Relaxed) != SlotDirty::CLEAN_RAW);
    if scroll_rect.is_none() && !any_dirty {
        return IncrementalFrameResult {
            scroll_rect: None,
            dirty_tiles: Vec::new(),
        };
    }

    let (dirty_tiles, actions, metas, reset_tile_cache, evict_slots) = if let Some(cache) =
        tile_cache.as_deref_mut()
    {
        let scans = slot_scans
            .as_mut()
            .expect("slot_scans allocated when tile_cache is Some")
            .as_slice();
        let plan = cache.plan_frame(&base_entries, Some(&dirty_slots), scans, has_alpha, mode);
        (
            plan.tile_rects,
            Some(plan.actions),
            Some(plan.metas),
            plan.reset_tile_cache,
            plan.evict_slots,
        )
    } else {
        let mut dirty_tiles = Vec::new();
        for entry in &base_entries {
            match *entry {
                BaseScanEntry::Strip1D { rect, slot_idx } => {
                    if dirty_slots[slot_idx].load(Ordering::Relaxed) != SlotDirty::CLEAN_RAW {
                        dirty_tiles.push(rect);
                    }
                },
                BaseScanEntry::Tile2D {
                    rect,
                    quad_slots,
                    n_quads,
                } => {
                    let mut states = [SlotDirty::Clean; 4];
                    let mut full_count = 0usize;
                    let mut dirty_count = 0usize;
                    for (q, &slot) in quad_slots[..n_quads].iter().enumerate() {
                        let st = SlotDirty::decode(dirty_slots[slot].load(Ordering::Relaxed));
                        states[q] = st;
                        match st {
                            SlotDirty::Clean => {},
                            SlotDirty::Full => {
                                full_count += 1;
                                dirty_count += 1;
                            },
                            SlotDirty::Cropped { .. } => {
                                dirty_count += 1;
                            },
                        }
                    }
                    if dirty_count == 0 {
                        continue;
                    }
                    if full_count == n_quads {
                        dirty_tiles.push(rect);
                    } else {
                        let (quads, _) = tile_quadrants(rect);
                        let mut full_grid = [None; 4];
                        let mut crop_grid = [None; 4];
                        for q in 0..n_quads {
                            let qrect = quads[q];
                            let slot = quad_grid_slot(rect, qrect);
                            match states[q] {
                                SlotDirty::Clean => {},
                                SlotDirty::Full => full_grid[slot] = Some(qrect),
                                SlotDirty::Cropped {
                                    min_x,
                                    max_x,
                                    min_y,
                                    max_y,
                                    ..
                                } => {
                                    crop_grid[slot] = Some((
                                        SlotDirty::cropped_rect(qrect, min_x, max_x, min_y, max_y),
                                        RectScanMeta::UNKNOWN,
                                    ));
                                },
                            }
                        }
                        merge_cropped_quad_slots(&mut crop_grid);
                        for s in 0..4 {
                            if let Some(r) = full_grid[s].or(crop_grid[s].map(|(r, _)| r)) {
                                dirty_tiles.push(r);
                            }
                        }
                    }
                },
            }
        }
        (dirty_tiles, None, None, false, Vec::new())
    };

    // Phase B: stream dirty tiles encoded directly from the updated `shadow`
    // buffer (skipping encoding completely for `CacheHit` and `Solid` tiles).
    let n_tiles = dirty_tiles.len();
    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
    let header = RawBufferHeader {
        frame_seq: 0,
        client,
        surface,
        metadata,
        incremental: true,
        scroll_rect,
        reset_tile_cache,
        evict_slots,
        tile_rects: dirty_tiles.clone(),
    };
    on_stream_ready(StreamingCompressedTiles {
        header,
        tiles: tiles_rx,
    });

    let shadow_ptr = shadow.pixels.as_ptr().cast::<Vec4u8>();
    // SAFETY: `shadow.pixels` has length `>= total_pixels * 4`, Phase A workers have
    // joined, and `Vec4u8` is `repr(C, packed)` with alignment 1.
    let shadow_aos = unsafe { BufferPointer::new(&shadow_ptr, total_pixels) };
    let is_hybrid = actions.is_some() && matches!(mode, CompressionMode::Hybrid { .. });
    let promotions: Vec<AtomicU8> = if is_hybrid {
        (0..n_tiles).map(|_| AtomicU8::new(0)).collect()
    } else {
        Vec::new()
    };
    let use_phase_b_affinity = if let Some(owners) = phase_a_owners {
        let n_encode_tiles = match actions.as_deref() {
            None => n_tiles,
            Some(acts) => acts
                .iter()
                .filter(|a| matches!(a, TileCacheAction::Put(_) | TileCacheAction::Uncached))
                .count(),
        };
        (n_encode_tiles >= PHASE_B_AFFINITY_MIN_ENCODE).then_some(owners)
    } else {
        None
    };
    if let Some(owners) = use_phase_b_affinity {
        let full_frame_2d = bands.is_none() && metadata.height as u32 >= MIN_2D_BAND_HEIGHT;
        run_phase_b_exact_affinity(
            compressor,
            tier,
            shadow_aos,
            has_alpha,
            stride_px,
            mode,
            is_hybrid,
            &sub_bands,
            owners,
            full_frame_2d,
            p,
            &dirty_tiles,
            actions.as_deref(),
            metas.as_deref(),
            &promotions,
            &tiles_tx,
        );
    } else {
        let next_tile = AtomicUsize::new(0);
        compressor.pool.execute(n_tiles, |state| {
            loop {
                let tile_idx = next_tile.fetch_add(1, Ordering::Relaxed);
                if tile_idx >= n_tiles {
                    break;
                }
                let rect = dirty_tiles[tile_idx];
                let action = actions
                    .as_ref()
                    .map_or(TileCacheAction::Uncached, |acts| acts[tile_idx]);
                let meta = metas
                    .as_ref()
                    .map_or(&TileEncodeMeta::UNKNOWN, |m| &m[tile_idx]);
                let tile = encode_incremental_tile(
                    tier,
                    shadow_aos,
                    has_alpha,
                    stride_px,
                    mode,
                    is_hybrid,
                    tile_idx,
                    rect,
                    action,
                    meta,
                    &promotions,
                    state,
                );
                if tiles_tx.send(tile).is_err() {
                    break;
                }
            }
        });
    }

    if is_hybrid
        && let Some(cache) = tile_cache
        && let Some(acts) = actions
    {
        for (tile_idx, action) in acts.into_iter().enumerate() {
            if let TileCacheAction::Put(slot_id) = action {
                let mask = promotions[tile_idx].load(Ordering::Relaxed);
                if mask != 0 {
                    cache.promote_slot_lossless(slot_id, mask);
                }
            }
        }
    }

    IncrementalFrameResult {
        scroll_rect,
        dirty_tiles,
    }
}

fn filter_and_compress_incremental_streaming_with_tier(
    tier: arch::Tier,
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    bands: Option<&[RowBand]>,
    shadow: &mut ShadowBuffer,
    tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) -> IncrementalFrameResult {
    let frame_bytes = (metadata.height as usize).saturating_mul(metadata.stride as usize);
    let enable_screening = tile_hash::has_hw_aes()
        && (compressor.pool.parallelism() > 1 || frame_bytes >= SCREEN_1T_MIN_BYTES);
    filter_and_compress_incremental_streaming_impl(
        tier,
        data,
        client,
        surface,
        metadata,
        mode,
        bands,
        shadow,
        tile_cache,
        compressor,
        enable_screening,
        on_stream_ready,
    )
}

/// Diffs `data` against `shadow` across `bands` (or the full frame when `bands`
/// is `None`), optionally detects and applies a vertical [`ScrollRect`], updates
/// `shadow` for every changed `128x128` quadrant or `< 8`-row 1D strip, and
/// streams only the dirty tiles (`incremental = true`) from `shadow`.
///
/// Returns the optional [`ScrollRect`] and transmitted dirty [`TileRect`]s in
/// deterministic raster order. When no pixels changed, returns an unchanged
/// result without invoking `on_stream_ready`.
#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress_incremental_streaming(
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    bands: Option<&[RowBand]>,
    shadow: &mut ShadowBuffer,
    tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) -> IncrementalFrameResult {
    filter_and_compress_incremental_streaming_with_tier(
        arch::Tier::detect(),
        data,
        client,
        surface,
        metadata,
        mode,
        bands,
        shadow,
        tile_cache,
        compressor,
        on_stream_ready,
    )
}

#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress(
    data: BufferPointer<u8>,
    metadata: BufferMetadata,
    mode: CompressionMode,
    compressor: &mut ShardingCompressor,
) -> CompressedTiles {
    let mut streaming_opt = None;
    filter_and_compress_full_streaming(
        data,
        ClientId(0),
        WlSurfaceId(0),
        metadata,
        mode,
        None,
        None,
        compressor,
        |streaming| {
            streaming_opt = Some(streaming);
        },
    );
    let streaming = streaming_opt.unwrap();
    let tile_rects = streaming.header.tile_rects;
    let mut tiles: Vec<CompressedTile> = streaming.tiles.into_iter().collect();
    tiles.sort_by_key(|t| t.tile_idx);
    CompressedTiles {
        metadata,
        tile_rects,
        tiles,
    }
}

/// Decodes a validated set of pairwise-disjoint `tile_rects` in parallel using
/// `tier`. When `output_buf` is `None`, runs in puts-only lockstep mode
/// applying cache resets, evictions, and `CachePut` tiles directly into
/// `tile_cache` without touching any surface buffer.
///
/// # Safety
/// Caller must have validated `metadata` (`byte_len = total_pixels * 4`) and
/// verified every rect in `tile_rects` via [`validate_tile_rects`].
unsafe fn unfilter_tiles_parallel_with_tier<'a>(
    tier: arch::Tier,
    decompressor: &mut ShardingDecompressor,
    metadata: BufferMetadata,
    byte_len: usize,
    reset_tile_cache: bool,
    evict_slots: &[u16],
    tile_rects: &[TileRect],
    output_buf: Option<&mut [u8]>,
    mut tile_cache: Option<&mut ClientTileCache>,
    next_tile: impl Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
) -> Result<()> {
    let puts_only = output_buf.is_none();
    if puts_only && tile_cache.is_none() {
        bail!("puts-only decode requires ClientTileCache");
    }
    if let Some(cache) = tile_cache.as_deref_mut() {
        cache
            .apply_header_ops(reset_tile_cache, evict_slots)
            .location(loc!())?;
    } else if reset_tile_cache || !evict_slots.is_empty() {
        bail!("received tile cache header ops without active ClientTileCache");
    }

    let frame = match output_buf {
        Some(buf) => {
            if buf.len() < byte_len {
                bail!("output_buf len {} < {byte_len}", buf.len());
            }
            let stride_px = (metadata.stride as usize) / 4;
            let aos: &mut [Vec4u8] = bytemuck::cast_slice_mut(&mut buf[..byte_len]);
            FrameRows {
                base: aos.as_mut_ptr().cast(),
                stride_px,
            }
        },
        None => FrameRows {
            base: ptr::NonNull::dangling().as_ptr(),
            stride_px: 0,
        },
    };

    let n_tiles = tile_rects.len();
    if n_tiles == 0 {
        return Ok(());
    }

    let has_alpha = metadata.format == BufferFormat::Argb8888;
    // Every `tile_rects` entry was validated in-bounds and pairwise disjoint
    // upfront; an atomic slot per index rejects duplicate or out-of-range
    // `tile_idx` without mutex contention before any worker writes `frame`.
    let claimed: Vec<AtomicBool> = (0..n_tiles).map(|_| AtomicBool::new(false)).collect();
    let completed = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    // Step 1: decode non-hit tiles in parallel and populate pairwise-disjoint
    // `CachePut` slots while recording `CacheHit` references.
    let step1_hits = {
        let cache_ctx = tile_cache.as_deref_mut().map(|cache| DecodeCacheCtx {
            slots: cache.sync_slots(),
            put_claimed: Box::new(std::array::from_fn(|_| AtomicBool::new(false))),
            frame_put_bytes: AtomicUsize::new(0),
            hit_refs: if puts_only {
                Vec::new()
            } else {
                (0..n_tiles).map(|_| AtomicU32::new(0)).collect()
            },
            hit_count: AtomicUsize::new(0),
            puts_only,
        });

        decompressor.pool.execute(n_tiles, |state| {
            // SAFETY: Caller validated `tile_rects` in-bounds and pairwise disjoint.
            unsafe {
                tier.unfilter_tiles_worker(
                    has_alpha,
                    tile_rects,
                    &frame,
                    &claimed,
                    &completed,
                    &first_err,
                    cache_ctx.as_ref(),
                    &next_tile,
                    state,
                );
            }
        });

        cache_ctx.map(|ctx| {
            (
                ctx.hit_refs,
                ctx.hit_count.into_inner(),
                ctx.frame_put_bytes.into_inner(),
            )
        })
    };

    // Recompute cache byte accounting after Step 1 joins whenever any `CachePut`
    // touched the cache (including partially failed frames so stale byte
    // accounting can never survive). Hit-only / solid-only frames modify zero
    // cache slots and skip the 1024-slot scan in O(1).
    let had_puts = step1_hits
        .as_ref()
        .is_some_and(|(_, _, put_bytes)| *put_bytes != 0);
    let cache_budget_res = if had_puts {
        tile_cache
            .as_deref_mut()
            .map(ClientTileCache::recompute_and_verify_bytes)
            .transpose()
    } else {
        Ok(None)
    };

    if let Some(err) = first_err.into_inner().unwrap() {
        return Err(err);
    }
    let done = completed.load(Ordering::Relaxed);
    if done != n_tiles {
        bail!("received {done} tiles, expected {n_tiles}");
    }
    cache_budget_res.location(loc!())?;

    // Step 2: validate and blit all `CacheHit` tiles in parallel from
    // immutable `ClientTileCache` slots into disjoint rects of `frame`.
    if !puts_only
        && let Some((hit_refs, hit_count, _)) = step1_hits
        && hit_count > 0
        && let Some(cache) = tile_cache.as_deref()
    {
        let _blit_span = debug_span!("tile_cache_hit_blit", hits = hit_count).entered();
        let mut hits = Vec::with_capacity(hit_count);
        for (tile_idx, cell) in hit_refs.into_iter().enumerate() {
            let raw = cell.into_inner();
            if raw != 0 {
                let hit_ref = CacheHitRef::from_u16((raw - 1) as u16).location(loc!())?;
                let valid = cache
                    .validate_hit(tile_idx, hit_ref, tile_rects[tile_idx], has_alpha)
                    .location(loc!())?;
                hits.push(valid);
            }
        }
        let n_hits = hits.len();
        let next_hit = AtomicUsize::new(0);
        decompressor.pool.execute(n_hits, |_| {
            // SAFETY: Every `hit` was validated via `validate_hit`, and
            // `tile_rects` are validated in-bounds and pairwise disjoint.
            unsafe {
                tier.blit_hits_worker(cache, &hits, tile_rects, &frame, &next_hit);
            }
        });
    }

    Ok(())
}

/// Decodes a validated set of pairwise-disjoint `tile_rects` in parallel.
///
/// # Safety
/// Caller must have validated `metadata` (`byte_len = total_pixels * 4`) and
/// verified every rect in `tile_rects` via [`validate_tile_rects`].
unsafe fn unfilter_tiles_parallel<'a>(
    decompressor: &mut ShardingDecompressor,
    metadata: BufferMetadata,
    byte_len: usize,
    reset_tile_cache: bool,
    evict_slots: &[u16],
    tile_rects: &[TileRect],
    output_buf: Option<&mut [u8]>,
    tile_cache: Option<&mut ClientTileCache>,
    next_tile: impl Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
) -> Result<()> {
    // SAFETY: Caller upholds validated-metadata and disjoint-tile invariants.
    unsafe {
        unfilter_tiles_parallel_with_tier(
            arch::Tier::detect(),
            decompressor,
            metadata,
            byte_len,
            reset_tile_cache,
            evict_slots,
            tile_rects,
            output_buf,
            tile_cache,
            next_tile,
        )
    }
}

pub struct ValidatedTileStream {
    stream: CompressedTileStream,
    byte_len: usize,
}

impl ValidatedTileStream {
    pub fn new(stream: CompressedTileStream) -> Result<Self> {
        let header = &stream.header;
        let total_pixels = header.metadata.validate().location(loc!())?;
        let byte_len = total_pixels
            .checked_mul(4)
            .context(loc!(), "pixel count overflow")?;
        let width = header.metadata.width as u32;
        let height = header.metadata.height as u32;
        let max_tiles = (u64::from(width) * u64::from(height)).min(MAX_FRAME_TILES as u64);
        if header.tile_rects.len() as u64 > max_tiles {
            bail!(
                "header tile_rects len {} exceeds maximum {max_tiles} for {width}x{height}",
                header.tile_rects.len(),
            );
        }
        if header.reset_tile_cache && !header.evict_slots.is_empty() {
            bail!("header cannot set both reset_tile_cache and evict_slots");
        }
        if !header.evict_slots.is_empty() {
            if header.evict_slots.len() > MAX_TILE_CACHE_SLOTS {
                bail!(
                    "evict_slots len {} exceeds MAX_TILE_CACHE_SLOTS ({MAX_TILE_CACHE_SLOTS})",
                    header.evict_slots.len()
                );
            }
            let mut seen = [false; MAX_TILE_CACHE_SLOTS];
            for &slot_id in &header.evict_slots {
                let idx = slot_id as usize;
                if idx >= MAX_TILE_CACHE_SLOTS {
                    bail!("evict slot_id {slot_id} out of range (max {MAX_TILE_CACHE_SLOTS})");
                }
                if seen[idx] {
                    bail!("duplicate evict slot_id {slot_id}");
                }
                seen[idx] = true;
            }
        }
        if let Some(sr) = header.scroll_rect {
            if !header.incremental {
                bail!("non-incremental frame cannot carry ScrollRect");
            }
            sr.validate_in_bounds(width, height).location(loc!())?;
        }
        let covered_pixels =
            validate_tile_rects(&header.tile_rects, width, height).location(loc!())?;
        if !header.incremental && covered_pixels != u64::from(width) * u64::from(height) {
            bail!(
                "non-incremental frame covers {covered_pixels} pixels, expected {width}x{height}",
            );
        }
        Ok(Self { stream, byte_len })
    }

    pub fn metadata(&self) -> BufferMetadata {
        self.stream.header.metadata
    }

    pub fn scroll_rect(&self) -> Option<ScrollRect> {
        self.stream.header.scroll_rect
    }

    pub fn incremental_tile_rects(&self) -> Option<&[TileRect]> {
        self.stream
            .header
            .incremental
            .then_some(self.stream.header.tile_rects.as_slice())
    }
}

#[instrument(skip_all, level = "debug")]
pub fn decompress_and_unfilter_streaming(
    validated: ValidatedTileStream,
    decompressor: &mut ShardingDecompressor,
    output_buf: &mut [u8],
    tile_cache: Option<&mut ClientTileCache>,
) -> Result<()> {
    let header = &validated.stream.header;
    let tiles_rx = &validated.stream.tiles;
    // SAFETY: `ValidatedTileStream::new` validated `metadata`, `byte_len`, and
    // `validate_tile_rects(tile_rects, width, height)`.
    unsafe {
        unfilter_tiles_parallel(
            decompressor,
            header.metadata,
            validated.byte_len,
            header.reset_tile_cache,
            &header.evict_slots,
            &header.tile_rects,
            Some(output_buf),
            tile_cache,
            || tiles_rx.recv().ok().map(|r| r.map(Cow::Owned)),
        )
    }
}

/// Applies header cache operations (`reset_tile_cache`, `evict_slots`) and
/// decodes only `CachePut` tiles directly into `tile_cache`, skipping surface
/// canvas blits. Used when a surface triggers `BufferResyncRequested` (e.g.
/// missing prior buffer) so connection-wide tile cache slots remain in lockstep
/// for other surfaces and already-pipelined frames.
#[instrument(skip_all, level = "debug")]
pub fn decompress_cache_puts_only(
    validated: ValidatedTileStream,
    decompressor: &mut ShardingDecompressor,
    tile_cache: &mut ClientTileCache,
) -> Result<()> {
    let header = &validated.stream.header;
    let tiles_rx = &validated.stream.tiles;
    // SAFETY: `ValidatedTileStream::new` validated `metadata`, `byte_len`, and
    // `validate_tile_rects(tile_rects, width, height)`.
    unsafe {
        unfilter_tiles_parallel(
            decompressor,
            header.metadata,
            validated.byte_len,
            header.reset_tile_cache,
            &header.evict_slots,
            &header.tile_rects,
            None,
            Some(tile_cache),
            || tiles_rx.recv().ok().map(|r| r.map(Cow::Owned)),
        )
    }
}

#[instrument(skip_all, level = "debug")]
pub fn decompress_and_unfilter(
    tiles: &CompressedTiles,
    decompressor: &mut ShardingDecompressor,
    output_buf: &mut [u8],
) -> Result<()> {
    let total_pixels = tiles.metadata.validate().location(loc!())?;
    let byte_len = total_pixels
        .checked_mul(4)
        .context(loc!(), "pixel count overflow")?;
    let width = tiles.metadata.width as u32;
    let height = tiles.metadata.height as u32;
    validate_tile_rects(&tiles.tile_rects, width, height).location(loc!())?;
    if tiles.tiles.len() != tiles.tile_rects.len() {
        bail!(
            "tiles len {} != tile_rects len {}",
            tiles.tiles.len(),
            tiles.tile_rects.len(),
        );
    }

    let n_tiles = tiles.tiles.len();
    let next_idx = AtomicUsize::new(0);
    // SAFETY: Validated `tiles.metadata` and `validate_tile_rects(&tiles.tile_rects)` above.
    unsafe {
        unfilter_tiles_parallel(
            decompressor,
            tiles.metadata,
            byte_len,
            false,
            &[],
            &tiles.tile_rects,
            Some(output_buf),
            None,
            || {
                let i = next_idx.fetch_add(1, Ordering::Relaxed);
                (i < n_tiles).then(|| Ok(Cow::Borrowed(&tiles.tiles[i])))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches;
    use std::cell::RefCell;
    use std::num::NonZeroUsize;

    use proptest::prelude::*;
    use rkyv::util::AlignedVec;

    use super::*;

    thread_local! {
        static TEST_CODEC: RefCell<Option<(ShardingCompressor, ShardingDecompressor)>> =
            const { RefCell::new(None) };
    }

    fn with_test_codec(f: impl FnOnce(&mut ShardingCompressor, &mut ShardingDecompressor)) {
        TEST_CODEC.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (compressor, decompressor) = borrow.get_or_insert_with(|| {
                (
                    ShardingCompressor::new(NonZeroUsize::new(4).unwrap(), 1).unwrap(),
                    ShardingDecompressor::new(NonZeroUsize::new(4).unwrap()).unwrap(),
                )
            });
            f(compressor, decompressor);
        });
    }

    fn test_vec(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i.wrapping_mul(37).wrapping_add(i / 17) & 0xFF) as u8)
            .collect()
    }

    fn assert_decodes_at_both_alignments(
        tiles: &CompressedTiles,
        expected: &[u8],
        decompressor: &mut ShardingDecompressor,
    ) {
        let mut buf = vec![0xAAu8; expected.len() + 68];
        let aligned_off = (64 - (buf.as_ptr() as usize & 63)) & 63;
        decompress_and_unfilter(
            tiles,
            decompressor,
            &mut buf[aligned_off..aligned_off + expected.len()],
        )
        .unwrap();
        assert_eq!(expected, &buf[aligned_off..aligned_off + expected.len()]);

        // Poison before unaligned decode (`STREAM = false` on x86) so solid
        // tiles cannot pass on stale bytes left behind by the aligned decode.
        buf.fill(0xAA);
        decompress_and_unfilter(
            tiles,
            decompressor,
            &mut buf[aligned_off + 4..aligned_off + 4 + expected.len()],
        )
        .unwrap();
        assert_eq!(
            expected,
            &buf[aligned_off + 4..aligned_off + 4 + expected.len()]
        );
    }

    fn test_roundtrip_2d(
        data: &[u8],
        width: i32,
        height: i32,
        format: BufferFormat,
        compressor: &mut ShardingCompressor,
        decompressor: &mut ShardingDecompressor,
    ) -> CompressedTiles {
        let metadata = BufferMetadata {
            width,
            height,
            stride: width * 4,
            format,
        };
        assert_eq!(data.len(), (width * height * 4) as usize);
        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        let tiles = filter_and_compress(buf_ptr, metadata, CompressionMode::Lossless, compressor);
        let mut expected = data.to_vec();
        if format == BufferFormat::Xrgb8888 {
            for px in expected.as_chunks_mut::<4>().0 {
                px[3] = 0xFF;
            }
        }
        assert_decodes_at_both_alignments(&tiles, &expected, decompressor);
        tiles
    }

    /// Compares `kernel` bit-for-bit against [`ScalarKernel`] across all edge
    /// widths (`1..=65`, `127..=129`, `255`, `256`, `300`) for every row
    /// primitive.
    #[inline(always)]
    pub(super) fn verify_kernel_matches_scalar<K: RowKernel>(kernel: K, tier_name: &str) {
        // Exhaustive `tw = 0..=256, 300` parity for `merge_bgrx_alpha_row`
        // (varying & constant alpha, aligned STREAM=true & unaligned STREAM=false)
        // and `merge_bgrx_alpha_row_in_place` against `ScalarKernel`.
        let merge_widths: Vec<usize> = if cfg!(miri) {
            (0..=33)
                .chain([63, 64, 65, 127, 128, 129, 255, 256, 300])
                .collect()
        } else {
            (0..=256).chain([300]).collect()
        };
        let mut merge_dst_bytes = vec![0xAAu8; (300 + 17) * 4];
        let merge_aligned_off = (64 - (merge_dst_bytes.as_ptr() as usize & 63)) & 63;
        for tw in merge_widths {
            let bgrx_src: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 19 + tw * 7) as u32;
                    Vec4u8(
                        ((k * k + 13) & 0xFF) as u8,
                        ((k * 31 + 5) & 0xFF) as u8,
                        ((k * 17 + 201) & 0xFF) as u8,
                        ((i * 13 + 7) & 0xFF) as u8,
                    )
                })
                .collect();
            let merge_alpha: Vec<u8> = (0..tw).map(|i| ((i * 37 + tw * 5) & 0xFF) as u8).collect();
            let mut ref_var = vec![Vec4u8(0, 0, 0, 0); tw];
            let mut ref_const = vec![Vec4u8(0, 0, 0, 0); tw];
            unsafe {
                ScalarKernel.merge_bgrx_alpha_row::<false, false>(
                    &bgrx_src,
                    &merge_alpha,
                    0,
                    slice::from_raw_parts_mut(ref_var.as_mut_ptr().cast(), tw),
                );
                ScalarKernel.merge_bgrx_alpha_row::<false, true>(
                    &bgrx_src,
                    &[],
                    0x7E,
                    slice::from_raw_parts_mut(ref_const.as_mut_ptr().cast(), tw),
                );
            }
            for (stream, byte_off) in [(true, merge_aligned_off), (false, merge_aligned_off + 4)] {
                let dst: &mut [Vec4u8] =
                    bytemuck::cast_slice_mut(&mut merge_dst_bytes[byte_off..byte_off + tw * 4]);
                dst.fill(Vec4u8(0, 0, 0, 0));
                let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                unsafe {
                    if stream && K::STREAM_ALIGN > 1 {
                        kernel.merge_bgrx_alpha_row::<true, false>(
                            &bgrx_src,
                            &merge_alpha,
                            0,
                            dst_uninit,
                        );
                    } else {
                        kernel.merge_bgrx_alpha_row::<false, false>(
                            &bgrx_src,
                            &merge_alpha,
                            0,
                            dst_uninit,
                        );
                    }
                }
                kernel.fence_stream();
                assert_eq!(
                    dst,
                    &ref_var[..],
                    "[{tier_name}] merge_bgrx_alpha_row varying tw={tw} stream={stream}"
                );

                dst.fill(Vec4u8(0, 0, 0, 0));
                let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                unsafe {
                    if stream && K::STREAM_ALIGN > 1 {
                        kernel.merge_bgrx_alpha_row::<true, true>(&bgrx_src, &[], 0x7E, dst_uninit);
                    } else {
                        kernel.merge_bgrx_alpha_row::<false, true>(
                            &bgrx_src,
                            &[],
                            0x7E,
                            dst_uninit,
                        );
                    }
                }
                kernel.fence_stream();
                assert_eq!(
                    dst,
                    &ref_const[..],
                    "[{tier_name}] merge_bgrx_alpha_row const tw={tw} stream={stream}"
                );

                dst.copy_from_slice(&bgrx_src);
                kernel.merge_bgrx_alpha_row_in_place::<false>(dst, &merge_alpha, 0);
                assert_eq!(
                    dst,
                    &ref_var[..],
                    "[{tier_name}] merge_bgrx_alpha_row_in_place varying tw={tw}"
                );

                dst.copy_from_slice(&bgrx_src);
                kernel.merge_bgrx_alpha_row_in_place::<true>(dst, &[], 0x7E);
                assert_eq!(
                    dst,
                    &ref_const[..],
                    "[{tier_name}] merge_bgrx_alpha_row_in_place const tw={tw}"
                );
            }
        }

        let widths: Vec<usize> = (1..=65).chain([127, 128, 129, 255, 256, 300]).collect();
        for tw in widths {
            let row_b = tw * 3;
            let raw0: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 19 + tw * 7) as u32;
                    Vec4u8(
                        ((k * k + 13) & 0xFF) as u8,
                        ((k * 31 + 5) & 0xFF) as u8,
                        ((k * 17 + 201) & 0xFF) as u8,
                        ((k * 23 + 3) & 0xFF) as u8,
                    )
                })
                .collect();
            let raw1: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 29 + tw * 11 + 3) as u32;
                    Vec4u8(
                        ((k * 13 + 7) & 0xFF) as u8,
                        ((k * k * 3 + 128) & 0xFF) as u8,
                        ((k * 41 + 99) & 0xFF) as u8,
                        ((k * 7 + 128) & 0xFF) as u8,
                    )
                })
                .collect();

            let p0 = raw0.as_ptr();
            let p1 = raw1.as_ptr();
            let bp0 = unsafe { BufferPointer::new(&p0, tw) };
            let bp1 = unsafe { BufferPointer::new(&p1, tw) };

            let (mut bgr0_s, mut bgr0_a) = (vec![0u8; row_b], vec![0u8; row_b]);
            let (mut bgr1_s, mut bgr1_a) = (vec![0u8; row_b], vec![0u8; row_b]);
            let (mut alp0_s, mut alp0_a) = (vec![0u8; tw], vec![0u8; tw]);
            let (mut alp1_s, mut alp1_a) = (vec![0u8; tw], vec![0u8; tw]);

            ScalarKernel.bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_s, &mut alp0_s);
            kernel.bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_a, &mut alp0_a);
            assert_eq!(
                bgr0_a, bgr0_s,
                "[{tier_name}] bgra_to_subgreen_bgr3<true> tw={tw}"
            );
            assert_eq!(alp0_a, alp0_s, "[{tier_name}] alpha extract tw={tw}");

            let mut bgr0_noalpha = vec![0u8; row_b];
            kernel.bgra_to_subgreen_bgr3::<false>(bp0, &mut bgr0_noalpha, &mut []);
            assert_eq!(
                bgr0_noalpha, bgr0_s,
                "[{tier_name}] bgra_to_subgreen_bgr3<false> tw={tw}"
            );

            ScalarKernel.bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_s, &mut alp1_s);
            kernel.bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_a, &mut alp1_a);
            assert_eq!(bgr1_a, bgr1_s);

            // Fused row-0 (`None`/`Sub`) and 4-predictor SADs + forward/inverse.
            let (mut f_s, mut f_a) = (vec![0u8; row_b], vec![0u8; row_b]);
            assert_eq!(
                kernel.row_sub3_forward_and_sads(&bgr0_a, &mut f_a),
                ScalarKernel.row_sub3_forward_and_sads(&bgr0_s, &mut f_s),
                "[{tier_name}] row_sub3_forward_and_sads tw={tw}"
            );
            assert_eq!(f_a, f_s, "[{tier_name}] sub3 fwd_and_sads out tw={tw}");

            let (mut up_s, mut up_a) = (vec![0xBBu8; row_b], vec![0xBBu8; row_b]);
            assert_eq!(
                kernel.row_up_forward_and_sads_all4(&bgr1_a, &bgr0_a, &mut up_a),
                ScalarKernel.row_up_forward_and_sads_all4(&bgr1_s, &bgr0_s, &mut up_s),
                "[{tier_name}] row_up_forward_and_sads_all4 tw={tw}"
            );
            assert_eq!(
                up_a, up_s,
                "[{tier_name}] row_up_forward_and_sads_all4 out tw={tw}"
            );

            // Sub3 forward + inverse
            f_a.fill(0xAA);
            f_s.fill(0xAA);
            kernel.row_sub3_forward(&bgr0_a, &mut f_a);
            ScalarKernel.row_sub3_forward(&bgr0_s, &mut f_s);
            assert_eq!(f_a, f_s, "[{tier_name}] sub3 fwd tw={tw}");
            let mut inv = vec![0xAAu8; row_b];
            kernel.row_sub3_inverse(&f_a, &mut inv);
            assert_eq!(inv, bgr0_s, "[{tier_name}] sub3 inv tw={tw}");

            // Up forward + inverse
            f_a.fill(0xAA);
            f_s.fill(0xAA);
            kernel.row_up_forward(&bgr1_a, &bgr0_a, &mut f_a);
            ScalarKernel.row_up_forward(&bgr1_s, &bgr0_s, &mut f_s);
            assert_eq!(f_a, f_s, "[{tier_name}] up fwd tw={tw}");
            inv.fill(0xAA);
            kernel.row_up_inverse(&f_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "[{tier_name}] up inv tw={tw}");

            // Sub3 in-place (on `up` -> produces `SubUp3`) + inverse
            let mut su_a = up_a.clone();
            let mut su_s = up_s.clone();
            kernel.row_sub3_forward_in_place(&mut su_a);
            ScalarKernel.row_sub3_forward_in_place(&mut su_s);
            assert_eq!(su_a, su_s, "[{tier_name}] sub3 in-place tw={tw}");
            inv.fill(0xAA);
            kernel.row_subup3_inverse(&su_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "[{tier_name}] subup3 inv tw={tw}");

            // End-to-end `select_and_filter_row` parity for row 0 and row > 0.
            let (mut sel_s, mut sel_a) = (vec![0xAAu8; row_b], vec![0xAAu8; row_b]);
            let p0_s = select_and_filter_row(ScalarKernel, 0, &bgr0_s, &[], &mut sel_s);
            let p0_a = select_and_filter_row(kernel, 0, &bgr0_a, &[], &mut sel_a);
            assert_eq!(
                p0_a, p0_s,
                "[{tier_name}] select_and_filter row 0 p tw={tw}"
            );
            assert_eq!(
                sel_a, sel_s,
                "[{tier_name}] select_and_filter row 0 out tw={tw}"
            );

            let p1_s = select_and_filter_row(ScalarKernel, 1, &bgr1_s, &bgr0_s, &mut sel_s);
            let p1_a = select_and_filter_row(kernel, 1, &bgr1_a, &bgr0_a, &mut sel_a);
            assert_eq!(
                p1_a, p1_s,
                "[{tier_name}] select_and_filter row 1 p tw={tw}"
            );
            assert_eq!(
                sel_a, sel_s,
                "[{tier_name}] select_and_filter row 1 out tw={tw}"
            );

            // Sub1 & Up on alpha plane
            let (mut af_s, mut af_a) = (vec![0u8; tw], vec![0u8; tw]);
            ScalarKernel.row_sub1_forward(&alp0_s, &mut af_s);
            kernel.row_sub1_forward(&alp0_a, &mut af_a);
            assert_eq!(af_a, af_s, "[{tier_name}] sub1 fwd tw={tw}");
            let mut a_inv = vec![0xAAu8; tw];
            kernel.row_sub1_inverse(&af_a, &mut a_inv);
            assert_eq!(a_inv, alp0_s, "[{tier_name}] sub1 inv tw={tw}");

            ScalarKernel.row_up_forward(&alp1_s, &alp0_s, &mut af_s);
            kernel.row_up_forward(&alp1_a, &alp0_a, &mut af_a);
            assert_eq!(af_a, af_s, "[{tier_name}] alpha up fwd tw={tw}");
            kernel.row_up_inverse(&af_a, &alp0_a, &mut a_inv);
            assert_eq!(a_inv, alp1_s, "[{tier_name}] alpha up inv tw={tw}");

            // Unpack BGR3 -> BGRA (constant alpha and varying alpha, both alignments)
            let mut dst_bytes = vec![0xAAu8; (tw + 17) * 4];
            let aligned_off = (64 - (dst_bytes.as_ptr() as usize & 63)) & 63;
            for (stream, byte_off) in [(true, aligned_off), (false, aligned_off + 4)] {
                let dst: &mut [Vec4u8] =
                    bytemuck::cast_slice_mut(&mut dst_bytes[byte_off..byte_off + tw * 4]);
                dst.fill(Vec4u8(0, 0, 0, 0));
                // SAFETY: `MaybeUninit<Vec4u8>` is `repr(transparent)` over `Vec4u8`.
                let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
                unsafe {
                    if stream && K::STREAM_ALIGN > 1 {
                        kernel
                            .subgreen_bgr3_to_bgra::<true, false>(&bgr0_a, &alp0_a, 0, dst_uninit);
                    } else {
                        kernel
                            .subgreen_bgr3_to_bgra::<false, false>(&bgr0_a, &alp0_a, 0, dst_uninit);
                    }
                }
                assert_eq!(
                    dst,
                    &raw0[..],
                    "[{tier_name}] unpack varying alpha tw={tw} stream={stream}"
                );

                dst.fill(Vec4u8(0, 0, 0, 0));
                // SAFETY: `MaybeUninit<Vec4u8>` is `repr(transparent)` over `Vec4u8`.
                let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
                unsafe {
                    if stream && K::STREAM_ALIGN > 1 {
                        kernel.subgreen_bgr3_to_bgra::<true, true>(&bgr0_a, &[], 0x7E, dst_uninit);
                    } else {
                        kernel.subgreen_bgr3_to_bgra::<false, true>(&bgr0_a, &[], 0x7E, dst_uninit);
                    }
                }
                for (got, exp) in dst.iter().zip(&raw0) {
                    assert_eq!(*got, Vec4u8(exp.0, exp.1, exp.2, 0x7E));
                }

                // Splat row
                let px = Vec4u8(0x11, 0x22, 0x33, 0x44);
                // SAFETY: `MaybeUninit<Vec4u8>` is `repr(transparent)` over `Vec4u8`.
                let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
                unsafe {
                    if stream && K::STREAM_ALIGN > 1 {
                        kernel.splat_row::<true>(dst_uninit, px);
                    } else {
                        kernel.splat_row::<false>(dst_uninit, px);
                    }
                }
                assert!(dst.iter().all(|&p| p == px));

                // Streaming row copy (`copy_row_stream`) on aligned 16-pixel-multiple rows.
                if stream && tw.is_multiple_of(16) {
                    dst.fill(Vec4u8(0, 0, 0, 0));
                    // SAFETY: `MaybeUninit<Vec4u8>` is `repr(transparent)` over `Vec4u8`.
                    let dst_uninit: &mut [MaybeUninit<Vec4u8>] =
                        unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
                    unsafe {
                        kernel.copy_row_stream(&raw0, dst_uninit);
                    }
                    kernel.fence_stream();
                    assert_eq!(dst, &raw0[..], "[{tier_name}] copy_row_stream tw={tw}");
                }
            }
            kernel.fence_stream();

            // `is_constant_plane` on constant + single perturbed byte at start/mid/end.
            let mut plane = vec![0x5Au8; tw];
            assert_eq!(kernel.is_constant_plane(&plane), Some(0x5A));
            plane[tw - 1] ^= 1;
            assert_eq!(
                kernel.is_constant_plane(&plane),
                if tw == 1 { Some(0x5B) } else { None }
            );
            if tw > 1 {
                plane[tw - 1] = 0x5A;
                plane[tw / 2] ^= 1;
                assert_eq!(kernel.is_constant_plane(&plane), None);
            }

            // `diff_and_update_row` matches `ScalarKernel` (including exact
            // tight pixel span) on identical, alpha-only, single-pixel, and
            // multi-pixel RGB-perturbed inputs across SIMD lane boundaries.
            let mut sh_s = raw0.clone();
            let mut sh_a = raw0.clone();
            assert_eq!(
                ScalarKernel.diff_and_update_row::<true>(bp0, &mut sh_s),
                None
            );
            assert_eq!(kernel.diff_and_update_row::<true>(bp0, &mut sh_a), None);
            assert_eq!(kernel.diff_and_update_row::<false>(bp0, &mut sh_a), None);

            for pos in [0, tw / 2, tw - 1] {
                let mut alt_a = raw0.clone();
                alt_a[pos].3 ^= 0x55;
                let pa = alt_a.as_ptr();
                let bpa = unsafe { BufferPointer::new(&pa, tw) };
                sh_s.clone_from(&raw0);
                sh_a.clone_from(&raw0);
                assert_eq!(
                    ScalarKernel.diff_and_update_row::<false>(bpa, &mut sh_s),
                    None
                );
                assert_eq!(kernel.diff_and_update_row::<false>(bpa, &mut sh_a), None);
                assert_eq!(
                    sh_a, raw0,
                    "[{tier_name}] xrgb alpha-only must not write tw={tw} pos={pos}"
                );
                assert_eq!(
                    ScalarKernel.diff_and_update_row::<true>(bpa, &mut sh_s),
                    Some((pos, pos + 1))
                );
                assert_eq!(
                    kernel.diff_and_update_row::<true>(bpa, &mut sh_a),
                    Some((pos, pos + 1)),
                    "[{tier_name}] argb alpha span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_a, alt_a,
                    "[{tier_name}] argb alpha diff tw={tw} pos={pos}"
                );
                assert_eq!(sh_a, sh_s);

                let mut alt_rgb = raw0.clone();
                alt_rgb[pos].1 ^= 0x33;
                alt_rgb[tw - 1].2 ^= 0x11;
                let pr = alt_rgb.as_ptr();
                let bpr = unsafe { BufferPointer::new(&pr, tw) };
                sh_s.clone_from(&raw0);
                sh_a.clone_from(&raw0);
                let exp_span = ScalarKernel.diff_and_update_row::<false>(bpr, &mut sh_s);
                let got_span = kernel.diff_and_update_row::<false>(bpr, &mut sh_a);
                assert_eq!(exp_span, Some((pos, tw)));
                assert_eq!(
                    got_span, exp_span,
                    "[{tier_name}] xrgb rgb span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_a, alt_rgb,
                    "[{tier_name}] xrgb rgb diff tw={tw} pos={pos}"
                );
                assert_eq!(sh_a, sh_s);
            }
            if tw >= 20 {
                // Interior scattered diffs crossing 4-, 8-, and 16-pixel SIMD
                // boundaries with unchanged prefix and suffix pixels.
                for (p0, p1) in [(1, 1), (3, 17), (5, tw - 4), (15, 16)] {
                    let mut alt = raw0.clone();
                    alt[p0].0 ^= 0x21;
                    alt[p1].2 ^= 0x42;
                    let pr = alt.as_ptr();
                    let bpr = unsafe { BufferPointer::new(&pr, tw) };
                    sh_s.clone_from(&raw0);
                    sh_a.clone_from(&raw0);
                    let exp_span = ScalarKernel.diff_and_update_row::<false>(bpr, &mut sh_s);
                    let got_span = kernel.diff_and_update_row::<false>(bpr, &mut sh_a);
                    assert_eq!(exp_span, Some((p0, p1 + 1)));
                    assert_eq!(
                        got_span, exp_span,
                        "[{tier_name}] interior span tw={tw} ({p0}..={p1})"
                    );
                    assert_eq!(sh_a, sh_s);
                }
            }
        }

        // Exhaustive `diff_and_update_row::<false/true>` sweep across every
        // width `0..=256` and `300` (compact subset under Miri), covering clean
        // rows, single-pixel RGB and alpha-only diffs across all 1/4/8/16/32/64
        // vector boundaries, and multi-chunk dirty spans.
        #[cfg(not(miri))]
        let diff_widths: Vec<usize> = (0..=256).chain([300]).collect();
        #[cfg(miri)]
        let diff_widths: Vec<usize> = vec![
            0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 300,
        ];
        for tw in diff_widths {
            let base: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 37 + tw * 13 + 5) as u32;
                    Vec4u8(
                        ((k * 11 + 3) & 0xFF) as u8,
                        ((k * 23 + 17) & 0xFF) as u8,
                        ((k * 41 + 9) & 0xFF) as u8,
                        ((k * 19 + 7) & 0xFF) as u8,
                    )
                })
                .collect();
            let p_base = base.as_ptr();
            let bp_base = unsafe { BufferPointer::new(&p_base, tw) };

            let mut sh_s = base.clone();
            let mut sh_k = base.clone();
            assert_eq!(
                kernel.diff_and_update_row::<false>(bp_base, &mut sh_k),
                None,
                "[{tier_name}] clean xrgb tw={tw}"
            );
            assert_eq!(
                kernel.diff_and_update_row::<true>(bp_base, &mut sh_k),
                None,
                "[{tier_name}] clean argb tw={tw}"
            );
            assert_eq!(sh_k, base, "[{tier_name}] clean mutated shadow tw={tw}");
            if tw == 0 {
                continue;
            }

            let probe_positions = [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, tw - 1];
            for &pos in &probe_positions {
                if pos >= tw {
                    continue;
                }
                // Alpha-only diff at `pos`.
                let mut alt_a = base.clone();
                alt_a[pos].3 ^= 0xA5;
                let pa = alt_a.as_ptr();
                let bpa = unsafe { BufferPointer::new(&pa, tw) };
                sh_s.clone_from(&base);
                sh_k.clone_from(&base);
                assert_eq!(
                    kernel.diff_and_update_row::<false>(bpa, &mut sh_k),
                    ScalarKernel.diff_and_update_row::<false>(bpa, &mut sh_s),
                    "[{tier_name}] alpha-only xrgb span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_k, base,
                    "[{tier_name}] alpha-only xrgb must not write shadow tw={tw} pos={pos}"
                );
                assert_eq!(
                    kernel.diff_and_update_row::<true>(bpa, &mut sh_k),
                    ScalarKernel.diff_and_update_row::<true>(bpa, &mut sh_s),
                    "[{tier_name}] alpha-only argb span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_k, sh_s,
                    "[{tier_name}] alpha-only argb shadow tw={tw} pos={pos}"
                );

                // Single-pixel RGB diff at `pos`.
                let mut alt_rgb = base.clone();
                alt_rgb[pos].0 ^= 0x5A;
                let pr = alt_rgb.as_ptr();
                let bpr = unsafe { BufferPointer::new(&pr, tw) };
                sh_s.clone_from(&base);
                sh_k.clone_from(&base);
                assert_eq!(
                    kernel.diff_and_update_row::<false>(bpr, &mut sh_k),
                    ScalarKernel.diff_and_update_row::<false>(bpr, &mut sh_s),
                    "[{tier_name}] single-px xrgb span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_k, sh_s,
                    "[{tier_name}] single-px xrgb shadow tw={tw} pos={pos}"
                );
                sh_s.clone_from(&base);
                sh_k.clone_from(&base);
                assert_eq!(
                    kernel.diff_and_update_row::<true>(bpr, &mut sh_k),
                    ScalarKernel.diff_and_update_row::<true>(bpr, &mut sh_s),
                    "[{tier_name}] single-px argb span tw={tw} pos={pos}"
                );
                assert_eq!(
                    sh_k, sh_s,
                    "[{tier_name}] single-px argb shadow tw={tw} pos={pos}"
                );
            }

            // Multi-chunk dirty spans crossing 16/32/64/128/256-px boundaries,
            // including an unrelated alpha-only diff in a clean outer chunk
            // under `HAS_ALPHA = false`.
            let span_pairs = [
                (0, tw - 1),
                (1, tw.saturating_sub(2)),
                (7, 33.min(tw - 1)),
                (15, 65.min(tw - 1)),
                (31, 129.min(tw - 1)),
                (63, 255.min(tw - 1)),
            ];
            for (p0, p1) in span_pairs {
                if p0 > p1 || p1 >= tw {
                    continue;
                }
                let mut alt = base.clone();
                alt[p0].1 ^= 0x3C;
                alt[p1].2 ^= 0xC3;
                if p0 >= 16 {
                    // Alpha-only perturbation in an earlier clean 16-px chunk
                    // must remain ignored under `HAS_ALPHA = false`.
                    alt[0].3 ^= 0x77;
                }
                let pa = alt.as_ptr();
                let bpa = unsafe { BufferPointer::new(&pa, tw) };
                sh_s.clone_from(&base);
                sh_k.clone_from(&base);
                let exp_xrgb = ScalarKernel.diff_and_update_row::<false>(bpa, &mut sh_s);
                let got_xrgb = kernel.diff_and_update_row::<false>(bpa, &mut sh_k);
                assert_eq!(
                    got_xrgb, exp_xrgb,
                    "[{tier_name}] multi-chunk xrgb span tw={tw} ({p0}..={p1})"
                );
                for j in 0..tw {
                    assert_eq!(
                        (sh_k[j].0, sh_k[j].1, sh_k[j].2),
                        (sh_s[j].0, sh_s[j].1, sh_s[j].2),
                        "[{tier_name}] multi-chunk xrgb rgb tw={tw} ({p0}..={p1}) px={j}"
                    );
                }

                sh_s.clone_from(&base);
                sh_k.clone_from(&base);
                let exp_argb = ScalarKernel.diff_and_update_row::<true>(bpa, &mut sh_s);
                let got_argb = kernel.diff_and_update_row::<true>(bpa, &mut sh_k);
                assert_eq!(
                    got_argb, exp_argb,
                    "[{tier_name}] multi-chunk argb span tw={tw} ({p0}..={p1})"
                );
                assert_eq!(
                    sh_k, sh_s,
                    "[{tier_name}] multi-chunk argb shadow tw={tw} ({p0}..={p1})"
                );
            }
        }

        // Preserve `copy_and_scan_row` and palette checks for the original width set.
        for tw in (1..=65).chain([127, 128, 129, 255, 256, 300]) {
            // `copy_and_scan_row` parity with `ScalarKernel` across solid,
            // garbage-X, translucent, and single-pixel diff inputs.
            let first_opq = u32::from_ne_bytes([0x12, 0x34, 0x56, 0xFF]);
            let first_tr = u32::from_ne_bytes([0x12, 0x34, 0x56, 0x80]);
            let mut row_xrgb = vec![0u8; tw * 4];
            for (i, px) in row_xrgb.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *px = [0x12, 0x34, 0x56, (i.wrapping_mul(29) & 0xFF) as u8];
            }
            let (mut dst_s, mut dst_k) = (vec![0xAAu8; tw * 4], vec![0xAAu8; tw * 4]);
            let r_s = unsafe {
                ScalarKernel.copy_and_scan_row::<false>(
                    row_xrgb.as_ptr(),
                    dst_s.as_mut_ptr(),
                    tw,
                    first_opq,
                )
            };
            let r_k = unsafe {
                kernel.copy_and_scan_row::<false>(
                    row_xrgb.as_ptr(),
                    dst_k.as_mut_ptr(),
                    tw,
                    first_opq,
                )
            };
            assert_eq!(r_s, (true, true));
            assert_eq!(
                r_k, r_s,
                "[{tier_name}] copy_and_scan_row xrgb solid tw={tw}"
            );
            assert_eq!(
                dst_k, row_xrgb,
                "[{tier_name}] copy_and_scan_row raw X bytes tw={tw}"
            );

            let mut row_argb = vec![0u8; tw * 4];
            for px in row_argb.as_chunks_mut::<4>().0 {
                *px = [0x12, 0x34, 0x56, 0xFF];
            }
            for pos in [0, tw / 2, tw - 1] {
                let mut alt = row_argb.clone();
                alt[pos * 4 + 3] = 0xFE;
                let r_s = unsafe {
                    ScalarKernel.copy_and_scan_row::<true>(
                        alt.as_ptr(),
                        dst_s.as_mut_ptr(),
                        tw,
                        first_opq,
                    )
                };
                let r_k = unsafe {
                    kernel.copy_and_scan_row::<true>(
                        alt.as_ptr(),
                        dst_k.as_mut_ptr(),
                        tw,
                        first_opq,
                    )
                };
                assert_eq!(r_s, (false, false));
                assert_eq!(
                    r_k, r_s,
                    "[{tier_name}] copy_and_scan_row argb fe tw={tw} pos={pos}"
                );
                assert_eq!(dst_k, alt);

                alt[pos * 4 + 3] = 0xFF;
                alt[pos * 4] ^= 0x01;
                let r_s = unsafe {
                    ScalarKernel.copy_and_scan_row::<true>(
                        alt.as_ptr(),
                        dst_s.as_mut_ptr(),
                        tw,
                        first_opq,
                    )
                };
                let r_k = unsafe {
                    kernel.copy_and_scan_row::<true>(
                        alt.as_ptr(),
                        dst_k.as_mut_ptr(),
                        tw,
                        first_opq,
                    )
                };
                assert_eq!(r_s, (false, true));
                assert_eq!(
                    r_k, r_s,
                    "[{tier_name}] copy_and_scan_row argb rgb diff tw={tw} pos={pos}"
                );
                assert_eq!(dst_k, alt);
            }
            let mut row_tr = vec![0u8; tw * 4];
            for px in row_tr.as_chunks_mut::<4>().0 {
                *px = [0x12, 0x34, 0x56, 0x80];
            }
            let r_s = unsafe {
                ScalarKernel.copy_and_scan_row::<true>(
                    row_tr.as_ptr(),
                    dst_s.as_mut_ptr(),
                    tw,
                    first_tr,
                )
            };
            let r_k = unsafe {
                kernel.copy_and_scan_row::<true>(row_tr.as_ptr(), dst_k.as_mut_ptr(), tw, first_tr)
            };
            assert_eq!(r_s, (true, false));
            assert_eq!(
                r_k, r_s,
                "[{tier_name}] copy_and_scan_row argb solid tr tw={tw}"
            );
            assert_eq!(dst_k, row_tr);
        }
        // Exhaustive width sweep (`0..=256` and `300`, covering every SIMD
        // main-loop, 48B/24B/12B/8px/4px tail, and scalar-tail boundary) for
        // `bgra_to_subgreen_bgr3`, the fused 4-predictor selector/filter,
        // `row_sub3_inverse`, `row_subup3_inverse`, `subgreen_bgr3_to_bgra`,
        // and fused `row_up_subgreen_bgr3_to_bgra` (all `STREAM`/`CONST_ALPHA`/
        // `WRITE_DEC` combinations + aligned/unaligned `dst`).
        let mut sweep_dst_bytes = vec![0xAAu8; (300 + 17) * 4];
        let sweep_aligned_off = (64 - (sweep_dst_bytes.as_ptr() as usize & 63)) & 63;
        for tw in (0..=256usize).chain([300]) {
            let row_b = tw * 3;
            let raw0: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 37 + tw * 13 + 1) as u32;
                    Vec4u8(
                        ((k * 19 + 3) & 0xFF) as u8,
                        ((k * 11 + 128) & 0xFF) as u8,
                        ((k * 23 + 77) & 0xFF) as u8,
                        ((k * 7 + 19) & 0xFF) as u8,
                    )
                })
                .collect();
            let raw1: Vec<Vec4u8> = (0..tw)
                .map(|i| {
                    let k = (i * 41 + tw * 5 + 9) as u32;
                    Vec4u8(
                        ((k * 13 + 128) & 0xFF) as u8,
                        ((k * 29 + 7) & 0xFF) as u8,
                        ((k * 17 + 211) & 0xFF) as u8,
                        ((k * 31 + 55) & 0xFF) as u8,
                    )
                })
                .collect();
            let p0 = raw0.as_ptr();
            let p1 = raw1.as_ptr();
            let bp0 = unsafe { BufferPointer::new(&p0, tw) };
            let bp1 = unsafe { BufferPointer::new(&p1, tw) };

            let (mut bgr0_s, mut bgr0_a) = (vec![0xCCu8; row_b], vec![0xCCu8; row_b]);
            let (mut bgr1_s, mut bgr1_a) = (vec![0xCCu8; row_b], vec![0xCCu8; row_b]);
            let (mut alp0_s, mut alp0_a) = (vec![0xCCu8; tw], vec![0xCCu8; tw]);
            let (mut alp1_s, mut alp1_a) = (vec![0xCCu8; tw], vec![0xCCu8; tw]);
            ScalarKernel.bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_s, &mut alp0_s);
            kernel.bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_a, &mut alp0_a);
            assert_eq!(bgr0_a, bgr0_s, "[{tier_name}] sweep subgreen<true> tw={tw}");
            assert_eq!(alp0_a, alp0_s, "[{tier_name}] sweep alpha tw={tw}");

            let mut bgr0_noa = vec![0xCCu8; row_b];
            kernel.bgra_to_subgreen_bgr3::<false>(bp0, &mut bgr0_noa, &mut []);
            assert_eq!(
                bgr0_noa, bgr0_s,
                "[{tier_name}] sweep subgreen<false> tw={tw}"
            );

            ScalarKernel.bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_s, &mut alp1_s);
            kernel.bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_a, &mut alp1_a);
            assert_eq!(bgr1_a, bgr1_s, "[{tier_name}] sweep subgreen row1 tw={tw}");
            assert_eq!(alp1_a, alp1_s, "[{tier_name}] sweep alpha row1 tw={tw}");

            let (mut up_s, mut up_a) = (vec![0xBBu8; row_b], vec![0xBBu8; row_b]);
            assert_eq!(
                kernel.row_up_forward_and_sads_all4(&bgr1_a, &bgr0_a, &mut up_a),
                ScalarKernel.row_up_forward_and_sads_all4(&bgr1_s, &bgr0_s, &mut up_s),
                "[{tier_name}] sweep row_up_forward_and_sads_all4 tw={tw}"
            );
            assert_eq!(up_a, up_s, "[{tier_name}] sweep up_out tw={tw}");

            // Inverse `Sub3` and `SubUp3` parity across every width `0..=256, 300`.
            let (mut sub_f, mut inv_a) = (vec![0xAAu8; row_b], vec![0x55u8; row_b]);
            ScalarKernel.row_sub3_forward(&bgr0_s, &mut sub_f);
            kernel.row_sub3_inverse(&sub_f, &mut inv_a);
            assert_eq!(
                inv_a, bgr0_s,
                "[{tier_name}] sweep row_sub3_inverse tw={tw}"
            );

            let (mut ip_s, mut ip_a) = (up_s.clone(), up_a.clone());
            kernel.row_sub3_forward_in_place(&mut ip_a);
            ScalarKernel.row_sub3_forward_in_place(&mut ip_s);
            assert_eq!(
                ip_a, ip_s,
                "[{tier_name}] sweep row_sub3_forward_in_place tw={tw}"
            );
            inv_a.fill(0x55);
            kernel.row_subup3_inverse(&ip_a, &bgr0_s, &mut inv_a);
            assert_eq!(
                inv_a, bgr1_s,
                "[{tier_name}] sweep row_subup3_inverse tw={tw}"
            );

            // `subgreen_bgr3_to_bgra` and fused `row_up_subgreen_bgr3_to_bgra`
            // across all `STREAM`/`CONST_ALPHA`/`WRITE_DEC` combinations and
            // both aligned and unaligned destinations.
            let mut dec_buf = vec![0x77u8; row_b];
            for (stream, byte_off) in [(true, sweep_aligned_off), (false, sweep_aligned_off + 4)] {
                let dst: &mut [Vec4u8] =
                    bytemuck::cast_slice_mut(&mut sweep_dst_bytes[byte_off..byte_off + tw * 4]);
                let do_stream = stream && K::STREAM_ALIGN > 1;

                // 1. `subgreen_bgr3_to_bgra` varying alpha & const alpha
                dst.fill(Vec4u8(0, 0, 0, 0));
                let dst_u: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                unsafe {
                    if do_stream {
                        kernel.subgreen_bgr3_to_bgra::<true, false>(&bgr1_s, &alp1_s, 0, dst_u);
                    } else {
                        kernel.subgreen_bgr3_to_bgra::<false, false>(&bgr1_s, &alp1_s, 0, dst_u);
                    }
                }
                assert_eq!(
                    dst,
                    &raw1[..],
                    "[{tier_name}] sweep subgreen_bgr3_to_bgra var tw={tw} stream={stream}"
                );

                dst.fill(Vec4u8(0, 0, 0, 0));
                let dst_u: &mut [MaybeUninit<Vec4u8>] =
                    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                unsafe {
                    if do_stream {
                        kernel.subgreen_bgr3_to_bgra::<true, true>(&bgr1_s, &[], 0xC3, dst_u);
                    } else {
                        kernel.subgreen_bgr3_to_bgra::<false, true>(&bgr1_s, &[], 0xC3, dst_u);
                    }
                }
                for (got, exp) in dst.iter().zip(&raw1) {
                    assert_eq!(
                        *got,
                        Vec4u8(exp.0, exp.1, exp.2, 0xC3),
                        "[{tier_name}] sweep subgreen_bgr3_to_bgra const tw={tw} stream={stream}"
                    );
                }

                // 2. Fused `row_up_subgreen_bgr3_to_bgra` (`WRITE_DEC = true/false`,
                //    `CONST_ALPHA = true/false`, `STREAM = true/false`).
                for write_dec in [true, false] {
                    dec_buf.fill(0x77);
                    dst.fill(Vec4u8(0, 0, 0, 0));
                    let dst_u: &mut [MaybeUninit<Vec4u8>] =
                        unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                    unsafe {
                        match (do_stream, write_dec) {
                            (true, true) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<true, false, true>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut dec_buf,
                                    &alp1_s,
                                    0,
                                    dst_u,
                                ),
                            (true, false) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<true, false, false>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut [],
                                    &alp1_s,
                                    0,
                                    dst_u,
                                ),
                            (false, true) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<false, false, true>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut dec_buf,
                                    &alp1_s,
                                    0,
                                    dst_u,
                                ),
                            (false, false) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<false, false, false>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut [],
                                    &alp1_s,
                                    0,
                                    dst_u,
                                ),
                        }
                    }
                    assert_eq!(
                        dst,
                        &raw1[..],
                        "[{tier_name}] sweep fused up var tw={tw} stream={stream} wd={write_dec}"
                    );
                    if write_dec {
                        assert_eq!(
                            dec_buf, bgr1_s,
                            "[{tier_name}] sweep fused up var cur_dec tw={tw} stream={stream}"
                        );
                    } else {
                        assert!(
                            dec_buf.iter().all(|&b| b == 0x77),
                            "[{tier_name}] sweep fused up var touched cur_dec when wd=false tw={tw}"
                        );
                    }

                    dec_buf.fill(0x77);
                    dst.fill(Vec4u8(0, 0, 0, 0));
                    let dst_u: &mut [MaybeUninit<Vec4u8>] =
                        unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), tw) };
                    unsafe {
                        match (do_stream, write_dec) {
                            (true, true) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<true, true, true>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut dec_buf,
                                    &[],
                                    0x9D,
                                    dst_u,
                                ),
                            (true, false) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<true, true, false>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut [],
                                    &[],
                                    0x9D,
                                    dst_u,
                                ),
                            (false, true) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<false, true, true>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut dec_buf,
                                    &[],
                                    0x9D,
                                    dst_u,
                                ),
                            (false, false) => kernel
                                .row_up_subgreen_bgr3_to_bgra::<false, true, false>(
                                    &up_s,
                                    &bgr0_s,
                                    &mut [],
                                    &[],
                                    0x9D,
                                    dst_u,
                                ),
                        }
                    }
                    for (got, exp) in dst.iter().zip(&raw1) {
                        assert_eq!(
                            *got,
                            Vec4u8(exp.0, exp.1, exp.2, 0x9D),
                            "[{tier_name}] sweep fused up const tw={tw} stream={stream} wd={write_dec}"
                        );
                    }
                    if write_dec {
                        assert_eq!(
                            dec_buf, bgr1_s,
                            "[{tier_name}] sweep fused up const cur_dec tw={tw} stream={stream}"
                        );
                    }
                }
            }
            kernel.fence_stream();

            let (mut o_s, mut o_a) = (vec![0xAAu8; row_b], vec![0xAAu8; row_b]);
            let m0_s = select_and_filter_row(ScalarKernel, 0, &bgr0_s, &[], &mut o_s);
            let m0_a = select_and_filter_row(kernel, 0, &bgr0_a, &[], &mut o_a);
            assert_eq!(m0_a, m0_s, "[{tier_name}] sweep sel r0 mode tw={tw}");
            assert_eq!(o_a, o_s, "[{tier_name}] sweep sel r0 out tw={tw}");

            let m1_s = select_and_filter_row(ScalarKernel, 1, &bgr1_s, &bgr0_s, &mut o_s);
            let m1_a = select_and_filter_row(kernel, 1, &bgr1_a, &bgr0_a, &mut o_a);
            assert_eq!(m1_a, m1_s, "[{tier_name}] sweep sel r1 mode tw={tw}");
            assert_eq!(o_a, o_s, "[{tier_name}] sweep sel r1 out tw={tw}");

            // Force each predictor (`0=None`, `1=Sub`, `2=Up`, `3=SubUp`) and
            // exact ties at 0 (`zeros`) to verify strict `<` tie-breaking.
            let zeros = vec![0u8; row_b];
            let (mut z_s, mut z_a) = (vec![0xEEu8; row_b], vec![0xEEu8; row_b]);
            assert_eq!(select_and_filter_row(kernel, 0, &zeros, &[], &mut z_a), 0);
            assert_eq!(z_a, zeros);
            assert_eq!(
                select_and_filter_row(kernel, 1, &zeros, &zeros, &mut z_a),
                0
            );
            assert_eq!(z_a, zeros);

            // Constant non-zero row over zero prev -> Sub wins (`cost_sub = 3*v < cost_none = row_b*v`)
            // and equals SubUp (`Sub == SubUp`), so strict `<` must pick `1` (`Sub`).
            let const_row = vec![5u8; row_b];
            let p_cs = select_and_filter_row(ScalarKernel, 1, &const_row, &zeros, &mut z_s);
            let p_ca = select_and_filter_row(kernel, 1, &const_row, &zeros, &mut z_a);
            assert_eq!(p_ca, p_cs, "[{tier_name}] tie Sub==SubUp tw={tw}");
            assert_eq!(z_a, z_s, "[{tier_name}] tie Sub==SubUp out tw={tw}");

            // Identical cur == prev -> Up == SubUp == 0 < None/Sub, strict `<` picks `2` (`Up`).
            let p_us = select_and_filter_row(ScalarKernel, 1, &bgr0_s, &bgr0_s, &mut z_s);
            let p_ua = select_and_filter_row(kernel, 1, &bgr0_a, &bgr0_a, &mut z_a);
            assert_eq!(p_ua, p_us, "[{tier_name}] tie Up==SubUp==0 tw={tw}");
            assert_eq!(z_a, z_s, "[{tier_name}] tie Up==SubUp==0 out tw={tw}");

            // Linear ramp shifted by constant delta from prev -> `Up` is constant,
            // so `SubUp` wins (`3`) for `tw >= 2`.
            let ramp_prev: Vec<u8> = (0..row_b).map(|j| ((j * 7) & 0xFF) as u8).collect();
            let ramp_cur: Vec<u8> = ramp_prev.iter().map(|&b| b.wrapping_add(11)).collect();
            let p_sus = select_and_filter_row(ScalarKernel, 1, &ramp_cur, &ramp_prev, &mut z_s);
            let p_sua = select_and_filter_row(kernel, 1, &ramp_cur, &ramp_prev, &mut z_a);
            assert_eq!(p_sua, p_sus, "[{tier_name}] SubUp winner tw={tw}");
            assert_eq!(z_a, z_s, "[{tier_name}] SubUp winner out tw={tw}");
        }

        // Verify `row_sub3_inverse` on arbitrary (including non-multiple-of-3) byte lengths `0..=65`.
        for len in 0..=65usize {
            let src: Vec<u8> = (0..len)
                .map(|j| ((j * 37 + len * 13 + 7) & 0xFF) as u8)
                .collect();
            let mut dst_s = vec![0xAAu8; len];
            let mut dst_k = vec![0x55u8; len];
            ScalarKernel.row_sub3_inverse(&src, &mut dst_s);
            kernel.row_sub3_inverse(&src, &mut dst_k);
            assert_eq!(
                dst_k, dst_s,
                "[{tier_name}] row_sub3_inverse arbitrary len={len}"
            );
        }

        // `i8::MIN` (`0x80` = `-128`) unsigned abs must equal `128` across all backends
        // (covers 16B, 32B, and 64B SIMD loops plus scalar tails).
        let min_bytes = [0x80u8; 73];
        let zero_bytes = [0u8; 73];
        let mut min_dst = [0u8; 73];
        let (min_none, min_sub) = kernel.row_sub3_forward_and_sads(&min_bytes, &mut min_dst);
        assert_eq!(min_none, 73 * 128, "[{tier_name}] i8::MIN row0 none sad");
        assert_eq!(min_sub, 3 * 128, "[{tier_name}] i8::MIN row0 sub sad");
        assert_eq!(
            kernel.row_up_forward_and_sads_all4(&min_bytes, &zero_bytes, &mut min_dst),
            [73 * 128, 3 * 128, 73 * 128, 3 * 128],
            "[{tier_name}] i8::MIN all4 sads"
        );
        assert_eq!(min_dst, min_bytes, "[{tier_name}] i8::MIN up out");
        assert_eq!(kernel.is_constant_plane(&[]), None);

        // Palette SIMD validation, packing, and 1/2/4/8-bpp row decoding
        // (including streaming and unaligned `idx_pos` offsets) must match
        // `ScalarKernel` bit-for-bit.
        verify_palette_kernel_matches_scalar(kernel, tier_name);

        // Scroll detection 64B probes, strip scoring, and aligned non-temporal
        // byte copy must match `ScalarKernel` bit-for-bit across all alignments
        // and edge cases.
        verify_scroll_kernel_matches_scalar(kernel, tier_name);
    }

    #[inline(always)]
    fn verify_scroll_kernel_matches_scalar<K: RowKernel>(kernel: K, tier_name: &str) {
        // 1. 16-pixel (`64 B`) scroll strip activity, curr hash, prev hash, and
        //    `score_strip_row` parity across `HAS_ALPHA = false / true`, unaligned
        //    offsets, solid/uniform, alpha-only, single-pixel diff, and `ok`
        //    threshold boundaries (`11` vs `12`).
        let mut curr_raw = vec![0u8; 256];
        let mut prev_raw = vec![0u8; 256];
        let mut prev_dst_raw = vec![0u8; 256];
        let textured: [Vec4u8; 16] = std::array::from_fn(|i| {
            let k = (i * 37 + 13) as u8;
            Vec4u8(
                k.wrapping_mul(3),
                k.wrapping_add(71),
                k.wrapping_mul(11),
                0x80u8.wrapping_add((i * 7) as u8),
            )
        });
        let solid = [Vec4u8(0x2A, 0x54, 0x7E, 0xFF); 16];

        let check_probes =
            |curr_px: &[Vec4u8; 16], prev_px: &[Vec4u8; 16], off: usize, tag: &str| {
                let mut cbuf = [0u8; 128];
                let mut pbuf = [0u8; 128];
                cbuf[off..off + 64].copy_from_slice(bytemuck::bytes_of(curr_px));
                pbuf[off..off + 64].copy_from_slice(bytemuck::bytes_of(prev_px));
                let cp = cbuf.as_ptr().wrapping_add(off).cast::<Vec4u8>();
                let pp = pbuf.as_ptr().wrapping_add(off).cast::<Vec4u8>();

                let (act_s0, act_k0) = unsafe {
                    (
                        ScalarKernel.probe_strip_activity::<false>(cp, pp),
                        kernel.probe_strip_activity::<false>(cp, pp),
                    )
                };
                assert_eq!(
                    act_k0, act_s0,
                    "[{tier_name}] probe_strip_activity<false> {tag} off={off}"
                );

                let (act_s1, act_k1) = unsafe {
                    (
                        ScalarKernel.probe_strip_activity::<true>(cp, pp),
                        kernel.probe_strip_activity::<true>(cp, pp),
                    )
                };
                assert_eq!(
                    act_k1, act_s1,
                    "[{tier_name}] probe_strip_activity<true> {tag} off={off}"
                );

                let (ch_s0, ch_k0) = unsafe {
                    (
                        ScalarKernel.probe_curr_strip_hash::<false>(cp, pp),
                        kernel.probe_curr_strip_hash::<false>(cp, pp),
                    )
                };
                assert_eq!(
                    ch_k0, ch_s0,
                    "[{tier_name}] probe_curr_strip_hash<false> {tag} off={off}"
                );

                let (ch_s1, ch_k1) = unsafe {
                    (
                        ScalarKernel.probe_curr_strip_hash::<true>(cp, pp),
                        kernel.probe_curr_strip_hash::<true>(cp, pp),
                    )
                };
                assert_eq!(
                    ch_k1, ch_s1,
                    "[{tier_name}] probe_curr_strip_hash<true> {tag} off={off}"
                );

                let (ph_s0, ph_k0) = unsafe {
                    (
                        ScalarKernel.hash_prev_strip_non_uniform::<false>(cp),
                        kernel.hash_prev_strip_non_uniform::<false>(cp),
                    )
                };
                assert_eq!(
                    ph_k0, ph_s0,
                    "[{tier_name}] hash_prev_strip_non_uniform<false> {tag} off={off}"
                );
                if ch_k0.0 {
                    assert_eq!(
                        ph_k0, ch_k0.1,
                        "[{tier_name}] curr vs prev hash<false> mismatch {tag} off={off}"
                    );
                } else {
                    assert_eq!(
                        ch_k0.1, None,
                        "[{tier_name}] unchanged curr hash<false> must be None {tag} off={off}"
                    );
                }

                let (ph_s1, ph_k1) = unsafe {
                    (
                        ScalarKernel.hash_prev_strip_non_uniform::<true>(cp),
                        kernel.hash_prev_strip_non_uniform::<true>(cp),
                    )
                };
                assert_eq!(
                    ph_k1, ph_s1,
                    "[{tier_name}] hash_prev_strip_non_uniform<true> {tag} off={off}"
                );
                if ch_k1.0 {
                    assert_eq!(
                        ph_k1, ch_k1.1,
                        "[{tier_name}] curr vs prev hash<true> mismatch {tag} off={off}"
                    );
                } else {
                    assert_eq!(
                        ch_k1.1, None,
                        "[{tier_name}] unchanged curr hash<true> must be None {tag} off={off}"
                    );
                }
            };

        for off in [0usize, 1, 3, 4, 7, 12, 16, 28, 32, 60] {
            check_probes(&textured, &textured, off, "identical_textured");
            check_probes(&solid, &solid, off, "identical_solid");
            check_probes(&solid, &textured, off, "solid_vs_textured");
            check_probes(&textured, &solid, off, "textured_vs_solid");
        }
        // Per-lane alpha-only, single-pixel RGB diff, and single-pixel non-uniformity.
        for lane in 0..16usize {
            let mut alpha_only = textured;
            alpha_only[lane].3 ^= 0x55;
            check_probes(&alpha_only, &textured, lane * 4, "alpha_only_textured");

            let mut solid_alpha_diff = solid;
            solid_alpha_diff[lane].3 ^= 0x33;
            check_probes(&solid_alpha_diff, &solid, 4, "alpha_only_solid");

            let mut rgb_diff = textured;
            rgb_diff[lane].1 ^= 0x01;
            check_probes(&rgb_diff, &textured, 8, "single_px_rgb_diff");

            let mut almost_solid = solid;
            almost_solid[lane].2 ^= 0x10;
            check_probes(&almost_solid, &solid, 12, "almost_solid_non_uniform");
        }

        // `score_strip_row`: sweep matching counts `ok = 0..=16` (including the
        // exact `11` vs `12` boundary), `moved = 0` vs `moved = 1` at each lane,
        // and alpha-only diffs under `HAS_ALPHA = false / true`.
        for ok in 0..=16usize {
            let mut curr = textured;
            for px in curr.iter_mut().skip(ok) {
                px.0 ^= 0xFF;
            }
            // Static (`prev_dst == curr`): `moved == 0`.
            let prev_src = textured;
            let prev_dst_static = curr;
            curr_raw[4..68].copy_from_slice(bytemuck::bytes_of(&curr));
            prev_raw[8..72].copy_from_slice(bytemuck::bytes_of(&prev_src));
            prev_dst_raw[12..76].copy_from_slice(bytemuck::bytes_of(&prev_dst_static));
            let cp = curr_raw.as_ptr().wrapping_add(4).cast::<Vec4u8>();
            let psp = prev_raw.as_ptr().wrapping_add(8).cast::<Vec4u8>();
            let pdp = prev_dst_raw.as_ptr().wrapping_add(12).cast::<Vec4u8>();
            let s0 = unsafe { ScalarKernel.score_strip_row::<false>(cp, psp, pdp) };
            let k0 = unsafe { kernel.score_strip_row::<false>(cp, psp, pdp) };
            assert_eq!(
                k0, s0,
                "[{tier_name}] score_strip_row<false> static ok={ok}"
            );
            let s1 = unsafe { ScalarKernel.score_strip_row::<true>(cp, psp, pdp) };
            let k1 = unsafe { kernel.score_strip_row::<true>(cp, psp, pdp) };
            assert_eq!(k1, s1, "[{tier_name}] score_strip_row<true> static ok={ok}");

            // Moved in a single lane (`0..16`), plus alpha-only move test.
            for move_lane in [0usize, 1, 7, 8, 11, 12, 15] {
                let mut pd_rgb = curr;
                pd_rgb[move_lane].2 ^= 0x7F;
                prev_dst_raw[12..76].copy_from_slice(bytemuck::bytes_of(&pd_rgb));
                let s0 = unsafe { ScalarKernel.score_strip_row::<false>(cp, psp, pdp) };
                let k0 = unsafe { kernel.score_strip_row::<false>(cp, psp, pdp) };
                assert_eq!(
                    k0, s0,
                    "[{tier_name}] score_strip_row<false> moved ok={ok} lane={move_lane}"
                );
                let s1 = unsafe { ScalarKernel.score_strip_row::<true>(cp, psp, pdp) };
                let k1 = unsafe { kernel.score_strip_row::<true>(cp, psp, pdp) };
                assert_eq!(
                    k1, s1,
                    "[{tier_name}] score_strip_row<true> moved ok={ok} lane={move_lane}"
                );

                let mut pd_alpha = curr;
                pd_alpha[move_lane].3 ^= 0x44;
                prev_dst_raw[12..76].copy_from_slice(bytemuck::bytes_of(&pd_alpha));
                let s0 = unsafe { ScalarKernel.score_strip_row::<false>(cp, psp, pdp) };
                let k0 = unsafe { kernel.score_strip_row::<false>(cp, psp, pdp) };
                assert_eq!(
                    k0, s0,
                    "[{tier_name}] score_strip_row<false> alpha_move ok={ok} lane={move_lane}"
                );
                let s1 = unsafe { ScalarKernel.score_strip_row::<true>(cp, psp, pdp) };
                let k1 = unsafe { kernel.score_strip_row::<true>(cp, psp, pdp) };
                assert_eq!(
                    k1, s1,
                    "[{tier_name}] score_strip_row<true> alpha_move ok={ok} lane={move_lane}"
                );
            }
        }

        // 2. `copy_bytes_stream` across lengths `0..=65, 127, 128, 129, 255, 256, 300`
        //    and all 64B-relative destination alignments with sentinel guard bytes.
        let mut src_buf = vec![0u8; 512];
        for (i, b) in src_buf.iter_mut().enumerate() {
            *b = ((i * 53 + 17) & 0xFF) as u8;
        }
        let mut dst_buf = vec![0xA5u8; 640];
        let base_align = (64 - (dst_buf.as_ptr() as usize & 63)) & 63;
        let lengths: Vec<usize> = (0..=65).chain([127, 128, 129, 255, 256, 300]).collect();
        let align_offsets = [0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63];
        for &dst_misalign in &align_offsets {
            let dst_off = 64 + base_align + dst_misalign;
            let src_off = (dst_misalign * 7) % 61;
            for &len in &lengths {
                dst_buf.fill(0xA5);
                // SAFETY: `src_off + len <= 512`, `dst_off + len <= 640`, non-overlapping.
                unsafe {
                    kernel.copy_bytes_stream(
                        src_buf.as_ptr().add(src_off),
                        dst_buf.as_mut_ptr().add(dst_off),
                        len,
                    );
                }
                kernel.fence_stream();
                assert!(
                    dst_buf[..dst_off].iter().all(|&b| b == 0xA5),
                    "[{tier_name}] copy_bytes_stream clobbered prefix len={len} misalign={dst_misalign}"
                );
                assert_eq!(
                    &dst_buf[dst_off..dst_off + len],
                    &src_buf[src_off..src_off + len],
                    "[{tier_name}] copy_bytes_stream payload mismatch len={len} misalign={dst_misalign}"
                );
                assert!(
                    dst_buf[dst_off + len..].iter().all(|&b| b == 0xA5),
                    "[{tier_name}] copy_bytes_stream clobbered suffix len={len} misalign={dst_misalign}"
                );
            }
        }
    }

    #[inline(always)]
    fn verify_palette_kernel_matches_scalar<K: RowKernel>(kernel: K, tier_name: &str) {
        let widths: Vec<usize> = (1..=65).chain([127, 128, 129, 255, 256, 300]).collect();

        // 1. `validate_palette_indices` across boundary counts and OOB positions.
        for n_colors in [2usize, 3, 4, 5, 15, 16, 17, 128, 255, 256] {
            for len in [
                0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 131,
            ] {
                let valid_byte: u8 = if n_colors <= 2 {
                    0xAA
                } else if n_colors == 3 {
                    0b10_01_00_10
                } else if n_colors <= 4 {
                    0xFF
                } else if n_colors <= 16 {
                    let hi = (n_colors - 1) as u8;
                    let lo = ((n_colors - 2) as u8) & 0x0F;
                    (hi << 4) | lo
                } else {
                    (n_colors - 1) as u8
                };
                let mut buf = vec![valid_byte; len];
                assert!(
                    kernel.validate_palette_indices(n_colors, &buf),
                    "[{tier_name}] validate valid n_colors={n_colors} len={len}"
                );
                if len > 0
                    && (n_colors == 3
                        || (5..16).contains(&n_colors)
                        || (17..256).contains(&n_colors))
                {
                    for pos in [0, len / 2, len - 1] {
                        let bad_vals: &[u8] = if n_colors == 3 {
                            &[0b00_00_00_11, 0b00_00_11_00, 0b00_11_00_00, 0b11_00_00_00]
                        } else if n_colors < 16 {
                            &[n_colors as u8, (n_colors as u8) << 4, 0x0F, 0xF0]
                        } else {
                            &[n_colors as u8, 255]
                        };
                        for &bad in bad_vals {
                            buf[pos] = bad;
                            assert_eq!(
                                kernel.validate_palette_indices(n_colors, &buf),
                                ScalarKernel.validate_palette_indices(n_colors, &buf),
                                "[{tier_name}] validate bad n_colors={n_colors} len={len} pos={pos} bad={bad:#x}"
                            );
                            assert!(
                                !kernel.validate_palette_indices(n_colors, &buf),
                                "[{tier_name}] missed OOB n_colors={n_colors} len={len} pos={pos} bad={bad:#x}"
                            );
                            buf[pos] = valid_byte;
                        }
                    }
                }
            }
        }

        // 2. `pack_palette_indices` and `decode_palette_{1,2,4,8}bpp_row` across
        //    widths, color counts, `idx_pos` offsets, `STREAM = true/false`, and
        //    aligned/unaligned destinations.
        let mut pal256_bytes = Vec::with_capacity(256 * 4);
        let mut lut8 = [0u32; MAX_PALETTE_COLORS];
        for (i, slot) in lut8.iter_mut().enumerate() {
            let px = [
                (i.wrapping_mul(37).wrapping_add(11) & 0xFF) as u8,
                (i.wrapping_mul(73).wrapping_add(29) & 0xFF) as u8,
                (i.wrapping_mul(113).wrapping_add(53) & 0xFF) as u8,
                if i % 3 == 0 {
                    0xFF
                } else {
                    (0x40 + (i & 0x7F)) as u8
                },
            ];
            pal256_bytes.extend_from_slice(&px);
            *slot = u32::from_ne_bytes(px);
        }
        let c0 = Vec4u8(
            pal256_bytes[0],
            pal256_bytes[1],
            pal256_bytes[2],
            pal256_bytes[3],
        );
        let c1 = Vec4u8(
            pal256_bytes[4],
            pal256_bytes[5],
            pal256_bytes[6],
            pal256_bytes[7],
        );
        let lut2: [Vec4u8; 4] = std::array::from_fn(|i| {
            Vec4u8(
                pal256_bytes[i * 4],
                pal256_bytes[i * 4 + 1],
                pal256_bytes[i * 4 + 2],
                pal256_bytes[i * 4 + 3],
            )
        });
        let lut4 = PaletteLut4::from_bytes(true, &pal256_bytes[..16 * 4], 16);

        for &tw in &widths {
            let max_off = 15usize;
            let total_px = tw + max_off;
            for n_colors in [2usize, 3, 4, 5, 15, 16, 17, 128, 255, 256] {
                let indices: Vec<u8> = (0..total_px)
                    .map(|i| {
                        ((i.wrapping_mul(19).wrapping_add(tw * 7 + n_colors)) % n_colors) as u8
                    })
                    .collect();
                let exp_len =
                    crate::serialization::framing::packed_palette_indices_len(total_px, n_colors);
                let mut packed_s = vec![0xA5u8; exp_len];
                let mut packed_k = vec![0x5Au8; exp_len];
                let len_s = ScalarKernel.pack_palette_indices(&indices, n_colors, &mut packed_s);
                let len_k = kernel.pack_palette_indices(&indices, n_colors, &mut packed_k);
                assert_eq!(len_k, exp_len);
                assert_eq!(len_k, len_s);
                assert_eq!(
                    packed_k, packed_s,
                    "[{tier_name}] pack_palette_indices n_colors={n_colors} total_px={total_px}"
                );
                assert!(
                    kernel.validate_palette_indices(n_colors, &packed_k),
                    "[{tier_name}] validate packed n_colors={n_colors} total_px={total_px}"
                );

                // Decode with both aligned (`idx_pos = 0`) and unaligned (`idx_pos = 1..=7`)
                // offsets, and both aligned+streaming (`STREAM = true` when `tw % 16 == 0`)
                // and unaligned (`STREAM = false`) output buffers.
                let mut dst_bytes = vec![0xAAu8; (tw + 17) * 4];
                let aligned_off = (64 - (dst_bytes.as_ptr() as usize & 63)) & 63;
                let mut exp_row = vec![MaybeUninit::uninit(); tw];

                let idx_offsets: &[usize] = if tw.is_multiple_of(16) {
                    &[0, 1, 2, 3, 5, 7, 8]
                } else {
                    &[0, 1, 3, 7]
                };
                for &idx_pos in idx_offsets {
                    if n_colors <= 2 {
                        unsafe {
                            ScalarKernel.decode_palette_1bpp_row::<false>(
                                c0,
                                c1,
                                &packed_k,
                                idx_pos,
                                &mut exp_row,
                            );
                        }
                    } else if n_colors <= 4 {
                        unsafe {
                            ScalarKernel.decode_palette_2bpp_row::<false>(
                                &lut2,
                                &packed_k,
                                idx_pos,
                                &mut exp_row,
                            );
                        }
                    } else if n_colors <= 16 {
                        unsafe {
                            ScalarKernel.decode_palette_4bpp_row::<false>(
                                &lut4,
                                &packed_k,
                                idx_pos,
                                &mut exp_row,
                            );
                        }
                    } else {
                        unsafe {
                            ScalarKernel.decode_palette_8bpp_row::<false>(
                                &lut8,
                                &packed_k[idx_pos..idx_pos + tw],
                                &mut exp_row,
                            );
                        }
                    }
                    let exp_init: Vec<Vec4u8> =
                        exp_row.iter().map(|m| unsafe { m.assume_init() }).collect();

                    let can_stream =
                        K::STREAM_ALIGN > 1 && tw.is_multiple_of(16) && idx_pos.is_multiple_of(8);
                    let modes: &[(bool, usize)] = if can_stream {
                        &[(true, aligned_off), (false, aligned_off + 4)]
                    } else {
                        &[(false, aligned_off), (false, aligned_off + 4)]
                    };
                    for &(stream, byte_off) in modes {
                        let dst: &mut [Vec4u8] =
                            bytemuck::cast_slice_mut(&mut dst_bytes[byte_off..byte_off + tw * 4]);
                        dst.fill(Vec4u8(0, 0, 0, 0));
                        let dst_uninit: &mut [MaybeUninit<Vec4u8>] = unsafe {
                            slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len())
                        };
                        unsafe {
                            if n_colors <= 2 {
                                if stream {
                                    kernel.decode_palette_1bpp_row::<true>(
                                        c0, c1, &packed_k, idx_pos, dst_uninit,
                                    );
                                } else {
                                    kernel.decode_palette_1bpp_row::<false>(
                                        c0, c1, &packed_k, idx_pos, dst_uninit,
                                    );
                                }
                            } else if n_colors <= 4 {
                                if stream {
                                    kernel.decode_palette_2bpp_row::<true>(
                                        &lut2, &packed_k, idx_pos, dst_uninit,
                                    );
                                } else {
                                    kernel.decode_palette_2bpp_row::<false>(
                                        &lut2, &packed_k, idx_pos, dst_uninit,
                                    );
                                }
                            } else if n_colors <= 16 {
                                if stream {
                                    kernel.decode_palette_4bpp_row::<true>(
                                        &lut4, &packed_k, idx_pos, dst_uninit,
                                    );
                                } else {
                                    kernel.decode_palette_4bpp_row::<false>(
                                        &lut4, &packed_k, idx_pos, dst_uninit,
                                    );
                                }
                            } else if stream {
                                kernel.decode_palette_8bpp_row::<true>(
                                    &lut8,
                                    &packed_k[idx_pos..idx_pos + tw],
                                    dst_uninit,
                                );
                            } else {
                                kernel.decode_palette_8bpp_row::<false>(
                                    &lut8,
                                    &packed_k[idx_pos..idx_pos + tw],
                                    dst_uninit,
                                );
                            }
                        }
                        if stream {
                            kernel.fence_stream();
                        }
                        assert_eq!(
                            dst,
                            &exp_init[..],
                            "[{tier_name}] decode_palette n_colors={n_colors} tw={tw} idx_pos={idx_pos} stream={stream}"
                        );
                    }
                }
            }
        }

        // 3. Whole-tile `decode_palette_tile` across odd/SIMD widths & heights,
        //    packed/padded strides, `allow_stream = false/true` (aligned & unaligned),
        //    `pal_has_alpha = false/true`, and upfront OOB rejection leaving zero
        //    partial destination writes.
        verify_palette_tile_decode(kernel, tier_name);
    }

    #[inline(always)]
    fn verify_palette_tile_decode<K: RowKernel>(kernel: K, tier_name: &str) {
        use palette::decode_palette_tile;
        use palette::parse_palette_header;
        use rkyv::util::AlignedVec;

        let color_counts: &[usize] = if cfg!(miri) {
            &[2, 3, 4, 5, 16, 17, 255, 256]
        } else {
            &[2, 3, 4, 5, 15, 16, 17, 128, 255, 256]
        };
        let widths: &[usize] = if cfg!(miri) {
            &[1, 3, 16, 17, 33]
        } else {
            &[
                1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256,
            ]
        };
        let heights: &[usize] = if cfg!(miri) {
            &[1, 3]
        } else {
            &[1, 3, 5, 17, 128]
        };

        let tier = arch::Tier::detect();
        let mut scratch = palette::PaletteScratch::new();
        let max_stride = 256 + 32;
        let max_h = if cfg!(miri) { 5 } else { 130 };
        let mut src_buf = vec![Vec4u8(0, 0, 0, 0xFF); max_stride * max_h];
        let mut aligned_bytes: AlignedVec<64> = AlignedVec::with_capacity(max_stride * max_h * 4);
        aligned_bytes.resize(max_stride * max_h * 4, 0xCC);

        for &tw in widths {
            for &th in heights {
                if !cfg!(miri) && th == 128 && !matches!(tw, 16 | 33 | 64 | 128 | 256) {
                    continue;
                }
                let n_px = tw * th;
                for &n_colors in color_counts {
                    for pal_has_alpha in [false, true] {
                        for (stride_px, x0, y0, allow_stream) in [
                            (tw, 0usize, 0usize, true),
                            (tw.div_ceil(16) * 16 + 16, 0, 1, true),
                            (tw + 13, 1, 1, true),
                            (tw + 7, 2, 0, false),
                        ] {
                            if !cfg!(miri) && th > 5 && x0 != 0 {
                                continue;
                            }
                            let actual_colors = n_colors.min(n_px);
                            if actual_colors < 2 {
                                continue;
                            }
                            for ry in 0..th {
                                for rx in 0..tw {
                                    let i = ry * tw + rx;
                                    let c = if i < actual_colors {
                                        i
                                    } else {
                                        (i.wrapping_mul(37) ^ (ry * 13)) % actual_colors
                                    };
                                    let a = if pal_has_alpha {
                                        0x40 + ((c * 19) & 0xBF) as u8
                                    } else {
                                        0xFF
                                    };
                                    src_buf[(y0 + ry) * stride_px + x0 + rx] = Vec4u8(
                                        (c.wrapping_mul(17) & 0xFF) as u8,
                                        (c.wrapping_mul(53).wrapping_add(7) & 0xFF) as u8,
                                        (c.wrapping_mul(97).wrapping_add(29) & 0xFF) as u8,
                                        a,
                                    );
                                }
                            }
                            let src_ptr = src_buf.as_ptr();
                            let bp = unsafe { BufferPointer::new(&src_ptr, src_buf.len()) };
                            let rect = TileRect {
                                x: x0 as u32,
                                y: y0 as u32,
                                w: tw as u32,
                                h: th as u32,
                            };
                            let got_n = if pal_has_alpha {
                                scratch.try_encode_2d::<true>(bp, stride_px, rect)
                            } else {
                                scratch.try_encode_2d::<false>(bp, stride_px, rect)
                            }
                            .unwrap();
                            assert_eq!(got_n, actual_colors);
                            scratch.pack_indices_range(tier, 0, n_px, got_n);
                            let packed = scratch.packed_indices(0, n_px, got_n);
                            let mut pal_bytes = Vec::with_capacity(got_n * 4);
                            for &c in &scratch.palette[..got_n] {
                                let b = c.to_le_bytes();
                                pal_bytes.extend_from_slice(if pal_has_alpha {
                                    &b[..]
                                } else {
                                    &b[..3]
                                });
                            }
                            let (parsed_n, exp_len) =
                                parse_palette_header(pal_has_alpha, &pal_bytes, n_px).unwrap();
                            assert_eq!(parsed_n, got_n);
                            assert_eq!(exp_len, packed.len());

                            let sentinel = Vec4u8(0xDE, 0xAD, 0xBE, 0xEF);
                            let dst_total = stride_px * (y0 + th + 1);
                            let dst_px = unsafe {
                                std::slice::from_raw_parts_mut(
                                    aligned_bytes.as_mut_ptr().cast::<MaybeUninit<Vec4u8>>(),
                                    dst_total,
                                )
                            };
                            dst_px.fill(MaybeUninit::new(sentinel));
                            let frame = FrameRows {
                                base: dst_px.as_mut_ptr(),
                                stride_px,
                            };
                            let streamed = unsafe {
                                decode_palette_tile(
                                    kernel,
                                    pal_has_alpha,
                                    &pal_bytes,
                                    parsed_n,
                                    packed,
                                    &frame,
                                    x0,
                                    y0,
                                    tw,
                                    th,
                                    allow_stream,
                                )
                            }
                            .unwrap();
                            if streamed {
                                kernel.fence_stream();
                            }
                            let expect_stream = allow_stream
                                && K::STREAM_ALIGN > 1
                                && tw.is_multiple_of(16)
                                && ((dst_px.as_ptr() as usize + (y0 * stride_px + x0) * 4)
                                    .is_multiple_of(K::STREAM_ALIGN))
                                && (stride_px * 4).is_multiple_of(K::STREAM_ALIGN);
                            assert_eq!(
                                streamed, expect_stream,
                                "[{tier_name}] streamed mismatch tw={tw} th={th} n={actual_colors} x0={x0} stride={stride_px}"
                            );
                            for ry in 0..(y0 + th + 1) {
                                for rx in 0..stride_px {
                                    let got = unsafe { dst_px[ry * stride_px + rx].assume_init() };
                                    let in_tile =
                                        ry >= y0 && ry < y0 + th && rx >= x0 && rx < x0 + tw;
                                    let exp = if in_tile {
                                        src_buf[ry * stride_px + rx]
                                    } else {
                                        sentinel
                                    };
                                    assert_eq!(
                                        got, exp,
                                        "[{tier_name}] pixel mismatch at ({rx},{ry}) tw={tw} th={th} n={actual_colors} alpha={pal_has_alpha} stream={allow_stream}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upfront OOB rejection across 2/4/8-bpp must leave canvas completely untouched.
        let sentinel = Vec4u8(0xAA, 0x55, 0xCC, 0x33);
        let mut canvas = vec![MaybeUninit::new(sentinel); 64 * 4];
        let frame = FrameRows {
            base: canvas.as_mut_ptr(),
            stride_px: 64,
        };
        let check_untouched = |canvas: &[MaybeUninit<Vec4u8>]| {
            for slot in canvas {
                assert_eq!(unsafe { slot.assume_init() }, sentinel);
            }
        };

        let pal3 = vec![1u8; 3 * 3];
        for bad_byte in [0usize, 31, 63] {
            let mut bad = vec![0b10_01_00_10u8; 64];
            bad[bad_byte] |= 0b1100_0000;
            assert!(
                unsafe {
                    decode_palette_tile(kernel, false, &pal3, 3, &bad, &frame, 0, 0, 64, 4, true)
                }
                .is_err(),
                "[{tier_name}] expected 2-bpp OOB error at byte {bad_byte}"
            );
            check_untouched(&canvas);
        }

        let pal15 = vec![2u8; 15 * 3];
        for bad_byte in [0usize, 31, 63, 127] {
            for bad_val in [0x0Fu8, 0xF0u8] {
                let mut bad = vec![0xE3u8; 128];
                bad[bad_byte] = bad_val;
                assert!(
                    unsafe {
                        decode_palette_tile(
                            kernel, false, &pal15, 15, &bad, &frame, 0, 0, 64, 4, true,
                        )
                    }
                    .is_err(),
                    "[{tier_name}] expected 4-bpp OOB error at byte {bad_byte} val={bad_val:#x}"
                );
                check_untouched(&canvas);
            }
        }

        let pal255 = vec![3u8; 255 * 3];
        for bad_byte in [0usize, 31, 63, 127, 255] {
            let mut bad = vec![254u8; 256];
            bad[bad_byte] = 255;
            assert!(
                unsafe {
                    decode_palette_tile(
                        kernel, false, &pal255, 255, &bad, &frame, 0, 0, 64, 4, true,
                    )
                }
                .is_err(),
                "[{tier_name}] expected 8-bpp OOB error at byte {bad_byte}"
            );
            check_untouched(&canvas);
        }
    }

    /// Compares every compiled, host-supported backend tier bit-for-bit against
    /// [`ScalarKernel`] across all edge widths for every row primitive.
    #[test]
    fn test_arch_kernel_matches_scalar_reference() {
        let tested = arch::verify_all_supported_kernels();
        assert_eq!(tested, arch::Tier::supported_tiers());
        assert!(!tested.is_empty());
    }

    /// Verifies exact `SlotDirty`, 128-bit keyed hash, [`RectScanMeta`], and
    /// `shadow` byte parity across every supported tier for quadrants
    /// transitioning from clean -> small crop -> large crop (`>= 1024 px`) ->
    /// `bbox.full` at row 0, mid-quadrant, and last row (under both
    /// `cache_non_empty = false` and `true`, `HAS_ALPHA = false` and `true`,
    /// clipped edge quadrants, `n_qx > STACK_HASH_CAP`, and scroll columns).
    #[test]
    fn test_quadrant_bbox_transitions_hash_meta_and_shadow_parity() {
        let key = tile_hash::TileHashKey::from_seed(0xC0DE_5168_CAFE_BEEF);
        let tiers = arch::Tier::supported_tiers();
        assert!(!tiers.is_empty());

        let fill_frame =
            |w: usize, h: usize, stride: usize, seed: u32, garbage_x: bool| -> Vec<u8> {
                let mut buf = vec![0u8; h * stride];
                for y in 0..h {
                    for x in 0..w {
                        let off = y * stride + x * 4;
                        let v = (x as u32)
                            .wrapping_mul(31)
                            .wrapping_add((y as u32).wrapping_mul(17))
                            .wrapping_add(seed);
                        buf[off] = (v & 0xFF) as u8;
                        buf[off + 1] = ((v >> 3) & 0xFF) as u8;
                        buf[off + 2] = ((v >> 7) & 0xFF) as u8;
                        buf[off + 3] = if garbage_x {
                            ((v >> 11) ^ 0xA5) as u8
                        } else {
                            0xFF
                        };
                    }
                }
                buf
            };

        let verify_band = |width: usize,
                           height: usize,
                           stride: usize,
                           base_shadow: &[u8],
                           src: &[u8],
                           has_alpha: bool,
                           cache_non_empty: bool,
                           with_scans: bool,
                           label: &str| {
            let n_qx = width.div_ceil(SUBTILE_W);
            let sub_bands = [ScanSubBand::RowBand2D {
                y0: 0,
                y1: height as u32,
                slot_base: 0,
            }];

            // Scalar reference via `ScalarKernel` + `hash_rect_from_shadow`.
            let mut ref_shadow = base_shadow.to_vec();
            let mut ref_slots = vec![SlotDirty::Clean; n_qx];
            let mut ref_hashes = vec![tile_hash::TileHashResult::EMPTY; n_qx];
            let max_crop_for = |cw: usize, ch: usize| (cw * ch * CROP_MAX_AREA_PCT) / 100;
            for qx in 0..n_qx {
                let x0 = qx * SUBTILE_W;
                let cw = (width - x0).min(SUBTILE_W);
                let mut bbox = QuadBbox::CLEAN;
                for y in 0..height {
                    let row_off = y * stride + x0 * 4;
                    let src_px: &[Vec4u8] = bytemuck::cast_slice(&src[row_off..row_off + cw * 4]);
                    let sp = src_px.as_ptr();
                    let bp = unsafe { BufferPointer::new(&sp, cw) };
                    let dst_px: &mut [Vec4u8] =
                        bytemuck::cast_slice_mut(&mut ref_shadow[row_off..row_off + cw * 4]);
                    if bbox.full {
                        dst_px.copy_from_slice(src_px);
                    } else {
                        let span = if has_alpha {
                            ScalarKernel.diff_and_update_row::<true>(bp, dst_px)
                        } else {
                            ScalarKernel.diff_and_update_row::<false>(bp, dst_px)
                        };
                        if let Some((fx, lx)) = span {
                            bbox.record_row(y, fx, lx, max_crop_for(cw, height));
                        }
                    }
                }
                let qrect = TileRect {
                    x: x0 as u32,
                    y: 0,
                    w: cw as u32,
                    h: height as u32,
                };
                if bbox.full {
                    ref_slots[qx] = SlotDirty::Full;
                    if with_scans {
                        ref_hashes[qx] = if has_alpha {
                            unsafe {
                                tile_hash::hash_rect_from_shadow::<true>(
                                    &key,
                                    ref_shadow.as_ptr(),
                                    stride,
                                    qrect,
                                )
                            }
                        } else {
                            unsafe {
                                tile_hash::hash_rect_from_shadow::<false>(
                                    &key,
                                    ref_shadow.as_ptr(),
                                    stride,
                                    qrect,
                                )
                            }
                        };
                    }
                } else if bbox.is_dirty() {
                    let area =
                        usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
                    let has_quad_hash =
                        with_scans && cache_non_empty && area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
                    ref_slots[qx] = SlotDirty::Cropped {
                        min_x: bbox.min_x,
                        max_x: bbox.max_x,
                        min_y: bbox.min_y,
                        max_y: bbox.max_y,
                        has_quad_hash,
                    };
                    if has_quad_hash {
                        ref_hashes[qx] = if has_alpha {
                            unsafe {
                                tile_hash::hash_rect_from_shadow::<true>(
                                    &key,
                                    ref_shadow.as_ptr(),
                                    stride,
                                    qrect,
                                )
                            }
                        } else {
                            unsafe {
                                tile_hash::hash_rect_from_shadow::<false>(
                                    &key,
                                    ref_shadow.as_ptr(),
                                    stride,
                                    qrect,
                                )
                            }
                        };
                    }
                }
            }

            let src_ptr = src.as_ptr();
            let bp = unsafe { BufferPointer::new(&src_ptr, src.len()) };
            let aos = unsafe { bp.cast::<Vec4u8>() };
            let stride_px = stride / 4;

            for &tier in &tiers {
                let mut shadow_buf = base_shadow.to_vec();
                let shadow = ShadowRows {
                    base: shadow_buf.as_mut_ptr(),
                    stride,
                };
                let dirty_slots: Vec<AtomicU32> = (0..n_qx)
                    .map(|_| AtomicU32::new(SlotDirty::CLEAN_RAW))
                    .collect();
                let mut scans = tile_cache::SlotScans::new(n_qx, key);
                let bucket = WorkBucket::new(0, sub_bands.len());
                let schedule = PhaseASchedule::single(&bucket, None);
                unsafe {
                    tier.diff_and_update_worker(
                        has_alpha,
                        bp,
                        aos,
                        &shadow,
                        width,
                        stride,
                        stride_px,
                        n_qx,
                        &sub_bands,
                        &dirty_slots,
                        with_scans.then_some(&scans),
                        cache_non_empty,
                        None,
                        &schedule,
                        0,
                    );
                }

                let scans_slice = scans.as_slice();
                for qx in 0..n_qx {
                    let got_slot = SlotDirty::decode(dirty_slots[qx].load(Ordering::Relaxed));
                    assert_eq!(
                        got_slot, ref_slots[qx],
                        "{label}: tier={tier:?} qx={qx} has_alpha={has_alpha} cache={cache_non_empty} scans={with_scans}"
                    );
                    let needs_hash = matches!(
                        got_slot,
                        SlotDirty::Full
                            | SlotDirty::Cropped {
                                has_quad_hash: true,
                                ..
                            }
                    ) && with_scans;
                    if needs_hash {
                        let got_scan = scans_slice[qx];
                        assert_eq!(
                            got_scan, ref_hashes[qx],
                            "{label}: hash/meta mismatch tier={tier:?} qx={qx} has_alpha={has_alpha} cache={cache_non_empty}"
                        );
                    }
                    // Verify canonical shadow bytes match `ref_shadow` on every pixel.
                    let x0 = qx * SUBTILE_W;
                    let cw = (width - x0).min(SUBTILE_W);
                    for y in 0..height {
                        let off = y * stride + x0 * 4;
                        let g = &shadow_buf[off..off + cw * 4];
                        let e = &ref_shadow[off..off + cw * 4];
                        if has_alpha {
                            assert_eq!(
                                g, e,
                                "{label}: argb shadow mismatch tier={tier:?} qx={qx} y={y}"
                            );
                        } else {
                            for px in 0..cw {
                                assert_eq!(
                                    &g[px * 4..px * 4 + 3],
                                    &e[px * 4..px * 4 + 3],
                                    "{label}: xrgb shadow mismatch tier={tier:?} qx={qx} y={y} px={px}"
                                );
                            }
                        }
                    }
                }
            }
        };

        // Case 1: Mixed transitions in one 7-quadrant band (clipped right edge
        // `width = 6 * 128 + 37 = 805`, `height = 128` or `32` under Miri):
        // - Q0: 100% clean (plus alpha-only noise under Xrgb)
        // - Q1: small crop `< 1024 px` (stays `Cropped { has_quad_hash: false }`)
        // - Q2: large crop `>= 1024 px`, never `bbox.full` (`Cropped { has_quad_hash }`)
        // - Q3: clean prefix -> crosses `>= 1024 px` at `ry = 12` -> `bbox.full` mid-band
        // - Q4: solid opaque tile transitioning to `bbox.full` mid-band
        // - Q5: escalates to `bbox.full` on the VERY LAST row (`ry = height - 1`)
        // - Q6: clipped right edge (`cw = 37`) solid translucent transitioning mid-band
        #[cfg(not(miri))]
        let band_h = 128usize;
        #[cfg(miri)]
        let band_h = 32usize;
        let width = 6 * SUBTILE_W + 37;
        let stride = (width + 5) * 4;
        for has_alpha in [false, true] {
            let base = fill_frame(width, band_h, stride, 11, !has_alpha);
            let mut src = base.clone();

            // Q0: alpha-only perturbation when `!has_alpha` (must remain Clean).
            if !has_alpha {
                for y in 0..band_h {
                    src[y * stride + 7 * 4 + 3] ^= 0x55;
                }
            }

            // Q1: small crop (`20 x 10 = 200 px < 1024 px`).
            for y in 3..13 {
                for x in 10..30 {
                    src[y * stride + (SUBTILE_W + x) * 4] ^= 0x33;
                }
            }

            // Q2: large crop (`64 x 20 = 1280 px >= 1024 px`, `< 45%` of quadrant).
            let q2_h = (band_h / 4).max(20).min(band_h);
            for y in 2..(2 + q2_h).min(band_h) {
                for x in 4..68 {
                    src[y * stride + (2 * SUBTILE_W + x) * 4 + 1] ^= 0x44;
                }
            }

            // Q3: clean prefix (`0..4`), then `128`-px wide strip causing
            // `>= 1024 px` at `ry = 11` and `bbox.full` mid-band.
            for y in 4..band_h {
                for x in 0..SUBTILE_W {
                    src[y * stride + (3 * SUBTILE_W + x) * 4 + 2] ^= 0x77;
                }
            }

            // Q4: solid opaque target color across entire quadrant (tests
            // `still_solid == true` through `StartAbsorb -> AbsorbShadow -> CopyAndAbsorb`).
            let solid_opq = [0x22u8, 0x44, 0x66, 0xFF];
            for y in 0..band_h {
                for x in 0..SUBTILE_W {
                    let off = y * stride + (4 * SUBTILE_W + x) * 4;
                    src[off..off + 4].copy_from_slice(&solid_opq);
                }
            }

            // Q5: one dirty pixel on row 0 (`x = 0`) and one dirty pixel on the
            // very last row (`y = band_h - 1, x = 127`) -> bbox jumps from
            // `1x1 = 1 px` directly to `128 x band_h` (`bbox.full`) on the LAST row!
            src[5 * SUBTILE_W * 4] ^= 0x19;
            src[(band_h - 1) * stride + (5 * SUBTILE_W + SUBTILE_W - 1) * 4 + 1] ^= 0x2A;

            // Q6: clipped right edge (`cw = 37`), solid translucent target color.
            let solid_tr = [0x10u8, 0x80, 0xC0, if has_alpha { 0x7F } else { 0x33 }];
            for y in 0..band_h {
                for x in 0..37 {
                    let off = y * stride + (6 * SUBTILE_W + x) * 4;
                    src[off..off + 4].copy_from_slice(&solid_tr);
                }
            }

            for cache_non_empty in [false, true] {
                for with_scans in [false, true] {
                    verify_band(
                        width,
                        band_h,
                        stride,
                        &base,
                        &src,
                        has_alpha,
                        cache_non_empty,
                        with_scans,
                        "mixed_transitions",
                    );
                }
            }
        }

        // Case 2: Row-0 immediate `bbox.full` escalation (`height = 1` band) +
        // all-quadrants-full early exit (`copy_and_absorb_band_rows` on `height = 16`).
        for (h_case, label) in [(1usize, "row0_full"), (16usize, "all_full_copy_absorb")] {
            let w_case = 3 * SUBTILE_W + 19;
            let st_case = w_case * 4;
            for has_alpha in [false, true] {
                let base = fill_frame(w_case, h_case, st_case, 5, !has_alpha);
                let mut src = fill_frame(w_case, h_case, st_case, 99, !has_alpha);
                // Make Q1 solid opaque to exercise mixed solid + non-solid in
                // `copy_and_absorb_band_rows`.
                for y in 0..h_case {
                    for x in 0..SUBTILE_W {
                        let off = y * st_case + (SUBTILE_W + x) * 4;
                        src[off..off + 4].copy_from_slice(&[0xAB, 0xCD, 0xEF, 0xFF]);
                    }
                }
                verify_band(
                    w_case, h_case, st_case, &base, &src, has_alpha, true, true, label,
                );
                verify_band(
                    w_case, h_case, st_case, &base, &src, has_alpha, false, true, label,
                );
            }
        }

        // Case 3: Wide frame exceeding `STACK_HASH_CAP` (`35` quadrants = 4463 px)
        // to exercise heap fallback in both active-band paths.
        #[cfg(not(miri))]
        {
            let wide_w = (STACK_HASH_CAP + 3) * SUBTILE_W - 17;
            let wide_h = 20usize;
            let wide_stride = wide_w * 4;
            let base = fill_frame(wide_w, wide_h, wide_stride, 7, true);
            let mut src = base.clone();
            // Trigger `bbox.full` on first, middle, and clipped last quadrant.
            for qx in [0, STACK_HASH_CAP, STACK_HASH_CAP + 2] {
                let x0 = qx * SUBTILE_W;
                let cw = (wide_w - x0).min(SUBTILE_W);
                for y in 0..wide_h {
                    for x in 0..cw {
                        src[y * wide_stride + (x0 + x) * 4] ^= 0x55;
                    }
                }
            }
            verify_band(
                wide_w,
                wide_h,
                wide_stride,
                &base,
                &src,
                false,
                true,
                true,
                "wide_heap_hashers",
            );
        }

        // Case 4: Scroll column (`move_up = true` and `false`) where quadrants
        // transition clean -> cropped -> `bbox.full` at row 0, mid-quadrant, and
        // last row across all supported tiers.
        let sw = 2 * SUBTILE_W + 31;
        let sh = 2 * SUBTILE_H;
        let s_stride = (sw + 3) * 4;
        let s_n_qx = sw.div_ceil(SUBTILE_W);
        let s_bands = [
            ScanSubBand::RowBand2D {
                y0: 0,
                y1: SUBTILE_H as u32,
                slot_base: 0,
            },
            ScanSubBand::RowBand2D {
                y0: SUBTILE_H as u32,
                y1: sh as u32,
                slot_base: s_n_qx,
            },
        ];
        let total_s_slots = 2 * s_n_qx;
        for &dy in &[-16i32, 16i32] {
            let sr = ScrollRect {
                x: 0,
                y: 0,
                width: sw as u32,
                height: sh as u32,
                dy,
            };
            for has_alpha in [false, true] {
                let base = fill_frame(sw, sh, s_stride, 21, !has_alpha);
                let mut expected_shadow = base.clone();
                // Apply reference scroll blit to `expected_shadow` first.
                let shift = dy.unsigned_abs() as usize;
                let copy_h = sh - shift;
                if dy < 0 {
                    for y in 0..copy_h {
                        let dst_off = y * s_stride;
                        let src_off = (y + shift) * s_stride;
                        expected_shadow.copy_within(src_off..src_off + sw * 4, dst_off);
                    }
                } else {
                    for y in (shift..sh).rev() {
                        let dst_off = y * s_stride;
                        let src_off = (y - shift) * s_stride;
                        expected_shadow.copy_within(src_off..src_off + sw * 4, dst_off);
                    }
                }
                // Build `src` as exact scrolled image + mid-quadrant full diff
                // in Q0 (top band) + last-row full diff in Q1 (bottom band).
                let mut src = expected_shadow.clone();
                for y in 0..SUBTILE_H {
                    for x in 0..SUBTILE_W {
                        src[y * s_stride + x * 4] ^= 0x41;
                    }
                }
                src[SUBTILE_H * s_stride + SUBTILE_W * 4] ^= 0x22;
                src[(sh - 1) * s_stride + (2 * SUBTILE_W - 1) * 4 + 2] ^= 0x33;

                let sp = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&sp, src.len()) };
                let aos = unsafe { bp.cast::<Vec4u8>() };
                let s_stride_px = s_stride / 4;

                for cache_non_empty in [false, true] {
                    // Run scalar tier as reference, compare every tier against it.
                    let mut ref_slots_raw = Vec::new();
                    let mut ref_scans_vec = Vec::new();
                    let mut ref_shadow_final = Vec::new();
                    for (t_idx, &tier) in tiers.iter().enumerate() {
                        let mut sh_buf = base.clone();
                        let shadow = ShadowRows {
                            base: sh_buf.as_mut_ptr(),
                            stride: s_stride,
                        };
                        let dirty_slots: Vec<AtomicU32> = (0..total_s_slots)
                            .map(|_| AtomicU32::new(SlotDirty::CLEAN_RAW))
                            .collect();
                        let mut scans = tile_cache::SlotScans::new(total_s_slots, key);
                        let next_sub = AtomicUsize::new(0);
                        unsafe {
                            tier.diff_and_update_scroll_worker(
                                has_alpha,
                                bp,
                                aos,
                                &shadow,
                                sw,
                                s_stride,
                                s_stride_px,
                                s_n_qx,
                                &s_bands,
                                sr,
                                &dirty_slots,
                                Some(&scans),
                                cache_non_empty,
                                &next_sub,
                            );
                        }
                        let raw_slots: Vec<u32> = dirty_slots
                            .iter()
                            .map(|s| s.load(Ordering::Relaxed))
                            .collect();
                        let scans_slice = scans.as_slice();
                        let mut got_scans = Vec::with_capacity(total_s_slots);
                        for (slot_idx, &raw) in raw_slots.iter().enumerate() {
                            let decoded = SlotDirty::decode(raw);
                            let needs_hash = matches!(
                                decoded,
                                SlotDirty::Full
                                    | SlotDirty::Cropped {
                                        has_quad_hash: true,
                                        ..
                                    }
                            );
                            if needs_hash {
                                let by = (slot_idx / s_n_qx) * SUBTILE_H;
                                let qx = slot_idx % s_n_qx;
                                let x0 = qx * SUBTILE_W;
                                let cw = (sw - x0).min(SUBTILE_W);
                                let qrect = TileRect {
                                    x: x0 as u32,
                                    y: by as u32,
                                    w: cw as u32,
                                    h: SUBTILE_H as u32,
                                };
                                let expected_hash = if has_alpha {
                                    unsafe {
                                        tile_hash::hash_rect_from_shadow::<true>(
                                            &key,
                                            sh_buf.as_ptr(),
                                            s_stride,
                                            qrect,
                                        )
                                    }
                                } else {
                                    unsafe {
                                        tile_hash::hash_rect_from_shadow::<false>(
                                            &key,
                                            sh_buf.as_ptr(),
                                            s_stride,
                                            qrect,
                                        )
                                    }
                                };
                                let got = scans_slice[slot_idx];
                                assert_eq!(
                                    got, expected_hash,
                                    "scroll dy={dy} tier={tier:?} slot={slot_idx} has_alpha={has_alpha} cache={cache_non_empty}"
                                );
                                got_scans.push(Some(got));
                            } else {
                                got_scans.push(None);
                            }
                        }
                        if t_idx == 0 {
                            ref_slots_raw = raw_slots;
                            ref_scans_vec = got_scans;
                            ref_shadow_final = sh_buf;
                        } else {
                            assert_eq!(raw_slots, ref_slots_raw, "scroll slots tier={tier:?}");
                            assert_eq!(got_scans, ref_scans_vec, "scroll scans tier={tier:?}");
                            for y in 0..sh {
                                let off = y * s_stride;
                                if has_alpha {
                                    assert_eq!(
                                        &sh_buf[off..off + sw * 4],
                                        &ref_shadow_final[off..off + sw * 4]
                                    );
                                } else {
                                    for x in 0..sw {
                                        assert_eq!(
                                            &sh_buf[off + x * 4..off + x * 4 + 3],
                                            &ref_shadow_final[off + x * 4..off + x * 4 + 3]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn test_x86_tier_detection_and_pruning() {
        let supported = arch::X86Tier::supported_tiers();
        assert!(!supported.is_empty());

        let mut expected = Vec::new();
        #[cfg(all(
            not(target_feature = "ssse3"),
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ))]
        expected.push(arch::X86Tier::Scalar);

        #[cfg(all(
            not(target_feature = "avx2"),
            not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            )),
        ))]
        if std::arch::is_x86_feature_detected!("ssse3") {
            expected.push(arch::X86Tier::Ssse3);
        }

        #[cfg(not(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        )))]
        if std::arch::is_x86_feature_detected!("avx2") {
            expected.push(arch::X86Tier::Avx2);
        }

        if std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
        {
            expected.push(arch::X86Tier::Avx512);
        }

        assert_eq!(supported, expected);

        let detected = arch::X86Tier::detect();
        assert_eq!(detected, *supported.last().unwrap());
        // Second call exercises the relaxed-atomic cache fast path.
        assert_eq!(arch::X86Tier::detect(), detected);

        #[cfg(all(
            target_feature = "avx512bw",
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
        ))]
        assert_eq!(std::mem::size_of::<arch::X86Tier>(), 0);
    }

    fn filter_and_compress_with_tier(
        tier: arch::Tier,
        data: BufferPointer<u8>,
        metadata: BufferMetadata,
        mode: CompressionMode,
        compressor: &mut ShardingCompressor,
    ) -> CompressedTiles {
        let mut streaming_opt = None;
        filter_and_compress_full_streaming_with_tier(
            tier,
            data,
            ClientId(0),
            WlSurfaceId(0),
            metadata,
            mode,
            None,
            None,
            compressor,
            |streaming| {
                streaming_opt = Some(streaming);
            },
        );
        let streaming = streaming_opt.unwrap();
        let tile_rects = streaming.header.tile_rects;
        let mut tiles: Vec<CompressedTile> = streaming.tiles.into_iter().collect();
        tiles.sort_by_key(|t| t.tile_idx);
        CompressedTiles {
            metadata,
            tile_rects,
            tiles,
        }
    }

    fn decompress_and_unfilter_with_tier(
        tier: arch::Tier,
        tiles: &CompressedTiles,
        decompressor: &mut ShardingDecompressor,
        output_buf: &mut [u8],
    ) -> Result<()> {
        let total_pixels = tiles.metadata.validate().location(loc!())?;
        let byte_len = total_pixels
            .checked_mul(4)
            .context(loc!(), "pixel count overflow")?;
        validate_tile_rects(
            &tiles.tile_rects,
            tiles.metadata.width as u32,
            tiles.metadata.height as u32,
        )
        .location(loc!())?;
        let n_tiles = tiles.tiles.len();
        let next_idx = AtomicUsize::new(0);
        // SAFETY: Validated `tiles.metadata` and `tiles.tile_rects` above.
        unsafe {
            unfilter_tiles_parallel_with_tier(
                tier,
                decompressor,
                tiles.metadata,
                byte_len,
                false,
                &[],
                &tiles.tile_rects,
                Some(output_buf),
                None,
                || {
                    let i = next_idx.fetch_add(1, Ordering::Relaxed);
                    (i < n_tiles).then(|| Ok(Cow::Borrowed(&tiles.tiles[i])))
                },
            )
        }
    }

    /// Exercises cross-tier encode -> decode interoperability (every compiled,
    /// host-supported `(enc_tier, dec_tier)` pair) across lossless AoS/palette/solid
    /// and hybrid translucent JPEG tiles, plus shadow-diff incremental parity.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cross_tier_encode_decode_and_shadow_diff_parity() {
        let tiers = arch::Tier::supported_tiers();
        assert!(!tiers.is_empty());

        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };

            // 1. Multi-tile `300x280` frame exercising 256x256 + clipped right/bottom
            //    edge tiles, all 4 row predictors, solid, palette, and varying alpha.
            let (w, h) = (300i32, 280i32);
            let mut lossless_frame = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let off = (y * w as usize + x) * 4;
                    let (b, g, r) = match (y / 32) % 4 {
                        0 => (
                            ((x * 131 + y * 59) & 0xFF) as u8,
                            ((x * 97 + y * 173) & 0xFF) as u8,
                            ((x * 61 + y * 211) & 0xFF) as u8,
                        ),
                        1 => ((x * 2) as u8, (x * 2 + 10) as u8, (x * 2 + 20) as u8),
                        2 => (
                            ((x * 37) & 0xFF) as u8,
                            ((x * 53) & 0xFF) as u8,
                            ((x * 71) & 0xFF) as u8,
                        ),
                        _ => ((x + y * 2) as u8, (x * 2 + y * 3) as u8, (x * 3 + y) as u8),
                    };
                    lossless_frame[off] = b;
                    lossless_frame[off + 1] = g;
                    lossless_frame[off + 2] = r;
                    lossless_frame[off + 3] = ((x * 5 + y * 11) & 0xFF) as u8;
                }
            }

            for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
                let meta = BufferMetadata {
                    width: w,
                    height: h,
                    stride: w * 4,
                    format,
                };
                let mut expected = lossless_frame.clone();
                if format == BufferFormat::Xrgb8888 {
                    for px in expected.as_chunks_mut::<4>().0 {
                        px[3] = 0xFF;
                    }
                }
                let ptr = lossless_frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, lossless_frame.len()) };

                let mut encoded_by_tier = Vec::with_capacity(tiers.len());
                for &enc_tier in &tiers {
                    let tiles = filter_and_compress_with_tier(
                        enc_tier,
                        bp,
                        meta,
                        CompressionMode::Lossless,
                        compressor,
                    );
                    encoded_by_tier.push((enc_tier, tiles));
                }

                // Every lossless tier must emit byte-identical wire payloads.
                for (enc_tier, tiles) in &encoded_by_tier[1..] {
                    assert_eq!(
                        tiles, &encoded_by_tier[0].1,
                        "lossless tile mismatch between {:?} and {enc_tier:?} ({format:?})",
                        encoded_by_tier[0].0
                    );
                }

                // Every `(enc_tier, dec_tier)` pair must decode bit-exact at both alignments.
                let mut buf = vec![0xAAu8; expected.len() + 68];
                let aligned_off = (64 - (buf.as_ptr() as usize & 63)) & 63;
                for (enc_tier, tiles) in &encoded_by_tier {
                    for &dec_tier in &tiers {
                        for off in [aligned_off, aligned_off + 4] {
                            buf.fill(0xAA);
                            decompress_and_unfilter_with_tier(
                                dec_tier,
                                tiles,
                                decompressor,
                                &mut buf[off..off + expected.len()],
                            )
                            .unwrap();
                            assert_eq!(
                                &buf[off..off + expected.len()],
                                expected.as_slice(),
                                "cross-tier decode mismatch enc={enc_tier:?} dec={dec_tier:?} off={off} ({format:?})"
                            );
                        }
                    }
                }
            }

            // 2. Hybrid translucent `256x256` mixed text/photo frame with varying alpha:
            //    cross-tier encode/decode preserves bit-exact alpha and text quadrants.
            let meta_256 = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Argb8888,
            };
            let mut hybrid_frame = vec![0u8; 256 * 256 * 4];
            fill_ui_text(&mut hybrid_frame, 256, tr(0, 0, 128, 128), true);
            fill_smooth_photo(&mut hybrid_frame, 256, tr(128, 0, 128, 128), 0xFF);
            fill_smooth_photo(&mut hybrid_frame, 256, tr(0, 128, 256, 128), 0xFF);
            for y in 0..256usize {
                for x in 0..256usize {
                    if x >= 128 || y >= 128 {
                        hybrid_frame[(y * 256 + x) * 4 + 3] = ((x + y * 3) & 0xFF) as u8;
                    }
                }
            }
            let hptr = hybrid_frame.as_ptr();
            let hbp = unsafe { BufferPointer::new(&hptr, hybrid_frame.len()) };
            for &enc_tier in &tiers {
                let tiles = filter_and_compress_with_tier(
                    enc_tier,
                    hbp,
                    meta_256,
                    CompressionMode::hybrid_default(),
                    compressor,
                );
                for &dec_tier in &tiers {
                    let mut out = vec![0x55u8; hybrid_frame.len()];
                    decompress_and_unfilter_with_tier(dec_tier, &tiles, decompressor, &mut out)
                        .unwrap();
                    assert!(
                        rect_psnr_rgb(&hybrid_frame, &out, 256, tr(0, 0, 128, 128)).is_infinite(),
                        "hybrid text quadrant mismatch enc={enc_tier:?} dec={dec_tier:?}"
                    );
                    for (i, (got, exp)) in out
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(hybrid_frame.as_chunks::<4>().0.iter())
                        .enumerate()
                    {
                        assert_eq!(
                            got[3], exp[3],
                            "hybrid alpha mismatch at px {i} enc={enc_tier:?} dec={dec_tier:?}"
                        );
                    }
                }
            }

            // 3. Incremental shadow diff + encode parity across all tiers.
            let inc_meta = BufferMetadata {
                width: 300,
                height: 280,
                stride: 300 * 4,
                format: BufferFormat::Argb8888,
            };
            let mut edited = lossless_frame.clone();
            edited[(10 * 300 + 20) * 4 + 3] ^= 0x40; // alpha-only change in TL quadrant
            edited[(200 * 300 + 270) * 4] ^= 0x33; // RGB change in right clipped quadrant
            let eptr = edited.as_ptr();
            let ebp = unsafe { BufferPointer::new(&eptr, edited.len()) };

            let mut ref_dirty = None;
            let mut ref_shadow = None;
            for &tier in &tiers {
                let mut shadow = ShadowBuffer::from_pixels(lossless_frame.clone());
                let mut streaming = None;
                let dirty = filter_and_compress_incremental_streaming_with_tier(
                    tier,
                    ebp,
                    ClientId(1),
                    WlSurfaceId(2),
                    inc_meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |s| streaming = Some(s),
                )
                .dirty_tiles;
                assert!(streaming.is_some());
                assert_eq!(
                    shadow.pixels(),
                    edited.as_slice(),
                    "shadow sync mismatch for {tier:?}"
                );
                if let Some(prev_dirty) = &ref_dirty {
                    assert_eq!(&dirty, prev_dirty, "dirty rects mismatch for {tier:?}");
                } else {
                    ref_dirty = Some(dirty);
                }
                if let Some(prev_shadow) = &ref_shadow {
                    assert_eq!(
                        shadow.pixels(),
                        prev_shadow,
                        "shadow parity mismatch for {tier:?}"
                    );
                } else {
                    ref_shadow = Some(shadow.into_pixels());
                }
            }
        });
    }

    #[test]
    fn test_partition_tiles_2d_and_thin_band_fallback() {
        let meta = BufferMetadata {
            width: 3840,
            height: 2160,
            stride: 3840 * 4,
            format: BufferFormat::Xrgb8888,
        };
        // Full 4K frame: 15 x 9 = 135 tiles (bottom row clipped to h = 112).
        let (full_subs, full, _) = plan_incremental_scan(&meta, None).unwrap();
        assert_eq!(full_subs.len(), 17);
        assert_eq!(full.len(), 15 * 9);
        assert_matches!(
            full[0],
            BaseScanEntry::Tile2D {
                rect: TileRect {
                    x: 0,
                    y: 0,
                    w: 256,
                    h: 256,
                },
                n_quads: 4,
                ..
            }
        );
        assert_matches!(
            *full.last().unwrap(),
            BaseScanEntry::Tile2D {
                rect: TileRect {
                    x: 3584,
                    y: 2048,
                    w: 256,
                    h: 112,
                },
                n_quads: 2,
                ..
            }
        );

        // Partial `< 8`-row bands emit 1 full-width 1D strip each.
        for h in [1u32, 7] {
            let (_, strip, _) = plan_incremental_scan(
                &meta,
                Some(&[RowBand {
                    y0: 100,
                    y1: 100 + h,
                }]),
            )
            .unwrap();
            assert_eq!(
                strip,
                vec![BaseScanEntry::Strip1D {
                    rect: TileRect {
                        x: 0,
                        y: 100,
                        w: 3840,
                        h,
                    },
                    slot_idx: 0,
                }]
            );
        }

        // Exact `band_h == 8` crosses into 2D `256x8` tiling (15 tiles on 4K).
        let (_, band8, _) =
            plan_incremental_scan(&meta, Some(&[RowBand { y0: 100, y1: 108 }])).unwrap();
        assert_eq!(band8.len(), 15);
        assert_matches!(
            band8[0],
            BaseScanEntry::Tile2D {
                rect: TileRect {
                    x: 0,
                    y: 100,
                    w: 256,
                    h: 8,
                },
                n_quads: 2,
                ..
            }
        );

        // Ultra-wide thin strip (`width = 20000, h = 7` -> `140,000 > TILE_PIXELS`) splits cleanly
        // so every tile remains `<= TILE_PIXELS`.
        let wide_meta = BufferMetadata {
            width: 20_000,
            height: 20,
            stride: 20_000 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let (_, wide_strips, _) =
            plan_incremental_scan(&wide_meta, Some(&[RowBand { y0: 0, y1: 7 }])).unwrap();
        assert_eq!(wide_strips.len(), 3);
        assert!(wide_strips.iter().all(
            |t| matches!(t, BaseScanEntry::Strip1D { rect, .. } if rect.checked_pixels().is_ok())
        ));
        assert_eq!(wide_strips.iter().map(|t| t.rect().w).sum::<u32>(), 20_000);

        // Empty bands -> 0 tiles; invalid bands -> Err.
        assert!(
            plan_incremental_scan(&meta, Some(&[]))
                .unwrap()
                .1
                .is_empty()
        );
        for bad_bands in [
            vec![RowBand { y0: 5, y1: 5 }],
            vec![RowBand { y0: 10, y1: 5 }],
            vec![RowBand { y0: 10, y1: 20 }, RowBand { y0: 15, y1: 25 }],
            vec![RowBand { y0: 2150, y1: 2161 }],
        ] {
            assert!(plan_incremental_scan(&meta, Some(&bad_bands)).is_err());
        }

        // Quadrant subdivision across 4-, 2-horizontal-, 2-vertical-, and 1-quadrant shapes.
        let (q4, n4) = tile_quadrants(TileRect {
            x: 256,
            y: 512,
            w: 200,
            h: 150,
        });
        assert_eq!(n4, 4);
        assert_eq!(
            &q4[..n4],
            &[
                TileRect {
                    x: 256,
                    y: 512,
                    w: 128,
                    h: 128
                },
                TileRect {
                    x: 384,
                    y: 512,
                    w: 72,
                    h: 128
                },
                TileRect {
                    x: 256,
                    y: 640,
                    w: 128,
                    h: 22
                },
                TileRect {
                    x: 384,
                    y: 640,
                    w: 72,
                    h: 22
                },
            ]
        );
        let (q2h, n2h) = tile_quadrants(TileRect {
            x: 0,
            y: 2048,
            w: 256,
            h: 112,
        });
        assert_eq!(n2h, 2);
        assert_eq!(
            &q2h[..n2h],
            &[
                TileRect {
                    x: 0,
                    y: 2048,
                    w: 128,
                    h: 112
                },
                TileRect {
                    x: 128,
                    y: 2048,
                    w: 128,
                    h: 112
                },
            ]
        );
        let (q2v, n2v) = tile_quadrants(TileRect {
            x: 1792,
            y: 0,
            w: 128,
            h: 256,
        });
        assert_eq!(n2v, 2);
        assert_eq!(
            &q2v[..n2v],
            &[
                TileRect {
                    x: 1792,
                    y: 0,
                    w: 128,
                    h: 128
                },
                TileRect {
                    x: 1792,
                    y: 128,
                    w: 128,
                    h: 128
                },
            ]
        );
        let (q1, n1) = tile_quadrants(TileRect {
            x: 1792,
            y: 960,
            w: 128,
            h: 120,
        });
        assert_eq!(n1, 1);
        assert_eq!(
            q1[0],
            TileRect {
                x: 1792,
                y: 960,
                w: 128,
                h: 120
            }
        );
    }

    #[test]
    fn test_validate_tile_rects_sweep_line() {
        let tr = |x, y, w, h| TileRect { x, y, w, h };

        // Full non-overlapping mixed 256x256 + 128x128 + edge tiling.
        let mixed = [
            tr(0, 0, 256, 256),
            tr(256, 0, 128, 128),
            tr(384, 0, 64, 128),
            tr(256, 128, 192, 128),
            tr(0, 256, 448, 44),
        ];
        assert_eq!(validate_tile_rects(&mixed, 448, 300).unwrap(), 448 * 300);
        // Shuffled order must validate identically.
        let mut rev = mixed;
        rev.reverse();
        assert_eq!(validate_tile_rects(&rev, 448, 300).unwrap(), 448 * 300);
        assert_eq!(validate_tile_rects(&[], 448, 300).unwrap(), 0);

        // Overlaps: partial corner, contained, enclosing arriving later, duplicate,
        // and horizontal overlap with a tall earlier strip.
        for bad in [
            vec![tr(0, 0, 40, 40), tr(39, 39, 25, 25)],
            vec![tr(0, 0, 128, 128), tr(10, 10, 20, 20)],
            vec![tr(20, 0, 20, 60), tr(0, 30, 60, 10)],
            vec![tr(0, 0, 32, 32), tr(0, 0, 32, 32)],
            vec![tr(100, 0, 10, 200), tr(0, 50, 101, 10)],
        ] {
            assert!(
                validate_tile_rects(&bad, 256, 256).is_err(),
                "expected overlap err for {bad:?}"
            );
        }

        // Out-of-bounds and zero-dim rects.
        for bad in [
            tr(32, 0, 33, 64),
            tr(0, 32, 64, 33),
            tr(0, 0, 0, 10),
            tr(0, 0, 10, 0),
        ] {
            assert!(validate_tile_rects(&[bad], 64, 64).is_err());
        }
    }

    #[test]
    fn test_shadow_diff_and_copy_miri_safe() {
        // Exercises `copy_rect_to_shadow`, `diff_and_update_strip`, and
        // `diff_and_update_row_band_2d` across every compiled, host-supported
        // tier (both packed and padded strides, partial-dirty and bulk-all-dirty
        // fast paths, Xrgb vs Argb).
        for tier in arch::Tier::supported_tiers() {
            for pad_bytes in [0usize, 12usize] {
                let width = 160usize;
                let height = 20usize;
                let stride = width * 4 + pad_bytes;
                let stride_px = stride / 4;
                let len = stride * height;

                let mut src = test_vec(len);
                let mut shadow = vec![0xAAu8; len];

                let src_ptr = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&src_ptr, len) };
                let aos = unsafe { bp.cast::<Vec4u8>() };
                let shadow_rows = ShadowRows {
                    base: shadow.as_mut_ptr(),
                    stride,
                };
                let rect_band = TileRect {
                    x: 0,
                    y: 0,
                    w: width as u32,
                    h: 12,
                };
                let rect_strip = TileRect {
                    x: 0,
                    y: 12,
                    w: width as u32,
                    h: 4,
                };

                unsafe {
                    copy_rect_to_shadow(bp, &shadow_rows, stride, rect_band);
                    copy_rect_to_shadow(bp, &shadow_rows, stride, rect_strip);
                    assert!(!tier.diff_and_update_strip::<true>(
                        aos,
                        &shadow_rows,
                        stride_px,
                        rect_strip
                    ));
                    assert!(!tier.diff_and_update_strip::<false>(
                        aos,
                        &shadow_rows,
                        stride_px,
                        rect_strip
                    ));
                }

                // Flip only byte 3 (X/alpha) in quadrant 0 of `[0..12)` -> ignored
                // on Xrgb, caught as tight `1x1` crop on Argb.
                let a_off = 5 * stride + 10 * 4 + 3;
                src[a_off] ^= 0xFF;
                let src_ptr = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&src_ptr, len) };
                let aos = unsafe { bp.cast::<Vec4u8>() };
                let slots_xrgb = [
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                ];
                unsafe {
                    tier.diff_and_update_row_band_2d::<false>(
                        bp,
                        aos,
                        &shadow_rows,
                        width,
                        stride,
                        0,
                        12,
                        &slots_xrgb,
                    );
                }
                assert_eq!(
                    SlotDirty::decode(slots_xrgb[0].load(Ordering::Relaxed)),
                    SlotDirty::Clean
                );
                assert_eq!(
                    SlotDirty::decode(slots_xrgb[1].load(Ordering::Relaxed)),
                    SlotDirty::Clean
                );
                assert_ne!(shadow[a_off], src[a_off]);

                let slots_argb = [
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                ];
                unsafe {
                    tier.diff_and_update_row_band_2d::<true>(
                        bp,
                        aos,
                        &shadow_rows,
                        width,
                        stride,
                        0,
                        12,
                        &slots_argb,
                    );
                }
                assert_eq!(
                    SlotDirty::decode(slots_argb[0].load(Ordering::Relaxed)),
                    SlotDirty::Cropped {
                        min_x: 10,
                        max_x: 11,
                        min_y: 5,
                        max_y: 6,
                        has_quad_hash: false,
                    }
                );
                assert_eq!(
                    SlotDirty::decode(slots_argb[1].load(Ordering::Relaxed)),
                    SlotDirty::Clean
                );
                assert_eq!(shadow[a_off], src[a_off]);

                // Flip RGB across wide spans in BOTH quadrants (`> 40%` of each
                // quadrant's `12`-row height) to exercise the all-quadrants-full
                // bulk copy fast path.
                for y in 0..8usize {
                    src[y * stride + 2 * 4] ^= 0x11;
                    src[y * stride + 120 * 4] ^= 0x11;
                    src[y * stride + 130 * 4] ^= 0x22;
                    src[y * stride + 158 * 4] ^= 0x22;
                }
                let q0_off = 2 * 4;
                let q1_off = 130 * 4;
                let last_off = 11 * stride + 150 * 4;
                src[last_off] ^= 0x33;
                let src_ptr = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&src_ptr, len) };
                let aos = unsafe { bp.cast::<Vec4u8>() };
                let slots_all = [
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                    AtomicU32::new(SlotDirty::CLEAN_RAW),
                ];
                unsafe {
                    tier.diff_and_update_row_band_2d::<false>(
                        bp,
                        aos,
                        &shadow_rows,
                        width,
                        stride,
                        0,
                        12,
                        &slots_all,
                    );
                }
                assert_eq!(
                    SlotDirty::decode(slots_all[0].load(Ordering::Relaxed)),
                    SlotDirty::Full
                );
                assert_eq!(
                    SlotDirty::decode(slots_all[1].load(Ordering::Relaxed)),
                    SlotDirty::Full
                );
                assert_eq!(shadow[q0_off], src[q0_off]);
                assert_eq!(shadow[q1_off], src[q1_off]);
                assert_eq!(shadow[last_off], src[last_off]);

                // Flip RGB in `rect_strip` (row 13 + trailing row 15).
                let strip_off = 13 * stride + 150 * 4;
                let strip_last = 15 * stride + 10 * 4;
                src[strip_off] ^= 0x5A;
                src[strip_last] ^= 0x3C;
                let src_ptr = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&src_ptr, len) };
                let aos = unsafe { bp.cast::<Vec4u8>() };
                unsafe {
                    assert!(tier.diff_and_update_strip::<false>(
                        aos,
                        &shadow_rows,
                        stride_px,
                        rect_strip
                    ));
                }
                assert_eq!(shadow[strip_off], src[strip_off]);
                assert_eq!(shadow[strip_last], src[strip_last]);

                // Stride padding on every row must stay untouched (`0xAA`).
                if pad_bytes > 0 {
                    for y in 0..height {
                        assert_eq!(
                            &shadow[y * stride + width * 4..(y + 1) * stride],
                            &vec![0xAAu8; pad_bytes][..]
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_roundtrip_2d_modes_and_shapes() {
        with_test_codec(|compressor, decompressor| {
            // Diverse 2D & 1D shapes including edge tiles (`1x1`, non-multiples of 8/12/32/256,
            // thin `< 8`-row full frames, multi-tile 2D frames).
            for (w, h) in [
                (1, 1),
                (3, 1),
                (7, 5),
                (16, 7),
                (17, 9),
                (33, 33),
                (256, 256),
                (257, 257),
                (520, 300),
                (1920, 1),
            ] {
                let data = test_vec((w * h * 4) as usize);
                test_roundtrip_2d(
                    &data,
                    w,
                    h,
                    BufferFormat::Argb8888,
                    compressor,
                    decompressor,
                );
                test_roundtrip_2d(
                    &data,
                    w,
                    h,
                    BufferFormat::Xrgb8888,
                    compressor,
                    decompressor,
                );
            }

            // Solid opaque and solid translucent tiles.
            let solid_opaque: Vec<u8> = [0x20u8, 0x70, 0xD0, 0xFF].repeat(300 * 280);
            let t_solid = test_roundtrip_2d(
                &solid_opaque,
                300,
                280,
                BufferFormat::Argb8888,
                compressor,
                decompressor,
            );
            assert!(t_solid.tiles.iter().all(|t| matches!(
                t.payload,
                TilePayload::Leaf(LeafPayload::Solid([0x20, 0x70, 0xD0, 0xFF]))
            )));
            test_roundtrip_2d(
                &solid_opaque,
                300,
                280,
                BufferFormat::Xrgb8888,
                compressor,
                decompressor,
            );

            let solid_trans: Vec<u8> = [0x20u8, 0x70, 0xD0, 0x80].repeat(300 * 280);
            let t_trans = test_roundtrip_2d(
                &solid_trans,
                300,
                280,
                BufferFormat::Argb8888,
                compressor,
                decompressor,
            );
            assert!(t_trans.tiles.iter().all(|t| matches!(
                t.payload,
                TilePayload::Leaf(LeafPayload::Solid([0x20, 0x70, 0xD0, 0x80]))
            )));
            test_roundtrip_2d(
                &solid_trans,
                300,
                280,
                BufferFormat::Xrgb8888,
                compressor,
                decompressor,
            );

            // Palette tiles across 1/2/4/8 bpp (opaque & translucent).
            for (n_colors, translucent) in [(2, false), (3, true), (15, false), (200, true)] {
                let (w, h) = (96i32, 64i32);
                let mut data = vec![0u8; (w * h * 4) as usize];
                for i in 0..(w * h) as usize {
                    let c = i % n_colors;
                    data[i * 4] = (c * 13) as u8;
                    data[i * 4 + 1] = (c * 29) as u8;
                    data[i * 4 + 2] = (c * 53) as u8;
                    data[i * 4 + 3] = if translucent {
                        0x40 + (c as u8 & 0x7F)
                    } else {
                        0xFF
                    };
                }
                let tiles = test_roundtrip_2d(
                    &data,
                    w,
                    h,
                    BufferFormat::Argb8888,
                    compressor,
                    decompressor,
                );
                assert_matches!(
                    tiles.tiles[0].payload,
                    TilePayload::Leaf(LeafPayload::Palette {
                        has_alpha,
                        ..
                    }) if has_alpha == translucent
                );
                test_roundtrip_2d(
                    &data,
                    w,
                    h,
                    BufferFormat::Xrgb8888,
                    compressor,
                    decompressor,
                );
            }

            // Force AosZstd (> 256 unique colors) exercising all 4 row predictors
            // `{None, Sub, Up, SubUp}` + opaque (`Omitted`), constant (`Constant(0x80)`),
            // and varying (`Uncompressed`/`Zstd`) alpha.
            let (w, h) = (128i32, 64i32);
            let mut rich = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let idx = (y * w as usize + x) * 4;
                    let (b, g, r) = match y / 16 {
                        // High-frequency noise -> prefers None
                        0 => (
                            ((x * 131 + y * 59) & 0xFF) as u8,
                            ((x * 97 + y * 173) & 0xFF) as u8,
                            ((x * 61 + y * 211) & 0xFF) as u8,
                        ),
                        // Horizontal smooth ramp -> prefers Sub
                        1 => ((x * 2) as u8, (x * 2 + 10) as u8, (x * 2 + 20) as u8),
                        // Vertical copy of previous row -> prefers Up
                        2 => (
                            ((x * 37) & 0xFF) as u8,
                            ((x * 53) & 0xFF) as u8,
                            ((x * 71) & 0xFF) as u8,
                        ),
                        // Bilinear gradient -> prefers SubUp
                        _ => ((x + y * 2) as u8, (x * 2 + y * 3) as u8, (x * 3 + y) as u8),
                    };
                    rich[idx] = b;
                    rich[idx + 1] = g;
                    rich[idx + 2] = r;
                    rich[idx + 3] = 0xFF;
                }
            }
            let t_opaque_argb = test_roundtrip_2d(
                &rich,
                w,
                h,
                BufferFormat::Argb8888,
                compressor,
                decompressor,
            );
            assert_matches!(
                t_opaque_argb.tiles[0].payload,
                TilePayload::Leaf(LeafPayload::AosZstd {
                    alpha: AlphaChannel::Omitted,
                    ..
                })
            );

            let mut const_a = rich.clone();
            for px in const_a.as_chunks_mut::<4>().0 {
                px[3] = 0x80;
            }
            let t_const_a = test_roundtrip_2d(
                &const_a,
                w,
                h,
                BufferFormat::Argb8888,
                compressor,
                decompressor,
            );
            assert_matches!(
                t_const_a.tiles[0].payload,
                TilePayload::Leaf(LeafPayload::AosZstd {
                    alpha: AlphaChannel::Constant(0x80),
                    ..
                })
            );

            let mut var_a = rich;
            for (i, px) in var_a.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = ((i * 7 + (i / 128) * 13) & 0xFF) as u8;
            }
            let t_var_a = test_roundtrip_2d(
                &var_a,
                w,
                h,
                BufferFormat::Argb8888,
                compressor,
                decompressor,
            );
            assert_matches!(
                t_var_a.tiles[0].payload,
                TilePayload::Leaf(LeafPayload::AosZstd {
                    alpha: AlphaChannel::Uncompressed(_) | AlphaChannel::Zstd(_),
                    ..
                })
            );
            test_roundtrip_2d(
                &var_a,
                w,
                h,
                BufferFormat::Xrgb8888,
                compressor,
                decompressor,
            );
        });
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn proptest_roundtrip_2d(
            w in 1i32..=320,
            h in 1i32..=300,
            seed in any::<u64>(),
            xrgb in any::<bool>(),
        ) {
            let n_px = (w as usize) * (h as usize);
            let mut data = vec![0u8; n_px * 4];
            let mut state = seed.wrapping_add(1);
            for px in data.as_chunks_mut::<4>().0 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                px.copy_from_slice(&((state >> 32) as u32).to_le_bytes());
            }
            let format = if xrgb {
                BufferFormat::Xrgb8888
            } else {
                BufferFormat::Argb8888
            };
            with_test_codec(|compressor, decompressor| {
                test_roundtrip_2d(&data, w, h, format, compressor, decompressor);
            });
        }
    }

    #[test]
    fn test_compute_partial_row_bands() {
        assert_eq!(compute_partial_row_bands(&[], 3840, 2160), None);
        for (w, h) in [(0, 2160), (-1, 2160), (3840, 0), (3840, -1)] {
            assert_eq!(
                compute_partial_row_bands(&[Rectangle::new(0, 0, 10, 10)], w, h),
                None,
                "expected None for dimensions ({w}, {h})"
            );
        }
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, 3840, 2160)], 3840, 2160),
            None
        );
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, i32::MAX, i32::MAX)], 3840, 2160),
            None
        );
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, 100, 1620)], 3840, 2160),
            None
        );
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, 100, 1619)], 3840, 2160),
            Some(vec![RowBand { y0: 0, y1: 1619 }])
        );
        assert_eq!(
            compute_partial_row_bands(
                &[
                    Rectangle::new(10, 500, 50, 40),
                    Rectangle::new(-10, -20, 100, 35),
                    Rectangle::new(3800, 2150, 100, 30),
                    Rectangle::new(0, 540, 20, 10),
                    Rectangle::new(0, 510, 20, 20),
                ],
                3840,
                2160,
            ),
            Some(vec![
                RowBand { y0: 0, y1: 15 },
                RowBand { y0: 500, y1: 550 },
                RowBand { y0: 2150, y1: 2160 },
            ])
        );
        for rect in [
            Rectangle::new(-100, -100, 20, 20),
            Rectangle::new(-20, 10, 20, 20),
            Rectangle::new(3840, 10, 20, 20),
            Rectangle::new(10, -20, 20, 20),
            Rectangle::new(10, 2160, 20, 20),
            Rectangle::new(10, 10, 0, 20),
            Rectangle::new(10, 10, 20, -5),
            Rectangle::new(10, 10, -5, 20),
        ] {
            assert_eq!(
                compute_partial_row_bands(&[rect], 3840, 2160),
                Some(Vec::new()),
                "expected empty bands for {rect:?}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_incremental_quadrant_skip_and_shadow_sync() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };

            for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
                for pad_bytes in [0i32, 16i32] {
                    // `400x360` produces:
                    // - (0, 0, 256, 256): 4 quadrants (128x128 each)
                    // - (256, 0, 144, 256): 4 clipped quadrants (128x128, 16x128, 128x128, 16x128)
                    // - (0, 256, 256, 104): 2 horizontal quadrants (128x104, 128x104)
                    // - (256, 256, 144, 104): 2 horizontal clipped quadrants (128x104, 16x104)
                    let width = 400i32;
                    let height = 360i32;
                    let stride = width * 4 + pad_bytes;
                    let stride_us = stride as usize;
                    let vis_bytes = (width * 4) as usize;
                    let len = stride_us * height as usize;
                    let metadata = BufferMetadata {
                        width,
                        height,
                        stride,
                        format,
                    };

                    let mut frame = test_vec(len);
                    let mut shadow = ShadowBuffer::new();
                    let mut client_canvas = AlignedVec::<64>::with_capacity(len);
                    client_canvas.resize(len, 0xCC);

                    let apply_stream = |streaming: StreamingCompressedTiles,
                                        decomp: &mut ShardingDecompressor,
                                        canvas: &mut [u8]| {
                        let mut wire = Vec::new();
                        streaming.framed_write(&mut wire).unwrap();
                        let mut rdr = wire.as_slice();
                        let mut rx_stream = None;
                        CompressedTileStream::read_from_stream(&mut rdr, |s| {
                            rx_stream = Some(s);
                            Ok(())
                        })
                        .unwrap();
                        assert!(rdr.is_empty());
                        let validated = ValidatedTileStream::new(rx_stream.unwrap()).unwrap();
                        decompress_and_unfilter_streaming(validated, decomp, canvas, None).unwrap();
                    };

                    let check_canvas = |canvas: &[u8], expected: &[u8]| {
                        for y in 0..height as usize {
                            let row = y * stride_us;
                            for x in 0..width as usize {
                                let px = row + x * 4;
                                let exp_a = if format == BufferFormat::Xrgb8888 {
                                    0xFF
                                } else {
                                    expected[px + 3]
                                };
                                assert_eq!(
                                    &canvas[px..px + 4],
                                    &[expected[px], expected[px + 1], expected[px + 2], exp_a],
                                    "pixel ({x},{y}) mismatch (format={format:?}, pad={pad_bytes})"
                                );
                            }
                            assert_eq!(
                                &canvas[row + vis_bytes..row + stride_us],
                                &vec![0xCCu8; pad_bytes as usize][..],
                                "stride padding modified on row {y}"
                            );
                        }
                    };

                    let set_px = |buf: &mut [u8], x: usize, y: usize, bgr_xor: u8, a_xor: u8| {
                        let off = y * stride_us + x * 4;
                        buf[off] ^= bgr_xor;
                        buf[off + 1] ^= bgr_xor.rotate_left(1);
                        buf[off + 2] ^= bgr_xor.rotate_left(2);
                        buf[off + 3] ^= a_xor;
                    };

                    // Step 0: Initial full sync populates `shadow` and client canvas.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        filter_and_compress_full_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            Some(&mut shadow),
                            None,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 1: Zero-dirty full-frame scan -> returns empty and NEVER calls `on_stream_ready`.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut called = false;
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |_| called = true,
                        );
                        assert!(diff.is_unchanged());
                        assert!(!called);
                    }

                    // Step 2: Perturbing ONLY the alpha/X byte -> skipped on Xrgb8888,
                    // caught as a tight `1x1` crop on Argb8888!
                    set_px(&mut frame, 10, 20, 0, 0x55);
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        if format == BufferFormat::Xrgb8888 {
                            assert!(diff.is_unchanged());
                            assert!(streaming.is_none());
                        } else {
                            assert_eq!(diff.scroll_rect, None);
                            assert_eq!(diff.dirty_tiles, vec![tr(10, 20, 1, 1)]);
                            apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                        }
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 3: Scattered single-pixel edits across 4 distinct quadrants
                    // emit tight `1x1` cropped rects in raster order.
                    set_px(&mut frame, 150, 10, 0x33, 0); // TR of (0,0,256,256)
                    set_px(&mut frame, 390, 5, 0x44, 0); // TR of (256,0,144,256)
                    set_px(&mut frame, 260, 200, 0x55, 0); // BL of (256,0,144,256)
                    set_px(&mut frame, 200, 300, 0x66, 0); // right of (0,256,256,104)
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(diff.scroll_rect, None);
                        assert_eq!(
                            diff.dirty_tiles,
                            vec![
                                tr(150, 10, 1, 1),
                                tr(390, 5, 1, 1),
                                tr(260, 200, 1, 1),
                                tr(200, 300, 1, 1),
                            ]
                        );
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 4: All 4 quadrants exceed the crop threshold in (0,0,256,256)
                    // -> coalesces to ONE `256x256` tile! Same for all 2 quadrants in
                    // bottom-right (256,256,144,104) -> ONE `144x104` tile!
                    for (qx0, qy0, qw, qh) in [
                        (0usize, 0usize, 128usize, 128usize),
                        (128, 0, 128, 128),
                        (0, 128, 128, 128),
                        (128, 128, 128, 128),
                        (256, 256, 128, 104),
                        (384, 256, 16, 104),
                    ] {
                        set_px(&mut frame, qx0 + 1, qy0 + 1, 0x11, 0);
                        set_px(&mut frame, qx0 + qw - 2, qy0 + qh - 2, 0x22, 0);
                    }
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(diff.scroll_rect, None);
                        assert_eq!(
                            diff.dirty_tiles,
                            vec![tr(0, 0, 256, 256), tr(256, 256, 144, 104)]
                        );
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 5: `< 8`-row partial band (1D strip): unchanged -> skipped;
                    // changed -> emits single 1D strip and updates shadow.
                    let thin_band = [RowBand { y0: 40, y1: 43 }];
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            Some(&thin_band),
                            &mut shadow,
                            None,
                            compressor,
                            |_| panic!("unchanged 1D strip must not emit stream"),
                        );
                        assert!(diff.is_unchanged());
                    }
                    set_px(&mut frame, 300, 41, 0x77, 0);
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            Some(&thin_band),
                            &mut shadow,
                            None,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(diff.scroll_rect, None);
                        assert_eq!(diff.dirty_tiles, vec![tr(0, 40, 400, 3)]);
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 6: Immediately re-scanning the full frame sees 0 dirty tiles,
                    // proving the shadow buffer stayed bit-exact across all edits.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |_| panic!("shadow desynchronized after multi-frame edits"),
                        );
                        assert!(diff.is_unchanged());
                    }

                    // Step 7: Empty bands slice returns empty immediately.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let diff = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            CompressionMode::Lossless,
                            Some(&[]),
                            &mut shadow,
                            None,
                            compressor,
                            |_| panic!("empty bands must not emit stream"),
                        );
                        assert!(diff.is_unchanged());
                    }
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_pipeline_errors_and_disjoint_validation() {
        with_test_codec(|compressor, decompressor| {
            let meta = BufferMetadata {
                width: 64,
                height: 64,
                stride: 64 * 4,
                format: BufferFormat::Argb8888,
            };
            let mut out = vec![0u8; 64 * 64 * 4];

            // Overlapping TileRects must be rejected before writing!
            let overlapping = CompressedTiles {
                metadata: meta,
                tile_rects: vec![
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 40,
                        h: 40,
                    },
                    TileRect {
                        x: 39,
                        y: 39,
                        w: 25,
                        h: 25,
                    },
                ],
                tiles: vec![
                    CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                    },
                    CompressedTile {
                        tile_idx: 1,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid([4, 5, 6, 0xFF])),
                    },
                ],
            };
            assert!(decompress_and_unfilter(&overlapping, decompressor, &mut out).is_err());

            // Duplicate `tile_idx` and out-of-range `tile_idx` must be rejected!
            let dup_idx = CompressedTiles {
                metadata: meta,
                tile_rects: vec![
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 32,
                        h: 64,
                    },
                    TileRect {
                        x: 32,
                        y: 0,
                        w: 32,
                        h: 64,
                    },
                ],
                tiles: vec![
                    CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                    },
                    CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid([4, 5, 6, 0xFF])),
                    },
                ],
            };
            assert!(decompress_and_unfilter(&dup_idx, decompressor, &mut out).is_err());

            let oob_idx = CompressedTiles {
                metadata: meta,
                tile_rects: vec![TileRect {
                    x: 0,
                    y: 0,
                    w: 64,
                    h: 64,
                }],
                tiles: vec![CompressedTile {
                    tile_idx: 1,
                    cache_put: None,
                    payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                }],
            };
            assert!(decompress_and_unfilter(&oob_idx, decompressor, &mut out).is_err());

            // Out-of-bounds TileRect (`x + w > width`, `y + h > height`, empty `w == 0`).
            for bad_rect in [
                TileRect {
                    x: 32,
                    y: 0,
                    w: 33,
                    h: 64,
                },
                TileRect {
                    x: 0,
                    y: 32,
                    w: 64,
                    h: 33,
                },
                TileRect {
                    x: 0,
                    y: 0,
                    w: 0,
                    h: 10,
                },
            ] {
                let bad = CompressedTiles {
                    metadata: meta,
                    tile_rects: vec![bad_rect],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out).is_err());
            }

            // Invalid row-0 predictor (`Up = 2`, `SubUp = 3`), wrong predictor length,
            // non-zero predictor padding bits, bad payload lengths, bad Zstd data.
            let r4x3 = TileRect {
                x: 0,
                y: 0,
                w: 4,
                h: 3,
            };
            for bad_payload in [
                LeafPayload::AosZstd {
                    predictors: vec![2], // Up on row 0
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![3], // SubUp on row 0
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![0, 0], // wrong len (expected 1)
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![0b01_00_00_00], // non-zero padding bits for h=3
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 35]), // != 36
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Zstd(vec![0xFF, 0x00, 0x01]),
                    alpha: AlphaChannel::Omitted,
                },
                LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 36]),
                    alpha: AlphaChannel::Uncompressed(vec![0u8; 11]), // != 12
                },
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                    data: EncodedBytes::Uncompressed(vec![0b11, 0, 0]), // index 3 >= 3
                },
            ] {
                let bad = CompressedTiles {
                    metadata: meta,
                    tile_rects: vec![r4x3],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(bad_payload),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out).is_err());
            }

            // Translucent payloads rejected on Xrgb8888.
            let xrgb_meta = BufferMetadata {
                format: BufferFormat::Xrgb8888,
                ..meta
            };
            for bad_xrgb in [
                LeafPayload::Solid([1, 2, 3, 0x80]),
                LeafPayload::Palette {
                    has_alpha: true,
                    palette: vec![1, 2, 3, 0x80, 4, 5, 6, 0xFF],
                    data: EncodedBytes::Uncompressed(vec![0, 0]),
                },
                LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 36]),
                    alpha: AlphaChannel::Constant(0x80),
                },
                LeafPayload::Jpeg {
                    data: vec![0xFF, 0xD8, 0xFF, 0xD9],
                    alpha: AlphaChannel::Constant(0x80),
                },
            ] {
                let bad = CompressedTiles {
                    metadata: xrgb_meta,
                    tile_rects: vec![r4x3],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(bad_xrgb),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out).is_err());
            }

            // Multi-tile frame: early drop in `on_stream_ready` (with and without shadow),
            // out-of-order arrival, undersized `output_buf`, missing tile, and upfront
            // `ValidatedTileStream::new` rejections.
            let meta_2 = BufferMetadata {
                width: 300,
                height: 20,
                stride: 300 * 4,
                format: BufferFormat::Argb8888,
            };
            let data_2 = test_vec(300 * 20 * 4);
            let ptr_2 = data_2.as_ptr();
            let bp_2 = unsafe { BufferPointer::new(&ptr_2, data_2.len()) };

            filter_and_compress_full_streaming(
                bp_2,
                ClientId(1),
                WlSurfaceId(1),
                meta_2,
                CompressionMode::Lossless,
                None,
                None,
                compressor,
                drop,
            );
            let mut shadow_dropped = ShadowBuffer::new();
            filter_and_compress_full_streaming(
                bp_2,
                ClientId(1),
                WlSurfaceId(1),
                meta_2,
                CompressionMode::Lossless,
                Some(&mut shadow_dropped),
                None,
                compressor,
                drop,
            );
            // Shadow must still be fully populated even when stream receiver dropped early.
            assert_eq!(shadow_dropped.pixels(), data_2.as_slice());

            let mut two_tiles =
                filter_and_compress(bp_2, meta_2, CompressionMode::Lossless, compressor);
            assert_eq!(two_tiles.tiles.len(), 2);
            two_tiles.tiles.reverse();
            let mut reversed_out = vec![0u8; data_2.len()];
            decompress_and_unfilter(&two_tiles, decompressor, &mut reversed_out).unwrap();
            assert_eq!(reversed_out, data_2);

            let mut short_out = vec![0u8; data_2.len() - 4];
            assert!(decompress_and_unfilter(&two_tiles, decompressor, &mut short_out).is_err());

            // Streaming missing tile (`done < n_tiles`).
            let (tx_short, rx_short) = crossbeam_channel::unbounded();
            tx_short.send(Ok(two_tiles.tiles[0].clone())).unwrap();
            drop(tx_short);
            let validated_short = ValidatedTileStream::new(CompressedTileStream {
                header: RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta_2,
                    incremental: false,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: two_tiles.tile_rects.clone(),
                },
                tiles: rx_short,
            })
            .unwrap();
            assert!(
                decompress_and_unfilter_streaming(
                    validated_short,
                    decompressor,
                    &mut reversed_out,
                    None,
                )
                .is_err()
            );

            // Upfront `ValidatedTileStream::new` rejections:
            // - non-incremental incomplete coverage (missing pixels / empty)
            // - non-incremental with ScrollRect
            // - invalid ScrollRect bounds (`dy == 0`, out-of-bounds)
            // - overlapping rects
            // - out-of-bounds rect
            // - zero-dim rect
            // - excessive tile count (> width * height)
            let bad_headers = [
                // Non-incremental with empty rects.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: false,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![],
                },
                // Non-incremental with ScrollRect.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: false,
                    scroll_rect: Some(ScrollRect {
                        x: 0,
                        y: 0,
                        width: 64,
                        height: 64,
                        dy: -4,
                    }),
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 64,
                        h: 64,
                    }],
                },
                // Incremental with invalid ScrollRect (`dy == 0`).
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    scroll_rect: Some(ScrollRect {
                        x: 0,
                        y: 0,
                        width: 64,
                        height: 64,
                        dy: 0,
                    }),
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![],
                },
                // Incremental with out-of-bounds ScrollRect.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    scroll_rect: Some(ScrollRect {
                        x: 32,
                        y: 0,
                        width: 33,
                        height: 64,
                        dy: -4,
                    }),
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![],
                },
                // Non-incremental covering only half the frame.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: false,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 32,
                        h: 64,
                    }],
                },
                // Incremental overlapping rects.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![
                        TileRect {
                            x: 0,
                            y: 0,
                            w: 40,
                            h: 40,
                        },
                        TileRect {
                            x: 39,
                            y: 0,
                            w: 20,
                            h: 40,
                        },
                    ],
                },
                // Out-of-bounds rect.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 32,
                        y: 0,
                        w: 33,
                        h: 64,
                    }],
                },
                // Zero-dim rect.
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 0,
                        h: 64,
                    }],
                },
                // Excessive tile count (`5 > 2 * 2`).
                RawBufferHeader {
                    frame_seq: 0,
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: BufferMetadata {
                        width: 2,
                        height: 2,
                        stride: 8,
                        format: BufferFormat::Xrgb8888,
                    },
                    incremental: true,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects: vec![
                        TileRect {
                            x: 0,
                            y: 0,
                            w: 1,
                            h: 1,
                        };
                        5
                    ],
                },
            ];
            for header in bad_headers {
                let (_tx, rx) = crossbeam_channel::unbounded();
                assert!(
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).is_err()
                );
            }
        });
    }

    fn fill_smooth_photo(buf: &mut [u8], stride_px: usize, rect: TileRect, alpha: u8) {
        for ry in 0..rect.h as usize {
            for rx in 0..rect.w as usize {
                let x = rect.x as usize + rx;
                let y = rect.y as usize + ry;
                let off = (y * stride_px + x) * 4;
                // Smooth low-frequency RGB surface (> 10k unique colors, no 35% dominant color).
                let fx = rx as f64 / rect.w as f64;
                let fy = ry as f64 / rect.h as f64;
                let b = (30.0 + 180.0 * fx + 25.0 * (fy * std::f64::consts::TAU).sin())
                    .clamp(0.0, 255.0) as u8;
                let g = (40.0 + 160.0 * fy + 30.0 * (fx * std::f64::consts::TAU).cos())
                    .clamp(0.0, 255.0) as u8;
                let r = (120.0 + 110.0 * ((fx + fy) * std::f64::consts::PI).sin()).clamp(0.0, 255.0)
                    as u8;
                buf[off] = b;
                buf[off + 1] = g;
                buf[off + 2] = r;
                buf[off + 3] = alpha;
            }
        }
    }

    fn fill_ui_text(buf: &mut [u8], stride_px: usize, rect: TileRect, cleartype: bool) {
        for ry in 0..rect.h as usize {
            for rx in 0..rect.w as usize {
                let x = rect.x as usize + rx;
                let y = rect.y as usize + ry;
                let off = (y * stride_px + x) * 4;
                // Solid white background on >= 60% of pixels + crisp dark glyph strokes.
                if ry % 8 < 5 || rx % 6 < 3 {
                    buf[off..off + 4].copy_from_slice(&[250, 250, 250, 0xFF]);
                } else if cleartype {
                    // > 256 unique subpixel-antialiased fringe RGB colors on solid background.
                    buf[off] = ((rx * 5 + ry * 3) & 0x7F) as u8;
                    buf[off + 1] = ((rx * 7 + ry * 11) & 0x7F) as u8;
                    buf[off + 2] = ((rx * 13 + ry * 17) & 0x7F) as u8;
                    buf[off + 3] = 0xFF;
                } else {
                    buf[off..off + 4].copy_from_slice(&[20, 24, 30, 0xFF]);
                }
            }
        }
    }

    fn rect_psnr_rgb(orig: &[u8], dec: &[u8], stride_px: usize, rect: TileRect) -> f64 {
        let mut sse = 0u64;
        for ry in 0..rect.h as usize {
            for rx in 0..rect.w as usize {
                let off = ((rect.y as usize + ry) * stride_px + rect.x as usize + rx) * 4;
                for c in 0..3 {
                    let d = i32::from(orig[off + c]) - i32::from(dec[off + c]);
                    sse += (d * d) as u64;
                }
            }
        }
        if sse == 0 {
            return f64::INFINITY;
        }
        let n_ch = (rect.w as u64) * (rect.h as u64) * 3;
        10.0 * (255.0 * 255.0 / (sse as f64 / n_ch as f64)).log10()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hybrid_quadtree_merge_split_and_quality() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let hybrid_444 = CompressionMode::hybrid_default();

            // 1. All-text `256x256` (both <= 256 palette AND > 256 ClearType with >= 35% bg)
            //    must NOT split and must decode bit-exact (`max_err == 0`).
            let meta_256 = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Xrgb8888,
            };
            for cleartype in [false, true] {
                let mut text_buf = vec![0u8; 256 * 256 * 4];
                fill_ui_text(&mut text_buf, 256, tr(0, 0, 256, 256), cleartype);
                let ptr = text_buf.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, text_buf.len()) };
                let tiles = filter_and_compress(bp, meta_256, hybrid_444, compressor);
                assert_eq!(tiles.tiles.len(), 1);
                if cleartype {
                    assert_matches!(
                        tiles.tiles[0].payload,
                        TilePayload::Leaf(LeafPayload::AosZstd { .. })
                    );
                } else {
                    assert_matches!(
                        tiles.tiles[0].payload,
                        TilePayload::Leaf(LeafPayload::Palette { .. })
                    );
                }
                assert_decodes_at_both_alignments(&tiles, &text_buf, decompressor);
            }

            // 2. All-photo `256x256` must merge into ONE `LeafPayload::Jpeg` across all
            //    subsampling modes, preserve high PSNR, and write byte 3 == `0xFF` on Xrgb8888.
            let mut photo_buf = vec![0u8; 256 * 256 * 4];
            fill_smooth_photo(&mut photo_buf, 256, tr(0, 0, 256, 256), 0x12);
            for (subsamp, min_psnr) in [
                (JpegSubsamp::Sub444, 38.0),
                (JpegSubsamp::Sub422, 36.0),
                (JpegSubsamp::Sub420, 34.0),
            ] {
                let mode = CompressionMode::Hybrid {
                    quality: 90,
                    subsamp,
                };
                let ptr = photo_buf.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, photo_buf.len()) };
                let tiles = filter_and_compress(bp, meta_256, mode, compressor);
                assert_eq!(tiles.tiles.len(), 1);
                assert_matches!(
                    tiles.tiles[0].payload,
                    TilePayload::Leaf(LeafPayload::Jpeg {
                        alpha: AlphaChannel::Omitted,
                        ..
                    })
                );
                let mut out = vec![0x33u8; photo_buf.len()];
                decompress_and_unfilter(&tiles, decompressor, &mut out).unwrap();
                assert!(
                    out.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF),
                    "Xrgb8888 JPEG decode must set byte 3 to 0xFF ({subsamp:?})"
                );
                let psnr = rect_psnr_rgb(&photo_buf, &out, 256, tr(0, 0, 256, 256));
                assert!(
                    psnr >= min_psnr,
                    "expected PSNR >= {min_psnr} dB for {subsamp:?}, got {psnr:.2}"
                );
            }

            // 3. Mixed `256x256` tile (TL palette UI text, TR ClearType text, BL & BR photos)
            //    must emit `TilePayload::Split` of 4 quadrants where text quadrants decode
            //    BIT-EXACT and photo quadrants decode as JPEG with high PSNR through wire framing.
            let mut mixed = vec![0u8; 256 * 256 * 4];
            let q_tl = tr(0, 0, 128, 128);
            let q_tr = tr(128, 0, 128, 128);
            let q_bl = tr(0, 128, 128, 128);
            let q_br = tr(128, 128, 128, 128);
            fill_ui_text(&mut mixed, 256, q_tl, false);
            fill_ui_text(&mut mixed, 256, q_tr, true);
            fill_smooth_photo(&mut mixed, 256, q_bl, 0xFF);
            fill_smooth_photo(&mut mixed, 256, q_br, 0xFF);

            let ptr = mixed.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, mixed.len()) };
            let mut streaming_opt = None;
            filter_and_compress_full_streaming(
                bp,
                ClientId(1),
                WlSurfaceId(2),
                meta_256,
                hybrid_444,
                None,
                None,
                compressor,
                |s| streaming_opt = Some(s),
            );
            let mut wire = Vec::new();
            streaming_opt.unwrap().framed_write(&mut wire).unwrap();

            let mut rx_stream = None;
            let mut rdr = wire.as_slice();
            CompressedTileStream::read_from_stream(&mut rdr, |s| {
                rx_stream = Some(s);
                Ok(())
            })
            .unwrap();
            assert!(rdr.is_empty());
            let stream = rx_stream.unwrap();
            let wire_tiles: Vec<CompressedTile> = stream
                .tiles
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            assert_eq!(wire_tiles.len(), 1);
            let TilePayload::Split(subs) = &wire_tiles[0].payload else {
                panic!("expected TilePayload::Split for mixed 256x256 tile");
            };
            assert_eq!(subs.len(), 4);
            assert_matches!(subs[0].payload, LeafPayload::Palette { .. });
            assert_matches!(subs[1].payload, LeafPayload::AosZstd { .. });
            assert_matches!(subs[2].payload, LeafPayload::Jpeg { .. });
            assert_matches!(subs[3].payload, LeafPayload::Jpeg { .. });

            let mut mixed_out = vec![0x55u8; mixed.len()];
            decompress_and_unfilter(
                &CompressedTiles {
                    metadata: meta_256,
                    tile_rects: stream.header.tile_rects,
                    tiles: wire_tiles,
                },
                decompressor,
                &mut mixed_out,
            )
            .unwrap();
            assert!(rect_psnr_rgb(&mixed, &mixed_out, 256, q_tl).is_infinite());
            assert!(rect_psnr_rgb(&mixed, &mixed_out, 256, q_tr).is_infinite());
            assert!(rect_psnr_rgb(&mixed, &mixed_out, 256, q_bl) >= 38.0);
            assert!(rect_psnr_rgb(&mixed, &mixed_out, 256, q_br) >= 38.0);

            // 4. Non-splittable `<= 128x128` photo encodes as a single JPEG leaf;
            //    thin/small `< 16`px photos stay lossless bit-exact.
            for (w, h, expect_jpeg) in [(100i32, 80i32, true), (256, 12, false), (12, 256, false)] {
                let meta = BufferMetadata {
                    width: w,
                    height: h,
                    stride: w * 4,
                    format: BufferFormat::Xrgb8888,
                };
                let mut buf = vec![0u8; (w * h * 4) as usize];
                fill_smooth_photo(&mut buf, w as usize, tr(0, 0, w as u32, h as u32), 0xFF);
                let ptr = buf.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, buf.len()) };
                let tiles = filter_and_compress(bp, meta, hybrid_444, compressor);
                assert_eq!(tiles.tiles.len(), 1);
                assert_eq!(
                    matches!(
                        tiles.tiles[0].payload,
                        TilePayload::Leaf(LeafPayload::Jpeg { .. })
                    ),
                    expect_jpeg
                );
                if !expect_jpeg {
                    assert_decodes_at_both_alignments(&tiles, &buf, decompressor);
                }
            }

            // 5. Mixed edge tile (`256x132`): top-left text, top-right photo, bottom
            //    `128x4` slivers (`h < MIN_JPEG_DIM`) must stay lossless inside `Split`.
            let meta_edge = BufferMetadata {
                width: 256,
                height: 132,
                stride: 256 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut edge_buf = vec![0u8; 256 * 132 * 4];
            fill_ui_text(&mut edge_buf, 256, tr(0, 0, 128, 128), false);
            fill_smooth_photo(&mut edge_buf, 256, tr(128, 0, 128, 128), 0xFF);
            fill_smooth_photo(&mut edge_buf, 256, tr(0, 128, 256, 4), 0xFF);
            let ptr = edge_buf.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, edge_buf.len()) };
            let edge_tiles = filter_and_compress(bp, meta_edge, hybrid_444, compressor);
            let TilePayload::Split(edge_subs) = &edge_tiles.tiles[0].payload else {
                panic!("expected Split for mixed 256x132 tile");
            };
            assert_eq!(edge_subs.len(), 4);
            assert_matches!(edge_subs[0].payload, LeafPayload::Palette { .. });
            assert_matches!(edge_subs[1].payload, LeafPayload::Jpeg { .. });
            assert!(!matches!(edge_subs[2].payload, LeafPayload::Jpeg { .. }));
            assert!(!matches!(edge_subs[3].payload, LeafPayload::Jpeg { .. }));
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hybrid_argb_translucent_jpeg_bit_exact_alpha() {
        with_test_codec(|compressor, decompressor| {
            let meta = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Argb8888,
            };
            let mode = CompressionMode::hybrid_default();

            // Constant translucent alpha (`0x77`).
            let mut const_a = vec![0u8; 256 * 256 * 4];
            fill_smooth_photo(
                &mut const_a,
                256,
                TileRect {
                    x: 0,
                    y: 0,
                    w: 256,
                    h: 256,
                },
                0x77,
            );
            let ptr = const_a.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, const_a.len()) };
            let tiles = filter_and_compress(bp, meta, mode, compressor);
            assert_matches!(
                tiles.tiles[0].payload,
                TilePayload::Leaf(LeafPayload::Jpeg {
                    alpha: AlphaChannel::Constant(0x77),
                    ..
                })
            );
            let mut out = vec![0u8; const_a.len()];
            decompress_and_unfilter(&tiles, decompressor, &mut out).unwrap();
            assert!(out.as_chunks::<4>().0.iter().all(|px| px[3] == 0x77));
            assert!(
                rect_psnr_rgb(
                    &const_a,
                    &out,
                    256,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 256,
                        h: 256
                    }
                ) >= 38.0
            );

            // Varying alpha gradient on photo -> bit-exact alpha channel on every pixel.
            let mut var_a = const_a;
            for y in 0..256usize {
                for x in 0..256usize {
                    var_a[(y * 256 + x) * 4 + 3] = ((x + y * 3) & 0xFF) as u8;
                }
            }
            let ptr = var_a.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, var_a.len()) };
            let tiles = filter_and_compress(bp, meta, mode, compressor);
            assert_matches!(
                tiles.tiles[0].payload,
                TilePayload::Leaf(LeafPayload::Jpeg {
                    alpha: AlphaChannel::Uncompressed(_) | AlphaChannel::Zstd(_),
                    ..
                })
            );
            out.fill(0);
            decompress_and_unfilter(&tiles, decompressor, &mut out).unwrap();
            for (i, (got, exp)) in out
                .as_chunks::<4>()
                .0
                .iter()
                .zip(var_a.as_chunks::<4>().0.iter())
                .enumerate()
            {
                assert_eq!(got[3], exp[3], "alpha mismatch at pixel {i}");
            }
            assert!(
                rect_psnr_rgb(
                    &var_a,
                    &out,
                    256,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 256,
                        h: 256
                    }
                ) >= 38.0
            );
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hybrid_incremental_skip_zero_generation_loss() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let meta = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mode = CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            };
            let mut frame = vec![0u8; 256 * 256 * 4];
            fill_smooth_photo(&mut frame, 256, tr(0, 0, 256, 256), 0xFF);
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; frame.len()];

            // Initial full sync encodes JPEG and snapshots raw `wl_shm` into `shadow`.
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, frame.len()) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    mode,
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            let canvas_after_f1 = canvas.clone();

            // Unchanged frame emits 0 tiles -> zero generation loss.
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, frame.len()) };
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    mode,
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |_| panic!("unchanged hybrid frame must emit 0 tiles"),
                );
                assert!(diff.is_unchanged());
            }

            // Edit only top-left `128x128` quadrant -> only TL transmitted; untouched
            // quadrants keep exact Frame-1 decoded bytes with zero generation loss.
            fill_ui_text(&mut frame, 256, tr(0, 0, 128, 128), false);
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, frame.len()) };
                let mut streaming = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    mode,
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                assert_eq!(diff.scroll_rect, None);
                assert_eq!(diff.dirty_tiles, vec![tr(0, 0, 128, 128)]);
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            // TL text quadrant is bit-exact; untouched quadrants are byte-identical to Frame 1.
            assert!(rect_psnr_rgb(&frame, &canvas, 256, tr(0, 0, 128, 128)).is_infinite());
            for y in 0..256usize {
                for x in 0..256usize {
                    if x >= 128 || y >= 128 {
                        let off = (y * 256 + x) * 4;
                        assert_eq!(&canvas[off..off + 4], &canvas_after_f1[off..off + 4]);
                    }
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_jpeg_and_split_decoder_errors() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };

            // Encode a valid `32x32` JPEG payload.
            let mut small_photo = vec![0u8; 32 * 32 * 4];
            fill_smooth_photo(&mut small_photo, 32, tr(0, 0, 32, 32), 0xFF);
            let ptr = small_photo.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, small_photo.len()) };
            let encoded = filter_and_compress(
                bp,
                BufferMetadata {
                    width: 32,
                    height: 32,
                    stride: 32 * 4,
                    format: BufferFormat::Xrgb8888,
                },
                CompressionMode::hybrid_default(),
                compressor,
            );
            let TilePayload::Leaf(LeafPayload::Jpeg {
                data: valid_32x32_jpeg,
                ..
            }) = &encoded.tiles[0].payload
            else {
                panic!("expected 32x32 JPEG leaf");
            };

            // Mismatched JPEG dimensions (both larger `64x64` and smaller `16x16` target rects)
            // must fail loudly instead of writing partial pixels or overflowing.
            for dim in [64i32, 16i32] {
                let meta = BufferMetadata {
                    width: dim,
                    height: dim,
                    stride: dim * 4,
                    format: BufferFormat::Xrgb8888,
                };
                let mut out = vec![0u8; (dim * dim * 4) as usize];
                let bad = CompressedTiles {
                    metadata: meta,
                    tile_rects: vec![tr(0, 0, dim as u32, dim as u32)],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Jpeg {
                            data: valid_32x32_jpeg.clone(),
                            alpha: AlphaChannel::Omitted,
                        }),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out).is_err());
            }

            // Empty and corrupt JPEG payloads must fail.
            let meta_32 = BufferMetadata {
                width: 32,
                height: 32,
                stride: 32 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut out_32 = vec![0u8; 32 * 32 * 4];
            for bad_bytes in [Vec::new(), vec![0x00, 0x01, 0x02, 0x03]] {
                let bad = CompressedTiles {
                    metadata: meta_32,
                    tile_rects: vec![tr(0, 0, 32, 32)],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Leaf(LeafPayload::Jpeg {
                            data: bad_bytes,
                            alpha: AlphaChannel::Omitted,
                        }),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out_32).is_err());
            }

            // Invalid Split tiles passed directly to `decompress_and_unfilter` must fail.
            let meta_256 = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut out_256 = vec![0u8; 256 * 256 * 4];
            let solid = LeafPayload::Solid([1, 2, 3, 0xFF]);
            for (parent_rect, bad_subs) in [
                // Non-splittable `128x128`
                (
                    tr(0, 0, 128, 128),
                    vec![
                        SubTile {
                            rect: tr(0, 0, 64, 128),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(64, 0, 64, 128),
                            payload: solid.clone(),
                        },
                    ],
                ),
                // Count 1 (< 2)
                (
                    tr(0, 0, 256, 256),
                    vec![SubTile {
                        rect: tr(0, 0, 256, 256),
                        payload: solid.clone(),
                    }],
                ),
                // Count 5 (> 4)
                (
                    tr(0, 0, 256, 256),
                    vec![
                        SubTile {
                            rect: tr(0, 0, 128, 128),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(128, 0, 128, 128),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(0, 128, 128, 128),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(128, 128, 64, 128),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(192, 128, 64, 128),
                            payload: solid.clone(),
                        },
                    ],
                ),
                // Overlapping sub-rects whose summed area equals parent area
                (
                    tr(0, 0, 256, 256),
                    vec![
                        SubTile {
                            rect: tr(0, 0, 160, 256),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(64, 0, 96, 256),
                            payload: solid.clone(),
                        },
                    ],
                ),
                // Incomplete coverage
                (
                    tr(0, 0, 256, 256),
                    vec![
                        SubTile {
                            rect: tr(0, 0, 128, 256),
                            payload: solid.clone(),
                        },
                        SubTile {
                            rect: tr(128, 0, 127, 256),
                            payload: solid,
                        },
                    ],
                ),
            ] {
                let bad = CompressedTiles {
                    metadata: meta_256,
                    tile_rects: vec![parent_rect],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::Split(bad_subs),
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out_256).is_err());
            }
        });
    }

    /// Generates a deterministic, textured document buffer (`doc_h` rows tall,
    /// `width` px wide, padded to `stride`) where every row has rich unique
    /// text/UI-like patterns.
    fn make_textured_document(
        width: usize,
        doc_h: usize,
        stride: usize,
        format: BufferFormat,
    ) -> Vec<u8> {
        let mut doc = vec![0xAAu8; doc_h * stride];
        for y in 0..doc_h {
            let line_id = y / 18;
            let row_in_line = y % 18;
            for x in 0..width {
                let off = y * stride + x * 4;
                let bg_luma = if (line_id % 5) == 0 { 244u8 } else { 250u8 };
                let is_glyph = (3..=14).contains(&row_in_line)
                    && x >= 24
                    && x < width.saturating_sub(24)
                    && ((x.wrapping_mul(31) ^ y.wrapping_mul(17) ^ line_id.wrapping_mul(131)) % 7)
                        < 2;
                let (b, g, r) = if is_glyph {
                    (
                        ((line_id * 29 + x * 3) & 0x7F) as u8,
                        ((line_id * 17 + y * 5) & 0x7F) as u8,
                        ((line_id * 43 + x * 7) & 0x7F) as u8,
                    )
                } else {
                    (bg_luma, bg_luma.wrapping_sub(2), bg_luma.wrapping_sub(4))
                };
                let a = if format == BufferFormat::Xrgb8888 {
                    // Deliberately non-0xFF X-byte to verify Xrgb8888 ignores byte 3.
                    ((x.wrapping_mul(13) ^ y.wrapping_mul(7)) & 0xFF) as u8
                } else {
                    180u8.wrapping_add(((line_id * 11 + x / 16) & 0x3F) as u8)
                };
                doc[off] = b;
                doc[off + 1] = g;
                doc[off + 2] = r;
                doc[off + 3] = a;
            }
        }
        doc
    }

    fn blit_doc_window(dst: &mut [u8], doc: &[u8], stride: usize, rect: TileRect, doc_y0: usize) {
        let x_bytes = rect.x as usize * 4;
        let row_bytes = rect.w as usize * 4;
        for ry in 0..rect.h as usize {
            let src_off = (doc_y0 + ry) * stride + x_bytes;
            let dst_off = (rect.y as usize + ry) * stride + x_bytes;
            dst[dst_off..dst_off + row_bytes].copy_from_slice(&doc[src_off..src_off + row_bytes]);
        }
    }

    fn assert_canvas_matches_frame(
        canvas: &[u8],
        expected: &[u8],
        metadata: BufferMetadata,
        pad_sentinel: u8,
    ) {
        let w = metadata.width as usize;
        let h = metadata.height as usize;
        let stride = metadata.stride as usize;
        let vis = w * 4;
        for y in 0..h {
            let row = y * stride;
            for x in 0..w {
                let off = row + x * 4;
                let exp_a = if metadata.format == BufferFormat::Xrgb8888 {
                    0xFF
                } else {
                    expected[off + 3]
                };
                assert_eq!(
                    &canvas[off..off + 4],
                    &[expected[off], expected[off + 1], expected[off + 2], exp_a],
                    "pixel mismatch at ({x}, {y}) format={:?}",
                    metadata.format
                );
            }
            if stride > vis {
                assert!(
                    canvas[row + vis..row + stride]
                        .iter()
                        .all(|&b| b == pad_sentinel),
                    "stride padding modified on row {y}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_blit_full_window_up_and_down_and_wire_reduction() {
        with_test_codec(|compressor, decompressor| {
            let width = 1024usize;
            let height = 768usize;
            for (format, pad) in [
                (BufferFormat::Xrgb8888, 0usize),
                (BufferFormat::Xrgb8888, 16usize),
                (BufferFormat::Argb8888, 8usize),
            ] {
                let stride = width * 4 + pad;
                let len = height * stride;
                let meta = BufferMetadata {
                    width: width as i32,
                    height: height as i32,
                    stride: stride as i32,
                    format,
                };
                let doc = make_textured_document(width, height + 600, stride, format);
                let full_rect = TileRect {
                    x: 0,
                    y: 0,
                    w: width as u32,
                    h: height as u32,
                };

                let mut scroll_y = 200usize;
                let mut frame = vec![0xBBu8; len];
                blit_doc_window(&mut frame, &doc, stride, full_rect, scroll_y);

                let mut shadow = ShadowBuffer::new();
                let mut canvas = vec![0xCCu8; len];

                // Initial full frame.
                let full_wire_len = {
                    let ptr = frame.as_ptr();
                    let bp = unsafe { BufferPointer::new(&ptr, len) };
                    let mut streaming = None;
                    filter_and_compress_full_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(2),
                        meta,
                        CompressionMode::Lossless,
                        Some(&mut shadow),
                        None,
                        compressor,
                        |s| streaming = Some(s),
                    );
                    let mut wire = Vec::new();
                    streaming.unwrap().framed_write(&mut wire).unwrap();
                    let mut rx = None;
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                        rx = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                    decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                        .unwrap();
                    wire.len()
                };
                assert_canvas_matches_frame(&canvas, &frame, meta, 0xCC);

                // Scroll up (content moves up -> dy < 0) and down (content moves down -> dy > 0).
                for (delta_doc, expected_dy) in [(36isize, -36i32), (120, -120), (-120, 120)] {
                    scroll_y = (scroll_y as isize + delta_doc) as usize;
                    blit_doc_window(&mut frame, &doc, stride, full_rect, scroll_y);
                    // Perturb X byte on Xrgb8888 to confirm detector ignores it.
                    if format == BufferFormat::Xrgb8888 {
                        for y in 0..height {
                            for x in 0..width {
                                frame[y * stride + x * 4 + 3] ^= 0x5A;
                            }
                        }
                    }

                    let ptr = frame.as_ptr();
                    let bp = unsafe { BufferPointer::new(&ptr, len) };
                    let mut streaming = None;
                    let diff = filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(2),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        None,
                        compressor,
                        |s| streaming = Some(s),
                    );
                    let sr = diff
                        .scroll_rect
                        .expect("expected ScrollRect to be detected");
                    assert_eq!(sr.dy, expected_dy, "format={format:?} pad={pad}");

                    let mut wire = Vec::new();
                    streaming.unwrap().framed_write(&mut wire).unwrap();
                    assert!(
                        wire.len() * 5 < full_wire_len,
                        "expected >5x wire reduction for dy={expected_dy}: scroll={} B vs full={} B",
                        wire.len(),
                        full_wire_len
                    );

                    let mut rx = None;
                    let mut rdr = wire.as_slice();
                    CompressedTileStream::read_from_stream(&mut rdr, |s| {
                        rx = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    assert!(rdr.is_empty());
                    let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                    let wire_sr = validated.scroll_rect().unwrap();
                    assert_eq!(wire_sr, sr);
                    apply_scroll_rect_in_place(&mut canvas, stride, wire_sr);
                    decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                        .unwrap();
                    assert_canvas_matches_frame(&canvas, &frame, meta, 0xCC);
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_blit_split_pane_with_sidebar_header_and_scrollbar_overlay() {
        with_test_codec(|compressor, decompressor| {
            let width = 1280usize;
            let height = 800usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };

            let sidebar_w = 260usize;
            let header_h = 64usize;
            let footer_h = 32usize;
            let scrollbar_w = 16usize;
            let pane_rect = TileRect {
                x: sidebar_w as u32,
                y: header_h as u32,
                w: (width - sidebar_w - scrollbar_w) as u32,
                h: (height - header_h - footer_h) as u32,
            };

            let doc = make_textured_document(width, height + 500, stride, BufferFormat::Xrgb8888);

            let render_split_frame = |scroll_y: usize| -> Vec<u8> {
                let mut buf = vec![0u8; len];
                // Fixed header + sidebar + footer chrome with rich distinct pixels.
                for y in 0..height {
                    for x in 0..width {
                        let off = y * stride + x * 4;
                        buf[off] = ((x * 5 + y * 3) & 0xFF) as u8;
                        buf[off + 1] = ((x * 2 + y * 7 + 40) & 0xFF) as u8;
                        buf[off + 2] = ((x * 11 + y * 13 + 80) & 0xFF) as u8;
                        buf[off + 3] = 0xFF;
                    }
                }
                // Scrolling content pane.
                blit_doc_window(&mut buf, &doc, stride, pane_rect, scroll_y);

                // Fixed scrollbar track + moving thumb at the right edge.
                let thumb_y = header_h + (scroll_y / 3).min(pane_rect.h as usize - 80);
                for y in header_h..(height - footer_h) {
                    let is_thumb = (thumb_y..thumb_y + 80).contains(&y);
                    let px = if is_thumb {
                        [110, 110, 120, 0xFF]
                    } else {
                        [225, 225, 230, 0xFF]
                    };
                    for x in (width - scrollbar_w)..width {
                        let off = y * stride + x * 4;
                        buf[off..off + 4].copy_from_slice(&px);
                    }
                }

                // Fixed floating overlay badge inside the scrolling pane.
                for y in 350..410 {
                    for x in 700..760 {
                        let off = y * stride + x * 4;
                        buf[off..off + 4].copy_from_slice(&[20, 120, 230, 0xFF]);
                    }
                }
                buf
            };

            let mut scroll_y = 100usize;
            let frame0 = render_split_frame(scroll_y);
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0xEEu8; len];
            {
                let ptr = frame0.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            assert_canvas_matches_frame(&canvas, &frame0, meta, 0xEE);

            for (delta_doc, expected_dy) in [(48isize, -48i32), (-64, 64)] {
                scroll_y = (scroll_y as isize + delta_doc) as usize;
                let next = render_split_frame(scroll_y);
                let ptr = next.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let mut streaming = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let sr = diff
                    .scroll_rect
                    .expect("expected split-pane ScrollRect to be detected");
                assert_eq!(sr.dy, expected_dy);
                // Must stay strictly outside fixed sidebar and header/footer.
                assert!(sr.x >= sidebar_w as u32);
                assert!(sr.x + sr.width <= (width - scrollbar_w) as u32);
                assert!(sr.y >= header_h as u32);
                assert!(sr.y + sr.height <= (height - footer_h) as u32);

                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                apply_scroll_rect_in_place(&mut canvas, stride, validated.scroll_rect().unwrap());
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
                assert_canvas_matches_frame(&canvas, &next, meta, 0xEE);
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_blit_hybrid_textsafe_zero_generation_loss() {
        with_test_codec(|compressor, decompressor| {
            let width = 1024usize;
            let height = 768usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };
            let mode = CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            };

            // Build a tall mixed document: left half crisp UI text, right half smooth photo.
            let doc_h = height + 256;
            let mut doc = make_textured_document(width, doc_h, stride, BufferFormat::Xrgb8888);
            fill_smooth_photo(
                &mut doc,
                width,
                TileRect {
                    x: 512,
                    y: 0,
                    w: 512,
                    h: doc_h as u32,
                },
                0xFF,
            );

            let full_rect = TileRect {
                x: 0,
                y: 0,
                w: width as u32,
                h: height as u32,
            };
            let mut frame = vec![0u8; len];
            blit_doc_window(&mut frame, &doc, stride, full_rect, 0);
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len];

            // Frame 1: full Hybrid encode.
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    mode,
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            let canvas_f1 = canvas.clone();

            // Frame 2: scroll down by 64 rows (`dy = -64`).
            blit_doc_window(&mut frame, &doc, stride, full_rect, 64);
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let mut streaming = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    mode,
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let sr = diff.scroll_rect.expect("expected Hybrid ScrollRect");
                assert_eq!(sr.dy, -64);

                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                apply_scroll_rect_in_place(&mut canvas, stride, validated.scroll_rect().unwrap());
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }

            // Scrolled photo pixels (`x >= 512, y < 640`) must be BYTE-IDENTICAL to
            // Frame 1 shifted up by 64 rows -> zero generation loss!
            for y in 0..640usize {
                for x in 512..width {
                    let dst = (y * width + x) * 4;
                    let src = ((y + 64) * width + x) * 4;
                    assert_eq!(
                        &canvas[dst..dst + 4],
                        &canvas_f1[src..src + 4],
                        "generation loss on scrolled Hybrid photo at ({x},{y})"
                    );
                }
            }
            // Newly exposed bottom photo strip (`y = 704..768, x >= 512`) decodes with high PSNR.
            assert!(
                rect_psnr_rgb(
                    &frame,
                    &canvas,
                    width,
                    TileRect {
                        x: 512,
                        y: 704,
                        w: 512,
                        h: 64
                    }
                ) >= 34.0
            );
            // Left UI text half stays bit-exact (`PSNR == inf`).
            assert!(
                rect_psnr_rgb(
                    &frame,
                    &canvas,
                    width,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 512,
                        h: height as u32
                    }
                )
                .is_infinite()
            );
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_blit_non_scroll_no_false_positives_and_zero_tile_scroll() {
        with_test_codec(|compressor, decompressor| {
            let width = 640usize;
            let height = 512usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };

            // 1. Solid color fill (unchanged and solid red -> solid blue) -> None.
            let solid_a: Vec<u8> = [10u8, 20, 30, 0xFF].repeat(width * height);
            let solid_b: Vec<u8> = [40u8, 50, 60, 0xFF].repeat(width * height);
            for (prev, curr) in [(&solid_a, &solid_a), (&solid_a, &solid_b)] {
                let ptr = curr.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                assert_eq!(detect_scroll_rect(bp, prev, meta, None), None);
            }

            // 2. Static textured frame -> None.
            let doc = make_textured_document(width, height + 200, stride, BufferFormat::Xrgb8888);
            let f0 = doc[..len].to_vec();
            {
                let ptr = f0.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                assert_eq!(detect_scroll_rect(bp, &f0, meta, None), None);
            }

            // 3. Random noise (unrelated full-frame repaint) -> None.
            let mut noise = vec![0u8; len];
            let mut prng = 0x9E37_79B9_7F4A_7C15u64;
            for chunk in noise.as_chunks_mut::<8>().0 {
                prng = prng.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = prng;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                chunk.copy_from_slice(&z.to_le_bytes());
            }
            {
                let ptr = noise.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                assert_eq!(detect_scroll_rect(bp, &f0, meta, None), None);
            }

            // 4. Small typing update (200x24 box) -> None.
            let mut typed = f0.clone();
            for y in 100..124 {
                for x in 80..280 {
                    let off = y * stride + x * 4;
                    typed[off] ^= 0x55;
                    typed[off + 1] ^= 0x33;
                }
            }
            {
                let ptr = typed.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                assert_eq!(detect_scroll_rect(bp, &f0, meta, None), None);
            }

            // 5. Partial RowBand shorter than 256 rows -> None even if buffer scrolled.
            let f_scrolled = doc[32 * stride..32 * stride + len].to_vec();
            {
                let ptr = f_scrolled.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let short_band = [RowBand { y0: 100, y1: 300 }];
                assert_eq!(detect_scroll_rect(bp, &f0, meta, Some(&short_band)), None);
                // Small dimensions `< 256` -> None.
                let small_meta = BufferMetadata {
                    width: 240,
                    height: 512,
                    stride: 240 * 4,
                    format: BufferFormat::Xrgb8888,
                };
                assert_eq!(detect_scroll_rect(bp, &f0, small_meta, None), None);
            }

            // 6. Zero-dirty-tile scroll: newly exposed bottom rows match the old
            //    bottom rows (solid white footer below scrolled text), so after
            //    ScrollRect is applied `dirty_tiles` is EMPTY — frame must still be
            //    emitted and reconstruct bit-exactly!
            let mut z_prev = vec![0xFFu8; len];
            // Put unique text in rows 64..448, solid white in 0..64 and 448..512.
            z_prev[64 * stride..448 * stride].copy_from_slice(&doc[0..384 * stride]);
            let mut z_curr = vec![0xFFu8; len];
            // Scroll up by 32 rows: text moves to 32..416; rows 416..512 are all white
            // (matching rows 448..512 in z_prev!).
            z_curr[32 * stride..416 * stride].copy_from_slice(&doc[0..384 * stride]);

            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0xFFu8; len];
            {
                let ptr = z_prev.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            let ptr = z_curr.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, len) };
            let mut streaming = None;
            let diff = filter_and_compress_incremental_streaming(
                bp,
                ClientId(1),
                WlSurfaceId(2),
                meta,
                CompressionMode::Lossless,
                None,
                &mut shadow,
                None,
                compressor,
                |s| streaming = Some(s),
            );
            assert!(!diff.is_unchanged());
            let sr = diff.scroll_rect.expect("expected ScrollRect");
            assert_eq!(sr.dy, -32);
            let mut wire = Vec::new();
            streaming.unwrap().framed_write(&mut wire).unwrap();
            let mut rx = None;
            CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                rx = Some(s);
                Ok(())
            })
            .unwrap();
            let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
            apply_scroll_rect_in_place(&mut canvas, stride, validated.scroll_rect().unwrap());
            decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None).unwrap();
            assert_canvas_matches_frame(&canvas, &z_curr, meta, 0xFF);
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_blit_parallel_replay_and_partial_row_bands() {
        with_test_codec(|compressor, decompressor| {
            // 1. Parallel column-stripe replay (`>= 512 KiB`) must match the
            //    serial implementation bit-for-bit across full-width,
            //    split-pane, padded-stride, up (`dy < 0`), and down (`dy > 0`)
            //    scrolls.
            let width = 1024usize;
            let height = 768usize;
            let stride = width * 4 + 32;
            let base = make_textured_document(width, height, stride, BufferFormat::Argb8888);
            for sr in [
                ScrollRect {
                    x: 0,
                    y: 0,
                    width: width as u32,
                    height: height as u32,
                    dy: -73,
                },
                ScrollRect {
                    x: 0,
                    y: 0,
                    width: width as u32,
                    height: height as u32,
                    dy: 91,
                },
                ScrollRect {
                    x: 137,
                    y: 48,
                    width: 811,
                    height: 680,
                    dy: -45,
                },
                ScrollRect {
                    x: 137,
                    y: 48,
                    width: 811,
                    height: 680,
                    dy: 61,
                },
            ] {
                let mut serial = base.clone();
                let mut parallel = base.clone();
                apply_scroll_rect_in_place(&mut serial, stride, sr);
                apply_scroll_rect_in_place_parallel(&mut parallel, stride, sr, decompressor);
                assert_eq!(parallel, serial, "parallel mismatch for {sr:?}");
            }

            // 2. Partial RowBands combining a `< 8`-row `Strip1D` top band and
            //    a scrolling `>= 256`-row `RowBand2D` band in the same
            //    incremental commit (tests mixed outside-scroll + fused
            //    scroll-column worker tasks).
            let stride0 = width * 4;
            let len0 = height * stride0;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride0 as i32,
                format: BufferFormat::Argb8888,
            };
            let doc = make_textured_document(width, height + 300, stride0, BufferFormat::Argb8888);
            let mut frame = doc[..len0].to_vec();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len0];
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len0) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(7),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }

            let bands = [RowBand { y0: 0, y1: 4 }, RowBand { y0: 64, y1: 640 }];
            for (delta_doc, expected_dy) in [(48isize, -48i32), (-48, 48)] {
                // Mutate the `< 8`-row top strip band.
                for y in 0..4 {
                    for x in 0..160 {
                        let off = y * stride0 + x * 4;
                        frame[off] ^= 0x5A;
                        frame[off + 2] ^= 0xA5;
                    }
                }
                // Scroll only rows `64..640` (within the second band).
                let doc_y0 = if delta_doc > 0 { 64 + 48usize } else { 64usize };
                let band_rect = TileRect {
                    x: 120,
                    y: 64,
                    w: width as u32 - 140,
                    h: 640 - 64,
                };
                blit_doc_window(&mut frame, &doc, stride0, band_rect, doc_y0);

                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, len0) };
                let mut streaming = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(7),
                    meta,
                    CompressionMode::Lossless,
                    Some(&bands),
                    &mut shadow,
                    None,
                    compressor,
                    |s| streaming = Some(s),
                );
                let sr = diff
                    .scroll_rect
                    .expect("expected ScrollRect inside partial RowBand");
                assert_eq!(sr.dy, expected_dy);
                assert!(sr.y >= 64 && sr.y + sr.height <= 640);
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                apply_scroll_rect_in_place_parallel(
                    &mut canvas,
                    stride0,
                    validated.scroll_rect().unwrap(),
                    decompressor,
                );
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
                assert_canvas_matches_frame(&canvas, &frame, meta, 0);
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_cache_aba_hit_rate_xrgb_argb_stride_edge_and_4k_two_screens() {
        with_test_codec(|compressor, decompressor| {
            let run_roundtrip = |streaming: StreamingCompressedTiles,
                                 decomp: &mut ShardingDecompressor,
                                 canvas: &mut [u8],
                                 client_cache: &mut ClientTileCache|
             -> (usize, usize, usize) {
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let mut rdr = wire.as_slice();
                let mut rx_stream = None;
                CompressedTileStream::read_from_stream(&mut rdr, |s| {
                    rx_stream = Some(s);
                    Ok(())
                })
                .unwrap();
                assert!(rdr.is_empty());
                let stream = rx_stream.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let n_hits = tiles
                    .iter()
                    .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                    .count();
                let n_tiles = tiles.len();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in tiles {
                    tx.send(Ok(t)).unwrap();
                }
                drop(tx);
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).unwrap();
                decompress_and_unfilter_streaming(validated, decomp, canvas, Some(client_cache))
                    .unwrap();
                (wire.len(), n_hits, n_tiles)
            };

            // 1. A -> B -> A with Xrgb8888 (varying X padding) and Argb8888,
            //    padded stride, and right/bottom `< 128px` edge tiles.
            let width = 540usize;
            let height = 410usize;
            for (format, pad) in [
                (BufferFormat::Xrgb8888, 16usize),
                (BufferFormat::Argb8888, 8usize),
            ] {
                let stride = width * 4 + pad;
                let len = height * stride;
                let meta = BufferMetadata {
                    width: width as i32,
                    height: height as i32,
                    stride: stride as i32,
                    format,
                };
                let doc_a = make_textured_document(width, height, stride, format);
                let mut doc_b = vec![0xBBu8; len];
                for y in 0..height {
                    for x in 0..width {
                        let off = y * stride + x * 4;
                        let tile_salt = (x >> 8) * 37 + (y >> 8) * 53;
                        doc_b[off] = ((x * 7 + y * 11 + tile_salt + 33) & 0xFF) as u8;
                        doc_b[off + 1] = ((x * 13 + y * 5 + tile_salt * 3 + 77) & 0xFF) as u8;
                        doc_b[off + 2] = ((x * 3 + y * 19 + tile_salt * 5 + 123) & 0xFF) as u8;
                        doc_b[off + 3] = ((x + y * 3 + tile_salt) & 0xFF) as u8;
                    }
                }

                let mut server_cache = ServerTileCache::new();
                let mut client_cache = ClientTileCache::new();
                let mut shadow = ShadowBuffer::new();
                let mut canvas = vec![0xCCu8; len];

                // Frame A (initial full sync populates cache).
                let (wire_a1, hits_a1, tiles_a1) = {
                    let ptr = doc_a.as_ptr();
                    let bp = unsafe { BufferPointer::new(&ptr, len) };
                    let mut streaming = None;
                    filter_and_compress_full_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(10),
                        meta,
                        CompressionMode::Lossless,
                        Some(&mut shadow),
                        Some(&mut server_cache),
                        compressor,
                        |s| streaming = Some(s),
                    );
                    run_roundtrip(
                        streaming.unwrap(),
                        decompressor,
                        &mut canvas,
                        &mut client_cache,
                    )
                };
                assert_eq!(hits_a1, 0);
                assert!(tiles_a1 > 0);
                assert_canvas_matches_frame(&canvas, &doc_a, meta, 0xCC);

                // Frame B (tab switch to B).
                let (_, hits_b, _) = {
                    let ptr = doc_b.as_ptr();
                    let bp = unsafe { BufferPointer::new(&ptr, len) };
                    let mut streaming = None;
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(10),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut server_cache),
                        compressor,
                        |s| streaming = Some(s),
                    );
                    run_roundtrip(
                        streaming.unwrap(),
                        decompressor,
                        &mut canvas,
                        &mut client_cache,
                    )
                };
                assert_eq!(hits_b, 0);
                assert_canvas_matches_frame(&canvas, &doc_b, meta, 0xCC);

                // Frame A again (perturb X byte on Xrgb8888 to prove X padding is ignored).
                let mut doc_a_ret = doc_a.clone();
                if format == BufferFormat::Xrgb8888 {
                    for y in 0..height {
                        for x in 0..width {
                            doc_a_ret[y * stride + x * 4 + 3] ^= 0xA5;
                        }
                    }
                }
                let (wire_a2, hits_a2, tiles_a2) = {
                    let ptr = doc_a_ret.as_ptr();
                    let bp = unsafe { BufferPointer::new(&ptr, len) };
                    let mut streaming = None;
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(10),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut server_cache),
                        compressor,
                        |s| streaming = Some(s),
                    );
                    run_roundtrip(
                        streaming.unwrap(),
                        decompressor,
                        &mut canvas,
                        &mut client_cache,
                    )
                };
                assert_eq!(
                    hits_a2, tiles_a2,
                    "expected 100% cache hit on return-to-A ({format:?})"
                );
                assert!(
                    wire_a2 * 10 < wire_a1,
                    "expected >10x wire reduction on A->B->A: {wire_a2} B vs {wire_a1} B"
                );
                assert_canvas_matches_frame(&canvas, &doc_a_ret, meta, 0xCC);

                // On Argb8888, flipping alpha in one pixel after switching back via B
                // must miss that quadrant while all other tiles still hit.
                if format == BufferFormat::Argb8888 {
                    let ptr_b = doc_b.as_ptr();
                    let bp_b = unsafe { BufferPointer::new(&ptr_b, len) };
                    let mut s_b = None;
                    filter_and_compress_incremental_streaming(
                        bp_b,
                        ClientId(1),
                        WlSurfaceId(10),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut server_cache),
                        compressor,
                        |s| s_b = Some(s),
                    );
                    run_roundtrip(s_b.unwrap(), decompressor, &mut canvas, &mut client_cache);

                    let mut doc_a_alpha = doc_a.clone();
                    doc_a_alpha[5 * stride + 5 * 4 + 3] ^= 0x40;
                    let ptr_aa = doc_a_alpha.as_ptr();
                    let bp_aa = unsafe { BufferPointer::new(&ptr_aa, len) };
                    let mut s_aa = None;
                    filter_and_compress_incremental_streaming(
                        bp_aa,
                        ClientId(1),
                        WlSurfaceId(10),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut server_cache),
                        compressor,
                        |s| s_aa = Some(s),
                    );
                    let (_, hits_aa, tiles_aa) =
                        run_roundtrip(s_aa.unwrap(), decompressor, &mut canvas, &mut client_cache);
                    assert_eq!(hits_aa + 1, tiles_aa);
                    assert_canvas_matches_frame(&canvas, &doc_a_alpha, meta, 0xCC);
                }
            }

            // 2. Two full 4K screens (`3840x2160` A -> B -> A) fit cleanly inside
            //    `MAX_TILE_CACHE_SLOTS = 1024` and `MAX_TILE_CACHE_BYTES = 64 MiB`
            //    with 100% cache hit rate on return-to-A!
            let w4k = 3840usize;
            let h4k = 2160usize;
            let stride4k = w4k * 4;
            let len4k = h4k * stride4k;
            let meta4k = BufferMetadata {
                width: w4k as i32,
                height: h4k as i32,
                stride: stride4k as i32,
                format: BufferFormat::Xrgb8888,
            };
            let mut tab_a = vec![0u8; len4k];
            let mut tab_b = vec![0u8; len4k];
            for ty in 0..h4k.div_ceil(128) {
                for tx in 0..w4k.div_ceil(128) {
                    let id = (ty * 32 + tx) as u32;
                    let px_a = [(id & 0xFF) as u8, ((id >> 8) & 0xFF) as u8, 0x10, 0xFF];
                    let px_b = [(id & 0xFF) as u8, ((id >> 8) & 0xFF) as u8, 0x90, 0xFF];
                    let y0 = ty * 128;
                    let y1 = (y0 + 128).min(h4k);
                    let x0 = tx * 128;
                    let x1 = (x0 + 128).min(w4k);
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let off = y * stride4k + x * 4;
                            tab_a[off..off + 4].copy_from_slice(&px_a);
                            tab_b[off..off + 4].copy_from_slice(&px_b);
                        }
                    }
                }
            }
            let mut srv4k = ServerTileCache::new();
            let mut cli4k = ClientTileCache::new();
            let mut shadow4k = ShadowBuffer::new();
            let mut canvas4k = vec![0u8; len4k];
            {
                let p = tab_a.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len4k) };
                let mut s = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(40),
                    meta4k,
                    CompressionMode::Lossless,
                    Some(&mut shadow4k),
                    Some(&mut srv4k),
                    compressor,
                    |st| s = Some(st),
                );
                let (_, hits, tiles) =
                    run_roundtrip(s.unwrap(), decompressor, &mut canvas4k, &mut cli4k);
                assert_eq!(hits, 0);
                assert_eq!(tiles, 15 * 9);
            }
            {
                let p = tab_b.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len4k) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(40),
                    meta4k,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow4k,
                    Some(&mut srv4k),
                    compressor,
                    |st| s = Some(st),
                );
                let (_, hits, tiles) =
                    run_roundtrip(s.unwrap(), decompressor, &mut canvas4k, &mut cli4k);
                assert_eq!(hits, 0);
                assert_eq!(tiles, 15 * 9);
            }
            assert!(srv4k.occupied_slots() <= MAX_TILE_CACHE_SLOTS);
            assert!(cli4k.used_bytes() <= MAX_TILE_CACHE_BYTES);
            assert_eq!(srv4k.used_bytes(), cli4k.used_bytes());
            {
                let p = tab_a.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len4k) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(40),
                    meta4k,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow4k,
                    Some(&mut srv4k),
                    compressor,
                    |st| s = Some(st),
                );
                let (_, hits, tiles) =
                    run_roundtrip(s.unwrap(), decompressor, &mut canvas4k, &mut cli4k);
                assert_eq!(hits, 15 * 9, "4K A->B->A must hit 100% of all 135 tiles");
                assert_eq!(tiles, 15 * 9);
                assert_eq!(canvas4k, tab_a);
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_cache_quadrant_subregion_hits_and_uncoalescing() {
        use crate::serialization::framing::CacheHitRegion;

        with_test_codec(|compressor, decompressor| {
            let width = 384usize;
            let height = 512usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };
            let base = make_textured_document(width, height, stride, BufferFormat::Xrgb8888);

            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len];

            let apply = |streaming: StreamingCompressedTiles,
                         decomp: &mut ShardingDecompressor,
                         canvas: &mut [u8],
                         cli: &mut ClientTileCache|
             -> Vec<CompressedTile> {
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).unwrap();
                decompress_and_unfilter_streaming(validated, decomp, canvas, Some(cli)).unwrap();
                tiles
            };

            // Frame 0: full frame caches four `256x256`/`128x256` parent slots.
            {
                let p = base.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                let tiles = apply(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(tiles.len(), 4);
                assert_canvas_matches_frame(&canvas, &base, meta, 0);
            }

            // Frame 1: open popup covering 1 quadrant from each of the 4 parent tiles
            // (including BottomLeft of the vertical-only 128x256 tile at `(256, 128)`!).
            let mut popup = base.clone();
            for y in 140..360 {
                for x in 140..360 {
                    let off = y * stride + x * 4;
                    popup[off..off + 4].copy_from_slice(&[
                        ((x + y) & 0xFF) as u8,
                        0x40,
                        0xC0,
                        0xFF,
                    ]);
                }
            }
            {
                let p = popup.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                let tiles = apply(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(tiles.len(), 4);
                assert!(tiles.iter().all(|t| t.cache_put.is_some()));
                assert_canvas_matches_frame(&canvas, &popup, meta, 0);
            }

            // Frame 2: close popup -> all 4 restored `128x128` quadrants MUST hit
            // sub-regions of Frame 0's cached parent slots immediately on first return!
            {
                let p = base.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                let tiles = apply(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(tiles.len(), 4);
                let mut regions = Vec::new();
                for t in &tiles {
                    match &t.payload {
                        TilePayload::CacheHit(href) => regions.push(href.region),
                        other => panic!("expected quadrant CacheHit on popup close: {other:?}"),
                    }
                }
                assert!(regions.contains(&CacheHitRegion::BottomRight));
                assert!(regions.contains(&CacheHitRegion::BottomLeft));
                assert!(regions.contains(&CacheHitRegion::TopRight));
                assert!(regions.contains(&CacheHitRegion::TopLeft));
                assert_canvas_matches_frame(&canvas, &base, meta, 0);
            }

            // Frame 3: partial-hit un-coalescing! Mutate all 4 quadrants of `(0,0,256,256)`
            // where BR matches the cached popup quadrant and TL/TR/BL are brand new ->
            // `plan_frame` must un-coalesce into 1 `CacheHit` + 3 `CachePut` quadrants.
            let mut mixed = base;
            for y in 0..128 {
                for x in 0..256 {
                    let off = y * stride + x * 4;
                    mixed[off] ^= 0x55;
                }
            }
            for y in 128..256 {
                for x in 0..128 {
                    let off = y * stride + x * 4;
                    mixed[off + 1] ^= 0x77;
                }
            }
            for y in 128..256 {
                for x in 128..256 {
                    let off = y * stride + x * 4;
                    mixed[off..off + 4].copy_from_slice(&popup[off..off + 4]);
                }
            }
            {
                let p = mixed.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                assert_eq!(diff.dirty_tiles.len(), 4);
                let tiles = apply(s.unwrap(), decompressor, &mut canvas, &mut cli);
                let hits = tiles
                    .iter()
                    .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                    .count();
                let puts = tiles.iter().filter(|t| t.cache_put.is_some()).count();
                assert_eq!((hits, puts), (1, 3));
                assert_canvas_matches_frame(&canvas, &mixed, meta, 0);
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_cache_intraframe_dedup_pinning_eviction_and_byte_cap() {
        with_test_codec(|compressor, decompressor| {
            let apply_full = |frame: &[u8],
                              meta: BufferMetadata,
                              srv: &mut ServerTileCache,
                              cli: &mut ClientTileCache,
                              comp: &mut ShardingCompressor,
                              decomp: &mut ShardingDecompressor|
             -> (RawBufferHeader, Vec<CompressedTile>, Vec<u8>) {
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, frame.len()) };
                let mut streaming = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(5),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    Some(srv),
                    comp,
                    |s| streaming = Some(s),
                );
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated = ValidatedTileStream::new(CompressedTileStream {
                    header: header.clone(),
                    tiles: rx,
                })
                .unwrap();
                let mut out = vec![0xAAu8; frame.len()];
                decompress_and_unfilter_streaming(validated, decomp, &mut out, Some(cli)).unwrap();
                (header, tiles, out)
            };

            // 1. Intra-frame dedup: `1024x256` frame with 4 identical `256x256` tiles ->
            //    exactly 1 tile is `CachePut(0)`, other 3 are `CacheHit(0)` in the SAME frame!
            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let meta_dup = BufferMetadata {
                width: 1024,
                height: 256,
                stride: 1024 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let tile_pat = make_textured_document(256, 256, 256 * 4, BufferFormat::Xrgb8888);
            let mut dup_frame = vec![0u8; 1024 * 256 * 4];
            for y in 0..256 {
                for rep in 0..4 {
                    let dst = y * 1024 * 4 + rep * 256 * 4;
                    let src = y * 256 * 4;
                    dup_frame[dst..dst + 256 * 4].copy_from_slice(&tile_pat[src..src + 256 * 4]);
                }
            }
            let (_, mut tiles, out) = apply_full(
                &dup_frame,
                meta_dup,
                &mut srv,
                &mut cli,
                compressor,
                decompressor,
            );
            tiles.sort_by_key(|t| t.tile_idx);
            assert_eq!(tiles.len(), 4);
            let put_tiles: Vec<&CompressedTile> =
                tiles.iter().filter(|t| t.cache_put.is_some()).collect();
            assert_eq!(put_tiles.len(), 1, "exactly 1 tile must be CachePut");
            let put_slot = put_tiles[0].cache_put.unwrap();
            assert_eq!(put_slot, 0, "first cold-fill slot must be 0");
            let hit_tiles: Vec<&CompressedTile> =
                tiles.iter().filter(|t| t.cache_put.is_none()).collect();
            assert_eq!(hit_tiles.len(), 3);
            for t in hit_tiles {
                assert_matches!(
                    t.payload,
                    TilePayload::CacheHit(href) if href.slot_id == put_slot
                );
            }
            assert_eq!(srv.occupied_slots(), 1);
            assert_eq!(cli.occupied_slots(), 1);
            assert_canvas_matches_frame(&out, &dup_frame, meta_dup, 0);

            // Repeat direct concurrent cold-fill encode on an 8-tile mixed frame
            // (5 unique + 3 duplicates) across parallel worker schedules: each
            // equivalence class must yield exactly 1 CachePut + matching
            // CacheHits, unique slots `0..5`, and bit-exact decode.
            let meta_det = BufferMetadata {
                width: 1024,
                height: 512,
                stride: 1024 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut det_frame = make_textured_document(1024, 512, 1024 * 4, BufferFormat::Xrgb8888);
            // Copy tile (0,0) into (512,0) and (256,256), and tile (256,0) into (768,256).
            for y in 0..256 {
                let row0 = y * 1024 * 4;
                let row1 = (y + 256) * 1024 * 4;
                let t0: Vec<u8> = det_frame[row0..row0 + 256 * 4].to_vec();
                let t1: Vec<u8> = det_frame[row0 + 256 * 4..row0 + 512 * 4].to_vec();
                det_frame[row0 + 512 * 4..row0 + 768 * 4].copy_from_slice(&t0);
                det_frame[row1 + 256 * 4..row1 + 512 * 4].copy_from_slice(&t0);
                det_frame[row1 + 768 * 4..row1 + 1024 * 4].copy_from_slice(&t1);
            }
            for _ in 0..4 {
                srv.clear();
                cli.clear();
                let (_, mut det_tiles, det_out) = apply_full(
                    &det_frame,
                    meta_det,
                    &mut srv,
                    &mut cli,
                    compressor,
                    decompressor,
                );
                det_tiles.sort_by_key(|t| t.tile_idx);
                assert_eq!(det_tiles.len(), 8);
                assert_eq!(srv.occupied_slots(), 5);
                assert_eq!(cli.occupied_slots(), 5);

                let mut used_slots = Vec::new();
                let check_group = |group: &[usize], used: &mut Vec<u16>| {
                    let mut group_put = None;
                    for &idx in group {
                        if let Some(s) = det_tiles[idx].cache_put {
                            assert!(group_put.is_none(), "duplicate Put in equivalence group");
                            group_put = Some(s);
                            used.push(s);
                        }
                    }
                    let slot = group_put.expect("equivalence group must have 1 CachePut");
                    for &idx in group {
                        if det_tiles[idx].cache_put.is_none() {
                            assert_matches!(
                                det_tiles[idx].payload,
                                TilePayload::CacheHit(href) if href.slot_id == slot
                            );
                        }
                    }
                };
                check_group(&[0, 2, 5], &mut used_slots);
                check_group(&[1, 7], &mut used_slots);
                check_group(&[3], &mut used_slots);
                check_group(&[4], &mut used_slots);
                check_group(&[6], &mut used_slots);
                used_slots.sort_unstable();
                assert_eq!(used_slots, vec![0, 1, 2, 3, 4]);
                assert_canvas_matches_frame(&det_out, &det_frame, meta_det, 0);
            }

            // 2. Eviction + byte-cap + pinning stress:
            //    Fill all 1024 slots with unique `128x128` (64 KiB) tiles = exact 64 MiB!
            //    Then emit a frame referencing the very oldest `128x128` tile (pinned by Pass 1!)
            //    plus new `256x256` (256 KiB) tiles -> forces multi-slot eviction (`evict_slots`),
            //    preserves pinned slot, and keeps `used_bytes <= MAX_TILE_CACHE_BYTES`.
            srv.clear();
            cli.clear();
            let meta_128 = BufferMetadata {
                width: 128,
                height: 128,
                stride: 128 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let make_unique_128 = |id: u32| -> Vec<u8> {
                let px = [
                    (id & 0xFF) as u8,
                    ((id >> 8) & 0xFF) as u8,
                    ((id >> 16) & 0xFF) as u8,
                    0xFF,
                ];
                let mut buf = px.repeat(128 * 128);
                // Perturb pixel 0 so the tile is non-solid and eligible for caching.
                buf[0] ^= 0xFF;
                buf[1] ^= 0x55;
                buf[2] ^= 0xAA;
                buf
            };
            for id in 0..MAX_TILE_CACHE_SLOTS as u32 {
                let f = make_unique_128(id);
                let (_, _, out) =
                    apply_full(&f, meta_128, &mut srv, &mut cli, compressor, decompressor);
                assert_eq!(out, f);
            }
            assert_eq!(srv.occupied_slots(), MAX_TILE_CACHE_SLOTS);
            assert_eq!(srv.used_bytes(), MAX_TILE_CACHE_BYTES);
            assert_eq!(cli.occupied_slots(), MAX_TILE_CACHE_SLOTS);
            assert_eq!(cli.used_bytes(), MAX_TILE_CACHE_BYTES);

            let meta_ev = BufferMetadata {
                width: 768,
                height: 256,
                stride: 768 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut ev_frame = vec![0u8; 768 * 256 * 4];
            let paint_box =
                |buf: &mut [u8], x0: usize, y0: usize, w: usize, h: usize, px: [u8; 4]| {
                    for y in y0..y0 + h {
                        for x in x0..x0 + w {
                            let off = (y * 768 + x) * 4;
                            buf[off..off + 4].copy_from_slice(&px);
                        }
                    }
                    let off0 = (y0 * 768 + x0) * 4;
                    buf[off0] ^= 0xFF;
                    buf[off0 + 1] ^= 0x55;
                    buf[off0 + 2] ^= 0xAA;
                };
            paint_box(&mut ev_frame, 0, 0, 128, 128, [0, 0, 0, 0xFF]); // oldest id=0!
            paint_box(&mut ev_frame, 128, 0, 128, 128, [1, 2, 200, 0xFF]);
            paint_box(&mut ev_frame, 0, 128, 128, 128, [2, 2, 200, 0xFF]);
            paint_box(&mut ev_frame, 128, 128, 128, 128, [3, 2, 200, 0xFF]);
            paint_box(&mut ev_frame, 256, 0, 256, 256, [4, 2, 200, 0xFF]);
            paint_box(&mut ev_frame, 512, 0, 256, 256, [5, 2, 200, 0xFF]);
            let (hdr_ev, tiles_ev, out_ev) = apply_full(
                &ev_frame,
                meta_ev,
                &mut srv,
                &mut cli,
                compressor,
                decompressor,
            );
            assert!(
                !hdr_ev.evict_slots.is_empty(),
                "replacing 64 KiB slots with 256 KiB slots must emit explicit evict_slots"
            );
            assert!(
                tiles_ev
                    .iter()
                    .any(|t| matches!(t.payload, TilePayload::CacheHit(_))),
                "oldest slot id=0 must be pinned and hit in Pass 1"
            );
            assert!(srv.used_bytes() <= MAX_TILE_CACHE_BYTES);
            assert_eq!(srv.used_bytes(), cli.used_bytes());
            assert_eq!(out_ev, ev_frame);

            // 3. Single giant frame with >1024 unique tiles gracefully emits `Uncached`
            //    for overflow tiles without evicting pinned slots!
            srv.clear();
            let mut entries = Vec::with_capacity(MAX_TILE_CACHE_SLOTS + 6);
            let mut scans = Vec::with_capacity(MAX_TILE_CACHE_SLOTS + 6);
            for i in 0..(MAX_TILE_CACHE_SLOTS + 6) {
                entries.push(BaseScanEntry::Strip1D {
                    rect: TileRect {
                        x: 0,
                        y: i as u32,
                        w: 16,
                        h: 1,
                    },
                    slot_idx: i,
                });
                let mut h = [0u8; 16];
                h[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
                scans.push(tile_hash::TileHashResult {
                    hash: h,
                    meta: RectScanMeta::UNKNOWN,
                });
            }
            let plan = srv.plan_frame(&entries, None, &scans, false, CompressionMode::Lossless);
            assert_eq!(plan.actions.len(), MAX_TILE_CACHE_SLOTS + 6);
            let n_put = plan
                .actions
                .iter()
                .filter(|a| matches!(a, TileCacheAction::Put(_)))
                .count();
            let n_uncached = plan
                .actions
                .iter()
                .filter(|a| matches!(a, TileCacheAction::Uncached))
                .count();
            assert_eq!(n_put, MAX_TILE_CACHE_SLOTS);
            assert_eq!(n_uncached, 6);
            assert_eq!(srv.occupied_slots(), MAX_TILE_CACHE_SLOTS);
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_cache_hybrid_lossless_lossy_quality_isolation() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let meta = BufferMetadata {
                width: 512,
                height: 512,
                stride: 512 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let len = 512 * 512 * 4;
            let mut frame_a = vec![0u8; len];
            fill_ui_text(&mut frame_a, 512, tr(0, 0, 256, 256), false);
            fill_smooth_photo(&mut frame_a, 512, tr(256, 0, 256, 256), 0xFF);
            fill_ui_text(&mut frame_a, 512, tr(0, 256, 128, 128), false);
            fill_smooth_photo(&mut frame_a, 512, tr(128, 256, 128, 128), 0xFF);
            fill_smooth_photo(&mut frame_a, 512, tr(0, 384, 256, 128), 0xFF);
            fill_smooth_photo(&mut frame_a, 512, tr(256, 256, 256, 256), 0xFF);

            let frame_b = vec![0x33u8; len];

            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len];

            let run_step = |frame: &[u8],
                            full: bool,
                            mode: CompressionMode,
                            shadow: &mut ShadowBuffer,
                            srv: &mut ServerTileCache,
                            cli: &mut ClientTileCache,
                            canvas: &mut [u8],
                            comp: &mut ShardingCompressor,
                            decomp: &mut ShardingDecompressor|
             -> (Vec<TileRect>, Vec<CompressedTile>) {
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut streaming = None;
                if full {
                    filter_and_compress_full_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(9),
                        meta,
                        mode,
                        Some(shadow),
                        Some(srv),
                        comp,
                        |s| streaming = Some(s),
                    );
                } else {
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(9),
                        meta,
                        mode,
                        None,
                        shadow,
                        Some(srv),
                        comp,
                        |s| streaming = Some(s),
                    );
                }
                let mut wire = Vec::new();
                streaming.unwrap().framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated = ValidatedTileStream::new(CompressedTileStream {
                    header: header.clone(),
                    tiles: rx,
                })
                .unwrap();
                decompress_and_unfilter_streaming(validated, decomp, canvas, Some(cli)).unwrap();
                (header.tile_rects, tiles)
            };

            let hybrid_85 = CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            };
            let hybrid_95 = CompressionMode::Hybrid {
                quality: 95,
                subsamp: JpegSubsamp::Sub444,
            };

            run_step(
                &frame_a,
                true,
                hybrid_85,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let hybrid_a_canvas = canvas.clone();
            run_step(
                &frame_b,
                false,
                hybrid_85,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );

            // Same Hybrid mode hits 100% with zero extra generation loss.
            let (_, tiles_same) = run_step(
                &frame_a,
                false,
                hybrid_85,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            assert!(
                tiles_same
                    .iter()
                    .all(|t| matches!(t.payload, TilePayload::CacheHit(_)))
            );
            assert_eq!(canvas, hybrid_a_canvas);

            // Higher Hybrid(q=95, Sub444): promoted lossless text tiles/quadrants HIT,
            // while q=85 lossy JPEG tiles MISS.
            run_step(
                &frame_b,
                false,
                hybrid_95,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let (rects_95, tiles_95) = run_step(
                &frame_a,
                false,
                hybrid_95,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let hit_rects_95: Vec<TileRect> = tiles_95
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                .map(|t| rects_95[t.tile_idx as usize])
                .collect();
            assert!(hit_rects_95.contains(&tr(0, 0, 256, 256)));
            assert!(hit_rects_95.contains(&tr(0, 256, 128, 128)));
            assert!(!hit_rects_95.contains(&tr(256, 0, 256, 256)));

            // Lossless mode: promoted lossless text tiles/quadrants HIT; lossy JPEG tiles
            // MISS and re-encode bit-exact.
            run_step(
                &frame_b,
                false,
                CompressionMode::Lossless,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let (rects_ll, tiles_ll) = run_step(
                &frame_a,
                false,
                CompressionMode::Lossless,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let hit_rects_ll: Vec<TileRect> = tiles_ll
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                .map(|t| rects_ll[t.tile_idx as usize])
                .collect();
            assert!(hit_rects_ll.contains(&tr(0, 0, 256, 256)));
            assert!(hit_rects_ll.contains(&tr(0, 256, 128, 128)));
            assert!(!hit_rects_ll.contains(&tr(256, 0, 256, 256)));
            assert_eq!(canvas, frame_a, "Lossless refresh must be 100% bit-exact");

            // Returning under Hybrid now hits the upgraded Lossless slots cleanly.
            run_step(
                &frame_b,
                false,
                hybrid_85,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let (_, tiles_upgraded) = run_step(
                &frame_a,
                false,
                hybrid_85,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            assert!(
                tiles_upgraded
                    .iter()
                    .all(|t| matches!(t.payload, TilePayload::CacheHit(_)))
            );
            assert_eq!(canvas, frame_a);
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_cache_with_scroll_rect_and_malformed_decode_rejections() {
        use crate::serialization::framing::CacheHitRegion;

        with_test_codec(|compressor, decompressor| {
            // 1. Scroll down then back up: ScrollRect + CacheHit + CachePut in the same frame!
            let width = 768usize;
            let height = 512usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };
            let doc = make_textured_document(width, height + 256, stride, BufferFormat::Xrgb8888);
            let f_top = doc[0..len].to_vec();
            let f_down = doc[128 * stride..128 * stride + len].to_vec();

            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len];

            let apply_scroll = |streaming: StreamingCompressedTiles,
                                decomp: &mut ShardingDecompressor,
                                canvas: &mut [u8],
                                cli: &mut ClientTileCache|
             -> (Option<ScrollRect>, Vec<CompressedTile>) {
                let mut wire = Vec::new();
                streaming.framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).unwrap();
                let sr = validated.scroll_rect();
                if let Some(sr) = sr {
                    apply_scroll_rect_in_place(canvas, stride, sr);
                }
                decompress_and_unfilter_streaming(validated, decomp, canvas, Some(cli)).unwrap();
                (sr, tiles)
            };

            {
                let p = f_top.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(3),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                apply_scroll(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_canvas_matches_frame(&canvas, &f_top, meta, 0);
            }
            {
                let p = f_down.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(3),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                let (sr, _) = apply_scroll(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(sr.unwrap().dy, -128);
                assert_canvas_matches_frame(&canvas, &f_down, meta, 0);
            }
            let mut f_back = f_top;
            f_back[10 * stride + (width - 10) * 4] ^= 0x55;
            {
                let p = f_back.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut s = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(3),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |st| s = Some(st),
                );
                let (sr, tiles) = apply_scroll(s.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(sr.unwrap().dy, 128);
                let hits = tiles
                    .iter()
                    .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                    .count();
                let puts = tiles.iter().filter(|t| t.cache_put.is_some()).count();
                assert!(
                    hits >= 5,
                    "expected cached top quadrants to hit: hits={hits}"
                );
                assert_eq!(puts, 1, "modified top-right quadrant must CachePut");
                assert_canvas_matches_frame(&canvas, &f_back, meta, 0);
            }

            // 2. Malformed decode & `ValidatedTileStream::new` rejections.
            let m64 = BufferMetadata {
                width: 64,
                height: 64,
                stride: 64 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let r64 = TileRect {
                x: 0,
                y: 0,
                w: 64,
                h: 64,
            };
            let make_hdr = |meta: BufferMetadata,
                            reset: bool,
                            evict: Vec<u16>,
                            rects: Vec<TileRect>| RawBufferHeader {
                frame_seq: 0,
                client: ClientId(1),
                surface: WlSurfaceId(1),
                metadata: meta,
                incremental: true,
                scroll_rect: None,
                reset_tile_cache: reset,
                evict_slots: evict,
                tile_rects: rects,
            };

            for bad_hdr in [
                make_hdr(m64, true, vec![0], vec![]),
                make_hdr(m64, false, vec![MAX_TILE_CACHE_SLOTS as u16], vec![]),
                make_hdr(m64, false, vec![5, 5], vec![]),
            ] {
                let (_tx, rx) = crossbeam_channel::unbounded();
                assert!(
                    ValidatedTileStream::new(CompressedTileStream {
                        header: bad_hdr,
                        tiles: rx,
                    })
                    .is_err()
                );
            }

            let decode_custom = |header: RawBufferHeader,
                                 tiles: Vec<CompressedTile>,
                                 cache: Option<&mut ClientTileCache>,
                                 decomp: &mut ShardingDecompressor|
             -> Result<()> {
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in tiles {
                    tx.send(Ok(t)).unwrap();
                }
                drop(tx);
                let byte_len = header.metadata.height as usize * header.metadata.stride as usize;
                let mut out = vec![0u8; byte_len];
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx })?;
                decompress_and_unfilter_streaming(validated, decomp, &mut out, cache)
            };

            let mut mal_cache = ClientTileCache::new();

            // CacheHit referencing empty slot -> Err.
            assert!(
                decode_custom(
                    make_hdr(m64, false, vec![], vec![r64]),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::CacheHit(CacheHitRef {
                            slot_id: 7,
                            region: CacheHitRegion::Full,
                        }),
                    }],
                    Some(&mut mal_cache),
                    decompressor,
                )
                .is_err()
            );

            // Populate slot 7 as 64x64 Xrgb8888.
            decode_custom(
                make_hdr(m64, false, vec![], vec![r64]),
                vec![CompressedTile {
                    tile_idx: 0,
                    cache_put: Some(7),
                    payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                }],
                Some(&mut mal_cache),
                decompressor,
            )
            .unwrap();

            // Dimension mismatch (32x32 rect hitting 64x64 Full slot) -> Err.
            assert!(
                decode_custom(
                    make_hdr(
                        m64,
                        false,
                        vec![],
                        vec![TileRect {
                            x: 0,
                            y: 0,
                            w: 32,
                            h: 32,
                        }]
                    ),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::CacheHit(CacheHitRef {
                            slot_id: 7,
                            region: CacheHitRegion::Full,
                        }),
                    }],
                    Some(&mut mal_cache),
                    decompressor,
                )
                .is_err()
            );

            // `has_alpha` mismatch (Argb8888 frame hitting Xrgb8888 slot 7) -> Err.
            let m64_argb = BufferMetadata {
                format: BufferFormat::Argb8888,
                ..m64
            };
            assert!(
                decode_custom(
                    make_hdr(m64_argb, false, vec![], vec![r64]),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::CacheHit(CacheHitRef {
                            slot_id: 7,
                            region: CacheHitRegion::Full,
                        }),
                    }],
                    Some(&mut mal_cache),
                    decompressor,
                )
                .is_err()
            );

            // Duplicate `cache_put` slot_id in the same frame -> Err.
            assert!(
                decode_custom(
                    make_hdr(
                        m64,
                        false,
                        vec![],
                        vec![
                            TileRect {
                                x: 0,
                                y: 0,
                                w: 32,
                                h: 64,
                            },
                            TileRect {
                                x: 32,
                                y: 0,
                                w: 32,
                                h: 64,
                            },
                        ]
                    ),
                    vec![
                        CompressedTile {
                            tile_idx: 0,
                            cache_put: Some(9),
                            payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                        },
                        CompressedTile {
                            tile_idx: 1,
                            cache_put: Some(9),
                            payload: TilePayload::Leaf(LeafPayload::Solid([4, 5, 6, 0xFF])),
                        },
                    ],
                    Some(&mut mal_cache),
                    decompressor,
                )
                .is_err()
            );

            // `CacheHit` with `cache_put: Some(_)` -> Err.
            assert!(
                decode_custom(
                    make_hdr(m64, false, vec![], vec![r64]),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: Some(8),
                        payload: TilePayload::CacheHit(CacheHitRef {
                            slot_id: 7,
                            region: CacheHitRegion::Full,
                        }),
                    }],
                    Some(&mut mal_cache),
                    decompressor,
                )
                .is_err()
            );

            // Cache ops when `tile_cache` is `None` -> Err.
            assert!(
                decode_custom(
                    make_hdr(m64, true, vec![], vec![]),
                    vec![],
                    None,
                    decompressor,
                )
                .is_err()
            );
            assert!(
                decode_custom(
                    make_hdr(m64, false, vec![0], vec![]),
                    vec![],
                    None,
                    decompressor,
                )
                .is_err()
            );
            assert!(
                decode_custom(
                    make_hdr(m64, false, vec![], vec![r64]),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: Some(0),
                        payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
                    }],
                    None,
                    decompressor,
                )
                .is_err()
            );
            assert!(
                decode_custom(
                    make_hdr(m64, false, vec![], vec![r64]),
                    vec![CompressedTile {
                        tile_idx: 0,
                        cache_put: None,
                        payload: TilePayload::CacheHit(CacheHitRef {
                            slot_id: 0,
                            region: CacheHitRegion::Full,
                        }),
                    }],
                    None,
                    decompressor,
                )
                .is_err()
            );
        });
    }

    #[test]
    fn test_merge_cropped_quad_slots_seams_and_disjoint_safety() {
        let tr = |x, y, w, h| TileRect { x, y, w, h };
        let unk = |x, y, w, h| (tr(x, y, w, h), RectScanMeta::UNKNOWN);
        let solid_a = RectScanMeta {
            solid_px: Some([10, 20, 30, 0xFF]),
            all_opaque: true,
        };
        let solid_b = RectScanMeta {
            solid_px: Some([40, 50, 60, 0xFF]),
            all_opaque: true,
        };
        let opaque = RectScanMeta {
            solid_px: None,
            all_opaque: true,
        };

        // Horizontal seam merge (`128x12 + 128x12 -> 256x12`), combining matching solid metadata.
        let mut g = [
            None,
            None,
            Some((tr(0, 244, 128, 12), solid_a)),
            Some((tr(128, 244, 128, 12), solid_a)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [None, None, Some((tr(0, 244, 256, 12), solid_a)), None]);

        // Vertical seam merge (`12x128 + 12x128 -> 12x256`), different solids drop solid_px but keep all_opaque.
        let mut g = [
            None,
            Some((tr(244, 0, 12, 128), solid_a)),
            None,
            Some((tr(244, 128, 12, 128), solid_b)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [None, Some((tr(244, 0, 12, 256), opaque)), None, None]);

        // 4-quadrant seam merge across both horizontal and vertical seams; any UNKNOWN degrades conservatively.
        let mut g = [
            Some((tr(0, 118, 128, 10), opaque)),
            Some((tr(128, 118, 128, 10), solid_a)),
            Some((tr(0, 128, 128, 10), solid_a)),
            Some(unk(128, 128, 128, 10)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [Some(unk(0, 118, 256, 20)), None, None, None]);

        // Safety invariant: when `TL+TR` merge horizontally and `BL` is cropped
        // while `BR` is `None` (clean / Full / CacheHit), vertical merge MUST
        // NOT fire because merging `TL+TR` with `BL` would overlap `BR`.
        let mut g = [
            Some(unk(0, 118, 128, 10)),
            Some(unk(128, 118, 128, 10)),
            Some(unk(0, 128, 128, 10)),
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(unk(0, 118, 256, 10)),
                None,
                Some(unk(0, 128, 128, 10)),
                None,
            ]
        );
        let rects: Vec<TileRect> = g
            .into_iter()
            .flatten()
            .map(|(r, _)| r)
            .chain([tr(128, 128, 128, 128)])
            .collect();
        validate_tile_rects(&rects, 256, 256).unwrap();

        // Non-touching gap (`x=124` vs `x=130`) must NOT merge.
        let mut g = [
            Some(unk(0, 10, 124, 12)),
            Some(unk(130, 10, 126, 12)),
            None,
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(unk(0, 10, 124, 12)),
                Some(unk(130, 10, 126, 12)),
                None,
                None,
            ]
        );

        // Excess bounding-box slack (`128x2 + 128x40` -> union `256x40 = 10240`,
        // sum = `5376`, slack = `4864` >> cap) must NOT merge.
        let mut g = [
            Some(unk(0, 10, 128, 2)),
            Some(unk(128, 10, 128, 40)),
            None,
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(unk(0, 10, 128, 2)),
                Some(unk(128, 10, 128, 40)),
                None,
                None,
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tight_bbox_crop_odd_shapes_alpha_hybrid_and_cross_tier_parity() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let tiers = arch::Tier::supported_tiers();
            let width = 300usize;
            let height = 280usize;

            for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
                for pad_bytes in [0usize, 16usize] {
                    let stride = width * 4 + pad_bytes;
                    let len = stride * height;
                    let meta = BufferMetadata {
                        width: width as i32,
                        height: height as i32,
                        stride: stride as i32,
                        format,
                    };
                    let base = test_vec(len);

                    // Construct an incremental frame exercising:
                    // - Q(0,0): scattered interior multi-pixel edit (`13..=29, 7..=25` -> `17x19` odd bbox)
                    // - Q(1,0): single-pixel `1x1` edit at `(191, 43)`
                    // - Q(0,1): alpha-only edit at `(55, 177)` (ignored on Xrgb, `1x1` on Argb)
                    // - Q(2,1): odd clipped right-edge edit (`259..=298, 140..=145` -> `40x6`)
                    let mut edited = base.clone();
                    edited[7 * stride + 13 * 4] ^= 0x55;
                    edited[25 * stride + 29 * 4 + 2] ^= 0x33;
                    edited[43 * stride + 191 * 4 + 1] ^= 0x77;
                    edited[177 * stride + 55 * 4 + 3] ^= 0x40;
                    edited[140 * stride + 259 * 4] ^= 0x19;
                    edited[145 * stride + 298 * 4 + 1] ^= 0x2A;

                    let mut expected_rects = vec![tr(13, 7, 17, 19), tr(191, 43, 1, 1)];
                    if format == BufferFormat::Argb8888 {
                        expected_rects.push(tr(55, 177, 1, 1));
                    }
                    expected_rects.push(tr(259, 140, 40, 6));

                    let mut ref_shadow = None;
                    for &tier in &tiers {
                        let mut shadow = ShadowBuffer::new();
                        let mut canvas = AlignedVec::<64>::with_capacity(len);
                        canvas.resize(len, 0xCC);
                        {
                            let p = base.as_ptr();
                            let bp = unsafe { BufferPointer::new(&p, len) };
                            let mut st = None;
                            filter_and_compress_full_streaming_with_tier(
                                tier,
                                bp,
                                ClientId(1),
                                WlSurfaceId(10),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut shadow),
                                None,
                                compressor,
                                |s| st = Some(s),
                            );
                            let mut wire = Vec::new();
                            st.unwrap().framed_write(&mut wire).unwrap();
                            let mut rx = None;
                            CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                                rx = Some(s);
                                Ok(())
                            })
                            .unwrap();
                            let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                            decompress_and_unfilter_streaming(
                                validated,
                                decompressor,
                                &mut canvas,
                                None,
                            )
                            .unwrap();
                        }

                        let p = edited.as_ptr();
                        let bp = unsafe { BufferPointer::new(&p, len) };
                        let mut st = None;
                        let diff = filter_and_compress_incremental_streaming_with_tier(
                            tier,
                            bp,
                            ClientId(1),
                            WlSurfaceId(10),
                            meta,
                            CompressionMode::Lossless,
                            None,
                            &mut shadow,
                            None,
                            compressor,
                            |s| st = Some(s),
                        );
                        assert_eq!(
                            diff.dirty_tiles, expected_rects,
                            "cropped rects mismatch tier={tier:?} format={format:?} pad={pad_bytes}"
                        );
                        let mut wire = Vec::new();
                        st.unwrap().framed_write(&mut wire).unwrap();
                        let mut rx = None;
                        CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                            rx = Some(s);
                            Ok(())
                        })
                        .unwrap();
                        let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                        decompress_and_unfilter_streaming(
                            validated,
                            decompressor,
                            &mut canvas,
                            None,
                        )
                        .unwrap();
                        assert_canvas_matches_frame(&canvas, &edited, meta, 0xCC);
                        if let Some(prev) = &ref_shadow {
                            assert_eq!(
                                shadow.pixels(),
                                prev,
                                "shadow mismatch across tiers ({tier:?}, {format:?})"
                            );
                        } else {
                            ref_shadow = Some(shadow.into_pixels());
                        }
                    }
                }
            }

            // Hybrid mode on odd-sized cropped patches (`17x19` photo -> JPEG,
            // `11x13` photo -> lossless fallback because `< MIN_JPEG_DIM`).
            let h_meta = BufferMetadata {
                width: 256,
                height: 256,
                stride: 256 * 4,
                format: BufferFormat::Argb8888,
            };
            let h_len = 256 * 256 * 4;
            let mut f0 = vec![0xFFu8; h_len];
            fill_ui_text(&mut f0, 256, tr(0, 0, 256, 256), true);
            let mut f1 = f0.clone();
            fill_smooth_photo(&mut f1, 256, tr(13, 11, 17, 19), 0x80);
            fill_smooth_photo(&mut f1, 256, tr(141, 17, 11, 13), 0xC0);

            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; h_len];
            {
                let p = f0.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, h_len) };
                let mut st = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(11),
                    h_meta,
                    CompressionMode::hybrid_default(),
                    Some(&mut shadow),
                    None,
                    compressor,
                    |s| st = Some(s),
                );
                let mut wire = Vec::new();
                st.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
            }
            {
                let p = f1.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, h_len) };
                let mut st = None;
                let diff = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(11),
                    h_meta,
                    CompressionMode::hybrid_default(),
                    None,
                    &mut shadow,
                    None,
                    compressor,
                    |s| st = Some(s),
                );
                assert_eq!(
                    diff.dirty_tiles,
                    vec![tr(13, 11, 17, 19), tr(141, 17, 11, 13)]
                );
                let mut wire = Vec::new();
                st.unwrap().framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(validated, decompressor, &mut canvas, None)
                    .unwrap();
                // `< MIN_JPEG_DIM` patch (`11x13`) stays strictly lossless, and alpha is
                // bit-exact everywhere including inside the `17x19` JPEG patch.
                assert!(rect_psnr_rgb(&f1, &canvas, 256, tr(141, 17, 11, 13)).is_infinite());
                assert!(rect_psnr_rgb(&f1, &canvas, 256, tr(13, 11, 17, 19)) > 30.0);
                for (i, (got, exp)) in canvas
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(f1.as_chunks::<4>().0.iter())
                    .enumerate()
                {
                    assert_eq!(got[3], exp[3], "hybrid cropped alpha mismatch at px {i}");
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tight_bbox_threshold_gating_and_tile_cache_isolation() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let width = 512usize;
            let height = 256usize;
            let stride = width * 4;
            let len = stride * height;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };

            let base = test_vec(len);
            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0u8; len];

            let apply = |st: StreamingCompressedTiles,
                         decomp: &mut ShardingDecompressor,
                         canvas: &mut [u8],
                         cli: &mut ClientTileCache|
             -> (Vec<TileRect>, Vec<CompressedTile>) {
                let mut wire = Vec::new();
                st.framed_write(&mut wire).unwrap();
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx.unwrap();
                let rects = stream.header.tile_rects.clone();
                let mut tiles: Vec<CompressedTile> =
                    stream.tiles.into_iter().map(Result::unwrap).collect();
                tiles.sort_by_key(|t| t.tile_idx);
                let (tx, ch_rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated = ValidatedTileStream::new(CompressedTileStream {
                    header: stream.header,
                    tiles: ch_rx,
                })
                .unwrap();
                decompress_and_unfilter_streaming(validated, decomp, canvas, Some(cli)).unwrap();
                (rects, tiles)
            };

            // 1. Initial full frame seeds the tile cache with 2 `256x256` tiles.
            {
                let p = base.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(20),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_canvas_matches_frame(&canvas, &base, meta, 0);
            }
            let initial_slots = srv.occupied_slots();
            let initial_bytes = srv.used_bytes();
            assert_eq!(initial_slots, 2);

            // 2. Repeated tiny transient edits (`<= 45%`, e.g. `16x20` cursor blinks)
            //    crop tightly and NEVER pollute or evict tile cache slots.
            for step in 1u8..=8 {
                let mut frame = base.clone();
                let cx = 20 + (step as usize) * 8;
                for y in 30..50 {
                    for x in cx..cx + 16 {
                        frame[y * stride + x * 4] ^= step.wrapping_mul(17);
                    }
                }
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(20),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                let (rects, tiles) = apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(rects.len(), 1);
                assert!(
                    rects[0].w <= 24 && rects[0].h == 20,
                    "expected tight crop: {rects:?}"
                );
                assert!(tiles.iter().all(|t| t.cache_put.is_none()));
                assert_eq!(srv.occupied_slots(), initial_slots);
                assert_eq!(srv.used_bytes(), initial_bytes);
                assert_canvas_matches_frame(&canvas, &frame, meta, 0);
            }

            // Reset back to `base`.
            {
                let p = base.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(20),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_canvas_matches_frame(&canvas, &base, meta, 0);
            }

            // 3. Popup-like overlay on tile `(0, 0, 256, 256)`: top half (`y=0..128`)
            //    is `100%` dirty (`Full` -> `CachePut`), while bottom quadrants only
            //    change `28` rows (`21.9% <= 45%`, `3584 px >= 1024` -> cropped
            //    `Uncached` merged to `256x28`, zero cache pollution).
            let mut popup = base.clone();
            for y in 0..156 {
                for x in 0..256 {
                    let off = y * stride + x * 4;
                    popup[off] ^= 0x6B ^ ((x >> 4) as u8);
                    popup[off + 1] ^= 0x3D ^ ((y >> 3) as u8);
                }
            }
            {
                let p = popup.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(20),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                let (rects, tiles) = apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(
                    rects,
                    vec![
                        tr(0, 0, 128, 128),
                        tr(128, 0, 128, 128),
                        tr(0, 128, 256, 28)
                    ]
                );
                assert!(tiles[0].cache_put.is_some());
                assert!(tiles[1].cache_put.is_some());
                assert!(
                    tiles[2].cache_put.is_none(),
                    "cropped bottom strip must not CachePut"
                );
                assert_canvas_matches_frame(&canvas, &popup, meta, 0);
            }

            // 4. Closing the popup reverts `(0, 0, 256, 256)` back to `base`: all 4
            //    quadrants carry hashes (`2 Full + 2 lookup-hashed Cropped`), so
            //    `PartialTile2D` coalesces them into ONE `256x256` parent `CacheHit`!
            {
                let p = base.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(20),
                    meta,
                    CompressionMode::Lossless,
                    None,
                    &mut shadow,
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                let (rects, tiles) = apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                assert_eq!(rects, vec![tr(0, 0, 256, 256)]);
                assert_eq!(tiles.len(), 1);
                assert!(matches!(tiles[0].payload, TilePayload::CacheHit(_)));
                assert_canvas_matches_frame(&canvas, &base, meta, 0);
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tight_bbox_scroll_exposed_strip_and_overlay_scrollbar() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };
            let width = 512usize;
            let height = 512usize;
            let sb_w = 12usize;

            for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
                let pad = 16usize;
                let stride = width * 4 + pad;
                let len = stride * height;
                let meta = BufferMetadata {
                    width: width as i32,
                    height: height as i32,
                    stride: stride as i32,
                    format,
                };

                // Build a textured scrollable document with a moving 12px right-edge
                // overlay scrollbar thumb.
                let mut doc = vec![0u8; (height + 200) * stride];
                for y in 0..(height + 200) {
                    for x in 0..width {
                        let off = y * stride + x * 4;
                        let v = ((x * 13 + y * 29 + (x ^ y) * 7) & 0xFF) as u8;
                        doc[off] = v;
                        doc[off + 1] = v.wrapping_add((y & 0x7F) as u8);
                        doc[off + 2] = v.wrapping_mul(3);
                        doc[off + 3] = if format == BufferFormat::Argb8888 {
                            0xE0 | ((x + y) & 0x1F) as u8
                        } else {
                            0xFF
                        };
                    }
                }
                let make_frame = |scroll_y: usize, thumb_y: usize| -> Vec<u8> {
                    let mut f = doc[scroll_y * stride..scroll_y * stride + len].to_vec();
                    for y in 0..height {
                        let in_thumb = y >= thumb_y && y < thumb_y + 96;
                        for x in (width - sb_w)..width {
                            let off = y * stride + x * 4;
                            f[off..off + 4].copy_from_slice(if in_thumb {
                                &[120, 120, 125, 0xFF]
                            } else {
                                &[230, 230, 232, 0xFF]
                            });
                        }
                    }
                    f
                };

                let f0 = make_frame(64, 40);
                let f1 = make_frame(88, 64); // dy = -24 exposed strip at bottom + scrollbar moved
                let f2 = make_frame(64, 40); // dy = +24 exposed strip at top + scrollbar restored

                let mut srv = ServerTileCache::new();
                let mut cli = ClientTileCache::new();
                let mut shadow = ShadowBuffer::new();
                let mut canvas = AlignedVec::<64>::with_capacity(len);
                canvas.resize(len, 0xCC);

                let apply = |st: StreamingCompressedTiles,
                             decomp: &mut ShardingDecompressor,
                             canvas: &mut [u8],
                             cli: &mut ClientTileCache|
                 -> (Option<ScrollRect>, Vec<TileRect>, Vec<CompressedTile>) {
                    let mut wire = Vec::new();
                    st.framed_write(&mut wire).unwrap();
                    let mut rx = None;
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                        rx = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    let stream = rx.unwrap();
                    let sr = stream.header.scroll_rect;
                    let rects = stream.header.tile_rects.clone();
                    let tiles: Vec<CompressedTile> =
                        stream.tiles.into_iter().map(Result::unwrap).collect();
                    let (tx, ch_rx) = crossbeam_channel::unbounded();
                    for t in &tiles {
                        tx.send(Ok(t.clone())).unwrap();
                    }
                    drop(tx);
                    let validated = ValidatedTileStream::new(CompressedTileStream {
                        header: stream.header,
                        tiles: ch_rx,
                    })
                    .unwrap();
                    if let Some(scroll_rect) = validated.scroll_rect() {
                        apply_scroll_rect_in_place_parallel(canvas, stride, scroll_rect, decomp);
                    }
                    decompress_and_unfilter_streaming(validated, decomp, canvas, Some(cli))
                        .unwrap();
                    (sr, rects, tiles)
                };

                {
                    let p = f0.as_ptr();
                    let bp = unsafe { BufferPointer::new(&p, len) };
                    let mut st = None;
                    filter_and_compress_full_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(21),
                        meta,
                        CompressionMode::Lossless,
                        Some(&mut shadow),
                        Some(&mut srv),
                        compressor,
                        |s| st = Some(s),
                    );
                    apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                    assert_canvas_matches_frame(&canvas, &f0, meta, 0xCC);
                }
                let slots_after_f0 = srv.occupied_slots();

                // Forward scroll (`dy = -24`) exposes a tight `24`-row bottom strip
                // (merged to `256x24` across horizontal seam) + tight `12`-px
                // scrollbar column without polluting the tile cache.
                {
                    let p = f1.as_ptr();
                    let bp = unsafe { BufferPointer::new(&p, len) };
                    let mut st = None;
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(21),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut srv),
                        compressor,
                        |s| st = Some(s),
                    );
                    let (sr, rects, tiles) =
                        apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                    assert_eq!(sr.expect("expected dy=-24 scroll").dy, -24);
                    assert!(
                        rects.contains(&tr(0, 488, 256, 24)),
                        "expected merged 256x24 bottom strip in {rects:?}"
                    );
                    assert!(
                        rects
                            .iter()
                            .any(|r| r.x == (width - sb_w) as u32 && r.w == sb_w as u32),
                        "expected tight 12px scrollbar crop in {rects:?}"
                    );
                    assert!(
                        tiles.iter().all(|t| t.cache_put.is_none()),
                        "cropped scroll strip/scrollbar must not pollute tile cache"
                    );
                    assert_eq!(srv.occupied_slots(), slots_after_f0);
                    assert_canvas_matches_frame(&canvas, &f1, meta, 0xCC);
                }

                // Reverse scroll (`dy = +24`) restores `f0` bit-exact.
                {
                    let p = f2.as_ptr();
                    let bp = unsafe { BufferPointer::new(&p, len) };
                    let mut st = None;
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(21),
                        meta,
                        CompressionMode::Lossless,
                        None,
                        &mut shadow,
                        Some(&mut srv),
                        compressor,
                        |s| st = Some(s),
                    );
                    let (sr, rects, _) = apply(st.unwrap(), decompressor, &mut canvas, &mut cli);
                    assert_eq!(sr.expect("expected dy=+24 scroll").dy, 24);
                    validate_tile_rects(&rects, width as u32, height as u32).unwrap();
                    assert_canvas_matches_frame(&canvas, &f2, meta, 0xCC);
                }
            }
        });
    }

    #[test]
    fn test_uninit_shadow_slot_and_row_scratch_miri_safe() {
        // 1. Uninitialized shadow spare-capacity population across packed and
        //    padded strides (`Strip1D` and `Tile2D` partitions).
        for (width, height) in [(300usize, 5usize), (300usize, 20usize)] {
            for pad_bytes in [0usize, 12usize] {
                let width_bytes = width * 4;
                let stride = width_bytes + pad_bytes;
                let byte_len = height * stride;
                let src = test_vec(byte_len);
                let src_ptr = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&src_ptr, byte_len) };

                let meta = BufferMetadata {
                    width: width as i32,
                    height: height as i32,
                    stride: stride as i32,
                    format: BufferFormat::Argb8888,
                };
                let (_, base_entries, _) = plan_incremental_scan(&meta, None).unwrap();

                let mut shadow = Vec::with_capacity(byte_len);
                let shadow_rows = ShadowRows {
                    base: shadow.as_mut_ptr(),
                    stride,
                };
                for entry in &base_entries {
                    let rect = entry.rect();
                    // SAFETY: `base_entries` partitions `[0, width) x [0, height)`
                    // strictly in-bounds and disjointly.
                    unsafe {
                        copy_rect_to_shadow(bp, &shadow_rows, stride, rect);
                        zero_shadow_row_padding(&shadow_rows, width_bytes, rect);
                    }
                }
                // SAFETY: Every pixel byte and right-edge stride padding byte
                // in `0..byte_len` was initialized above.
                unsafe { shadow.set_len(byte_len) };

                for y in 0..height {
                    let row = y * stride;
                    assert_eq!(
                        &shadow[row..row + width_bytes],
                        &src[row..row + width_bytes],
                        "pixel mismatch at row {y} ({width}x{height}, pad={pad_bytes})"
                    );
                    assert_eq!(
                        &shadow[row + width_bytes..row + stride],
                        &vec![0u8; pad_bytes][..],
                        "stride padding not zeroed at row {y} ({width}x{height}, pad={pad_bytes})"
                    );
                }
            }
        }

        // 2. `ClientTileCache` uninit slot target + commit + invalidate.
        let mut cli = ClientTileCache::new();
        let rect = TileRect {
            x: 16,
            y: 8,
            w: 12,
            h: 6,
        };
        let px = Vec4u8(0x11, 0x22, 0x33, 0xFF);
        {
            let slots = cli.sync_slots();
            // SAFETY: Single-threaded exclusive ownership of slot 3.
            let slot_frame =
                unsafe { ClientTileCache::prepare_slot_target_uninit(&slots, 0, 3, rect) };
            for ry in 0..rect.h as usize {
                // SAFETY: Row lies strictly inside `slot_frame` (`12x6`).
                let row = unsafe {
                    slice::from_raw_parts_mut(slot_frame.row_ptr(ry, 0), rect.w as usize)
                };
                // SAFETY: `STREAM = false` has no alignment requirement.
                unsafe { ScalarKernel.splat_row::<false>(row, px) };
            }
            // SAFETY: All `12 * 6` pixels were initialized above.
            let committed = unsafe { ClientTileCache::commit_slot(&slots, 3, rect, false) };
            assert_eq!(committed.len(), (rect.w * rect.h) as usize);
            assert!(committed.iter().all(|&p| p == px));
        }
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.occupied_slots(), 1);
        assert_eq!(cli.used_bytes(), (rect.w * rect.h * 4) as usize);

        // `evict()` retains Vec capacity and reuses the exact same backing
        // allocation on a subsequent `CachePut` of the same tile size.
        let (ptr_before, cap_before, occ_before) = cli.slot_buf_info(3);
        assert!(occ_before);
        assert!(cap_before >= (rect.w * rect.h) as usize);
        cli.evict(3).unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        let (ptr_evicted, cap_evicted, occ_evicted) = cli.slot_buf_info(3);
        assert!(!occ_evicted);
        assert_eq!(ptr_evicted, ptr_before);
        assert_eq!(cap_evicted, cap_before);
        {
            let slots = cli.sync_slots();
            // SAFETY: Single-threaded exclusive ownership of slot 3.
            let slot_frame =
                unsafe { ClientTileCache::prepare_slot_target_uninit(&slots, 0, 3, rect) };
            for ry in 0..rect.h as usize {
                // SAFETY: Row lies strictly inside `slot_frame` (`12x6`).
                let row = unsafe {
                    slice::from_raw_parts_mut(slot_frame.row_ptr(ry, 0), rect.w as usize)
                };
                // SAFETY: `STREAM = false` has no alignment requirement.
                unsafe { ScalarKernel.splat_row::<false>(row, px) };
            }
            // SAFETY: All `12 * 6` pixels were initialized above.
            let _ = unsafe { ClientTileCache::commit_slot(&slots, 3, rect, false) };
        }
        cli.recompute_and_verify_bytes().unwrap();
        let (ptr_reused, cap_reused, occ_reused) = cli.slot_buf_info(3);
        assert!(occ_reused);
        assert_eq!(ptr_reused, ptr_before);
        assert_eq!(cap_reused, cap_before);

        // In-band `reset_tile_cache` preserves capacity; `clear()` releases it.
        cli.apply_header_ops(true, &[]).unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        assert_eq!(cli.slot_buf_info(3), (ptr_before, cap_before, false));
        cli.clear();
        assert_eq!(cli.capacity_bytes(), 0);

        // `SlotScans::as_slice` zero-copy read after writes under Miri.
        let mut scans = SlotScans::new(3, tile_hash::TileHashKey::from_seed(42));
        let sample = tile_hash::TileHashResult {
            hash: [7u8; 16],
            meta: RectScanMeta {
                solid_px: Some([1, 2, 3, 0xFF]),
                all_opaque: true,
            },
        };
        // SAFETY: Exclusive single-threaded write to slot 1.
        unsafe { scans.write(1, sample) };
        assert_eq!(scans.as_slice()[1], sample);

        {
            let slots = cli.sync_slots();
            // SAFETY: Single-threaded exclusive ownership of slot 3.
            let _slot_frame =
                unsafe { ClientTileCache::prepare_slot_target_uninit(&slots, 0, 3, rect) };
            // Simulate decode error before `commit_slot`.
            unsafe { ClientTileCache::invalidate_slot(&slots, 3) };
        }
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        assert_eq!(cli.capacity_bytes(), 0);

        // 3. `select_and_filter_row` across short cropped (`tw < 256`),
        //    full `256`, and wide 1D strip (`tw > 256`).
        for tw in [1usize, 7, 16, 128, 256, 300] {
            let row_b = tw * 3;
            let row0 = test_vec(row_b);
            let row1: Vec<u8> = row0
                .iter()
                .enumerate()
                .map(|(i, &b)| b.wrapping_add(((i % 7) as u8).wrapping_sub(3)))
                .collect();
            let mut filt0 = vec![0u8; row_b];
            let mut filt1 = vec![0u8; row_b];
            let p0 = select_and_filter_row(ScalarKernel, 0, &row0, &[], &mut filt0);
            assert!(p0 < 2, "row 0 predictor must be None or Sub, got {p0}");
            let p1 = select_and_filter_row(ScalarKernel, 1, &row1, &row0, &mut filt1);

            let mut dec0 = vec![0u8; row_b];
            match p0 {
                0 => dec0.copy_from_slice(&filt0),
                1 => ScalarKernel.row_sub3_inverse(&filt0, &mut dec0),
                _ => unreachable!(),
            }
            assert_eq!(dec0, row0, "row 0 roundtrip failed for tw={tw}");

            let mut dec1 = vec![0u8; row_b];
            match p1 {
                0 => dec1.copy_from_slice(&filt1),
                1 => ScalarKernel.row_sub3_inverse(&filt1, &mut dec1),
                2 => ScalarKernel.row_up_inverse(&filt1, &dec0, &mut dec1),
                _ => ScalarKernel.row_subup3_inverse(&filt1, &dec0, &mut dec1),
            }
            assert_eq!(dec1, row1, "row 1 roundtrip failed for tw={tw}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_uninit_shadow_full_frame_and_puts_only_soundness() {
        use crate::serialization::framing::CacheHitRegion;

        with_test_codec(|compressor, decompressor| {
            let assert_shadow_exact = |shadow: &[u8],
                                       src: &[u8],
                                       width: usize,
                                       height: usize,
                                       stride: usize,
                                       ctx: &str| {
                let width_bytes = width * 4;
                let pad_bytes = stride - width_bytes;
                assert_eq!(shadow.len(), height * stride, "shadow len mismatch ({ctx})");
                for y in 0..height {
                    let row = y * stride;
                    assert_eq!(
                        &shadow[row..row + width_bytes],
                        &src[row..row + width_bytes],
                        "shadow pixel mismatch row {y} ({ctx})"
                    );
                    if pad_bytes > 0 {
                        assert_eq!(
                            &shadow[row + width_bytes..row + stride],
                            &vec![0u8; pad_bytes][..],
                            "shadow padding not zeroed row {y} ({ctx})"
                        );
                    }
                }
            };

            let decode_stream =
                |st: StreamingCompressedTiles,
                 decomp: &mut ShardingDecompressor,
                 canvas: &mut [u8],
                 cli: Option<&mut ClientTileCache>| {
                    let mut wire = Vec::new();
                    st.framed_write(&mut wire).unwrap();
                    let mut rx = None;
                    CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                        rx = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                    decompress_and_unfilter_streaming(validated, decomp, canvas, cli).unwrap();
                };

            // 1. Uninit shadow full frame across tightly packed and padded strides,
            //    `Strip1D` (`300x5`) and multi-tile `Tile2D` (`300x270`), both
            //    formats, cold-fill / warm-2-pass / no-cache, and early receiver drop.
            for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
                for (width, height) in [(300usize, 5usize), (300usize, 270usize)] {
                    for pad_bytes in [0usize, 16usize] {
                        let stride = width * 4 + pad_bytes;
                        let byte_len = height * stride;
                        let meta = BufferMetadata {
                            width: width as i32,
                            height: height as i32,
                            stride: stride as i32,
                            format,
                        };
                        let frame_a = test_vec(byte_len);
                        let frame_b: Vec<u8> = frame_a.iter().map(|&b| b ^ 0x5A).collect();
                        let mut canvas = vec![0xCCu8; byte_len];

                        // (a) `tile_cache = Some` cold-fill (fresh empty shadow).
                        let mut srv = ServerTileCache::new();
                        let mut cli = ClientTileCache::new();
                        let mut shadow = ShadowBuffer::new();
                        {
                            let ptr = frame_a.as_ptr();
                            let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                            let mut st = None;
                            filter_and_compress_full_streaming(
                                bp,
                                ClientId(1),
                                WlSurfaceId(1),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut shadow),
                                Some(&mut srv),
                                compressor,
                                |s| st = Some(s),
                            );
                            decode_stream(st.unwrap(), decompressor, &mut canvas, Some(&mut cli));
                        }
                        assert_shadow_exact(
                            shadow.pixels(),
                            &frame_a,
                            width,
                            height,
                            stride,
                            "cold_fill",
                        );
                        assert_canvas_matches_frame(&canvas, &frame_a, meta, 0xCC);
                        assert!(srv.occupied_slots() > 0);

                        // (b) `tile_cache = Some` warm 2-pass with oversized sentinel shadow
                        //     (verifies shrink + zeroed padding with no stale `0xDD` bytes).
                        let mut oversized_shadow =
                            ShadowBuffer::from_pixels(vec![0xDDu8; byte_len * 3]);
                        {
                            let ptr = frame_b.as_ptr();
                            let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                            let mut st = None;
                            filter_and_compress_full_streaming(
                                bp,
                                ClientId(1),
                                WlSurfaceId(1),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut oversized_shadow),
                                Some(&mut srv),
                                compressor,
                                |s| st = Some(s),
                            );
                            decode_stream(st.unwrap(), decompressor, &mut canvas, Some(&mut cli));
                        }
                        assert_shadow_exact(
                            oversized_shadow.pixels(),
                            &frame_b,
                            width,
                            height,
                            stride,
                            "warm_2pass",
                        );
                        assert!(
                            oversized_shadow.capacity() <= byte_len.saturating_mul(2),
                            "oversized shadow did not shrink"
                        );
                        assert_canvas_matches_frame(&canvas, &frame_b, meta, 0xCC);

                        // (c) `tile_cache = Some` cold-fill with early receiver drop
                        //     (`disconnected = true`).
                        let mut srv_drop = ServerTileCache::new();
                        let mut shadow_cold_drop = ShadowBuffer::new();
                        {
                            let ptr = frame_a.as_ptr();
                            let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                            filter_and_compress_full_streaming(
                                bp,
                                ClientId(1),
                                WlSurfaceId(1),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut shadow_cold_drop),
                                Some(&mut srv_drop),
                                compressor,
                                drop,
                            );
                        }
                        assert_shadow_exact(
                            shadow_cold_drop.pixels(),
                            &frame_a,
                            width,
                            height,
                            stride,
                            "cold_fill_drop",
                        );

                        // (d) `tile_cache = None` normal.
                        let mut shadow_nc = ShadowBuffer::new();
                        {
                            let ptr = frame_a.as_ptr();
                            let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                            let mut st = None;
                            filter_and_compress_full_streaming(
                                bp,
                                ClientId(1),
                                WlSurfaceId(1),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut shadow_nc),
                                None,
                                compressor,
                                |s| st = Some(s),
                            );
                            decode_stream(st.unwrap(), decompressor, &mut canvas, None);
                        }
                        assert_shadow_exact(
                            shadow_nc.pixels(),
                            &frame_a,
                            width,
                            height,
                            stride,
                            "nocache",
                        );
                        assert_canvas_matches_frame(&canvas, &frame_a, meta, 0xCC);

                        // (e) `tile_cache = None` early receiver drop (`disconnected = true`).
                        let mut shadow_nc_drop = ShadowBuffer::new();
                        {
                            let ptr = frame_b.as_ptr();
                            let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                            filter_and_compress_full_streaming(
                                bp,
                                ClientId(1),
                                WlSurfaceId(1),
                                meta,
                                CompressionMode::Lossless,
                                Some(&mut shadow_nc_drop),
                                None,
                                compressor,
                                drop,
                            );
                        }
                        assert_shadow_exact(
                            shadow_nc_drop.pixels(),
                            &frame_b,
                            width,
                            height,
                            stride,
                            "nocache_drop",
                        );
                    }
                }
            }

            // 2. `decompress_cache_puts_only` success (uninit slot commit + bit-exact
            //    subsequent `CacheHit` replay) and decode error (slot invalidated cleanly).
            let width = 300usize;
            let height = 270usize;
            let stride = width * 4;
            let byte_len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };
            let frame = make_textured_document(width, height, stride, BufferFormat::Xrgb8888);
            let mut srv = ServerTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut wire_puts = Vec::new();
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                let mut st = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                st.unwrap().framed_write(&mut wire_puts).unwrap();
            }
            let mut wire_hits = Vec::new();
            {
                let ptr = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
                let mut st = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(2),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    compressor,
                    |s| st = Some(s),
                );
                st.unwrap().framed_write(&mut wire_hits).unwrap();
            }

            let mut cli = ClientTileCache::new();
            {
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire_puts.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_cache_puts_only(validated, decompressor, &mut cli).unwrap();
            }
            assert_eq!(cli.occupied_slots(), srv.occupied_slots());
            assert_eq!(cli.used_bytes(), srv.used_bytes());

            let mut canvas = vec![0xCCu8; byte_len];
            {
                let mut rx = None;
                CompressedTileStream::read_from_stream(&mut wire_hits.as_slice(), |s| {
                    rx = Some(s);
                    Ok(())
                })
                .unwrap();
                let validated = ValidatedTileStream::new(rx.unwrap()).unwrap();
                decompress_and_unfilter_streaming(
                    validated,
                    decompressor,
                    &mut canvas,
                    Some(&mut cli),
                )
                .unwrap();
            }
            assert_canvas_matches_frame(&canvas, &frame, meta, 0xCC);

            // Decode error inside `decompress_cache_puts_only` (both on an empty slot
            // and overwriting an occupied slot) must invalidate the slot cleanly without
            // committing uninitialized memory.
            let r64 = TileRect {
                x: 0,
                y: 0,
                w: 64,
                h: 64,
            };
            let m64 = BufferMetadata {
                width: 64,
                height: 64,
                stride: 64 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let occupied_before = cli.occupied_slots();
            let used_before = cli.used_bytes();
            let empty_slot = 900u16;
            let occupied_slot = 0u16;

            for target_slot in [empty_slot, occupied_slot] {
                let (tx, rx) = crossbeam_channel::unbounded();
                tx.send(Ok(CompressedTile {
                    tile_idx: 0,
                    cache_put: Some(target_slot),
                    // Translucent Solid on Xrgb8888 triggers an immediate decode error.
                    payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0x80])),
                }))
                .unwrap();
                drop(tx);
                let validated = ValidatedTileStream::new(CompressedTileStream {
                    header: RawBufferHeader {
                        frame_seq: 0,
                        client: ClientId(1),
                        surface: WlSurfaceId(2),
                        metadata: m64,
                        incremental: true,
                        scroll_rect: None,
                        reset_tile_cache: false,
                        evict_slots: Vec::new(),
                        tile_rects: vec![r64],
                    },
                    tiles: rx,
                })
                .unwrap();
                assert!(decompress_cache_puts_only(validated, decompressor, &mut cli).is_err());
                // Target slot must now be empty (a CacheHit referencing it must fail).
                assert!(
                    cli.validate_hit(
                        0,
                        CacheHitRef {
                            slot_id: target_slot,
                            region: CacheHitRegion::Full,
                        },
                        r64,
                        false,
                    )
                    .is_err()
                );
            }
            assert_eq!(cli.occupied_slots(), occupied_before - 1);
            assert!(cli.used_bytes() < used_before);
        });
    }

    #[test]
    fn test_miri_copy_and_scan_rect_to_shadow_uninit() {
        for (width, height) in [(1usize, 1usize), (17, 5), (33, 9)] {
            for pad_bytes in [0usize, 12usize] {
                let stride = width * 4 + pad_bytes;
                let byte_len = height * stride;
                let width_bytes = width * 4;
                let rect = TileRect {
                    x: 0,
                    y: 0,
                    w: width as u32,
                    h: height as u32,
                };

                // 1. Solid Xrgb with varying garbage byte 3.
                let mut src = vec![0x55u8; byte_len];
                for y in 0..height {
                    for x in 0..width {
                        let off = y * stride + x * 4;
                        src[off] = 0x10;
                        src[off + 1] = 0x20;
                        src[off + 2] = 0x30;
                        src[off + 3] = ((y * 13 + x * 7) & 0xFF) as u8;
                    }
                }
                let mut shadow: Vec<u8> = Vec::with_capacity(byte_len);
                let sr = ShadowRows {
                    base: shadow.as_mut_ptr(),
                    stride,
                };
                let p = src.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, byte_len) };
                let meta = unsafe {
                    let m = copy_and_scan_rect_to_shadow_with::<ScalarKernel, false>(
                        ScalarKernel,
                        bp,
                        &sr,
                        stride,
                        rect,
                    );
                    zero_shadow_row_padding(&sr, width_bytes, rect);
                    shadow.set_len(byte_len);
                    m
                };
                assert_eq!(
                    meta,
                    RectScanMeta {
                        solid_px: Some([0x10, 0x20, 0x30, 0xFF]),
                        all_opaque: true,
                    }
                );
                for y in 0..height {
                    let row = y * stride;
                    assert_eq!(
                        &shadow[row..row + width_bytes],
                        &src[row..row + width_bytes]
                    );
                }

                // 2. Non-solid Argb opaque, then single pixel alpha = 0xFE at end.
                for y in 0..height {
                    for x in 0..width {
                        let off = y * stride + x * 4;
                        src[off] = (x & 0xFF) as u8;
                        src[off + 1] = (y & 0xFF) as u8;
                        src[off + 2] = 0x77;
                        src[off + 3] = 0xFF;
                    }
                }
                let mut shadow2: Vec<u8> = Vec::with_capacity(byte_len);
                let sr2 = ShadowRows {
                    base: shadow2.as_mut_ptr(),
                    stride,
                };
                let p2 = src.as_ptr();
                let bp2 = unsafe { BufferPointer::new(&p2, byte_len) };
                let meta2 = unsafe {
                    let m = copy_and_scan_rect_to_shadow_with::<ScalarKernel, true>(
                        ScalarKernel,
                        bp2,
                        &sr2,
                        stride,
                        rect,
                    );
                    zero_shadow_row_padding(&sr2, width_bytes, rect);
                    shadow2.set_len(byte_len);
                    m
                };
                assert!(meta2.all_opaque);
                assert_eq!(meta2.solid_px.is_some(), width == 1 && height == 1);

                src[(height - 1) * stride + (width - 1) * 4 + 3] = 0xFE;
                let mut shadow3: Vec<u8> = Vec::with_capacity(byte_len);
                let sr3 = ShadowRows {
                    base: shadow3.as_mut_ptr(),
                    stride,
                };
                let p3 = src.as_ptr();
                let bp3 = unsafe { BufferPointer::new(&p3, byte_len) };
                let meta3 = unsafe {
                    let m = copy_and_scan_rect_to_shadow_with::<ScalarKernel, true>(
                        ScalarKernel,
                        bp3,
                        &sr3,
                        stride,
                        rect,
                    );
                    zero_shadow_row_padding(&sr3, width_bytes, rect);
                    shadow3.set_len(byte_len);
                    m
                };
                assert!(!meta3.all_opaque);
                assert_eq!(meta3.solid_px.is_some(), width == 1 && height == 1);

                // 3. Matching 3-point pre-probe pixels `(0,0)`, `(w/2,h/2)`, `(w-1,h-1)`
                //    with a differing 4th non-probe pixel must still report non-solid / non-opaque.
                if width > 1 && height > 1 {
                    for y in 0..height {
                        for x in 0..width {
                            let off = y * stride + x * 4;
                            src[off..off + 4].copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]);
                        }
                    }
                    // Perturb top-right `(width - 1, 0)` (not one of the 3 probe points).
                    let fourth_off = (width - 1) * 4;
                    src[fourth_off] ^= 0x01;
                    let mut shadow4: Vec<u8> = Vec::with_capacity(byte_len);
                    let sr4 = ShadowRows {
                        base: shadow4.as_mut_ptr(),
                        stride,
                    };
                    let p4 = src.as_ptr();
                    let bp4 = unsafe { BufferPointer::new(&p4, byte_len) };
                    let meta4_xrgb = unsafe {
                        copy_and_scan_rect_to_shadow_with::<ScalarKernel, false>(
                            ScalarKernel,
                            bp4,
                            &sr4,
                            stride,
                            rect,
                        )
                    };
                    assert_eq!(
                        meta4_xrgb,
                        RectScanMeta {
                            solid_px: None,
                            all_opaque: true,
                        }
                    );
                    src[fourth_off] ^= 0x01;
                    src[fourth_off + 3] = 0xFE;
                    let p5 = src.as_ptr();
                    let bp5 = unsafe { BufferPointer::new(&p5, byte_len) };
                    let meta4_argb = unsafe {
                        copy_and_scan_rect_to_shadow_with::<ScalarKernel, true>(
                            ScalarKernel,
                            bp5,
                            &sr4,
                            stride,
                            rect,
                        )
                    };
                    assert_eq!(
                        meta4_argb,
                        RectScanMeta {
                            solid_px: None,
                            all_opaque: false,
                        }
                    );
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_in_register_solid_cache_non_pollution_and_argb_opaque_roundtrip() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };

            let run_full = |frame: &[u8],
                            meta: BufferMetadata,
                            mode: CompressionMode,
                            shadow: Option<&mut ShadowBuffer>,
                            srv: Option<&mut ServerTileCache>,
                            cli: Option<&mut ClientTileCache>,
                            comp: &mut ShardingCompressor,
                            decomp: &mut ShardingDecompressor|
             -> (Vec<CompressedTile>, Vec<u8>) {
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, frame.len()) };
                let mut st = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(7),
                    meta,
                    mode,
                    shadow,
                    srv,
                    comp,
                    |s| st = Some(s),
                );
                let mut wire = Vec::new();
                st.unwrap().framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).unwrap();
                let mut out = vec![0xA5u8; frame.len()];
                decompress_and_unfilter_streaming(validated, decomp, &mut out, cli).unwrap();
                (tiles, out)
            };

            let run_inc = |frame: &[u8],
                           meta: BufferMetadata,
                           mode: CompressionMode,
                           shadow: &mut ShadowBuffer,
                           srv: Option<&mut ServerTileCache>,
                           cli: Option<&mut ClientTileCache>,
                           canvas: &mut [u8],
                           comp: &mut ShardingCompressor,
                           decomp: &mut ShardingDecompressor|
             -> Vec<CompressedTile> {
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, frame.len()) };
                let mut st = None;
                let res = filter_and_compress_incremental_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(7),
                    meta,
                    mode,
                    None,
                    shadow,
                    srv,
                    comp,
                    |s| st = Some(s),
                );
                if res.is_unchanged() {
                    return Vec::new();
                }
                let mut wire = Vec::new();
                st.unwrap().framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let header = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated =
                    ValidatedTileStream::new(CompressedTileStream { header, tiles: rx }).unwrap();
                if let Some(sr) = validated.scroll_rect() {
                    apply_scroll_rect_in_place(canvas, meta.stride as usize, sr);
                }
                decompress_and_unfilter_streaming(validated, decomp, canvas, cli).unwrap();
                tiles
            };

            // 1. All-solid frames (Xrgb with garbage X bytes, Argb opaque, Argb translucent):
            //    MUST allocate 0 ServerTileCache and 0 ClientTileCache slots!
            let width = 512usize;
            let height = 512usize;
            let stride = width * 4;
            let len = height * stride;
            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();

            for (fmt, px, garbage_x) in [
                (BufferFormat::Xrgb8888, [0x11, 0x22, 0x33, 0xFF], true),
                (BufferFormat::Argb8888, [0x44, 0x55, 0x66, 0xFF], false),
                (BufferFormat::Argb8888, [0x77, 0x88, 0x99, 0x80], false),
            ] {
                let meta = BufferMetadata {
                    width: width as i32,
                    height: height as i32,
                    stride: stride as i32,
                    format: fmt,
                };
                let mut frame = vec![0u8; len];
                for (i, dst) in frame.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    *dst = [
                        px[0],
                        px[1],
                        px[2],
                        if garbage_x { (i * 19) as u8 } else { px[3] },
                    ];
                }
                let (tiles, out) = run_full(
                    &frame,
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut shadow),
                    Some(&mut srv),
                    Some(&mut cli),
                    compressor,
                    decompressor,
                );
                assert_eq!(srv.occupied_slots(), 0, "solid frame polluted server cache");
                assert_eq!(cli.occupied_slots(), 0, "solid frame polluted client cache");
                assert_eq!(tiles.len(), 4);
                for t in &tiles {
                    assert!(t.cache_put.is_none());
                    assert_eq!(t.payload, TilePayload::Leaf(LeafPayload::Solid(px)));
                }
                assert_canvas_matches_frame(&out, &frame, meta, 0xA5);
            }

            // 2. Mixed 512x512 frame:
            //    - (0,0) 256x256: wholly solid -> 0 cache slots
            //    - (256,0) 256x256: 2 solid quadrants + 2 complex quadrants -> 1 cache slot, hits on return
            //    - (0,256) 256x256: all 4 complex quadrants -> 1 cache slot, hits on return
            //    - (256,256) 256x256: wholly solid translucent (or solid Xrgb) -> 0 cache slots
            let meta_xrgb = BufferMetadata {
                width: 512,
                height: 512,
                stride: 512 * 4,
                format: BufferFormat::Xrgb8888,
            };
            let mut mixed = vec![0u8; len];
            for px in mixed.as_chunks_mut::<4>().0 {
                *px = [0x10, 0x20, 0x30, 0xFF];
            }
            fill_ui_text(&mut mixed, 512, tr(256, 0, 128, 128), false);
            fill_smooth_photo(&mut mixed, 512, tr(384, 128, 128, 128), 0xFF);
            fill_ui_text(&mut mixed, 512, tr(0, 256, 256, 256), false);

            let (tiles_m1, mut canvas) = run_full(
                &mixed,
                meta_xrgb,
                CompressionMode::Lossless,
                Some(&mut shadow),
                Some(&mut srv),
                Some(&mut cli),
                compressor,
                decompressor,
            );
            assert_eq!(
                srv.occupied_slots(),
                2,
                "only the 2 non-solid tiles should cache"
            );
            assert_eq!(cli.occupied_slots(), 2);
            let n_solid = tiles_m1
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::Leaf(LeafPayload::Solid(_))))
                .count();
            assert_eq!(n_solid, 2);
            assert_canvas_matches_frame(&canvas, &mixed, meta_xrgb, 0xA5);

            // Switch to solid screen (0 slots added), then back to `mixed`:
            // the 2 non-solid tiles hit, the 2 solid tiles emit Solid without touching cache!
            let solid_b = [0xEEu8, 0xDD, 0xCC, 0xFF].repeat(512 * 512);
            let tiles_b = run_inc(
                &solid_b,
                meta_xrgb,
                CompressionMode::Lossless,
                &mut shadow,
                Some(&mut srv),
                Some(&mut cli),
                &mut canvas,
                compressor,
                decompressor,
            );
            assert!(tiles_b.iter().all(|t| t.cache_put.is_none()
                && matches!(t.payload, TilePayload::Leaf(LeafPayload::Solid(_)))));
            assert_eq!(srv.occupied_slots(), 2);
            assert_canvas_matches_frame(&canvas, &solid_b, meta_xrgb, 0xA5);

            let tiles_ret = run_inc(
                &mixed,
                meta_xrgb,
                CompressionMode::Lossless,
                &mut shadow,
                Some(&mut srv),
                Some(&mut cli),
                &mut canvas,
                compressor,
                decompressor,
            );
            let ret_hits = tiles_ret
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                .count();
            let ret_solids = tiles_ret
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::Leaf(LeafPayload::Solid(_))))
                .count();
            assert_eq!(ret_hits, 2, "both complex + mixed 256x256 tiles must hit");
            assert_eq!(ret_solids, 2, "both solid 256x256 tiles must emit Solid");
            assert_eq!(srv.occupied_slots(), 2);
            assert_canvas_matches_frame(&canvas, &mixed, meta_xrgb, 0xA5);

            // Partial incremental update turning one quadrant of (0,256) solid:
            // emits Solid with cache_put = None and does not allocate a cache slot.
            let mut partial = mixed.clone();
            for y in 256..384 {
                for x in 0..128 {
                    let off = (y * 512 + x) * 4;
                    partial[off..off + 4].copy_from_slice(&[0x55, 0x66, 0x77, 0xFF]);
                }
            }
            let tiles_part = run_inc(
                &partial,
                meta_xrgb,
                CompressionMode::Lossless,
                &mut shadow,
                Some(&mut srv),
                Some(&mut cli),
                &mut canvas,
                compressor,
                decompressor,
            );
            assert_eq!(tiles_part.len(), 1);
            assert!(tiles_part[0].cache_put.is_none());
            assert_eq!(
                tiles_part[0].payload,
                TilePayload::Leaf(LeafPayload::Solid([0x55, 0x66, 0x77, 0xFF]))
            );
            assert_eq!(srv.occupied_slots(), 2);
            assert_canvas_matches_frame(&canvas, &partial, meta_xrgb, 0xA5);

            // 3. Argb8888 all-opaque & mixed opaque/translucent bit-exact roundtrips
            //    across Lossless & Hybrid, full-frame (cold-fill, warm 2-pass,
            //    no-cache w/ shadow, no-cache w/o shadow), and incremental.
            let meta_argb = BufferMetadata {
                width: 512,
                height: 512,
                stride: 512 * 4,
                format: BufferFormat::Argb8888,
            };
            let mut argb_opaque = vec![0u8; len];
            fill_ui_text(&mut argb_opaque, 512, tr(0, 0, 256, 256), false);
            fill_smooth_photo(&mut argb_opaque, 512, tr(256, 0, 256, 256), 0xFF);
            fill_ui_text(&mut argb_opaque, 512, tr(0, 256, 128, 128), false);
            fill_smooth_photo(&mut argb_opaque, 512, tr(128, 256, 128, 128), 0xFF);
            fill_smooth_photo(&mut argb_opaque, 512, tr(0, 384, 256, 128), 0xFF);
            for y in 256..512 {
                for x in 256..512 {
                    let off = (y * 512 + x) * 4;
                    argb_opaque[off..off + 4].copy_from_slice(&[0x22, 0x33, 0x44, 0xFF]);
                }
            }
            // Force all alpha bytes to 0xFF.
            for px in argb_opaque.as_chunks_mut::<4>().0 {
                px[3] = 0xFF;
            }

            // Mixed translucent Argb8888: solid translucent, single alpha=0xFE pixel,
            // translucent gradient photo, and opaque text quadrants in the same frame!
            let mut argb_mixed = argb_opaque.clone();
            // (0,0) tile: single pixel with alpha = 0xFE at (127, 127) -> non-opaque!
            argb_mixed[(127 * 512 + 127) * 4 + 3] = 0xFE;
            // (256,0) tile: translucent photo (alpha = 0xA0) in TL, opaque photo elsewhere.
            fill_smooth_photo(&mut argb_mixed, 512, tr(256, 0, 128, 128), 0xA0);
            // (0,256) tile: solid translucent quadrant in TL, photo in TR, opaque text in bottom.
            for y in 256..384 {
                for x in 0..128 {
                    let off = (y * 512 + x) * 4;
                    argb_mixed[off..off + 4].copy_from_slice(&[0x12, 0x34, 0x56, 0x80]);
                }
            }
            // (256,256) tile: wholly solid translucent.
            for y in 256..512 {
                for x in 256..512 {
                    let off = (y * 512 + x) * 4;
                    argb_mixed[off..off + 4].copy_from_slice(&[0x99, 0xAA, 0xBB, 0x80]);
                }
            }

            let hybrid = CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            };

            for mode in [CompressionMode::Lossless, hybrid] {
                let mut srv_a = ServerTileCache::new();
                let mut cli_a = ClientTileCache::new();
                let mut sh_a = ShadowBuffer::new();

                // (a) Cold-fill opaque Argb8888: verify omitted alpha on wire + exact alpha=0xFF decode.
                let (t_opq, out_opq) = run_full(
                    &argb_opaque,
                    meta_argb,
                    mode,
                    Some(&mut sh_a),
                    Some(&mut srv_a),
                    Some(&mut cli_a),
                    compressor,
                    decompressor,
                );
                for t in &t_opq {
                    let check_leaf = |leaf: &LeafPayload| match leaf {
                        LeafPayload::Solid(px) => assert_eq!(px[3], 0xFF),
                        LeafPayload::Palette { has_alpha, .. } => assert!(!has_alpha),
                        LeafPayload::AosZstd { alpha, .. } | LeafPayload::Jpeg { alpha, .. } => {
                            assert_eq!(*alpha, AlphaChannel::Omitted);
                        },
                    };
                    match &t.payload {
                        TilePayload::Leaf(l) => check_leaf(l),
                        TilePayload::Split(subs) => {
                            for s in subs {
                                check_leaf(&s.payload);
                            }
                        },
                        TilePayload::CacheHit(_) => {},
                    }
                }
                if mode == CompressionMode::Lossless {
                    assert_eq!(out_opq, argb_opaque);
                } else {
                    assert!(out_opq.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF));
                }

                // (b) Warm 2-pass mixed translucent Argb8888.
                let (_, out_mix) = run_full(
                    &argb_mixed,
                    meta_argb,
                    mode,
                    Some(&mut sh_a),
                    Some(&mut srv_a),
                    Some(&mut cli_a),
                    compressor,
                    decompressor,
                );
                if mode == CompressionMode::Lossless {
                    assert_eq!(out_mix, argb_mixed);
                } else {
                    // Exact lossless alpha channel on every single pixel even in Hybrid mode!
                    for (got, exp) in out_mix
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(argb_mixed.as_chunks::<4>().0.iter())
                    {
                        assert_eq!(got[3], exp[3], "Hybrid Argb alpha mismatch");
                    }
                }

                // (c) Incremental transition mixed -> opaque -> mixed.
                let mut canv_a = out_mix;
                run_inc(
                    &argb_opaque,
                    meta_argb,
                    mode,
                    &mut sh_a,
                    Some(&mut srv_a),
                    Some(&mut cli_a),
                    &mut canv_a,
                    compressor,
                    decompressor,
                );
                if mode == CompressionMode::Lossless {
                    assert_eq!(canv_a, argb_opaque);
                } else {
                    assert!(canv_a.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF));
                }
                run_inc(
                    &argb_mixed,
                    meta_argb,
                    mode,
                    &mut sh_a,
                    Some(&mut srv_a),
                    Some(&mut cli_a),
                    &mut canv_a,
                    compressor,
                    decompressor,
                );
                if mode == CompressionMode::Lossless {
                    assert_eq!(canv_a, argb_mixed);
                } else {
                    for (got, exp) in canv_a
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(argb_mixed.as_chunks::<4>().0.iter())
                    {
                        assert_eq!(got[3], exp[3]);
                    }
                }

                // (d) No-cache with shadow (`copy_and_scan_rect_to_shadow`) and without shadow.
                let mut sh_nc = ShadowBuffer::new();
                let (_, out_nc_sh) = run_full(
                    &argb_opaque,
                    meta_argb,
                    mode,
                    Some(&mut sh_nc),
                    None,
                    None,
                    compressor,
                    decompressor,
                );
                let (_, out_nc_mix) = run_full(
                    &argb_mixed,
                    meta_argb,
                    mode,
                    Some(&mut sh_nc),
                    None,
                    None,
                    compressor,
                    decompressor,
                );
                let (_, out_nc_nosh) = run_full(
                    &argb_mixed,
                    meta_argb,
                    mode,
                    None,
                    None,
                    None,
                    compressor,
                    decompressor,
                );
                if mode == CompressionMode::Lossless {
                    assert_eq!(out_nc_sh, argb_opaque);
                    assert_eq!(out_nc_mix, argb_mixed);
                    assert_eq!(out_nc_nosh, argb_mixed);
                } else {
                    assert!(out_nc_sh.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF));
                    for (got, exp) in out_nc_mix
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(argb_mixed.as_chunks::<4>().0.iter())
                    {
                        assert_eq!(got[3], exp[3]);
                    }
                    for (got, exp) in out_nc_nosh
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(argb_mixed.as_chunks::<4>().0.iter())
                    {
                        assert_eq!(got[3], exp[3]);
                    }
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_client_tile_cache_evict_resync_capacity_retention_and_rss_cap() {
        let mut cli = ClientTileCache::new();
        let rect_256 = TileRect {
            x: 0,
            y: 0,
            w: 256,
            h: 256,
        };
        let rect_128 = TileRect {
            x: 0,
            y: 0,
            w: 128,
            h: 128,
        };
        let rect_160 = TileRect {
            x: 0,
            y: 0,
            w: 160,
            h: 256,
        };
        let put_solid = |cli: &mut ClientTileCache,
                         worker_idx: usize,
                         slot_id: u16,
                         rect: TileRect,
                         px: Vec4u8| {
            let slots = cli.sync_slots();
            // SAFETY: Exclusive single-threaded ownership of `slot_id`.
            let sf = unsafe {
                ClientTileCache::prepare_slot_target_uninit(&slots, worker_idx, slot_id, rect)
            };
            for ry in 0..rect.h as usize {
                // SAFETY: Row lies strictly inside `sf`.
                let row = unsafe { slice::from_raw_parts_mut(sf.row_ptr(ry, 0), rect.w as usize) };
                // SAFETY: `STREAM = false` requires no extra alignment.
                unsafe { ScalarKernel.splat_row::<false>(row, px) };
            }
            // SAFETY: Every pixel in `rect` was initialized above.
            let _ = unsafe { ClientTileCache::commit_slot(&slots, slot_id, rect, false) };
        };

        // 1. `evict()` retains backing `Vec` capacity and reuses the exact same
        //    backing allocation on subsequent `CachePut` of the same tile size
        //    (via per-worker LIFO recycle shards for `256 KiB` tiles and in-place
        //    for smaller edge/quadrant tiles).
        put_solid(&mut cli, 0, 10, rect_256, Vec4u8(1, 2, 3, 0xFF));
        put_solid(&mut cli, 0, 12, rect_128, Vec4u8(9, 8, 7, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        let (ptr0, cap0, occ0) = cli.slot_buf_info(10);
        let (ptr12, cap12, occ12) = cli.slot_buf_info(12);
        assert!(occ0 && occ12);
        assert_eq!(cap0, 256 * 256);
        assert_eq!(cap12, 128 * 128);

        cli.evict(10).unwrap();
        cli.evict(12).unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        assert!(!cli.slot_buf_info(10).2);
        assert_eq!(cli.slot_buf_info(12), (ptr12, cap12, false));
        assert_eq!(cli.capacity_bytes(), (256 * 256 + 128 * 128) * 4);

        put_solid(&mut cli, 0, 10, rect_256, Vec4u8(4, 5, 6, 0xFF));
        put_solid(&mut cli, 0, 12, rect_128, Vec4u8(6, 5, 4, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(
            cli.slot_buf_info(10),
            (ptr0, cap0, true),
            "evicted 256 KiB slot must reuse exact backing allocation with zero reallocation"
        );
        assert_eq!(
            cli.slot_buf_info(12),
            (ptr12, cap12, true),
            "evicted 64 KiB slot must reuse exact backing allocation in place"
        );

        // Cross-slot 256 KiB reuse and 256 KiB -> smaller tile overwrite
        // recycling back into the worker's LIFO shard.
        cli.evict(10).unwrap();
        put_solid(&mut cli, 0, 42, rect_256, Vec4u8(3, 3, 3, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.slot_buf_info(42), (ptr0, cap0, true));
        put_solid(&mut cli, 0, 42, rect_128, Vec4u8(5, 5, 5, 0xFF));
        put_solid(&mut cli, 0, 10, rect_256, Vec4u8(4, 5, 6, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.slot_buf_info(10), (ptr0, cap0, true));
        cli.evict(42).unwrap();

        // 2. In-band `reset_tile_cache = true` retains capacity across resync
        //    and returns each `256 KiB` buffer to its owning worker's shard
        //    (preserving L2/L3 worker affinity even when slot IDs swap).
        put_solid(&mut cli, 1, 11, rect_256, Vec4u8(7, 8, 9, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        let (ptr11, cap11, _) = cli.slot_buf_info(11);
        cli.apply_header_ops(true, &[]).unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        assert_eq!(cli.capacity_bytes(), (2 * 256 * 256 + 2 * 128 * 128) * 4);

        put_solid(&mut cli, 1, 10, rect_256, Vec4u8(10, 11, 12, 0xFF));
        put_solid(&mut cli, 0, 11, rect_256, Vec4u8(13, 14, 15, 0xFF));
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(
            cli.slot_buf_info(10),
            (ptr11, cap11, true),
            "worker 1 must reclaim its own L2-hot buffer even on a different slot"
        );
        assert_eq!(
            cli.slot_buf_info(11),
            (ptr0, cap0, true),
            "worker 0 must reclaim its own L2-hot buffer even on a different slot"
        );

        // 3. `ClientTileCache::clear()` releases ALL backing memory (even when
        //    slots were already marked empty via `reset_in_band()`).
        cli.reset_in_band();
        assert_eq!(cli.occupied_slots(), 0);
        assert!(cli.capacity_bytes() > 0);
        cli.clear();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);
        assert_eq!(cli.capacity_bytes(), 0);

        // 4. `recompute_and_verify_bytes()` strictly trims recycled stacks and
        //    unoccupied warm slot capacities whenever total capacity exceeds
        //    `MAX_TILE_CACHE_BYTES`.
        for s in 500u16..540 {
            put_solid(&mut cli, 0, s, rect_128, Vec4u8(0x11, 0x22, 0x33, 0xFF));
        }
        for s in 540u16..550 {
            put_solid(
                &mut cli,
                (s as usize) % 4,
                s,
                rect_256,
                Vec4u8(0x22, 0x33, 0x44, 0xFF),
            );
        }
        cli.recompute_and_verify_bytes().unwrap();
        for s in 500u16..550 {
            cli.evict(s).unwrap();
        }
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.capacity_bytes(), 40 * 64 * 1024 + 10 * 256 * 1024);

        for s in 0u16..256 {
            put_solid(
                &mut cli,
                (s as usize) % 4,
                s,
                rect_256,
                Vec4u8(0x44, 0x55, 0x66, 0xFF),
            );
        }
        assert!(
            cli.capacity_bytes() > MAX_TILE_CACHE_BYTES,
            "precondition: warm unoccupied slots push raw capacity over 64 MiB"
        );
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.occupied_slots(), 256);
        assert_eq!(cli.used_bytes(), MAX_TILE_CACHE_BYTES);
        assert!(
            cli.capacity_bytes() <= MAX_TILE_CACHE_BYTES,
            "recompute_and_verify_bytes must enforce capacity_bytes <= MAX_TILE_CACHE_BYTES, got {}",
            cli.capacity_bytes()
        );

        // 5. Occupied slack cap: overwrite 160 slots (256 KiB cap -> 160 KiB
        //    live, recycling the 256 KiB buffers into worker shards) and fill
        //    60 new 256 KiB slots within the 64 MiB live budget -> raw capacity
        //    exceeds 64 MiB; `recompute_and_verify_bytes()` must strictly
        //    enforce `capacity_bytes <= MAX_TILE_CACHE_BYTES` while preserving
        //    every occupied slot's pixels.
        for s in 0u16..160 {
            put_solid(&mut cli, 0, s, rect_160, Vec4u8(0x77, 0x88, 0x99, 0xFF));
        }
        for s in 256u16..316 {
            put_solid(&mut cli, 0, s, rect_256, Vec4u8(0xAA, 0xBB, 0xCC, 0xFF));
        }
        assert!(cli.capacity_bytes() > MAX_TILE_CACHE_BYTES);
        cli.recompute_and_verify_bytes().unwrap();
        assert!(cli.used_bytes() <= MAX_TILE_CACHE_BYTES);
        assert!(cli.capacity_bytes() <= MAX_TILE_CACHE_BYTES);
        let hit = cli
            .validate_hit(
                0,
                CacheHitRef {
                    slot_id: 0,
                    region: crate::serialization::framing::CacheHitRegion::Full,
                },
                rect_160,
                false,
            )
            .unwrap();
        let (src, stride_px) = cli.slot_hit_src(hit);
        assert_eq!(stride_px, 160);
        assert!(src.iter().all(|&p| p == Vec4u8(0x77, 0x88, 0x99, 0xFF)));

        // 6. Occupied non-256 KiB slack shrink_to_fit(): fill 400 slots with
        //    `160x256` (`160 KiB` = `64 MiB` total), overwrite 200 of them with
        //    `128x256` (`128 KiB` live, below the 2x shrink threshold -> keeps
        //    `160 KiB` capacity), and fill 50 new `128x256` slots (`6.4 MiB`).
        //    Live bytes = `60.8 MiB <= 64 MiB`, raw capacity = `70.4 MiB > 64 MiB`
        //    with zero recycled or unoccupied buffers; `recompute_and_verify_bytes`
        //    must shrink occupied slack to enforce `<= 64 MiB` with bit-exact pixels.
        cli.clear();
        let rect_128x256 = TileRect {
            x: 0,
            y: 0,
            w: 128,
            h: 256,
        };
        for s in 0u16..400 {
            put_solid(&mut cli, 0, s, rect_160, Vec4u8(0x12, 0x34, 0x56, 0xFF));
        }
        cli.recompute_and_verify_bytes().unwrap();
        for s in 0u16..200 {
            put_solid(&mut cli, 0, s, rect_128x256, Vec4u8(0x65, 0x43, 0x21, 0xFF));
        }
        for s in 400u16..450 {
            put_solid(&mut cli, 0, s, rect_128x256, Vec4u8(0x98, 0x76, 0x54, 0xFF));
        }
        assert!(cli.capacity_bytes() > MAX_TILE_CACHE_BYTES);
        cli.recompute_and_verify_bytes().unwrap();
        assert!(cli.used_bytes() <= MAX_TILE_CACHE_BYTES);
        assert!(cli.capacity_bytes() <= MAX_TILE_CACHE_BYTES);
        let hit2 = cli
            .validate_hit(
                0,
                CacheHitRef {
                    slot_id: 0,
                    region: crate::serialization::framing::CacheHitRegion::Full,
                },
                rect_128x256,
                false,
            )
            .unwrap();
        let (src2, stride2) = cli.slot_hit_src(hit2);
        assert_eq!(stride2, 128);
        assert!(src2.iter().all(|&p| p == Vec4u8(0x65, 0x43, 0x21, 0xFF)));

        cli.clear();
        assert_eq!(cli.capacity_bytes(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_direct_cold_fill_concurrent_dedup_hybrid_disconnect_and_overflow() {
        with_test_codec(|compressor, decompressor| {
            let tr = |x, y, w, h| TileRect { x, y, w, h };

            // 1. Concurrent direct cold-fill with Hybrid lossless promotion:
            //    text tiles & text quadrants promote to Lossless, photo JPEG
            //    tiles remain Lossy, solid tiles allocate 0 slots, and duplicate
            //    tiles deduplicate intra-frame.
            let width = 768usize;
            let height = 512usize;
            let stride = width * 4;
            let len = height * stride;
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format: BufferFormat::Xrgb8888,
            };
            let mut frame = vec![0u8; len];
            // (0,0): UI text 256x256 (promotes Lossless)
            fill_ui_text(&mut frame, width, tr(0, 0, 256, 256), false);
            // (256,0): exact duplicate of (0,0) -> intra-frame CacheHit!
            for y in 0..256 {
                let row = y * stride;
                let copy: Vec<u8> = frame[row..row + 256 * 4].to_vec();
                frame[row + 256 * 4..row + 512 * 4].copy_from_slice(&copy);
            }
            // (512,0): wholly solid -> 0 cache slots!
            for y in 0..256 {
                for x in 512..768 {
                    let off = y * stride + x * 4;
                    frame[off..off + 4].copy_from_slice(&[0x20, 0x40, 0x60, 0xFF]);
                }
            }
            // (0,256): pure smooth photo -> JPEG (stays Lossy)
            fill_smooth_photo(&mut frame, width, tr(0, 256, 256, 256), 0xFF);
            // (256,256): mixed Split (TL text promotes Lossless, rest photo JPEG)
            fill_ui_text(&mut frame, width, tr(256, 256, 128, 128), false);
            fill_smooth_photo(&mut frame, width, tr(384, 256, 128, 128), 0xFF);
            fill_smooth_photo(&mut frame, width, tr(256, 384, 256, 128), 0xFF);
            // (512,256): unique UI text
            fill_ui_text(&mut frame, width, tr(512, 256, 256, 256), false);
            frame[(260 * width + 520) * 4] ^= 0x7F;

            let hybrid = CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            };
            let mut srv = ServerTileCache::new();
            let mut cli = ClientTileCache::new();
            let mut shadow = ShadowBuffer::new();
            let mut canvas = vec![0xAAu8; len];

            let send_recv = |f: &[u8],
                             full: bool,
                             mode: CompressionMode,
                             sh: &mut ShadowBuffer,
                             srv: &mut ServerTileCache,
                             cli: &mut ClientTileCache,
                             canv: &mut [u8],
                             comp: &mut ShardingCompressor,
                             decomp: &mut ShardingDecompressor|
             -> (RawBufferHeader, Vec<CompressedTile>) {
                let p = f.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                let mut st = None;
                if full {
                    filter_and_compress_full_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(11),
                        meta,
                        mode,
                        Some(sh),
                        Some(srv),
                        comp,
                        |s| st = Some(s),
                    );
                } else {
                    filter_and_compress_incremental_streaming(
                        bp,
                        ClientId(1),
                        WlSurfaceId(11),
                        meta,
                        mode,
                        None,
                        sh,
                        Some(srv),
                        comp,
                        |s| st = Some(s),
                    );
                }
                let mut wire = Vec::new();
                st.unwrap().framed_write(&mut wire).unwrap();
                let mut rx_opt = None;
                CompressedTileStream::read_from_stream(&mut wire.as_slice(), |s| {
                    rx_opt = Some(s);
                    Ok(())
                })
                .unwrap();
                let stream = rx_opt.unwrap();
                let hdr = stream.header.clone();
                let tiles: Vec<CompressedTile> = stream
                    .tiles
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                let (tx, rx) = crossbeam_channel::unbounded();
                for t in &tiles {
                    tx.send(Ok(t.clone())).unwrap();
                }
                drop(tx);
                let validated = ValidatedTileStream::new(CompressedTileStream {
                    header: hdr.clone(),
                    tiles: rx,
                })
                .unwrap();
                decompress_and_unfilter_streaming(validated, decomp, canv, Some(cli)).unwrap();
                (hdr, tiles)
            };

            let (_, tiles_cold) = send_recv(
                &frame,
                true,
                hybrid,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            // 6 base tiles: 1 solid (0 slots), 2 identical text (1 slot + 1 hit),
            // 1 photo, 1 split, 1 unique text -> exactly 4 slots!
            assert_eq!(srv.occupied_slots(), 4);
            assert_eq!(cli.occupied_slots(), 4);
            assert_eq!(
                tiles_cold
                    .iter()
                    .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                    .count(),
                1
            );
            assert_eq!(
                tiles_cold
                    .iter()
                    .filter(|t| matches!(t.payload, TilePayload::Leaf(LeafPayload::Solid(_))))
                    .count(),
                1
            );

            // Switch to solid screen (no cache pollution), then return in
            // Lossless mode: promoted text tiles & promoted Split TL quadrant
            // MUST hit, while JPEG photo tile MUST miss and decode bit-exact!
            let solid_bg = [0x10u8, 0x10, 0x10, 0xFF].repeat(width * height);
            send_recv(
                &solid_bg,
                false,
                CompressionMode::Lossless,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let (hdr_ll, tiles_ll) = send_recv(
                &frame,
                false,
                CompressionMode::Lossless,
                &mut shadow,
                &mut srv,
                &mut cli,
                &mut canvas,
                compressor,
                decompressor,
            );
            let hit_rects: Vec<TileRect> = tiles_ll
                .iter()
                .filter(|t| matches!(t.payload, TilePayload::CacheHit(_)))
                .map(|t| hdr_ll.tile_rects[t.tile_idx as usize])
                .collect();
            assert!(hit_rects.contains(&tr(0, 0, 256, 256)));
            assert!(hit_rects.contains(&tr(256, 0, 256, 256)));
            assert!(hit_rects.contains(&tr(512, 256, 256, 256)));
            assert!(hit_rects.contains(&tr(256, 256, 128, 128)));
            assert!(!hit_rects.contains(&tr(0, 256, 256, 256)));
            assert_canvas_matches_frame(&canvas, &frame, meta, 0);

            // 2. Early disconnect mid-cold-fill resets `ServerTileCache` and
            //    marks `reset_tile_cache = true` on the next frame while still
            //    completely initializing `shadow`.
            let mut srv_disc = ServerTileCache::new();
            let mut sh_disc = ShadowBuffer::new();
            {
                let p = frame.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, len) };
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(11),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut sh_disc),
                    Some(&mut srv_disc),
                    compressor,
                    drop,
                );
            }
            assert_eq!(sh_disc.as_slice(), frame.as_slice());
            assert_eq!(srv_disc.occupied_slots(), 0);
            assert_eq!(srv_disc.used_bytes(), 0);
            let mut cli_disc = ClientTileCache::new();
            let (hdr_after_disc, _) = send_recv(
                &frame,
                true,
                CompressionMode::Lossless,
                &mut sh_disc,
                &mut srv_disc,
                &mut cli_disc,
                &mut canvas,
                compressor,
                decompressor,
            );
            assert!(
                hdr_after_disc.reset_tile_cache,
                "first frame after mid-cold-fill disconnect must set reset_tile_cache = true"
            );
            assert_canvas_matches_frame(&canvas, &frame, meta, 0);

            // 3. Direct cold-fill overflow: > MAX_TILE_CACHE_SLOTS and
            //    > MAX_TILE_CACHE_BYTES cleanly return `TileCacheAction::Uncached`.
            let mut srv_slots = ServerTileCache::new();
            srv_slots.begin_cold_fill();
            let r16 = tr(0, 0, 16, 16);
            let mut puts = 0usize;
            let mut uncached = 0usize;
            for i in 0..(MAX_TILE_CACHE_SLOTS + 8) {
                let mut h = [0u8; 16];
                h[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
                match srv_slots.claim_cold_tile(r16, h, None, false, TileQuality::Lossless) {
                    TileCacheAction::Put(_) => puts += 1,
                    TileCacheAction::Uncached => uncached += 1,
                    other => panic!("unexpected action {other:?}"),
                }
            }
            assert_eq!(puts, MAX_TILE_CACHE_SLOTS);
            assert_eq!(uncached, 8);
            assert_eq!(srv_slots.occupied_slots(), MAX_TILE_CACHE_SLOTS);

            let mut srv_bytes = ServerTileCache::new();
            srv_bytes.begin_cold_fill();
            let r256 = tr(0, 0, 256, 256);
            let max_256_slots = MAX_TILE_CACHE_BYTES / (256 * 256 * 4);
            let mut b_puts = 0usize;
            let mut b_uncached = 0usize;
            for i in 0..(max_256_slots + 5) {
                let mut h = [0u8; 16];
                h[..8].copy_from_slice(&(i as u64 + 10_000).to_le_bytes());
                match srv_bytes.claim_cold_tile(r256, h, None, false, TileQuality::Lossless) {
                    TileCacheAction::Put(_) => b_puts += 1,
                    TileCacheAction::Uncached => b_uncached += 1,
                    other => panic!("unexpected action {other:?}"),
                }
            }
            assert_eq!(b_puts, max_256_slots);
            assert_eq!(b_uncached, 5);
            assert_eq!(srv_bytes.used_bytes(), MAX_TILE_CACHE_BYTES);
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_hybrid_tile_single_pass_bit_exact_parity() {
        // Exact oracle for RGB `is_text_safe` matching the reference two-pass
        // hybrid classifier (`<= 256` unique RGB colors OR `>= 35%` dominant RGB).
        fn ref_is_text_safe(frame: &[u8], stride_px: usize, rect: TileRect) -> bool {
            let mut counts = std::collections::HashMap::<u32, usize>::new();
            let n_px = rect.w as usize * rect.h as usize;
            for ry in 0..rect.h as usize {
                for rx in 0..rect.w as usize {
                    let off = ((rect.y as usize + ry) * stride_px + rect.x as usize + rx) * 4;
                    let rgb =
                        u32::from_le_bytes([frame[off], frame[off + 1], frame[off + 2], 0xFF]);
                    *counts.entry(rgb).or_insert(0) += 1;
                }
            }
            if counts.len() <= 256 {
                return true;
            }
            counts.values().copied().max().unwrap_or(0) * 100 >= n_px * 35
        }

        fn ref_filter_hybrid_tile<const HAS_ALPHA: bool>(
            tier: arch::Tier,
            frame: &[u8],
            aos: BufferPointer<Vec4u8>,
            stride_px: usize,
            rect: TileRect,
            quality: u8,
            subsamp: JpegSubsamp,
            meta: &TileEncodeMeta,
            state: &mut CompressorWorkerState,
        ) -> TilePayload {
            if let Some(px) = meta.tile.solid_px {
                return TilePayload::Leaf(LeafPayload::Solid(px));
            }
            let tw = rect.w as usize;
            let th = rect.h as usize;
            if tw < MIN_JPEG_DIM || th < MIN_JPEG_DIM {
                return TilePayload::Leaf(filter_lossless_leaf::<HAS_ALPHA>(
                    tier, aos, stride_px, rect, meta.tile, state,
                ));
            }
            if let Some(leaf) = try_encode_palette_leaf::<HAS_ALPHA>(
                tier,
                aos,
                stride_px,
                rect,
                meta.tile.all_opaque,
                state,
            ) {
                return TilePayload::Leaf(leaf);
            }
            if tw <= SUBTILE_W && th <= SUBTILE_H {
                return TilePayload::Leaf(if ref_is_text_safe(frame, stride_px, rect) {
                    if HAS_ALPHA && !meta.tile.all_opaque {
                        tier.filter_aos_zstd_leaf::<true>(aos, stride_px, rect, state)
                    } else {
                        tier.filter_aos_zstd_leaf::<false>(aos, stride_px, rect, state)
                    }
                } else if HAS_ALPHA && !meta.tile.all_opaque {
                    encode_jpeg_leaf::<true>(
                        tier, aos, false, stride_px, rect, quality, subsamp, state,
                    )
                } else {
                    encode_jpeg_leaf::<false>(
                        tier, aos, false, stride_px, rect, quality, subsamp, state,
                    )
                });
            }
            let (quads, n_quads) = tile_quadrants(rect);
            let mut safe = [false; 4];
            let mut text_safe_count = 0usize;
            for q in 0..n_quads {
                safe[q] = meta.quads[q].solid_px.is_some()
                    || ref_is_text_safe(frame, stride_px, quads[q]);
                text_safe_count += usize::from(safe[q]);
            }
            if text_safe_count == n_quads {
                TilePayload::Leaf(if HAS_ALPHA && !meta.tile.all_opaque {
                    tier.filter_aos_zstd_leaf::<true>(aos, stride_px, rect, state)
                } else {
                    tier.filter_aos_zstd_leaf::<false>(aos, stride_px, rect, state)
                })
            } else if text_safe_count == 0 {
                TilePayload::Leaf(if HAS_ALPHA && !meta.tile.all_opaque {
                    encode_jpeg_leaf::<true>(
                        tier, aos, false, stride_px, rect, quality, subsamp, state,
                    )
                } else {
                    encode_jpeg_leaf::<false>(
                        tier, aos, false, stride_px, rect, quality, subsamp, state,
                    )
                })
            } else {
                let mut sub_tiles = Vec::with_capacity(n_quads);
                for q in 0..n_quads {
                    let qrect = quads[q];
                    let qmeta = meta.quads[q];
                    let payload = if let Some(px) = qmeta.solid_px {
                        LeafPayload::Solid(px)
                    } else if safe[q] {
                        filter_lossless_leaf::<HAS_ALPHA>(tier, aos, stride_px, qrect, qmeta, state)
                    } else if (qrect.w as usize) < MIN_JPEG_DIM || (qrect.h as usize) < MIN_JPEG_DIM
                    {
                        if HAS_ALPHA && !qmeta.all_opaque {
                            tier.filter_aos_zstd_leaf::<true>(aos, stride_px, qrect, state)
                        } else {
                            tier.filter_aos_zstd_leaf::<false>(aos, stride_px, qrect, state)
                        }
                    } else if HAS_ALPHA && !qmeta.all_opaque {
                        encode_jpeg_leaf::<true>(
                            tier, aos, false, stride_px, qrect, quality, subsamp, state,
                        )
                    } else {
                        encode_jpeg_leaf::<false>(
                            tier, aos, false, stride_px, qrect, quality, subsamp, state,
                        )
                    };
                    sub_tiles.push(SubTile {
                        rect: qrect,
                        payload,
                    });
                }
                TilePayload::Split(sub_tiles)
            }
        }

        fn scan_meta<const HAS_ALPHA: bool>(
            frame: &[u8],
            stride_px: usize,
            rect: TileRect,
        ) -> RectScanMeta {
            let mut first = None;
            let mut solid = true;
            let mut all_opaque = true;
            for ry in 0..rect.h as usize {
                for rx in 0..rect.w as usize {
                    let off = ((rect.y as usize + ry) * stride_px + rect.x as usize + rx) * 4;
                    let mut px = [frame[off], frame[off + 1], frame[off + 2], frame[off + 3]];
                    if !HAS_ALPHA {
                        px[3] = 0xFF;
                    } else if px[3] != 0xFF {
                        all_opaque = false;
                    }
                    match first {
                        None => first = Some(px),
                        Some(f) if f != px => solid = false,
                        _ => {},
                    }
                }
            }
            RectScanMeta {
                solid_px: if solid { first } else { None },
                all_opaque,
            }
        }

        fn build_tile_meta<const HAS_ALPHA: bool>(
            frame: &[u8],
            stride_px: usize,
            rect: TileRect,
        ) -> TileEncodeMeta {
            let tile = scan_meta::<HAS_ALPHA>(frame, stride_px, rect);
            let mut quads = [RectScanMeta::UNKNOWN; 4];
            if rect.w as usize > SUBTILE_W || rect.h as usize > SUBTILE_H {
                let (qs, nq) = tile_quadrants(rect);
                for q in 0..nq {
                    quads[q] = scan_meta::<HAS_ALPHA>(frame, stride_px, qs[q]);
                }
            }
            TileEncodeMeta { tile, quads }
        }

        fn fill_solid(buf: &mut [u8], stride: usize, r: TileRect, px: [u8; 4]) {
            for ry in 0..r.h as usize {
                for rx in 0..r.w as usize {
                    let off = ((r.y as usize + ry) * stride + r.x as usize + rx) * 4;
                    buf[off..off + 4].copy_from_slice(&px);
                }
            }
        }

        fn fill_n_colors(
            buf: &mut [u8],
            stride: usize,
            r: TileRect,
            n_colors: usize,
            salt: usize,
            alpha_fn: impl Fn(usize) -> u8,
        ) {
            for ry in 0..r.h as usize {
                for rx in 0..r.w as usize {
                    let i = ry * r.w as usize + rx;
                    let c = (i % n_colors) + salt;
                    let off = ((r.y as usize + ry) * stride + r.x as usize + rx) * 4;
                    buf[off] = (c & 0xFF) as u8;
                    buf[off + 1] = ((c >> 8) & 0xFF) as u8;
                    buf[off + 2] = ((c * 37) & 0xFF) as u8;
                    buf[off + 3] = alpha_fn(i);
                }
            }
        }

        fn fill_dominant_bg(
            buf: &mut [u8],
            stride: usize,
            r: TileRect,
            dom_count: usize,
            late: bool,
            alpha_fn: impl Fn(usize) -> u8,
        ) {
            let n = r.w as usize * r.h as usize;
            for ry in 0..r.h as usize {
                for rx in 0..r.w as usize {
                    let i = ry * r.w as usize + rx;
                    let is_dom = if late {
                        i >= n - dom_count
                    } else {
                        i < dom_count
                    };
                    let off = ((r.y as usize + ry) * stride + r.x as usize + rx) * 4;
                    if is_dom {
                        buf[off] = 240;
                        buf[off + 1] = 242;
                        buf[off + 2] = 245;
                    } else {
                        let u = i + 500;
                        buf[off] = (u & 0xFF) as u8;
                        buf[off + 1] = ((u >> 8) & 0xFF) as u8;
                        buf[off + 2] = ((u.wrapping_mul(13)) & 0xFF) as u8;
                    }
                    buf[off + 3] = alpha_fn(i);
                }
            }
        }

        let tr = |x, y, w, h| TileRect { x, y, w, h };
        let tier = arch::Tier::detect();
        let stride = 300usize;
        let canvas_h = 280usize;
        let mut comp_got = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();
        let mut comp_ref = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();

        let mut check_case = |label: &str, buf: &[u8], rect: TileRect| {
            let ptr = buf.as_ptr().cast::<Vec4u8>();
            let aos = unsafe { BufferPointer::new(&ptr, buf.len() / 4) };
            let meta_xrgb = build_tile_meta::<false>(buf, stride, rect);
            let meta_argb = build_tile_meta::<true>(buf, stride, rect);

            for (m_tag, m_x, m_a) in [
                ("known", meta_xrgb, meta_argb),
                ("unknown", TileEncodeMeta::UNKNOWN, TileEncodeMeta::UNKNOWN),
            ] {
                let got_pair = Mutex::new(None);
                comp_got.pool.execute(1, |st_got| {
                    let gx = filter_hybrid_tile::<false>(
                        tier,
                        aos,
                        true,
                        stride,
                        rect,
                        85,
                        JpegSubsamp::Sub420,
                        &m_x,
                        st_got,
                    );
                    let ga = filter_hybrid_tile::<true>(
                        tier,
                        aos,
                        true,
                        stride,
                        rect,
                        85,
                        JpegSubsamp::Sub420,
                        &m_a,
                        st_got,
                    );
                    *got_pair.lock().unwrap() = Some((gx, ga));
                });
                let ref_pair = Mutex::new(None);
                comp_ref.pool.execute(1, |st_ref| {
                    let rx = ref_filter_hybrid_tile::<false>(
                        tier,
                        buf,
                        aos,
                        stride,
                        rect,
                        85,
                        JpegSubsamp::Sub420,
                        &m_x,
                        st_ref,
                    );
                    let ra = ref_filter_hybrid_tile::<true>(
                        tier,
                        buf,
                        aos,
                        stride,
                        rect,
                        85,
                        JpegSubsamp::Sub420,
                        &m_a,
                        st_ref,
                    );
                    *ref_pair.lock().unwrap() = Some((rx, ra));
                });
                let (got_x, got_a) = got_pair.into_inner().unwrap().unwrap();
                let (ref_x, ref_a) = ref_pair.into_inner().unwrap().unwrap();
                assert_eq!(got_x, ref_x, "Xrgb8888 mismatch on {label} ({m_tag})");
                assert_eq!(got_a, ref_a, "Argb8888 mismatch on {label} ({m_tag})");
            }
        };

        // 1. Non-splittable leaves (`<= 128x128`): Solid, 2/16/256 palette, 257 boundary,
        //    DominantBg (below & above 35%, early & late, translucent RGB dominant), Photo, thin strip.
        for rect in [
            tr(8, 6, 128, 128),
            tr(4, 4, 73, 51),
            tr(0, 0, 16, 16),
            tr(0, 0, 128, 8),
        ] {
            let n = rect.w as usize * rect.h as usize;
            let thresh = (n * 35).div_ceil(100);
            let mut buf = vec![0xAAu8; stride * canvas_h * 4];

            fill_solid(&mut buf, stride, rect, [10, 20, 30, 0xFF]);
            check_case("leaf_solid", &buf, rect);

            for n_col in [2usize, 16, 256] {
                fill_n_colors(&mut buf, stride, rect, n_col, 0, |i| {
                    0x80 + ((i * 7) & 0x7F) as u8
                });
                check_case("leaf_palette_var_alpha", &buf, rect);
                fill_n_colors(&mut buf, stride, rect, n_col, 0, |_| 0xFF);
                check_case("leaf_palette_opaque", &buf, rect);
            }

            if n >= 257 {
                fill_n_colors(&mut buf, stride, rect, 257, 0, |_| 0xFF);
                check_case("leaf_257_photo", &buf, rect);

                fill_dominant_bg(&mut buf, stride, rect, thresh - 1, false, |_| 0xFF);
                check_case("leaf_dom_below_early", &buf, rect);

                fill_dominant_bg(&mut buf, stride, rect, thresh, false, |_| 0xFF);
                check_case("leaf_dom_at_early", &buf, rect);

                fill_dominant_bg(&mut buf, stride, rect, thresh, true, |_| 0xFF);
                check_case("leaf_dom_at_late", &buf, rect);

                fill_dominant_bg(&mut buf, stride, rect, thresh, true, |i| {
                    0x80 + (i & 0x3F) as u8
                });
                check_case("leaf_dom_at_late_translucent", &buf, rect);
            }
        }

        // 2. Splittable `256x256` tiles:
        //    (a) Whole-tile palette, (b) All-Photo, (c) All-TextSafe (>256 global,
        //        both `ovf_ry < 128` and `ovf_ry >= 128`!), (d) Mixed Split with
        //        Solid + Palette + DominantBg + Photo in both `ovf_ry < 128` and
        //        `ovf_ry >= 128` layouts!
        let r256 = tr(12, 8, 256, 256);
        let q0 = tr(12, 8, 128, 128);
        let q1 = tr(140, 8, 128, 128);
        let q2 = tr(12, 136, 128, 128);
        let q3 = tr(140, 136, 128, 128);
        let mut buf = vec![0x55u8; stride * canvas_h * 4];

        // (a) Whole-tile <= 256 palette
        fill_n_colors(&mut buf, stride, r256, 200, 0, |_| 0xFF);
        check_case("split_whole_palette", &buf, r256);

        // (b) All-Photo
        fill_n_colors(&mut buf, stride, r256, 10_000, 0, |_| 0xFF);
        check_case("split_all_photo", &buf, r256);

        // (c1) All-TextSafe with `ovf_ry >= 128` (Q0+Q1 <= 200 colors total, Q2+Q3 add 200 more)
        fill_n_colors(&mut buf, stride, q0, 100, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q1, 100, 100, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q2, 100, 200, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q3, 100, 300, |_| 0xFF);
        check_case("split_all_textsafe_ovf_ge_128", &buf, r256);

        // (c2) All-TextSafe with `ovf_ry < 128` (Q0+Q1 > 256 colors, each quad DominantBg)
        let q_thresh = (128 * 128 * 35usize).div_ceil(100);
        fill_dominant_bg(&mut buf, stride, q0, q_thresh + 200, false, |_| 0xFF);
        fill_dominant_bg(&mut buf, stride, q1, q_thresh + 200, true, |_| 0xFF);
        fill_dominant_bg(&mut buf, stride, q2, q_thresh + 200, false, |i| {
            0x90 + (i & 0x1F) as u8
        });
        fill_n_colors(&mut buf, stride, q3, 12, 0, |_| 0xFF);
        check_case("split_all_textsafe_ovf_lt_128_mixed_alpha", &buf, r256);

        // (d1) Mixed Split with `ovf_ry < 128`: Q0 Solid, Q1 Photo, Q2 Palette, Q3 DominantBg
        fill_solid(&mut buf, stride, q0, [11, 22, 33, 0xFF]);
        fill_n_colors(&mut buf, stride, q1, 8000, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q2, 64, 0, |i| 0xC0 + (i & 3) as u8);
        fill_dominant_bg(&mut buf, stride, q3, q_thresh, true, |_| 0xFF);
        check_case("split_mixed_ovf_lt_128", &buf, r256);

        // (d2) Mixed Split with `ovf_ry >= 128`: Q0+Q1 <= 200 colors total (deferred!), Q2 Palette, Q3 Photo
        fill_n_colors(&mut buf, stride, q0, 80, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q1, 80, 80, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q2, 120, 400, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q3, 8000, 1000, |_| 0xFF);
        check_case("split_mixed_ovf_ge_128", &buf, r256);

        // (d3) Translucent mixed split where Q0 has <= 256 RGB colors but > 256 BGRA colors
        fill_n_colors(&mut buf, stride, q0, 32, 0, |i| 0x80 + (i & 0x3F) as u8);
        fill_n_colors(&mut buf, stride, q1, 16, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, q2, 8000, 0, |i| 0xA0 + (i & 0x1F) as u8);
        fill_solid(&mut buf, stride, q3, [44, 55, 66, 0x80]);
        check_case("split_mixed_translucent_rgb_vs_bgra", &buf, r256);

        // 3. Splittable 2-quadrant edge tiles:
        //    - Vertical `100x200` with `ovf_ry >= 128` (top palette, bottom photo)
        let rv = tr(10, 10, 100, 200);
        fill_n_colors(&mut buf, stride, tr(10, 10, 100, 128), 32, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, tr(10, 138, 100, 72), 4000, 0, |_| 0xFF);
        check_case("edge_vertical_2quad_ovf_ge_128", &buf, rv);

        //    - Horizontal `200x100` (left palette, right photo)
        let rh = tr(10, 10, 200, 100);
        fill_n_colors(&mut buf, stride, tr(10, 10, 128, 100), 16, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, tr(138, 10, 72, 100), 4000, 0, |_| 0xFF);
        check_case("edge_horizontal_2quad", &buf, rh);

        //    - Bottom-strip `< MIN_JPEG_DIM` (`256x136`: top palette, bottom `128x8` photo)
        let re = tr(0, 0, 256, 136);
        fill_n_colors(&mut buf, stride, tr(0, 0, 256, 128), 20, 0, |_| 0xFF);
        fill_n_colors(&mut buf, stride, tr(0, 128, 256, 8), 2000, 0, |_| 0xFF);
        check_case("edge_256x136_thin_bottom_photo", &buf, re);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_palette_streaming_roundtrip_and_malformed_unfilter_all_tiers() {
        use rkyv::util::AlignedVec;

        use crate::serialization::framing::EncodedBytes;

        let mut comp = ShardingCompressor::new(NonZeroUsize::new(2).unwrap(), 1).unwrap();
        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(2).unwrap()).unwrap();

        // Full encode + decode roundtrip for 1/2/4/8-bpp palette frames across
        // cached (`tile_cache = Some`) and uncached (`tile_cache = None`, direct
        // `allow_stream = true` NT stores) with both opaque Xrgb8888 and
        // translucent Argb8888 formats.
        for &(w, h) in &[(256usize, 256usize), (305, 193)] {
            let stride = w * 4;
            for n_colors in [2usize, 3, 4, 5, 15, 16, 17, 128, 255, 256] {
                for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
                    let mut src = vec![0u8; h * stride];
                    for y in 0..h {
                        for x in 0..w {
                            let i = y * w + x;
                            let c = if i < n_colors {
                                i
                            } else {
                                (i.wrapping_mul(31) ^ (y * 17)) % n_colors
                            };
                            let off = y * stride + x * 4;
                            src[off] = (c.wrapping_mul(17) & 0xFF) as u8;
                            src[off + 1] = (c.wrapping_mul(53).wrapping_add(9) & 0xFF) as u8;
                            src[off + 2] = (c.wrapping_mul(97).wrapping_add(23) & 0xFF) as u8;
                            src[off + 3] = if format == BufferFormat::Argb8888 {
                                0x50 + ((c * 13) & 0xAF) as u8
                            } else {
                                0xFF
                            };
                        }
                    }
                    let meta = BufferMetadata {
                        width: w as i32,
                        height: h as i32,
                        stride: stride as i32,
                        format,
                    };

                    for use_cache in [false, true] {
                        let mut srv = ServerTileCache::new();
                        let mut cli = ClientTileCache::new();
                        let ptr = src.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, src.len()) };
                        let mut stream_opt = None;
                        filter_and_compress_full_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(1),
                            meta,
                            CompressionMode::Lossless,
                            None,
                            use_cache.then_some(&mut srv),
                            &mut comp,
                            |s| stream_opt = Some(s),
                        );
                        let mut wire = Vec::new();
                        stream_opt.unwrap().framed_write(&mut wire).unwrap();

                        let mut dec_stream = None;
                        let mut rdr = wire.as_slice();
                        CompressedTileStream::read_from_stream(&mut rdr, |s| {
                            dec_stream = Some(s);
                            Ok(())
                        })
                        .unwrap();
                        let validated = ValidatedTileStream::new(dec_stream.unwrap()).unwrap();
                        let mut dst: AlignedVec<64> = AlignedVec::with_capacity(src.len());
                        dst.resize(src.len(), 0xEE);
                        decompress_and_unfilter_streaming(
                            validated,
                            &mut decomp,
                            &mut dst,
                            use_cache.then_some(&mut cli),
                        )
                        .unwrap();
                        assert_eq!(
                            dst.as_slice(),
                            src.as_slice(),
                            "roundtrip mismatch {w}x{h} n_colors={n_colors} fmt={format:?} cache={use_cache}"
                        );
                    }
                }
            }
        }

        // End-to-end malformed `LeafPayload::Palette` rejection through
        // `unfilter_tiles_parallel_with_tier` across every compiled tier,
        // verifying zero partial canvas corruption.
        let malformed_cases: Vec<(&str, LeafPayload)> = vec![
            (
                "bad_pal_len_not_mult_3",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1, 2, 3, 4],
                    data: EncodedBytes::Uncompressed(vec![0; 32]),
                },
            ),
            (
                "n_colors_0",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![],
                    data: EncodedBytes::Uncompressed(vec![]),
                },
            ),
            (
                "n_colors_1",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1, 2, 3],
                    data: EncodedBytes::Uncompressed(vec![]),
                },
            ),
            (
                "n_colors_257",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![0; 257 * 3],
                    data: EncodedBytes::Uncompressed(vec![0; 64 * 4]),
                },
            ),
            (
                "wrong_packed_len",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![0; 2 * 3],
                    data: EncodedBytes::Uncompressed(vec![0; 31]),
                },
            ),
            (
                "oob_2bpp_tail",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1; 3 * 3],
                    data: EncodedBytes::Uncompressed({
                        let mut v = vec![0b10_01_00_10u8; 64];
                        v[63] = 0b1100_0000;
                        v
                    }),
                },
            ),
            (
                "oob_4bpp_mid",
                LeafPayload::Palette {
                    has_alpha: true,
                    palette: vec![1; 15 * 4],
                    data: EncodedBytes::Uncompressed({
                        let mut v = vec![0xE3u8; 128];
                        v[63] = 0xF0;
                        v
                    }),
                },
            ),
            (
                "oob_8bpp_mid",
                LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1; 255 * 3],
                    data: EncodedBytes::Uncompressed({
                        let mut v = vec![254u8; 256];
                        v[129] = 255;
                        v
                    }),
                },
            ),
        ];

        let meta = BufferMetadata {
            width: 64,
            height: 4,
            stride: 64 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let tile_rects = [TileRect {
            x: 0,
            y: 0,
            w: 64,
            h: 4,
        }];
        for tier in arch::Tier::supported_tiers() {
            for (name, payload) in &malformed_cases {
                let mut canvas: AlignedVec<64> = AlignedVec::with_capacity(64 * 4 * 4);
                canvas.resize(64 * 4 * 4, 0xA5);
                let tile = CompressedTile {
                    tile_idx: 0,
                    cache_put: None,
                    payload: TilePayload::Leaf(payload.clone()),
                };
                let emitted = AtomicBool::new(false);
                let res = unsafe {
                    unfilter_tiles_parallel_with_tier(
                        tier,
                        &mut decomp,
                        meta,
                        canvas.len(),
                        false,
                        &[],
                        &tile_rects,
                        Some(&mut canvas),
                        None,
                        || {
                            (!emitted.swap(true, Ordering::Relaxed))
                                .then(|| Ok(Cow::Borrowed(&tile)))
                        },
                    )
                };
                assert!(
                    res.is_err(),
                    "tier={tier:?} accepted malformed palette case {name}"
                );
                assert!(
                    canvas.iter().all(|&b| b == 0xA5),
                    "tier={tier:?} partially corrupted canvas on malformed case {name}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_aos_mixed_predictor_transitions_across_tiers() {
        // Explicit sequence exercising every `(prev_mode -> cur_mode)` transition
        // among `{0=None, 1=Sub, 2=Up, 3=SubUp}`, including `Up -> None/Sub`
        // (where fused `Up` skips writing `cur_dec`), `Up -> Up/SubUp` (where
        // fused `Up` must populate `cur_dec` for the next row), and ending on
        // `Up` as the final row (`next_needs_prev == false`).
        let pred_seq: [u8; 20] = [0, 2, 0, 1, 2, 1, 2, 2, 3, 2, 3, 3, 0, 3, 1, 3, 2, 0, 2, 2];
        let th = pred_seq.len();
        let mut predictors = vec![0u8; th.div_ceil(4)];
        for (ry, &p) in pred_seq.iter().enumerate() {
            predictors[ry >> 2] |= p << ((ry & 3) * 2);
        }

        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let widths = [
            1usize, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15, 16, 17, 23, 24, 25, 31, 32, 40, 48, 63,
            64, 72, 120, 127, 128, 255, 256,
        ];

        for &tw in &widths {
            let row_b = tw * 3;
            for (format, alpha_kind) in [
                (BufferFormat::Xrgb8888, 0u8),
                (BufferFormat::Argb8888, 1),
                (BufferFormat::Argb8888, 2),
            ] {
                let mut expected_px = Vec::with_capacity(tw * th);
                let mut raw_bgr_rows = Vec::with_capacity(th);
                let mut raw_alpha = Vec::with_capacity(tw * th);
                for ry in 0..th {
                    let mut row_px = Vec::with_capacity(tw);
                    for x in 0..tw {
                        let k = (ry * 131 + x * 37 + tw * 17 + usize::from(alpha_kind) * 53) as u32;
                        let a = match alpha_kind {
                            0 => 0xFF,
                            1 => 0xA7,
                            _ => ((k * 19 + 7) & 0xFF) as u8,
                        };
                        row_px.push(Vec4u8(
                            ((k * 13 + 5) & 0xFF) as u8,
                            ((k * 29 + 11) & 0xFF) as u8,
                            ((k * 43 + 17) & 0xFF) as u8,
                            a,
                        ));
                    }
                    let mut bgr = vec![0u8; row_b];
                    let mut alp = vec![0u8; tw];
                    let p = row_px.as_ptr();
                    let bp = unsafe { BufferPointer::new(&p, tw) };
                    ScalarKernel.bgra_to_subgreen_bgr3::<true>(bp, &mut bgr, &mut alp);
                    expected_px.extend_from_slice(&row_px);
                    raw_bgr_rows.push(bgr);
                    raw_alpha.extend_from_slice(&alp);
                }

                let mut filt_bgr = vec![0u8; row_b * th];
                for ry in 0..th {
                    let dst = &mut filt_bgr[ry * row_b..(ry + 1) * row_b];
                    match pred_seq[ry] {
                        0 => dst.copy_from_slice(&raw_bgr_rows[ry]),
                        1 => ScalarKernel.row_sub3_forward(&raw_bgr_rows[ry], dst),
                        2 => ScalarKernel.row_up_forward(
                            &raw_bgr_rows[ry],
                            &raw_bgr_rows[ry - 1],
                            dst,
                        ),
                        _ => {
                            ScalarKernel.row_up_forward(
                                &raw_bgr_rows[ry],
                                &raw_bgr_rows[ry - 1],
                                dst,
                            );
                            ScalarKernel.row_sub3_forward_in_place(dst);
                        },
                    }
                }

                let alpha = match alpha_kind {
                    0 => AlphaChannel::Omitted,
                    1 => AlphaChannel::Constant(0xA7),
                    _ => {
                        let mut a_filt = vec![0u8; tw * th];
                        ScalarKernel.row_sub1_forward(&raw_alpha[..tw], &mut a_filt[..tw]);
                        for ry in 1..th {
                            ScalarKernel.row_up_forward(
                                &raw_alpha[ry * tw..(ry + 1) * tw],
                                &raw_alpha[(ry - 1) * tw..ry * tw],
                                &mut a_filt[ry * tw..(ry + 1) * tw],
                            );
                        }
                        AlphaChannel::Uncompressed(a_filt)
                    },
                };

                let payload = LeafPayload::AosZstd {
                    predictors: predictors.clone(),
                    color: EncodedBytes::Uncompressed(filt_bgr),
                    alpha,
                };
                let meta = BufferMetadata {
                    width: tw as i32,
                    height: th as i32,
                    stride: (tw * 4) as i32,
                    format,
                };
                let tile_rects = [TileRect {
                    x: 0,
                    y: 0,
                    w: tw as u32,
                    h: th as u32,
                }];
                let expected_bytes: &[u8] = bytemuck::cast_slice(&expected_px);

                for tier in arch::Tier::supported_tiers() {
                    for use_cache in [false, true] {
                        let mut cache = ClientTileCache::new();
                        let mut canvas: AlignedVec<64> = AlignedVec::with_capacity(tw * th * 4);
                        canvas.resize(tw * th * 4, 0x5A);
                        let tile = CompressedTile {
                            tile_idx: 0,
                            cache_put: use_cache.then_some(0),
                            payload: TilePayload::Leaf(payload.clone()),
                        };
                        let emitted = AtomicBool::new(false);
                        unsafe {
                            unfilter_tiles_parallel_with_tier(
                                tier,
                                &mut decomp,
                                meta,
                                canvas.len(),
                                false,
                                &[],
                                &tile_rects,
                                Some(&mut canvas),
                                use_cache.then_some(&mut cache),
                                || {
                                    (!emitted.swap(true, Ordering::Relaxed))
                                        .then(|| Ok(Cow::Borrowed(&tile)))
                                },
                            )
                            .unwrap();
                        }
                        assert_eq!(
                            canvas.as_slice(),
                            expected_bytes,
                            "tier={tier:?} tw={tw} alpha_kind={alpha_kind} use_cache={use_cache}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_scroll_probes_and_row_blit_cross_tier_parity() {
        let tiers = arch::Tier::supported_tiers();
        assert!(!tiers.is_empty());

        // 1. `detect_scroll_rect_with_tier` parity across all supported tiers
        //    for scrolling (`Xrgb8888`/`Argb8888`, up/down, split-pane) and
        //    non-scrolling (unchanged, alpha-only XRGB, typing, solid fill, tab
        //    switch) frames.
        let width = 512usize;
        let height = 400usize;
        let stride = width * 4 + 16;
        let doc = make_textured_document(width, height + 160, stride, BufferFormat::Argb8888);
        let prev = doc[..height * stride].to_vec();

        let make_split_scroll = |base: &[u8], dy: i32| -> Vec<u8> {
            let mut curr = base.to_vec();
            let sx = 69usize;
            let sw = 365usize;
            let sy0 = 24usize;
            let sh = 352usize;
            let shift = dy.unsigned_abs() as usize;
            let copy_h = sh - shift;
            if dy < 0 {
                for ry in 0..copy_h {
                    let s = (sy0 + shift + ry) * stride + sx * 4;
                    let d = (sy0 + ry) * stride + sx * 4;
                    curr[d..d + sw * 4].copy_from_slice(&base[s..s + sw * 4]);
                }
                for ry in copy_h..sh {
                    let s = (height + ry - copy_h) * stride + sx * 4;
                    let d = (sy0 + ry) * stride + sx * 4;
                    curr[d..d + sw * 4].copy_from_slice(&doc[s..s + sw * 4]);
                }
            } else {
                for ry in 0..copy_h {
                    let s = (sy0 + ry) * stride + sx * 4;
                    let d = (sy0 + shift + ry) * stride + sx * 4;
                    curr[d..d + sw * 4].copy_from_slice(&base[s..s + sw * 4]);
                }
                for ry in 0..shift {
                    let s = (height + ry) * stride + sx * 4;
                    let d = (sy0 + ry) * stride + sx * 4;
                    curr[d..d + sw * 4].copy_from_slice(&doc[s..s + sw * 4]);
                }
            }
            curr
        };

        let mut curr_up = vec![0u8; height * stride];
        curr_up.copy_from_slice(&doc[48 * stride..(height + 48) * stride]);
        let mut curr_down = prev.clone();
        curr_down[52 * stride..height * stride].copy_from_slice(&prev[..(height - 52) * stride]);
        curr_down[..52 * stride].copy_from_slice(&doc[height * stride..(height + 52) * stride]);
        let curr_split_up = make_split_scroll(&prev, -37);
        let curr_split_down = make_split_scroll(&prev, 43);

        let mut curr_alpha_only = prev.clone();
        for y in 0..height {
            for x in 0..width {
                curr_alpha_only[y * stride + x * 4 + 3] ^= 0x77;
            }
        }
        let mut curr_typing = prev.clone();
        for y in 100..118 {
            for x in 80..220 {
                curr_typing[y * stride + x * 4] ^= 0x5A;
            }
        }
        let curr_solid = vec![0x42u8; height * stride];
        let curr_tab_switch: Vec<u8> = prev.iter().map(|&b| b.wrapping_add(0x6D) ^ 0x3C).collect();

        for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
            let meta = BufferMetadata {
                width: width as i32,
                height: height as i32,
                stride: stride as i32,
                format,
            };
            let cases: [(&str, &[u8], Option<i32>); 9] = [
                ("full_up", &curr_up, Some(-48)),
                ("full_down", &curr_down, Some(52)),
                ("split_up", &curr_split_up, Some(-37)),
                ("split_down", &curr_split_down, Some(43)),
                ("unchanged", &prev, None),
                ("alpha_only", &curr_alpha_only, None),
                ("typing", &curr_typing, None),
                ("solid", &curr_solid, None),
                ("tab_switch", &curr_tab_switch, None),
            ];
            for (name, curr_bytes, expected_dy) in cases {
                let p = curr_bytes.as_ptr();
                let bp = unsafe { BufferPointer::new(&p, curr_bytes.len()) };
                let baseline =
                    scroll::detect_scroll_rect_with_tier(bp, &prev, meta, None, tiers[0]);
                match expected_dy {
                    Some(dy) => {
                        let sr = baseline.unwrap_or_else(|| {
                            panic!("expected scroll dy={dy} on {name} ({format:?})")
                        });
                        assert_eq!(sr.dy, dy, "unexpected dy on {name} ({format:?})");
                    },
                    None => assert_eq!(
                        baseline, None,
                        "false positive scroll on {name} ({format:?})"
                    ),
                }
                for &tier in &tiers[1..] {
                    let got = scroll::detect_scroll_rect_with_tier(bp, &prev, meta, None, tier);
                    assert_eq!(
                        got, baseline,
                        "tier={tier:?} vs {:?} mismatch on {name} ({format:?})",
                        tiers[0]
                    );
                }
            }
        }

        // 2. Direct `tier.apply_scroll_rows_stream` / `apply_scroll_rows_temporal`
        //    and `tier.copy_packed_rows_stream` parity across all supported tiers,
        //    odd widths, aligned/unaligned `rect.x`, padded strides, and every
        //    `dy` overlap/disjoint boundary.
        let rect_xs: &[u32] = if cfg!(miri) {
            &[0, 1, 7, 16, 17]
        } else {
            &[0, 1, 3, 7, 8, 15, 16, 17]
        };
        let widths: Vec<u32> = if cfg!(miri) {
            vec![1, 2, 7, 15, 16, 17, 33, 64, 65]
        } else {
            (1u32..=65).chain([127, 128, 129, 255, 256]).collect()
        };
        let scroll_h = 36u32;
        let dys: &[i32] = &[
            -1, 1, -7, 7, -12, // `|dy| == copy_h / 2` (`12 == 24 / 2`)
            12, -15, // `|dy| > copy_h / 2` yet still overlapping (`15 < 21`)
            15, -18, // `|dy| == copy_h` (exact disjoint boundary)
            18, -25, // `|dy| > copy_h` (strictly disjoint)
            25, -35, // `|dy| == height - 1` (`copy_h == 1`)
            35,
        ];
        for &stride_pad in &[0usize, 4, 20, 64] {
            for &rx in rect_xs {
                for &rw in &widths {
                    let frame_w = rx as usize + rw as usize + 3;
                    let frame_h = scroll_h as usize + 4;
                    let st = frame_w * 4 + stride_pad;
                    let base: Vec<u8> = (0..frame_h * st)
                        .map(|i| ((i * 29 + rw as usize * 7 + rx as usize) & 0xFF) as u8)
                        .collect();

                    // Test `copy_packed_rows_stream` per tier.
                    let packed: Vec<u8> = (0..7 * rw as usize * 4)
                        .map(|i| ((i * 41 + 9) & 0xFF) as u8)
                        .collect();
                    let mut ref_dst = base.clone();
                    for r in 0..7usize {
                        let d = (2 + r) * st + rx as usize * 4;
                        ref_dst[d..d + rw as usize * 4].copy_from_slice(
                            &packed[r * rw as usize * 4..(r + 1) * rw as usize * 4],
                        );
                    }
                    for &tier in &tiers {
                        let mut dst = base.clone();
                        tier.copy_packed_rows_stream(
                            &packed,
                            &mut dst,
                            st,
                            rx as usize * 4,
                            2,
                            rw as usize * 4,
                            7,
                        );
                        assert_eq!(
                            dst, ref_dst,
                            "copy_packed_rows_stream mismatch tier={tier:?} rx={rx} rw={rw} pad={stride_pad}"
                        );
                    }

                    // Test streaming and temporal scroll-row blits.
                    // Sweep all `dy` cases on representative widths and a fast
                    // subset on the dense `1..=65` width sweep.
                    let active_dys: &[i32] = if matches!(
                        rw,
                        1 | 7
                            | 15
                            | 16
                            | 17
                            | 31
                            | 32
                            | 33
                            | 63
                            | 64
                            | 65
                            | 127
                            | 128
                            | 129
                            | 255
                            | 256
                    ) {
                        dys
                    } else {
                        &[-1, 1, -12, 18, -25, 35]
                    };
                    for &dy in active_dys {
                        let sr = ScrollRect {
                            x: rx,
                            y: 2,
                            width: rw,
                            height: scroll_h,
                            dy,
                        };
                        let mut expected = base.clone();
                        apply_scroll_rect_in_place(&mut expected, st, sr);
                        let shift = dy.unsigned_abs() as usize;
                        let copy_h = scroll_h as usize - shift;
                        let move_up = dy < 0;
                        let src_y0 = if move_up { 2 + shift } else { 2 };
                        let dst_y0 = if move_up { 2 } else { 2 + shift };
                        let disjoint = shift >= copy_h;
                        for &tier in &tiers {
                            for stream in [false, true] {
                                let mut buf = base.clone();
                                unsafe {
                                    if stream {
                                        tier.apply_scroll_rows_stream(
                                            buf.as_mut_ptr(),
                                            st,
                                            rx as usize * 4,
                                            rw as usize * 4,
                                            src_y0,
                                            dst_y0,
                                            copy_h,
                                            !move_up,
                                            disjoint,
                                        );
                                    } else {
                                        scroll::apply_scroll_rows_temporal(
                                            buf.as_mut_ptr(),
                                            st,
                                            rx as usize * 4,
                                            rw as usize * 4,
                                            src_y0,
                                            dst_y0,
                                            copy_h,
                                            !move_up,
                                            disjoint,
                                        );
                                    }
                                }
                                assert_eq!(
                                    buf, expected,
                                    "apply_scroll_rows mismatch tier={tier:?} stream={stream} sr={sr:?} pad={stride_pad}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scroll_parallel_blit_cross_tier_parity() {
        let tiers = arch::Tier::supported_tiers();
        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(4).unwrap()).unwrap();

        // Full-width and partial-pane parallel blits (`>= 512 KiB`) across
        // 64B-aligned and unaligned `x`, padded strides (`stride % 64 == 0` and
        // `stride % 64 != 0`), `3840`-wide 4K width, and every overlap/disjoint
        // `dy` regime (`|dy| == 1`, `< copy_h/2`, `== copy_h/2`, `> copy_h/2`
        // overlapping, `== copy_h` disjoint, `> copy_h` disjoint, `== height-1`).
        let geometries: &[(usize, usize, &[(u32, u32, u32, u32)])] = &[
            (
                320,
                1024 * 4,
                &[
                    (0, 0, 1024, 320),
                    (1, 8, 1007, 300),
                    (3, 8, 993, 300),
                    (7, 8, 977, 300),
                    (8, 8, 960, 300),
                    (15, 8, 945, 300),
                    (16, 8, 944, 300),
                    (17, 8, 943, 300),
                ],
            ),
            (
                320,
                1024 * 4 + 20, // `stride % 64 != 0`
                &[(0, 0, 1024, 320), (17, 12, 929, 296)],
            ),
            (
                160,
                3840 * 4, // 4K width, `2.46 MiB` (`>= 512 KiB`)
                &[(0, 0, 3840, 160), (773, 12, 3051, 140)],
            ),
        ];

        for &(fh, stride, rects) in geometries {
            let base: Vec<u8> = (0..fh * stride)
                .map(|i| ((i * 31 + (i >> 8) * 17) & 0xFF) as u8)
                .collect();
            for &(rx, ry, rw, rh) in rects {
                let h = rh as i32;
                let dys = [
                    -1,
                    1,
                    -(h / 4),
                    h / 4,
                    -(h / 3), // `|dy| == copy_h / 2`
                    h / 3,
                    -(h * 2 / 5), // `|dy| > copy_h / 2`, still overlapping (`shift < copy_h`)
                    h * 2 / 5,
                    -(h / 2), // `|dy| == copy_h` (disjoint boundary)
                    h / 2,
                    -(h * 3 / 5), // `|dy| > copy_h` (disjoint horizontal bands)
                    h * 3 / 5,
                    -(h - 1), // `copy_h == 1`
                    h - 1,
                ];
                for dy in dys {
                    let sr = ScrollRect {
                        x: rx,
                        y: ry,
                        width: rw,
                        height: rh,
                        dy,
                    };
                    let mut expected = base.clone();
                    apply_scroll_rect_in_place(&mut expected, stride, sr);
                    for &tier in &tiers {
                        for stream in [false, true] {
                            let mut got = base.clone();
                            scroll::apply_scroll_rect_in_place_parallel_with(
                                &mut got,
                                stride,
                                sr,
                                &mut decomp,
                                tier,
                                stream,
                            );
                            assert_eq!(
                                got, expected,
                                "parallel scroll mismatch tier={tier:?} stream={stream} sr={sr:?} stride={stride}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Generates a deterministic high-entropy photographic frame guaranteed to
    /// exceed `TextSafe` thresholds on every tile (forcing `LeafPayload::Jpeg`).
    fn make_photo_bgra(
        w: usize,
        h: usize,
        stride: usize,
        alpha_fn: impl Fn(usize, usize) -> u8,
    ) -> Vec<u8> {
        let mut buf = vec![0xA5u8; h * stride];
        for y in 0..h {
            for x in 0..w {
                let mut z = ((x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let off = y * stride + x * 4;
                buf[off] = ((x * 5 + y * 3) as u8).wrapping_add((z & 0x3F) as u8);
                buf[off + 1] = ((x * 2 + y * 7) as u8).wrapping_add(((z >> 8) & 0x3F) as u8);
                buf[off + 2] = ((x * 9 + y * 4) as u8).wrapping_add(((z >> 16) & 0x3F) as u8);
                buf[off + 3] = alpha_fn(x, y);
            }
        }
        buf
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_turbojpeg_bgra_sets_alpha_0xff_strided_and_edges() {
        let mut comp = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();
        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(1).unwrap()).unwrap();
        for &(tw, th) in &[
            (16, 16),
            (17, 19),
            (31, 33),
            (64, 48),
            (128, 128),
            (255, 19),
            (256, 256),
        ] {
            let stride_px = tw + 13;
            let src = make_photo_bgra(stride_px, th + 5, stride_px * 4, |x, y| {
                ((x * 11 + y * 7) & 0xFE) as u8
            });
            let jpeg_slot = Mutex::new(Vec::new());
            comp.pool.execute(1, |state| {
                let tile_ptr = unsafe { src.as_ptr().add((3 * stride_px + 7) * 4) };
                let out = unsafe {
                    state
                        .compress_jpeg_strided_raw(
                            tile_ptr,
                            tw,
                            stride_px * 4,
                            th,
                            85,
                            JpegSubsamp::Sub420,
                        )
                        .unwrap()
                };
                *jpeg_slot.lock().unwrap() = out;
            });
            let jpeg = jpeg_slot.into_inner().unwrap();
            // Compare strided encode bit-for-bit against packed encode.
            let mut packed = vec![0u8; tw * th * 4];
            for ry in 0..th {
                let s = ((3 + ry) * stride_px + 7) * 4;
                packed[ry * tw * 4..(ry + 1) * tw * 4].copy_from_slice(&src[s..s + tw * 4]);
            }
            let jpeg_packed_slot = Mutex::new(Vec::new());
            comp.pool.execute(1, |state| {
                state.jpeg_pixels.resize(packed.len(), 0);
                state.jpeg_pixels[..packed.len()].copy_from_slice(&packed);
                let out = state
                    .compress_jpeg_from_scratch(tw, th, 85, JpegSubsamp::Sub420)
                    .unwrap();
                *jpeg_packed_slot.lock().unwrap() = out;
            });
            let jpeg_packed = jpeg_packed_slot.into_inner().unwrap();
            assert_eq!(
                jpeg, jpeg_packed,
                "strided vs packed JPEG mismatch {tw}x{th}"
            );

            // Decode strided into a poisoned canvas (both aligned and +4 unaligned)
            // and verify every decoded pixel's byte 3 is 0xFF and surrounding
            // sentinel bytes are completely untouched.
            for &x_off in &[8usize, 9usize] {
                let dst_stride_px = tw + x_off + 11;
                let dst_h = th + 4;
                let canvas_cell = Mutex::new(vec![0x5Au8; dst_h * dst_stride_px * 4]);
                decomp.pool.execute(1, |state| {
                    let mut canvas = canvas_cell.lock().unwrap();
                    let dst_ptr =
                        unsafe { canvas.as_mut_ptr().add((2 * dst_stride_px + x_off) * 4) };
                    unsafe {
                        state
                            .decompress_jpeg_into_raw(&jpeg, tw, th, dst_ptr, dst_stride_px * 4)
                            .unwrap();
                    }
                });
                let canvas = canvas_cell.into_inner().unwrap();
                for y in 0..dst_h {
                    for x in 0..dst_stride_px {
                        let off = (y * dst_stride_px + x) * 4;
                        if (2..2 + th).contains(&y) && (x_off..x_off + tw).contains(&x) {
                            assert_eq!(
                                canvas[off + 3],
                                0xFF,
                                "BGRA byte 3 != 0xFF at ({x},{y}) for {tw}x{th} x_off={x_off}"
                            );
                        } else {
                            assert_eq!(
                                &canvas[off..off + 4],
                                &[0x5A; 4],
                                "strided decode overwrote out-of-rect pixel ({x},{y}) for {tw}x{th}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hybrid_strided_jpeg_corners_edges_and_alpha() {
        let mut comp = ShardingCompressor::new(NonZeroUsize::new(4).unwrap(), 1).unwrap();
        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(4).unwrap()).unwrap();
        let mode = CompressionMode::Hybrid {
            quality: 85,
            subsamp: JpegSubsamp::Sub420,
        };

        // Cover:
        // - 533x301 with padded stride (`+12` bytes): exercises all 4 corners,
        //   non-right-edge bottom tile `(0, 256, 256, 45)`, middle bottom tile
        //   `(256, 256, 256, 45)`, and bottom-right corner `(512, 256, 21, 45)`.
        // - 512x512 mixed Split tiles (top-left solid, top-right photo,
        //   bottom-left palette, bottom-right photo) so cached & uncached
        //   Split JPEG quadrant decoding runs at non-zero quadrant offsets.
        let cases: &[(usize, usize, usize, BufferFormat, u8)] = &[
            (533, 301, 533 * 4, BufferFormat::Xrgb8888, 0),
            (533, 301, 533 * 4 + 12, BufferFormat::Xrgb8888, 0),
            (533, 301, 533 * 4, BufferFormat::Argb8888, 1), // opaque 0xFF
            (533, 301, 533 * 4 + 12, BufferFormat::Argb8888, 2), // constant 0x7E
            (533, 301, 533 * 4 + 12, BufferFormat::Argb8888, 3), // varying alpha
            (512, 512, 512 * 4, BufferFormat::Xrgb8888, 4), // mixed Split Xrgb
            (512, 512, 512 * 4, BufferFormat::Argb8888, 5), // mixed Split translucent
        ];

        for &(w, h, stride, format, kind) in cases {
            let mut src = match kind {
                0 => make_photo_bgra(w, h, stride, |x, y| ((x + y) & 0xFF) as u8),
                1 => make_photo_bgra(w, h, stride, |_, _| 0xFF),
                2 => make_photo_bgra(w, h, stride, |_, _| 0x7E),
                3 => make_photo_bgra(w, h, stride, |x, y| {
                    0x20u8.wrapping_add(((x * 3 + y * 5) & 0xDF) as u8)
                }),
                4 | 5 => {
                    let mut buf = make_photo_bgra(w, h, stride, |x, y| {
                        if kind == 4 {
                            0xFF
                        } else {
                            0x40u8.wrapping_add(((x + y * 3) & 0xBF) as u8)
                        }
                    });
                    // Make top-left 128x128 quadrant of each 256x256 tile solid
                    // and bottom-left 128x128 a 4-color palette to force Split!
                    for ty in (0..h).step_by(256) {
                        for tx in (0..w).step_by(256) {
                            for ry in 0..128 {
                                for rx in 0..128 {
                                    let off = (ty + ry) * stride + (tx + rx) * 4;
                                    let a = if kind == 4 { 0xFF } else { 0xE0 };
                                    buf[off..off + 4].copy_from_slice(&[0xF0, 0xF0, 0xF0, a]);
                                    let off2 = (ty + 128 + ry) * stride + (tx + rx) * 4;
                                    let c = ((rx ^ ry) & 3) as u8 * 60;
                                    buf[off2..off2 + 4].copy_from_slice(&[c, c, 0x20, a]);
                                }
                            }
                        }
                    }
                    buf
                },
                _ => unreachable!(),
            };
            // Zero stride padding bytes so comparison matches zeroed shadow padding.
            if stride > w * 4 {
                for y in 0..h {
                    src[y * stride + w * 4..(y + 1) * stride].fill(0);
                }
            }
            let meta = BufferMetadata {
                width: w as i32,
                height: h as i32,
                stride: stride as i32,
                format,
            };
            let p = src.as_ptr();
            let bp = unsafe { BufferPointer::new(&p, src.len()) };

            // 1. Stateless bounce path (`shadow_out = None`) produces baseline wire & pixels.
            let stateless_tiles = filter_and_compress(bp, meta, mode, &mut comp);
            let mut stateless_out = vec![0xAAu8; src.len()];
            decompress_and_unfilter(&stateless_tiles, &mut decomp, &mut stateless_out).unwrap();

            // 2. Owned-shadow cold-fill + cached decode must produce bit-identical
            //    wire tiles AND bit-identical decoded pixels (and exact lossless alpha!).
            for use_cache in [false, true] {
                let mut shadow = ShadowBuffer::new();
                let mut srv_cache = ServerTileCache::new();
                let mut cli_cache = ClientTileCache::new();
                let mut stream_opt = None;
                filter_and_compress_full_streaming(
                    bp,
                    ClientId(1),
                    WlSurfaceId(1),
                    meta,
                    mode,
                    Some(&mut shadow),
                    use_cache.then_some(&mut srv_cache),
                    &mut comp,
                    |s| stream_opt = Some(s),
                );
                let mut wire = Vec::new();
                stream_opt.unwrap().framed_write(&mut wire).unwrap();

                let read_stream = |wire_bytes: &[u8]| -> ValidatedTileStream {
                    let mut stream_opt = None;
                    let mut rdr = wire_bytes;
                    CompressedTileStream::read_from_stream(&mut rdr, |s| {
                        stream_opt = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    ValidatedTileStream::new(stream_opt.unwrap()).unwrap()
                };
                let validated = read_stream(&wire);

                // Test at both 64B-aligned and +4 unaligned destination offsets.
                let mut out_storage = vec![0xCCu8; src.len() + 68];
                let aligned_off = (64 - (out_storage.as_ptr() as usize & 63)) & 63;
                for &byte_off in &[aligned_off, aligned_off + 4] {
                    let mut cc = ClientTileCache::new();
                    let validated2 = read_stream(&wire);
                    let out = &mut out_storage[byte_off..byte_off + src.len()];
                    out.fill(0x55);
                    decompress_and_unfilter_streaming(
                        validated2,
                        &mut decomp,
                        out,
                        use_cache.then_some(&mut cc),
                    )
                    .unwrap();
                    for y in 0..h {
                        for x in 0..w {
                            let off = y * stride + x * 4;
                            assert_eq!(
                                &out[off..off + 4],
                                &stateless_out[off..off + 4],
                                "strided vs stateless pixel mismatch ({x},{y}) kind={kind} cache={use_cache} off={byte_off}"
                            );
                            let expected_a = if format == BufferFormat::Xrgb8888 {
                                0xFF
                            } else {
                                src[off + 3]
                            };
                            assert_eq!(
                                out[off + 3],
                                expected_a,
                                "alpha mismatch ({x},{y}) kind={kind} cache={use_cache}"
                            );
                        }
                    }
                }

                // Also test incremental Phase-B strided encode/decode after mutating bottom-right.
                let mut out_main = vec![0u8; src.len()];
                decompress_and_unfilter_streaming(
                    validated,
                    &mut decomp,
                    &mut out_main,
                    use_cache.then_some(&mut cli_cache),
                )
                .unwrap();
                let mut src2 = src.clone();
                for y in (h - 40)..h {
                    for x in (w - 40)..w {
                        let off = y * stride + x * 4;
                        src2[off] ^= 0x33;
                        src2[off + 1] ^= 0x55;
                        src2[off + 2] ^= 0x11;
                        if format == BufferFormat::Argb8888 && matches!(kind, 3 | 5) {
                            src2[off + 3] ^= 0x22;
                        }
                    }
                }
                let p2 = src2.as_ptr();
                let bp2 = unsafe { BufferPointer::new(&p2, src2.len()) };
                let mut inc_stream = None;
                let res = filter_and_compress_incremental_streaming(
                    bp2,
                    ClientId(1),
                    WlSurfaceId(1),
                    meta,
                    mode,
                    None,
                    &mut shadow,
                    use_cache.then_some(&mut srv_cache),
                    &mut comp,
                    |s| inc_stream = Some(s),
                );
                assert!(!res.is_unchanged());
                let mut wire2 = Vec::new();
                inc_stream.unwrap().framed_write(&mut wire2).unwrap();
                let val2 = read_stream(&wire2);
                decompress_and_unfilter_streaming(
                    val2,
                    &mut decomp,
                    &mut out_main,
                    use_cache.then_some(&mut cli_cache),
                )
                .unwrap();
                for y in 0..h {
                    for x in 0..w {
                        let off = y * stride + x * 4;
                        let expected_a = if format == BufferFormat::Xrgb8888 {
                            0xFF
                        } else {
                            src2[off + 3]
                        };
                        assert_eq!(
                            out_main[off + 3],
                            expected_a,
                            "incremental alpha mismatch ({x},{y}) kind={kind} cache={use_cache}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_phase_ab_work_steal_partition_and_partial_wake() {
        for &p in &[1usize, 2, 3, 4, 8, 16, 64] {
            for &total in &[0usize, 1, 2, 7, 15, 16, 17, 30, 64, 135, 510, 600] {
                let mut cursor = 0usize;
                let mut buckets = Vec::with_capacity(p);
                for w in 0..p {
                    let (start, end) = contiguous_slice(total, p, w);
                    assert_eq!(start, cursor, "gap/overlap for total={total} p={p} w={w}");
                    assert!(start <= end && end <= total);
                    let len = end - start;
                    let base = total / p;
                    assert!(len == base || len == base + 1);
                    buckets.push(WorkBucket::new(start, end));
                    cursor = end;
                }
                assert_eq!(cursor, total);

                // Even-sized slices come before odd-sized slices when `base` is
                // even or when `base` is odd and `rem > 0`, keeping `256`-px
                // tile pairs `(2k, 2k+1)` together whenever `total >= 2`.
                if total >= 2 && (total / p) >= 2 && (total / p).is_multiple_of(2) {
                    let (s0, e0) = contiguous_slice(total, p, 0);
                    assert!((e0 - s0).is_multiple_of(2));
                }

                // Partial worker wake subset: only a subset of workers runs
                // (e.g. inline worker `0` plus at most 2 arbitrary workers),
                // and every task `0..total` must still be claimed exactly once.
                let active: Vec<usize> = if p == 1 {
                    vec![0]
                } else if p <= 4 {
                    vec![0, p - 1]
                } else {
                    vec![0, p / 3, p - 1]
                };
                let mut seen = vec![0u32; total];
                let mut iters: Vec<WorkStealIter<'_>> = active
                    .iter()
                    .map(|&w| WorkStealIter::new(&buckets, w))
                    .collect();
                // Interleave claims round-robin across the partial active set.
                loop {
                    let mut any = false;
                    for iter in &mut iters {
                        if let Some(idx) = iter.next_idx() {
                            seen[idx] += 1;
                            any = true;
                        }
                    }
                    if !any {
                        break;
                    }
                }
                assert!(
                    seen.iter().all(|&c| c == 1),
                    "exact-once violated for total={total} p={p}: {seen:?}"
                );
            }
        }
    }

    #[test]
    fn test_phase_b_owner_non_scroll_full_partial_and_clamping() {
        let meta = BufferMetadata {
            width: 512,
            height: 512,
            stride: 512 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let (sub_bands, _, _) = plan_incremental_scan(&meta, None).unwrap();
        assert_eq!(sub_bands.len(), 4);
        let owners: Vec<AtomicU8> = [0u8, 1, 2, 99].into_iter().map(AtomicU8::new).collect();
        // `256`-tall tiles alternate top/bottom sub-band across odd/even `x / TILE_W`.
        let r_even = TileRect {
            x: 0,
            y: 0,
            w: 256,
            h: 256,
        };
        let r_odd = TileRect {
            x: 256,
            y: 0,
            w: 256,
            h: 256,
        };
        assert_eq!(
            phase_b_owner_non_scroll(&sub_bands, &owners, r_even, true, 4),
            0
        );
        assert_eq!(
            phase_b_owner_non_scroll(&sub_bands, &owners, r_odd, true, 4),
            1
        );
        // Out-of-range owner clamps cleanly to `p - 1`.
        let r_bottom = TileRect {
            x: 0,
            y: 384,
            w: 128,
            h: 128,
        };
        assert_eq!(
            phase_b_owner_non_scroll(&sub_bands, &owners, r_bottom, true, 4),
            3
        );

        // Partial `RowBand`s mixing `Strip1D` and `RowBand2D`.
        let bands = [RowBand { y0: 4, y1: 6 }, RowBand { y0: 128, y1: 260 }];
        let (p_subs, _, _) = plan_incremental_scan(&meta, Some(&bands)).unwrap();
        let p_owners: Vec<AtomicU8> = (0..p_subs.len())
            .map(|i| AtomicU8::new((i % 4) as u8))
            .collect();
        assert_eq!(
            sub_band_idx_for_rect(
                &p_subs,
                TileRect {
                    x: 0,
                    y: 4,
                    w: 512,
                    h: 2,
                },
                false,
            ),
            0,
        );
        let mid_idx = sub_band_idx_for_rect(
            &p_subs,
            TileRect {
                x: 0,
                y: 256,
                w: 128,
                h: 4,
            },
            false,
        );
        assert_eq!(mid_idx, p_subs.len() - 1);
        assert_eq!(
            phase_b_owner_non_scroll(
                &p_subs,
                &p_owners,
                TileRect {
                    x: 0,
                    y: 256,
                    w: 128,
                    h: 4,
                },
                false,
                4,
            ),
            mid_idx % 4,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_phase_ab_affinity_multithreaded_incremental_parity() {
        // Exercises:
        // 1) tiny edit (`n_encode < PHASE_B_AFFINITY_MIN_ENCODE`, fallback path),
        // 2) partial-wake affinity (`PHASE_B_AFFINITY_MIN_ENCODE <= n_tiles < p`),
        // 3) full multi-band affinity (`n_tiles >= p`, Lossless & Hybrid, with cache),
        // 4) heap-spilled `home_tiles` (`n_tiles > STACK_HOME_TILES_CAP`) and
        //    heap-spilled `phase_a_owners` (`n_sub > MAX_STACK_OWNERS`).
        let mut comp_1t = ShardingCompressor::new(NonZeroUsize::new(1).unwrap(), 1).unwrap();
        let mut comp_mt = ShardingCompressor::new(NonZeroUsize::new(12).unwrap(), 1).unwrap();
        let mut decomp = ShardingDecompressor::new(NonZeroUsize::new(4).unwrap()).unwrap();

        let w = 1024usize;
        let h = 1024usize;
        let stride = w * 4;
        let meta = BufferMetadata {
            width: w as i32,
            height: h as i32,
            stride: stride as i32,
            format: BufferFormat::Xrgb8888,
        };
        let mut base = vec![0u8; h * stride];
        for y in 0..h {
            for x in 0..w {
                let off = y * stride + x * 4;
                base[off] = (x.wrapping_mul(7) ^ y.wrapping_mul(3)) as u8;
                base[off + 1] = (y.wrapping_mul(11) ^ (x >> 1)) as u8;
                base[off + 2] = (x.wrapping_add(y).wrapping_mul(5)) as u8;
                base[off + 3] = 0xFF;
            }
        }

        let read_stream = |wire_bytes: &[u8]| -> ValidatedTileStream {
            let mut stream_opt = None;
            let mut rdr = wire_bytes;
            CompressedTileStream::read_from_stream(&mut rdr, |s| {
                stream_opt = Some(s);
                Ok(())
            })
            .unwrap();
            ValidatedTileStream::new(stream_opt.unwrap()).unwrap()
        };

        // Case 1 (2 dirty quadrants -> fallback), Case 2 (10 dirty quadrants ->
        // partial wake with p=12), Case 3 (all 64 quadrants dirty -> full affinity).
        let dirty_quad_counts = [2usize, 10, 64];
        for &mode in &[
            CompressionMode::Lossless,
            CompressionMode::Hybrid {
                quality: 85,
                subsamp: JpegSubsamp::Sub420,
            },
        ] {
            for &n_dirty in &dirty_quad_counts {
                let mut f1 = base.clone();
                let n_qx = w / SUBTILE_W;
                for q in 0..n_dirty {
                    let qx = q % n_qx;
                    let qy = q / n_qx;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let y = qy * SUBTILE_H + dy * 4;
                            let x = qx * SUBTILE_W + dx * 4;
                            let off = y * stride + x * 4;
                            f1[off] ^= 0x5A;
                            f1[off + 1] ^= 0xA5;
                            f1[off + 2] ^= 0x3C;
                        }
                    }
                }

                let mut run = |comp: &mut ShardingCompressor| {
                    let mut shadow = ShadowBuffer::new();
                    let mut srv = ServerTileCache::new();
                    let mut cli = ClientTileCache::new();
                    let mut out = vec![0u8; base.len()];
                    let p0 = base.as_ptr();
                    let bp0 = unsafe { BufferPointer::new(&p0, base.len()) };
                    let mut s0 = None;
                    filter_and_compress_full_streaming(
                        bp0,
                        ClientId(1),
                        WlSurfaceId(1),
                        meta,
                        mode,
                        Some(&mut shadow),
                        Some(&mut srv),
                        comp,
                        |s| s0 = Some(s),
                    );
                    let mut w0 = Vec::new();
                    s0.unwrap().framed_write(&mut w0).unwrap();
                    decompress_and_unfilter_streaming(
                        read_stream(&w0),
                        &mut decomp,
                        &mut out,
                        Some(&mut cli),
                    )
                    .unwrap();

                    let p1 = f1.as_ptr();
                    let bp1 = unsafe { BufferPointer::new(&p1, f1.len()) };
                    let mut s1 = None;
                    let res = filter_and_compress_incremental_streaming(
                        bp1,
                        ClientId(1),
                        WlSurfaceId(1),
                        meta,
                        mode,
                        None,
                        &mut shadow,
                        Some(&mut srv),
                        comp,
                        |s| s1 = Some(s),
                    );
                    let mut w1 = Vec::new();
                    s1.unwrap().framed_write(&mut w1).unwrap();
                    decompress_and_unfilter_streaming(
                        read_stream(&w1),
                        &mut decomp,
                        &mut out,
                        Some(&mut cli),
                    )
                    .unwrap();
                    assert_eq!(shadow.pixels(), f1.as_slice());
                    if mode == CompressionMode::Lossless {
                        assert_eq!(out, f1);
                    }
                    (res.dirty_tiles, out, srv.occupied_slots())
                };

                let (rects_1t, out_1t, slots_1t) = run(&mut comp_1t);
                let (rects_mt, out_mt, slots_mt) = run(&mut comp_mt);
                assert_eq!(
                    rects_mt, rects_1t,
                    "rects mismatch n_dirty={n_dirty} {mode:?}"
                );
                assert_eq!(
                    out_mt, out_1t,
                    "decoded pixels mismatch n_dirty={n_dirty} {mode:?}"
                );
                assert_eq!(
                    slots_mt, slots_1t,
                    "slots mismatch n_dirty={n_dirty} {mode:?}"
                );
            }
        }

        // Case 4: heap-spill paths (`n_sub > MAX_STACK_OWNERS` AND
        // `dirty_tiles.len() > STACK_HOME_TILES_CAP`).
        let tall_w = 1024usize;
        let tall_h = (MAX_STACK_OWNERS + 2) * SUBTILE_H; // 66 sub-bands, 66*8 = 528 quadrants
        let tall_stride = tall_w * 4;
        let tall_meta = BufferMetadata {
            width: tall_w as i32,
            height: tall_h as i32,
            stride: tall_stride as i32,
            format: BufferFormat::Xrgb8888,
        };
        let mut shadow = ShadowBuffer::from_pixels(vec![0u8; tall_h * tall_stride]);
        let mut tall_f1 = shadow.pixels().to_vec();
        // Flip one pixel per quadrant so all 528 quadrants emit cropped tiles
        // (`528 > STACK_HOME_TILES_CAP = 512`), forcing both heap spills.
        for qy in 0..(tall_h / SUBTILE_H) {
            for qx in 0..(tall_w / SUBTILE_W) {
                let y = qy * SUBTILE_H + 10;
                let x = qx * SUBTILE_W + 10;
                let off = y * tall_stride + x * 4;
                tall_f1[off] = 0x77;
                tall_f1[off + 3] = 0xFF;
            }
        }
        let p_tall = tall_f1.as_ptr();
        let bp_tall = unsafe { BufferPointer::new(&p_tall, tall_f1.len()) };
        let mut stream = None;
        let res = filter_and_compress_incremental_streaming(
            bp_tall,
            ClientId(1),
            WlSurfaceId(2),
            tall_meta,
            CompressionMode::Lossless,
            None,
            &mut shadow,
            None,
            &mut comp_mt,
            |s| stream = Some(s),
        );
        assert!(
            res.dirty_tiles.len() > STACK_HOME_TILES_CAP,
            "expected > {} cropped tiles, got {}",
            STACK_HOME_TILES_CAP,
            res.dirty_tiles.len(),
        );
        let mut wire = Vec::new();
        stream.unwrap().framed_write(&mut wire).unwrap();
        let mut out = vec![0u8; tall_f1.len()];
        decompress_and_unfilter_streaming(read_stream(&wire), &mut decomp, &mut out, None).unwrap();
        assert_eq!(shadow.pixels(), tall_f1.as_slice());
        assert_eq!(out, tall_f1);
    }

    #[test]
    fn test_half_dram_hash_screening_correctness_and_invalidation() {
        let mut comp = ShardingCompressor::new(
            NonZeroUsize::new(if cfg!(miri) { 1 } else { 4 }).unwrap(),
            1,
        )
        .unwrap();
        let mut decomp =
            ShardingDecompressor::new(NonZeroUsize::new(if cfg!(miri) { 1 } else { 2 }).unwrap())
                .unwrap();

        // Non-128-multiple width & height (`5 * 128 + 29 = 669`, `2 * 128 + 20 = 276`)
        // with non-tight row stride (`+ 16` bytes padding) to cover interior, right-edge,
        // and bottom-edge quadrants across both `Xrgb8888` and `Argb8888`.
        let w = if cfg!(miri) {
            3 * SUBTILE_W + 13
        } else {
            5 * SUBTILE_W + 29
        };
        let h = if cfg!(miri) {
            SUBTILE_H + 16
        } else {
            2 * SUBTILE_H + 20
        };
        let stride = (w + 4) * 4;
        let len = h * stride;
        let n_qy = h.div_ceil(SUBTILE_H);

        let make_frame = |seed: u8, has_alpha: bool| -> Vec<u8> {
            let mut out = vec![0u8; len];
            for y in 0..h {
                for x in 0..w {
                    let off = y * stride + x * 4;
                    out[off] = seed.wrapping_add(((x * 13 + y * 7) & 0xFF) as u8);
                    out[off + 1] = seed.wrapping_add(((x * 5 + y * 11) & 0xFF) as u8);
                    out[off + 2] = seed.wrapping_add(((x * 17 + y * 3) & 0xFF) as u8);
                    out[off + 3] = if has_alpha {
                        0x80u8.wrapping_add(((x + y * 3) & 0x7F) as u8)
                    } else {
                        ((x * 19 + y) & 0xFF) as u8
                    };
                }
            }
            out
        };

        let run_pair_and_assert_exact =
            |tier: arch::Tier,
             meta: BufferMetadata,
             frames: &[(&[u8], Option<&[RowBand]>)],
             comp: &mut ShardingCompressor,
             decomp: &mut ShardingDecompressor| {
                let mut sh_screen = ShadowBuffer::new();
                let mut sh_plain = ShadowBuffer::new();
                let mut srv_screen = ServerTileCache::new();
                let mut srv_plain = ServerTileCache::new();
                let mut cli_screen = ClientTileCache::new();
                let mut cli_plain = ClientTileCache::new();
                let mut out_screen = vec![0u8; len];
                let mut out_plain = vec![0u8; len];

                // Frame 0: full sync seeds `quad_hashes`.
                let (f0, _) = frames[0];
                let p0 = f0.as_ptr();
                let bp0 = unsafe { BufferPointer::new(&p0, len) };
                let mut s_scr = None;
                let mut s_pln = None;
                filter_and_compress_full_streaming_with_tier(
                    tier,
                    bp0,
                    ClientId(1),
                    WlSurfaceId(1),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut sh_screen),
                    Some(&mut srv_screen),
                    comp,
                    |s| s_scr = Some(s),
                );
                filter_and_compress_full_streaming_with_tier(
                    tier,
                    bp0,
                    ClientId(1),
                    WlSurfaceId(1),
                    meta,
                    CompressionMode::Lossless,
                    Some(&mut sh_plain),
                    Some(&mut srv_plain),
                    comp,
                    |s| s_pln = Some(s),
                );
                assert_eq!(sh_screen.valid_band_count(), n_qy);
                let mut w_scr = Vec::new();
                let mut w_pln = Vec::new();
                s_scr.unwrap().framed_write(&mut w_scr).unwrap();
                s_pln.unwrap().framed_write(&mut w_pln).unwrap();
                let decode = |wire: &[u8],
                              out: &mut [u8],
                              cli: &mut ClientTileCache,
                              decomp: &mut ShardingDecompressor| {
                    if wire.is_empty() {
                        return;
                    }
                    let mut rx = None;
                    let mut cursor = wire;
                    CompressedTileStream::read_from_stream(&mut cursor, |s| {
                        rx = Some(s);
                        Ok(())
                    })
                    .unwrap();
                    let val = ValidatedTileStream::new(rx.unwrap()).unwrap();
                    if let Some(sr) = val.scroll_rect() {
                        apply_scroll_rect_in_place(out, stride, sr);
                    }
                    decompress_and_unfilter_streaming(val, decomp, out, Some(cli)).unwrap();
                };
                decode(&w_scr, &mut out_screen, &mut cli_screen, decomp);
                decode(&w_pln, &mut out_plain, &mut cli_plain, decomp);
                assert_eq!(out_screen, out_plain);

                for (step, &(frame, bands)) in frames[1..].iter().enumerate() {
                    let p = frame.as_ptr();
                    let bp = unsafe { BufferPointer::new(&p, len) };
                    let mut st_scr = None;
                    let mut st_pln = None;
                    let res_scr = filter_and_compress_incremental_streaming_impl(
                        tier,
                        bp,
                        ClientId(1),
                        WlSurfaceId(1),
                        meta,
                        CompressionMode::Lossless,
                        bands,
                        &mut sh_screen,
                        Some(&mut srv_screen),
                        comp,
                        true,
                        |s| st_scr = Some(s),
                    );
                    let res_pln = filter_and_compress_incremental_streaming_impl(
                        tier,
                        bp,
                        ClientId(1),
                        WlSurfaceId(1),
                        meta,
                        CompressionMode::Lossless,
                        bands,
                        &mut sh_plain,
                        Some(&mut srv_plain),
                        comp,
                        false,
                        |s| st_pln = Some(s),
                    );
                    assert_eq!(
                        res_scr, res_pln,
                        "step {step}: IncrementalFrameResult mismatch"
                    );
                    let mut wb_scr = Vec::new();
                    let mut wb_pln = Vec::new();
                    if let Some(s) = st_scr {
                        s.framed_write(&mut wb_scr).unwrap();
                    }
                    if let Some(s) = st_pln {
                        s.framed_write(&mut wb_pln).unwrap();
                    }
                    assert_eq!(wb_scr.len(), wb_pln.len(), "step {step}: wire len mismatch");
                    decode(&wb_scr, &mut out_screen, &mut cli_screen, decomp);
                    decode(&wb_pln, &mut out_plain, &mut cli_plain, decomp);
                    assert_eq!(out_screen, out_plain, "step {step}: decoded pixel mismatch");
                    assert_eq!(
                        srv_screen.occupied_slots(),
                        srv_plain.occupied_slots(),
                        "step {step}: cache slot count mismatch"
                    );
                }
                sh_screen
            };

        let tiers = arch::Tier::supported_tiers();
        for format in [BufferFormat::Xrgb8888, BufferFormat::Argb8888] {
            let has_alpha = format == BufferFormat::Argb8888;
            let meta = BufferMetadata {
                width: w as i32,
                height: h as i32,
                stride: stride as i32,
                format,
            };
            let f0 = make_frame(1, has_alpha);
            // Step 1: exact same pixels (plus X-byte perturbation on Xrgb8888 -> MUST stay Clean!).
            let mut f1 = f0.clone();
            if !has_alpha {
                for y in 0..h {
                    f1[y * stride + 5 * 4 + 3] ^= 0x77;
                }
            }
            // Step 2: 1 small cropped edit (< 1024 px) in Q0 + 1 alpha-only edit in right edge.
            let mut f2 = f1.clone();
            for y in 10..20 {
                for x in 10..20 {
                    f2[y * stride + x * 4] ^= 0x55;
                }
            }
            f2[(15 * stride) + (w - 3) * 4 + 3] ^= 0x40;
            // Step 3: identical to Step 2 (verifies prev_quad_hashes updated on small crop & edge!).
            let f3 = f2.clone();
            // Step 4: trigger mid-band bailout (`> SCREEN_MAX_MISMATCHES` dirty quads in band 0,
            // mixing small crop, large crop >= 1024 px, and `bbox.full` + right edge).
            let mut f4 = f3.clone();
            for y in 0..16 {
                for x in 0..8 {
                    f4[y * stride + x * 4 + 1] ^= 0x33;
                }
            }
            for y in 4..44 {
                for x in SUBTILE_W..(SUBTILE_W + 40) {
                    f4[y * stride + x * 4 + 2] ^= 0x44;
                }
            }
            for y in 0..SUBTILE_H {
                for x in (2 * SUBTILE_W)..(3 * SUBTILE_W) {
                    f4[y * stride + x * 4] ^= 0x66;
                }
            }
            for y in 0..SUBTILE_H {
                for x in (3 * SUBTILE_W)..w {
                    f4[y * stride + x * 4 + 1] ^= 0x22;
                }
            }
            // Step 5: full-frame redraw (triggers Tier-0 dense_dirty_bands skip + free re-seed).
            let f5 = make_frame(99, has_alpha);
            // Step 6: identical to Step 5 (must be Clean with valid bands).
            let f6 = f5.clone();
            // Step 7: unaligned partial band (`y = 4..8` Strip1D + `y = 10..40` unaligned RowBand2D).
            let mut f7 = f6.clone();
            for y in 4..40 {
                for x in 0..32 {
                    f7[y * stride + x * 4] ^= 0x19;
                }
            }
            let partial_bands = [RowBand { y0: 4, y1: 8 }, RowBand { y0: 10, y1: 40 }];
            // Step 8: follow-up full-frame incremental commit after unaligned partial invalidation.
            let f8 = f7.clone();

            let seq: [(&[u8], Option<&[RowBand]>); 9] = [
                (&f0, None),
                (&f1, None),
                (&f2, None),
                (&f3, None),
                (&f4, None),
                (&f5, None),
                (&f6, None),
                (&f7, Some(&partial_bands)),
                (&f8, None),
            ];
            for &tier in &tiers {
                let mut sh = run_pair_and_assert_exact(tier, meta, &seq, &mut comp, &mut decomp);
                // Band 0 was invalidated by unaligned `partial_bands`, while untouched bands stay valid.
                assert_eq!(sh.valid_band_count(), n_qy - 1);
                // Mutating via `pixels_mut()` invalidates all cached bands.
                sh.pixels_mut()[0] ^= 1;
                assert_eq!(sh.valid_band_count(), 0);
            }
        }

        // Verify key-change invalidation & early-disconnect invalidation.
        let meta = BufferMetadata {
            width: w as i32,
            height: h as i32,
            stride: stride as i32,
            format: BufferFormat::Xrgb8888,
        };
        let f0 = make_frame(7, false);
        let p0 = f0.as_ptr();
        let bp0 = unsafe { BufferPointer::new(&p0, len) };
        let mut sh = ShadowBuffer::new();
        let mut srv1 = ServerTileCache::new();
        filter_and_compress_full_streaming(
            bp0,
            ClientId(1),
            WlSurfaceId(1),
            meta,
            CompressionMode::Lossless,
            Some(&mut sh),
            Some(&mut srv1),
            &mut comp,
            drop,
        );
        assert_eq!(
            sh.valid_band_count(),
            0,
            "early disconnect cold-fill must invalidate quad hashes"
        );

        let mut srv2 = ServerTileCache::new();
        let mut st = None;
        filter_and_compress_full_streaming(
            bp0,
            ClientId(1),
            WlSurfaceId(1),
            meta,
            CompressionMode::Lossless,
            Some(&mut sh),
            Some(&mut srv2),
            &mut comp,
            |s| st = Some(s),
        );
        drop(st);
        assert_eq!(sh.valid_band_count(), n_qy);

        // Swapping to a different `ServerTileCache` (different random `TileHashKey`)
        // must cleanly invalidate and still detect a 1-pixel edit.
        let mut srv3 = ServerTileCache::new();
        let mut f_edit = f0;
        f_edit[4] ^= 0xFF;
        let pe = f_edit.as_ptr();
        let bpe = unsafe { BufferPointer::new(&pe, len) };
        let mut st_edit = None;
        let res = filter_and_compress_incremental_streaming(
            bpe,
            ClientId(1),
            WlSurfaceId(1),
            meta,
            CompressionMode::Lossless,
            None,
            &mut sh,
            Some(&mut srv3),
            &mut comp,
            |s| st_edit = Some(s),
        );
        assert_eq!(res.dirty_tiles.len(), 1);
        assert_eq!(sh.valid_band_count(), 0);
        assert_eq!(sh.pixels(), f_edit.as_slice());
    }
}
