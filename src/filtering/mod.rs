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
    use crate::filtering::ScalarKernel;
    use crate::filtering::ScanSubBand;
    use crate::filtering::ShadowRows;
    use crate::filtering::tile_cache::SlotHashes;
    use crate::filtering::tile_cache::ValidatedCacheHit;
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
                );
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
            slot_hashes: Option<&SlotHashes>,
            cache_non_empty: bool,
            next_sub: &AtomicUsize,
            hash_buf: &mut Vec<u8>,
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
                        slot_hashes,
                        cache_non_empty,
                        next_sub,
                        hash_buf,
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
                        slot_hashes,
                        cache_non_empty,
                        next_sub,
                        hash_buf,
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
            slot_hashes: Option<&SlotHashes>,
            cache_non_empty: bool,
            next_sub: &AtomicUsize,
            hash_buf: &mut Vec<u8>,
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
                        slot_hashes,
                        cache_non_empty,
                        next_sub,
                        hash_buf,
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
                        slot_hashes,
                        cache_non_empty,
                        next_sub,
                        hash_buf,
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
            // SAFETY: Caller upholds disjoint-row shadow invariants.
            unsafe {
                let mut hash_buf = Vec::new();
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
                    &mut hash_buf,
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
        super::tests::verify_kernel_matches_scalar(ScalarKernel, "scalar");
        vec![Tier]
    }
}

pub(crate) mod palette;
pub(crate) mod scroll;
pub(crate) mod tile_cache;

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
use zstd::bulk::Decompressor;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::palette::TextSafeClass;
use crate::filtering::palette::decode_palette_row;
use crate::filtering::palette::prepare_palette_lut;
use crate::filtering::tile_cache::SlotHashes;
use crate::filtering::tile_cache::SyncClientSlots;
use crate::filtering::tile_cache::TileCacheAction;
use crate::filtering::tile_cache::ValidatedCacheHit;
use crate::filtering::tile_cache::payload_lossless_promotion_mask;
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
/// full-quadrant BLAKE3 hash against a non-empty [`ServerTileCache`]. Tiny
/// transient edits (`< 1,024` px: cursors, status digits, single-character
/// typing) skip hashing completely, while larger cropped strips (such as a
/// popup border returning on close) still hit cached full quadrants without
/// polluting the cache on misses.
const CROP_CACHE_LOOKUP_MIN_PIXELS: usize = 1024;

/// Maximum horizontal quadrants tracked on the stack inside a single
/// `RowBand2D` (`64 * 128 = 8192` px 8K width without heap allocation).
const STACK_QX_CAP: usize = 64;

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

    fn sad_signed_bytes(self, data: &[u8]) -> u64;

    fn row_sub3_forward_sad(self, src: &[u8], dst: &mut [u8]) -> u64;

    fn row_up_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64;

    fn row_subup3_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64;

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

    fn fence_stream(self);
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
    fn sad_signed_bytes(self, data: &[u8]) -> u64 {
        data.iter()
            .map(|&b| u64::from((b as i8).unsigned_abs()))
            .sum()
    }

    #[inline(always)]
    fn row_sub3_forward_sad(self, src: &[u8], dst: &mut [u8]) -> u64 {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let mut sum = 0u64;
        for j in 0..len {
            let p = if j >= 3 { src[j - 3] } else { 0 };
            let d = src[j].wrapping_sub(p);
            dst[j] = d;
            sum += u64::from((d as i8).unsigned_abs());
        }
        sum
    }

    #[inline(always)]
    fn row_up_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let mut sum = 0u64;
        for j in 0..len {
            let d = cur[j].wrapping_sub(prev[j]);
            dst[j] = d;
            sum += u64::from((d as i8).unsigned_abs());
        }
        sum
    }

    #[inline(always)]
    fn row_subup3_forward_sad(self, cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        let mut sum = 0u64;
        for j in 0..len {
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
    fn fence_stream(self) {}
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
    cand_sub: &mut [u8],
    cand_up: &mut [u8],
    cand_su: &mut [u8],
) -> u8 {
    let mut best_p = 0u8;
    let mut best_cost = kernel.sad_signed_bytes(cur_raw);
    if best_cost == 0 {
        filt_row.copy_from_slice(cur_raw);
        return 0;
    }

    let cost_sub = kernel.row_sub3_forward_sad(cur_raw, cand_sub);
    if cost_sub < best_cost {
        best_cost = cost_sub;
        best_p = 1;
    }
    if ry > 0 && best_cost > 0 {
        let cost_up = kernel.row_up_forward_sad(cur_raw, prev_raw, cand_up);
        if cost_up < best_cost {
            best_cost = cost_up;
            best_p = 2;
        }
        if best_cost > 0 {
            let cost_su = kernel.row_subup3_forward_sad(cur_raw, prev_raw, cand_su);
            if cost_su < best_cost {
                best_p = 3;
            }
        }
    }
    let chosen: &[u8] = match best_p {
        0 => cur_raw,
        1 => cand_sub,
        2 => cand_up,
        _ => cand_su,
    };
    filt_row.copy_from_slice(chosen);
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

fn try_encode_palette_leaf<const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
    state: &mut CompressorWorkerState,
) -> Option<LeafPayload> {
    let (n_colors, packed_len) = state
        .palette
        .try_encode_2d::<HAS_ALPHA>(aos, stride_px, rect)?;
    if n_colors == 1 {
        return Some(LeafPayload::Solid(state.palette.palette[0].to_le_bytes()));
    }
    let pal_has_alpha = HAS_ALPHA
        && state.palette.palette[..n_colors]
            .iter()
            .any(|&c| (c >> 24) != 0xFF);
    let entry_bytes = if pal_has_alpha { 4 } else { 3 };
    let mut palette_bytes = Vec::with_capacity(n_colors * entry_bytes);
    for &c in &state.palette.palette[..n_colors] {
        let b = c.to_le_bytes();
        palette_bytes.extend_from_slice(if pal_has_alpha { &b[..] } else { &b[..3] });
    }
    let data = compress_bytes(
        &mut state.compressor,
        &mut state.zstd_scratch,
        &state.palette.packed[..packed_len],
    );
    Some(LeafPayload::Palette {
        has_alpha: pal_has_alpha,
        palette: palette_bytes,
        data,
    })
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
    ensure_len(&mut state.cand_sub, row_b);
    if th > 1 {
        ensure_len(&mut state.raw_row_b, row_b);
        ensure_len(&mut state.cand_up, row_b);
        ensure_len(&mut state.cand_su, row_b);
    }
    if HAS_ALPHA {
        ensure_len(&mut state.raw_alpha, TILE_PIXELS);
    }

    let raw_row_a = &mut state.raw_row_a[..row_b];
    let raw_row_b = &mut state.raw_row_b[..if th > 1 { row_b } else { 0 }];
    let filt_bgr = &mut state.filt_bgr[..total_b];
    let cand_sub = &mut state.cand_sub[..row_b];
    let cand_up = &mut state.cand_up[..if th > 1 { row_b } else { 0 }];
    let cand_su = &mut state.cand_su[..if th > 1 { row_b } else { 0 }];
    let raw_alpha = &mut state.raw_alpha[..if HAS_ALPHA { tw * th } else { 0 }];

    let mut predictors = vec![0u8; th.div_ceil(4)];
    let mut use_a = true;
    for (ry, filt_row) in filt_bgr.chunks_exact_mut(row_b).enumerate() {
        let row_start = (y0 + ry) * stride_px + x0;
        let (_, after_start) = aos.split_at(row_start);
        let (row_ptr, _) = after_start.split_at(tw);

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

        let best_p = select_and_filter_row::<K>(
            kernel, ry, cur_raw, prev_raw, filt_row, cand_sub, cand_up, cand_su,
        );
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
    state: &mut CompressorWorkerState,
) -> LeafPayload {
    if let Some(leaf) = try_encode_palette_leaf::<HAS_ALPHA>(aos, stride_px, rect, state) {
        return leaf;
    }
    tier.filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state)
}

fn encode_jpeg_leaf<const HAS_ALPHA: bool>(
    tier: arch::Tier,
    aos: BufferPointer<Vec4u8>,
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

    // Copy out of `wl_shm` via `BufferPointer` into owned L2-hot worker memory
    // before passing `&[u8]` to TurboJPEG (preserves SIGBUS / aliasing safety).
    ensure_len(&mut state.jpeg_pixels, n_px * 4);
    if HAS_ALPHA {
        ensure_len(&mut state.raw_alpha, TILE_PIXELS);
    }
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

    let data = state
        .compress_jpeg(tw, th, quality, subsamp)
        .expect("turbojpeg compression failed for valid tile");
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
    stride_px: usize,
    rect: TileRect,
    quality: u8,
    subsamp: JpegSubsamp,
    state: &mut CompressorWorkerState,
) -> TilePayload {
    let tw = rect.w as usize;
    let th = rect.h as usize;

    // Small/thin tiles (`< 16` px in either dimension, including `< 8`-row 1D
    // strips) always stay lossless where JPEG headers would bloat the payload.
    if tw < MIN_JPEG_DIM || th < MIN_JPEG_DIM {
        return TilePayload::Leaf(filter_lossless_leaf::<HAS_ALPHA>(
            tier, aos, stride_px, rect, state,
        ));
    }

    // Fast path: if the whole tile fits in `<= 256` unique colors, every
    // quadrant is guaranteed `TextSafe` — emit the encoded `Solid`/`Palette`
    // leaf directly without re-scanning.
    if let Some(leaf) = try_encode_palette_leaf::<HAS_ALPHA>(aos, stride_px, rect, state) {
        return TilePayload::Leaf(leaf);
    }

    // Non-splittable (`<= 128x128`): classify directly (`try_encode_palette_leaf`
    // already ran above, so `TextSafe` goes straight to `AosZstd`).
    if tw <= SUBTILE_W && th <= SUBTILE_H {
        return TilePayload::Leaf(
            if state
                .palette
                .classify_text_safe(aos, stride_px, rect)
                .is_text_safe()
            {
                tier.filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state)
            } else {
                encode_jpeg_leaf::<HAS_ALPHA>(tier, aos, stride_px, rect, quality, subsamp, state)
            },
        );
    }

    // Splittable (`> 128` wide or tall): classify each `128x128` quadrant and
    // merge uniform tiles into a single leaf or emit a non-recursive `Split`.
    let (quads, n_quads) = tile_quadrants(rect);
    let mut classes = [TextSafeClass::Photo; 4];
    let mut text_safe_count = 0usize;
    for q in 0..n_quads {
        let cls = state.palette.classify_text_safe(aos, stride_px, quads[q]);
        classes[q] = cls;
        text_safe_count += usize::from(cls.is_text_safe());
    }

    if text_safe_count == n_quads {
        TilePayload::Leaf(tier.filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, rect, state))
    } else if text_safe_count == 0 {
        TilePayload::Leaf(encode_jpeg_leaf::<HAS_ALPHA>(
            tier, aos, stride_px, rect, quality, subsamp, state,
        ))
    } else {
        let mut sub_tiles = Vec::with_capacity(n_quads);
        for q in 0..n_quads {
            let qrect = quads[q];
            let payload = match classes[q] {
                TextSafeClass::PaletteCandidate => {
                    filter_lossless_leaf::<HAS_ALPHA>(tier, aos, stride_px, qrect, state)
                },
                TextSafeClass::DominantBg => {
                    tier.filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, qrect, state)
                },
                TextSafeClass::Photo => {
                    if (qrect.w as usize) < MIN_JPEG_DIM || (qrect.h as usize) < MIN_JPEG_DIM {
                        tier.filter_aos_zstd_leaf::<HAS_ALPHA>(aos, stride_px, qrect, state)
                    } else {
                        encode_jpeg_leaf::<HAS_ALPHA>(
                            tier, aos, stride_px, qrect, quality, subsamp, state,
                        )
                    }
                },
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
    stride_px: usize,
    rect: TileRect,
    mode: CompressionMode,
    state: &mut CompressorWorkerState,
) -> TilePayload {
    match mode {
        CompressionMode::Lossless => TilePayload::Leaf(filter_lossless_leaf::<HAS_ALPHA>(
            tier, aos, stride_px, rect, state,
        )),
        CompressionMode::Hybrid { quality, subsamp } => {
            filter_hybrid_tile::<HAS_ALPHA>(tier, aos, stride_px, rect, quality, subsamp, state)
        },
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
/// from all other tiles being decoded concurrently.
#[inline(always)]
unsafe fn decode_jpeg_alpha_rows_with<K: RowKernel>(
    kernel: K,
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
    ensure_len(alpha_row_a, tw);
    if th > 1 {
        ensure_len(alpha_row_b, tw);
    }
    let mut use_a = true;
    for ry in 0..th {
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
        let src_row = &jpeg_rows[ry * tw..(ry + 1) * tw];
        // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
        let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
        for ((dst, &px), &a) in dst_row.iter_mut().zip(src_row).zip(a_cur.iter()) {
            dst.write(Vec4u8(px.0, px.1, px.2, a));
        }
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
    state.decompress_jpeg(data, tw, th).location(loc!())?;
    let (const_alpha, alpha_filt_slice) = prepare_alpha_decode::<HAS_ALPHA>(
        alpha,
        n_px,
        &mut state.alpha_filt,
        &mut state.decompressor,
    )
    .location(loc!())?;

    let jpeg_rows: &[Vec4u8] = bytemuck::cast_slice(&state.jpeg_pixels[..n_px * 4]);
    if const_alpha == Some(0xFF) {
        // `PixelFormat::BGRA` already sets byte 3 to `0xFF`.
        for ry in 0..th {
            let src_row = &jpeg_rows[ry * tw..(ry + 1) * tw];
            // SAFETY: Caller guarantees `rect` is in-bounds and disjoint and
            // `src_row` does not overlap `frame`.
            unsafe {
                ptr::copy_nonoverlapping(
                    src_row.as_ptr(),
                    frame.row_ptr(y0 + ry, x0).cast::<Vec4u8>(),
                    tw,
                );
            }
        }
    } else if let Some(a) = const_alpha {
        for ry in 0..th {
            let src_row = &jpeg_rows[ry * tw..(ry + 1) * tw];
            // SAFETY: Caller guarantees `rect` is in-bounds and disjoint.
            let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
            for (dst, &px) in dst_row.iter_mut().zip(src_row) {
                dst.write(Vec4u8(px.0, px.1, px.2, a));
            }
        }
    } else if HAS_ALPHA {
        let a_filt = alpha_filt_slice.unwrap();
        // SAFETY: Caller guarantees `rect` is in-bounds and disjoint.
        unsafe {
            tier.decode_jpeg_alpha_rows(
                a_filt,
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
    Ok(false)
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
            let entry_bytes = if *pal_alpha { 4 } else { 3 };
            if !palette.len().is_multiple_of(entry_bytes) {
                bail!(
                    "palette byte len {} is not a multiple of {entry_bytes}",
                    palette.len()
                );
            }
            let n_colors = palette.len() / entry_bytes;
            if !(2..=256).contains(&n_colors) {
                bail!("invalid palette color count {n_colors}");
            }
            let expected_len =
                crate::serialization::framing::packed_palette_indices_len(n_px, n_colors);
            let packed = decode_encoded_bytes(
                data,
                expected_len,
                &mut state.filt_bgr,
                &mut state.decompressor,
            )
            .location(loc!())?;
            let (lut, dec_n) =
                prepare_palette_lut(*pal_alpha, palette, packed, n_px).location(loc!())?;
            let mut idx_pos = 0usize;
            for ry in 0..th {
                // SAFETY: Caller guarantees `rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                decode_palette_row(&lut, dec_n, packed, &mut idx_pos, dst_row).location(loc!())?;
            }
            Ok(false)
        },
        LeafPayload::AosZstd {
            predictors,
            color,
            alpha,
        } => {
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

            let mut streamed = false;
            let mut use_a = true;
            for ry in 0..th {
                let c = ry * row_b;
                let filt_row = &filt_bgr[c..c + row_b];
                let p_mode = predictor_at(predictors, ry);

                let (cur_dec, prev_dec): (&mut [u8], &[u8]) = if use_a {
                    (
                        &mut state.dec_row_a[..row_b],
                        &state.dec_row_b[..if ry > 0 { row_b } else { 0 }],
                    )
                } else {
                    (&mut state.dec_row_b[..row_b], &state.dec_row_a[..row_b])
                };

                let decoded_bgr: &[u8] = match p_mode {
                    0 => {
                        let next_needs_prev = ry + 1 < th && predictor_at(predictors, ry + 1) >= 2;
                        if next_needs_prev {
                            cur_dec.copy_from_slice(filt_row);
                        }
                        filt_row
                    },
                    1 => {
                        kernel.row_sub3_inverse(filt_row, cur_dec);
                        cur_dec
                    },
                    2 => {
                        kernel.row_up_inverse(filt_row, prev_dec, cur_dec);
                        cur_dec
                    },
                    _ => {
                        kernel.row_subup3_inverse(filt_row, prev_dec, cur_dec);
                        cur_dec
                    },
                };

                // SAFETY: Caller guarantees `rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                let use_stream = allow_stream
                    && K::STREAM_ALIGN > 1
                    && tw.is_multiple_of(16)
                    && (dst_row.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN);

                if let Some(a) = const_alpha {
                    if use_stream {
                        // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                        unsafe {
                            kernel.subgreen_bgr3_to_bgra::<true, true>(decoded_bgr, &[], a, dst_row)
                        };
                        streamed = true;
                    } else {
                        // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                        unsafe {
                            kernel.subgreen_bgr3_to_bgra::<false, true>(
                                decoded_bgr,
                                &[],
                                a,
                                dst_row,
                            );
                        }
                    }
                } else {
                    let a_filt = alpha_filt_slice.unwrap();
                    let (a_cur, a_prev): (&mut [u8], &[u8]) = if use_a {
                        (
                            &mut state.alpha_row_a[..tw],
                            &state.alpha_row_b[..if ry > 0 { tw } else { 0 }],
                        )
                    } else {
                        (&mut state.alpha_row_b[..tw], &state.alpha_row_a[..tw])
                    };
                    if ry == 0 {
                        kernel.row_sub1_inverse(&a_filt[..tw], a_cur);
                    } else {
                        kernel.row_up_inverse(&a_filt[ry * tw..(ry + 1) * tw], a_prev, a_cur);
                    }
                    if use_stream {
                        // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                        unsafe {
                            kernel.subgreen_bgr3_to_bgra::<true, false>(
                                decoded_bgr,
                                a_cur,
                                0,
                                dst_row,
                            );
                        }
                        streamed = true;
                    } else {
                        // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                        unsafe {
                            kernel.subgreen_bgr3_to_bgra::<false, false>(
                                decoded_bgr,
                                a_cur,
                                0,
                                dst_row,
                            );
                        }
                    }
                }
                use_a = !use_a;
            }

            Ok(streamed)
        },
        LeafPayload::Jpeg { data, alpha } => {
            // SAFETY: Caller guarantees `rect` shifted by `dst_origin` is in-bounds and disjoint.
            unsafe {
                unfilter_jpeg_leaf::<HAS_ALPHA>(tier, rect, dst_origin, data, alpha, frame, state)
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
                    ClientTileCache::prepare_slot_target_uninit(&ctx.slots, slot_id, rect)
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

/// Raw pointer handle to the server's per-surface shadow pixel buffer
/// preserving provenance for disjoint per-tile mutable writes across workers.
struct ShadowRows {
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
///   - bit `7`: `has_quad_hash` (lookup-only full-quadrant BLAKE3 hash written
///     to [`SlotHashes`])
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
struct QuadBbox {
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
/// any `None` (clean, full, or cache-hit) sibling quadrant.
#[inline]
pub(crate) fn merge_cropped_quad_slots(grid: &mut [Option<TileRect>; 4]) {
    let mut merged_top = false;
    if let (Some(a), Some(b)) = (grid[0], grid[1])
        && let Some(m) = try_merge_adjacent(a, b, true)
    {
        grid[0] = Some(m);
        grid[1] = None;
        merged_top = true;
    }

    let mut merged_bot = false;
    if let (Some(a), Some(b)) = (grid[2], grid[3])
        && let Some(m) = try_merge_adjacent(a, b, true)
    {
        grid[2] = Some(m);
        grid[3] = None;
        merged_bot = true;
    }

    // Only attempt vertical merges when both rows merged horizontally (all 4
    // quadrants were cropped misses) or neither did (each rect remains inside
    // its own column), preventing a wide `TL+TR` merge from overlapping `BR`.
    if merged_top == merged_bot {
        if let (Some(a), Some(b)) = (grid[0], grid[2])
            && let Some(m) = try_merge_adjacent(a, b, false)
        {
            grid[0] = Some(m);
            grid[2] = None;
        }
        if let (Some(a), Some(b)) = (grid[1], grid[3])
            && let Some(m) = try_merge_adjacent(a, b, false)
        {
            grid[1] = Some(m);
            grid[3] = None;
        }
    }
}

/// Scans a `< 8`-row 1D strip linearly against `shadow`, updating `shadow`
/// from the first differing pixel onward.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
#[inline(always)]
unsafe fn diff_and_update_strip_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    stride_px: usize,
    rect: TileRect,
) -> bool {
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
            for rem_ry in (ry + 1)..h {
                let (_, rem_after) = aos.split_at((y0 + rem_ry) * stride_px + x0);
                let (rem_src, _) = rem_after.split_at(w);
                // SAFETY: Caller guarantees `rect` is in-bounds and disjoint across workers.
                let rem_dst = unsafe {
                    slice::from_raw_parts_mut(
                        shadow.row_ptr(y0 + rem_ry, x_bytes).cast::<Vec4u8>(),
                        w,
                    )
                };
                rem_src.copy_to_nonoverlapping(rem_dst);
            }
            return true;
        }
    }
    false
}

/// Scans a horizontal `<= 128`-row 2D sub-band linearly row-by-row across all
/// `n_qx` horizontal quadrants, tracking tight dirty bounding boxes per
/// quadrant and falling back to bulk contiguous `memcpy` / fused copy+hash once
/// all quadrants in the band exceed the crop area threshold.
///
/// # Safety
/// Rows `[y0, y1)` must be strictly in-bounds within `shadow` and disjoint from
/// all rows concurrently accessed by other threads.
#[inline(always)]
unsafe fn diff_and_update_row_band_2d_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    data: BufferPointer<u8>,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    width: usize,
    stride: usize,
    y0: usize,
    y1: usize,
    slot_base: usize,
    dirty_slots: &[AtomicU32],
    slot_hashes: Option<&SlotHashes>,
    cache_non_empty: bool,
    hash_buf: &mut Vec<u8>,
) {
    let stride_px = stride / 4;
    let ch = y1 - y0;
    let n_qx = dirty_slots.len();
    let mut stack_bboxes = [QuadBbox::CLEAN; STACK_QX_CAP];
    let mut heap_bboxes;
    let qx_bboxes: &mut [QuadBbox] = if n_qx <= STACK_QX_CAP {
        &mut stack_bboxes[..n_qx]
    } else {
        heap_bboxes = vec![QuadBbox::CLEAN; n_qx];
        &mut heap_bboxes
    };
    let mut full_count = 0usize;

    for y in y0..y1 {
        if full_count == n_qx {
            if let Some(hashes) = slot_hashes {
                // Fuse remaining row copy (`y..y1`) and BLAKE3 hashing per
                // `<= 128x128` quadrant so each 64 KiB quadrant stays L1/L2-hot
                // instead of streaming the entire `RowBand2D` out of L2 first.
                let copy_from_ry = y - y0;
                let mut x = 0usize;
                for (qx, slot) in dirty_slots.iter().enumerate() {
                    let cw = (width - x).min(SUBTILE_W);
                    let qrect = TileRect {
                        x: x as u32,
                        y: y0 as u32,
                        w: cw as u32,
                        h: ch as u32,
                    };
                    // SAFETY: `[y0, y1)` is strictly disjoint across workers.
                    let h = unsafe {
                        tile_cache::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                            data,
                            shadow.base,
                            stride,
                            qrect,
                            copy_from_ry,
                            hash_buf,
                        )
                    };
                    // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                    unsafe { hashes.write(slot_base + qx, h) };
                    slot.store(SlotDirty::FULL_RAW, Ordering::Relaxed);
                    x += cw;
                }
                return;
            }
            let vis_bytes = width * 4;
            if stride == vis_bytes {
                let off = y * stride;
                let bytes = (y1 - y) * stride;
                let (_, after) = data.split_at(off);
                let (chunk, _) = after.split_at(bytes);
                // SAFETY: `[y0, y1)` is strictly disjoint across workers.
                let dst = unsafe { slice::from_raw_parts_mut(shadow.row_ptr(y, 0), bytes) };
                chunk.copy_to_nonoverlapping(dst);
            } else {
                for ry in y..y1 {
                    let off = ry * stride;
                    let (_, after) = data.split_at(off);
                    let (chunk, _) = after.split_at(vis_bytes);
                    // SAFETY: `[y0, y1)` is strictly disjoint across workers.
                    let dst =
                        unsafe { slice::from_raw_parts_mut(shadow.row_ptr(ry, 0), vis_bytes) };
                    chunk.copy_to_nonoverlapping(dst);
                }
            }
            break;
        }

        let ry = y - y0;
        let (_, row_after) = aos.split_at(y * stride_px);
        let (row_aos, _) = row_after.split_at(width);
        // SAFETY: `[y0, y1)` is strictly disjoint across workers and `Vec4u8` has alignment 1.
        let dst_row =
            unsafe { slice::from_raw_parts_mut(shadow.row_ptr(y, 0).cast::<Vec4u8>(), width) };

        let mut x = 0usize;
        for bbox in qx_bboxes.iter_mut() {
            let cw = (width - x).min(SUBTILE_W);
            let (_, after_x) = row_aos.split_at(x);
            let (src_chunk, _) = after_x.split_at(cw);
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

    let mut x = 0usize;
    for (qx, (slot, bbox)) in dirty_slots.iter().zip(qx_bboxes.iter()).enumerate() {
        let cw = (width - x).min(SUBTILE_W);
        if bbox.full {
            if let Some(hashes) = slot_hashes {
                let qrect = TileRect {
                    x: x as u32,
                    y: y0 as u32,
                    w: cw as u32,
                    h: ch as u32,
                };
                // SAFETY: `[y0, y1)` is in-bounds, finalized, and exclusively owned by this worker.
                let h = unsafe {
                    tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                        shadow.base.cast_const(),
                        stride,
                        qrect,
                        hash_buf,
                    )
                };
                // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                unsafe { hashes.write(slot_base + qx, h) };
            }
            slot.store(SlotDirty::FULL_RAW, Ordering::Relaxed);
        } else if bbox.is_dirty() {
            let crop_area =
                usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
            let has_quad_hash = cache_non_empty
                && slot_hashes.is_some()
                && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
            if has_quad_hash && let Some(hashes) = slot_hashes {
                let qrect = TileRect {
                    x: x as u32,
                    y: y0 as u32,
                    w: cw as u32,
                    h: ch as u32,
                };
                // SAFETY: `[y0, y1)` is in-bounds, finalized, and exclusively owned by this worker.
                let h = unsafe {
                    tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                        shadow.base.cast_const(),
                        stride,
                        qrect,
                        hash_buf,
                    )
                };
                // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                unsafe { hashes.write(slot_base + qx, h) };
            }
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
        x += cw;
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
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    width: usize,
    stride: usize,
    stride_px: usize,
    scroll_bands: &[ScanSubBand],
    sr: ScrollRect,
    qx: usize,
    dirty_slots: &[AtomicU32],
    slot_hashes: Option<&SlotHashes>,
    cache_non_empty: bool,
    hash_buf: &mut Vec<u8>,
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
    let cx0_bytes = cx0 * 4;

    let shift = sr.dy.unsigned_abs() as usize;
    let copy_h = sr.height as usize - shift;
    let (sr_dst_y0, sr_dst_y1) = if sr.dy < 0 {
        (sr.y as usize, sr.y as usize + copy_h)
    } else {
        (sr.y as usize + shift, sr.y as usize + sr.height as usize)
    };
    let move_up = sr.dy < 0;

    let mut process_band = |band: &ScanSubBand| {
        let ScanSubBand::RowBand2D { y0, y1, slot_base } = *band else {
            unreachable!("scroll bands must be RowBand2D");
        };
        let y0 = y0 as usize;
        let y1 = y1 as usize;
        let ch = y1 - y0;
        let max_crop_area = (cw * ch * CROP_MAX_AREA_PCT) / 100;
        let mut bbox = QuadBbox::CLEAN;
        let band_overlaps_dst = ov_bytes > 0 && y1 > sr_dst_y0 && y0 < sr_dst_y1;

        let mut process_row = |y: usize| {
            let (_, row_after) = aos.split_at(y * stride_px + cx0);
            let (src_chunk, _) = row_after.split_at(cw);
            if bbox.full {
                // SAFETY: Column stripe `[cx0, cx1)` is exclusively owned by
                // this worker across `scroll_bands`, and `Vec4u8` has align 1.
                let dst_chunk = unsafe {
                    slice::from_raw_parts_mut(shadow.row_ptr(y, cx0_bytes).cast::<Vec4u8>(), cw)
                };
                src_chunk.copy_to_nonoverlapping(dst_chunk);
            } else {
                if band_overlaps_dst && y >= sr_dst_y0 && y < sr_dst_y1 {
                    let src_y = (y as isize - sr.dy as isize) as usize;
                    // SAFETY: `src_y != y` (`dy != 0`), `[ov_start, ov_end)` lies
                    // within this worker's exclusive column stripe `[cx0, cx1)`,
                    // and overlap-safe row traversal guarantees `shadow[src_y]`
                    // has not yet been overwritten.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            shadow.row_ptr(src_y, ov_x_bytes),
                            shadow.row_ptr(y, ov_x_bytes),
                            ov_bytes,
                        );
                    }
                }
                // SAFETY: Created after `ptr::copy_nonoverlapping` completes;
                // column stripe `[cx0, cx1)` is exclusively owned by this worker.
                let dst_chunk = unsafe {
                    slice::from_raw_parts_mut(shadow.row_ptr(y, cx0_bytes).cast::<Vec4u8>(), cw)
                };
                if let Some((fx, lx)) =
                    kernel.diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk)
                {
                    bbox.record_row(y - y0, fx, lx, max_crop_area);
                }
            }
        };

        if move_up {
            for y in y0..y1 {
                process_row(y);
            }
        } else {
            for y in (y0..y1).rev() {
                process_row(y);
            }
        }

        if bbox.full {
            if let Some(hashes) = slot_hashes {
                let qrect = TileRect {
                    x: cx0 as u32,
                    y: y0 as u32,
                    w: cw as u32,
                    h: ch as u32,
                };
                // SAFETY: Column stripe `[cx0, cx1)` over `[y0, y1)` is in-bounds,
                // finalized, and exclusively owned by this worker.
                let h = unsafe {
                    tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                        shadow.base.cast_const(),
                        stride,
                        qrect,
                        hash_buf,
                    )
                };
                // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                unsafe { hashes.write(slot_base + qx, h) };
            }
            dirty_slots[slot_base + qx].store(SlotDirty::FULL_RAW, Ordering::Relaxed);
        } else if bbox.is_dirty() {
            let crop_area =
                usize::from(bbox.max_x - bbox.min_x) * usize::from(bbox.max_y - bbox.min_y);
            let has_quad_hash = cache_non_empty
                && slot_hashes.is_some()
                && crop_area >= CROP_CACHE_LOOKUP_MIN_PIXELS;
            if has_quad_hash && let Some(hashes) = slot_hashes {
                let qrect = TileRect {
                    x: cx0 as u32,
                    y: y0 as u32,
                    w: cw as u32,
                    h: ch as u32,
                };
                // SAFETY: Column stripe `[cx0, cx1)` over `[y0, y1)` is in-bounds,
                // finalized, and exclusively owned by this worker.
                let h = unsafe {
                    tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                        shadow.base.cast_const(),
                        stride,
                        qrect,
                        hash_buf,
                    )
                };
                // SAFETY: `slot_base + qx` is exclusively owned by this worker.
                unsafe { hashes.write(slot_base + qx, h) };
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
    };

    if move_up {
        for band in scroll_bands {
            process_band(band);
        }
    } else {
        for band in scroll_bands.iter().rev() {
            process_band(band);
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
    slot_hashes: Option<&SlotHashes>,
    cache_non_empty: bool,
    next_sub: &AtomicUsize,
    hash_buf: &mut Vec<u8>,
) {
    let n_sub = sub_bands.len();
    loop {
        let idx = next_sub.fetch_add(1, Ordering::Relaxed);
        if idx >= n_sub {
            break;
        }
        match sub_bands[idx] {
            ScanSubBand::Strip1D { rect, slot_idx } => {
                // SAFETY: `sub_bands` cover disjoint regions in `shadow`.
                let dirty = unsafe {
                    diff_and_update_strip_with::<K, HAS_ALPHA>(kernel, aos, shadow, stride_px, rect)
                };
                if dirty {
                    if let Some(hashes) = slot_hashes {
                        // SAFETY: `rect` is in-bounds, finalized, and exclusively owned.
                        let h = unsafe {
                            tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                                shadow.base.cast_const(),
                                stride,
                                rect,
                                hash_buf,
                            )
                        };
                        // SAFETY: `slot_idx` is exclusively owned by this worker.
                        unsafe { hashes.write(slot_idx, h) };
                    }
                    dirty_slots[slot_idx].store(SlotDirty::FULL_RAW, Ordering::Relaxed);
                }
            },
            ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                let slots = &dirty_slots[slot_base..slot_base + n_qx];
                // SAFETY: `sub_bands` cover disjoint row ranges in `shadow`.
                unsafe {
                    diff_and_update_row_band_2d_with::<K, HAS_ALPHA>(
                        kernel,
                        data,
                        aos,
                        shadow,
                        width,
                        stride,
                        y0 as usize,
                        y1 as usize,
                        slot_base,
                        slots,
                        slot_hashes,
                        cache_non_empty,
                        hash_buf,
                    );
                }
            },
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
    slot_hashes: Option<&SlotHashes>,
    cache_non_empty: bool,
    next_sub: &AtomicUsize,
    hash_buf: &mut Vec<u8>,
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
                    aos,
                    shadow,
                    width,
                    stride,
                    stride_px,
                    scroll_bands,
                    sr,
                    task,
                    dirty_slots,
                    slot_hashes,
                    cache_non_empty,
                    hash_buf,
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
                    let dirty = unsafe {
                        diff_and_update_strip_with::<K, HAS_ALPHA>(
                            kernel, aos, shadow, stride_px, rect,
                        )
                    };
                    if dirty {
                        if let Some(hashes) = slot_hashes {
                            // SAFETY: `rect` is in-bounds, finalized, and exclusively owned.
                            let h = unsafe {
                                tile_cache::hash_rect_from_shadow::<HAS_ALPHA>(
                                    shadow.base.cast_const(),
                                    stride,
                                    rect,
                                    hash_buf,
                                )
                            };
                            // SAFETY: `slot_idx` is exclusively owned by this worker.
                            unsafe { hashes.write(slot_idx, h) };
                        }
                        dirty_slots[slot_idx].store(SlotDirty::FULL_RAW, Ordering::Relaxed);
                    }
                },
                ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                    let slots = &dirty_slots[slot_base..slot_base + n_qx];
                    // SAFETY: Outside `sb_start..sb_end`, disjoint from `sr`.
                    unsafe {
                        diff_and_update_row_band_2d_with::<K, HAS_ALPHA>(
                            kernel,
                            data,
                            aos,
                            shadow,
                            width,
                            stride,
                            y0 as usize,
                            y1 as usize,
                            slot_base,
                            slots,
                            slot_hashes,
                            cache_non_empty,
                            hash_buf,
                        );
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
    shadow_out: Option<&mut Vec<u8>>,
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

    let width_bytes = metadata.width as usize * 4;
    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let (_, base_entries, total_slots) =
        plan_incremental_scan(&metadata, None).expect("full-frame tile plan cannot fail");

    let mut shadow_opt = shadow_out;
    if let Some(shadow) = shadow_opt.as_deref_mut() {
        shadow.clear();
        if shadow.capacity() < byte_len {
            shadow.reserve_exact(byte_len);
        } else if shadow.capacity() > byte_len.saturating_mul(2) {
            shadow.shrink_to(byte_len);
        }
    }

    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };

    if let Some(cache) = tile_cache {
        if cache.occupied_slots() == 0 {
            // Fast single-pass cold-fill: when the cache starts empty, no
            // inter-frame hits or partial quadrant un-coalescing can occur, so
            // `tile_rects` is known upfront and each worker fuses
            // copy-to-shadow + BLAKE3 hashing + intra-frame dedup + encoding
            // while the tile's `<= 256 KiB` pixels stay hot in L2 cache.
            let tile_rects: Vec<TileRect> = base_entries
                .iter()
                .copied()
                .map(BaseScanEntry::rect)
                .collect();
            let n_tiles = tile_rects.len();
            let reset_tile_cache = cache.take_reset_pending();
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

            let shadow_rows = shadow_opt.as_deref_mut().map(|shadow| ShadowRows {
                base: shadow.as_mut_ptr(),
                stride,
            });
            let is_hybrid = matches!(mode, CompressionMode::Hybrid { .. });
            let promotions: Vec<AtomicU8> = if is_hybrid {
                (0..MAX_TILE_CACHE_SLOTS)
                    .map(|_| AtomicU8::new(0))
                    .collect()
            } else {
                Vec::new()
            };
            let cold_state = Mutex::new(cache.new_cold_fill_state(n_tiles, total_slots));
            let frontier = tile_cache::ColdFillFrontier::new(n_tiles);
            let next_hash = AtomicUsize::new(0);
            let disconnected = AtomicBool::new(false);
            compressor.pool.execute(n_tiles, |state| {
                let shadow_ptr;
                let encode_aos = if let Some(shadow) = &shadow_rows {
                    shadow_ptr = shadow.base.cast_const().cast::<Vec4u8>();
                    // SAFETY: `shadow.base` points to `total_pixels * 4` bytes of
                    // allocated capacity; each tile's rect is read only after its
                    // `publish_hash` -> `advance` -> `planned_up_to` Release/Acquire
                    // happens-before edge completes its exclusive write.
                    unsafe { BufferPointer::new(&shadow_ptr, total_pixels) }
                } else {
                    aos
                };

                let encode_and_send =
                    |enc_idx: usize, action: TileCacheAction, state: &mut CompressorWorkerState| {
                        let rect = base_entries[enc_idx].rect();
                        let tile = match action {
                            TileCacheAction::Hit(hit_ref) => CompressedTile {
                                tile_idx: enc_idx as u32,
                                cache_put: None,
                                payload: TilePayload::CacheHit(hit_ref),
                            },
                            TileCacheAction::Put(slot_id) => {
                                let payload = if has_alpha {
                                    filter_tile::<true>(
                                        tier, encode_aos, stride_px, rect, mode, state,
                                    )
                                } else {
                                    filter_tile::<false>(
                                        tier, encode_aos, stride_px, rect, mode, state,
                                    )
                                };
                                if is_hybrid {
                                    promotions[slot_id as usize].store(
                                        payload_lossless_promotion_mask(rect, &payload),
                                        Ordering::Relaxed,
                                    );
                                }
                                CompressedTile {
                                    tile_idx: enc_idx as u32,
                                    cache_put: Some(slot_id),
                                    payload,
                                }
                            },
                            TileCacheAction::Uncached => {
                                let payload = if has_alpha {
                                    filter_tile::<true>(
                                        tier, encode_aos, stride_px, rect, mode, state,
                                    )
                                } else {
                                    filter_tile::<false>(
                                        tier, encode_aos, stride_px, rect, mode, state,
                                    )
                                };
                                CompressedTile {
                                    tile_idx: enc_idx as u32,
                                    cache_put: None,
                                    payload,
                                }
                            },
                        };
                        if tiles_tx.send(tile).is_err() {
                            disconnected.store(true, Ordering::Relaxed);
                        }
                    };

                loop {
                    if disconnected.load(Ordering::Relaxed) {
                        if let Some(shadow) = &shadow_rows {
                            let h = next_hash.fetch_add(1, Ordering::Relaxed);
                            if h >= n_tiles {
                                break;
                            }
                            let rect = base_entries[h].rect();
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

                    let h = next_hash.fetch_add(1, Ordering::Relaxed);
                    if h < n_tiles {
                        let entry = base_entries[h];
                        let (hash, sub_quads) = match entry {
                            BaseScanEntry::Strip1D { rect, .. } => {
                                let digest = if let Some(shadow) = &shadow_rows {
                                    // SAFETY: `base_entries` are pairwise disjoint and in-bounds.
                                    unsafe {
                                        let h = if has_alpha {
                                            tile_cache::copy_and_hash_rect_to_shadow::<true>(
                                                data,
                                                shadow.base,
                                                stride,
                                                rect,
                                                0,
                                                &mut state.hash_buf,
                                            )
                                        } else {
                                            tile_cache::copy_and_hash_rect_to_shadow::<false>(
                                                data,
                                                shadow.base,
                                                stride,
                                                rect,
                                                0,
                                                &mut state.hash_buf,
                                            )
                                        };
                                        zero_shadow_row_padding(shadow, width_bytes, rect);
                                        h
                                    }
                                } else if has_alpha {
                                    tile_cache::hash_rect_from_ptr::<true>(
                                        aos,
                                        stride_px,
                                        rect,
                                        &mut state.hash_buf,
                                    )
                                } else {
                                    tile_cache::hash_rect_from_ptr::<false>(
                                        aos,
                                        stride_px,
                                        rect,
                                        &mut state.hash_buf,
                                    )
                                };
                                (digest, None)
                            },
                            BaseScanEntry::Tile2D { rect, n_quads, .. } => {
                                let (quads, _) = tile_quadrants(rect);
                                let mut quad_hashes = [[0u8; 16]; 4];
                                for q in 0..n_quads {
                                    quad_hashes[q] = if let Some(shadow) = &shadow_rows {
                                        // SAFETY: `quads[q]` lies inside exclusively owned `rect`.
                                        unsafe {
                                            if has_alpha {
                                                tile_cache::copy_and_hash_rect_to_shadow::<true>(
                                                    data,
                                                    shadow.base,
                                                    stride,
                                                    quads[q],
                                                    0,
                                                    &mut state.hash_buf,
                                                )
                                            } else {
                                                tile_cache::copy_and_hash_rect_to_shadow::<false>(
                                                    data,
                                                    shadow.base,
                                                    stride,
                                                    quads[q],
                                                    0,
                                                    &mut state.hash_buf,
                                                )
                                            }
                                        }
                                    } else if has_alpha {
                                        tile_cache::hash_rect_from_ptr::<true>(
                                            aos,
                                            stride_px,
                                            quads[q],
                                            &mut state.hash_buf,
                                        )
                                    } else {
                                        tile_cache::hash_rect_from_ptr::<false>(
                                            aos,
                                            stride_px,
                                            quads[q],
                                            &mut state.hash_buf,
                                        )
                                    };
                                }
                                if let Some(shadow) = &shadow_rows {
                                    // SAFETY: `rect` is exclusively owned by this worker and
                                    // disjoint from all other `base_entries`.
                                    unsafe { zero_shadow_row_padding(shadow, width_bytes, rect) };
                                }
                                if n_quads == 1 {
                                    (quad_hashes[0], None)
                                } else {
                                    let parent_hash = tile_cache::combine_quad_hashes(
                                        rect.w,
                                        rect.h,
                                        &quad_hashes[..n_quads],
                                    );
                                    (parent_hash, Some((quads, quad_hashes, n_quads)))
                                }
                            },
                        };

                        // SAFETY: `h` was uniquely claimed from `next_hash`.
                        unsafe { frontier.publish_hash(h, hash, sub_quads) };
                        frontier.advance(&cold_state, &base_entries);

                        // Fast path: if the deterministic frontier already passed `h`,
                        // encode `h` immediately while its pixels sit hot in L2 cache.
                        if let Some(action) = frontier.try_claim_encode(h) {
                            encode_and_send(h, action, state);
                        } else if let Some((enc_idx, action)) = frontier.try_claim_next_planned() {
                            encode_and_send(enc_idx, action, state);
                        }
                        continue;
                    }

                    if let Some((enc_idx, action)) = frontier.try_claim_next_planned() {
                        encode_and_send(enc_idx, action, state);
                        continue;
                    }
                    if frontier.is_done() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            });

            if let Some(shadow) = shadow_opt {
                // SAFETY: `base_entries` partitions `[0, width) x [0, height)`,
                // every tile was written (even on early stream disconnect), and
                // right-edge tiles zeroed trailing row-stride padding, so all
                // `byte_len` bytes are initialized.
                unsafe { shadow.set_len(byte_len) };
            }

            if disconnected.load(Ordering::Relaxed) {
                cache.reset_for_peer();
            } else {
                cache.commit_cold_fill(
                    cold_state.into_inner().unwrap().into_pending(),
                    has_alpha,
                    mode,
                    &promotions,
                );
            }
            return;
        }

        // Phase A: copy all base tiles to `shadow` (when requested) and compute
        // 128-bit BLAKE3 digests for every quadrant while hot in L1/L2 cache.
        let slot_hashes = SlotHashes::new(total_slots);
        {
            let _hash_span = debug_span!("tile_cache_hash_full").entered();
            let shadow_rows = shadow_opt.as_deref_mut().map(|shadow| ShadowRows {
                base: shadow.as_mut_ptr(),
                stride,
            });
            let n_entries = base_entries.len();
            let next_entry = AtomicUsize::new(0);
            compressor.pool.execute(n_entries, |state| {
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
                                let h = unsafe {
                                    let h = if has_alpha {
                                        tile_cache::copy_and_hash_rect_to_shadow::<true>(
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                            &mut state.hash_buf,
                                        )
                                    } else {
                                        tile_cache::copy_and_hash_rect_to_shadow::<false>(
                                            data,
                                            shadow.base,
                                            stride,
                                            rect,
                                            0,
                                            &mut state.hash_buf,
                                        )
                                    };
                                    zero_shadow_row_padding(shadow, width_bytes, rect);
                                    h
                                };
                                // SAFETY: `slot_idx` is exclusively owned by this entry.
                                unsafe { slot_hashes.write(slot_idx, h) };
                            },
                            BaseScanEntry::Tile2D {
                                rect,
                                quad_slots,
                                n_quads,
                            } => {
                                let (quads, _) = tile_quadrants(rect);
                                for q in 0..n_quads {
                                    // SAFETY: `quads[q]` lies inside exclusively owned `rect`.
                                    let h = unsafe {
                                        if has_alpha {
                                            tile_cache::copy_and_hash_rect_to_shadow::<true>(
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                                &mut state.hash_buf,
                                            )
                                        } else {
                                            tile_cache::copy_and_hash_rect_to_shadow::<false>(
                                                data,
                                                shadow.base,
                                                stride,
                                                quads[q],
                                                0,
                                                &mut state.hash_buf,
                                            )
                                        }
                                    };
                                    // SAFETY: `quad_slots[q]` is exclusively owned by this entry.
                                    unsafe { slot_hashes.write(quad_slots[q], h) };
                                }
                                // SAFETY: `rect` is exclusively owned by this worker and
                                // disjoint from all other `base_entries`.
                                unsafe { zero_shadow_row_padding(shadow, width_bytes, rect) };
                            },
                        }
                    } else {
                        match entry {
                            BaseScanEntry::Strip1D { rect, slot_idx } => {
                                let h = if has_alpha {
                                    tile_cache::hash_rect_from_ptr::<true>(
                                        aos,
                                        stride_px,
                                        rect,
                                        &mut state.hash_buf,
                                    )
                                } else {
                                    tile_cache::hash_rect_from_ptr::<false>(
                                        aos,
                                        stride_px,
                                        rect,
                                        &mut state.hash_buf,
                                    )
                                };
                                // SAFETY: `slot_idx` is exclusively owned by this entry.
                                unsafe { slot_hashes.write(slot_idx, h) };
                            },
                            BaseScanEntry::Tile2D {
                                rect,
                                quad_slots,
                                n_quads,
                            } => {
                                let (quads, _) = tile_quadrants(rect);
                                for q in 0..n_quads {
                                    let h = if has_alpha {
                                        tile_cache::hash_rect_from_ptr::<true>(
                                            aos,
                                            stride_px,
                                            quads[q],
                                            &mut state.hash_buf,
                                        )
                                    } else {
                                        tile_cache::hash_rect_from_ptr::<false>(
                                            aos,
                                            stride_px,
                                            quads[q],
                                            &mut state.hash_buf,
                                        )
                                    };
                                    // SAFETY: `quad_slots[q]` is exclusively owned by this entry.
                                    unsafe { slot_hashes.write(quad_slots[q], h) };
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
            unsafe { shadow.set_len(byte_len) };
        }

        let hashes = slot_hashes.into_vec();
        let plan = cache.plan_frame(&base_entries, None, &hashes, has_alpha, mode);
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
            shadow_ptr_storage = shadow.as_ptr().cast::<Vec4u8>();
            // SAFETY: Phase A workers have joined, `shadow.len() == total_pixels * 4`,
            // and `Vec4u8` has alignment 1.
            unsafe { BufferPointer::new(&shadow_ptr_storage, total_pixels) }
        } else {
            aos
        };

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
                let tile = match plan.actions[tile_idx] {
                    TileCacheAction::Hit(hit_ref) => CompressedTile {
                        tile_idx: tile_idx as u32,
                        cache_put: None,
                        payload: TilePayload::CacheHit(hit_ref),
                    },
                    TileCacheAction::Put(slot_id) => {
                        let payload = if has_alpha {
                            filter_tile::<true>(tier, encode_aos, stride_px, rect, mode, state)
                        } else {
                            filter_tile::<false>(tier, encode_aos, stride_px, rect, mode, state)
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
                            filter_tile::<true>(tier, encode_aos, stride_px, rect, mode, state)
                        } else {
                            filter_tile::<false>(tier, encode_aos, stride_px, rect, mode, state)
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
        base: shadow.as_mut_ptr(),
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
            let encode_aos = if let Some(shadow) = &shadow_rows {
                // SAFETY: `rects_for_workers` are pairwise disjoint and in-bounds,
                // and each `tile_idx` is claimed once.
                unsafe {
                    copy_rect_to_shadow(data, shadow, stride, rect);
                    zero_shadow_row_padding(shadow, width_bytes, rect);
                }
                if disconnected {
                    continue;
                }
                shadow_ptr = shadow.base.cast_const().cast::<Vec4u8>();
                // SAFETY: `shadow.base` points to `total_pixels * 4` bytes of
                // allocated capacity and this worker holds exclusive access to
                // `rect`'s initialized pixels after `copy_rect_to_shadow`.
                unsafe { BufferPointer::new(&shadow_ptr, total_pixels) }
            } else {
                aos
            };
            let payload = if has_alpha {
                filter_tile::<true>(tier, encode_aos, stride_px, rect, mode, state)
            } else {
                filter_tile::<false>(tier, encode_aos, stride_px, rect, mode, state)
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
        unsafe { shadow.set_len(byte_len) };
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
    shadow_out: Option<&mut Vec<u8>>,
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

fn filter_and_compress_incremental_streaming_with_tier(
    tier: arch::Tier,
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    mode: CompressionMode,
    bands: Option<&[RowBand]>,
    shadow: &mut [u8],
    mut tile_cache: Option<&mut ServerTileCache>,
    compressor: &mut ShardingCompressor,
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
    assert!(shadow.len() >= byte_len);

    let (sub_bands, base_entries, total_slots) =
        plan_incremental_scan(&metadata, bands).expect("server generated invalid row bands");
    if sub_bands.is_empty() {
        return IncrementalFrameResult {
            scroll_rect: None,
            dirty_tiles: Vec::new(),
        };
    }

    let width = metadata.width as usize;
    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let n_qx = width.div_ceil(SUBTILE_W);
    let n_sub = sub_bands.len();
    let dirty_slots: Vec<AtomicU32> = (0..total_slots)
        .map(|_| AtomicU32::new(SlotDirty::CLEAN_RAW))
        .collect();
    let slot_hashes = tile_cache.is_some().then(|| SlotHashes::new(total_slots));
    let cache_non_empty = tile_cache
        .as_deref()
        .is_some_and(|c| c.occupied_slots() > 0);
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };

    // Detect vertical scroll blit and fuse overlap-safe parallel column-stripe
    // shadow scrolling into Phase A when present, leaving the non-scroll Phase
    // A worker loop untouched for zero codegen overhead on non-scroll frames.
    let scroll_rect = scroll::detect_scroll_rect(data, shadow, metadata, bands);

    // Phase A: parallel dirty scan + tight bbox tracking + shadow update (+
    // dirty quadrant hashing when `tile_cache` is enabled, with optional fused
    // scroll).
    {
        let shadow_rows = ShadowRows {
            base: shadow.as_mut_ptr(),
            stride,
        };
        let next_sub = AtomicUsize::new(0);
        if let Some(sr) = scroll_rect {
            let (sb_start, sb_end) = scroll_sub_band_range(&sub_bands, sr);
            let phase_a_tasks = (sub_bands.len() - (sb_end - sb_start)) + n_qx;
            compressor.pool.execute(phase_a_tasks, |state| {
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
                        slot_hashes.as_ref(),
                        cache_non_empty,
                        &next_sub,
                        &mut state.hash_buf,
                    );
                }
            });
        } else {
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
                        slot_hashes.as_ref(),
                        cache_non_empty,
                        &next_sub,
                        &mut state.hash_buf,
                    );
                }
            });
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

    let (dirty_tiles, actions, reset_tile_cache, evict_slots) = if let Some(cache) =
        tile_cache.as_deref_mut()
    {
        let hashes = slot_hashes
            .expect("slot_hashes allocated when tile_cache is Some")
            .into_vec();
        let plan = cache.plan_frame(&base_entries, Some(&dirty_slots), &hashes, has_alpha, mode);
        (
            plan.tile_rects,
            Some(plan.actions),
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
                                    crop_grid[slot] = Some(SlotDirty::cropped_rect(
                                        qrect, min_x, max_x, min_y, max_y,
                                    ));
                                },
                            }
                        }
                        merge_cropped_quad_slots(&mut crop_grid);
                        for s in 0..4 {
                            if let Some(r) = full_grid[s].or(crop_grid[s]) {
                                dirty_tiles.push(r);
                            }
                        }
                    }
                },
            }
        }
        (dirty_tiles, None, false, Vec::new())
    };

    // Phase B: stream dirty tiles encoded directly from the updated `shadow`
    // buffer (skipping encoding completely for `CacheHit` tiles).
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

    let shadow_ptr = shadow.as_ptr().cast::<Vec4u8>();
    // SAFETY: `shadow` has length `>= total_pixels * 4`, Phase A workers have
    // joined, and `Vec4u8` is `repr(C, packed)` with alignment 1.
    let shadow_aos = unsafe { BufferPointer::new(&shadow_ptr, total_pixels) };
    let is_hybrid = actions.is_some() && matches!(mode, CompressionMode::Hybrid { .. });
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
            let rect = dirty_tiles[tile_idx];
            let action = actions
                .as_ref()
                .map_or(TileCacheAction::Uncached, |acts| acts[tile_idx]);
            let tile = match action {
                TileCacheAction::Hit(hit_ref) => CompressedTile {
                    tile_idx: tile_idx as u32,
                    cache_put: None,
                    payload: TilePayload::CacheHit(hit_ref),
                },
                TileCacheAction::Put(slot_id) => {
                    let payload = if has_alpha {
                        filter_tile::<true>(tier, shadow_aos, stride_px, rect, mode, state)
                    } else {
                        filter_tile::<false>(tier, shadow_aos, stride_px, rect, mode, state)
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
                        filter_tile::<true>(tier, shadow_aos, stride_px, rect, mode, state)
                    } else {
                        filter_tile::<false>(tier, shadow_aos, stride_px, rect, mode, state)
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
    shadow: &mut [u8],
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

        cache_ctx.map(|ctx| (ctx.hit_refs, ctx.hit_count.into_inner()))
    };

    // Always recompute cache byte accounting after Step 1 joins so even a
    // partially failed frame never leaves stale byte accounting.
    let cache_budget_res = tile_cache
        .as_deref_mut()
        .map(ClientTileCache::recompute_and_verify_bytes)
        .transpose();

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
        && let Some((hit_refs, hit_count)) = step1_hits
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

            assert_eq!(
                kernel.sad_signed_bytes(&bgr0_a),
                ScalarKernel.sad_signed_bytes(&bgr0_s),
                "[{tier_name}] sad_signed_bytes tw={tw}"
            );

            // Sub3 forward + inverse
            let (mut f_s, mut f_a) = (vec![0u8; row_b], vec![0u8; row_b]);
            assert_eq!(
                kernel.row_sub3_forward_sad(&bgr0_a, &mut f_a),
                ScalarKernel.row_sub3_forward_sad(&bgr0_s, &mut f_s),
                "[{tier_name}] sub3 sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "[{tier_name}] sub3 fwd tw={tw}");
            let mut inv = vec![0xAAu8; row_b];
            kernel.row_sub3_inverse(&f_a, &mut inv);
            assert_eq!(inv, bgr0_s, "[{tier_name}] sub3 inv tw={tw}");

            // Up forward + inverse
            assert_eq!(
                kernel.row_up_forward_sad(&bgr1_a, &bgr0_a, &mut f_a),
                ScalarKernel.row_up_forward_sad(&bgr1_s, &bgr0_s, &mut f_s),
                "[{tier_name}] up sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "[{tier_name}] up fwd tw={tw}");
            inv.fill(0xAA);
            kernel.row_up_inverse(&f_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "[{tier_name}] up inv tw={tw}");

            // SubUp3 forward + inverse
            assert_eq!(
                kernel.row_subup3_forward_sad(&bgr1_a, &bgr0_a, &mut f_a),
                ScalarKernel.row_subup3_forward_sad(&bgr1_s, &bgr0_s, &mut f_s),
                "[{tier_name}] subup3 sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "[{tier_name}] subup3 fwd tw={tw}");
            inv.fill(0xAA);
            kernel.row_subup3_inverse(&f_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "[{tier_name}] subup3 inv tw={tw}");

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
        // `i8::MIN` (`0x80` = `-128`) unsigned abs must equal `128` across all backends
        // (covers 16B, 32B, and 64B SIMD loops plus scalar tails).
        let min_bytes = [0x80u8; 73];
        assert_eq!(
            kernel.sad_signed_bytes(&min_bytes),
            73 * 128,
            "[{tier_name}] i8::MIN sad"
        );
        assert_eq!(kernel.is_constant_plane(&[]), None);
    }

    /// Compares every compiled, host-supported backend tier bit-for-bit against
    /// [`ScalarKernel`] across all edge widths for every row primitive.
    #[test]
    fn test_arch_kernel_matches_scalar_reference() {
        let tested = arch::verify_all_supported_kernels();
        assert_eq!(tested, arch::Tier::supported_tiers());
        assert!(!tested.is_empty());
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
                let mut shadow = lossless_frame.clone();
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
                assert_eq!(shadow, edited, "shadow sync mismatch for {tier:?}");
                if let Some(prev_dirty) = &ref_dirty {
                    assert_eq!(&dirty, prev_dirty, "dirty rects mismatch for {tier:?}");
                } else {
                    ref_dirty = Some(dirty);
                }
                if let Some(prev_shadow) = &ref_shadow {
                    assert_eq!(&shadow, prev_shadow, "shadow parity mismatch for {tier:?}");
                } else {
                    ref_shadow = Some(shadow);
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
                    let mut shadow = Vec::new();
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
            let mut shadow_dropped = Vec::new();
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
            assert_eq!(shadow_dropped, data_2);

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
            let mut shadow = Vec::new();
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

                let mut shadow = Vec::new();
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
            let mut shadow = Vec::new();
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
            let mut shadow = Vec::new();
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

            let mut shadow = Vec::new();
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
            let mut shadow = Vec::new();
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
                let mut shadow = Vec::new();
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
            let mut shadow4k = Vec::new();
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
            let mut shadow = Vec::new();
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
            //    first tile is `CachePut(slot)`, other 3 are `CacheHit(slot)` in the SAME frame!
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
            let put_slot = tiles[0].cache_put.expect("tile 0 must be CachePut");
            assert_eq!(put_slot, 0, "deterministic raster-order slot 0 for tile 0");
            for t in &tiles[1..] {
                assert_eq!(t.cache_put, None);
                assert_matches!(
                    t.payload,
                    TilePayload::CacheHit(href) if href.slot_id == put_slot
                );
            }
            assert_canvas_matches_frame(&out, &dup_frame, meta_dup, 0);

            // Repeat cold-fill encode on an 8-tile mixed frame (unique +
            // duplicates) to verify deterministic raster-order slot IDs and
            // dedup winners across parallel worker scheduling.
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
            let expected_decisions = [
                (0u32, Some(0u16), None),
                (1, Some(1), None),
                (2, None, Some(0u16)),
                (3, Some(2), None),
                (4, Some(3), None),
                (5, None, Some(0)),
                (6, Some(4), None),
                (7, None, Some(1)),
            ];
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
                let got: Vec<(u32, Option<u16>, Option<u16>)> = det_tiles
                    .iter()
                    .map(|t| {
                        let hit_slot = match t.payload {
                            TilePayload::CacheHit(href) => Some(href.slot_id),
                            _ => None,
                        };
                        (t.tile_idx, t.cache_put, hit_slot)
                    })
                    .collect();
                assert_eq!(got, expected_decisions);
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
            let make_solid_128 = |id: u32| -> Vec<u8> {
                let px = [
                    (id & 0xFF) as u8,
                    ((id >> 8) & 0xFF) as u8,
                    ((id >> 16) & 0xFF) as u8,
                    0xFF,
                ];
                px.repeat(128 * 128)
            };
            for id in 0..MAX_TILE_CACHE_SLOTS as u32 {
                let f = make_solid_128(id);
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
            let mut hashes = Vec::with_capacity(MAX_TILE_CACHE_SLOTS + 6);
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
                hashes.push(h);
            }
            let plan = srv.plan_frame(&entries, None, &hashes, false, CompressionMode::Lossless);
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
            let mut shadow = Vec::new();
            let mut canvas = vec![0u8; len];

            let run_step = |frame: &[u8],
                            full: bool,
                            mode: CompressionMode,
                            shadow: &mut Vec<u8>,
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
            let mut shadow = Vec::new();
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

        // Horizontal seam merge (`128x12 + 128x12 -> 256x12`).
        let mut g = [
            None,
            None,
            Some(tr(0, 244, 128, 12)),
            Some(tr(128, 244, 128, 12)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [None, None, Some(tr(0, 244, 256, 12)), None]);

        // Vertical seam merge (`12x128 + 12x128 -> 12x256`).
        let mut g = [
            None,
            Some(tr(244, 0, 12, 128)),
            None,
            Some(tr(244, 128, 12, 128)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [None, Some(tr(244, 0, 12, 256)), None, None]);

        // 4-quadrant seam merge across both horizontal and vertical seams.
        let mut g = [
            Some(tr(0, 118, 128, 10)),
            Some(tr(128, 118, 128, 10)),
            Some(tr(0, 128, 128, 10)),
            Some(tr(128, 128, 128, 10)),
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(g, [Some(tr(0, 118, 256, 20)), None, None, None]);

        // Safety invariant: when `TL+TR` merge horizontally and `BL` is cropped
        // while `BR` is `None` (clean / Full / CacheHit), vertical merge MUST
        // NOT fire because merging `TL+TR` with `BL` would overlap `BR`.
        let mut g = [
            Some(tr(0, 118, 128, 10)),
            Some(tr(128, 118, 128, 10)),
            Some(tr(0, 128, 128, 10)),
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(tr(0, 118, 256, 10)),
                None,
                Some(tr(0, 128, 128, 10)),
                None,
            ]
        );
        let rects: Vec<TileRect> = g
            .into_iter()
            .flatten()
            .chain([tr(128, 128, 128, 128)])
            .collect();
        validate_tile_rects(&rects, 256, 256).unwrap();

        // Non-touching gap (`x=124` vs `x=130`) must NOT merge.
        let mut g = [
            Some(tr(0, 10, 124, 12)),
            Some(tr(130, 10, 126, 12)),
            None,
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(tr(0, 10, 124, 12)),
                Some(tr(130, 10, 126, 12)),
                None,
                None,
            ]
        );

        // Excess bounding-box slack (`128x2 + 128x40` -> union `256x40 = 10240`,
        // sum = `5376`, slack = `4864` >> cap) must NOT merge.
        let mut g = [
            Some(tr(0, 10, 128, 2)),
            Some(tr(128, 10, 128, 40)),
            None,
            None,
        ];
        merge_cropped_quad_slots(&mut g);
        assert_eq!(
            g,
            [
                Some(tr(0, 10, 128, 2)),
                Some(tr(128, 10, 128, 40)),
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
                        let mut shadow = Vec::new();
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
                                &shadow, prev,
                                "shadow mismatch across tiers ({tier:?}, {format:?})"
                            );
                        } else {
                            ref_shadow = Some(shadow);
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

            let mut shadow = Vec::new();
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
            let mut shadow = Vec::new();
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
                let mut shadow = Vec::new();
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
                unsafe { ClientTileCache::prepare_slot_target_uninit(&slots, 3, rect) };
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

        {
            let slots = cli.sync_slots();
            // SAFETY: Single-threaded exclusive ownership of slot 3.
            let _slot_frame =
                unsafe { ClientTileCache::prepare_slot_target_uninit(&slots, 3, rect) };
            // Simulate decode error before `commit_slot`.
            unsafe { ClientTileCache::invalidate_slot(&slots, 3) };
        }
        cli.recompute_and_verify_bytes().unwrap();
        assert_eq!(cli.occupied_slots(), 0);
        assert_eq!(cli.used_bytes(), 0);

        // 3. `select_and_filter_row` with reusable candidate scratch across
        //    short cropped (`tw < 256`), full `256`, and wide 1D strip (`tw > 256`).
        let mut cand_sub = vec![0u8; TILE_W * 3];
        let mut cand_up = vec![0u8; TILE_W * 3];
        let mut cand_su = vec![0u8; TILE_W * 3];
        for tw in [1usize, 7, 16, 128, 256, 300] {
            let row_b = tw * 3;
            ensure_len(&mut cand_sub, row_b);
            ensure_len(&mut cand_up, row_b);
            ensure_len(&mut cand_su, row_b);
            let row0 = test_vec(row_b);
            let row1: Vec<u8> = row0
                .iter()
                .enumerate()
                .map(|(i, &b)| b.wrapping_add(((i % 7) as u8).wrapping_sub(3)))
                .collect();
            let mut filt0 = vec![0u8; row_b];
            let mut filt1 = vec![0u8; row_b];
            let p0 = select_and_filter_row(
                ScalarKernel,
                0,
                &row0,
                &[],
                &mut filt0,
                &mut cand_sub[..row_b],
                &mut [],
                &mut [],
            );
            assert!(p0 < 2, "row 0 predictor must be None or Sub, got {p0}");
            let p1 = select_and_filter_row(
                ScalarKernel,
                1,
                &row1,
                &row0,
                &mut filt1,
                &mut cand_sub[..row_b],
                &mut cand_up[..row_b],
                &mut cand_su[..row_b],
            );

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
                        let mut shadow = Vec::new();
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
                        assert_shadow_exact(&shadow, &frame_a, width, height, stride, "cold_fill");
                        assert_canvas_matches_frame(&canvas, &frame_a, meta, 0xCC);
                        assert!(srv.occupied_slots() > 0);

                        // (b) `tile_cache = Some` warm 2-pass with oversized sentinel shadow
                        //     (verifies shrink + zeroed padding with no stale `0xDD` bytes).
                        let mut oversized_shadow = vec![0xDDu8; byte_len * 3];
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
                            &oversized_shadow,
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
                        let mut shadow_cold_drop = Vec::new();
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
                            &shadow_cold_drop,
                            &frame_a,
                            width,
                            height,
                            stride,
                            "cold_fill_drop",
                        );

                        // (d) `tile_cache = None` normal.
                        let mut shadow_nc = Vec::new();
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
                        assert_shadow_exact(&shadow_nc, &frame_a, width, height, stride, "nocache");
                        assert_canvas_matches_frame(&canvas, &frame_a, meta, 0xCC);

                        // (e) `tile_cache = None` early receiver drop (`disconnected = true`).
                        let mut shadow_nc_drop = Vec::new();
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
                            &shadow_nc_drop,
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
            let mut shadow = Vec::new();
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
}
