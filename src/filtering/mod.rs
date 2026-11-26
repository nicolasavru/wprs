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

//! 2D tiled pixel filtering, incremental sub-tile diffing, and lossless
//! compression/decompression.
//!
//! Pixel buffers are partitioned into `256x256` tiles (with a full-width 1D
//! strip fallback for thin `< 8`-row partial bands so 1-row updates avoid
//! tile-header fragmentation). On incremental commits against a per-surface
//! shadow buffer, a parallel linear horizontal-band SIMD pass skips unchanged
//! `128x128` quadrants (coalescing all-dirty quadrants back into a `256x256`
//! tile and omitting empty commits completely). Each dirty tile is encoded on
//! a worker thread inside L1/L2 scratch memory as:
//! - [`TilePayload::Solid`] for 1-color tiles,
//! - [`TilePayload::Palette`] (1/2/4/8 bpp + Zstd) for `<= 256` unique colors,
//! - [`TilePayload::AosZstd`] using fused 3-byte `SubGreen(B-G, G, R-G)` AoS,
//!   per-row adaptive `{None, Sub, Up, SubUp}` prediction (2 bits/row), a
//!   single Zstd level-1 (`MinMatch(7)`) stream, and an optional separate alpha
//!   plane omitted completely on `Xrgb8888` and opaque `Argb8888` tiles.
//!
//! Architecture-specific backends:
//! - `x86`: native AVX2 (256-bit) and SSSE3 (128-bit) with scalar fallback on
//!   baseline SSE2 / Miri
//! - `neon`: AArch64 NEON structure load/store (`vld4q`/`vst3q`, `vld3q`/`vst4q`)

#[cfg(target_arch = "x86_64")]
#[path = "x86.rs"]
mod arch;

#[cfg(target_arch = "aarch64")]
#[path = "neon.rs"]
mod arch;

pub(crate) mod palette;

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::slice;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use zstd::bulk::Decompressor;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::palette::decode_palette_row;
use crate::filtering::palette::prepare_palette_lut;
use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::framing::AlphaChannel;
use crate::serialization::framing::CompressedTile;
use crate::serialization::framing::CompressedTileStream;
use crate::serialization::framing::CompressedTiles;
use crate::serialization::framing::EncodedBytes;
use crate::serialization::framing::MAX_FRAME_TILES;
use crate::serialization::framing::RawBufferHeader;
use crate::serialization::framing::RowBand;
use crate::serialization::framing::SUBTILE_H;
use crate::serialization::framing::SUBTILE_W;
use crate::serialization::framing::StreamingCompressedTiles;
use crate::serialization::framing::TILE_H;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TILE_W;
use crate::serialization::framing::TilePayload;
use crate::serialization::framing::TileRect;
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

/// Architecture-specific row kernels operating on exact-length slices (no slack
/// bytes required).
pub(crate) trait RowKernel {
    const STREAM_ALIGN: usize;

    fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        dst_bgr: &mut [u8],
        dst_alpha: &mut [u8],
    );

    fn sad_signed_bytes(data: &[u8]) -> u64;

    fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64;

    fn row_up_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64;

    fn row_subup3_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64;

    fn row_sub3_inverse(src: &[u8], dst: &mut [u8]);

    fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]);

    fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]);

    /// # Safety
    /// When `STREAM` is `true`, `dst.as_mut_ptr()` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [Vec4u8],
    );

    /// # Safety
    /// When `STREAM` is `true`, `dst.as_mut_ptr()` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8);

    fn is_constant_plane(data: &[u8]) -> Option<u8>;

    fn row_sub1_forward(src: &[u8], dst: &mut [u8]);

    fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]);

    fn row_sub1_inverse(src: &[u8], dst: &mut [u8]);

    /// Compares `src` (via [`BufferPointer`], never creating `&[u8]` over
    /// `wl_shm`) against `shadow` pixel-by-pixel (`BGRA` when `HAS_ALPHA`, or
    /// ignoring byte 3 when `!HAS_ALPHA`). Returns `false` without writing
    /// `shadow` when all pixels match; otherwise updates the differing and
    /// subsequent pixels in `shadow` from `src` and returns `true`.
    fn diff_and_update_row<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> bool;

    fn fence_stream();
}

/// Portable scalar reference implementation of [`RowKernel`], also used as the
/// baseline `x86-64` (SSE2-only) and Miri backend.
#[cfg_attr(
    not(any(test, miri, all(target_arch = "x86_64", not(target_feature = "ssse3")))),
    allow(dead_code)
)]
pub(crate) struct ScalarKernel;

impl RowKernel for ScalarKernel {
    const STREAM_ALIGN: usize = 1;

    fn bgra_to_subgreen_bgr3<const HAS_ALPHA: bool>(
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

    fn sad_signed_bytes(data: &[u8]) -> u64 {
        data.iter()
            .map(|&b| u64::from((b as i8).unsigned_abs()))
            .sum()
    }

    fn row_sub3_forward_sad(src: &[u8], dst: &mut [u8]) -> u64 {
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

    fn row_up_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
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

    fn row_subup3_forward_sad(cur: &[u8], prev: &[u8], dst: &mut [u8]) -> u64 {
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

    fn row_sub3_inverse(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            let p = if j >= 3 { dst[j - 3] } else { 0 };
            dst[j] = src[j].wrapping_add(p);
        }
    }

    fn row_up_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            dst[j] = src[j].wrapping_add(prev[j]);
        }
    }

    fn row_subup3_inverse(src: &[u8], prev: &[u8], dst: &mut [u8]) {
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

    unsafe fn subgreen_bgr3_to_bgra<const STREAM: bool, const CONST_ALPHA: bool>(
        src_bgr: &[u8],
        alpha_row: &[u8],
        const_alpha: u8,
        dst: &mut [Vec4u8],
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
            *out = Vec4u8(
                src_bgr[s].wrapping_add(g),
                g,
                src_bgr[s + 2].wrapping_add(g),
                a,
            );
        }
    }

    unsafe fn splat_row<const STREAM: bool>(dst: &mut [Vec4u8], pixel: Vec4u8) {
        dst.fill(pixel);
    }

    fn is_constant_plane(data: &[u8]) -> Option<u8> {
        let first = *data.first()?;
        data[1..].iter().all(|&b| b == first).then_some(first)
    }

    fn row_sub1_forward(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let mut prev = 0u8;
        for j in 0..len {
            let cur = src[j];
            dst[j] = cur.wrapping_sub(prev);
            prev = cur;
        }
    }

    fn row_up_forward(cur: &[u8], prev: &[u8], dst: &mut [u8]) {
        let len = cur.len();
        debug_assert_eq!(prev.len(), len);
        debug_assert_eq!(dst.len(), len);
        for j in 0..len {
            dst[j] = cur[j].wrapping_sub(prev[j]);
        }
    }

    fn row_sub1_inverse(src: &[u8], dst: &mut [u8]) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        let mut acc = 0u8;
        for j in 0..len {
            acc = acc.wrapping_add(src[j]);
            dst[j] = acc;
        }
    }

    fn diff_and_update_row<const HAS_ALPHA: bool>(
        src: BufferPointer<Vec4u8>,
        shadow: &mut [Vec4u8],
    ) -> bool {
        let n = src.len();
        debug_assert_eq!(shadow.len(), n);
        let src_p = src.ptr();
        for j in 0..n {
            // SAFETY: `j < n == src.len()`.
            let s = unsafe { src_p.add(j).read() };
            let d = shadow[j];
            let differs = if HAS_ALPHA {
                s != d
            } else {
                s.0 != d.0 || s.1 != d.1 || s.2 != d.2
            };
            if differs {
                shadow[j] = s;
                if j + 1 < n {
                    src.split_at(j + 1)
                        .1
                        .copy_to_nonoverlapping(&mut shadow[j + 1..]);
                }
                return true;
            }
        }
        false
    }

    fn fence_stream() {}
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
#[inline]
fn select_and_filter_row<K: RowKernel>(
    ry: usize,
    cur_raw: &[u8],
    prev_raw: &[u8],
    filt_row: &mut [u8],
    cand_row: &mut [u8],
) -> u8 {
    let row_b = cur_raw.len();
    let mut best_p = 0u8;
    let mut best_cost = K::sad_signed_bytes(cur_raw);

    if row_b <= TILE_W * 3 {
        // Common 2D tile path (`row_b <= 768 B`): evaluate candidates in L1d
        // stack buffers and copy only the single winner into `filt_row`.
        if best_cost == 0 {
            filt_row.copy_from_slice(cur_raw);
            return 0;
        }
        let mut cand_sub = [0u8; TILE_W * 3];
        let mut cand_up = [0u8; TILE_W * 3];
        let mut cand_su = [0u8; TILE_W * 3];

        let cost_sub = K::row_sub3_forward_sad(cur_raw, &mut cand_sub[..row_b]);
        if cost_sub < best_cost {
            best_cost = cost_sub;
            best_p = 1;
        }
        if ry > 0 && best_cost > 0 {
            let cost_up = K::row_up_forward_sad(cur_raw, prev_raw, &mut cand_up[..row_b]);
            if cost_up < best_cost {
                best_cost = cost_up;
                best_p = 2;
            }
            if best_cost > 0 {
                let cost_su = K::row_subup3_forward_sad(cur_raw, prev_raw, &mut cand_su[..row_b]);
                if cost_su < best_cost {
                    best_p = 3;
                }
            }
        }
        let chosen = match best_p {
            0 => cur_raw,
            1 => &cand_sub[..row_b],
            2 => &cand_up[..row_b],
            _ => &cand_su[..row_b],
        };
        filt_row.copy_from_slice(chosen);
        return best_p;
    }

    // Wide `< 8`-row 1D strip path (`row_b > 768 B`): reuse worker heap scratch.
    filt_row.copy_from_slice(cur_raw);
    if best_cost == 0 {
        return 0;
    }
    let cost_sub = K::row_sub3_forward_sad(cur_raw, &mut cand_row[..row_b]);
    if cost_sub < best_cost {
        best_cost = cost_sub;
        best_p = 1;
        filt_row.copy_from_slice(&cand_row[..row_b]);
    }
    if ry > 0 && best_cost > 0 {
        let cost_up = K::row_up_forward_sad(cur_raw, prev_raw, &mut cand_row[..row_b]);
        if cost_up < best_cost {
            best_cost = cost_up;
            best_p = 2;
            filt_row.copy_from_slice(&cand_row[..row_b]);
        }
        if best_cost > 0 {
            let cost_su = K::row_subup3_forward_sad(cur_raw, prev_raw, &mut cand_row[..row_b]);
            if cost_su < best_cost {
                best_p = 3;
                filt_row.copy_from_slice(&cand_row[..row_b]);
            }
        }
    }
    best_p
}

fn filter_tile<K: RowKernel, const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
    state: &mut CompressorWorkerState,
) -> TilePayload {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;
    debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

    if let Some((n_colors, packed_len)) = state
        .palette
        .try_encode_2d::<HAS_ALPHA>(aos, stride_px, rect)
    {
        if n_colors == 1 {
            let px = state.palette.palette[0].to_le_bytes();
            return TilePayload::Solid(px);
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
        return TilePayload::Palette {
            has_alpha: pal_has_alpha,
            palette: palette_bytes,
            data,
        };
    }

    // Direct 3-byte `SubGreen(BGR)` AoS + per-row adaptive `{None, Sub, Up, SubUp}`
    // streamed through two L1d-resident row buffers (`raw_row_a`, `raw_row_b`).
    let row_b = tw * 3;
    let total_b = row_b * th;
    ensure_len(&mut state.raw_row_a, row_b);
    if th > 1 {
        ensure_len(&mut state.raw_row_b, row_b);
    }
    if row_b > TILE_W * 3 {
        ensure_len(&mut state.cand_row, row_b);
    }
    if HAS_ALPHA {
        ensure_len(&mut state.raw_alpha, TILE_PIXELS);
    }

    let mut predictors = vec![0u8; th.div_ceil(4)];
    let mut use_a = true;
    for ry in 0..th {
        let row_start = (y0 + ry) * stride_px + x0;
        let (_, after_start) = aos.split_at(row_start);
        let (row_ptr, _) = after_start.split_at(tw);

        let (cur_raw, prev_raw): (&mut [u8], &[u8]) = if use_a {
            (
                &mut state.raw_row_a[..row_b],
                &state.raw_row_b[..if ry > 0 { row_b } else { 0 }],
            )
        } else {
            (&mut state.raw_row_b[..row_b], &state.raw_row_a[..row_b])
        };

        if HAS_ALPHA {
            K::bgra_to_subgreen_bgr3::<true>(
                row_ptr,
                cur_raw,
                &mut state.raw_alpha[ry * tw..(ry + 1) * tw],
            );
        } else {
            K::bgra_to_subgreen_bgr3::<false>(row_ptr, cur_raw, &mut []);
        }

        let filt_row = &mut state.filt_bgr[ry * row_b..(ry + 1) * row_b];
        let best_p =
            select_and_filter_row::<K>(ry, cur_raw, prev_raw, filt_row, &mut state.cand_row);
        predictors[ry >> 2] |= best_p << ((ry & 3) * 2);
        use_a = !use_a;
    }

    let color = compress_bytes(
        &mut state.compressor,
        &mut state.zstd_scratch,
        &state.filt_bgr[..total_b],
    );

    let alpha = if !HAS_ALPHA {
        AlphaChannel::Omitted
    } else {
        let raw_alpha = &state.raw_alpha[..n_px];
        match K::is_constant_plane(raw_alpha) {
            Some(0xFF) => AlphaChannel::Omitted,
            Some(a) => AlphaChannel::Constant(a),
            None => {
                ensure_len(&mut state.filt_alpha, TILE_PIXELS);
                K::row_sub1_forward(&raw_alpha[..tw], &mut state.filt_alpha[..tw]);
                for ry in 1..th {
                    K::row_up_forward(
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
    };

    TilePayload::AosZstd {
        predictors,
        color,
        alpha,
    }
}

// --- fused 2D tile decompress + unfilter ---

/// Raw pointer handle to the destination frame buffer preserving pointer
/// provenance for disjoint per-row mutable borrows across parallel workers.
struct FrameRows {
    base: *mut Vec4u8,
    stride_px: usize,
}

// SAFETY: `unfilter_tiles_parallel` validates every accepted `TileRect` is
// strictly in-bounds and pairwise disjoint before any worker calls `row_mut`,
// so concurrent workers never alias overlapping pixels.
unsafe impl Send for FrameRows {}
unsafe impl Sync for FrameRows {}

impl FrameRows {
    /// # Safety
    /// Caller must guarantee `(y, x..x + w)` lies within the validated frame
    /// bounds.
    #[inline]
    unsafe fn row_ptr(&self, y: usize, x: usize) -> *mut Vec4u8 {
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

/// # Safety
/// `rect` must already be validated in-bounds within `frame` and disjoint from
/// all other tiles being decoded concurrently.
unsafe fn unfilter_tile<K: RowKernel, const HAS_ALPHA: bool>(
    rect: TileRect,
    payload: &TilePayload,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    let n_px = tw * th;
    debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

    match payload {
        TilePayload::Solid(px) => {
            if !HAS_ALPHA && px[3] != 0xFF {
                bail!("translucent Solid tile on Xrgb8888 buffer");
            }
            let pixel = Vec4u8(px[0], px[1], px[2], px[3]);
            let mut streamed = false;
            for ry in 0..th {
                // SAFETY: Caller guarantees `tile.rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                if K::STREAM_ALIGN > 1
                    && tw.is_multiple_of(16)
                    && (dst_row.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN)
                {
                    // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                    unsafe { K::splat_row::<true>(dst_row, pixel) };
                    streamed = true;
                } else {
                    // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                    unsafe { K::splat_row::<false>(dst_row, pixel) };
                }
            }
            Ok(streamed)
        },
        TilePayload::Palette {
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
                // SAFETY: Caller guarantees `tile.rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                decode_palette_row(&lut, dec_n, packed, &mut idx_pos, dst_row).location(loc!())?;
            }
            Ok(false)
        },
        TilePayload::AosZstd {
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

            let (const_alpha, alpha_filt_slice): (Option<u8>, Option<&[u8]>) = match alpha {
                AlphaChannel::Omitted => (Some(0xFF), None),
                AlphaChannel::Constant(a) => {
                    if !HAS_ALPHA {
                        bail!("AlphaChannel::Constant on Xrgb8888 buffer");
                    }
                    (Some(*a), None)
                },
                AlphaChannel::Uncompressed(bytes) => {
                    if !HAS_ALPHA {
                        bail!("AlphaChannel::Uncompressed on Xrgb8888 buffer");
                    }
                    if bytes.len() != n_px {
                        bail!("uncompressed alpha len {} != expected {n_px}", bytes.len());
                    }
                    (None, Some(bytes.as_slice()))
                },
                AlphaChannel::Zstd(bytes) => {
                    if !HAS_ALPHA {
                        bail!("AlphaChannel::Zstd on Xrgb8888 buffer");
                    }
                    ensure_len(&mut state.alpha_filt, TILE_PIXELS);
                    let written = state
                        .decompressor
                        .decompress_to_buffer(bytes, &mut state.alpha_filt[..n_px])
                        .location(loc!())?;
                    if written != n_px {
                        bail!("decompressed alpha len {written} != expected {n_px}");
                    }
                    (None, Some(&state.alpha_filt[..n_px]))
                },
            };

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
                        K::row_sub3_inverse(filt_row, cur_dec);
                        cur_dec
                    },
                    2 => {
                        K::row_up_inverse(filt_row, prev_dec, cur_dec);
                        cur_dec
                    },
                    _ => {
                        K::row_subup3_inverse(filt_row, prev_dec, cur_dec);
                        cur_dec
                    },
                };

                // SAFETY: Caller guarantees `tile.rect` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                let use_stream = K::STREAM_ALIGN > 1
                    && tw.is_multiple_of(16)
                    && (dst_row.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN);

                if let Some(a) = const_alpha {
                    if use_stream {
                        // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                        unsafe {
                            K::subgreen_bgr3_to_bgra::<true, true>(decoded_bgr, &[], a, dst_row)
                        };
                        streamed = true;
                    } else {
                        // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                        unsafe {
                            K::subgreen_bgr3_to_bgra::<false, true>(decoded_bgr, &[], a, dst_row);
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
                        K::row_sub1_inverse(&a_filt[..tw], a_cur);
                    } else {
                        K::row_up_inverse(&a_filt[ry * tw..(ry + 1) * tw], a_prev, a_cur);
                    }
                    if use_stream {
                        // SAFETY: Verified `dst_row` alignment matches `K::STREAM_ALIGN`.
                        unsafe {
                            K::subgreen_bgr3_to_bgra::<true, false>(decoded_bgr, a_cur, 0, dst_row);
                        }
                        streamed = true;
                    } else {
                        // SAFETY: `STREAM = false` imposes no extra alignment requirement.
                        unsafe {
                            K::subgreen_bgr3_to_bgra::<false, false>(
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

/// Splits a 2D base tile (`w <= TILE_W`, `h <= TILE_H`) into 1..=4 clipped
/// `128x128` quadrants in raster order.
#[inline]
fn tile_quadrants(rect: TileRect) -> ([TileRect; 4], usize) {
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

/// Copies `rect` from `src` (`wl_shm` via [`BufferPointer`]) into `shadow`.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
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
    for ry in 0..rect.h as usize {
        let row_off = (y0 + ry) * stride + x_bytes;
        let (_, after_start) = src.split_at(row_off);
        let (src_row, _) = after_start.split_at(row_bytes);
        // SAFETY: Caller guarantees `rect` is in-bounds and disjoint across workers.
        let dst_row =
            unsafe { slice::from_raw_parts_mut(shadow.row_ptr(y0 + ry, x_bytes), row_bytes) };
        src_row.copy_to_nonoverlapping(dst_row);
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
enum BaseScanEntry {
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

/// Scans a `< 8`-row 1D strip linearly against `shadow`, updating `shadow`
/// from the first differing pixel onward.
///
/// # Safety
/// `rect` must be strictly in-bounds within `shadow` and disjoint from all
/// regions concurrently accessed by other threads.
unsafe fn diff_and_update_strip<const HAS_ALPHA: bool>(
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
        if arch::Kernel::diff_and_update_row::<HAS_ALPHA>(src_row, dst_row) {
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
/// `n_qx` horizontal quadrants, updating `shadow` for dirty quadrants and
/// falling back to bulk contiguous `memcpy` once all quadrants in the band are
/// dirty.
///
/// # Safety
/// Rows `[y0, y1)` must be strictly in-bounds within `shadow` and disjoint from
/// all rows concurrently accessed by other threads.
unsafe fn diff_and_update_row_band_2d<const HAS_ALPHA: bool>(
    data: BufferPointer<u8>,
    aos: BufferPointer<Vec4u8>,
    shadow: &ShadowRows,
    width: usize,
    stride: usize,
    y0: usize,
    y1: usize,
    dirty_slots: &[AtomicBool],
) {
    let stride_px = stride / 4;
    let n_qx = dirty_slots.len();
    let mut qx_dirty = vec![false; n_qx];
    let mut dirty_count = 0usize;

    for y in y0..y1 {
        if dirty_count == n_qx {
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

        let (_, row_after) = aos.split_at(y * stride_px);
        let (row_aos, _) = row_after.split_at(width);
        // SAFETY: `[y0, y1)` is strictly disjoint across workers and `Vec4u8` has alignment 1.
        let dst_row =
            unsafe { slice::from_raw_parts_mut(shadow.row_ptr(y, 0).cast::<Vec4u8>(), width) };

        let mut x = 0usize;
        for is_dirty in &mut qx_dirty {
            let cw = (width - x).min(SUBTILE_W);
            let (_, after_x) = row_aos.split_at(x);
            let (src_chunk, _) = after_x.split_at(cw);
            let dst_chunk = &mut dst_row[x..x + cw];
            if *is_dirty {
                src_chunk.copy_to_nonoverlapping(dst_chunk);
            } else if arch::Kernel::diff_and_update_row::<HAS_ALPHA>(src_chunk, dst_chunk) {
                *is_dirty = true;
                dirty_count += 1;
            }
            x += cw;
        }
    }

    for (slot, &dirty) in dirty_slots.iter().zip(&qx_dirty) {
        if dirty {
            slot.store(true, Ordering::Relaxed);
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
    shadow_out: Option<&mut Vec<u8>>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) {
    assert!(data.len().is_multiple_of(4));
    let total_pixels = metadata
        .validate()
        .expect("server passed invalid BufferMetadata");
    let byte_len = total_pixels * 4;
    assert!(data.len() >= byte_len);

    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let tile_rects: Vec<TileRect> = plan_incremental_scan(&metadata, None)
        .expect("full-frame tile plan cannot fail")
        .1
        .into_iter()
        .map(BaseScanEntry::rect)
        .collect();
    let n_tiles = tile_rects.len();
    let rects_for_workers = tile_rects.clone();

    let shadow_rows = shadow_out.map(|shadow| {
        if shadow.len() < byte_len {
            shadow.resize(byte_len, 0);
        } else {
            shadow.truncate(byte_len);
            if shadow.capacity() > byte_len.saturating_mul(2) {
                shadow.shrink_to(byte_len);
            }
        }
        ShadowRows {
            base: shadow.as_mut_ptr(),
            stride,
        }
    });

    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
    let header = RawBufferHeader {
        client,
        surface,
        metadata,
        incremental: false,
        tile_rects,
    };
    on_stream_ready(StreamingCompressedTiles {
        header,
        tiles: tiles_rx,
    });

    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };
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
                unsafe { copy_rect_to_shadow(data, shadow, stride, rect) };
                if disconnected {
                    continue;
                }
                shadow_ptr = shadow.base.cast_const().cast::<Vec4u8>();
                // SAFETY: `shadow.base` is valid for `total_pixels` `Vec4u8`s and
                // this worker holds exclusive access to `rect`'s pixels after the
                // `&mut [u8]` row writes in `copy_rect_to_shadow` ended.
                unsafe { BufferPointer::new(&shadow_ptr, total_pixels) }
            } else {
                aos
            };
            let payload = if has_alpha {
                filter_tile::<arch::Kernel, true>(encode_aos, stride_px, rect, state)
            } else {
                filter_tile::<arch::Kernel, false>(encode_aos, stride_px, rect, state)
            };

            if tiles_tx
                .send(CompressedTile {
                    tile_idx: tile_idx as u32,
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
}

/// Diffs `data` against `shadow` across `bands` (or the full frame when `bands`
/// is `None`), updates `shadow` for every changed `128x128` quadrant or
/// `< 8`-row 1D strip, and streams only the dirty tiles (`incremental = true`)
/// from `shadow`.
///
/// Returns the transmitted dirty [`TileRect`]s in deterministic raster order.
/// When no pixels changed, returns an empty `Vec` without invoking
/// `on_stream_ready`.
#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress_incremental_streaming(
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    bands: Option<&[RowBand]>,
    shadow: &mut [u8],
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) -> Vec<TileRect> {
    assert!(data.len().is_multiple_of(4));
    let total_pixels = metadata
        .validate()
        .expect("server passed invalid BufferMetadata");
    let byte_len = total_pixels * 4;
    assert!(data.len() >= byte_len);
    assert!(shadow.len() >= byte_len);

    let (sub_bands, base_entries, total_slots) =
        plan_incremental_scan(&metadata, bands).expect("server generated invalid row bands");
    if sub_bands.is_empty() {
        return Vec::new();
    }

    let width = metadata.width as usize;
    let stride = metadata.stride as usize;
    let stride_px = stride / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let n_qx = width.div_ceil(SUBTILE_W);
    let n_sub = sub_bands.len();
    let dirty_flags: Vec<AtomicBool> = (0..total_slots).map(|_| AtomicBool::new(false)).collect();
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };

    // Phase A: parallel linear horizontal-band dirty scan + shadow update.
    {
        let shadow_rows = ShadowRows {
            base: shadow.as_mut_ptr(),
            stride,
        };
        let next_sub = AtomicUsize::new(0);
        compressor.pool.execute(n_sub, |_| {
            loop {
                let idx = next_sub.fetch_add(1, Ordering::Relaxed);
                if idx >= n_sub {
                    break;
                }
                match sub_bands[idx] {
                    ScanSubBand::Strip1D { rect, slot_idx } => {
                        let dirty = if has_alpha {
                            // SAFETY: `sub_bands` cover disjoint regions in `shadow`.
                            unsafe {
                                diff_and_update_strip::<true>(aos, &shadow_rows, stride_px, rect)
                            }
                        } else {
                            // SAFETY: `sub_bands` cover disjoint regions in `shadow`.
                            unsafe {
                                diff_and_update_strip::<false>(aos, &shadow_rows, stride_px, rect)
                            }
                        };
                        if dirty {
                            dirty_flags[slot_idx].store(true, Ordering::Relaxed);
                        }
                    },
                    ScanSubBand::RowBand2D { y0, y1, slot_base } => {
                        let slots = &dirty_flags[slot_base..slot_base + n_qx];
                        if has_alpha {
                            // SAFETY: `sub_bands` cover disjoint row ranges in `shadow`.
                            unsafe {
                                diff_and_update_row_band_2d::<true>(
                                    data,
                                    aos,
                                    &shadow_rows,
                                    width,
                                    stride,
                                    y0 as usize,
                                    y1 as usize,
                                    slots,
                                );
                            }
                        } else {
                            // SAFETY: `sub_bands` cover disjoint row ranges in `shadow`.
                            unsafe {
                                diff_and_update_row_band_2d::<false>(
                                    data,
                                    aos,
                                    &shadow_rows,
                                    width,
                                    stride,
                                    y0 as usize,
                                    y1 as usize,
                                    slots,
                                );
                            }
                        }
                    },
                }
            }
        });
    }

    let mut dirty_tiles = Vec::new();
    for entry in &base_entries {
        match *entry {
            BaseScanEntry::Strip1D { rect, slot_idx } => {
                if dirty_flags[slot_idx].load(Ordering::Relaxed) {
                    dirty_tiles.push(rect);
                }
            },
            BaseScanEntry::Tile2D {
                rect,
                quad_slots,
                n_quads,
            } => {
                let mut mask = 0u8;
                for (q, &slot) in quad_slots[..n_quads].iter().enumerate() {
                    if dirty_flags[slot].load(Ordering::Relaxed) {
                        mask |= 1u8 << q;
                    }
                }
                if mask == 0 {
                    continue;
                }
                let all_mask = (1u8 << n_quads) - 1;
                if mask == all_mask {
                    dirty_tiles.push(rect);
                } else {
                    let (quads, _) = tile_quadrants(rect);
                    for (q, &quad) in quads[..n_quads].iter().enumerate() {
                        if (mask & (1u8 << q)) != 0 {
                            dirty_tiles.push(quad);
                        }
                    }
                }
            },
        }
    }

    if dirty_tiles.is_empty() {
        return dirty_tiles;
    }

    // Phase B: stream dirty tiles encoded directly from the updated `shadow`
    // buffer so transmitted pixels are guaranteed consistent with `shadow`.
    let n_tiles = dirty_tiles.len();
    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
    let header = RawBufferHeader {
        client,
        surface,
        metadata,
        incremental: true,
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
    let next_tile = AtomicUsize::new(0);
    compressor.pool.execute(n_tiles, |state| {
        loop {
            let tile_idx = next_tile.fetch_add(1, Ordering::Relaxed);
            if tile_idx >= n_tiles {
                break;
            }
            let rect = dirty_tiles[tile_idx];
            let payload = if has_alpha {
                filter_tile::<arch::Kernel, true>(shadow_aos, stride_px, rect, state)
            } else {
                filter_tile::<arch::Kernel, false>(shadow_aos, stride_px, rect, state)
            };

            if tiles_tx
                .send(CompressedTile {
                    tile_idx: tile_idx as u32,
                    payload,
                })
                .is_err()
            {
                break;
            }
        }
    });

    dirty_tiles
}

#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress(
    data: BufferPointer<u8>,
    metadata: BufferMetadata,
    compressor: &mut ShardingCompressor,
) -> CompressedTiles {
    let mut streaming_opt = None;
    filter_and_compress_full_streaming(
        data,
        ClientId(0),
        WlSurfaceId(0),
        metadata,
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

/// Decodes a validated set of pairwise-disjoint `tile_rects` in parallel.
///
/// # Safety
/// Caller must have validated `metadata` (`byte_len = total_pixels * 4`) and
/// verified every rect in `tile_rects` via [`validate_tile_rects`].
unsafe fn unfilter_tiles_parallel<'a>(
    decompressor: &mut ShardingDecompressor,
    metadata: BufferMetadata,
    byte_len: usize,
    tile_rects: &[TileRect],
    output_buf: &mut [u8],
    next_tile: impl Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
) -> Result<()> {
    if output_buf.len() < byte_len {
        bail!("output_buf len {} < {byte_len}", output_buf.len());
    }

    let n_tiles = tile_rects.len();
    if n_tiles == 0 {
        if let Some(item) = next_tile() {
            item?;
            bail!("received unexpected tile for 0-tile frame");
        }
        return Ok(());
    }

    let stride_px = (metadata.stride as usize) / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let aos: &mut [Vec4u8] = bytemuck::cast_slice_mut(&mut output_buf[..byte_len]);
    let frame = FrameRows {
        base: aos.as_mut_ptr(),
        stride_px,
    };

    // Every `tile_rects` entry was validated in-bounds and pairwise disjoint
    // upfront; an atomic slot per index rejects duplicate or out-of-range
    // `tile_idx` without mutex contention before any worker writes `frame`.
    let claimed: Vec<AtomicBool> = (0..n_tiles).map(|_| AtomicBool::new(false)).collect();
    let completed = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    decompressor.pool.execute(n_tiles, |state| {
        let mut streamed_any = false;
        while let Some(item) = next_tile() {
            let res = (|| -> Result<bool> {
                let tile = item?;
                let idx = tile.tile_idx as usize;
                let Some(slot) = claimed.get(idx) else {
                    bail!("tile_idx {idx} out of range (n_tiles={n_tiles})");
                };
                if slot.swap(true, Ordering::AcqRel) {
                    bail!("duplicate tile_idx {idx}");
                }
                let rect = tile_rects[idx];
                // SAFETY: Caller validated `tile_rects` in-bounds and pairwise
                // disjoint, and `claimed[idx]` guarantees exclusive ownership
                // of `rect` for this worker.
                unsafe {
                    if has_alpha {
                        unfilter_tile::<arch::Kernel, true>(rect, &tile.payload, &frame, state)
                    } else {
                        unfilter_tile::<arch::Kernel, false>(rect, &tile.payload, &frame, state)
                    }
                }
            })();

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
            arch::Kernel::fence_stream();
        }
    });

    if let Some(err) = first_err.into_inner().unwrap() {
        return Err(err);
    }
    let done = completed.load(Ordering::Relaxed);
    if done != n_tiles {
        bail!("received {done} tiles, expected {n_tiles}");
    }
    Ok(())
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
) -> Result<()> {
    let metadata = validated.stream.header.metadata;
    let tile_rects = &validated.stream.header.tile_rects;
    let tiles_rx = &validated.stream.tiles;
    // SAFETY: `ValidatedTileStream::new` validated `metadata`, `byte_len`, and
    // `validate_tile_rects(tile_rects, width, height)`.
    unsafe {
        unfilter_tiles_parallel(
            decompressor,
            metadata,
            validated.byte_len,
            tile_rects,
            output_buf,
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
            &tiles.tile_rects,
            output_buf,
            || {
                let i = next_idx.fetch_add(1, Ordering::Relaxed);
                (i < n_tiles).then(|| Ok(Cow::Borrowed(&tiles.tiles[i])))
            },
        )
    }
}

#[cfg(test)]
mod tests {
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
        let mut buf = vec![0xAAu8; expected.len() + 36];
        let aligned_off = (32 - (buf.as_ptr() as usize & 31)) & 31;
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

        let tiles = filter_and_compress(buf_ptr, metadata, compressor);
        let mut expected = data.to_vec();
        if format == BufferFormat::Xrgb8888 {
            for px in expected.as_chunks_mut::<4>().0 {
                px[3] = 0xFF;
            }
        }
        assert_decodes_at_both_alignments(&tiles, &expected, decompressor);
        tiles
    }

    /// Compares `arch::Kernel` bit-for-bit against [`ScalarKernel`] across all
    /// edge widths (`1..=65`, `255`, `256`, `300`) for every row primitive.
    #[test]
    fn test_arch_kernel_matches_scalar_reference() {
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

            ScalarKernel::bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_s, &mut alp0_s);
            arch::Kernel::bgra_to_subgreen_bgr3::<true>(bp0, &mut bgr0_a, &mut alp0_a);
            assert_eq!(bgr0_a, bgr0_s, "bgra_to_subgreen_bgr3<true> tw={tw}");
            assert_eq!(alp0_a, alp0_s, "alpha extract tw={tw}");

            let mut bgr0_noalpha = vec![0u8; row_b];
            arch::Kernel::bgra_to_subgreen_bgr3::<false>(bp0, &mut bgr0_noalpha, &mut []);
            assert_eq!(bgr0_noalpha, bgr0_s, "bgra_to_subgreen_bgr3<false> tw={tw}");

            ScalarKernel::bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_s, &mut alp1_s);
            arch::Kernel::bgra_to_subgreen_bgr3::<true>(bp1, &mut bgr1_a, &mut alp1_a);
            assert_eq!(bgr1_a, bgr1_s);

            assert_eq!(
                arch::Kernel::sad_signed_bytes(&bgr0_a),
                ScalarKernel::sad_signed_bytes(&bgr0_s),
                "sad_signed_bytes tw={tw}"
            );

            // Sub3 forward + inverse
            let (mut f_s, mut f_a) = (vec![0u8; row_b], vec![0u8; row_b]);
            assert_eq!(
                arch::Kernel::row_sub3_forward_sad(&bgr0_a, &mut f_a),
                ScalarKernel::row_sub3_forward_sad(&bgr0_s, &mut f_s),
                "sub3 sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "sub3 fwd tw={tw}");
            let mut inv = vec![0xAAu8; row_b];
            arch::Kernel::row_sub3_inverse(&f_a, &mut inv);
            assert_eq!(inv, bgr0_s, "sub3 inv tw={tw}");

            // Up forward + inverse
            assert_eq!(
                arch::Kernel::row_up_forward_sad(&bgr1_a, &bgr0_a, &mut f_a),
                ScalarKernel::row_up_forward_sad(&bgr1_s, &bgr0_s, &mut f_s),
                "up sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "up fwd tw={tw}");
            inv.fill(0xAA);
            arch::Kernel::row_up_inverse(&f_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "up inv tw={tw}");

            // SubUp3 forward + inverse
            assert_eq!(
                arch::Kernel::row_subup3_forward_sad(&bgr1_a, &bgr0_a, &mut f_a),
                ScalarKernel::row_subup3_forward_sad(&bgr1_s, &bgr0_s, &mut f_s),
                "subup3 sad tw={tw}"
            );
            assert_eq!(f_a, f_s, "subup3 fwd tw={tw}");
            inv.fill(0xAA);
            arch::Kernel::row_subup3_inverse(&f_a, &bgr0_a, &mut inv);
            assert_eq!(inv, bgr1_s, "subup3 inv tw={tw}");

            // Sub1 & Up on alpha plane
            let (mut af_s, mut af_a) = (vec![0u8; tw], vec![0u8; tw]);
            ScalarKernel::row_sub1_forward(&alp0_s, &mut af_s);
            arch::Kernel::row_sub1_forward(&alp0_a, &mut af_a);
            assert_eq!(af_a, af_s, "sub1 fwd tw={tw}");
            let mut a_inv = vec![0xAAu8; tw];
            arch::Kernel::row_sub1_inverse(&af_a, &mut a_inv);
            assert_eq!(a_inv, alp0_s, "sub1 inv tw={tw}");

            ScalarKernel::row_up_forward(&alp1_s, &alp0_s, &mut af_s);
            arch::Kernel::row_up_forward(&alp1_a, &alp0_a, &mut af_a);
            assert_eq!(af_a, af_s, "alpha up fwd tw={tw}");
            arch::Kernel::row_up_inverse(&af_a, &alp0_a, &mut a_inv);
            assert_eq!(a_inv, alp1_s, "alpha up inv tw={tw}");

            // Unpack BGR3 -> BGRA (constant alpha and varying alpha, both alignments)
            let mut dst_buf = vec![Vec4u8(0xAA, 0xAA, 0xAA, 0xAA); tw + 8];
            let aligned_off = ((32 - (dst_buf.as_ptr() as usize & 31)) & 31) / 4;
            for (stream, off) in [(true, aligned_off), (false, aligned_off + 1)] {
                let dst = &mut dst_buf[off..off + tw];
                dst.fill(Vec4u8(0, 0, 0, 0));
                unsafe {
                    if stream && arch::Kernel::STREAM_ALIGN > 1 {
                        arch::Kernel::subgreen_bgr3_to_bgra::<true, false>(
                            &bgr0_a, &alp0_a, 0, dst,
                        );
                    } else {
                        arch::Kernel::subgreen_bgr3_to_bgra::<false, false>(
                            &bgr0_a, &alp0_a, 0, dst,
                        );
                    }
                }
                assert_eq!(
                    dst,
                    &raw0[..],
                    "unpack varying alpha tw={tw} stream={stream}"
                );

                dst.fill(Vec4u8(0, 0, 0, 0));
                unsafe {
                    if stream && arch::Kernel::STREAM_ALIGN > 1 {
                        arch::Kernel::subgreen_bgr3_to_bgra::<true, true>(&bgr0_a, &[], 0x7E, dst);
                    } else {
                        arch::Kernel::subgreen_bgr3_to_bgra::<false, true>(&bgr0_a, &[], 0x7E, dst);
                    }
                }
                for (got, exp) in dst.iter().zip(&raw0) {
                    assert_eq!(*got, Vec4u8(exp.0, exp.1, exp.2, 0x7E));
                }

                // Splat row
                let px = Vec4u8(0x11, 0x22, 0x33, 0x44);
                unsafe {
                    if stream && arch::Kernel::STREAM_ALIGN > 1 {
                        arch::Kernel::splat_row::<true>(dst, px);
                    } else {
                        arch::Kernel::splat_row::<false>(dst, px);
                    }
                }
                assert!(dst.iter().all(|&p| p == px));
            }
            arch::Kernel::fence_stream();

            // `is_constant_plane` on constant + single perturbed byte at start/mid/end.
            let mut plane = vec![0x5Au8; tw];
            assert_eq!(arch::Kernel::is_constant_plane(&plane), Some(0x5A));
            plane[tw - 1] ^= 1;
            assert_eq!(
                arch::Kernel::is_constant_plane(&plane),
                if tw == 1 { Some(0x5B) } else { None }
            );
            if tw > 1 {
                plane[tw - 1] = 0x5A;
                plane[tw / 2] ^= 1;
                assert_eq!(arch::Kernel::is_constant_plane(&plane), None);
            }

            // `diff_and_update_row` matches `ScalarKernel` on identical, alpha-only,
            // and RGB-perturbed pixels at start/mid/end.
            let mut sh_s = raw0.clone();
            let mut sh_a = raw0.clone();
            assert!(!ScalarKernel::diff_and_update_row::<true>(bp0, &mut sh_s));
            assert!(!arch::Kernel::diff_and_update_row::<true>(bp0, &mut sh_a));
            assert!(!arch::Kernel::diff_and_update_row::<false>(bp0, &mut sh_a));

            for pos in [0, tw / 2, tw - 1] {
                let mut alt_a = raw0.clone();
                alt_a[pos].3 ^= 0x55;
                let pa = alt_a.as_ptr();
                let bpa = unsafe { BufferPointer::new(&pa, tw) };
                sh_s.clone_from(&raw0);
                sh_a.clone_from(&raw0);
                assert!(!ScalarKernel::diff_and_update_row::<false>(bpa, &mut sh_s));
                assert!(!arch::Kernel::diff_and_update_row::<false>(bpa, &mut sh_a));
                assert_eq!(
                    sh_a, raw0,
                    "xrgb alpha-only must not write tw={tw} pos={pos}"
                );
                assert!(ScalarKernel::diff_and_update_row::<true>(bpa, &mut sh_s));
                assert!(arch::Kernel::diff_and_update_row::<true>(bpa, &mut sh_a));
                assert_eq!(sh_a, alt_a, "argb alpha diff tw={tw} pos={pos}");
                assert_eq!(sh_a, sh_s);

                let mut alt_rgb = raw0.clone();
                alt_rgb[pos].1 ^= 0x33;
                alt_rgb[tw - 1].2 ^= 0x11;
                let pr = alt_rgb.as_ptr();
                let bpr = unsafe { BufferPointer::new(&pr, tw) };
                sh_s.clone_from(&raw0);
                sh_a.clone_from(&raw0);
                assert!(ScalarKernel::diff_and_update_row::<false>(bpr, &mut sh_s));
                assert!(arch::Kernel::diff_and_update_row::<false>(bpr, &mut sh_a));
                assert_eq!(sh_a, alt_rgb, "xrgb rgb diff tw={tw} pos={pos}");
                assert_eq!(sh_a, sh_s);
            }
        }
        // `i8::MIN` (`0x80` = `-128`) unsigned abs must equal `128` across all backends.
        let min_bytes = [0x80u8; 41];
        assert_eq!(arch::Kernel::sad_signed_bytes(&min_bytes), 41 * 128);
        assert_eq!(arch::Kernel::is_constant_plane(&[]), None);
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
        assert!(matches!(
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
        ));
        assert!(matches!(
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
        ));

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
        assert!(matches!(
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
        ));

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
        // `diff_and_update_row_band_2d` (both packed and padded strides,
        // partial-dirty and bulk-all-dirty fast paths, Xrgb vs Argb).
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
                assert!(!diff_and_update_strip::<true>(
                    aos,
                    &shadow_rows,
                    stride_px,
                    rect_strip
                ));
                assert!(!diff_and_update_strip::<false>(
                    aos,
                    &shadow_rows,
                    stride_px,
                    rect_strip
                ));
            }

            // Flip only byte 3 (X/alpha) in quadrant 0 of `[0..12)` -> ignored
            // on Xrgb, caught on Argb.
            let a_off = 5 * stride + 10 * 4 + 3;
            src[a_off] ^= 0xFF;
            let src_ptr = src.as_ptr();
            let bp = unsafe { BufferPointer::new(&src_ptr, len) };
            let aos = unsafe { bp.cast::<Vec4u8>() };
            let slots_xrgb = [AtomicBool::new(false), AtomicBool::new(false)];
            unsafe {
                diff_and_update_row_band_2d::<false>(
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
            assert!(!slots_xrgb[0].load(Ordering::Relaxed));
            assert!(!slots_xrgb[1].load(Ordering::Relaxed));
            assert_ne!(shadow[a_off], src[a_off]);

            let slots_argb = [AtomicBool::new(false), AtomicBool::new(false)];
            unsafe {
                diff_and_update_row_band_2d::<true>(
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
            assert!(slots_argb[0].load(Ordering::Relaxed));
            assert!(!slots_argb[1].load(Ordering::Relaxed));
            assert_eq!(shadow[a_off], src[a_off]);

            // Flip RGB in BOTH quadrants on row 0 AND another pixel on row 11
            // to exercise the all-quadrants-dirty bulk copy fast path.
            let q0_off = 2 * 4;
            let q1_off = 140 * 4;
            let last_off = 11 * stride + 150 * 4;
            src[q0_off] ^= 0x11;
            src[q1_off] ^= 0x22;
            src[last_off] ^= 0x33;
            let src_ptr = src.as_ptr();
            let bp = unsafe { BufferPointer::new(&src_ptr, len) };
            let aos = unsafe { bp.cast::<Vec4u8>() };
            let slots_all = [AtomicBool::new(false), AtomicBool::new(false)];
            unsafe {
                diff_and_update_row_band_2d::<false>(
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
            assert!(slots_all[0].load(Ordering::Relaxed));
            assert!(slots_all[1].load(Ordering::Relaxed));
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
                assert!(diff_and_update_strip::<false>(
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
            assert!(
                t_solid
                    .tiles
                    .iter()
                    .all(|t| matches!(t.payload, TilePayload::Solid([0x20, 0x70, 0xD0, 0xFF])))
            );
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
            assert!(
                t_trans
                    .tiles
                    .iter()
                    .all(|t| matches!(t.payload, TilePayload::Solid([0x20, 0x70, 0xD0, 0x80])))
            );
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
                assert!(matches!(
                    tiles.tiles[0].payload,
                    TilePayload::Palette {
                        has_alpha,
                        ..
                    } if has_alpha == translucent
                ));
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
            assert!(matches!(
                t_opaque_argb.tiles[0].payload,
                TilePayload::AosZstd {
                    alpha: AlphaChannel::Omitted,
                    ..
                }
            ));

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
            assert!(matches!(
                t_const_a.tiles[0].payload,
                TilePayload::AosZstd {
                    alpha: AlphaChannel::Constant(0x80),
                    ..
                }
            ));

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
            assert!(matches!(
                t_var_a.tiles[0].payload,
                TilePayload::AosZstd {
                    alpha: AlphaChannel::Uncompressed(_) | AlphaChannel::Zstd(_),
                    ..
                }
            ));
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
                        decompress_and_unfilter_streaming(validated, decomp, canvas).unwrap();
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
                            Some(&mut shadow),
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
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            None,
                            &mut shadow,
                            compressor,
                            |_| called = true,
                        );
                        assert!(dirty.is_empty());
                        assert!(!called);
                    }

                    // Step 2: Perturbing ONLY the alpha/X byte -> skipped on Xrgb8888,
                    // caught on Argb8888!
                    set_px(&mut frame, 10, 20, 0, 0x55);
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            None,
                            &mut shadow,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        if format == BufferFormat::Xrgb8888 {
                            assert!(dirty.is_empty());
                            assert!(streaming.is_none());
                        } else {
                            assert_eq!(dirty, vec![tr(0, 0, 128, 128)]);
                            apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                        }
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 3: 1 dirty quadrant in (0,0,256,256) + 2 dirty quadrants in clipped
                    // right edge (256,0,144,256) + 1 dirty quadrant in bottom edge (0,256,256,104).
                    set_px(&mut frame, 150, 10, 0x33, 0); // TR of (0,0,256,256) -> (128,0,128,128)
                    set_px(&mut frame, 390, 5, 0x44, 0); // TR of (256,0,144,256) -> (384,0,16,128)
                    set_px(&mut frame, 260, 200, 0x55, 0); // BL of (256,0,144,256) -> (256,128,128,128)
                    set_px(&mut frame, 200, 300, 0x66, 0); // right of (0,256,256,104) -> (128,256,128,104)
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            None,
                            &mut shadow,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(
                            dirty,
                            vec![
                                tr(128, 0, 128, 128),
                                tr(384, 0, 16, 128),
                                tr(256, 128, 128, 128),
                                tr(128, 256, 128, 104),
                            ]
                        );
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 4: All 4 quadrants dirty in (0,0,256,256) -> coalesces to ONE `256x256`
                    // tile! Also all 2 quadrants dirty in bottom-right (256,256,144,104) -> ONE
                    // `144x104` tile!
                    set_px(&mut frame, 1, 1, 0x11, 0);
                    set_px(&mut frame, 130, 1, 0x11, 0);
                    set_px(&mut frame, 1, 130, 0x11, 0);
                    set_px(&mut frame, 130, 130, 0x11, 0);
                    set_px(&mut frame, 260, 260, 0x22, 0);
                    set_px(&mut frame, 395, 355, 0x22, 0);
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            None,
                            &mut shadow,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(dirty, vec![tr(0, 0, 256, 256), tr(256, 256, 144, 104)]);
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 5: `< 8`-row partial band (1D strip): unchanged -> skipped;
                    // changed -> emits single 1D strip and updates shadow.
                    let thin_band = [RowBand { y0: 40, y1: 43 }];
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            Some(&thin_band),
                            &mut shadow,
                            compressor,
                            |_| panic!("unchanged 1D strip must not emit stream"),
                        );
                        assert!(dirty.is_empty());
                    }
                    set_px(&mut frame, 300, 41, 0x77, 0);
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let mut streaming = None;
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            Some(&thin_band),
                            &mut shadow,
                            compressor,
                            |s| streaming = Some(s),
                        );
                        assert_eq!(dirty, vec![tr(0, 40, 400, 3)]);
                        apply_stream(streaming.unwrap(), decompressor, &mut client_canvas);
                    }
                    check_canvas(&client_canvas, &frame);

                    // Step 6: Immediately re-scanning the full frame sees 0 dirty tiles,
                    // proving the shadow buffer stayed bit-exact across all edits.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            None,
                            &mut shadow,
                            compressor,
                            |_| panic!("shadow desynchronized after multi-frame edits"),
                        );
                        assert!(dirty.is_empty());
                    }

                    // Step 7: Empty bands slice returns empty immediately.
                    {
                        let ptr = frame.as_ptr();
                        let bp = unsafe { BufferPointer::new(&ptr, len) };
                        let dirty = filter_and_compress_incremental_streaming(
                            bp,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            Some(&[]),
                            &mut shadow,
                            compressor,
                            |_| panic!("empty bands must not emit stream"),
                        );
                        assert!(dirty.is_empty());
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
                        payload: TilePayload::Solid([1, 2, 3, 0xFF]),
                    },
                    CompressedTile {
                        tile_idx: 1,
                        payload: TilePayload::Solid([4, 5, 6, 0xFF]),
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
                        payload: TilePayload::Solid([1, 2, 3, 0xFF]),
                    },
                    CompressedTile {
                        tile_idx: 0,
                        payload: TilePayload::Solid([4, 5, 6, 0xFF]),
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
                    payload: TilePayload::Solid([1, 2, 3, 0xFF]),
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
                        payload: TilePayload::Solid([1, 2, 3, 0xFF]),
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
                TilePayload::AosZstd {
                    predictors: vec![2], // Up on row 0
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![3], // SubUp on row 0
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![0, 0], // wrong len (expected 1)
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![0b01_00_00_00], // non-zero padding bits for h=3
                    color: EncodedBytes::Uncompressed(vec![0u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 35]), // != 36
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Zstd(vec![0xFF, 0x00, 0x01]),
                    alpha: AlphaChannel::Omitted,
                },
                TilePayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 36]),
                    alpha: AlphaChannel::Uncompressed(vec![0u8; 11]), // != 12
                },
                TilePayload::Palette {
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
                        payload: bad_payload,
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
                TilePayload::Solid([1, 2, 3, 0x80]),
                TilePayload::Palette {
                    has_alpha: true,
                    palette: vec![1, 2, 3, 0x80, 4, 5, 6, 0xFF],
                    data: EncodedBytes::Uncompressed(vec![0, 0]),
                },
                TilePayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Uncompressed(vec![0u8; 36]),
                    alpha: AlphaChannel::Constant(0x80),
                },
            ] {
                let bad = CompressedTiles {
                    metadata: xrgb_meta,
                    tile_rects: vec![r4x3],
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        payload: bad_xrgb,
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
                Some(&mut shadow_dropped),
                compressor,
                drop,
            );
            // Shadow must still be fully populated even when stream receiver dropped early.
            assert_eq!(shadow_dropped, data_2);

            let mut two_tiles = filter_and_compress(bp_2, meta_2, compressor);
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
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta_2,
                    incremental: false,
                    tile_rects: two_tiles.tile_rects.clone(),
                },
                tiles: rx_short,
            })
            .unwrap();
            assert!(
                decompress_and_unfilter_streaming(validated_short, decompressor, &mut reversed_out)
                    .is_err()
            );

            // Upfront `ValidatedTileStream::new` rejections:
            // - non-incremental incomplete coverage (missing pixels / empty)
            // - overlapping rects
            // - out-of-bounds rect
            // - zero-dim rect
            // - excessive tile count (> width * height)
            let bad_headers = [
                // Non-incremental with empty rects.
                RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: false,
                    tile_rects: vec![],
                },
                // Non-incremental covering only half the frame.
                RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: false,
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 32,
                        h: 64,
                    }],
                },
                // Incremental overlapping rects.
                RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
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
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    tile_rects: vec![TileRect {
                        x: 32,
                        y: 0,
                        w: 33,
                        h: 64,
                    }],
                },
                // Zero-dim rect.
                RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta,
                    incremental: true,
                    tile_rects: vec![TileRect {
                        x: 0,
                        y: 0,
                        w: 0,
                        h: 64,
                    }],
                },
                // Excessive tile count (`5 > 2 * 2`).
                RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: BufferMetadata {
                        width: 2,
                        height: 2,
                        stride: 8,
                        format: BufferFormat::Xrgb8888,
                    },
                    incremental: true,
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
}
