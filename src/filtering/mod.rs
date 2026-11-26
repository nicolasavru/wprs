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

//! 2D tiled pixel filtering and lossless compression/decompression.
//!
//! Pixel buffers are partitioned into self-describing 2D `256x256` tiles (with
//! a full-width 1D strip fallback for thin `< 8`-row partial bands so 1-row
//! updates avoid tile-header fragmentation). Each tile is encoded on a worker
//! thread inside L1/L2 scratch memory as:
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
use std::slice;
use std::sync::Mutex;
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
use crate::serialization::framing::RawBufferHeader;
use crate::serialization::framing::RowBand;
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
) -> CompressedTile {
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
            return CompressedTile {
                rect,
                payload: TilePayload::Solid(px),
            };
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
        return CompressedTile {
            rect,
            payload: TilePayload::Palette {
                has_alpha: pal_has_alpha,
                palette: palette_bytes,
                data,
            },
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

    CompressedTile {
        rect,
        payload: TilePayload::AosZstd {
            predictors,
            color,
            alpha,
        },
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
/// `tile.rect` must already be validated in-bounds within `frame` and disjoint
/// from all other tiles being decoded concurrently.
unsafe fn unfilter_tile<K: RowKernel, const HAS_ALPHA: bool>(
    tile: &CompressedTile,
    frame: &FrameRows,
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let x0 = tile.rect.x as usize;
    let y0 = tile.rect.y as usize;
    let tw = tile.rect.w as usize;
    let th = tile.rect.h as usize;
    let n_px = tw * th;
    debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

    match &tile.payload {
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

fn push_region_tiles(out: &mut Vec<TileRect>, width: u32, y0: u32, y1: u32) {
    let band_h = y1 - y0;
    if band_h < MIN_2D_BAND_HEIGHT {
        // Emit a single full-width 1D strip for normal displays, splitting
        // horizontally only if `width * band_h > TILE_PIXELS` so every tile
        // stays within L2 scratch bounds.
        let max_strip_w = (TILE_PIXELS as u32 / band_h).max(1);
        let mut x = 0u32;
        while x < width {
            let w = (width - x).min(max_strip_w);
            out.push(TileRect {
                x,
                y: y0,
                w,
                h: band_h,
            });
            x += w;
        }
        return;
    }

    let mut y = y0;
    while y < y1 {
        let h = (y1 - y).min(TILE_H as u32);
        let mut x = 0u32;
        while x < width {
            let w = (width - x).min(TILE_W as u32);
            out.push(TileRect { x, y, w, h });
            x += w;
        }
        y += h;
    }
}

/// Validates `bands` (when `Some`: sorted, non-overlapping, non-empty,
/// `y1 <= height`) and partitions the full frame (`None`) or each damaged band
/// into clipped `256x256` 2D tiles (with `< 8`-row 1D strip fallback).
/// `metadata` must already pass [`BufferMetadata::validate`].
fn partition_tiles(metadata: &BufferMetadata, bands: Option<&[RowBand]>) -> Result<Vec<TileRect>> {
    let width = metadata.width as u32;
    let height = metadata.height as u32;
    let mut rects = Vec::new();
    match bands {
        None => push_region_tiles(&mut rects, width, 0, height),
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
                push_region_tiles(&mut rects, width, band.y0, band.y1);
            }
        },
    }
    Ok(rects)
}

// --- public API ---

#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress_streaming(
    data: BufferPointer<u8>,
    client: ClientId,
    surface: WlSurfaceId,
    metadata: BufferMetadata,
    bands: Option<Vec<RowBand>>,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedTiles),
) {
    assert!(data.len().is_multiple_of(4));
    let total_pixels = metadata
        .validate()
        .expect("server passed invalid BufferMetadata");
    assert!(data.len() >= total_pixels * 4);
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };
    let stride_px = (metadata.stride as usize) / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();

    let tile_rects =
        partition_tiles(&metadata, bands.as_deref()).expect("server generated invalid row bands");
    let n_tiles = tile_rects.len();

    let header = RawBufferHeader {
        client,
        surface,
        metadata,
        n_tiles: n_tiles as u32,
        bands,
    };
    on_stream_ready(StreamingCompressedTiles {
        header,
        tiles: tiles_rx,
    });

    let next_tile = AtomicUsize::new(0);
    compressor.pool.execute(n_tiles, |state| {
        loop {
            let tile_idx = next_tile.fetch_add(1, Ordering::Relaxed);
            if tile_idx >= n_tiles {
                break;
            }
            let rect = tile_rects[tile_idx];
            let tile = if has_alpha {
                filter_tile::<arch::Kernel, true>(aos, stride_px, rect, state)
            } else {
                filter_tile::<arch::Kernel, false>(aos, stride_px, rect, state)
            };

            if tiles_tx.send(tile).is_err() {
                break;
            }
        }
    });
}

#[instrument(skip_all, level = "debug")]
pub fn filter_and_compress(
    data: BufferPointer<u8>,
    metadata: BufferMetadata,
    compressor: &mut ShardingCompressor,
) -> CompressedTiles {
    let mut streaming_opt = None;
    filter_and_compress_streaming(
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
    let mut tiles: Vec<CompressedTile> = streaming_opt.unwrap().tiles.into_iter().collect();
    tiles.sort_by_key(|t| (t.rect.y, t.rect.x));
    CompressedTiles { metadata, tiles }
}

fn unfilter_tiles_parallel<'a>(
    decompressor: &mut ShardingDecompressor,
    metadata: BufferMetadata,
    expected_tiles: usize,
    output_buf: &mut [u8],
    next_tile: impl Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
) -> Result<()> {
    let total_pixels = metadata.validate().location(loc!())?;
    let byte_len = total_pixels
        .checked_mul(4)
        .context(loc!(), "pixel count overflow")?;
    if output_buf.len() < byte_len {
        bail!("output_buf len {} < {byte_len}", output_buf.len());
    }

    let width = metadata.width as u32;
    let height = metadata.height as u32;
    let stride_px = (metadata.stride as usize) / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let aos: &mut [Vec4u8] = bytemuck::cast_slice_mut(&mut output_buf[..byte_len]);
    let frame = FrameRows {
        base: aos.as_mut_ptr(),
        stride_px,
    };

    // Track every accepted tile rect under a short mutex so overlapping or
    // out-of-bounds wire tiles are rejected BEFORE any worker writes to `frame`.
    let accepted: Mutex<Vec<TileRect>> = Mutex::new(Vec::with_capacity(expected_tiles.min(1024)));
    let completed = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    decompressor.pool.execute(expected_tiles, |state| {
        let mut streamed_any = false;
        while let Some(item) = next_tile() {
            let res = (|| -> Result<bool> {
                let tile = item?;
                tile.rect
                    .validate_in_bounds(width, height)
                    .location(loc!())?;
                {
                    let mut guard = accepted.lock().unwrap();
                    if guard.len() >= expected_tiles {
                        bail!("received more than expected {expected_tiles} tiles");
                    }
                    if let Some(conflict) = guard.iter().find(|prev| prev.overlaps(tile.rect)) {
                        bail!(
                            "overlapping TileRect {:?} conflicts with {:?}",
                            tile.rect,
                            conflict
                        );
                    }
                    guard.push(tile.rect);
                }
                // SAFETY: `tile.rect` passed `validate_in_bounds(width, height)`
                // and pairwise disjointness against every other accepted tile
                // in this frame before writing into `frame`.
                unsafe {
                    if has_alpha {
                        unfilter_tile::<arch::Kernel, true>(&tile, &frame, state)
                    } else {
                        unfilter_tile::<arch::Kernel, false>(&tile, &frame, state)
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
    if done != expected_tiles {
        bail!("received {done} tiles, expected {expected_tiles}");
    }
    Ok(())
}

pub struct ValidatedTileStream {
    stream: CompressedTileStream,
    expected_tiles: usize,
}

impl ValidatedTileStream {
    pub fn new(stream: CompressedTileStream) -> Result<Self> {
        stream.header.metadata.validate().location(loc!())?;
        let expected_tiles =
            partition_tiles(&stream.header.metadata, stream.header.bands.as_deref())
                .location(loc!())?
                .len();
        if stream.header.n_tiles as usize != expected_tiles {
            bail!(
                "header n_tiles {} != expected {expected_tiles}",
                stream.header.n_tiles,
            );
        }
        Ok(Self {
            stream,
            expected_tiles,
        })
    }

    pub fn metadata(&self) -> BufferMetadata {
        self.stream.header.metadata
    }

    pub fn bands(&self) -> Option<&[RowBand]> {
        self.stream.header.bands.as_deref()
    }
}

#[instrument(skip_all, level = "debug")]
pub fn decompress_and_unfilter_streaming(
    validated: ValidatedTileStream,
    decompressor: &mut ShardingDecompressor,
    output_buf: &mut [u8],
) -> Result<()> {
    let metadata = validated.stream.header.metadata;
    let tiles_rx = &validated.stream.tiles;
    unfilter_tiles_parallel(
        decompressor,
        metadata,
        validated.expected_tiles,
        output_buf,
        || tiles_rx.recv().ok().map(|r| r.map(Cow::Owned)),
    )
}

#[instrument(skip_all, level = "debug")]
pub fn decompress_and_unfilter(
    tiles: &CompressedTiles,
    decompressor: &mut ShardingDecompressor,
    output_buf: &mut [u8],
) -> Result<()> {
    let n_tiles = tiles.tiles.len();
    let next_idx = AtomicUsize::new(0);
    unfilter_tiles_parallel(decompressor, tiles.metadata, n_tiles, output_buf, || {
        let i = next_idx.fetch_add(1, Ordering::Relaxed);
        (i < n_tiles).then(|| Ok(Cow::Borrowed(&tiles.tiles[i])))
    })
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
        let full = partition_tiles(&meta, None).unwrap();
        assert_eq!(full.len(), 15 * 9);
        assert_eq!(
            full[0],
            TileRect {
                x: 0,
                y: 0,
                w: 256,
                h: 256,
            }
        );
        assert_eq!(
            *full.last().unwrap(),
            TileRect {
                x: 3584,
                y: 2048,
                w: 256,
                h: 112,
            }
        );

        // Partial `< 8`-row bands emit 1 full-width 1D strip each.
        for h in [1u32, 7] {
            let strip = partition_tiles(
                &meta,
                Some(&[RowBand {
                    y0: 100,
                    y1: 100 + h,
                }]),
            )
            .unwrap();
            assert_eq!(
                strip,
                vec![TileRect {
                    x: 0,
                    y: 100,
                    w: 3840,
                    h,
                }]
            );
        }

        // Exact `band_h == 8` crosses into 2D `256x8` tiling (15 tiles on 4K).
        let band8 = partition_tiles(&meta, Some(&[RowBand { y0: 100, y1: 108 }])).unwrap();
        assert_eq!(band8.len(), 15);
        assert_eq!(
            band8[0],
            TileRect {
                x: 0,
                y: 100,
                w: 256,
                h: 8,
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
        let wide_strips = partition_tiles(&wide_meta, Some(&[RowBand { y0: 0, y1: 7 }])).unwrap();
        assert_eq!(wide_strips.len(), 3);
        assert!(wide_strips.iter().all(|r| r.checked_pixels().is_ok()));
        assert_eq!(wide_strips.iter().map(|r| r.w).sum::<u32>(), 20_000);

        // Empty bands -> 0 tiles; invalid bands -> Err.
        assert!(partition_tiles(&meta, Some(&[])).unwrap().is_empty());
        for bad_bands in [
            vec![RowBand { y0: 5, y1: 5 }],
            vec![RowBand { y0: 10, y1: 5 }],
            vec![RowBand { y0: 10, y1: 20 }, RowBand { y0: 15, y1: 25 }],
            vec![RowBand { y0: 2150, y1: 2161 }],
        ] {
            assert!(partition_tiles(&meta, Some(&bad_bands)).is_err());
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
    fn test_partial_row_bands_streaming_roundtrip() {
        with_test_codec(|compressor, decompressor| {
            for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
                for pad_bytes in [0i32, 16i32] {
                    let width = 480i32;
                    let height = 600i32;
                    let stride = width * 4 + pad_bytes;
                    let len = (stride * height) as usize;
                    let mut frame0 = test_vec(len);
                    let frame1: Vec<u8> = (0..len).map(|i| ((i * 11 + 7) % 255) as u8).collect();
                    let mut frame1_expected = frame1.clone();
                    if format == BufferFormat::Xrgb8888 {
                        for px in frame0.as_chunks_mut::<4>().0 {
                            px[3] = 0xFF;
                        }
                        for px in frame1_expected.as_chunks_mut::<4>().0 {
                            px[3] = 0xFF;
                        }
                    }

                    // Band 1: 1 row (`< 8` -> 1D strip fallback); Band 2: 180 rows (2D tiles).
                    let partial_bands = compute_partial_row_bands(
                        &[
                            Rectangle::new(0, 3, width, 1),
                            Rectangle::new(50, 300, 100, 180),
                        ],
                        width,
                        height,
                    )
                    .unwrap();

                    for bands_case in [None, Some(Vec::new()), Some(partial_bands)] {
                        let metadata = BufferMetadata {
                            width,
                            height,
                            stride,
                            format,
                        };
                        let ptr = frame1.as_ptr();
                        let buf_ptr = unsafe { BufferPointer::new(&ptr, len) };

                        let mut streaming_opt = None;
                        filter_and_compress_streaming(
                            buf_ptr,
                            ClientId(1),
                            WlSurfaceId(2),
                            metadata,
                            bands_case.clone(),
                            compressor,
                            |s| streaming_opt = Some(s),
                        );
                        let mut wire = Vec::new();
                        streaming_opt.unwrap().framed_write(&mut wire).unwrap();

                        let mut client_canvas = AlignedVec::<64>::with_capacity(len);
                        client_canvas.extend_from_slice(&frame0);
                        let mut wire_slice = wire.as_slice();
                        let mut tile_stream_opt = None;
                        CompressedTileStream::read_from_stream(&mut wire_slice, |tile_stream| {
                            tile_stream_opt = Some(tile_stream);
                            Ok(())
                        })
                        .unwrap();
                        decompress_and_unfilter_streaming(
                            ValidatedTileStream::new(tile_stream_opt.unwrap()).unwrap(),
                            decompressor,
                            &mut client_canvas,
                        )
                        .unwrap();

                        let visible_bytes = (width * 4) as usize;
                        let stride_us = stride as usize;
                        for y in 0..height as usize {
                            let row_damaged = match &bands_case {
                                None => true,
                                Some(bs) => bs
                                    .iter()
                                    .any(|b| (b.y0 as usize..b.y1 as usize).contains(&y)),
                            };
                            let row_off = y * stride_us;
                            let exp_row = if row_damaged {
                                &frame1_expected[row_off..row_off + visible_bytes]
                            } else {
                                &frame0[row_off..row_off + visible_bytes]
                            };
                            assert_eq!(
                                &client_canvas[row_off..row_off + visible_bytes],
                                exp_row,
                                "row {y} mismatch (format={format:?}, pad={pad_bytes})"
                            );
                            // Stride padding bytes must remain untouched.
                            assert_eq!(
                                &client_canvas[row_off + visible_bytes..row_off + stride_us],
                                &frame0[row_off + visible_bytes..row_off + stride_us],
                                "stride padding modified on row {y}"
                            );
                        }
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
                tiles: vec![
                    CompressedTile {
                        rect: TileRect {
                            x: 0,
                            y: 0,
                            w: 40,
                            h: 40,
                        },
                        payload: TilePayload::Solid([1, 2, 3, 0xFF]),
                    },
                    CompressedTile {
                        rect: TileRect {
                            x: 39,
                            y: 39,
                            w: 25,
                            h: 25,
                        },
                        payload: TilePayload::Solid([4, 5, 6, 0xFF]),
                    },
                ],
            };
            assert!(decompress_and_unfilter(&overlapping, decompressor, &mut out).is_err());

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
                    tiles: vec![CompressedTile {
                        rect: bad_rect,
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
                    tiles: vec![CompressedTile {
                        rect: r4x3,
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
                    tiles: vec![CompressedTile {
                        rect: r4x3,
                        payload: bad_xrgb,
                    }],
                };
                assert!(decompress_and_unfilter(&bad, decompressor, &mut out).is_err());
            }

            // Multi-tile frame: early drop in `on_stream_ready`, out-of-order arrival,
            // undersized `output_buf`, missing tile, and mismatched `header.n_tiles`.
            let meta_2 = BufferMetadata {
                width: 300,
                height: 20,
                stride: 300 * 4,
                format: BufferFormat::Argb8888,
            };
            let data_2 = test_vec(300 * 20 * 4);
            let ptr_2 = data_2.as_ptr();
            let bp_2 = unsafe { BufferPointer::new(&ptr_2, data_2.len()) };

            filter_and_compress_streaming(
                bp_2,
                ClientId(1),
                WlSurfaceId(1),
                meta_2,
                None,
                compressor,
                drop,
            );

            let mut two_tiles = filter_and_compress(bp_2, meta_2, compressor);
            assert_eq!(two_tiles.tiles.len(), 2);
            two_tiles.tiles.reverse();
            let mut reversed_out = vec![0u8; data_2.len()];
            decompress_and_unfilter(&two_tiles, decompressor, &mut reversed_out).unwrap();
            assert_eq!(reversed_out, data_2);

            let mut short_out = vec![0u8; data_2.len() - 4];
            assert!(decompress_and_unfilter(&two_tiles, decompressor, &mut short_out).is_err());

            // Streaming missing tile (`done < expected_tiles`).
            let (tx_short, rx_short) = crossbeam_channel::unbounded();
            tx_short.send(Ok(two_tiles.tiles[0].clone())).unwrap();
            drop(tx_short);
            let validated_short = ValidatedTileStream::new(CompressedTileStream {
                header: RawBufferHeader {
                    client: ClientId(1),
                    surface: WlSurfaceId(1),
                    metadata: meta_2,
                    n_tiles: 2,
                    bands: None,
                },
                tiles: rx_short,
            })
            .unwrap();
            assert!(
                decompress_and_unfilter_streaming(validated_short, decompressor, &mut reversed_out)
                    .is_err()
            );

            // Mismatched `header.n_tiles`.
            for bad_n_tiles in [0u32, 2] {
                let (_tx, rx) = crossbeam_channel::unbounded();
                let stream = CompressedTileStream {
                    header: RawBufferHeader {
                        client: ClientId(1),
                        surface: WlSurfaceId(1),
                        metadata: meta,
                        n_tiles: bad_n_tiles,
                        bands: None,
                    },
                    tiles: rx,
                };
                assert!(ValidatedTileStream::new(stream).is_err());
            }
        });
    }
}
