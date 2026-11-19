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

//! AoS↔SoA transposition and spatial/color filtering for pixel buffers.
//!
//! The pipeline partitions pixel buffers into self-contained L2-resident tiles
//! (`TILE_PIXELS`), transposes each tile between array-of-structs (RGBA
//! interleaved) and struct-of-arrays (channels separated) inside per-worker
//! scratch memory, applies subtract_green (VP8L color decorrelation) and
//! running_difference (delta encoding), and compresses/decompresses directly in
//! cache on the same worker thread.
//!
//! Architecture-specific backends:
//! - `x86`: permutation-based transpose (SSE2 through AVX2)
//! - `neon`: hardware structure load/store (vld4q/vst4q)
//!
//! Ref:
//! * <https://afrantzis.com/pixel-format-guide/wayland_drm.html>
//! * <https://en.algorithmica.org/hpc/algorithms/prefix/>
#[cfg(target_arch = "x86_64")]
#[path = "x86.rs"]
mod arch;

#[cfg(target_arch = "aarch64")]
#[path = "neon.rs"]
mod arch;

use std::borrow::Cow;
use std::slice;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use itertools::izip;
use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;

use crate::buffer_pointer::BufferPointer;
use crate::buffer_pointer::KnownSizeBufferPointer;
use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::framing::CompressedTile;
use crate::serialization::framing::CompressedTileStream;
use crate::serialization::framing::CompressedTiles;
use crate::serialization::framing::RawBufferHeader;
use crate::serialization::framing::RowBand;
use crate::serialization::framing::StreamingCompressedTiles;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileChannel;
use crate::serialization::geometry::Rectangle;
use crate::serialization::wayland::BufferFormat;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::WlSurfaceId;
use crate::sharding_compression::CompressorWorkerState;
use crate::sharding_compression::DecompressorWorkerState;
use crate::sharding_compression::ShardingCompressor;
use crate::sharding_compression::ShardingDecompressor;
use crate::sharding_compression::compress_channel_slice;
use crate::vec4u8::Vec4u8;

/// Architecture-specific 32-pixel block kernel.
pub(crate) trait FilterKernel {
    type ForwardCarry: Copy + Send;
    type InverseCarry: Copy + Send;
    const STREAM_ALIGN: usize;

    fn zero_forward() -> Self::ForwardCarry;

    fn zero_inverse() -> Self::InverseCarry;

    fn or_accum_is_zero(acc: Self::ForwardCarry) -> (bool, bool, bool, bool);

    fn forward_block<const HAS_ALPHA: bool, const ACCUM_OR: bool>(
        input: KnownSizeBufferPointer<Vec4u8, 32>,
        out0: &mut [u8; 32],
        out1: &mut [u8; 32],
        out2: &mut [u8; 32],
        out3: &mut [u8; 32],
        prev: Self::ForwardCarry,
        or_acc: &mut Self::ForwardCarry,
    ) -> Self::ForwardCarry;

    /// # Safety
    /// When `STREAM` is `true`, `out_blocks` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn splat_blocks<const STREAM: bool>(out_blocks: &mut [[Vec4u8; 32]], pixel: Vec4u8);

    /// # Safety
    /// When `STREAM` is `true`, `out` must be aligned to `Self::STREAM_ALIGN`.
    unsafe fn inverse_block<const STREAM: bool, const CONST_ALPHA: bool>(
        in0: &[u8; 32],
        in1: &[u8; 32],
        in2: &[u8; 32],
        in3: &[u8; 32],
        const_alpha: u8,
        out: &mut [Vec4u8; 32],
        prev: Self::InverseCarry,
    ) -> Self::InverseCarry;

    fn fence_stream();
}

// --- fused tile forward + compress ---

#[inline]
fn encode_channel(
    data: &[u8],
    tail_chunks_zero: bool,
    compressor: &mut Compressor<'static>,
    zstd_scratch: &mut Vec<u8>,
) -> TileChannel {
    if data.len() >= 32
        && data.len().is_multiple_of(32)
        && tail_chunks_zero
        && data[1..32] == [0u8; 31]
    {
        TileChannel::ConstantDelta(data[0])
    } else {
        compress_channel_slice(compressor, zstd_scratch, data)
    }
}

fn filter_tile<K: FilterKernel, const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    tile_idx: u32,
    state: &mut CompressorWorkerState,
) -> CompressedTile {
    let tile_len = aos.len();
    debug_assert!(tile_len > 0 && tile_len <= TILE_PIXELS);
    let lim = (tile_len / 32) * 32;
    let (aos_blocks, aos_rem) = aos.split_at(lim);

    let [s0_buf, s1_buf, s2_buf, s3_buf] = &mut state.soa_scratch.0;
    let s0 = &mut s0_buf[..tile_len];
    let s1 = &mut s1_buf[..tile_len];
    let s2 = &mut s2_buf[..tile_len];
    let s3 = &mut s3_buf[..tile_len];

    let mut or_acc = K::zero_forward();
    let mut prev = K::zero_forward();
    let mut aos_chunks = aos_blocks.array_chunks::<32>();
    let mut s0_chunks = s0.as_chunks_mut::<32>().0.iter_mut();
    let mut s1_chunks = s1.as_chunks_mut::<32>().0.iter_mut();
    let mut s2_chunks = s2.as_chunks_mut::<32>().0.iter_mut();
    let mut s3_chunks = s3.as_chunks_mut::<32>().0.iter_mut();

    if let Some(first_aos) = aos_chunks.next() {
        prev = K::forward_block::<HAS_ALPHA, false>(
            first_aos,
            s0_chunks.next().unwrap(),
            s1_chunks.next().unwrap(),
            s2_chunks.next().unwrap(),
            s3_chunks.next().unwrap(),
            prev,
            &mut or_acc,
        );
    }
    for (aos_c, s0_c, s1_c, s2_c, s3_c) in
        izip!(aos_chunks, s0_chunks, s1_chunks, s2_chunks, s3_chunks)
    {
        prev =
            K::forward_block::<HAS_ALPHA, true>(aos_c, s0_c, s1_c, s2_c, s3_c, prev, &mut or_acc);
    }

    for (i, px) in aos_rem.into_iter().enumerate() {
        s0[lim + i] = px.0;
        s1[lim + i] = px.1;
        s2[lim + i] = px.2;
        if HAS_ALPHA {
            s3[lim + i] = px.3;
        }
    }

    let (z0, z1, z2, z3) = K::or_accum_is_zero(or_acc);
    let compressor = &mut state.compressor;
    let zstd_scratch = &mut state.zstd_scratch;
    let ch0 = encode_channel(s0, z0, compressor, zstd_scratch);
    let ch1 = encode_channel(s1, z1, compressor, zstd_scratch);
    let ch2 = encode_channel(s2, z2, compressor, zstd_scratch);
    let ch3 = if HAS_ALPHA {
        encode_channel(s3, z3, compressor, zstd_scratch)
    } else {
        TileChannel::Omitted
    };

    CompressedTile {
        tile_idx,
        channels: [ch0, ch1, ch2, ch3],
    }
}

// --- fused tile decompress + unfilter ---

const ZERO_CHUNK: [u8; 32] = [0; 32];

enum ChannelChunks<'a> {
    Slice(slice::Iter<'a, [u8; 32]>),
    Constant {
        first_chunk: [u8; 32],
        emitted_first: bool,
    },
}

impl<'a> ChannelChunks<'a> {
    #[inline]
    fn new(data: &'a [u8], const_delta: Option<u8>) -> Self {
        match const_delta {
            Some(d) => {
                let mut first_chunk = [0u8; 32];
                first_chunk[0] = d;
                Self::Constant {
                    first_chunk,
                    emitted_first: false,
                }
            },
            None => Self::Slice(data.as_chunks::<32>().0.iter()),
        }
    }

    #[inline]
    fn next_chunk(&mut self) -> &[u8; 32] {
        match self {
            Self::Slice(iter) => iter.next().unwrap(),
            Self::Constant {
                first_chunk,
                emitted_first,
            } => {
                if !*emitted_first {
                    *emitted_first = true;
                    first_chunk
                } else {
                    &ZERO_CHUNK
                }
            },
        }
    }
}

/// # Safety
/// When `STREAM` is `true`, `aos` must be aligned to `K::STREAM_ALIGN`.
unsafe fn unfilter_block<K: FilterKernel, const STREAM: bool, const HAS_ALPHA: bool>(
    soa0: &[u8],
    soa1: &[u8],
    soa2: &[u8],
    soa3: &[u8],
    c0: Option<u8>,
    c1: Option<u8>,
    c2: Option<u8>,
    c3: Option<u8>,
    aos: &mut [Vec4u8],
) {
    const { assert!(size_of::<[Vec4u8; 32]>().is_multiple_of(K::STREAM_ALIGN)) };
    let out_chunks = aos.as_chunks_mut::<32>().0;

    if let (Some(d0), Some(d1), Some(d2)) = (c0, c1, c2)
        && (!HAS_ALPHA || c3.is_some())
    {
        let a = if HAS_ALPHA { c3.unwrap() } else { 0xFF };
        let pixel = Vec4u8(d0.wrapping_add(d1), d1, d2.wrapping_add(d1), a);
        // SAFETY: Caller guarantees `out_chunks` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
        unsafe { K::splat_blocks::<STREAM>(out_chunks, pixel) };
        return;
    }

    let mut prev = K::zero_inverse();
    if !HAS_ALPHA || c3.is_some() {
        let const_alpha = if HAS_ALPHA { c3.unwrap() } else { 0xFF };

        if c0.is_none() && c1.is_none() && c2.is_none() {
            for (s0, s1, s2, out) in izip!(
                soa0.as_chunks::<32>().0,
                soa1.as_chunks::<32>().0,
                soa2.as_chunks::<32>().0,
                out_chunks,
            ) {
                // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
                prev = unsafe {
                    K::inverse_block::<STREAM, true>(
                        s0,
                        s1,
                        s2,
                        &ZERO_CHUNK,
                        const_alpha,
                        out,
                        prev,
                    )
                };
            }
        } else {
            let mut ch0 = ChannelChunks::new(soa0, c0);
            let mut ch1 = ChannelChunks::new(soa1, c1);
            let mut ch2 = ChannelChunks::new(soa2, c2);
            for out in out_chunks {
                // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
                prev = unsafe {
                    K::inverse_block::<STREAM, true>(
                        ch0.next_chunk(),
                        ch1.next_chunk(),
                        ch2.next_chunk(),
                        &ZERO_CHUNK,
                        const_alpha,
                        out,
                        prev,
                    )
                };
            }
        }
    } else if c0.is_none() && c1.is_none() && c2.is_none() {
        for (s0, s1, s2, s3, out) in izip!(
            soa0.as_chunks::<32>().0,
            soa1.as_chunks::<32>().0,
            soa2.as_chunks::<32>().0,
            soa3.as_chunks::<32>().0,
            out_chunks,
        ) {
            // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
            prev = unsafe { K::inverse_block::<STREAM, false>(s0, s1, s2, s3, 0, out, prev) };
        }
    } else {
        let mut ch0 = ChannelChunks::new(soa0, c0);
        let mut ch1 = ChannelChunks::new(soa1, c1);
        let mut ch2 = ChannelChunks::new(soa2, c2);
        for (s3, out) in izip!(soa3.as_chunks::<32>().0, out_chunks) {
            // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
            prev = unsafe {
                K::inverse_block::<STREAM, false>(
                    ch0.next_chunk(),
                    ch1.next_chunk(),
                    ch2.next_chunk(),
                    s3,
                    0,
                    out,
                    prev,
                )
            };
        }
    }
}

#[inline]
fn decode_channel<'a>(
    channel: &'a TileChannel,
    tile_len: usize,
    scratch: &'a mut [u8; TILE_PIXELS],
    decompressor: &mut Decompressor<'static>,
) -> Result<(&'a [u8], Option<u8>)> {
    match channel {
        TileChannel::ConstantDelta(d) => {
            if tile_len < 32 || !tile_len.is_multiple_of(32) {
                bail!("ConstantDelta channel invalid for tile_len {tile_len}");
            }
            Ok((&[][..], Some(*d)))
        },
        TileChannel::Uncompressed(data) => {
            if data.len() != tile_len {
                bail!("uncompressed channel len {} != {tile_len}", data.len());
            }
            Ok((data.as_slice(), None))
        },
        TileChannel::Zstd(data) => {
            let out = &mut scratch[..tile_len];
            let written = decompressor
                .decompress_to_buffer(data, out)
                .location(loc!())?;
            if written != tile_len {
                bail!("decompressed channel len {written} != {tile_len}");
            }
            Ok((out, None))
        },
        TileChannel::Omitted => bail!("unexpected Omitted channel on active plane"),
    }
}

fn unfilter_tile<K: FilterKernel, const HAS_ALPHA: bool>(
    tile: &CompressedTile,
    aos: &mut [Vec4u8],
    state: &mut DecompressorWorkerState,
) -> Result<bool> {
    let tile_len = aos.len();
    debug_assert!(tile_len > 0 && tile_len <= TILE_PIXELS);
    let [buf0, buf1, buf2, buf3] = &mut state.soa_scratch.0;

    let (s0, c0) = decode_channel(&tile.channels[0], tile_len, buf0, &mut state.decompressor)
        .location(loc!())?;
    let (s1, c1) = decode_channel(&tile.channels[1], tile_len, buf1, &mut state.decompressor)
        .location(loc!())?;
    let (s2, c2) = decode_channel(&tile.channels[2], tile_len, buf2, &mut state.decompressor)
        .location(loc!())?;
    let (s3, c3) = if HAS_ALPHA {
        decode_channel(&tile.channels[3], tile_len, buf3, &mut state.decompressor)
            .location(loc!())?
    } else {
        (&[][..], None)
    };

    let lim = (tile_len / 32) * 32;
    let stream = (aos.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN);

    if stream {
        // SAFETY: `stream` verified `aos.as_mut_ptr()` is aligned to
        // `K::STREAM_ALIGN`.
        unsafe {
            unfilter_block::<K, true, HAS_ALPHA>(s0, s1, s2, s3, c0, c1, c2, c3, aos);
        }
    } else {
        // SAFETY: `STREAM` is `false`, imposing no alignment requirement beyond `Vec4u8`.
        unsafe {
            unfilter_block::<K, false, HAS_ALPHA>(s0, s1, s2, s3, c0, c1, c2, c3, aos);
        }
    }

    for i in lim..tile_len {
        let a = if HAS_ALPHA { s3[i] } else { 0xFF };
        aos[i] = Vec4u8(s0[i], s1[i], s2[i], a);
    }

    Ok(stream && lim > 0)
}

// --- public API ---

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

/// Validates that `bands` are sorted and non-overlapping, each band is
/// non-empty (`y0 < y1`), and `y1 <= metadata.height` (an empty `bands` slice
/// is valid), then splits them into `<= TILE_PIXELS` pixel spans
/// `(start_pixel, len_pixels)`. `metadata` must already pass
/// `BufferMetadata::validate`.
fn tile_spans_for_bands(
    bands: &[RowBand],
    metadata: &BufferMetadata,
) -> Result<Vec<(usize, usize)>> {
    let row_pixels = (metadata.stride as usize) / 4;
    let height_u32 = metadata.height as u32;
    let mut spans = Vec::new();
    let mut prev_y1 = 0u32;

    for band in bands {
        if band.y0 < prev_y1 || band.y0 >= band.y1 || band.y1 > height_u32 {
            bail!(
                "invalid or overlapping RowBand [{}..{}) for height {height_u32}",
                band.y0,
                band.y1
            );
        }
        prev_y1 = band.y1;
        let band_start = (band.y0 as usize) * row_pixels;
        let band_end = (band.y1 as usize) * row_pixels;
        let mut cur = band_start;
        while cur < band_end {
            let len = (band_end - cur).min(TILE_PIXELS);
            spans.push((cur, len));
            cur += len;
        }
    }

    Ok(spans)
}

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
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };
    let total_pixels = aos.len();
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();

    let spans_opt = bands
        .as_ref()
        .map(|b| tile_spans_for_bands(b, &metadata).expect("server generated invalid row bands"));
    let n_tiles = match &spans_opt {
        None => total_pixels.div_ceil(TILE_PIXELS),
        Some(spans) => spans.len(),
    };

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
            let (start, tile_len) = match &spans_opt {
                None => {
                    let start = tile_idx * TILE_PIXELS;
                    (start, (total_pixels - start).min(TILE_PIXELS))
                },
                Some(spans) => spans[tile_idx],
            };
            let (_, after_start) = aos.split_at(start);
            let (aos_tile, _) = after_start.split_at(tile_len);

            let tile = if has_alpha {
                filter_tile::<arch::Kernel, true>(aos_tile, tile_idx as u32, state)
            } else {
                filter_tile::<arch::Kernel, false>(aos_tile, tile_idx as u32, state)
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
    format: BufferFormat,
    compressor: &mut ShardingCompressor,
) -> CompressedTiles {
    assert!(data.len().is_multiple_of(4));
    let stride = i32::try_from(data.len()).expect("buffer byte length exceeds i32::MAX");
    let total_pixels = data.len() / 4;
    let metadata = BufferMetadata {
        width: total_pixels as i32,
        height: 1,
        stride,
        format,
    };
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
    tiles.sort_by_key(|t| t.tile_idx);
    CompressedTiles {
        format,
        total_pixels,
        tiles,
    }
}

fn unfilter_tiles_parallel<'a>(
    decompressor: &mut ShardingDecompressor,
    total_pixels: usize,
    spans: Option<&[(usize, usize)]>,
    has_alpha: bool,
    output_buf: &mut [u8],
    next_tile: impl Fn() -> Option<Result<Cow<'a, CompressedTile>>> + Sync,
) -> Result<()> {
    let byte_len = total_pixels
        .checked_mul(4)
        .context(loc!(), "pixel count overflow")?;
    if output_buf.len() < byte_len {
        bail!("output_buf len {} < {byte_len}", output_buf.len());
    }

    let expected_tiles = spans.map_or_else(|| total_pixels.div_ceil(TILE_PIXELS), <[_]>::len);
    let aos: &mut [Vec4u8] = bytemuck::cast_slice_mut(&mut output_buf[..byte_len]);
    let aos_base = aos.as_mut_ptr() as usize;
    let claimed: Vec<AtomicBool> = (0..expected_tiles)
        .map(|_| AtomicBool::new(false))
        .collect();
    let completed = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    decompressor.pool.execute(expected_tiles, |state| {
        let mut streamed_any = false;
        while let Some(item) = next_tile() {
            let res = (|| -> Result<bool> {
                let tile = item?;
                let idx = tile.tile_idx as usize;
                if idx >= expected_tiles {
                    bail!("tile_idx {idx} >= {expected_tiles}");
                }
                if claimed[idx].swap(true, Ordering::Relaxed) {
                    bail!("duplicate tile_idx {idx}");
                }
                let (start, tile_len) = match spans {
                    None => {
                        let start = idx * TILE_PIXELS;
                        (start, (total_pixels - start).min(TILE_PIXELS))
                    },
                    Some(spans) => spans[idx],
                };
                // SAFETY: `spans` (when `Some`, produced by `tile_spans_for_bands`
                // over the same validated `metadata` that yielded `total_pixels`)
                // or fixed `TILE_PIXELS` chunks are disjoint ranges inside
                // `0..total_pixels`, and `claimed[idx].swap(true)` guarantees
                // exclusive access to `[start .. start + tile_len)` within `aos`
                // for the duration of `pool.execute`.
                let aos_tile = unsafe {
                    slice::from_raw_parts_mut((aos_base as *mut Vec4u8).add(start), tile_len)
                };
                if has_alpha {
                    unfilter_tile::<arch::Kernel, true>(&tile, aos_tile, state)
                } else {
                    unfilter_tile::<arch::Kernel, false>(&tile, aos_tile, state)
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
    total_pixels: usize,
    spans: Option<Vec<(usize, usize)>>,
}

impl ValidatedTileStream {
    pub fn new(stream: CompressedTileStream) -> Result<Self> {
        let total_pixels = stream.header.metadata.validate().location(loc!())?;
        let spans = match &stream.header.bands {
            None => None,
            Some(bands) => {
                Some(tile_spans_for_bands(bands, &stream.header.metadata).location(loc!())?)
            },
        };
        let expected_tiles = match &spans {
            None => total_pixels.div_ceil(TILE_PIXELS),
            Some(spans) => spans.len(),
        };
        if stream.header.n_tiles as usize != expected_tiles {
            bail!(
                "header n_tiles {} != expected {expected_tiles}",
                stream.header.n_tiles,
            );
        }
        Ok(Self {
            stream,
            total_pixels,
            spans,
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
    let has_alpha = validated.stream.header.metadata.format == BufferFormat::Argb8888;
    let tiles_rx = &validated.stream.tiles;
    unfilter_tiles_parallel(
        decompressor,
        validated.total_pixels,
        validated.spans.as_deref(),
        has_alpha,
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
    let total_pixels = tiles.total_pixels;
    let n_tiles = tiles.tiles.len();
    let has_alpha = tiles.format == BufferFormat::Argb8888;
    let next_idx = AtomicUsize::new(0);
    unfilter_tiles_parallel(
        decompressor,
        total_pixels,
        None,
        has_alpha,
        output_buf,
        || {
            let i = next_idx.fetch_add(1, Ordering::Relaxed);
            (i < n_tiles).then(|| Ok(Cow::Borrowed(&tiles.tiles[i])))
        },
    )
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

    fn materialize_channel(ch: &TileChannel, len: usize) -> Vec<u8> {
        match ch {
            TileChannel::ConstantDelta(first_byte) => {
                let mut out = vec![0u8; len];
                if len > 0 {
                    out[0] = *first_byte;
                }
                out
            },
            TileChannel::Uncompressed(data) => data.clone(),
            TileChannel::Zstd(data) => zstd::bulk::decompress(data, len).unwrap(),
            TileChannel::Omitted => panic!("unexpected omitted channel"),
        }
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

        // Poison the buffer before the misaligned (`STREAM = false` on x86,
        // `STREAM_ALIGN = 32`) decode so 4-byte-periodic solid tiles cannot
        // pass by reading stale bytes left behind by the aligned decode.
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

    fn test_roundtrip_impl(
        data: &[u8],
        compressor: &mut ShardingCompressor,
        decompressor: &mut ShardingDecompressor,
    ) {
        assert!(data.len().is_multiple_of(4));

        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        let tiles = filter_and_compress(buf_ptr, BufferFormat::Argb8888, compressor);
        assert_decodes_at_both_alignments(&tiles, data, decompressor);
    }

    fn test_roundtrip_xrgb_impl(
        data: &[u8],
        compressor: &mut ShardingCompressor,
        decompressor: &mut ShardingDecompressor,
    ) {
        assert!(data.len().is_multiple_of(4));

        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        let tiles = filter_and_compress(buf_ptr, BufferFormat::Xrgb8888, compressor);
        assert!(
            tiles
                .tiles
                .iter()
                .all(|t| t.channels[3] == TileChannel::Omitted)
        );

        let mut expected = data.to_vec();
        for px in expected.as_chunks_mut::<4>().0 {
            px[3] = 0xFF;
        }
        assert_decodes_at_both_alignments(&tiles, &expected, decompressor);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_roundtrip() {
        with_test_codec(|compressor, decompressor| {
            for n in [
                0,
                4,
                8,
                12,
                16,
                20,
                24,
                28,
                32,
                36,
                120,
                124,
                128,
                132,
                248,
                252,
                256,
                260,
                1016,
                1020,
                1024,
                1028,
                2040,
                2044,
                2048,
                2052,
                100,
                1920 * 1080,
                32768 * 4 + 4,
                TILE_PIXELS * 4 + 4,
                1008 * 9513 * 4,
                1008 * 951 * 4,
            ] {
                let v = test_vec(n);
                test_roundtrip_impl(&v, compressor, decompressor);
                test_roundtrip_xrgb_impl(&v, compressor, decompressor);
            }

            // Multi-tile solid-color roundtrips exercising `splat_blocks` and
            // constant-value tiles/tails.
            let solid: Vec<u8> = [0x20u8, 0x70, 0xD0, 0xFF]
                .into_iter()
                .cycle()
                .take(TILE_PIXELS * 4 * 6 + 20)
                .collect();
            test_roundtrip_impl(&solid, compressor, decompressor);
            test_roundtrip_xrgb_impl(&solid, compressor, decompressor);

            // Translucent solid (`alpha = 0x80 != 0xFF`) + constant-value tail
            // `tile_len = 36` (`36 >= 32` but not a multiple of 32, failing
            // `encode_channel`'s `data.len().is_multiple_of(32)` guard).
            let translucent_solid: Vec<u8> = [0x20u8, 0x70, 0xD0, 0x80]
                .into_iter()
                .cycle()
                .take(TILE_PIXELS * 4 + 36 * 4)
                .collect();
            test_roundtrip_impl(&translucent_solid, compressor, decompressor);
            test_roundtrip_xrgb_impl(&translucent_solid, compressor, decompressor);

            // Exhaustively cover all 16 `{Constant, Slice}` channel combinations
            // (`bit 0 = B-G`, `bit 1 = G`, `bit 2 = R-G`, `bit 3 = A`; set bit
            // yields a constant non-zero delta, clear bit yields a varying slice).
            // Set `px[1]` (G) before `px[0]`/`px[2]` so `B - G` and `R - G` stay
            // constant even when G varies.
            for mask in 0u8..16 {
                let mut data = test_vec(TILE_PIXELS * 4 + 128);
                for px in data.as_chunks_mut::<4>().0 {
                    if mask & 2 != 0 {
                        px[1] = 77;
                    }
                    if mask & 1 != 0 {
                        px[0] = px[1].wrapping_add(9);
                    }
                    if mask & 4 != 0 {
                        px[2] = px[1].wrapping_add(123);
                    }
                    if mask & 8 != 0 {
                        px[3] = 0xCC;
                    }
                }
                test_roundtrip_impl(&data, compressor, decompressor);
                if mask & 8 == 0 {
                    test_roundtrip_xrgb_impl(&data, compressor, decompressor);
                }
            }

            // Flat first 64 pixels followed by varying pixels: exercises
            // `HAS_ALPHA = false` (`Xrgb8888`) `ACCUM_OR` accumulation on c0..c2.
            let mut late_vary = [0x20u8, 0x70, 0xD0, 0xFF].repeat(64);
            late_vary.extend(test_vec(64 * 4));
            test_roundtrip_xrgb_impl(&late_vary, compressor, decompressor);
        });
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn proptest_roundtrip(mut arr in proptest::collection::vec(0..=u8::MAX, 0..1_000_000)) {
            arr.truncate((arr.len() / 4) * 4);
            assert!(arr.len() % 4 == 0);
            with_test_codec(|compressor, decompressor| {
                test_roundtrip_impl(&arr, compressor, decompressor);
            });
        }
    }

    /// Verify that the filtered SoA output matches known-good values across
    /// multiple 32-pixel SIMD blocks (including nonlinear `subtract_green`,
    /// block-boundary `ForwardCarry`, and both 128-bit lanes of `ACCUM_OR` tail
    /// blocks). The filter math (`wrapping_sub` decorrelation + prefix-difference)
    /// is byte-exact across all x86 SIMD tiers and aarch64 NEON for
    /// cross-architecture wire compatibility.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filtered_output_deterministic() {
        let n = 96usize; // 3 SIMD blocks of 32 pixels
        let check_filtered_against_reference =
            |data: &[u8], compressor: &mut ShardingCompressor, label: &str| {
                let n = data.len() / 4;
                let lim = (n / 32) * 32;
                let data_ptr = data.as_ptr();
                let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

                let mut tiles = filter_and_compress(buf_ptr, BufferFormat::Argb8888, compressor);
                assert_eq!(tiles.tiles.len(), 1);
                let tile = tiles.tiles.pop().unwrap();

                let mut expected = [vec![0u8; n], vec![0u8; n], vec![0u8; n], vec![0u8; n]];
                for i in 0..n {
                    expected[0][i] = data[i * 4];
                    expected[1][i] = data[i * 4 + 1];
                    expected[2][i] = data[i * 4 + 2];
                    expected[3][i] = data[i * 4 + 3];
                }
                let [ch0, ch1, ch2, ..] = &mut expected;
                for (b, (g, r)) in ch0[..lim]
                    .iter_mut()
                    .zip(ch1[..lim].iter().zip(ch2[..lim].iter_mut()))
                {
                    *b = b.wrapping_sub(*g);
                    *r = r.wrapping_sub(*g);
                }
                for ch in &mut expected {
                    let mut prev = 0u8;
                    for v in &mut ch[..lim] {
                        let orig = *v;
                        *v = orig.wrapping_sub(prev);
                        prev = orig;
                    }
                }

                for (c, exp) in expected.iter().enumerate() {
                    let actual = materialize_channel(&tile.channels[c], n);
                    assert_eq!(&actual, exp, "channel {c} mismatch ({label})");
                }
                tile
            };

        with_test_codec(|compressor, _| {
            let mut poly = vec![0u8; n * 4];
            for i in 0..n {
                let k = i as u32;
                poly[i * 4] = ((k * k * 3 + 19) & 0xFF) as u8;
                poly[i * 4 + 1] = ((k * 17 + (k >> 2) * 41 + 5) & 0xFF) as u8;
                poly[i * 4 + 2] = ((k * k * 11 + k * 7 + 200) & 0xFF) as u8;
                poly[i * 4 + 3] = 0xFF;
            }
            let poly_tile = check_filtered_against_reference(&poly, compressor, "poly");
            assert_eq!(poly_tile.channels[3], TileChannel::ConstantDelta(0xFF));

            let base = [0x12u8, 0x34, 0x56, 0xFF];
            let expected_const_first = [
                base[0].wrapping_sub(base[1]),
                base[1],
                base[2].wrapping_sub(base[1]),
                base[3],
            ];
            // Exact `data.len() == 32` minimum and non-multiple-of-32 (`36`) boundaries
            // of `encode_channel`'s `ConstantDelta` guard.
            let tile_32 = check_filtered_against_reference(&base.repeat(32), compressor, "len=32");
            for (c, &first_byte) in expected_const_first.iter().enumerate() {
                assert_eq!(tile_32.channels[c], TileChannel::ConstantDelta(first_byte));
            }
            // For `len=36`, the 4-pixel raw non-zero tail mismatches a materialized
            // `ConstantDelta` (whose bytes `1..36` are zero), so
            // `check_filtered_against_reference` catches a missing
            // `is_multiple_of(32)` guard without coupling to `MIN_SIZE_TO_COMPRESS`.
            check_filtered_against_reference(&base.repeat(36), compressor, "len=36");
            // Step perturbation (`idx..n` flipped) leaves a single non-zero delta at
            // `idx` without changing `data[0]`: `1`/`31` bound `encode_channel`'s
            // first-block `data[1..32]` check, `32`/`48` hit the lo/hi 16-byte halves
            // of the second SIMD block (`ACCUM_OR`), and `95` is the last byte.
            let cases = std::iter::once((0usize, None))
                .chain((0..4).flat_map(|ch| [1usize, 31, 32, 48, 95].map(|idx| (ch, Some(idx)))));
            for (perturb_ch, perturb_idx) in cases {
                let mut data = base.repeat(n);
                if let Some(idx) = perturb_idx {
                    for i in idx..n {
                        data[i * 4 + perturb_ch] ^= 1;
                    }
                }
                let label = format!("ch={perturb_ch},idx={perturb_idx:?}");
                let tile = check_filtered_against_reference(&data, compressor, &label);
                for (c, &first_byte) in expected_const_first.iter().enumerate() {
                    // Perturbing G (`perturb_ch == 1`) also perturbs B (`0`) and R (`2`)
                    // via `subtract_green`.
                    let expect_const = perturb_idx.is_none()
                        || (c != perturb_ch && !(perturb_ch == 1 && (c == 0 || c == 2)));
                    assert_eq!(
                        tile.channels[c] == TileChannel::ConstantDelta(first_byte),
                        expect_const,
                        "unexpected ConstantDelta for channel {c} ({label}): {:?}",
                        tile.channels[c]
                    );
                }
            }
        });
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
        // 75% threshold: 1620 / 2160 == 75% -> full frame; 1619 / 2160 < 75% -> partial.
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, 100, 1620)], 3840, 2160),
            None
        );
        assert_eq!(
            compute_partial_row_bands(&[Rectangle::new(0, 0, 100, 1619)], 3840, 2160),
            Some(vec![RowBand { y0: 0, y1: 1619 }])
        );
        // Out-of-order, overlapping, adjacent, bottom-touching (`0 < y0 < y1 == height`),
        // and negative-coord rects merge cleanly.
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
        // Off-screen, edge-touching (`x1 == 0`, `x0 == width`, `y1 == 0`, `y0 == height`),
        // and non-positive-dimension rects yield an empty partial update (`0` damaged rows).
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
    fn test_tile_spans_for_bands() {
        let small_meta = BufferMetadata {
            width: 256,
            height: 20,
            stride: 256 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let spans = tile_spans_for_bands(&[RowBand { y0: 2, y1: 12 }], &small_meta).unwrap();
        assert_eq!(spans, vec![(2 * 256, 10 * 256)]);

        let large_meta = BufferMetadata {
            width: 512,
            height: 400,
            stride: 512 * 4,
            format: BufferFormat::Xrgb8888,
        };
        let multi_tile_spans =
            tile_spans_for_bands(&[RowBand { y0: 0, y1: 300 }], &large_meta).unwrap();
        assert_eq!(
            multi_tile_spans,
            vec![
                (0, TILE_PIXELS),
                (TILE_PIXELS, TILE_PIXELS),
                (2 * TILE_PIXELS, 300 * 512 - 2 * TILE_PIXELS),
            ]
        );

        let meta_64 = BufferMetadata {
            width: 64,
            height: 64,
            stride: 64 * 4,
            format: BufferFormat::Xrgb8888,
        };
        // `y1 == height` boundary is accepted.
        assert_eq!(
            tile_spans_for_bands(&[RowBand { y0: 60, y1: 64 }], &meta_64).unwrap(),
            vec![(60 * 64, 4 * 64)]
        );
        // Non-monotonic (`y0 == y1`, `y0 > y1`), overlapping/unsorted, and `y1 > height`.
        for bad_bands in [
            vec![RowBand { y0: 5, y1: 5 }],
            vec![RowBand { y0: 10, y1: 5 }],
            vec![RowBand { y0: 10, y1: 20 }, RowBand { y0: 15, y1: 25 }],
            vec![RowBand { y0: 50, y1: 70 }],
        ] {
            assert!(tile_spans_for_bands(&bad_bands, &meta_64).is_err());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_partial_row_bands_streaming_roundtrip() {
        with_test_codec(|compressor, decompressor| {
            // Parameterized over format, padded stride, full-frame (`None`),
            // empty partial (`Some(vec![])`), small `<= 512 B` Uncompressed
            // band, and multi-tile Zstd bands.
            for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
                for pad_bytes in [0i32, 16i32] {
                    let width = 480i32;
                    let height = 600i32;
                    let stride = width * 4 + pad_bytes;
                    let len = (stride * height) as usize;
                    let mut frame0 = test_vec(len);
                    // Keep byte 3 strictly `< 0xFF` across all `bands_case`s so
                    // `Xrgb8888` alpha overwrite (`0xFF`) is verified on partial
                    // bands and `rem > 0` scalar tails too.
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

                    // Odd row `y = 3` (`3 * 1936 == 16 mod 32` when `pad_bytes = 16`)
                    // + even row `y = 300` (`0 mod 32`) on a 64-byte-aligned canvas
                    // exercises both `STREAM = false` and `STREAM = true` on x86
                    // (`STREAM_ALIGN = 32`).
                    let partial_bands = compute_partial_row_bands(
                        &[
                            Rectangle::new(0, 3, width, 1), // <= 512 B Uncompressed tile
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

                        let expected = match &bands_case {
                            None => frame1_expected.clone(),
                            Some(bands) => {
                                let mut exp = frame0.clone();
                                for b in bands {
                                    let s = b.y0 as usize * stride as usize;
                                    let e = b.y1 as usize * stride as usize;
                                    exp[s..e].copy_from_slice(&frame1_expected[s..e]);
                                }
                                exp
                            },
                        };
                        assert_eq!(&client_canvas[..], expected.as_slice());
                    }
                }
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_pipeline_errors() {
        with_test_codec(|compressor, decompressor| {
            // Malformed wire error cases must return `Err` cleanly rather than panicking.
            let bad_const_delta = CompressedTiles {
                format: BufferFormat::Argb8888,
                total_pixels: 50, // not a multiple of 32
                tiles: vec![CompressedTile {
                    tile_idx: 0,
                    channels: [
                        TileChannel::ConstantDelta(1),
                        TileChannel::ConstantDelta(2),
                        TileChannel::ConstantDelta(3),
                        TileChannel::ConstantDelta(4),
                    ],
                }],
            };
            let mut out = vec![0u8; 50 * 4];
            assert!(decompress_and_unfilter(&bad_const_delta, decompressor, &mut out).is_err());

            for bad_ch0 in [
                TileChannel::Omitted,
                TileChannel::Uncompressed(vec![0u8; 31]),
                TileChannel::Uncompressed(vec![0u8; 33]),
                TileChannel::Zstd(zstd::bulk::compress(&[0u8; 16], 1).unwrap()),
                TileChannel::Zstd(vec![0xFF, 0x00, 0x01]),
            ] {
                let bad_tile = CompressedTiles {
                    format: BufferFormat::Argb8888,
                    total_pixels: 32,
                    tiles: vec![CompressedTile {
                        tile_idx: 0,
                        channels: [
                            bad_ch0,
                            TileChannel::ConstantDelta(2),
                            TileChannel::ConstantDelta(3),
                            TileChannel::ConstantDelta(4),
                        ],
                    }],
                };
                assert!(
                    decompress_and_unfilter(&bad_tile, decompressor, &mut out[..32 * 4]).is_err()
                );
            }

            let two_tile_pixels = TILE_PIXELS + 64;
            let two_tile_data = test_vec(two_tile_pixels * 4);
            let two_tile_ptr = two_tile_data.as_ptr();
            let two_tile_buf = unsafe { BufferPointer::new(&two_tile_ptr, two_tile_data.len()) };

            // Dropping `StreamingCompressedTiles` inside `on_stream_ready` (e.g. on
            // client disconnect) must let workers exit cleanly and leave the pool reusable.
            filter_and_compress_streaming(
                two_tile_buf,
                ClientId(1),
                WlSurfaceId(1),
                BufferMetadata {
                    width: two_tile_pixels as i32,
                    height: 1,
                    stride: (two_tile_pixels * 4) as i32,
                    format: BufferFormat::Argb8888,
                },
                None,
                compressor,
                drop,
            );

            // Out-of-order tile arrival (`[tile_1, tile_0]`) must unfilter identically to
            // `[tile_0, tile_1]`.
            let mut two_tiles =
                filter_and_compress(two_tile_buf, BufferFormat::Argb8888, compressor);
            assert_eq!(two_tiles.tiles.len(), 2);
            two_tiles.tiles.reverse();
            assert_eq!(two_tiles.tiles[0].tile_idx, 1);
            assert_eq!(two_tiles.tiles[1].tile_idx, 0);
            let mut reversed_out = vec![0u8; two_tile_data.len()];
            decompress_and_unfilter(&two_tiles, decompressor, &mut reversed_out).unwrap();
            assert_eq!(reversed_out, two_tile_data);

            // Missing tiles (`done < expected_tiles`).
            for bad_tiles in [
                CompressedTiles {
                    format: BufferFormat::Argb8888,
                    total_pixels: two_tile_pixels,
                    tiles: Vec::new(),
                },
                CompressedTiles {
                    format: BufferFormat::Argb8888,
                    total_pixels: two_tile_pixels,
                    tiles: vec![two_tiles.tiles[1].clone()],
                },
            ] {
                assert!(
                    decompress_and_unfilter(&bad_tiles, decompressor, &mut reversed_out).is_err()
                );
            }

            // Undersized `output_buf`.
            let mut short_out = vec![0u8; two_tile_data.len() - 4];
            assert!(decompress_and_unfilter(&two_tiles, decompressor, &mut short_out).is_err());

            // Out-of-range `tile_idx >= expected_tiles`.
            let mut oob_idx = two_tiles.clone();
            oob_idx.tiles[0].tile_idx = 2;
            assert!(decompress_and_unfilter(&oob_idx, decompressor, &mut reversed_out).is_err());

            // Duplicate `tile_idx = 0` on a 2-tile frame.
            let mut dup_idx = two_tiles.clone();
            dup_idx.tiles[0] = dup_idx.tiles[1].clone();
            assert_eq!(dup_idx.tiles[0].tile_idx, 0);
            assert_eq!(dup_idx.tiles[1].tile_idx, 0);
            assert!(decompress_and_unfilter(&dup_idx, decompressor, &mut reversed_out).is_err());

            // Mismatched `header.n_tiles` must fail header validation cleanly.
            for bad_n_tiles in [0, 2] {
                let (_tx, rx) = crossbeam_channel::unbounded();
                let mismatched_n_tiles_stream = CompressedTileStream {
                    header: RawBufferHeader {
                        client: ClientId(1),
                        surface: WlSurfaceId(1),
                        metadata: BufferMetadata {
                            width: 64,
                            height: 64,
                            stride: 64 * 4,
                            format: BufferFormat::Argb8888,
                        },
                        n_tiles: bad_n_tiles,
                        bands: None,
                    },
                    tiles: rx,
                };
                assert!(
                    ValidatedTileStream::new(mismatched_n_tiles_stream).is_err(),
                    "expected error for mismatched n_tiles={bad_n_tiles}"
                );
            }
        });
    }
}
