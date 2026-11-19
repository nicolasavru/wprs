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
//! The pipeline transposes pixel data between array-of-structs (RGBA
//! interleaved) and struct-of-arrays (channels separated), applies
//! subtract_green (VP8L color decorrelation) and running_difference (delta
//! encoding) for improved entropy, then feeds the result to zstd.
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

/// Bytes per compression shard fed to zstd. Balances parallelism granularity
/// against zstd context-window utilisation.
const COMPRESSION_BLOCK_SIZE: usize = 128 * 1024;
const N_FILTER_THREADS: usize = 4;

use std::cmp;
use std::mem;
use std::slice;

use itertools::izip;
use lagoon::ThreadPool;

use crate::buffer_pointer::BufferPointer;
use crate::buffer_pointer::KnownSizeBufferPointer;
use crate::prelude::*;
use crate::serialization::wayland::BufferFormat;
use crate::sharding_compression::CompressedShards;
use crate::sharding_compression::DecompressedShards;
use crate::sharding_compression::ShardingCompressor;
use crate::sharding_compression::ShardingCompressorSink;
use crate::sharding_compression::StreamingCompressedShards;
use crate::vec4u8::Vec4u8;

/// Architecture-specific 32-pixel block kernel.
pub(crate) trait FilterKernel {
    type ForwardCarry: Copy + Send;
    type InverseCarry: Copy + Send;
    const STREAM_ALIGN: usize;

    fn zero_forward() -> Self::ForwardCarry;

    fn zero_inverse() -> (
        Self::InverseCarry,
        Self::InverseCarry,
        Self::InverseCarry,
        Self::InverseCarry,
    );

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

    fn advance_inverse_carry(carry: Self::InverseCarry, delta: u8) -> (Self::InverseCarry, u8);

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
        prev: (
            Self::InverseCarry,
            Self::InverseCarry,
            Self::InverseCarry,
            Self::InverseCarry,
        ),
    ) -> (
        Self::InverseCarry,
        Self::InverseCarry,
        Self::InverseCarry,
        Self::InverseCarry,
    );

    fn fence_stream();
}

// --- shared parallel pipeline ---

fn shard_indices_and_chunk_size(len: usize, n_channels: usize) -> (Vec<usize>, usize) {
    if len == 0 {
        return (Vec::new(), 32);
    }

    let n_blocks = len / 32;
    let lim = n_blocks * 32;
    let rem = len % 32;
    let blocks_per_thread = cmp::max(n_blocks / N_FILTER_THREADS, 1);
    let thread_chunk_size = blocks_per_thread * 32;

    let mut base_offsets = Vec::new();
    let mut thread_start = 0;
    while thread_start < lim {
        let thread_end = cmp::min(thread_start + thread_chunk_size, lim);
        let mut idx = thread_start;
        while idx < thread_end {
            base_offsets.push(idx);
            idx += cmp::min(thread_end - idx, COMPRESSION_BLOCK_SIZE);
        }
        thread_start = thread_end;
    }
    if rem > 0 {
        base_offsets.push(lim);
    }

    let mut indices = Vec::with_capacity(base_offsets.len() * n_channels);
    for ch in 0..n_channels {
        let ch_offset = ch * len;
        indices.extend(base_offsets.iter().map(|&idx| idx + ch_offset));
    }
    (indices, thread_chunk_size)
}

fn emit_or_reuse_channel_shard(
    compressor: &ShardingCompressorSink,
    shard_idx: usize,
    soa_len: usize,
    tail_chunks_zero: bool,
    buf: &mut Vec<u8>,
) {
    if tail_chunks_zero && buf[1..32] == [0u8; 31] {
        compressor.send_constant_delta_shard(shard_idx, soa_len, buf[0]);
    } else {
        compressor.compress_shard(shard_idx, mem::take(buf));
    }
}

#[instrument(skip_all, level = "debug")]
fn aos_to_soa_parallel_compression<K: FilterKernel, const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    thread_chunk_size: usize,
    compressor: ShardingCompressorSink,
) {
    if aos.is_empty() {
        return;
    }

    let len = aos.len();
    let lim = (len / 32) * 32;
    let rem = len % 32;
    let (aos_to_lim, aos_remainder) = aos.split_at(lim);

    debug_span!("aos_to_soa_u8_32x4_loop").in_scope(|| {
        ThreadPool::global().scoped(|s| {
            for (thread_idx, aos) in aos_to_lim.chunks(thread_chunk_size).enumerate() {
                let compressor = compressor.clone();
                s.run(move || {
                    let mut idx = thread_idx * thread_chunk_size;
                    let mut prev = K::zero_forward();
                    let mut soa0 = Vec::new();
                    let mut soa1 = Vec::new();
                    let mut soa2 = Vec::new();
                    let mut soa3 = Vec::new();
                    let mut dummy3 = [0u8; 32];

                    for aos in aos.chunks(COMPRESSION_BLOCK_SIZE) {
                        let soa_len = cmp::min(aos.len(), COMPRESSION_BLOCK_SIZE);
                        soa0.resize(soa_len, 0);
                        soa1.resize(soa_len, 0);
                        soa2.resize(soa_len, 0);
                        if HAS_ALPHA {
                            soa3.resize(soa_len, 0);
                        }

                        let mut aos_chunks = aos.array_chunks::<32>();
                        let mut s0_chunks = soa0.as_chunks_mut::<32>().0.iter_mut();
                        let mut s1_chunks = soa1.as_chunks_mut::<32>().0.iter_mut();
                        let mut s2_chunks = soa2.as_chunks_mut::<32>().0.iter_mut();
                        let mut or_acc = K::zero_forward();

                        if HAS_ALPHA {
                            let mut s3_chunks = soa3.as_chunks_mut::<32>().0.iter_mut();
                            if let Some(first_aos) = aos_chunks.next() {
                                prev = K::forward_block::<true, false>(
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
                                prev = K::forward_block::<true, true>(
                                    aos_c,
                                    s0_c,
                                    s1_c,
                                    s2_c,
                                    s3_c,
                                    prev,
                                    &mut or_acc,
                                );
                            }
                        } else {
                            if let Some(first_aos) = aos_chunks.next() {
                                prev = K::forward_block::<false, false>(
                                    first_aos,
                                    s0_chunks.next().unwrap(),
                                    s1_chunks.next().unwrap(),
                                    s2_chunks.next().unwrap(),
                                    &mut dummy3,
                                    prev,
                                    &mut or_acc,
                                );
                            }
                            for (aos_c, s0_c, s1_c, s2_c) in
                                izip!(aos_chunks, s0_chunks, s1_chunks, s2_chunks)
                            {
                                prev = K::forward_block::<false, true>(
                                    aos_c,
                                    s0_c,
                                    s1_c,
                                    s2_c,
                                    &mut dummy3,
                                    prev,
                                    &mut or_acc,
                                );
                            }
                        }

                        let (z0, z1, z2, z3) = K::or_accum_is_zero(or_acc);
                        emit_or_reuse_channel_shard(&compressor, idx, soa_len, z0, &mut soa0);
                        emit_or_reuse_channel_shard(&compressor, idx + len, soa_len, z1, &mut soa1);
                        emit_or_reuse_channel_shard(
                            &compressor,
                            idx + 2 * len,
                            soa_len,
                            z2,
                            &mut soa2,
                        );
                        if HAS_ALPHA {
                            emit_or_reuse_channel_shard(
                                &compressor,
                                idx + 3 * len,
                                soa_len,
                                z3,
                                &mut soa3,
                            );
                        }

                        idx += aos.len();
                    }
                });
            }
        });
    });

    if rem > 0 {
        let mut rem0 = vec![0u8; rem];
        let mut rem1 = vec![0u8; rem];
        let mut rem2 = vec![0u8; rem];
        let mut rem3 = if HAS_ALPHA {
            vec![0u8; rem]
        } else {
            Vec::new()
        };

        for (i, s) in aos_remainder.into_iter().enumerate() {
            rem0[i] = s.0;
            rem1[i] = s.1;
            rem2[i] = s.2;
            if HAS_ALPHA {
                rem3[i] = s.3;
            }
        }

        compressor.compress_shard(len - rem, rem0);
        compressor.compress_shard(2 * len - rem, rem1);
        compressor.compress_shard(3 * len - rem, rem2);
        if HAS_ALPHA {
            compressor.compress_shard(4 * len - rem, rem3);
        }
    }
}

const ZERO_CHUNK: [u8; 32] = [0; 32];

enum ChannelChunks<'a> {
    Slice(slice::Iter<'a, [u8; 32]>),
    Constant {
        first_chunk: [u8; 32],
        emitted_first: bool,
        remaining: usize,
    },
}

impl<'a> ChannelChunks<'a> {
    #[inline]
    fn new(data: &'a [u8], n_chunks: usize, const_delta: Option<u8>) -> Self {
        match const_delta {
            Some(d) => {
                let mut first_chunk = [0u8; 32];
                first_chunk[0] = d;
                Self::Constant {
                    first_chunk,
                    emitted_first: false,
                    remaining: n_chunks,
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
                remaining,
            } => {
                debug_assert!(*remaining > 0);
                *remaining -= 1;
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
    prev: &mut (
        K::InverseCarry,
        K::InverseCarry,
        K::InverseCarry,
        K::InverseCarry,
    ),
) {
    const { assert!(size_of::<[Vec4u8; 32]>().is_multiple_of(K::STREAM_ALIGN)) };
    let out_chunks = aos.as_chunks_mut::<32>().0;
    let n_chunks = out_chunks.len();

    if let (Some(d0), Some(d1), Some(d2)) = (c0, c1, c2)
        && (!HAS_ALPHA || c3.is_some())
    {
        let (p0, b_diff) = K::advance_inverse_carry(prev.0, d0);
        let (p1, g) = K::advance_inverse_carry(prev.1, d1);
        let (p2, r_diff) = K::advance_inverse_carry(prev.2, d2);
        prev.0 = p0;
        prev.1 = p1;
        prev.2 = p2;
        let a = if HAS_ALPHA {
            let (p3, a) = K::advance_inverse_carry(prev.3, c3.unwrap());
            prev.3 = p3;
            a
        } else {
            0xFF
        };
        let pixel = Vec4u8(b_diff.wrapping_add(g), g, r_diff.wrapping_add(g), a);
        // SAFETY: Caller guarantees `out_chunks` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
        unsafe { K::splat_blocks::<STREAM>(out_chunks, pixel) };
        return;
    }

    if !HAS_ALPHA || c3.is_some() {
        let const_alpha = if HAS_ALPHA {
            let (p3, a) = K::advance_inverse_carry(prev.3, c3.unwrap());
            prev.3 = p3;
            a
        } else {
            0xFF
        };

        if c0.is_none() && c1.is_none() && c2.is_none() {
            for (s0, s1, s2, out) in izip!(
                soa0.as_chunks::<32>().0,
                soa1.as_chunks::<32>().0,
                soa2.as_chunks::<32>().0,
                out_chunks,
            ) {
                // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
                *prev = unsafe {
                    K::inverse_block::<STREAM, true>(
                        s0,
                        s1,
                        s2,
                        &ZERO_CHUNK,
                        const_alpha,
                        out,
                        *prev,
                    )
                };
            }
        } else {
            let mut ch0 = ChannelChunks::new(soa0, n_chunks, c0);
            let mut ch1 = ChannelChunks::new(soa1, n_chunks, c1);
            let mut ch2 = ChannelChunks::new(soa2, n_chunks, c2);
            for out in out_chunks {
                // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
                *prev = unsafe {
                    K::inverse_block::<STREAM, true>(
                        ch0.next_chunk(),
                        ch1.next_chunk(),
                        ch2.next_chunk(),
                        &ZERO_CHUNK,
                        const_alpha,
                        out,
                        *prev,
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
            *prev = unsafe { K::inverse_block::<STREAM, false>(s0, s1, s2, s3, 0, out, *prev) };
        }
    } else {
        let mut ch0 = ChannelChunks::new(soa0, n_chunks, c0);
        let mut ch1 = ChannelChunks::new(soa1, n_chunks, c1);
        let mut ch2 = ChannelChunks::new(soa2, n_chunks, c2);
        for (s3, out) in izip!(soa3.as_chunks::<32>().0, out_chunks) {
            // SAFETY: Caller guarantees `out` satisfies `K::STREAM_ALIGN` when `STREAM` is `true`.
            *prev = unsafe {
                K::inverse_block::<STREAM, false>(
                    ch0.next_chunk(),
                    ch1.next_chunk(),
                    ch2.next_chunk(),
                    s3,
                    0,
                    out,
                    *prev,
                )
            };
        }
    }
}

#[instrument(skip_all, level = "debug")]
fn soa_to_aos_parallel<K: FilterKernel, const HAS_ALPHA: bool>(
    soa: &DecompressedShards,
    aos: &mut [Vec4u8],
) {
    let len = aos.len();
    let n_channels = if HAS_ALPHA { 4 } else { 3 };
    assert_eq!(soa.len(), len * n_channels);
    if len == 0 {
        return;
    }

    let (soa0, rest) = soa.data.split_at(len);
    let (soa1, rest) = rest.split_at(len);
    let (soa2, soa3) = if HAS_ALPHA {
        rest.split_at(len)
    } else {
        (rest, &[][..])
    };

    let n_blocks = len / 32;
    let lim = n_blocks * 32;
    let blocks_per_thread = cmp::max(n_blocks / N_FILTER_THREADS, 1);
    let thread_chunk_size = blocks_per_thread * 32;

    if lim > 0 {
        let constant_deltas = &soa.constant_deltas;
        debug_span!("soa_to_aos_u8_32x4_loop").in_scope(|| {
            ThreadPool::global().scoped(|s| {
                for (thread_idx, aos_thread) in aos[..lim].chunks_mut(thread_chunk_size).enumerate()
                {
                    let thread_start = thread_idx * thread_chunk_size;
                    let thread_end = thread_start + aos_thread.len();
                    let s0_thread = &soa0[thread_start..thread_end];
                    let s1_thread = &soa1[thread_start..thread_end];
                    let s2_thread = &soa2[thread_start..thread_end];
                    let s3_thread = if HAS_ALPHA {
                        &soa3[thread_start..thread_end]
                    } else {
                        &[][..]
                    };

                    s.run(move || {
                        let stream =
                            (aos_thread.as_mut_ptr() as usize).is_multiple_of(K::STREAM_ALIGN);
                        let mut prev = K::zero_inverse();
                        let mut rel = 0;
                        while rel < aos_thread.len() {
                            let block_len =
                                cmp::min(aos_thread.len() - rel, COMPRESSION_BLOCK_SIZE);
                            let idx = thread_start + rel;
                            let c0 = constant_deltas.get(&idx).copied();
                            let c1 = constant_deltas.get(&(idx + len)).copied();
                            let c2 = constant_deltas.get(&(idx + 2 * len)).copied();
                            let c3 = if HAS_ALPHA {
                                constant_deltas.get(&(idx + 3 * len)).copied()
                            } else {
                                None
                            };
                            let s3_sub = if HAS_ALPHA {
                                &s3_thread[rel..rel + block_len]
                            } else {
                                &[][..]
                            };

                            if stream {
                                // SAFETY: `stream` verified `aos_thread` is aligned to `K::STREAM_ALIGN`.
                                unsafe {
                                    unfilter_block::<K, true, HAS_ALPHA>(
                                        &s0_thread[rel..rel + block_len],
                                        &s1_thread[rel..rel + block_len],
                                        &s2_thread[rel..rel + block_len],
                                        s3_sub,
                                        c0,
                                        c1,
                                        c2,
                                        c3,
                                        &mut aos_thread[rel..rel + block_len],
                                        &mut prev,
                                    );
                                }
                            } else {
                                // SAFETY: `STREAM` is `false`.
                                unsafe {
                                    unfilter_block::<K, false, HAS_ALPHA>(
                                        &s0_thread[rel..rel + block_len],
                                        &s1_thread[rel..rel + block_len],
                                        &s2_thread[rel..rel + block_len],
                                        s3_sub,
                                        c0,
                                        c1,
                                        c2,
                                        c3,
                                        &mut aos_thread[rel..rel + block_len],
                                        &mut prev,
                                    );
                                }
                            }

                            rel += block_len;
                        }
                        if stream {
                            K::fence_stream();
                        }
                    });
                }
            });
        });
    }

    for i in lim..len {
        let a = if HAS_ALPHA { soa3[i] } else { 0xFF };
        aos[i] = Vec4u8(soa0[i], soa1[i], soa2[i], a);
    }
}

// --- public API ---

pub fn filter_and_compress_streaming(
    data: BufferPointer<u8>,
    format: BufferFormat,
    compressor: &mut ShardingCompressor,
    on_stream_ready: impl FnOnce(StreamingCompressedShards),
) {
    assert!(data.len().is_multiple_of(4));
    // SAFETY: Vec4u8 is a repr(C, packed) wrapper around [u8; 4].
    let aos = unsafe { data.cast::<Vec4u8>() };
    let has_alpha = format == BufferFormat::Argb8888;
    let n_channels = if has_alpha { 4 } else { 3 };
    let (indices, thread_chunk_size) = shard_indices_and_chunk_size(aos.len(), n_channels);
    let (sink, streaming) = compressor.begin_streaming(indices, aos.len() * n_channels);
    on_stream_ready(streaming);
    if has_alpha {
        aos_to_soa_parallel_compression::<arch::Kernel, true>(aos, thread_chunk_size, sink);
    } else {
        aos_to_soa_parallel_compression::<arch::Kernel, false>(aos, thread_chunk_size, sink);
    }
}

pub fn filter_and_compress(
    data: BufferPointer<u8>,
    format: BufferFormat,
    compressor: &mut ShardingCompressor,
) -> CompressedShards {
    let mut streaming_opt = None;
    filter_and_compress_streaming(data, format, compressor, |streaming| {
        streaming_opt = Some(streaming);
    });
    streaming_opt.unwrap().collect_shards()
}

pub fn unfilter(data: &DecompressedShards, format: BufferFormat, output_buf: &mut [u8]) {
    let aos = bytemuck::cast_slice_mut(output_buf);
    match format {
        BufferFormat::Argb8888 => soa_to_aos_parallel::<arch::Kernel, true>(data, aos),
        BufferFormat::Xrgb8888 => soa_to_aos_parallel::<arch::Kernel, false>(data, aos),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::num::NonZeroUsize;

    use fallible_iterator::IteratorExt;
    use proptest::prelude::*;

    use super::*;
    use crate::sharding_compression::CompressedShard;
    use crate::sharding_compression::ShardingDecompressor;

    thread_local! {
        static TEST_CODEC: RefCell<Option<(ShardingCompressor, ShardingDecompressor)>> =
            const { RefCell::new(None) };
    }

    fn with_test_codec(f: impl FnOnce(&mut ShardingCompressor, &mut ShardingDecompressor)) {
        TEST_CODEC.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (compressor, decompressor) = borrow.get_or_insert_with(|| {
                (
                    ShardingCompressor::new(NonZeroUsize::new(8).unwrap(), 1).unwrap(),
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

    fn test_roundtrip_impl(
        data: &[u8],
        compressor: &mut ShardingCompressor,
        decompressor: &mut ShardingDecompressor,
    ) {
        assert!(data.len().is_multiple_of(4));

        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        let shards = filter_and_compress(buf_ptr, BufferFormat::Argb8888, compressor);
        let indices = shards.indices();
        let uncompressed_size = shards.uncompressed_size();

        let soa = decompressor
            .decompress_to_owned(
                &indices,
                uncompressed_size,
                shards
                    .shards
                    .into_iter()
                    .map(Ok::<CompressedShard, anyhow::Error>)
                    .transpose_into_fallible(),
            )
            .unwrap();

        let mut buf = vec![0u8; data.len() + 36];
        let aligned_off = (32 - (buf.as_ptr() as usize & 31)) & 31;
        unfilter(
            &soa,
            BufferFormat::Argb8888,
            &mut buf[aligned_off..aligned_off + data.len()],
        );
        assert_eq!(data, &buf[aligned_off..aligned_off + data.len()]);

        unfilter(
            &soa,
            BufferFormat::Argb8888,
            &mut buf[aligned_off + 4..aligned_off + 4 + data.len()],
        );
        assert_eq!(data, &buf[aligned_off + 4..aligned_off + 4 + data.len()]);
    }

    fn test_roundtrip_xrgb_impl(
        data: &[u8],
        compressor: &mut ShardingCompressor,
        decompressor: &mut ShardingDecompressor,
    ) {
        assert!(data.len().is_multiple_of(4));

        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        let shards = filter_and_compress(buf_ptr, BufferFormat::Xrgb8888, compressor);
        let indices = shards.indices();
        assert_eq!(shards.uncompressed_size(), (data.len() / 4) * 3);

        let soa = decompressor
            .decompress_to_owned(
                &indices,
                shards.uncompressed_size(),
                shards
                    .shards
                    .into_iter()
                    .map(Ok::<CompressedShard, anyhow::Error>)
                    .transpose_into_fallible(),
            )
            .unwrap();

        let mut expected = data.to_vec();
        for px in expected.as_chunks_mut::<4>().0 {
            px[3] = 0xFF;
        }

        let mut out = vec![0u8; data.len()];
        unfilter(&soa, BufferFormat::Xrgb8888, &mut out);
        assert_eq!(expected, out);
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
                1008 * 9513 * 4,
                1008 * 951 * 4,
            ] {
                let v = test_vec(n);
                test_roundtrip_impl(&v, compressor, decompressor);
                test_roundtrip_xrgb_impl(&v, compressor, decompressor);
            }

            // Multi-shard solid-color roundtrips exercising `ConstantDelta`.
            let solid: Vec<u8> = [0x20u8, 0x70, 0xD0, 0xFF]
                .into_iter()
                .cycle()
                .take(COMPRESSION_BLOCK_SIZE * 4 * 6 + 20)
                .collect();
            test_roundtrip_impl(&solid, compressor, decompressor);
            test_roundtrip_xrgb_impl(&solid, compressor, decompressor);

            // Exhaustively cover all 16 `{Constant, Slice}` channel combinations
            // (`bit 0 = B-G`, `bit 1 = G`, `bit 2 = R-G`, `bit 3 = A`; set bit
            // yields a constant non-zero delta, clear bit yields a varying slice).
            // Set `px[1]` (G) before `px[0]`/`px[2]` so `B - G` and `R - G` stay
            // constant even when G varies.
            for mask in 0u8..16 {
                let mut data = test_vec(COMPRESSION_BLOCK_SIZE * 4 + 128);
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

    /// Verify that the filtered SoA output matches known-good values.
    ///
    /// This test is architecture-independent: both x86 and aarch64 must
    /// produce identical compressed shards for the same input, since a
    /// buffer filtered on one architecture may be unfiltered on another.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filtered_output_deterministic() {
        // 128 bytes = 32 pixels = exactly one SIMD block on both architectures.
        let mut data = [0u8; 128];
        for i in 0..32usize {
            let k = i as u32;
            data[i * 4] = ((k * k * 3 + 19) & 0xFF) as u8;
            data[i * 4 + 1] = ((k * 17 + (k >> 2) * 41 + 5) & 0xFF) as u8;
            data[i * 4 + 2] = ((k * k * 11 + k * 7 + 200) & 0xFF) as u8;
            data[i * 4 + 3] = if i == 17 { 0xFE } else { 0xFF };
        }
        let data_ptr = data.as_ptr();
        let buf_ptr = unsafe { BufferPointer::new(&data_ptr, data.len()) };

        with_test_codec(|compressor, decompressor| {
            let shards = filter_and_compress(buf_ptr, BufferFormat::Argb8888, compressor);
            let indices = shards.indices();
            let mut soa = decompressor
                .decompress_to_owned(
                    &indices,
                    shards.uncompressed_size(),
                    shards
                        .shards
                        .into_iter()
                        .map(Ok::<CompressedShard, anyhow::Error>)
                        .transpose_into_fallible(),
                )
                .unwrap();
            for (&idx, &first_byte) in &soa.constant_deltas {
                soa.data[idx] = first_byte;
            }

            // SoA layout: 4 channel arrays concatenated, each of length n_pixels.
            let n = 32;
            assert!(soa.len() >= 4 * n);
            let s0 = &soa.data[0..n];
            let s1 = &soa.data[n..2 * n];
            let s2 = &soa.data[2 * n..3 * n];
            let s3 = &soa.data[3 * n..4 * n];

            // Compute expected values via scalar reference implementation.
            // Forward: transpose → subtract_green → running_difference.
            let mut expected = [vec![0u8; n], vec![0u8; n], vec![0u8; n], vec![0u8; n]];
            for i in 0..n {
                expected[0][i] = data[i * 4]; // channel 0
                expected[1][i] = data[i * 4 + 1]; // channel 1 (green)
                expected[2][i] = data[i * 4 + 2]; // channel 2
                expected[3][i] = data[i * 4 + 3]; // channel 3
            }
            // subtract_green: ch0 -= ch1, ch2 -= ch1.
            let [ch0, ch1, ch2, ..] = &mut expected;
            for (b, (g, r)) in ch0.iter_mut().zip(ch1.iter().zip(ch2.iter_mut())) {
                *b = b.wrapping_sub(*g);
                *r = r.wrapping_sub(*g);
            }
            // running_difference per channel.
            for ch in &mut expected {
                let mut prev = 0u8;
                for v in ch.iter_mut() {
                    let orig = *v;
                    *v = orig.wrapping_sub(prev);
                    prev = orig;
                }
            }

            assert_eq!(s0, &expected[0], "channel 0 mismatch");
            assert_eq!(s1, &expected[1], "channel 1 mismatch");
            assert_eq!(s2, &expected[2], "channel 2 mismatch");
            assert_eq!(s3, &expected[3], "channel 3 mismatch");
        });
    }
}
