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

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::io::Write;
use std::num::NonZeroUsize;
use std::thread;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use divbuf::DivBufMut;
use divbuf::DivBufShared;
use fallible_iterator::FallibleIterator;
use fallible_iterator::IteratorExt;
use rkyv::Archive;
use rkyv::Deserialize;
use rkyv::Serialize;
use rkyv::rancor::Error as RancorError;
use rkyv::util::AlignedVec;
use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;

use crate::arc_slice::ArcSlice;
use crate::prelude::*;
use crate::serialization::framing::Framed;

// TODO: benchmark this and pick a value based on that.
pub const MIN_SIZE_TO_COMPRESS: usize = 4096;
const MAX_QUEUED_UNCOMPRESSED_SHARDS: usize = 64;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub enum ShardCompression {
    Uncompressed,
    Zstd,
    ConstantDelta(u8),
}

#[derive(Clone, Eq, PartialEq, Archive, Deserialize, Serialize)]
pub struct CompressedShard {
    pub idx: usize,
    pub uncompressed_size: usize,
    pub compression: ShardCompression,
    pub data: Vec<u8>,
}

impl CompressedShard {
    pub fn len(&self) -> usize {
        match self.compression {
            ShardCompression::ConstantDelta(_) => 1,
            ShardCompression::Uncompressed | ShardCompression::Zstd => self.data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for CompressedShard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedShard")
            .field("idx", &self.idx)
            .field("compression", &self.compression)
            .field("data", &format_args!("Vec<u8>[{:?}]", self.data.len()))
            .finish()
    }
}

impl Framed for CompressedShard {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        self.idx.framed_write(stream).location(loc!())?;
        self.uncompressed_size
            .framed_write(stream)
            .location(loc!())?;
        match self.compression {
            ShardCompression::Uncompressed => {
                0u8.framed_write(stream).location(loc!())?;
                self.data.framed_write(stream).location(loc!())?;
            },
            ShardCompression::Zstd => {
                1u8.framed_write(stream).location(loc!())?;
                self.data.framed_write(stream).location(loc!())?;
            },
            ShardCompression::ConstantDelta(first_byte) => {
                2u8.framed_write(stream).location(loc!())?;
                first_byte.framed_write(stream).location(loc!())?;
            },
        }
        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let idx = usize::framed_read(stream).location(loc!())?;
        let uncompressed_size = usize::framed_read(stream).location(loc!())?;
        let tag = u8::framed_read(stream).location(loc!())?;
        // TODO: this fails on client disconnection
        let (compression, data) = match tag {
            0 => (
                ShardCompression::Uncompressed,
                Vec::<u8>::framed_read(stream).location(loc!())?,
            ),
            1 => (
                ShardCompression::Zstd,
                Vec::<u8>::framed_read(stream).location(loc!())?,
            ),
            2 => (
                ShardCompression::ConstantDelta(u8::framed_read(stream).location(loc!())?),
                Vec::new(),
            ),
            _ => bail!("invalid ShardCompression tag: {tag}"),
        };
        Ok(Self {
            idx,
            uncompressed_size,
            compression,
            data,
        })
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct DecompressedShards {
    pub data: Vec<u8>,
    pub constant_deltas: HashMap<usize, u8>,
}

impl DecompressedShards {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Default, Archive, Deserialize, Serialize)]
pub struct CompressedShards {
    pub shards: Vec<CompressedShard>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompressedShardsRef<'a> {
    pub shards: &'a CompressedShard,
}

impl CompressedShards {
    pub fn new(mut shards: Vec<CompressedShard>) -> Self {
        shards.sort_by_key(|k| k.idx);
        Self { shards }
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    pub fn indices(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.idx).collect()
    }

    pub fn uncompressed_size(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.uncompressed_size)
            .sum()
    }

    pub fn size(&self) -> usize {
        self.shards.iter().map(CompressedShard::len).sum()
    }

    #[instrument(skip_all, level = "debug")]
    pub fn streaming_framed_decompress_with<F, T, R: Read>(
        stream: &mut R,
        decompressor: &mut ShardingDecompressor,
        f: F,
    ) -> Result<T>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let serialized_indices = AlignedVec::framed_read(stream).location(loc!())?;
        let indices =
            rkyv::from_bytes::<Vec<usize>, RancorError>(&serialized_indices).location(loc!())?;
        debug!("read indices: {:?}", indices);

        let uncompressed_size = usize::framed_read(stream).location(loc!())?;
        debug!("read uncompressed_size: {:?}", uncompressed_size);

        let shards = (0..indices.len())
            .map(|_| CompressedShard::framed_read(stream))
            .transpose_into_fallible();
        debug!("read data");

        decompressor.decompress_with(&indices, uncompressed_size, shards, f)
    }

    #[instrument(skip_all, level = "debug")]
    pub fn streaming_framed_decompress_to_owned<R: Read>(
        stream: &mut R,
        decompressor: &mut ShardingDecompressor,
    ) -> Result<DecompressedShards> {
        let serialized_indices = AlignedVec::framed_read(stream).location(loc!())?;
        let indices =
            rkyv::from_bytes::<Vec<usize>, RancorError>(&serialized_indices).location(loc!())?;
        debug!("read indices: {:?}", indices);

        let uncompressed_size = usize::framed_read(stream).location(loc!())?;
        debug!("read uncompressed_size: {:?}", uncompressed_size);

        let shards = (0..indices.len())
            .map(|_| CompressedShard::framed_read(stream))
            .transpose_into_fallible();
        debug!("read data");

        decompressor.decompress_to_owned(&indices, uncompressed_size, shards)
    }
}

impl Framed for CompressedShards {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        let indices = self.indices();
        rkyv::to_bytes::<RancorError>(&indices)
            .location(loc!())?
            .framed_write(stream)
            .location(loc!())?;

        self.uncompressed_size()
            .framed_write(stream)
            .location(loc!())?;

        for shard in self.shards.iter() {
            shard.framed_write(stream).location(loc!())?;
            // Flush here instaed of after writing all the frames so that the client
            // can start decompressing the shards sooner.
            stream.flush().location(loc!())?;
        }

        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let serialized_indices = AlignedVec::framed_read(stream).location(loc!())?;
        let indices =
            rkyv::from_bytes::<Vec<usize>, RancorError>(&serialized_indices).location(loc!())?;
        let shards: Vec<CompressedShard> = (0..indices.len())
            .map(|_| CompressedShard::framed_read(stream))
            .transpose_into_fallible()
            .collect()
            .location(loc!())?;

        Ok(Self { shards })
    }
}

const MAX_COALESCED_SHARD_BYTES: usize = 256 * 1024;

pub struct StreamingCompressedShards {
    pub indices: Vec<usize>,
    pub uncompressed_size: usize,
    pub shards: Receiver<CompressedShard>,
}

impl StreamingCompressedShards {
    pub fn framed_write<W: Write>(self, stream: &mut W) -> Result<usize> {
        rkyv::to_bytes::<RancorError>(&self.indices)
            .location(loc!())?
            .framed_write(stream)
            .location(loc!())?;

        self.uncompressed_size
            .framed_write(stream)
            .location(loc!())?;

        let mut remaining = self.indices.len();
        let mut total_compressed_size = 0;

        while remaining > 0 {
            let shard = self
                .shards
                .recv()
                .context(loc!(), "shard sender dropped before sending all shards")?;
            let mut batch_bytes = shard.len();
            total_compressed_size += shard.len();
            shard.framed_write(stream).location(loc!())?;
            remaining -= 1;

            // Coalesce any shards that are already ready in the channel into the
            // same flush, while flushing immediately as soon as the channel is
            // empty so the client can decompress completed shards while we are
            // still compressing subsequent ones.
            while remaining > 0
                && batch_bytes < MAX_COALESCED_SHARD_BYTES
                && let Ok(shard) = self.shards.try_recv()
            {
                batch_bytes += shard.len();
                total_compressed_size += shard.len();
                shard.framed_write(stream).location(loc!())?;
                remaining -= 1;
            }

            stream.flush().location(loc!())?;
        }

        Ok(total_compressed_size)
    }

    #[instrument(skip_all, level = "debug")]
    pub fn collect_shards(self) -> CompressedShards {
        let shards = CompressedShards::new(self.shards.into_iter().collect());
        debug_assert_eq!(self.indices, shards.indices());
        shards
    }
}

type CompressorJob = (
    usize,
    Box<dyn AsRef<[u8]> + Send + Sync + 'static>,
    Sender<CompressedShard>,
);

fn spawn_compressor(compression_level: i32, input_rx: Receiver<CompressorJob>) -> Result<()> {
    let mut compressor = Compressor::new(compression_level).location(loc!())?;
    compressor.long_distance_matching(true).location(loc!())?;
    thread::spawn(move || {
        // The iterator (and, consequently, the thread) will terminate when all
        // the input senders (which are all in the ShardingCompressor) are
        // dropped.
        for (idx, input, output_tx) in input_rx {
            let input = (*input).as_ref();
            let (compression, data) = if input.len() > MIN_SIZE_TO_COMPRESS {
                (ShardCompression::Zstd, compressor.compress(input).unwrap())
            } else {
                (ShardCompression::Uncompressed, input.to_vec())
            };

            // Ignores error if the receiver dropped early (e.g., on client
            // disconnect).
            _ = output_tx.send(CompressedShard {
                idx,
                uncompressed_size: input.len(),
                compression,
                data,
            });
        }
    });
    Ok(())
}

#[derive(Clone)]
pub struct ShardingCompressorSink {
    compressor_input: Sender<CompressorJob>,
    shard_tx: Sender<CompressedShard>,
}

impl ShardingCompressorSink {
    pub fn compress_shard(&self, idx: usize, data: impl AsRef<[u8]> + Send + Sync + 'static) {
        self.compressor_input
            .send((idx, Box::new(data), self.shard_tx.clone()))
            .unwrap();
    }

    pub fn send_constant_delta_shard(&self, idx: usize, uncompressed_size: usize, first_byte: u8) {
        _ = self.shard_tx.send(CompressedShard {
            idx,
            uncompressed_size,
            compression: ShardCompression::ConstantDelta(first_byte),
            data: Vec::new(),
        });
    }
}

pub struct ShardingCompressor {
    compressor_input: Sender<CompressorJob>,
}

impl ShardingCompressor {
    pub fn new(n_compressors: NonZeroUsize, compression_level: i32) -> Result<Self> {
        let (compressor_input_tx, compressor_input_rx) =
            crossbeam_channel::bounded(MAX_QUEUED_UNCOMPRESSED_SHARDS);
        for _ in 0..n_compressors.get() {
            spawn_compressor(compression_level, compressor_input_rx.clone()).location(loc!())?;
        }

        Ok(Self {
            compressor_input: compressor_input_tx,
        })
    }

    pub fn begin_streaming(
        &self,
        indices: Vec<usize>,
        uncompressed_size: usize,
    ) -> (ShardingCompressorSink, StreamingCompressedShards) {
        let (shard_tx, shard_rx) = crossbeam_channel::unbounded();
        (
            ShardingCompressorSink {
                compressor_input: self.compressor_input.clone(),
                shard_tx,
            },
            StreamingCompressedShards {
                indices,
                uncompressed_size,
                shards: shard_rx,
            },
        )
    }

    #[instrument(skip_all, level = "debug")]
    pub fn compress(&mut self, n_shards: NonZeroUsize, data: ArcSlice<u8>) -> CompressedShards {
        let len = data.len();
        if len == 0 {
            return CompressedShards::default();
        }
        let chunk_size = (len / n_shards).max(1);
        debug!("chunk_size: {:?}", chunk_size);
        let indices = (0..len).step_by(chunk_size).collect();
        let (sink, streaming) = self.begin_streaming(indices, len);
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            sink.compress_shard(i * chunk_size, chunk);
        }
        drop(sink);
        streaming.collect_shards()
    }
}

/// # Panics
/// If there is a bug and the decompression buffer wasn't resized to be large enough.
pub fn spawn_decompressor(
    input_rx: Receiver<(CompressedShard, DivBufMut)>,
    output_tx: Sender<()>,
) -> Result<()> {
    let mut decompressor = Decompressor::new().location(loc!())?;
    thread::spawn(move || {
        // The iterator (and, consequently, the thread) will terminate when all
        // the input senders (which are all in the ShardingDecompressor) are
        // dropped.
        for (input, mut output) in input_rx.iter() {
            let _span = debug_span!("decompressor").entered();
            match input.compression {
                ShardCompression::Zstd => {
                    // We made DivBufMut large enough, so this should never panic.
                    decompressor
                        .decompress_to_buffer(&input.data, output.as_mut())
                        .unwrap();
                },
                ShardCompression::Uncompressed => {
                    // The last output block will be larger than the data.
                    let output = &mut output[0..input.data.len()];
                    output.copy_from_slice(&input.data);
                },
                ShardCompression::ConstantDelta(first_byte) => {
                    if input.uncompressed_size > 0 {
                        output[0] = first_byte;
                        output[1..input.uncompressed_size].fill(0);
                    }
                },
            }
            drop(output); // release our handle

            // This will be an error when the ShardingDecompressor is dropped,
            // but the for loop (and consequently this thread) will terminate at
            // the same time for the same reason.
            _ = output_tx.send(());
        }
    });
    Ok(())
}

pub struct ShardingDecompressor {
    decompressor_input: Sender<(CompressedShard, DivBufMut)>,
    decompressor_output: Receiver<()>,
    buffer: DivBufShared,
}

impl ShardingDecompressor {
    pub fn new(n_decompressors: NonZeroUsize) -> Result<Self> {
        // These channels will have at most n_shards items in them, but we only
        // know n_shards when decompress is called, not now.
        let (decompressor_input_tx, decompressor_input_rx) = crossbeam_channel::unbounded();
        let (decompressor_output_tx, decompressor_output_rx) = crossbeam_channel::unbounded();

        for _ in 0..n_decompressors.get() {
            spawn_decompressor(
                decompressor_input_rx.clone(),
                decompressor_output_tx.clone(),
            )
            .location(loc!())?;
        }

        Ok(Self {
            decompressor_input: decompressor_input_tx,
            decompressor_output: decompressor_output_rx,
            buffer: DivBufShared::from(vec![0; 64 * 1024]),
        })
    }

    fn dispatch_shards_into<E: std::convert::From<anyhow::Error> + Send + Sync + 'static>(
        decompressor_input: &Sender<(CompressedShard, DivBufMut)>,
        target_buffer: &DivBufShared,
        indices: &[usize],
        mut compressed_shards: impl FallibleIterator<Item = CompressedShard, Error = E>,
        dispatched: &mut usize,
        mut constant_deltas: Option<&mut HashMap<usize, u8>>,
    ) -> std::result::Result<(), E>
    where
        Result<(), E>: anyhow::Context<(), E>,
    {
        let mut mut_buf = target_buffer.try_mut().location(loc!())?;

        let mut prev_idx = 0;
        let mut output_divbufs: Vec<DivBufMut> = indices
            .iter()
            // The first index is 0, we don't need to split on it.
            .skip(1)
            .map(|idx| {
                let buf = mut_buf.split_to(idx - prev_idx);
                prev_idx = *idx;
                buf
            })
            .collect();
        output_divbufs.push(mut_buf);

        let mut index_divbuf_map: HashMap<usize, DivBufMut> =
            indices.iter().cloned().zip(output_divbufs).collect();

        while let Some(shard) = compressed_shards.next()? {
            let divbuf = index_divbuf_map.remove(&shard.idx).unwrap();
            if let Some(deltas) = constant_deltas.as_deref_mut()
                && let ShardCompression::ConstantDelta(first_byte) = shard.compression
            {
                deltas.insert(shard.idx, first_byte);
                drop(divbuf);
                continue;
            }
            decompressor_input.send((shard, divbuf)).unwrap();
            *dispatched += 1;
        }

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn iter_shards(&self, n: usize) -> impl Iterator<Item = ()> + '_ {
        // Will only panic is the other end disconnected, which should never
        // happen.
        (0..n).map(|_| self.decompressor_output.recv().unwrap())
    }

    #[instrument(skip_all, level = "debug")]
    fn collect_shards(&self, n: usize) {
        self.iter_shards(n).for_each(|_| {});
    }

    /// IMPORTANT:
    /// * Indices must be sorted.
    /// * If indices != compressed_shards.collect().indices(), this function
    ///   will either panic or return corrupt data.
    #[instrument(skip_all, level = "debug")]
    pub fn decompress_with<F, T, E: std::convert::From<anyhow::Error> + Send + Sync + 'static>(
        &mut self,
        indices: &[usize],
        uncompressed_size: usize,
        compressed_shards: impl FallibleIterator<Item = CompressedShard, Error = E>,
        f: F,
    ) -> Result<T>
    where
        Result<(), E>: anyhow::Context<(), E>,
        F: FnOnce(&[u8]) -> Result<T>,
    {
        if indices.is_empty() {
            bail!("Cannot call decompress_with on empty indices.");
        }

        if uncompressed_size > self.buffer.len() {
            self.buffer = DivBufShared::from(vec![0; uncompressed_size]);
        }

        let mut dispatched = 0;
        Self::dispatch_shards_into(
            &self.decompressor_input,
            &self.buffer,
            indices,
            compressed_shards,
            &mut dispatched,
            None,
        )
        .location(loc!())?;
        self.collect_shards(dispatched);

        // We dropped mut_buf and all the buffers in the decompressor threads
        // have been freed already, so this should never fail.
        let decompressed_data = self.buffer.try_const().unwrap().split_to(uncompressed_size);
        f(&decompressed_data)
    }

    /// IMPORTANT: see note on decompress_with.
    #[instrument(skip_all, level = "debug")]
    pub fn decompress_to_owned<E: std::convert::From<anyhow::Error> + Send + Sync + 'static>(
        &mut self,
        indices: &[usize],
        uncompressed_size: usize,
        compressed_shards: impl FallibleIterator<Item = CompressedShard, Error = E>,
    ) -> Result<DecompressedShards>
    where
        Result<(), E>: anyhow::Context<(), E>,
    {
        if indices.is_empty() {
            return Ok(DecompressedShards::default());
        }

        // Allocate >= 36 MB (> glibc's 32 MiB DEFAULT_MMAP_THRESHOLD_MAX) so
        // calloc uses demand-zero mmap pages instead of single-threaded heap
        // memset, leaving ConstantDelta pages and unused tail completely
        // unfaulted.
        let alloc_size = uncompressed_size.max(36_000_000);
        let buf = DivBufShared::from(vec![0; alloc_size]);
        let mut dispatched = 0;
        let mut constant_deltas = HashMap::new();
        Self::dispatch_shards_into(
            &self.decompressor_input,
            &buf,
            indices,
            compressed_shards,
            &mut dispatched,
            Some(&mut constant_deltas),
        )
        .location(loc!())?;
        self.collect_shards(dispatched);

        // We dropped mut_buf and all the buffers in the decompressor threads
        // have been freed already, so this should never fail.
        let mut data: Vec<u8> = buf.try_into().unwrap();
        data.truncate(uncompressed_size);
        Ok(DecompressedShards {
            data,
            constant_deltas,
        })
    }
}
