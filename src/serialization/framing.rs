// Copyright 2025 Google LLC
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

use std::convert::TryInto;
use std::fmt;
use std::io::Read;
use std::io::Write;
use std::mem;

use crossbeam_channel::Receiver;
use rkyv::util::AlignedVec;
use static_assertions::const_assert;
use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;
use zstd::zstd_safe;

use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::wayland::BufferFormat;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::WlSurfaceId;

const_assert!(mem::size_of::<usize>() >= mem::size_of::<u32>());

/// Pixels per self-contained L2-resident tile (`64 KiB` per channel, `256 KiB`
/// 4-channel SoA scratch, fitting inside a 512 KiB L2 cache alongside zstd's
/// hash table).
pub const TILE_PIXELS: usize = 64 * 1024;

pub trait Framed: Sized {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()>;
    fn framed_read<R: Read>(stream: &mut R) -> Result<Self>;
}

impl Framed for u8 {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        stream.write_all(&self.to_be_bytes()).location(loc!())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let mut buf = [0u8; mem::size_of::<Self>()];
        stream.read_exact(&mut buf).location(loc!())?;
        Ok(Self::from_be_bytes(buf))
    }
}

impl Framed for u32 {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        stream.write_all(&self.to_be_bytes()).location(loc!())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let mut buf = [0u8; mem::size_of::<Self>()];
        stream.read_exact(&mut buf).location(loc!())?;
        Ok(Self::from_be_bytes(buf))
    }
}

impl Framed for usize {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        u32::try_from(*self)
            .context(loc!(), "framed usize exceeds u32::MAX")?
            .framed_write(stream)
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        u32::framed_read(stream).map(|u| u.try_into().unwrap())
    }
}

impl Framed for Vec<u8> {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        self.len().framed_write(stream).location(loc!())?;
        stream.write_all(self).location(loc!())?;
        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let len = u32::framed_read(stream).location(loc!())?;
        // Pre-size up to `TILE_PIXELS` (64 KiB) so legitimate tile channels
        // allocate once without `vec![0; len]`'s redundant zero-fill, while
        // corrupt wire lengths (`u32::MAX`) cannot trigger multi-GiB allocations
        // before reading from `stream`.
        let mut buf = Self::with_capacity((len as usize).min(TILE_PIXELS));
        let read = stream
            .take(len as u64)
            .read_to_end(&mut buf)
            .location(loc!())?;
        if read != len as usize {
            bail!("unexpected EOF reading framed Vec<u8>: expected {len} bytes, got {read}");
        }
        Ok(buf)
    }
}

impl Framed for String {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        let bytes = self.as_bytes();
        bytes.len().framed_write(stream).location(loc!())?;
        stream.write_all(bytes).location(loc!())?;
        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let bytes = Vec::<u8>::framed_read(stream).location(loc!())?;
        Self::from_utf8(bytes).location(loc!())
    }
}

/// Maximum payload size sent uncompressed by [`ObjectCompressor`] and accepted
/// on raw frames (`tag == 0`) by [`ObjectDecompressor`] (wire-format contract;
/// ~1 Gbps breakeven for serial level-1 object framing).
const MAX_RAW_OBJECT_BYTES: usize = 4096;

pub struct ObjectCompressor {
    compressor: Compressor<'static>,
    zstd_scratch: Vec<u8>,
}

impl ObjectCompressor {
    pub fn new(compression_level: i32) -> Result<Self> {
        Ok(Self {
            compressor: Compressor::new(compression_level).location(loc!())?,
            zstd_scratch: Vec::new(),
        })
    }

    pub fn write_framed<W: Write>(&mut self, data: &[u8], stream: &mut W) -> Result<usize> {
        data.len().framed_write(stream).location(loc!())?;
        if data.len() > MAX_RAW_OBJECT_BYTES {
            let bound = zstd_safe::compress_bound(data.len());
            self.zstd_scratch.clear();
            self.zstd_scratch.reserve(bound);
            let compressed_len = self
                .compressor
                .compress_to_buffer(data, &mut self.zstd_scratch)
                .location(loc!())?;
            1u8.framed_write(stream).location(loc!())?;
            compressed_len.framed_write(stream).location(loc!())?;
            stream
                .write_all(&self.zstd_scratch[..compressed_len])
                .location(loc!())?;
            Ok(compressed_len)
        } else {
            0u8.framed_write(stream).location(loc!())?;
            stream.write_all(data).location(loc!())?;
            Ok(data.len())
        }
    }
}

/// Resizes `buf` only when `len` exceeds the high-water mark (`AlignedVec::resize`
/// zero-fills newly grown bytes) so messages within the high-water mark slice
/// `&mut buf[..len]` without re-zeroing. Callers must validate `len` before
/// calling so a corrupt header cannot trigger multi-GiB allocations.
fn grow_output(buf: &mut AlignedVec, len: usize) -> &mut [u8] {
    if len > buf.len() {
        buf.resize(len, 0);
    }
    &mut buf[..len]
}

pub struct ObjectDecompressor {
    decompressor: Decompressor<'static>,
    compressed_scratch: Vec<u8>,
    output_buf: AlignedVec,
}

impl ObjectDecompressor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            decompressor: Decompressor::new().location(loc!())?,
            compressed_scratch: Vec::new(),
            output_buf: AlignedVec::with_capacity(64 * 1024),
        })
    }

    #[instrument(skip_all, level = "debug")]
    pub fn read_framed_with<R: Read, T>(
        &mut self,
        stream: &mut R,
        f: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        let uncompressed_size = usize::framed_read(stream).location(loc!())?;
        let is_compressed = match u8::framed_read(stream).location(loc!())? {
            0 => false,
            1 => true,
            tag => bail!("invalid object compression tag: {tag}"),
        };

        let output_slice: &[u8] = if is_compressed {
            let compressed_len = usize::framed_read(stream).location(loc!())?;
            self.compressed_scratch.clear();
            let read = stream
                .take(compressed_len as u64)
                .read_to_end(&mut self.compressed_scratch)
                .location(loc!())?;
            if read != compressed_len {
                bail!(
                    "unexpected EOF reading compressed object: expected {compressed_len} bytes, got {read}"
                );
            }
            let frame_size = zstd_safe::get_frame_content_size(&self.compressed_scratch)
                .ok()
                .flatten();
            if frame_size != Some(uncompressed_size as u64) {
                bail!(
                    "zstd frame content size {frame_size:?} does not match header uncompressed_size {uncompressed_size}"
                );
            }
            let out = grow_output(&mut self.output_buf, uncompressed_size);
            // zstd enforces decoded length == `frameContentSize` (validated above).
            self.decompressor
                .decompress_to_buffer(&self.compressed_scratch, out)
                .location(loc!())?;
            out
        } else {
            if uncompressed_size > MAX_RAW_OBJECT_BYTES {
                bail!(
                    "uncompressed object length {uncompressed_size} exceeds MAX_RAW_OBJECT_BYTES ({MAX_RAW_OBJECT_BYTES})"
                );
            }
            let out = grow_output(&mut self.output_buf, uncompressed_size);
            stream.read_exact(out).location(loc!())?;
            out
        };

        f(output_slice)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RawBufferHeader {
    pub client: ClientId,
    pub surface: WlSurfaceId,
    pub metadata: BufferMetadata,
    pub n_tiles: u32,
}

const RAW_BUFFER_HEADER_LEN: usize = 33;

impl Framed for RawBufferHeader {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        let mut buf = [0u8; RAW_BUFFER_HEADER_LEN];
        buf[0..8].copy_from_slice(&self.client.0.to_be_bytes());
        buf[8..16].copy_from_slice(&self.surface.0.to_be_bytes());
        buf[16..20].copy_from_slice(&self.metadata.width.to_be_bytes());
        buf[20..24].copy_from_slice(&self.metadata.height.to_be_bytes());
        buf[24..28].copy_from_slice(&self.metadata.stride.to_be_bytes());
        buf[28] = match self.metadata.format {
            BufferFormat::Argb8888 => 0,
            BufferFormat::Xrgb8888 => 1,
        };
        buf[29..33].copy_from_slice(&self.n_tiles.to_be_bytes());
        stream.write_all(&buf).location(loc!())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let mut buf = [0u8; RAW_BUFFER_HEADER_LEN];
        stream.read_exact(&mut buf).location(loc!())?;
        let client = ClientId(u64::from_be_bytes(buf[0..8].try_into().unwrap()));
        let surface = WlSurfaceId(u64::from_be_bytes(buf[8..16].try_into().unwrap()));
        let width = i32::from_be_bytes(buf[16..20].try_into().unwrap());
        let height = i32::from_be_bytes(buf[20..24].try_into().unwrap());
        let stride = i32::from_be_bytes(buf[24..28].try_into().unwrap());
        let format = match buf[28] {
            0 => BufferFormat::Argb8888,
            1 => BufferFormat::Xrgb8888,
            tag => bail!("invalid BufferFormat tag: {tag}"),
        };
        let n_tiles = u32::from_be_bytes(buf[29..33].try_into().unwrap());
        Ok(Self {
            client,
            surface,
            metadata: BufferMetadata {
                width,
                height,
                stride,
                format,
            },
            n_tiles,
        })
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TileChannel {
    Omitted,
    ConstantDelta(u8),
    Uncompressed(Vec<u8>),
    Zstd(Vec<u8>),
}

impl TileChannel {
    fn compressed_len(&self) -> usize {
        match self {
            Self::Omitted => 0,
            Self::ConstantDelta(_) => 1,
            Self::Uncompressed(data) | Self::Zstd(data) => data.len(),
        }
    }

    fn write_framed<W: Write>(&self, stream: &mut W) -> Result<()> {
        match self {
            Self::Omitted => bail!("attempted to write omitted channel"),
            Self::Uncompressed(data) => {
                0u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
            Self::Zstd(data) => {
                1u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
            Self::ConstantDelta(first_byte) => {
                2u8.framed_write(stream).location(loc!())?;
                first_byte.framed_write(stream).location(loc!())?;
            },
        }
        Ok(())
    }

    fn read_framed<R: Read>(stream: &mut R) -> Result<Self> {
        match u8::framed_read(stream).location(loc!())? {
            0 => Ok(Self::Uncompressed(
                Vec::<u8>::framed_read(stream).location(loc!())?,
            )),
            1 => Ok(Self::Zstd(Vec::<u8>::framed_read(stream).location(loc!())?)),
            2 => Ok(Self::ConstantDelta(
                u8::framed_read(stream).location(loc!())?,
            )),
            tag => bail!("invalid TileChannel tag: {tag}"),
        }
    }
}

impl fmt::Debug for TileChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Omitted => write!(f, "Omitted"),
            Self::ConstantDelta(b) => write!(f, "ConstantDelta({b})"),
            Self::Uncompressed(v) => write!(f, "Uncompressed([{}])", v.len()),
            Self::Zstd(v) => write!(f, "Zstd([{}])", v.len()),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompressedTile {
    pub tile_idx: u32,
    pub channels: [TileChannel; 4],
}

impl CompressedTile {
    fn compressed_len(&self) -> usize {
        self.channels.iter().map(TileChannel::compressed_len).sum()
    }

    fn write_framed<W: Write>(&self, has_alpha: bool, stream: &mut W) -> Result<()> {
        self.tile_idx.framed_write(stream).location(loc!())?;
        self.channels[0].write_framed(stream).location(loc!())?;
        self.channels[1].write_framed(stream).location(loc!())?;
        self.channels[2].write_framed(stream).location(loc!())?;
        if has_alpha {
            self.channels[3].write_framed(stream).location(loc!())?;
        }
        Ok(())
    }

    fn read_framed<R: Read>(has_alpha: bool, stream: &mut R) -> Result<Self> {
        let tile_idx = u32::framed_read(stream).location(loc!())?;
        let ch0 = TileChannel::read_framed(stream).location(loc!())?;
        let ch1 = TileChannel::read_framed(stream).location(loc!())?;
        let ch2 = TileChannel::read_framed(stream).location(loc!())?;
        let ch3 = if has_alpha {
            TileChannel::read_framed(stream).location(loc!())?
        } else {
            TileChannel::Omitted
        };
        Ok(Self {
            tile_idx,
            channels: [ch0, ch1, ch2, ch3],
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompressedTiles {
    pub format: BufferFormat,
    pub total_pixels: usize,
    pub tiles: Vec<CompressedTile>,
}

impl CompressedTiles {
    pub fn size(&self) -> usize {
        self.tiles.iter().map(CompressedTile::compressed_len).sum()
    }
}

const MAX_COALESCED_TILE_BYTES: usize = 256 * 1024;

pub struct StreamingCompressedTiles {
    pub header: RawBufferHeader,
    pub tiles: Receiver<CompressedTile>,
}

impl StreamingCompressedTiles {
    pub fn uncompressed_size(&self) -> usize {
        let n_channels = match self.header.metadata.format {
            BufferFormat::Argb8888 => 4,
            BufferFormat::Xrgb8888 => 3,
        };
        let row_pixels = (self.header.metadata.stride as usize) / 4;
        (self.header.metadata.height as usize) * row_pixels * n_channels
    }

    pub fn framed_write<W: Write>(self, stream: &mut W) -> Result<usize> {
        let has_alpha = self.header.metadata.format == BufferFormat::Argb8888;
        self.header.framed_write(stream).location(loc!())?;
        // Flush the header immediately so `wprsc` can prepare the destination
        // `wl_shm` canvas while the first tile is still compressing.
        stream.flush().location(loc!())?;

        let mut remaining = self.header.n_tiles as usize;
        let mut total_compressed_size = 0;

        while remaining > 0 {
            let tile = self
                .tiles
                .recv()
                .context(loc!(), "tile sender dropped before sending all tiles")?;
            let mut batch_bytes = tile.compressed_len();
            total_compressed_size += batch_bytes;
            tile.write_framed(has_alpha, stream).location(loc!())?;
            remaining -= 1;

            while remaining > 0
                && batch_bytes < MAX_COALESCED_TILE_BYTES
                && let Ok(tile) = self.tiles.try_recv()
            {
                let tile_len = tile.compressed_len();
                batch_bytes += tile_len;
                total_compressed_size += tile_len;
                tile.write_framed(has_alpha, stream).location(loc!())?;
                remaining -= 1;
            }

            stream.flush().location(loc!())?;
        }

        Ok(total_compressed_size)
    }
}

pub struct CompressedTileStream {
    pub header: RawBufferHeader,
    pub tiles: Receiver<Result<CompressedTile>>,
}

impl CompressedTileStream {
    pub fn read_from_stream<R: Read, F>(stream: &mut R, on_stream: F) -> Result<()>
    where
        F: FnOnce(Self) -> Result<()>,
    {
        let header = RawBufferHeader::framed_read(stream).location(loc!())?;
        let has_alpha = header.metadata.format == BufferFormat::Argb8888;
        let n_tiles = header.n_tiles;
        let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
        on_stream(Self {
            header,
            tiles: tiles_rx,
        })
        .location(loc!())?;

        for _ in 0..n_tiles {
            match CompressedTile::read_framed(has_alpha, stream) {
                Ok(tile) => {
                    // Drain all `n_tiles` off `stream` even if the consumer dropped
                    // `tiles_rx` early (e.g. when `handle_buffer` fails to decode) so
                    // unread tile bytes cannot desynchronize the next `MessageType` on
                    // the socket.
                    _ = tiles_tx.send(Ok(tile));
                },
                Err(err) => {
                    _ = tiles_tx.send(Err(anyhow!("{err:#}")));
                    return Err(err).location(loc!());
                },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usize_framed_write_u32_overflow() {
        assert!(
            ((u32::MAX as usize) + 1)
                .framed_write(&mut Vec::new())
                .is_err()
        );
    }

    #[test]
    fn test_vec_u8_framed_read_short_read_and_roundtrip() {
        assert!(Vec::<u8>::framed_read(&mut [0u8, 0, 0, 4, 1, 2].as_slice()).is_err());
        assert!(Vec::<u8>::framed_read(&mut [0xFFu8; 4].as_slice()).is_err());
        assert_eq!(
            Vec::<u8>::framed_read(&mut [0u8, 0, 0, 2, 1, 2].as_slice()).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn test_raw_buffer_header_framing_and_errors() {
        for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
            let hdr = RawBufferHeader {
                client: ClientId(42),
                surface: WlSurfaceId(99),
                metadata: BufferMetadata {
                    width: 640,
                    height: 480,
                    stride: 640 * 4 + 16,
                    format,
                },
                n_tiles: 5,
            };
            let mut wire = Vec::new();
            hdr.framed_write(&mut wire).unwrap();
            let decoded = RawBufferHeader::framed_read(&mut wire.as_slice()).unwrap();
            assert_eq!(decoded, hdr);
            assert_eq!(decoded.metadata.validate().unwrap(), 480 * (640 + 4));
        }

        // Invalid BufferFormat tag at byte 28 must return Err.
        let mut bad = [0u8; RAW_BUFFER_HEADER_LEN];
        bad[28] = 9;
        assert!(RawBufferHeader::framed_read(&mut bad.as_slice()).is_err());

        // CompressedTile Xrgb8888 / Argb8888 framing and invalid channel tags.
        let xrgb_tile = CompressedTile {
            tile_idx: 3,
            channels: [
                TileChannel::ConstantDelta(7),
                TileChannel::Uncompressed(vec![1, 2, 3, 4]),
                TileChannel::Zstd(vec![9, 8]),
                TileChannel::Omitted,
            ],
        };
        let mut tile_wire = Vec::new();
        xrgb_tile.write_framed(false, &mut tile_wire).unwrap();
        let decoded_xrgb = CompressedTile::read_framed(false, &mut tile_wire.as_slice()).unwrap();
        assert_eq!(decoded_xrgb, xrgb_tile);
        assert!(xrgb_tile.write_framed(true, &mut Vec::new()).is_err());

        let argb_tile = CompressedTile {
            tile_idx: 4,
            channels: [
                TileChannel::ConstantDelta(7),
                TileChannel::Uncompressed(vec![1, 2, 3, 4]),
                TileChannel::Zstd(vec![9, 8]),
                TileChannel::ConstantDelta(0x80),
            ],
        };
        let mut argb_wire = Vec::new();
        argb_tile.write_framed(true, &mut argb_wire).unwrap();
        let decoded_argb = CompressedTile::read_framed(true, &mut argb_wire.as_slice()).unwrap();
        assert_eq!(decoded_argb, argb_tile);

        assert!(TileChannel::Omitted.write_framed(&mut Vec::new()).is_err());
        assert!(TileChannel::read_framed(&mut [9u8].as_slice()).is_err());
    }

    #[test]
    fn test_streaming_compressed_tiles_and_errors() {
        let header = RawBufferHeader {
            client: ClientId(7),
            surface: WlSurfaceId(11),
            metadata: BufferMetadata {
                width: 64,
                height: 64,
                stride: 64 * 4 + 16,
                format: BufferFormat::Xrgb8888,
            },
            n_tiles: 2,
        };
        let tile0 = CompressedTile {
            tile_idx: 0,
            channels: [
                TileChannel::ConstantDelta(1),
                TileChannel::Uncompressed(vec![2, 3, 4, 5]),
                TileChannel::Zstd(vec![7, 8, 9]),
                TileChannel::Omitted,
            ],
        };
        let tile1 = CompressedTile {
            tile_idx: 1,
            channels: [
                TileChannel::Uncompressed(vec![9, 8, 7, 6, 5]),
                TileChannel::ConstantDelta(5),
                TileChannel::Zstd(vec![1, 2, 3, 4]),
                TileChannel::Omitted,
            ],
        };
        assert_eq!(
            CompressedTiles {
                format: BufferFormat::Xrgb8888,
                total_pixels: 64 * 68,
                tiles: vec![tile0.clone(), tile1.clone()],
            }
            .size(),
            18
        );

        // `StreamingCompressedTiles::uncompressed_size` accounts for padded
        // `stride` (`68` pixels/row) and `BufferFormat` channel count.
        for (format, expected) in [
            (BufferFormat::Xrgb8888, 64 * 68 * 3),
            (BufferFormat::Argb8888, 64 * 68 * 4),
        ] {
            let (_tx_u, rx_u) = crossbeam_channel::unbounded();
            let mut h = header.clone();
            h.metadata.format = format;
            let s = StreamingCompressedTiles {
                header: h,
                tiles: rx_u,
            };
            assert_eq!(s.uncompressed_size(), expected);
        }

        // Full 2-tile roundtrip (exercises `recv()` + `try_recv()` coalescing).
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(tile0.clone()).unwrap();
        tx.send(tile1.clone()).unwrap();
        drop(tx);
        let streaming = StreamingCompressedTiles {
            header: header.clone(),
            tiles: rx,
        };
        let mut wire = Vec::new();
        let written_bytes = streaming.framed_write(&mut wire).unwrap();
        assert_eq!(written_bytes, 18);

        let mut received_header = None;
        let mut tiles_rx_opt = None;
        CompressedTileStream::read_from_stream(&mut wire.as_slice(), |stream| {
            received_header = Some(stream.header);
            tiles_rx_opt = Some(stream.tiles);
            Ok(())
        })
        .unwrap();
        assert_eq!(received_header.unwrap(), header);
        let received_tiles: Vec<CompressedTile> = tiles_rx_opt
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(received_tiles, vec![tile0.clone(), tile1]);

        // Verify immediate header flush, `MAX_COALESCED_TILE_BYTES` batch cap
        // (flushing after `t0 + t1` even with `t2` already queued), and
        // `try_recv()`-empty flush before `t3` arrives.
        struct FlushWriter {
            buf: Vec<u8>,
            flush_offsets: Vec<usize>,
            send_last_at: usize,
            deferred: Option<(crossbeam_channel::Sender<CompressedTile>, CompressedTile)>,
        }
        impl Write for FlushWriter {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                self.buf.extend_from_slice(data);
                Ok(data.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                self.flush_offsets.push(self.buf.len());
                if self.buf.len() >= self.send_last_at
                    && let Some((tx_last, last_tile)) = self.deferred.take()
                {
                    tx_last.send(last_tile).unwrap();
                }
                Ok(())
            }
        }

        let half_cap = MAX_COALESCED_TILE_BYTES / 2;
        let large0 = CompressedTile {
            tile_idx: 0,
            channels: [
                TileChannel::Uncompressed(vec![0xAA; half_cap - 2]),
                TileChannel::ConstantDelta(1),
                TileChannel::ConstantDelta(2),
                TileChannel::Omitted,
            ],
        };
        let large1 = CompressedTile {
            tile_idx: 1,
            channels: [
                TileChannel::Uncompressed(vec![0xBB; half_cap - 2]),
                TileChannel::ConstantDelta(3),
                TileChannel::ConstantDelta(4),
                TileChannel::Omitted,
            ],
        };
        let small2 = CompressedTile {
            tile_idx: 2,
            channels: [
                TileChannel::ConstantDelta(5),
                TileChannel::ConstantDelta(6),
                TileChannel::ConstantDelta(7),
                TileChannel::Omitted,
            ],
        };
        let small3 = CompressedTile {
            tile_idx: 3,
            channels: [
                TileChannel::ConstantDelta(8),
                TileChannel::ConstantDelta(9),
                TileChannel::ConstantDelta(10),
                TileChannel::Omitted,
            ],
        };
        assert_eq!(
            large0.compressed_len() + large1.compressed_len(),
            MAX_COALESCED_TILE_BYTES
        );

        let mut batch_header = header.clone();
        batch_header.n_tiles = 4;
        let mut expected_hdr_wire = Vec::new();
        batch_header.framed_write(&mut expected_hdr_wire).unwrap();
        let mut expected_b1 = expected_hdr_wire.clone();
        large0.write_framed(false, &mut expected_b1).unwrap();
        large1.write_framed(false, &mut expected_b1).unwrap();
        let mut expected_b2 = expected_b1.clone();
        small2.write_framed(false, &mut expected_b2).unwrap();
        let mut expected_b3 = expected_b2.clone();
        small3.write_framed(false, &mut expected_b3).unwrap();

        let (tx_batch, rx_batch) = crossbeam_channel::unbounded();
        tx_batch.send(large0).unwrap();
        tx_batch.send(large1).unwrap();
        tx_batch.send(small2).unwrap();
        let mut writer = FlushWriter {
            buf: Vec::new(),
            flush_offsets: Vec::new(),
            send_last_at: expected_b2.len(),
            deferred: Some((tx_batch, small3)),
        };
        let batch_written = StreamingCompressedTiles {
            header: batch_header,
            tiles: rx_batch,
        }
        .framed_write(&mut writer)
        .unwrap();
        assert_eq!(batch_written, MAX_COALESCED_TILE_BYTES + 6);
        assert_eq!(
            writer.flush_offsets,
            vec![
                expected_hdr_wire.len(),
                expected_b1.len(),
                expected_b2.len(),
                expected_b3.len(),
            ]
        );
        assert_eq!(writer.buf, expected_b3);

        // Dropped sender before `n_tiles` are sent must return `Err` from the
        // outer `recv()`, both before the first tile and after the coalescing
        // `try_recv()` loop stops on disconnect.
        for sent_before_drop in [0usize, 1usize] {
            let (tx_short, rx_short) = crossbeam_channel::unbounded();
            for _ in 0..sent_before_drop {
                tx_short.send(tile0.clone()).unwrap();
            }
            drop(tx_short);
            let short_stream = StreamingCompressedTiles {
                header: header.clone(),
                tiles: rx_short,
            };
            assert!(short_stream.framed_write(&mut Vec::new()).is_err());
        }

        // Mid-stream truncation after 1 of 2 tiles must return `Err` from
        // `read_from_stream` AND forward an `Err` item onto `tiles_rx`.
        let mut truncated_wire = Vec::new();
        header.framed_write(&mut truncated_wire).unwrap();
        tile0.write_framed(false, &mut truncated_wire).unwrap();
        let mut truncated_rx_opt = None;
        assert!(
            CompressedTileStream::read_from_stream(&mut truncated_wire.as_slice(), |stream| {
                truncated_rx_opt = Some(stream.tiles);
                Ok(())
            })
            .is_err()
        );
        let items: Vec<Result<CompressedTile>> = truncated_rx_opt.unwrap().into_iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap(), &tile0);
        assert!(items[1].is_err());

        // Early drop of `stream.tiles` inside `on_stream` must still drain all
        // `n_tiles` off `stream` without consuming trailing bytes of the next frame.
        let mut wire_with_trailer = wire.clone();
        wire_with_trailer.push(0xFE);
        let mut reader = wire_with_trailer.as_slice();
        CompressedTileStream::read_from_stream(&mut reader, |stream| {
            drop(stream.tiles);
            Ok(())
        })
        .unwrap();
        assert_eq!(reader, &[0xFE]);

        // Returning `Err` from `on_stream` must abort immediately after reading
        // `RawBufferHeader` without consuming tile bytes.
        let mut header_only = Vec::new();
        header.framed_write(&mut header_only).unwrap();
        let mut err_reader = wire.as_slice();
        assert!(
            CompressedTileStream::read_from_stream(&mut err_reader, |_| {
                bail!("consumer rejected stream")
            })
            .is_err()
        );
        assert_eq!(err_reader, &wire[header_only.len()..]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_object_compressor_roundtrip_and_errors() {
        let mut comp = ObjectCompressor::new(1).unwrap();
        let mut decomp = ObjectDecompressor::new().unwrap();

        // Oversized `uncompressed_size` (`> MAX_RAW_OBJECT_BYTES`) on a raw
        // frame must return `Err` without growing `output_buf`.
        let mut oversized_raw = Vec::new();
        (MAX_RAW_OBJECT_BYTES + 1)
            .framed_write(&mut oversized_raw)
            .unwrap();
        0u8.framed_write(&mut oversized_raw).unwrap();
        oversized_raw.resize(oversized_raw.len() + MAX_RAW_OBJECT_BYTES + 1, 0xAB);
        assert!(
            decomp
                .read_framed_with(&mut oversized_raw.as_slice(), |_| Ok(()))
                .is_err()
        );
        assert!(decomp.output_buf.is_empty());

        let small: Vec<u8> = (0..512).map(|i| (i % 251) as u8).collect();
        let boundary = vec![0x5Au8; MAX_RAW_OBJECT_BYTES];
        let above_boundary: Vec<u8> = (0..MAX_RAW_OBJECT_BYTES + 1)
            .map(|i| (i % 251) as u8)
            .collect();
        let large: Vec<u8> = (0..16_384).map(|i| ((i * 7 + 3) % 256) as u8).collect();

        // Trailing `&small` after `&large` exercises shrinking below `output_buf`'s high-water mark.
        for payload in [&small, &boundary, &above_boundary, &large, &small] {
            let mut wire = Vec::new();
            let written = comp.write_framed(payload, &mut wire).unwrap();
            let expect_compressed = payload.len() > MAX_RAW_OBJECT_BYTES;
            assert_eq!(wire[4], u8::from(expect_compressed));
            assert_eq!(
                written,
                if expect_compressed {
                    wire.len() - 9
                } else {
                    payload.len()
                }
            );
            let mut reader = wire.as_slice();
            let out = decomp
                .read_framed_with(&mut reader, |slice| Ok(slice.to_vec()))
                .unwrap();
            assert_eq!(&out, payload);
            assert!(reader.is_empty());
        }

        // Mismatched `uncompressed_size` header on a Zstd-compressed frame must
        // return `Err` without growing `output_buf`.
        let mut mismatched_len = Vec::new();
        comp.write_framed(&large, &mut mismatched_len).unwrap();
        assert_eq!(mismatched_len[4], 1); // Zstd-compressed tag after 4-byte u32 length
        mismatched_len[0..4].copy_from_slice(&(large.len() as u32 + 64).to_be_bytes());
        assert!(
            decomp
                .read_framed_with(&mut mismatched_len.as_slice(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(decomp.output_buf.len(), large.len());

        // Zstd frame omitting `frameContentSize` (`None`) must return `Err`
        // without growing `output_buf`.
        let big = vec![0u8; 1024 * 1024];
        let mut no_size_comp = Compressor::new(1).unwrap();
        no_size_comp
            .set_parameter(zstd_safe::CParameter::ContentSizeFlag(false))
            .unwrap();
        let no_size_payload = no_size_comp.compress(&big).unwrap();
        let mut no_size_wire = Vec::new();
        big.len().framed_write(&mut no_size_wire).unwrap();
        1u8.framed_write(&mut no_size_wire).unwrap();
        no_size_payload
            .len()
            .framed_write(&mut no_size_wire)
            .unwrap();
        no_size_wire.extend_from_slice(&no_size_payload);
        assert!(
            decomp
                .read_framed_with(&mut no_size_wire.as_slice(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(decomp.output_buf.len(), large.len());

        // Truncated compressed payload (`read != compressed_len`) must return `Err`.
        let mut short_compressed = Vec::new();
        comp.write_framed(&large, &mut short_compressed).unwrap();
        short_compressed[5..9].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(
            decomp
                .read_framed_with(&mut short_compressed.as_slice(), |_| Ok(()))
                .is_err()
        );

        // Invalid compression tag must return Err.
        let mut bad_tag = Vec::new();
        16usize.framed_write(&mut bad_tag).unwrap();
        9u8.framed_write(&mut bad_tag).unwrap();
        assert!(
            decomp
                .read_framed_with(&mut bad_tag.as_slice(), |_| Ok(()))
                .is_err()
        );
    }
}
