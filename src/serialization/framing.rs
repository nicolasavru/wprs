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
use std::str::FromStr;

use crossbeam_channel::Receiver;
use rkyv::Archive;
use rkyv::Deserialize;
use rkyv::Serialize;
use rkyv::util::AlignedVec;
use static_assertions::const_assert;
use zstd::bulk::Compressor;
use zstd::bulk::Decompressor;
use zstd::zstd_safe;

use crate::prelude::*;
use crate::serialization::ClientId;
use crate::serialization::geometry::Rectangle;
use crate::serialization::wayland::BufferFormat;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::WlSurfaceId;

const_assert!(mem::size_of::<usize>() >= mem::size_of::<u32>());

/// Minimum allowed JPEG quality for [`CompressionMode::Hybrid`].
pub const MIN_JPEG_QUALITY: u8 = 1;

/// Maximum allowed JPEG quality for [`CompressionMode::Hybrid`].
pub const MAX_JPEG_QUALITY: u8 = 100;

/// Default JPEG quality for [`CompressionMode::Hybrid`].
pub const DEFAULT_JPEG_QUALITY: u8 = 90;

/// Chroma subsampling mode for lossy TurboJPEG tiles in hybrid compression.
#[derive(
    Archive,
    Deserialize,
    Serialize,
    serde_derive::Deserialize,
    serde_derive::Serialize,
    Debug,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    Default,
)]
pub enum JpegSubsamp {
    #[default]
    Sub444,
    Sub422,
    Sub420,
}

impl From<JpegSubsamp> for turbojpeg::Subsamp {
    fn from(subsamp: JpegSubsamp) -> Self {
        match subsamp {
            JpegSubsamp::Sub444 => Self::None,
            JpegSubsamp::Sub422 => Self::Sub2x1,
            JpegSubsamp::Sub420 => Self::Sub2x2,
        }
    }
}

impl fmt::Display for JpegSubsamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sub444 => "444",
            Self::Sub422 => "422",
            Self::Sub420 => "420",
        })
    }
}

impl FromStr for JpegSubsamp {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "444" => Ok(Self::Sub444),
            "422" => Ok(Self::Sub422),
            "420" => Ok(Self::Sub420),
            _ => bail!("invalid JPEG subsampling {s:?}: expected \"444\", \"422\", or \"420\""),
        }
    }
}

/// Client-selected frame compression mode negotiated over [`crate::serialization::Event::SetCompressionMode`].
#[derive(
    Archive,
    Deserialize,
    Serialize,
    serde_derive::Deserialize,
    serde_derive::Serialize,
    Debug,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Default,
)]
pub enum CompressionMode {
    #[default]
    Lossless,
    Hybrid {
        quality: u8,
        subsamp: JpegSubsamp,
    },
}

impl CompressionMode {
    pub const fn hybrid_default() -> Self {
        Self::Hybrid {
            quality: DEFAULT_JPEG_QUALITY,
            subsamp: JpegSubsamp::Sub444,
        }
    }

    pub fn validate(self) -> Result<()> {
        match self {
            Self::Lossless => Ok(()),
            Self::Hybrid { quality, .. } => {
                if !(MIN_JPEG_QUALITY..=MAX_JPEG_QUALITY).contains(&quality) {
                    bail!(
                        "invalid JPEG quality {quality}: expected {MIN_JPEG_QUALITY}..={MAX_JPEG_QUALITY}"
                    );
                }
                Ok(())
            },
        }
    }
}

/// Nominal 2D tile width in pixels.
pub const TILE_W: usize = 256;

/// Nominal 2D tile height in pixels.
pub const TILE_H: usize = 256;

/// Sub-tile quadrant width in pixels for incremental dirty-region skipping.
pub const SUBTILE_W: usize = 128;

/// Sub-tile quadrant height in pixels for incremental dirty-region skipping.
pub const SUBTILE_H: usize = 128;

/// Maximum pixels in any self-contained L2-resident tile (`256 * 256 = 65,536`,
/// `192 KiB` packed 3-byte `SubGreen(BGR)` payload).
pub const TILE_PIXELS: usize = TILE_W * TILE_H;

/// Upper bound on planned tiles in a single [`RawBufferHeader`] (covers 8K
/// down to `16x16` tiles while capping header memory at `<= 4 MiB`).
pub const MAX_FRAME_TILES: usize = 262_144;

/// Maximum number of slots in the connection-wide inter-frame tile cache.
/// Holds two full 4K screens (`270` `256x256` tiles = `63.3 MiB`) or up to
/// `1024` `128x128` quadrants (`64 MiB`).
pub const MAX_TILE_CACHE_SLOTS: usize = 1024;

/// Hard cap on total decoded pixel bytes stored across all tile cache slots
/// (`64 MiB` = `16,777,216` BGRA pixels).
pub const MAX_TILE_CACHE_BYTES: usize = 64 * 1024 * 1024;

const_assert!(MAX_TILE_CACHE_SLOTS == 1024);

/// Which region of a cached tile slot a [`TilePayload::CacheHit`] copies.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum CacheHitRegion {
    /// Entire cached tile (`rect.w == slot.w && rect.h == slot.h`).
    Full = 0,
    /// Top-left `<= 128x128` quadrant of a cached splittable tile.
    TopLeft = 1,
    /// Top-right `<= 128x128` quadrant of a cached tile with `slot.w > 128`.
    TopRight = 2,
    /// Bottom-left `<= 128x128` quadrant of a cached tile with `slot.h > 128`.
    BottomLeft = 3,
    /// Bottom-right `<= 128x128` quadrant of a cached tile with `slot.w > 128 && slot.h > 128`.
    BottomRight = 4,
}

const CACHE_SLOT_MASK: u16 = (MAX_TILE_CACHE_SLOTS - 1) as u16;
const CACHE_REGION_SHIFT: u32 = 10;

/// Compact wire reference to a cached tile slot (or one of its `128x128`
/// quadrants), packed into 16 bits (`10`-bit `slot_id` + `3`-bit `region`).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct CacheHitRef {
    pub slot_id: u16,
    pub region: CacheHitRegion,
}

impl CacheHitRef {
    #[inline]
    pub fn to_u16(self) -> Result<u16> {
        if self.slot_id as usize >= MAX_TILE_CACHE_SLOTS {
            bail!(
                "tile cache slot_id {} out of range (max {})",
                self.slot_id,
                MAX_TILE_CACHE_SLOTS
            );
        }
        Ok(self.slot_id | (u16::from(self.region as u8) << CACHE_REGION_SHIFT))
    }

    #[inline]
    pub fn from_u16(raw: u16) -> Result<Self> {
        let region_bits = raw >> CACHE_REGION_SHIFT;
        let region = match region_bits {
            0 => CacheHitRegion::Full,
            1 => CacheHitRegion::TopLeft,
            2 => CacheHitRegion::TopRight,
            3 => CacheHitRegion::BottomLeft,
            4 => CacheHitRegion::BottomRight,
            _ => bail!("invalid CacheHitRef region bits: {region_bits} (raw={raw:#06x})"),
        };
        let slot_id = raw & CACHE_SLOT_MASK;
        debug_assert!((slot_id as usize) < MAX_TILE_CACHE_SLOTS);
        Ok(Self { slot_id, region })
    }

    /// Validates that `self` is compatible with destination tile `rect`.
    #[inline]
    pub fn validate_for_rect(self, rect: TileRect) -> Result<()> {
        if self.slot_id as usize >= MAX_TILE_CACHE_SLOTS {
            bail!(
                "tile cache slot_id {} out of range (max {})",
                self.slot_id,
                MAX_TILE_CACHE_SLOTS
            );
        }
        if self.region != CacheHitRegion::Full
            && (rect.w > SUBTILE_W as u32 || rect.h > SUBTILE_H as u32)
        {
            bail!(
                "quadrant CacheHitRegion::{:?} incompatible with tile rect {rect:?}",
                self.region
            );
        }
        Ok(())
    }
}

/// Returns the exact bitpacked index byte length for `n_pixels` pixels indexed
/// into an `n_colors`-entry palette (`1`/`2`/`4`/`8` bpp for `2`/`3..=4`/
/// `5..=16`/`17..=256` colors).
#[inline]
pub(crate) fn packed_palette_indices_len(n_pixels: usize, n_colors: usize) -> usize {
    if n_colors <= 1 {
        0
    } else if n_colors <= 2 {
        n_pixels.div_ceil(8)
    } else if n_colors <= 4 {
        n_pixels.div_ceil(4)
    } else if n_colors <= 16 {
        n_pixels.div_ceil(2)
    } else {
        n_pixels
    }
}

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

impl Framed for u16 {
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

/// Reads a `u32`-length-prefixed byte vector after enforcing `len <= max_len`
/// before allocating.
fn read_bounded_vec<R: Read>(stream: &mut R, max_len: usize) -> Result<Vec<u8>> {
    let len = u32::framed_read(stream).location(loc!())? as usize;
    if len > max_len {
        bail!("framed payload length {len} exceeds maximum {max_len}");
    }
    read_exact_vec(stream, len)
}

/// Reads an exact known `len` bytes from `stream`, capping initial capacity so
/// truncated streams cannot trigger oversized pre-allocations.
fn read_exact_vec<R: Read>(stream: &mut R, len: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(len.min(TILE_PIXELS));
    let read = stream
        .take(len as u64)
        .read_to_end(&mut buf)
        .location(loc!())?;
    if read != len {
        bail!("unexpected EOF reading framed bytes: expected {len} bytes, got {read}");
    }
    Ok(buf)
}

impl Framed for Vec<u8> {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        self.len().framed_write(stream).location(loc!())?;
        stream.write_all(self).location(loc!())?;
        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let len = u32::framed_read(stream).location(loc!())? as usize;
        read_exact_vec(stream, len)
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct RowBand {
    pub y0: u32,
    pub y1: u32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

const TILE_RECT_LEN: usize = 16;

impl TileRect {
    /// Returns the pixel count if non-empty and within [`TILE_PIXELS`].
    #[inline]
    pub fn checked_pixels(self) -> Result<usize> {
        if self.w == 0 || self.h == 0 {
            bail!("empty TileRect: {self:?}");
        }
        let pixels = u64::from(self.w)
            .checked_mul(u64::from(self.h))
            .context(loc!(), "TileRect pixel count overflow")?;
        if pixels > TILE_PIXELS as u64 {
            bail!("TileRect {self:?} pixel count {pixels} exceeds TILE_PIXELS ({TILE_PIXELS})");
        }
        Ok(pixels as usize)
    }

    /// Validates that `self` is non-empty, within [`TILE_PIXELS`], and strictly
    /// inside `[0, width) x [0, height)` without integer overflow.
    #[inline]
    pub fn validate_in_bounds(self, width: u32, height: u32) -> Result<usize> {
        let pixels = self.checked_pixels().location(loc!())?;
        let x1 = self
            .x
            .checked_add(self.w)
            .context(loc!(), "TileRect x + w overflow")?;
        let y1 = self
            .y
            .checked_add(self.h)
            .context(loc!(), "TileRect y + h overflow")?;
        if x1 > width || y1 > height {
            bail!("TileRect {self:?} out of bounds for {width}x{height}");
        }
        Ok(pixels)
    }

    /// Returns `true` if `self` and `other` share at least one pixel.
    #[inline]
    pub fn overlaps(self, other: Self) -> bool {
        let ax0 = u64::from(self.x);
        let ax1 = ax0 + u64::from(self.w);
        let ay0 = u64::from(self.y);
        let ay1 = ay0 + u64::from(self.h);

        let bx0 = u64::from(other.x);
        let bx1 = bx0 + u64::from(other.w);
        let by0 = u64::from(other.y);
        let by1 = by0 + u64::from(other.h);

        ax0 < bx1 && bx0 < ax1 && ay0 < by1 && by0 < ay1
    }

    fn write_framed<W: Write>(self, stream: &mut W) -> Result<()> {
        let mut buf = [0u8; TILE_RECT_LEN];
        buf[0..4].copy_from_slice(&self.x.to_be_bytes());
        buf[4..8].copy_from_slice(&self.y.to_be_bytes());
        buf[8..12].copy_from_slice(&self.w.to_be_bytes());
        buf[12..16].copy_from_slice(&self.h.to_be_bytes());
        stream.write_all(&buf).location(loc!())
    }

    fn read_framed<R: Read>(stream: &mut R) -> Result<(Self, usize)> {
        let mut buf = [0u8; TILE_RECT_LEN];
        stream.read_exact(&mut buf).location(loc!())?;
        let rect = Self {
            x: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            y: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            w: u32::from_be_bytes(buf[8..12].try_into().unwrap()),
            h: u32::from_be_bytes(buf[12..16].try_into().unwrap()),
        };
        let pixels = rect.checked_pixels().location(loc!())?;
        Ok((rect, pixels))
    }
}

/// Vertical scroll-blit region replayed before decoding incremental tiles.
///
/// Shifts pixels within `[x, x + width) x [y, y + height)` vertically by `dy`
/// rows:
/// - `dy < 0` (content moves up): copies rows `[y + |dy|, y + height)` to
///   `[y, y + height - |dy|)`.
/// - `dy > 0` (content moves down): copies rows `[y, y + height - dy)` to
///   `[y + dy, y + height)`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ScrollRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub dy: i32,
}

const SCROLL_RECT_LEN: usize = 20;

impl ScrollRect {
    /// Validates that `self` is non-empty, strictly inside
    /// `[0, buf_width) x [0, buf_height)` without integer overflow, and shifts
    /// by a non-zero `dy` with `|dy| < height`.
    #[inline]
    pub fn validate_in_bounds(self, buf_width: u32, buf_height: u32) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            bail!("empty ScrollRect: {self:?}");
        }
        let x1 = self
            .x
            .checked_add(self.width)
            .context(loc!(), "ScrollRect x + width overflow")?;
        let y1 = self
            .y
            .checked_add(self.height)
            .context(loc!(), "ScrollRect y + height overflow")?;
        if x1 > buf_width || y1 > buf_height {
            bail!("ScrollRect {self:?} out of bounds for {buf_width}x{buf_height}");
        }
        if self.dy == 0 {
            bail!("zero dy in ScrollRect: {self:?}");
        }
        if self.dy.unsigned_abs() >= self.height {
            bail!(
                "ScrollRect dy={} shift magnitude >= height {} ({self:?})",
                self.dy,
                self.height
            );
        }
        Ok(())
    }

    /// Returns the destination rectangle overwritten by this scroll blit as a
    /// 2D pixel rectangle (`w * h` may exceed [`TILE_PIXELS`]).
    ///
    /// Caller must have validated `self` via [`Self::validate_in_bounds`].
    #[inline]
    pub fn dst_rect(self) -> TileRect {
        let shift = self.dy.unsigned_abs();
        let dst_y = if self.dy < 0 { self.y } else { self.y + shift };
        TileRect {
            x: self.x,
            y: dst_y,
            w: self.width,
            h: self.height - shift,
        }
    }

    /// Returns the destination rectangle in buffer-space damage coordinates.
    ///
    /// Caller must have validated `self` against a valid [`BufferMetadata`].
    #[inline]
    pub fn dst_damage_rect(self) -> Rectangle<i32> {
        let dst = self.dst_rect();
        Rectangle::new(dst.x as i32, dst.y as i32, dst.w as i32, dst.h as i32)
    }

    fn write_framed<W: Write>(self, stream: &mut W) -> Result<()> {
        let mut buf = [0u8; SCROLL_RECT_LEN];
        buf[0..4].copy_from_slice(&self.x.to_be_bytes());
        buf[4..8].copy_from_slice(&self.y.to_be_bytes());
        buf[8..12].copy_from_slice(&self.width.to_be_bytes());
        buf[12..16].copy_from_slice(&self.height.to_be_bytes());
        buf[16..20].copy_from_slice(&self.dy.to_be_bytes());
        stream.write_all(&buf).location(loc!())
    }

    fn read_framed<R: Read>(stream: &mut R) -> Result<Self> {
        let mut buf = [0u8; SCROLL_RECT_LEN];
        stream.read_exact(&mut buf).location(loc!())?;
        Ok(Self {
            x: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            y: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            width: u32::from_be_bytes(buf[8..12].try_into().unwrap()),
            height: u32::from_be_bytes(buf[12..16].try_into().unwrap()),
            dy: i32::from_be_bytes(buf[16..20].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RawBufferHeader {
    pub frame_seq: u64,
    pub client: ClientId,
    pub surface: WlSurfaceId,
    pub metadata: BufferMetadata,
    pub incremental: bool,
    pub scroll_rect: Option<ScrollRect>,
    /// When `true`, `wprsc` clears its connection-wide tile cache before
    /// decoding any tiles in this frame.
    pub reset_tile_cache: bool,
    /// Optional explicit tile cache slot evictions applied on `wprsc` before
    /// decoding this frame so `wprsc` strictly honors [`MAX_TILE_CACHE_BYTES`]
    /// when a larger tile replaces multiple smaller cached slots.
    pub evict_slots: Vec<u16>,
    pub tile_rects: Vec<TileRect>,
}

const RAW_BUFFER_HEADER_LEN: usize = 42;
const HEADER_FLAG_INCREMENTAL: u8 = 0x01;
const HEADER_FLAG_SCROLL_RECT: u8 = 0x02;
const HEADER_FLAG_RESET_TILE_CACHE: u8 = 0x04;
const HEADER_FLAG_EVICT_SLOTS: u8 = 0x08;
const HEADER_FLAGS_VALID: u8 = HEADER_FLAG_INCREMENTAL
    | HEADER_FLAG_SCROLL_RECT
    | HEADER_FLAG_RESET_TILE_CACHE
    | HEADER_FLAG_EVICT_SLOTS;

impl Framed for RawBufferHeader {
    fn framed_write<W: Write>(&self, stream: &mut W) -> Result<()> {
        if let Some(sr) = self.scroll_rect {
            if !self.incremental {
                bail!("non-incremental RawBufferHeader cannot carry ScrollRect");
            }
            let width = u32::try_from(self.metadata.width)
                .context(loc!(), "negative BufferMetadata width")?;
            let height = u32::try_from(self.metadata.height)
                .context(loc!(), "negative BufferMetadata height")?;
            sr.validate_in_bounds(width, height).location(loc!())?;
        }
        if self.reset_tile_cache && !self.evict_slots.is_empty() {
            bail!("RawBufferHeader cannot set both reset_tile_cache and evict_slots");
        }
        if self.evict_slots.len() > MAX_TILE_CACHE_SLOTS {
            bail!(
                "evict_slots len {} exceeds MAX_TILE_CACHE_SLOTS ({MAX_TILE_CACHE_SLOTS})",
                self.evict_slots.len()
            );
        }
        if !self.evict_slots.is_empty() {
            let mut seen = [false; MAX_TILE_CACHE_SLOTS];
            for &slot_id in &self.evict_slots {
                let idx = slot_id as usize;
                if idx >= MAX_TILE_CACHE_SLOTS {
                    bail!("evict_slots contains out-of-range slot_id {slot_id}");
                }
                if seen[idx] {
                    bail!("evict_slots contains duplicate slot_id {slot_id}");
                }
                seen[idx] = true;
            }
        }
        let n_tiles = u32::try_from(self.tile_rects.len())
            .context(loc!(), "tile_rects length exceeds u32::MAX")?;
        let mut flags = 0u8;
        if self.incremental {
            flags |= HEADER_FLAG_INCREMENTAL;
        }
        if self.scroll_rect.is_some() {
            flags |= HEADER_FLAG_SCROLL_RECT;
        }
        if self.reset_tile_cache {
            flags |= HEADER_FLAG_RESET_TILE_CACHE;
        }
        if !self.evict_slots.is_empty() {
            flags |= HEADER_FLAG_EVICT_SLOTS;
        }
        let mut buf = [0u8; RAW_BUFFER_HEADER_LEN];
        buf[0..8].copy_from_slice(&self.frame_seq.to_be_bytes());
        buf[8..16].copy_from_slice(&self.client.0.to_be_bytes());
        buf[16..24].copy_from_slice(&self.surface.0.to_be_bytes());
        buf[24..28].copy_from_slice(&self.metadata.width.to_be_bytes());
        buf[28..32].copy_from_slice(&self.metadata.height.to_be_bytes());
        buf[32..36].copy_from_slice(&self.metadata.stride.to_be_bytes());
        buf[36] = match self.metadata.format {
            BufferFormat::Argb8888 => 0,
            BufferFormat::Xrgb8888 => 1,
        };
        buf[37] = flags;
        buf[38..42].copy_from_slice(&n_tiles.to_be_bytes());
        stream.write_all(&buf).location(loc!())?;
        if let Some(sr) = self.scroll_rect {
            sr.write_framed(stream).location(loc!())?;
        }
        if !self.evict_slots.is_empty() {
            (self.evict_slots.len() as u16)
                .framed_write(stream)
                .location(loc!())?;
            for &slot_id in &self.evict_slots {
                slot_id.framed_write(stream).location(loc!())?;
            }
        }
        for &rect in &self.tile_rects {
            rect.write_framed(stream).location(loc!())?;
        }
        Ok(())
    }

    fn framed_read<R: Read>(stream: &mut R) -> Result<Self> {
        let mut buf = [0u8; RAW_BUFFER_HEADER_LEN];
        stream.read_exact(&mut buf).location(loc!())?;
        let frame_seq = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let client = ClientId(u64::from_be_bytes(buf[8..16].try_into().unwrap()));
        let surface = WlSurfaceId(u64::from_be_bytes(buf[16..24].try_into().unwrap()));
        let width = i32::from_be_bytes(buf[24..28].try_into().unwrap());
        let height = i32::from_be_bytes(buf[28..32].try_into().unwrap());
        let stride = i32::from_be_bytes(buf[32..36].try_into().unwrap());
        let format = match buf[36] {
            0 => BufferFormat::Argb8888,
            1 => BufferFormat::Xrgb8888,
            tag => bail!("invalid BufferFormat tag: {tag}"),
        };
        let flags = buf[37];
        if (flags & !HEADER_FLAGS_VALID) != 0 {
            bail!("invalid RawBufferHeader flags: {flags}");
        }
        let incremental = (flags & HEADER_FLAG_INCREMENTAL) != 0;
        let has_scroll_rect = (flags & HEADER_FLAG_SCROLL_RECT) != 0;
        let reset_tile_cache = (flags & HEADER_FLAG_RESET_TILE_CACHE) != 0;
        let has_evict_slots = (flags & HEADER_FLAG_EVICT_SLOTS) != 0;
        if has_scroll_rect && !incremental {
            bail!("invalid RawBufferHeader flags: {flags} (ScrollRect requires incremental)");
        }
        if reset_tile_cache && has_evict_slots {
            bail!("invalid RawBufferHeader flags: {flags} (reset_tile_cache and evict_slots)");
        }
        let n_tiles = u32::from_be_bytes(buf[38..42].try_into().unwrap()) as usize;
        let metadata = BufferMetadata {
            width,
            height,
            stride,
            format,
        };
        metadata.validate().location(loc!())?;
        let width_u32 = width as u32;
        let height_u32 = height as u32;
        let max_tiles = (u64::from(width_u32) * u64::from(height_u32)).min(MAX_FRAME_TILES as u64);
        if n_tiles as u64 > max_tiles {
            bail!("tile count {n_tiles} exceeds maximum {max_tiles} for {width}x{height}");
        }
        let scroll_rect = if has_scroll_rect {
            let sr = ScrollRect::read_framed(stream).location(loc!())?;
            sr.validate_in_bounds(width_u32, height_u32)
                .location(loc!())?;
            Some(sr)
        } else {
            None
        };
        let evict_slots = if has_evict_slots {
            let n_evict = u16::framed_read(stream).location(loc!())? as usize;
            if n_evict == 0 || n_evict > MAX_TILE_CACHE_SLOTS {
                bail!("invalid evict_slots count {n_evict}: expected 1..={MAX_TILE_CACHE_SLOTS}");
            }
            let mut seen = [false; MAX_TILE_CACHE_SLOTS];
            let mut slots = Vec::with_capacity(n_evict);
            for _ in 0..n_evict {
                let slot_id = u16::framed_read(stream).location(loc!())?;
                let idx = slot_id as usize;
                if idx >= MAX_TILE_CACHE_SLOTS {
                    bail!("evict_slots contains out-of-range slot_id {slot_id}");
                }
                if seen[idx] {
                    bail!("evict_slots contains duplicate slot_id {slot_id}");
                }
                seen[idx] = true;
                slots.push(slot_id);
            }
            slots
        } else {
            Vec::new()
        };
        let mut tile_rects = Vec::with_capacity(n_tiles.min(1024));
        for _ in 0..n_tiles {
            let (rect, _) = TileRect::read_framed(stream).location(loc!())?;
            rect.validate_in_bounds(width_u32, height_u32)
                .location(loc!())?;
            tile_rects.push(rect);
        }
        Ok(Self {
            frame_seq,
            client,
            surface,
            metadata,
            incremental,
            scroll_rect,
            reset_tile_cache,
            evict_slots,
            tile_rects,
        })
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum EncodedBytes {
    Uncompressed(Vec<u8>),
    Zstd(Vec<u8>),
}

impl EncodedBytes {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Uncompressed(data) | Self::Zstd(data) => data.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn write_framed<W: Write>(
        &self,
        expected_uncompressed_len: usize,
        stream: &mut W,
    ) -> Result<()> {
        match self {
            Self::Uncompressed(data) => {
                if data.len() != expected_uncompressed_len {
                    bail!(
                        "uncompressed payload len {} != expected {expected_uncompressed_len}",
                        data.len()
                    );
                }
                0u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
            Self::Zstd(data) => {
                1u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
        }
        Ok(())
    }

    fn read_framed<R: Read>(expected_uncompressed_len: usize, stream: &mut R) -> Result<Self> {
        match u8::framed_read(stream).location(loc!())? {
            0 => {
                let len = u32::framed_read(stream).location(loc!())? as usize;
                if len != expected_uncompressed_len {
                    bail!("uncompressed payload len {len} != expected {expected_uncompressed_len}");
                }
                Ok(Self::Uncompressed(
                    read_exact_vec(stream, len).location(loc!())?,
                ))
            },
            1 => {
                let max_len = zstd_safe::compress_bound(expected_uncompressed_len);
                Ok(Self::Zstd(
                    read_bounded_vec(stream, max_len).location(loc!())?,
                ))
            },
            tag => bail!("invalid EncodedBytes tag: {tag}"),
        }
    }
}

impl fmt::Debug for EncodedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uncompressed(v) => write!(f, "Uncompressed([{}])", v.len()),
            Self::Zstd(v) => write!(f, "Zstd([{}])", v.len()),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum AlphaChannel {
    Omitted,
    Constant(u8),
    Uncompressed(Vec<u8>),
    Zstd(Vec<u8>),
}

impl AlphaChannel {
    fn payload_len(&self) -> usize {
        match self {
            Self::Omitted => 0,
            Self::Constant(_) => 1,
            Self::Uncompressed(data) | Self::Zstd(data) => data.len(),
        }
    }

    fn write_framed<W: Write>(&self, expected_pixels: usize, stream: &mut W) -> Result<()> {
        match self {
            Self::Omitted => bail!("attempted to write omitted AlphaChannel"),
            Self::Constant(a) => {
                0u8.framed_write(stream).location(loc!())?;
                a.framed_write(stream).location(loc!())?;
            },
            Self::Uncompressed(data) => {
                if data.len() != expected_pixels {
                    bail!(
                        "uncompressed alpha len {} != expected {expected_pixels}",
                        data.len()
                    );
                }
                1u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
            Self::Zstd(data) => {
                2u8.framed_write(stream).location(loc!())?;
                data.framed_write(stream).location(loc!())?;
            },
        }
        Ok(())
    }

    fn read_framed<R: Read>(expected_pixels: usize, stream: &mut R) -> Result<Self> {
        match u8::framed_read(stream).location(loc!())? {
            0 => Ok(Self::Constant(u8::framed_read(stream).location(loc!())?)),
            1 => {
                let len = u32::framed_read(stream).location(loc!())? as usize;
                if len != expected_pixels {
                    bail!("uncompressed alpha len {len} != expected {expected_pixels}");
                }
                Ok(Self::Uncompressed(
                    read_exact_vec(stream, len).location(loc!())?,
                ))
            },
            2 => {
                let max_len = zstd_safe::compress_bound(expected_pixels);
                Ok(Self::Zstd(
                    read_bounded_vec(stream, max_len).location(loc!())?,
                ))
            },
            tag => bail!("invalid AlphaChannel tag: {tag}"),
        }
    }
}

impl fmt::Debug for AlphaChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Omitted => write!(f, "Omitted"),
            Self::Constant(a) => write!(f, "Constant({a})"),
            Self::Uncompressed(v) => write!(f, "Uncompressed([{}])", v.len()),
            Self::Zstd(v) => write!(f, "Zstd([{}])", v.len()),
        }
    }
}

/// Upper bound on valid JPEG payload size for a tile with `pixels` pixels,
/// enforced before allocating on wire reads (`<= 458,752 B` for a `256x256`
/// tile, exceeding `tj3JPEGBufSize` even for Q100 4:4:4 random noise).
#[inline]
pub(crate) fn max_jpeg_bytes(pixels: usize) -> usize {
    pixels * 6 + 65_536
}

/// Returns `true` iff `rect` exceeds `SUBTILE_W x SUBTILE_H` and may be
/// subdivided into `2..=4` sub-tiles.
#[inline]
pub(crate) fn is_splittable_rect(rect: TileRect) -> bool {
    rect.w > SUBTILE_W as u32 || rect.h > SUBTILE_H as u32
}

/// Validates that `sub_rect` is non-empty, strictly within `parent`, and
/// disjoint from every rect in `prior`.
fn validate_sub_rect_step(
    parent: TileRect,
    prior: &[SubTile],
    sub_rect: TileRect,
) -> Result<usize> {
    let sub_px = sub_rect.checked_pixels().location(loc!())?;
    let parent_x1 = parent
        .x
        .checked_add(parent.w)
        .context(loc!(), "parent TileRect x + w overflow")?;
    let parent_y1 = parent
        .y
        .checked_add(parent.h)
        .context(loc!(), "parent TileRect y + h overflow")?;
    let sx1 = sub_rect
        .x
        .checked_add(sub_rect.w)
        .context(loc!(), "SubTile x + w overflow")?;
    let sy1 = sub_rect
        .y
        .checked_add(sub_rect.h)
        .context(loc!(), "SubTile y + h overflow")?;
    if sub_rect.x < parent.x || sub_rect.y < parent.y || sx1 > parent_x1 || sy1 > parent_y1 {
        bail!("SubTile {sub_rect:?} lies outside parent TileRect {parent:?}");
    }
    if prior.iter().any(|prev| prev.rect.overlaps(sub_rect)) {
        bail!("SubTile {sub_rect:?} overlaps an earlier sub-tile in parent {parent:?}");
    }
    Ok(sub_px)
}

/// Validates that `sub_tiles` forms a strict non-overlapping partition of
/// splittable `parent` (`2..=4` sub-tiles covering exact area `parent.w * parent.h`).
pub(crate) fn validate_split_rects(parent: TileRect, sub_tiles: &[SubTile]) -> Result<()> {
    if !is_splittable_rect(parent) {
        bail!("Split payload on non-splittable tile {parent:?}");
    }
    if !(2..=4).contains(&sub_tiles.len()) {
        bail!(
            "invalid Split sub-tile count {}: expected 2..=4",
            sub_tiles.len()
        );
    }
    let parent_area = parent.checked_pixels().location(loc!())? as u64;
    let mut covered_area = 0u64;
    for (i, sub) in sub_tiles.iter().enumerate() {
        let sub_px = validate_sub_rect_step(parent, &sub_tiles[..i], sub.rect).location(loc!())?;
        covered_area += sub_px as u64;
    }
    if covered_area != parent_area {
        bail!("Split sub-tiles cover {covered_area} pixels, expected {parent_area} for {parent:?}");
    }
    Ok(())
}

#[derive(Clone, Eq, PartialEq)]
pub enum LeafPayload {
    Solid([u8; 4]),
    Palette {
        has_alpha: bool,
        palette: Vec<u8>,
        data: EncodedBytes,
    },
    AosZstd {
        predictors: Vec<u8>,
        color: EncodedBytes,
        alpha: AlphaChannel,
    },
    Jpeg {
        data: Vec<u8>,
        alpha: AlphaChannel,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SubTile {
    pub rect: TileRect,
    pub payload: LeafPayload,
}

#[derive(Clone, Eq, PartialEq)]
pub enum TilePayload {
    Leaf(LeafPayload),
    Split(Vec<SubTile>),
    CacheHit(CacheHitRef),
}

const PAYLOAD_TAG_SOLID_OPAQUE: u8 = 0;
const PAYLOAD_TAG_SOLID_ALPHA: u8 = 1;
const PAYLOAD_TAG_PALETTE_OPAQUE: u8 = 2;
const PAYLOAD_TAG_PALETTE_ALPHA: u8 = 3;
const PAYLOAD_TAG_AOS_OPAQUE: u8 = 4;
const PAYLOAD_TAG_AOS_ALPHA: u8 = 5;
const PAYLOAD_TAG_JPEG_OPAQUE: u8 = 6;
const PAYLOAD_TAG_JPEG_ALPHA: u8 = 7;
const PAYLOAD_TAG_SPLIT: u8 = 8;
const PAYLOAD_TAG_CACHE_HIT: u8 = 9;
const PAYLOAD_FLAG_CACHE_PUT: u8 = 0x80;

impl LeafPayload {
    fn payload_len(&self) -> usize {
        match self {
            Self::Solid(px) => {
                if px[3] == 0xFF {
                    3
                } else {
                    4
                }
            },
            Self::Palette { palette, data, .. } => 1 + palette.len() + data.len(),
            Self::AosZstd {
                predictors,
                color,
                alpha,
            } => predictors.len() + color.len() + alpha.payload_len(),
            Self::Jpeg { data, alpha } => data.len() + alpha.payload_len(),
        }
    }

    fn write_framed_with_put<W: Write>(
        &self,
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        cache_put: Option<u16>,
        stream: &mut W,
    ) -> Result<()> {
        let put_flag = if cache_put.is_some() {
            PAYLOAD_FLAG_CACHE_PUT
        } else {
            0
        };
        let write_tag = |tag: u8, stream: &mut W| -> Result<()> {
            (tag | put_flag).framed_write(stream).location(loc!())?;
            if let Some(slot_id) = cache_put {
                slot_id.framed_write(stream).location(loc!())?;
            }
            Ok(())
        };
        match self {
            Self::Solid(px) => {
                if px[3] == 0xFF {
                    write_tag(PAYLOAD_TAG_SOLID_OPAQUE, stream)?;
                    stream.write_all(&px[0..3]).location(loc!())?;
                } else {
                    if !has_alpha {
                        bail!("translucent Solid tile on Xrgb8888 buffer");
                    }
                    write_tag(PAYLOAD_TAG_SOLID_ALPHA, stream)?;
                    stream.write_all(px).location(loc!())?;
                }
            },
            Self::Palette {
                has_alpha: pal_alpha,
                palette,
                data,
            } => {
                if *pal_alpha && !has_alpha {
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
                let tag = if *pal_alpha {
                    PAYLOAD_TAG_PALETTE_ALPHA
                } else {
                    PAYLOAD_TAG_PALETTE_OPAQUE
                };
                write_tag(tag, stream)?;
                ((n_colors - 2) as u8)
                    .framed_write(stream)
                    .location(loc!())?;
                stream.write_all(palette).location(loc!())?;
                let expected_len = packed_palette_indices_len(pixels, n_colors);
                data.write_framed(expected_len, stream).location(loc!())?;
            },
            Self::AosZstd {
                predictors,
                color,
                alpha,
            } => {
                let expected_pred_len = (rect.h as usize).div_ceil(4);
                if predictors.len() != expected_pred_len {
                    bail!(
                        "predictors len {} != expected {expected_pred_len}",
                        predictors.len()
                    );
                }
                let color_bytes = pixels * 3;
                if matches!(alpha, AlphaChannel::Omitted) {
                    write_tag(PAYLOAD_TAG_AOS_OPAQUE, stream)?;
                    stream.write_all(predictors).location(loc!())?;
                    color.write_framed(color_bytes, stream).location(loc!())?;
                } else {
                    if !has_alpha {
                        bail!("non-omitted AlphaChannel on Xrgb8888 buffer");
                    }
                    write_tag(PAYLOAD_TAG_AOS_ALPHA, stream)?;
                    stream.write_all(predictors).location(loc!())?;
                    color.write_framed(color_bytes, stream).location(loc!())?;
                    alpha.write_framed(pixels, stream).location(loc!())?;
                }
            },
            Self::Jpeg { data, alpha } => {
                let max_bytes = max_jpeg_bytes(pixels);
                if data.is_empty() || data.len() > max_bytes {
                    bail!(
                        "invalid JPEG payload len {} (expected 1..={max_bytes})",
                        data.len()
                    );
                }
                if matches!(alpha, AlphaChannel::Omitted) {
                    write_tag(PAYLOAD_TAG_JPEG_OPAQUE, stream)?;
                    data.framed_write(stream).location(loc!())?;
                } else {
                    if !has_alpha {
                        bail!("non-omitted AlphaChannel on Xrgb8888 buffer");
                    }
                    write_tag(PAYLOAD_TAG_JPEG_ALPHA, stream)?;
                    data.framed_write(stream).location(loc!())?;
                    alpha.write_framed(pixels, stream).location(loc!())?;
                }
            },
        }
        Ok(())
    }

    fn write_framed<W: Write>(
        &self,
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        stream: &mut W,
    ) -> Result<()> {
        self.write_framed_with_put(rect, pixels, has_alpha, None, stream)
    }

    fn read_framed_with_tag<R: Read>(
        tag: u8,
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        stream: &mut R,
    ) -> Result<Self> {
        match tag {
            PAYLOAD_TAG_SOLID_OPAQUE => {
                let mut bgr = [0u8; 3];
                stream.read_exact(&mut bgr).location(loc!())?;
                Ok(Self::Solid([bgr[0], bgr[1], bgr[2], 0xFF]))
            },
            PAYLOAD_TAG_SOLID_ALPHA => {
                if !has_alpha {
                    bail!("unexpected translucent Solid tile on Xrgb8888 buffer");
                }
                let mut bgra = [0u8; 4];
                stream.read_exact(&mut bgra).location(loc!())?;
                Ok(Self::Solid(bgra))
            },
            PAYLOAD_TAG_PALETTE_OPAQUE | PAYLOAD_TAG_PALETTE_ALPHA => {
                let pal_alpha = tag == PAYLOAD_TAG_PALETTE_ALPHA;
                if pal_alpha && !has_alpha {
                    bail!("unexpected translucent Palette tile on Xrgb8888 buffer");
                }
                let n_colors_minus_2 = u8::framed_read(stream).location(loc!())?;
                if n_colors_minus_2 > 254 {
                    bail!("invalid palette count tag: {n_colors_minus_2}");
                }
                let n_colors = usize::from(n_colors_minus_2) + 2;
                let entry_bytes = if pal_alpha { 4 } else { 3 };
                let palette = read_exact_vec(stream, n_colors * entry_bytes).location(loc!())?;
                let expected_len = packed_palette_indices_len(pixels, n_colors);
                let data = EncodedBytes::read_framed(expected_len, stream).location(loc!())?;
                Ok(Self::Palette {
                    has_alpha: pal_alpha,
                    palette,
                    data,
                })
            },
            PAYLOAD_TAG_AOS_OPAQUE | PAYLOAD_TAG_AOS_ALPHA => {
                let with_alpha = tag == PAYLOAD_TAG_AOS_ALPHA;
                if with_alpha && !has_alpha {
                    bail!("unexpected alpha payload on Xrgb8888 buffer");
                }
                let pred_len = (rect.h as usize).div_ceil(4);
                let predictors = read_exact_vec(stream, pred_len).location(loc!())?;
                let color = EncodedBytes::read_framed(pixels * 3, stream).location(loc!())?;
                let alpha = if with_alpha {
                    AlphaChannel::read_framed(pixels, stream).location(loc!())?
                } else {
                    AlphaChannel::Omitted
                };
                Ok(Self::AosZstd {
                    predictors,
                    color,
                    alpha,
                })
            },
            PAYLOAD_TAG_JPEG_OPAQUE | PAYLOAD_TAG_JPEG_ALPHA => {
                let with_alpha = tag == PAYLOAD_TAG_JPEG_ALPHA;
                if with_alpha && !has_alpha {
                    bail!("unexpected alpha payload on Xrgb8888 buffer");
                }
                let data = read_bounded_vec(stream, max_jpeg_bytes(pixels)).location(loc!())?;
                if data.is_empty() {
                    bail!("empty JPEG payload");
                }
                let alpha = if with_alpha {
                    AlphaChannel::read_framed(pixels, stream).location(loc!())?
                } else {
                    AlphaChannel::Omitted
                };
                Ok(Self::Jpeg { data, alpha })
            },
            _ => bail!("invalid LeafPayload tag: {tag}"),
        }
    }

    fn read_framed<R: Read>(
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        stream: &mut R,
    ) -> Result<Self> {
        let tag = u8::framed_read(stream).location(loc!())?;
        Self::read_framed_with_tag(tag, rect, pixels, has_alpha, stream)
    }
}

impl fmt::Debug for LeafPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solid(px) => write!(f, "Solid({px:?})"),
            Self::Palette {
                has_alpha,
                palette,
                data,
            } => f
                .debug_struct("Palette")
                .field("has_alpha", has_alpha)
                .field("palette_bytes", &palette.len())
                .field("data", data)
                .finish(),
            Self::AosZstd {
                predictors,
                color,
                alpha,
            } => f
                .debug_struct("AosZstd")
                .field("predictors_bytes", &predictors.len())
                .field("color", color)
                .field("alpha", alpha)
                .finish(),
            Self::Jpeg { data, alpha } => f
                .debug_struct("Jpeg")
                .field("bytes", &data.len())
                .field("alpha", alpha)
                .finish(),
        }
    }
}

impl TilePayload {
    fn payload_len(&self) -> usize {
        match self {
            Self::Leaf(leaf) => leaf.payload_len(),
            Self::Split(subs) => {
                1 + subs
                    .iter()
                    .map(|s| TILE_RECT_LEN + 1 + s.payload.payload_len())
                    .sum::<usize>()
            },
            Self::CacheHit(_) => 2,
        }
    }

    fn write_framed<W: Write>(
        &self,
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        cache_put: Option<u16>,
        stream: &mut W,
    ) -> Result<()> {
        if let Some(slot_id) = cache_put
            && slot_id as usize >= MAX_TILE_CACHE_SLOTS
        {
            bail!("tile cache_put slot_id {slot_id} out of range (max {MAX_TILE_CACHE_SLOTS})");
        }
        match self {
            Self::Leaf(leaf) => {
                leaf.write_framed_with_put(rect, pixels, has_alpha, cache_put, stream)
            },
            Self::Split(sub_tiles) => {
                validate_split_rects(rect, sub_tiles).location(loc!())?;
                let tag = PAYLOAD_TAG_SPLIT
                    | if cache_put.is_some() {
                        PAYLOAD_FLAG_CACHE_PUT
                    } else {
                        0
                    };
                tag.framed_write(stream).location(loc!())?;
                if let Some(slot_id) = cache_put {
                    slot_id.framed_write(stream).location(loc!())?;
                }
                (sub_tiles.len() as u8)
                    .framed_write(stream)
                    .location(loc!())?;
                for sub in sub_tiles {
                    let sub_px = sub.rect.checked_pixels().location(loc!())?;
                    sub.rect.write_framed(stream).location(loc!())?;
                    sub.payload
                        .write_framed(sub.rect, sub_px, has_alpha, stream)
                        .location(loc!())?;
                }
                Ok(())
            },
            Self::CacheHit(hit_ref) => {
                if cache_put.is_some() {
                    bail!("CacheHit payload cannot also carry cache_put");
                }
                hit_ref.validate_for_rect(rect).location(loc!())?;
                let packed = hit_ref.to_u16().location(loc!())?;
                PAYLOAD_TAG_CACHE_HIT
                    .framed_write(stream)
                    .location(loc!())?;
                packed.framed_write(stream).location(loc!())?;
                Ok(())
            },
        }
    }

    fn read_framed<R: Read>(
        rect: TileRect,
        pixels: usize,
        has_alpha: bool,
        stream: &mut R,
    ) -> Result<(Option<u16>, Self)> {
        let raw_tag = u8::framed_read(stream).location(loc!())?;
        let is_put = (raw_tag & PAYLOAD_FLAG_CACHE_PUT) != 0;
        let tag = raw_tag & !PAYLOAD_FLAG_CACHE_PUT;
        if tag == PAYLOAD_TAG_CACHE_HIT {
            if is_put {
                bail!("CacheHit payload cannot also carry CachePut flag");
            }
            let packed = u16::framed_read(stream).location(loc!())?;
            let hit_ref = CacheHitRef::from_u16(packed).location(loc!())?;
            hit_ref.validate_for_rect(rect).location(loc!())?;
            return Ok((None, Self::CacheHit(hit_ref)));
        }
        if tag > PAYLOAD_TAG_SPLIT {
            bail!("invalid TilePayload tag: {raw_tag}");
        }
        let cache_put = if is_put {
            let slot_id = u16::framed_read(stream).location(loc!())?;
            if slot_id as usize >= MAX_TILE_CACHE_SLOTS {
                bail!("tile cache_put slot_id {slot_id} out of range (max {MAX_TILE_CACHE_SLOTS})");
            }
            Some(slot_id)
        } else {
            None
        };
        if tag == PAYLOAD_TAG_SPLIT {
            if !is_splittable_rect(rect) {
                bail!("unexpected Split payload on non-splittable tile {rect:?}");
            }
            let n_sub = usize::from(u8::framed_read(stream).location(loc!())?);
            if !(2..=4).contains(&n_sub) {
                bail!("invalid Split sub-tile count {n_sub}: expected 2..=4");
            }
            let mut sub_tiles = Vec::with_capacity(n_sub);
            let mut covered_area = 0u64;
            for _ in 0..n_sub {
                let (sub_rect, sub_px) = TileRect::read_framed(stream).location(loc!())?;
                validate_sub_rect_step(rect, &sub_tiles, sub_rect).location(loc!())?;
                let payload = LeafPayload::read_framed(sub_rect, sub_px, has_alpha, stream)
                    .location(loc!())?;
                covered_area += sub_px as u64;
                sub_tiles.push(SubTile {
                    rect: sub_rect,
                    payload,
                });
            }
            if covered_area != pixels as u64 {
                bail!(
                    "Split sub-tiles cover {covered_area} pixels, expected {pixels} for {rect:?}"
                );
            }
            Ok((cache_put, Self::Split(sub_tiles)))
        } else {
            Ok((
                cache_put,
                Self::Leaf(
                    LeafPayload::read_framed_with_tag(tag, rect, pixels, has_alpha, stream)
                        .location(loc!())?,
                ),
            ))
        }
    }
}

impl fmt::Debug for TilePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(leaf) => fmt::Debug::fmt(leaf, f),
            Self::Split(subs) => f.debug_tuple("Split").field(subs).finish(),
            Self::CacheHit(hit_ref) => f.debug_tuple("CacheHit").field(hit_ref).finish(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompressedTile {
    pub tile_idx: u32,
    pub cache_put: Option<u16>,
    pub payload: TilePayload,
}

impl CompressedTile {
    pub fn compressed_len(&self) -> usize {
        usize::from(self.cache_put.is_some()) * 2 + self.payload.payload_len()
    }

    fn write_framed<W: Write>(
        &self,
        rect: TileRect,
        has_alpha: bool,
        stream: &mut W,
    ) -> Result<()> {
        let pixels = rect.checked_pixels().location(loc!())?;
        self.tile_idx.framed_write(stream).location(loc!())?;
        self.payload
            .write_framed(rect, pixels, has_alpha, self.cache_put, stream)
            .location(loc!())
    }

    fn read_framed<R: Read>(
        tile_rects: &[TileRect],
        has_alpha: bool,
        stream: &mut R,
    ) -> Result<Self> {
        let tile_idx = u32::framed_read(stream).location(loc!())?;
        let rect = *tile_rects.get(tile_idx as usize).with_context(loc!(), || {
            format!(
                "tile_idx {tile_idx} out of bounds for {} tiles",
                tile_rects.len()
            )
        })?;
        let pixels = rect.checked_pixels().location(loc!())?;
        let (cache_put, payload) =
            TilePayload::read_framed(rect, pixels, has_alpha, stream).location(loc!())?;
        Ok(Self {
            tile_idx,
            cache_put,
            payload,
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompressedTiles {
    pub metadata: BufferMetadata,
    pub tile_rects: Vec<TileRect>,
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
        let total_pixels: usize = self
            .header
            .tile_rects
            .iter()
            .map(|r| r.w as usize * r.h as usize)
            .sum();
        total_pixels * n_channels
    }

    pub fn framed_write<W: Write>(self, stream: &mut W) -> Result<usize> {
        let has_alpha = self.header.metadata.format == BufferFormat::Argb8888;
        self.header.framed_write(stream).location(loc!())?;
        // Flush the header immediately so `wprsc` can prepare the destination
        // `wl_shm` canvas (including any 2-slot swapchain rect copies) while
        // the first tile is still compressing.
        stream.flush().location(loc!())?;

        let tile_rects = &self.header.tile_rects;
        let mut remaining = tile_rects.len();
        let mut total_compressed_size = 0;

        while remaining > 0 {
            let tile = self
                .tiles
                .recv()
                .context(loc!(), "tile sender dropped before sending all tiles")?;
            let rect = *tile_rects
                .get(tile.tile_idx as usize)
                .with_context(loc!(), || {
                    format!(
                        "tile_idx {} out of bounds for {} tiles",
                        tile.tile_idx,
                        tile_rects.len()
                    )
                })?;
            let mut batch_bytes = tile.compressed_len();
            total_compressed_size += batch_bytes;
            tile.write_framed(rect, has_alpha, stream)
                .location(loc!())?;
            remaining -= 1;

            while remaining > 0
                && batch_bytes < MAX_COALESCED_TILE_BYTES
                && let Ok(tile) = self.tiles.try_recv()
            {
                let rect = *tile_rects
                    .get(tile.tile_idx as usize)
                    .with_context(loc!(), || {
                        format!(
                            "tile_idx {} out of bounds for {} tiles",
                            tile.tile_idx,
                            tile_rects.len()
                        )
                    })?;
                let tile_len = tile.compressed_len();
                batch_bytes += tile_len;
                total_compressed_size += tile_len;
                tile.write_framed(rect, has_alpha, stream)
                    .location(loc!())?;
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
        let tile_rects = header.tile_rects.clone();
        let n_tiles = tile_rects.len();
        let (tiles_tx, tiles_rx) = crossbeam_channel::unbounded();
        on_stream(Self {
            header,
            tiles: tiles_rx,
        })
        .location(loc!())?;

        for _ in 0..n_tiles {
            match CompressedTile::read_framed(&tile_rects, has_alpha, stream) {
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
    fn test_tile_rect_bounds_and_overlap() {
        let r = TileRect {
            x: 10,
            y: 20,
            w: 256,
            h: 256,
        };
        assert_eq!(r.checked_pixels().unwrap(), TILE_PIXELS);
        assert_eq!(r.validate_in_bounds(266, 276).unwrap(), TILE_PIXELS);
        assert!(r.validate_in_bounds(265, 276).is_err());
        assert!(r.validate_in_bounds(266, 275).is_err());

        // Empty, oversized (> TILE_PIXELS), and u32::MAX overflow rects.
        for bad in [
            TileRect {
                x: 0,
                y: 0,
                w: 0,
                h: 10,
            },
            TileRect {
                x: 0,
                y: 0,
                w: 10,
                h: 0,
            },
            TileRect {
                x: 0,
                y: 0,
                w: 257,
                h: 256,
            },
            TileRect {
                x: u32::MAX,
                y: 0,
                w: 1,
                h: 1,
            },
            TileRect {
                x: 0,
                y: u32::MAX,
                w: 1,
                h: 1,
            },
        ] {
            assert!(bad.validate_in_bounds(1000, 1000).is_err());
        }

        // Adjacent tiles do not overlap; overlapping tiles do.
        let a = TileRect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        assert!(!a.overlaps(TileRect {
            x: 10,
            y: 0,
            w: 10,
            h: 10,
        }));
        assert!(!a.overlaps(TileRect {
            x: 0,
            y: 10,
            w: 10,
            h: 10,
        }));
        assert!(a.overlaps(TileRect {
            x: 9,
            y: 9,
            w: 10,
            h: 10,
        }));
        assert!(a.overlaps(a));
    }

    #[test]
    fn test_raw_buffer_header_and_tile_payload_framing() {
        for format in [BufferFormat::Argb8888, BufferFormat::Xrgb8888] {
            for (incremental, tile_rects) in [
                (
                    false,
                    vec![
                        TileRect {
                            x: 0,
                            y: 0,
                            w: 256,
                            h: 256,
                        },
                        TileRect {
                            x: 256,
                            y: 0,
                            w: 256,
                            h: 256,
                        },
                    ],
                ),
                (true, vec![]),
                (
                    true,
                    vec![
                        TileRect {
                            x: 128,
                            y: 0,
                            w: 128,
                            h: 128,
                        },
                        TileRect {
                            x: 512,
                            y: 256,
                            w: 128,
                            h: 224,
                        },
                    ],
                ),
            ] {
                let hdr = RawBufferHeader {
                    frame_seq: 123_456_789,
                    client: ClientId(42),
                    surface: WlSurfaceId(99),
                    metadata: BufferMetadata {
                        width: 640,
                        height: 480,
                        stride: 640 * 4 + 16,
                        format,
                    },
                    incremental,
                    scroll_rect: None,
                    reset_tile_cache: false,
                    evict_slots: Vec::new(),
                    tile_rects,
                };
                let mut wire = Vec::new();
                hdr.framed_write(&mut wire).unwrap();
                let decoded = RawBufferHeader::framed_read(&mut wire.as_slice()).unwrap();
                assert_eq!(decoded, hdr);
                assert_eq!(decoded.metadata.validate().unwrap(), 480 * (640 + 4));
            }
        }

        // Valid fixed prefix helper for negative header tests (`640x480` Xrgb).
        let make_header_prefix = |w: i32, h: i32, stride: i32, fmt: u8, inc: u8, n_tiles: u32| {
            let mut buf = [0u8; RAW_BUFFER_HEADER_LEN];
            buf[24..28].copy_from_slice(&w.to_be_bytes());
            buf[28..32].copy_from_slice(&h.to_be_bytes());
            buf[32..36].copy_from_slice(&stride.to_be_bytes());
            buf[36] = fmt;
            buf[37] = inc;
            buf[38..42].copy_from_slice(&n_tiles.to_be_bytes());
            buf
        };

        // Invalid BufferFormat, incremental tag, invalid metadata, and excessive
        // tile count (`> width * height` or `> MAX_FRAME_TILES`) must fail
        // before reading/allocating tile rects.
        assert!(
            RawBufferHeader::framed_read(
                &mut make_header_prefix(640, 480, 640 * 4, 9, 0, 0).as_slice()
            )
            .is_err()
        );
        assert!(
            RawBufferHeader::framed_read(
                &mut make_header_prefix(640, 480, 640 * 4, 1, 2, 0).as_slice()
            )
            .is_err()
        );
        assert!(
            RawBufferHeader::framed_read(
                &mut make_header_prefix(0, 480, 640 * 4, 1, 0, 0).as_slice()
            )
            .is_err()
        );
        // Excessive tile count (`> width * height` or `> MAX_FRAME_TILES`) must
        // fail BEFORE reading or allocating any `TileRect`s.
        let unit_rect = TileRect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        };
        let mut too_many_for_2x2 = make_header_prefix(2, 2, 8, 1, 1, 5).to_vec();
        for _ in 0..5 {
            unit_rect.write_framed(&mut too_many_for_2x2).unwrap();
        }
        let mut rdr = too_many_for_2x2.as_slice();
        assert!(RawBufferHeader::framed_read(&mut rdr).is_err());
        assert_eq!(rdr.len(), 5 * TILE_RECT_LEN);

        let mut too_many_global =
            make_header_prefix(3840, 2160, 3840 * 4, 1, 1, MAX_FRAME_TILES as u32 + 1).to_vec();
        unit_rect.write_framed(&mut too_many_global).unwrap();
        let mut rdr = too_many_global.as_slice();
        assert!(RawBufferHeader::framed_read(&mut rdr).is_err());
        assert_eq!(rdr.len(), TILE_RECT_LEN);

        assert!(
            RawBufferHeader::framed_read(
                &mut make_header_prefix(3840, 2160, 3840 * 4, 1, 1, u32::MAX).as_slice()
            )
            .is_err()
        );

        // Out-of-bounds TileRect in header must fail immediately on read.
        let mut oob_hdr = make_header_prefix(64, 64, 64 * 4, 1, 1, 1).to_vec();
        TileRect {
            x: 32,
            y: 0,
            w: 33,
            h: 64,
        }
        .write_framed(&mut oob_hdr)
        .unwrap();
        assert!(RawBufferHeader::framed_read(&mut oob_hdr.as_slice()).is_err());

        // Roundtrip all TilePayload variants across Xrgb8888 and Argb8888.
        let rect = TileRect {
            x: 16,
            y: 32,
            w: 4,
            h: 3,
        };
        let tile_rects = [rect];
        let opaque_tiles = [
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Solid([10, 20, 30, 0xFF])),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1, 2, 3, 4, 5, 6],
                    data: EncodedBytes::Uncompressed(vec![0x5A, 0x05]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Palette {
                    has_alpha: false,
                    palette: vec![1; 17 * 3],
                    data: EncodedBytes::Zstd(vec![9, 8, 7]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::AosZstd {
                    predictors: vec![0b00_10_01_00],
                    color: EncodedBytes::Uncompressed(vec![7u8; 4 * 3 * 3]),
                    alpha: AlphaChannel::Omitted,
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Jpeg {
                    data: vec![0xFF, 0xD8, 0xFF, 0xD9],
                    alpha: AlphaChannel::Omitted,
                }),
            },
        ];
        for tile in &opaque_tiles {
            for has_alpha in [false, true] {
                let mut wire = Vec::new();
                tile.write_framed(rect, has_alpha, &mut wire).unwrap();
                let decoded =
                    CompressedTile::read_framed(&tile_rects, has_alpha, &mut wire.as_slice())
                        .unwrap();
                assert_eq!(&decoded, tile);
            }
        }

        let translucent_tiles = [
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Solid([10, 20, 30, 0x80])),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Palette {
                    has_alpha: true,
                    palette: vec![1, 2, 3, 0x40, 4, 5, 6, 0xFF],
                    data: EncodedBytes::Uncompressed(vec![0x5A, 0x05]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Zstd(vec![1, 2, 3, 4]),
                    alpha: AlphaChannel::Constant(0x80),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Zstd(vec![1, 2, 3, 4]),
                    alpha: AlphaChannel::Uncompressed(vec![0x55; 12]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::AosZstd {
                    predictors: vec![0],
                    color: EncodedBytes::Zstd(vec![1, 2, 3, 4]),
                    alpha: AlphaChannel::Zstd(vec![5, 6]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Jpeg {
                    data: vec![0xFF, 0xD8, 0xAA, 0xFF, 0xD9],
                    alpha: AlphaChannel::Constant(0x80),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Jpeg {
                    data: vec![0xFF, 0xD8, 0xBB, 0xFF, 0xD9],
                    alpha: AlphaChannel::Uncompressed(vec![0x42; 12]),
                }),
            },
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::Leaf(LeafPayload::Jpeg {
                    data: vec![0xFF, 0xD8, 0xCC, 0xFF, 0xD9],
                    alpha: AlphaChannel::Zstd(vec![7, 8, 9]),
                }),
            },
        ];
        for tile in &translucent_tiles {
            let mut wire = Vec::new();
            tile.write_framed(rect, true, &mut wire).unwrap();
            let decoded =
                CompressedTile::read_framed(&tile_rects, true, &mut wire.as_slice()).unwrap();
            assert_eq!(&decoded, tile);
            // Translucent payloads must fail on Xrgb8888 (both write and read).
            assert!(tile.write_framed(rect, false, &mut Vec::new()).is_err());
            assert!(CompressedTile::read_framed(&tile_rects, false, &mut wire.as_slice()).is_err());
        }

        // Out-of-range `tile_idx` must fail before reading payload.
        let mut oob_idx_wire = Vec::new();
        1u32.framed_write(&mut oob_idx_wire).unwrap();
        assert!(
            CompressedTile::read_framed(&tile_rects, true, &mut oob_idx_wire.as_slice()).is_err()
        );

        // Oversized TileRect (`> TILE_PIXELS`) must fail on `TileRect::read_framed`.
        let mut oversized_rect_wire = Vec::new();
        TileRect {
            x: 0,
            y: 0,
            w: 512,
            h: 512,
        }
        .write_framed(&mut oversized_rect_wire)
        .unwrap();
        assert!(TileRect::read_framed(&mut oversized_rect_wire.as_slice()).is_err());

        // Oversized Zstd length (`compress_bound(36) + 1`) with full payload bytes
        // present must fail BEFORE reading or allocating the payload.
        let max_zstd = zstd_safe::compress_bound(4 * 3 * 3);
        let mut oversized_zstd = Vec::new();
        0u32.framed_write(&mut oversized_zstd).unwrap();
        PAYLOAD_TAG_AOS_OPAQUE
            .framed_write(&mut oversized_zstd)
            .unwrap();
        oversized_zstd.push(0); // predictors (1 byte)
        1u8.framed_write(&mut oversized_zstd).unwrap(); // EncodedBytes::Zstd
        (max_zstd + 1).framed_write(&mut oversized_zstd).unwrap();
        let header_end = oversized_zstd.len();
        oversized_zstd.resize(header_end + max_zstd + 1, 0xAB);
        let mut rdr = oversized_zstd.as_slice();
        assert!(CompressedTile::read_framed(&tile_rects, false, &mut rdr).is_err());
        assert_eq!(rdr.len(), max_zstd + 1);

        // Wrong uncompressed length on AosZstd color must fail before allocating.
        let mut wrong_raw_len = Vec::new();
        0u32.framed_write(&mut wrong_raw_len).unwrap();
        PAYLOAD_TAG_AOS_OPAQUE
            .framed_write(&mut wrong_raw_len)
            .unwrap();
        wrong_raw_len.push(0);
        0u8.framed_write(&mut wrong_raw_len).unwrap();
        u32::MAX.framed_write(&mut wrong_raw_len).unwrap();
        assert!(
            CompressedTile::read_framed(&tile_rects, false, &mut wrong_raw_len.as_slice()).is_err()
        );

        // Invalid palette count tag `255` (`257` colors) must fail.
        let mut bad_pal_count = Vec::new();
        0u32.framed_write(&mut bad_pal_count).unwrap();
        PAYLOAD_TAG_PALETTE_OPAQUE
            .framed_write(&mut bad_pal_count)
            .unwrap();
        bad_pal_count.push(255);
        assert!(
            CompressedTile::read_framed(&tile_rects, false, &mut bad_pal_count.as_slice()).is_err()
        );

        // Invalid payload tag, EncodedBytes tag, and AlphaChannel tag.
        let mut bad_tag = Vec::new();
        0u32.framed_write(&mut bad_tag).unwrap();
        bad_tag.push(99);
        assert!(CompressedTile::read_framed(&tile_rects, true, &mut bad_tag.as_slice()).is_err());
        assert!(EncodedBytes::read_framed(16, &mut [9u8].as_slice()).is_err());
        assert!(AlphaChannel::read_framed(16, &mut [9u8].as_slice()).is_err());
        assert!(
            AlphaChannel::Omitted
                .write_framed(16, &mut Vec::new())
                .is_err()
        );
    }

    #[test]
    fn test_streaming_compressed_tiles_and_errors() {
        let rect0 = TileRect {
            x: 0,
            y: 0,
            w: 64,
            h: 10,
        };
        let rect1 = TileRect {
            x: 0,
            y: 20,
            w: 64,
            h: 6,
        };
        let header = RawBufferHeader {
            frame_seq: 1,
            client: ClientId(7),
            surface: WlSurfaceId(11),
            metadata: BufferMetadata {
                width: 64,
                height: 64,
                stride: 64 * 4 + 16,
                format: BufferFormat::Xrgb8888,
            },
            incremental: true,
            scroll_rect: None,
            reset_tile_cache: false,
            evict_slots: Vec::new(),
            tile_rects: vec![rect0, rect1],
        };
        let tile0 = CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Solid([1, 2, 3, 0xFF])),
        };
        let tile1 = CompressedTile {
            tile_idx: 1,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::AosZstd {
                predictors: vec![0, 0],
                color: EncodedBytes::Zstd(vec![1, 2, 3, 4, 5]),
                alpha: AlphaChannel::Omitted,
            }),
        };
        assert_eq!(
            CompressedTiles {
                metadata: header.metadata,
                tile_rects: header.tile_rects.clone(),
                tiles: vec![tile0.clone(), tile1.clone()],
            }
            .size(),
            3 + 2 + 5
        );

        // `StreamingCompressedTiles::uncompressed_size` accounts for exact
        // `tile_rects` pixel area and `BufferFormat` channel count.
        let full_rect = vec![TileRect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        }];
        for (format, rects, expected) in [
            (
                BufferFormat::Xrgb8888,
                header.tile_rects.clone(),
                16 * 64 * 3,
            ),
            (
                BufferFormat::Argb8888,
                header.tile_rects.clone(),
                16 * 64 * 4,
            ),
            (BufferFormat::Xrgb8888, full_rect.clone(), 64 * 64 * 3),
            (BufferFormat::Argb8888, full_rect, 64 * 64 * 4),
        ] {
            let (_tx_u, rx_u) = crossbeam_channel::unbounded();
            let mut h = header.clone();
            h.metadata.format = format;
            h.tile_rects = rects;
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
        assert_eq!(written_bytes, 10);

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
        let large_rects = [
            TileRect {
                x: 0,
                y: 0,
                w: 256,
                h: 256,
            },
            TileRect {
                x: 256,
                y: 0,
                w: 256,
                h: 256,
            },
            TileRect {
                x: 0,
                y: 256,
                w: 256,
                h: 256,
            },
            TileRect {
                x: 256,
                y: 256,
                w: 256,
                h: 256,
            },
        ];
        let pred_len = 256usize.div_ceil(4);
        let large0 = CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::AosZstd {
                predictors: vec![0; pred_len],
                color: EncodedBytes::Zstd(vec![0xAA; half_cap - pred_len]),
                alpha: AlphaChannel::Omitted,
            }),
        };
        let large1 = CompressedTile {
            tile_idx: 1,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::AosZstd {
                predictors: vec![0; pred_len],
                color: EncodedBytes::Zstd(vec![0xBB; half_cap - pred_len]),
                alpha: AlphaChannel::Omitted,
            }),
        };
        let small2 = CompressedTile {
            tile_idx: 2,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Solid([5, 6, 7, 0xFF])),
        };
        let small3 = CompressedTile {
            tile_idx: 3,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Solid([8, 9, 10, 0xFF])),
        };
        assert_eq!(
            large0.compressed_len() + large1.compressed_len(),
            MAX_COALESCED_TILE_BYTES
        );

        let batch_header = RawBufferHeader {
            frame_seq: 2,
            client: ClientId(7),
            surface: WlSurfaceId(11),
            metadata: BufferMetadata {
                width: 512,
                height: 512,
                stride: 512 * 4,
                format: BufferFormat::Xrgb8888,
            },
            incremental: false,
            scroll_rect: None,
            reset_tile_cache: false,
            evict_slots: Vec::new(),
            tile_rects: large_rects.to_vec(),
        };
        let mut expected_hdr_wire = Vec::new();
        batch_header.framed_write(&mut expected_hdr_wire).unwrap();
        let mut expected_b1 = expected_hdr_wire.clone();
        large0
            .write_framed(large_rects[0], false, &mut expected_b1)
            .unwrap();
        large1
            .write_framed(large_rects[1], false, &mut expected_b1)
            .unwrap();
        let mut expected_b2 = expected_b1.clone();
        small2
            .write_framed(large_rects[2], false, &mut expected_b2)
            .unwrap();
        let mut expected_b3 = expected_b2.clone();
        small3
            .write_framed(large_rects[3], false, &mut expected_b3)
            .unwrap();

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

        // Dropped sender before `n_tiles` are sent must return `Err`.
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

        // Mid-stream truncation after 1 of 2 tiles must return `Err` AND forward an `Err` item.
        let mut truncated_wire = Vec::new();
        header.framed_write(&mut truncated_wire).unwrap();
        tile0
            .write_framed(rect0, false, &mut truncated_wire)
            .unwrap();
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

        // Early drop of `stream.tiles` must still drain all `n_tiles` off `stream`.
        let mut wire_with_trailer = wire.clone();
        wire_with_trailer.push(0xFE);
        let mut reader = wire_with_trailer.as_slice();
        CompressedTileStream::read_from_stream(&mut reader, |stream| {
            drop(stream.tiles);
            Ok(())
        })
        .unwrap();
        assert_eq!(reader, &[0xFE]);

        // Returning `Err` from `on_stream` must abort immediately after reading `RawBufferHeader`.
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

        let mut mismatched_len = Vec::new();
        comp.write_framed(&large, &mut mismatched_len).unwrap();
        assert_eq!(mismatched_len[4], 1);
        mismatched_len[0..4].copy_from_slice(&(large.len() as u32 + 64).to_be_bytes());
        assert!(
            decomp
                .read_framed_with(&mut mismatched_len.as_slice(), |_| Ok(()))
                .is_err()
        );
        assert_eq!(decomp.output_buf.len(), large.len());

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

        let mut short_compressed = Vec::new();
        comp.write_framed(&large, &mut short_compressed).unwrap();
        short_compressed[5..9].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(
            decomp
                .read_framed_with(&mut short_compressed.as_slice(), |_| Ok(()))
                .is_err()
        );

        let mut bad_tag = Vec::new();
        16usize.framed_write(&mut bad_tag).unwrap();
        9u8.framed_write(&mut bad_tag).unwrap();
        assert!(
            decomp
                .read_framed_with(&mut bad_tag.as_slice(), |_| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn test_compression_mode_and_jpeg_subsamp() {
        assert_eq!(JpegSubsamp::default(), JpegSubsamp::Sub444);
        assert_eq!(CompressionMode::default(), CompressionMode::Lossless);
        assert_eq!(
            CompressionMode::hybrid_default(),
            CompressionMode::Hybrid {
                quality: 90,
                subsamp: JpegSubsamp::Sub444,
            }
        );

        for (s, subsamp, tj) in [
            ("444", JpegSubsamp::Sub444, turbojpeg::Subsamp::None),
            ("422", JpegSubsamp::Sub422, turbojpeg::Subsamp::Sub2x1),
            ("420", JpegSubsamp::Sub420, turbojpeg::Subsamp::Sub2x2),
        ] {
            assert_eq!(JpegSubsamp::from_str(s).unwrap(), subsamp);
            assert_eq!(subsamp.to_string(), s);
            assert_eq!(turbojpeg::Subsamp::from(subsamp), tj);
        }
        for bad in ["", "411", "440", "gray", "Sub444"] {
            assert!(JpegSubsamp::from_str(bad).is_err());
        }

        assert!(CompressionMode::Lossless.validate().is_ok());
        assert!(CompressionMode::hybrid_default().validate().is_ok());
        assert!(
            CompressionMode::Hybrid {
                quality: 1,
                subsamp: JpegSubsamp::Sub420,
            }
            .validate()
            .is_ok()
        );
        assert!(
            CompressionMode::Hybrid {
                quality: 100,
                subsamp: JpegSubsamp::Sub422,
            }
            .validate()
            .is_ok()
        );
        for bad_q in [0u8, 101, 255] {
            assert!(
                CompressionMode::Hybrid {
                    quality: bad_q,
                    subsamp: JpegSubsamp::Sub444,
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn test_jpeg_and_split_framing_and_bounds() {
        let tr = |x, y, w, h| TileRect { x, y, w, h };
        let parent_256 = tr(256, 0, 256, 256);
        let parent_256x128 = tr(0, 256, 256, 128);
        let small_128 = tr(0, 0, 128, 128);

        // 4-quadrant Split roundtrip (mixed Solid + Palette + AosZstd + Jpeg).
        let split4 = CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Split(vec![
                SubTile {
                    rect: tr(256, 0, 128, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
                SubTile {
                    rect: tr(384, 0, 128, 128),
                    payload: LeafPayload::Jpeg {
                        data: vec![0xFF, 0xD8, 0x01, 0xFF, 0xD9],
                        alpha: AlphaChannel::Omitted,
                    },
                },
                SubTile {
                    rect: tr(256, 128, 128, 128),
                    payload: LeafPayload::Palette {
                        has_alpha: false,
                        palette: vec![10, 20, 30, 40, 50, 60],
                        data: EncodedBytes::Zstd(vec![1, 2, 3]),
                    },
                },
                SubTile {
                    rect: tr(384, 128, 128, 128),
                    payload: LeafPayload::Jpeg {
                        data: vec![0xFF, 0xD8, 0x02, 0xFF, 0xD9],
                        alpha: AlphaChannel::Constant(0x80),
                    },
                },
            ]),
        };
        let mut wire = Vec::new();
        split4.write_framed(parent_256, true, &mut wire).unwrap();
        // Translucent sub-tile fails on Xrgb8888 (both write and read).
        assert!(
            split4
                .write_framed(parent_256, false, &mut Vec::new())
                .is_err()
        );
        assert!(CompressedTile::read_framed(&[parent_256], false, &mut wire.as_slice()).is_err());
        let decoded =
            CompressedTile::read_framed(&[parent_256], true, &mut wire.as_slice()).unwrap();
        assert_eq!(decoded, split4);

        // 2-quadrant Split on `256x128` roundtrip.
        let split2 = CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Split(vec![
                SubTile {
                    rect: tr(0, 256, 128, 128),
                    payload: LeafPayload::Solid([9, 8, 7, 0xFF]),
                },
                SubTile {
                    rect: tr(128, 256, 128, 128),
                    payload: LeafPayload::Jpeg {
                        data: vec![0xFF, 0xD8, 0xFF, 0xD9],
                        alpha: AlphaChannel::Omitted,
                    },
                },
            ]),
        };
        let mut wire2 = Vec::new();
        split2
            .write_framed(parent_256x128, false, &mut wire2)
            .unwrap();
        assert_eq!(
            CompressedTile::read_framed(&[parent_256x128], false, &mut wire2.as_slice()).unwrap(),
            split2
        );

        // Oversized JPEG length (`max_jpeg_bytes + 1`) rejected BEFORE reading/allocating payload.
        let small_px = small_128.checked_pixels().unwrap();
        let over_jpeg = max_jpeg_bytes(small_px) + 1;
        let mut bad_jpeg_wire = Vec::new();
        0u32.framed_write(&mut bad_jpeg_wire).unwrap();
        PAYLOAD_TAG_JPEG_OPAQUE
            .framed_write(&mut bad_jpeg_wire)
            .unwrap();
        over_jpeg.framed_write(&mut bad_jpeg_wire).unwrap();
        let hdr_len = bad_jpeg_wire.len();
        bad_jpeg_wire.resize(hdr_len + 16, 0xCD);
        let mut rdr = bad_jpeg_wire.as_slice();
        assert!(CompressedTile::read_framed(&[small_128], false, &mut rdr).is_err());
        assert_eq!(rdr.len(), 16);

        // Empty JPEG payload rejected on both write and read.
        let empty_jpeg = CompressedTile {
            tile_idx: 0,
            cache_put: None,
            payload: TilePayload::Leaf(LeafPayload::Jpeg {
                data: Vec::new(),
                alpha: AlphaChannel::Omitted,
            }),
        };
        assert!(
            empty_jpeg
                .write_framed(small_128, false, &mut Vec::new())
                .is_err()
        );
        let mut empty_jpeg_wire = Vec::new();
        0u32.framed_write(&mut empty_jpeg_wire).unwrap();
        PAYLOAD_TAG_JPEG_OPAQUE
            .framed_write(&mut empty_jpeg_wire)
            .unwrap();
        0u32.framed_write(&mut empty_jpeg_wire).unwrap();
        assert!(
            CompressedTile::read_framed(&[small_128], false, &mut empty_jpeg_wire.as_slice())
                .is_err()
        );

        // Split on non-splittable `<= 128x128` tile rejected before reading count.
        let mut nonsplit_wire = Vec::new();
        0u32.framed_write(&mut nonsplit_wire).unwrap();
        PAYLOAD_TAG_SPLIT.framed_write(&mut nonsplit_wire).unwrap();
        2u8.framed_write(&mut nonsplit_wire).unwrap();
        let mut rdr = nonsplit_wire.as_slice();
        assert!(CompressedTile::read_framed(&[small_128], false, &mut rdr).is_err());
        assert_eq!(rdr.len(), 1);

        // Invalid Split count (`0`, `1`, `5`) rejected BEFORE reading sub-tiles.
        for bad_count in [0u8, 1, 5, 255] {
            let mut bad_cnt_wire = Vec::new();
            0u32.framed_write(&mut bad_cnt_wire).unwrap();
            PAYLOAD_TAG_SPLIT.framed_write(&mut bad_cnt_wire).unwrap();
            bad_count.framed_write(&mut bad_cnt_wire).unwrap();
            bad_cnt_wire.extend_from_slice(&[0xEE; 16]);
            let mut rdr = bad_cnt_wire.as_slice();
            assert!(CompressedTile::read_framed(&[parent_256], false, &mut rdr).is_err());
            assert_eq!(rdr.len(), 16);
        }

        // Nested Split (tag `PAYLOAD_TAG_SPLIT` inside a SubTile) must fail!
        let mut nested_wire = Vec::new();
        0u32.framed_write(&mut nested_wire).unwrap();
        PAYLOAD_TAG_SPLIT.framed_write(&mut nested_wire).unwrap();
        2u8.framed_write(&mut nested_wire).unwrap();
        tr(0, 256, 128, 128).write_framed(&mut nested_wire).unwrap();
        PAYLOAD_TAG_SPLIT.framed_write(&mut nested_wire).unwrap();
        assert!(
            CompressedTile::read_framed(&[parent_256x128], false, &mut nested_wire.as_slice())
                .is_err()
        );

        // Out-of-bounds, overlapping, and incomplete-coverage Splits must fail on read and validate.
        for bad_subs in [
            // Out of bounds (`x` shifted past parent right edge)
            vec![
                SubTile {
                    rect: tr(0, 256, 128, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
                SubTile {
                    rect: tr(129, 256, 128, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
            ],
            // Overlapping sub-tiles whose summed area equals parent area (only overlap check catches this).
            vec![
                SubTile {
                    rect: tr(0, 256, 160, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
                SubTile {
                    rect: tr(64, 256, 96, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
            ],
            // Incomplete coverage (hole in parent)
            vec![
                SubTile {
                    rect: tr(0, 256, 128, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
                SubTile {
                    rect: tr(128, 256, 127, 128),
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
            ],
        ] {
            assert!(validate_split_rects(parent_256x128, &bad_subs).is_err());
            let mut bad_wire = Vec::new();
            0u32.framed_write(&mut bad_wire).unwrap();
            PAYLOAD_TAG_SPLIT.framed_write(&mut bad_wire).unwrap();
            (bad_subs.len() as u8).framed_write(&mut bad_wire).unwrap();
            for sub in &bad_subs {
                sub.rect.write_framed(&mut bad_wire).unwrap();
                sub.payload
                    .write_framed(
                        sub.rect,
                        sub.rect.checked_pixels().unwrap(),
                        false,
                        &mut bad_wire,
                    )
                    .unwrap();
            }
            assert!(
                CompressedTile::read_framed(&[parent_256x128], false, &mut bad_wire.as_slice())
                    .is_err()
            );
        }
    }

    #[test]
    fn test_scroll_rect_framing_and_malformed_bounds() {
        let meta = BufferMetadata {
            width: 1920,
            height: 1080,
            stride: 1920 * 4,
            format: BufferFormat::Xrgb8888,
        };

        // Roundtrip both scroll directions (`dy < 0` and `dy > 0`), with and without tiles.
        for (sr, expected_dst, tiles) in [
            (
                ScrollRect {
                    x: 0,
                    y: 80,
                    width: 1920,
                    height: 1000,
                    dy: -36,
                },
                TileRect {
                    x: 0,
                    y: 80,
                    w: 1920,
                    h: 964,
                },
                vec![TileRect {
                    x: 0,
                    y: 1024,
                    w: 256,
                    h: 56,
                }],
            ),
            (
                ScrollRect {
                    x: 300,
                    y: 60,
                    width: 1608,
                    height: 1000,
                    dy: 120,
                },
                TileRect {
                    x: 300,
                    y: 180,
                    w: 1608,
                    h: 880,
                },
                vec![],
            ),
        ] {
            sr.validate_in_bounds(1920, 1080).unwrap();
            assert_eq!(sr.dst_rect(), expected_dst);
            assert_eq!(
                sr.dst_damage_rect(),
                Rectangle::new(
                    expected_dst.x as i32,
                    expected_dst.y as i32,
                    expected_dst.w as i32,
                    expected_dst.h as i32,
                )
            );
            let hdr = RawBufferHeader {
                frame_seq: 77,
                client: ClientId(5),
                surface: WlSurfaceId(9),
                metadata: meta,
                incremental: true,
                scroll_rect: Some(sr),
                reset_tile_cache: false,
                evict_slots: Vec::new(),
                tile_rects: tiles.clone(),
            };
            let mut wire = Vec::new();
            hdr.framed_write(&mut wire).unwrap();
            assert_eq!(
                wire.len(),
                RAW_BUFFER_HEADER_LEN + SCROLL_RECT_LEN + tiles.len() * TILE_RECT_LEN
            );
            assert_eq!(wire[37], HEADER_FLAG_INCREMENTAL | HEADER_FLAG_SCROLL_RECT);
            let decoded = RawBufferHeader::framed_read(&mut wire.as_slice()).unwrap();
            assert_eq!(decoded, hdr);
        }

        // Non-scroll header adds zero wire bytes (still 42 B fixed prefix).
        let plain_hdr = RawBufferHeader {
            frame_seq: 78,
            client: ClientId(5),
            surface: WlSurfaceId(9),
            metadata: meta,
            incremental: true,
            scroll_rect: None,
            reset_tile_cache: false,
            evict_slots: Vec::new(),
            tile_rects: vec![],
        };
        let mut plain_wire = Vec::new();
        plain_hdr.framed_write(&mut plain_wire).unwrap();
        assert_eq!(plain_wire.len(), RAW_BUFFER_HEADER_LEN);

        // Non-incremental header with `scroll_rect: Some(_)` must fail on write AND read.
        let valid_sr = ScrollRect {
            x: 0,
            y: 0,
            width: 640,
            height: 480,
            dy: -16,
        };
        let bad_non_inc = RawBufferHeader {
            frame_seq: 1,
            client: ClientId(1),
            surface: WlSurfaceId(1),
            metadata: meta,
            incremental: false,
            scroll_rect: Some(valid_sr),
            reset_tile_cache: false,
            evict_slots: Vec::new(),
            tile_rects: vec![],
        };
        assert!(bad_non_inc.framed_write(&mut Vec::new()).is_err());

        // Malformed ScrollRect bounds: `dy == 0`, `|dy| >= height`, `i32::MIN`,
        // `width == 0`, `height == 0`, out-of-bounds, and u32::MAX overflow.
        let bad_scrolls = [
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
                dy: 0,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
                dy: 480,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
                dy: -480,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
                dy: -481,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480,
                dy: i32::MIN,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 0,
                height: 480,
                dy: -10,
            },
            ScrollRect {
                x: 0,
                y: 0,
                width: 640,
                height: 0,
                dy: -10,
            },
            ScrollRect {
                x: 1900,
                y: 0,
                width: 21,
                height: 480,
                dy: -10,
            },
            ScrollRect {
                x: 0,
                y: 1000,
                width: 640,
                height: 81,
                dy: -10,
            },
            ScrollRect {
                x: u32::MAX,
                y: 0,
                width: 1,
                height: 480,
                dy: -10,
            },
            ScrollRect {
                x: 0,
                y: u32::MAX,
                width: 640,
                height: 1,
                dy: -1,
            },
        ];

        for bad_sr in bad_scrolls {
            assert!(bad_sr.validate_in_bounds(1920, 1080).is_err());
            let mut wire = [0u8; RAW_BUFFER_HEADER_LEN].to_vec();
            wire[24..28].copy_from_slice(&1920i32.to_be_bytes());
            wire[28..32].copy_from_slice(&1080i32.to_be_bytes());
            wire[32..36].copy_from_slice(&(1920i32 * 4).to_be_bytes());
            wire[36] = 1; // Xrgb8888
            wire[37] = HEADER_FLAG_INCREMENTAL | HEADER_FLAG_SCROLL_RECT;
            wire[38..42].copy_from_slice(&0u32.to_be_bytes());
            bad_sr.write_framed(&mut wire).unwrap();
            assert!(
                RawBufferHeader::framed_read(&mut wire.as_slice()).is_err(),
                "expected error for malformed ScrollRect {bad_sr:?}"
            );
        }

        // Invalid flags (`0x02` = non-incremental + scroll, `0x10` = reserved bit,
        // `reset_tile_cache | evict_slots`) and truncated ScrollRect payload must fail cleanly.
        for bad_flag in [
            HEADER_FLAG_SCROLL_RECT,
            HEADER_FLAG_RESET_TILE_CACHE | HEADER_FLAG_EVICT_SLOTS,
            0x10,
            0xFF,
        ] {
            let mut wire = [0u8; RAW_BUFFER_HEADER_LEN].to_vec();
            wire[24..28].copy_from_slice(&640i32.to_be_bytes());
            wire[28..32].copy_from_slice(&480i32.to_be_bytes());
            wire[32..36].copy_from_slice(&(640i32 * 4).to_be_bytes());
            wire[36] = 1;
            wire[37] = bad_flag;
            valid_sr.write_framed(&mut wire).unwrap();
            assert!(RawBufferHeader::framed_read(&mut wire.as_slice()).is_err());
        }
        let mut truncated = [0u8; RAW_BUFFER_HEADER_LEN + 10];
        truncated[24..28].copy_from_slice(&640i32.to_be_bytes());
        truncated[28..32].copy_from_slice(&480i32.to_be_bytes());
        truncated[32..36].copy_from_slice(&(640i32 * 4).to_be_bytes());
        truncated[36] = 1;
        truncated[37] = HEADER_FLAG_INCREMENTAL | HEADER_FLAG_SCROLL_RECT;
        assert!(RawBufferHeader::framed_read(&mut truncated.as_slice()).is_err());
    }

    #[test]
    fn test_tile_cache_framing_and_malformed_bounds() {
        let meta = BufferMetadata {
            width: 512,
            height: 512,
            stride: 512 * 4,
            format: BufferFormat::Argb8888,
        };
        let rect_256 = TileRect {
            x: 0,
            y: 0,
            w: 256,
            h: 256,
        };
        let rect_128 = TileRect {
            x: 256,
            y: 0,
            w: 128,
            h: 128,
        };

        // 1. Header roundtrip with `reset_tile_cache` and with `evict_slots`.
        let reset_hdr = RawBufferHeader {
            frame_seq: 1,
            client: ClientId(2),
            surface: WlSurfaceId(3),
            metadata: meta,
            incremental: false,
            scroll_rect: None,
            reset_tile_cache: true,
            evict_slots: Vec::new(),
            tile_rects: vec![rect_256],
        };
        let mut wire = Vec::new();
        reset_hdr.framed_write(&mut wire).unwrap();
        assert_eq!(wire.len(), RAW_BUFFER_HEADER_LEN + TILE_RECT_LEN);
        assert_eq!(wire[37], HEADER_FLAG_RESET_TILE_CACHE);
        assert_eq!(
            RawBufferHeader::framed_read(&mut wire.as_slice()).unwrap(),
            reset_hdr
        );

        let evict_hdr = RawBufferHeader {
            frame_seq: 2,
            client: ClientId(2),
            surface: WlSurfaceId(3),
            metadata: meta,
            incremental: true,
            scroll_rect: None,
            reset_tile_cache: false,
            evict_slots: vec![0, 511, (MAX_TILE_CACHE_SLOTS - 1) as u16],
            tile_rects: vec![rect_128],
        };
        let mut evict_wire = Vec::new();
        evict_hdr.framed_write(&mut evict_wire).unwrap();
        assert_eq!(
            evict_wire.len(),
            RAW_BUFFER_HEADER_LEN + 2 + 3 * 2 + TILE_RECT_LEN
        );
        assert_eq!(
            evict_wire[37],
            HEADER_FLAG_INCREMENTAL | HEADER_FLAG_EVICT_SLOTS
        );
        assert_eq!(
            RawBufferHeader::framed_read(&mut evict_wire.as_slice()).unwrap(),
            evict_hdr
        );

        // Contradictory `reset_tile_cache + evict_slots`, out-of-range, and
        // duplicate `evict_slots` must fail on write AND read.
        for bad_hdr in [
            RawBufferHeader {
                reset_tile_cache: true,
                evict_slots: vec![1],
                ..evict_hdr.clone()
            },
            RawBufferHeader {
                evict_slots: vec![MAX_TILE_CACHE_SLOTS as u16],
                ..evict_hdr.clone()
            },
            RawBufferHeader {
                evict_slots: vec![7, 7],
                ..evict_hdr.clone()
            },
        ] {
            assert!(bad_hdr.framed_write(&mut Vec::new()).is_err());
        }

        // Malformed wire `evict_slots`: count=0, count=1025, slot=1024, duplicate slot.
        let make_evict_wire = |count: u16, slots: &[u16]| {
            let mut w = [0u8; RAW_BUFFER_HEADER_LEN].to_vec();
            w[24..28].copy_from_slice(&512i32.to_be_bytes());
            w[28..32].copy_from_slice(&512i32.to_be_bytes());
            w[32..36].copy_from_slice(&(512i32 * 4).to_be_bytes());
            w[36] = 1;
            w[37] = HEADER_FLAG_EVICT_SLOTS;
            count.framed_write(&mut w).unwrap();
            for &s in slots {
                s.framed_write(&mut w).unwrap();
            }
            w
        };
        for bad_wire in [
            make_evict_wire(0, &[]),
            make_evict_wire((MAX_TILE_CACHE_SLOTS + 1) as u16, &[]),
            make_evict_wire(1, &[MAX_TILE_CACHE_SLOTS as u16]),
            make_evict_wire(2, &[5, 5]),
        ] {
            assert!(RawBufferHeader::framed_read(&mut bad_wire.as_slice()).is_err());
        }

        // 2. Roundtrip `CacheHit` (all 5 regions) and `CachePut` (Leaf + Split).
        for region in [
            CacheHitRegion::Full,
            CacheHitRegion::TopLeft,
            CacheHitRegion::TopRight,
            CacheHitRegion::BottomLeft,
            CacheHitRegion::BottomRight,
        ] {
            let target_rect = if region == CacheHitRegion::Full {
                rect_256
            } else {
                rect_128
            };
            let hit = CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::CacheHit(CacheHitRef {
                    slot_id: (MAX_TILE_CACHE_SLOTS - 1) as u16,
                    region,
                }),
            };
            let mut w = Vec::new();
            hit.write_framed(target_rect, true, &mut w).unwrap();
            assert_eq!(w.len(), 4 + 1 + 2);
            let dec = CompressedTile::read_framed(&[target_rect], true, &mut w.as_slice()).unwrap();
            assert_eq!(dec, hit);
        }

        let put_solid = CompressedTile {
            tile_idx: 0,
            cache_put: Some(42),
            payload: TilePayload::Leaf(LeafPayload::Solid([10, 20, 30, 0xFF])),
        };
        let mut w = Vec::new();
        put_solid.write_framed(rect_128, false, &mut w).unwrap();
        assert_eq!(w.len(), 4 + 1 + 2 + 3);
        assert_eq!(
            CompressedTile::read_framed(&[rect_128], false, &mut w.as_slice()).unwrap(),
            put_solid
        );

        let put_split = CompressedTile {
            tile_idx: 0,
            cache_put: Some(1023),
            payload: TilePayload::Split(vec![
                SubTile {
                    rect: TileRect {
                        x: 0,
                        y: 0,
                        w: 128,
                        h: 256,
                    },
                    payload: LeafPayload::Solid([1, 2, 3, 0xFF]),
                },
                SubTile {
                    rect: TileRect {
                        x: 128,
                        y: 0,
                        w: 128,
                        h: 256,
                    },
                    payload: LeafPayload::Solid([4, 5, 6, 0xFF]),
                },
            ]),
        };
        let mut ws = Vec::new();
        put_split.write_framed(rect_256, false, &mut ws).unwrap();
        assert_eq!(
            CompressedTile::read_framed(&[rect_256], false, &mut ws.as_slice()).unwrap(),
            put_split
        );

        // 3. Reject malformed `CacheHit` / `CachePut` combinations on write & read.
        // CacheHit + CachePut together.
        assert!(
            CompressedTile {
                tile_idx: 0,
                cache_put: Some(0),
                payload: TilePayload::CacheHit(CacheHitRef {
                    slot_id: 0,
                    region: CacheHitRegion::Full,
                }),
            }
            .write_framed(rect_128, false, &mut Vec::new())
            .is_err()
        );
        // Out-of-range `cache_put` slot_id.
        assert!(
            CompressedTile {
                tile_idx: 0,
                cache_put: Some(MAX_TILE_CACHE_SLOTS as u16),
                payload: TilePayload::Leaf(LeafPayload::Solid([0, 0, 0, 0xFF])),
            }
            .write_framed(rect_128, false, &mut Vec::new())
            .is_err()
        );
        // Out-of-range `CacheHit` slot_id.
        assert!(
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::CacheHit(CacheHitRef {
                    slot_id: MAX_TILE_CACHE_SLOTS as u16,
                    region: CacheHitRegion::Full,
                }),
            }
            .write_framed(rect_128, false, &mut Vec::new())
            .is_err()
        );
        // Quadrant `CacheHitRegion` targeting a `256x256` rect must fail.
        assert!(
            CompressedTile {
                tile_idx: 0,
                cache_put: None,
                payload: TilePayload::CacheHit(CacheHitRef {
                    slot_id: 0,
                    region: CacheHitRegion::TopLeft,
                }),
            }
            .write_framed(rect_256, false, &mut Vec::new())
            .is_err()
        );

        // Wire-level malformed checks:
        // - `PAYLOAD_FLAG_CACHE_PUT | PAYLOAD_TAG_CACHE_HIT`
        let mut hit_with_put = Vec::new();
        0u32.framed_write(&mut hit_with_put).unwrap();
        (PAYLOAD_FLAG_CACHE_PUT | PAYLOAD_TAG_CACHE_HIT)
            .framed_write(&mut hit_with_put)
            .unwrap();
        0u16.framed_write(&mut hit_with_put).unwrap();
        assert!(
            CompressedTile::read_framed(&[rect_128], false, &mut hit_with_put.as_slice()).is_err()
        );

        // - Invalid `CacheHitRef` region bits (`5`, `7`, high reserved bits set)
        for bad_packed in [5u16 << 10, 7u16 << 10, 0x8000u16] {
            assert!(CacheHitRef::from_u16(bad_packed).is_err());
            let mut bw = Vec::new();
            0u32.framed_write(&mut bw).unwrap();
            PAYLOAD_TAG_CACHE_HIT.framed_write(&mut bw).unwrap();
            bad_packed.framed_write(&mut bw).unwrap();
            assert!(CompressedTile::read_framed(&[rect_128], false, &mut bw.as_slice()).is_err());
        }

        // - Wire `CachePut` with `slot_id == 1024`
        let mut bad_put = Vec::new();
        0u32.framed_write(&mut bad_put).unwrap();
        (PAYLOAD_FLAG_CACHE_PUT | PAYLOAD_TAG_SOLID_OPAQUE)
            .framed_write(&mut bad_put)
            .unwrap();
        (MAX_TILE_CACHE_SLOTS as u16)
            .framed_write(&mut bad_put)
            .unwrap();
        bad_put.extend_from_slice(&[1, 2, 3]);
        assert!(CompressedTile::read_framed(&[rect_128], false, &mut bad_put.as_slice()).is_err());

        // - Wire quadrant `CacheHit` for a `256x256` rect
        let mut quad_on_256 = Vec::new();
        0u32.framed_write(&mut quad_on_256).unwrap();
        PAYLOAD_TAG_CACHE_HIT
            .framed_write(&mut quad_on_256)
            .unwrap();
        CacheHitRef {
            slot_id: 0,
            region: CacheHitRegion::BottomRight,
        }
        .to_u16()
        .unwrap()
        .framed_write(&mut quad_on_256)
        .unwrap();
        assert!(
            CompressedTile::read_framed(&[rect_256], false, &mut quad_on_256.as_slice()).is_err()
        );
    }
}
