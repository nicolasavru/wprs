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

//! Fast tile palette encoder/decoder (`<= 256` unique colors, 1/2/4/8 bpp).
//!
//! Uses a 512-slot open-addressed table (4 KiB, L1d-resident) with a generation
//! counter to avoid per-tile clearing, bailing out immediately on the 257th
//! unique color so photo tiles exit within a few hundred pixels.

use crate::buffer_pointer::BufferPointer;
use crate::prelude::*;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::packed_palette_indices_len;
use crate::vec4u8::Vec4u8;

const MAX_PALETTE_COLORS: usize = 256;
const HASH_SLOTS: usize = 512;
const HASH_MASK: usize = HASH_SLOTS - 1;

fn pack_indices(indices: &[u8], n_colors: usize, packed: &mut [u8]) -> usize {
    let n = indices.len();
    let out_len = packed_palette_indices_len(n, n_colors);
    if n_colors <= 1 {
        return 0;
    } else if n_colors <= 2 {
        for (oi, chunk) in indices.chunks(8).enumerate() {
            let mut b = 0u8;
            for (bit, &v) in chunk.iter().enumerate() {
                b |= (v & 1) << bit;
            }
            packed[oi] = b;
        }
    } else if n_colors <= 4 {
        for (oi, chunk) in indices.chunks(4).enumerate() {
            let mut b = 0u8;
            for (s, &v) in chunk.iter().enumerate() {
                b |= (v & 3) << (s * 2);
            }
            packed[oi] = b;
        }
    } else if n_colors <= 16 {
        let pairs = n / 2;
        for oi in 0..pairs {
            packed[oi] = (indices[oi * 2] & 0x0F) | ((indices[oi * 2 + 1] & 0x0F) << 4);
        }
        if n % 2 == 1 {
            packed[pairs] = indices[n - 1] & 0x0F;
        }
    } else {
        packed[..n].copy_from_slice(indices);
    }
    out_len
}

pub(crate) struct PaletteScratch {
    keys: [u32; HASH_SLOTS],
    vals: [u8; HASH_SLOTS],
    gens: [u32; HASH_SLOTS],
    cur_gen: u32,
    pub(crate) palette: [u32; MAX_PALETTE_COLORS],
    indices: Vec<u8>,
    pub(crate) packed: Vec<u8>,
}

impl PaletteScratch {
    pub(crate) fn new() -> Self {
        Self {
            keys: [0; HASH_SLOTS],
            vals: [0; HASH_SLOTS],
            gens: [0; HASH_SLOTS],
            cur_gen: 0,
            palette: [0; MAX_PALETTE_COLORS],
            indices: vec![0; TILE_PIXELS],
            packed: vec![0; TILE_PIXELS],
        }
    }

    /// Scans `rect` in `aos` (stride `stride_px` pixels). Returns
    /// `Some((n_colors, packed_len))` if the tile contains at most 256 unique
    /// colors, populating `self.palette[..n_colors]` and
    /// `self.packed[..packed_len]`. Returns `None` immediately upon seeing a
    /// 257th unique color.
    ///
    /// Reads directly from `BufferPointer` without creating a shared `&[u8]`
    /// slice over client shm memory.
    #[inline]
    pub(crate) fn try_encode_2d<const HAS_ALPHA: bool>(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> Option<(usize, usize)> {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;
        debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            self.gens.fill(0);
            self.cur_gen = 1;
        }
        let cur_epoch = self.cur_gen;
        let alpha_or = if HAS_ALPHA { 0u32 } else { 0xFF00_0000u32 };

        let mut count = 0usize;
        let mut last_color = 0u32;
        let mut last_idx = 0u8;
        let mut out_i = 0usize;

        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos` (`(y0 + ry) * stride_px + x0 + tw <= aos.len()`).
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            for x in 0..tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` `Vec4u8` reads.
                let px = unsafe { row_ptr.add(x).read() };
                let c = u32::from_le_bytes([px.0, px.1, px.2, px.3]) | alpha_or;
                if count > 0 && c == last_color {
                    self.indices[out_i] = last_idx;
                    out_i += 1;
                    continue;
                }
                let mut h = (c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
                loop {
                    if self.gens[h] != cur_epoch {
                        if count >= MAX_PALETTE_COLORS {
                            return None;
                        }
                        self.gens[h] = cur_epoch;
                        self.keys[h] = c;
                        let idx = count as u8;
                        self.vals[h] = idx;
                        self.palette[count] = c;
                        count += 1;
                        last_color = c;
                        last_idx = idx;
                        self.indices[out_i] = idx;
                        out_i += 1;
                        break;
                    }
                    if self.keys[h] == c {
                        let idx = self.vals[h];
                        last_color = c;
                        last_idx = idx;
                        self.indices[out_i] = idx;
                        out_i += 1;
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
            }
        }

        if count == 1 {
            return Some((1, 0));
        }
        let packed_len = pack_indices(&self.indices[..n_px], count, &mut self.packed);
        Some((count, packed_len))
    }
}

/// Validates `palette_bytes` and `packed`, returning the `[Vec4u8; 256]` lookup
/// table and active color count `n_colors` (`2..=256`).
pub(crate) fn prepare_palette_lut(
    pal_has_alpha: bool,
    palette_bytes: &[u8],
    packed: &[u8],
    n_pixels: usize,
) -> Result<([Vec4u8; MAX_PALETTE_COLORS], usize)> {
    let entry_bytes = if pal_has_alpha { 4 } else { 3 };
    if !palette_bytes.len().is_multiple_of(entry_bytes) {
        bail!(
            "palette byte len {} is not a multiple of {entry_bytes}",
            palette_bytes.len()
        );
    }
    let n_colors = palette_bytes.len() / entry_bytes;
    if !(2..=MAX_PALETTE_COLORS).contains(&n_colors) {
        bail!("invalid palette color count {n_colors}");
    }
    let expected_len = packed_palette_indices_len(n_pixels, n_colors);
    if packed.len() != expected_len {
        bail!(
            "packed palette indices len {} != expected {expected_len}",
            packed.len()
        );
    }

    // Validate canonical zero padding bits in the trailing byte and out-of-range
    // 8-bpp indices up front so row decoding needs no bounds checks on 8 bpp.
    if n_colors <= 2 {
        let rem = n_pixels & 7;
        if rem != 0 && (packed[n_pixels >> 3] >> rem) != 0 {
            bail!("non-zero trailing bits in 1-bpp palette stream");
        }
    } else if n_colors <= 4 {
        let rem = n_pixels & 3;
        if rem != 0 && (packed[n_pixels >> 2] >> (rem * 2)) != 0 {
            bail!("non-zero trailing bits in 2-bpp palette stream");
        }
    } else if n_colors <= 16 {
        if (n_pixels & 1) != 0 && (packed[n_pixels >> 1] >> 4) != 0 {
            bail!("non-zero trailing bits in 4-bpp palette stream");
        }
    } else if n_colors < MAX_PALETTE_COLORS && packed.iter().any(|&v| usize::from(v) >= n_colors) {
        bail!("palette index out of range for {n_colors}-color palette");
    }

    let mut lut = [Vec4u8(0, 0, 0, 0xFF); MAX_PALETTE_COLORS];
    if pal_has_alpha {
        for (dst, chunk) in lut[..n_colors]
            .iter_mut()
            .zip(palette_bytes.as_chunks::<4>().0)
        {
            *dst = Vec4u8(chunk[0], chunk[1], chunk[2], chunk[3]);
        }
    } else {
        for (dst, chunk) in lut[..n_colors]
            .iter_mut()
            .zip(palette_bytes.as_chunks::<3>().0)
        {
            *dst = Vec4u8(chunk[0], chunk[1], chunk[2], 0xFF);
        }
    }

    Ok((lut, n_colors))
}

/// Decodes one row (`dst_row.len()` pixels) from `packed` starting at pixel
/// offset `*idx_pos`, advancing `*idx_pos` by `dst_row.len()`.
#[inline]
pub(crate) fn decode_palette_row(
    lut: &[Vec4u8; MAX_PALETTE_COLORS],
    n_colors: usize,
    packed: &[u8],
    idx_pos: &mut usize,
    dst_row: &mut [Vec4u8],
) -> Result<()> {
    let mut pos = *idx_pos;
    if n_colors <= 2 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 3];
            *dst = lut[usize::from((b >> (pos & 7)) & 1)];
            pos += 1;
        }
    } else if n_colors <= 4 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 2];
            let v = usize::from((b >> ((pos & 3) * 2)) & 3);
            if v >= n_colors {
                bail!("2-bpp palette index {v} >= {n_colors}");
            }
            *dst = lut[v];
            pos += 1;
        }
    } else if n_colors <= 16 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 1];
            let v = usize::from(if (pos & 1) == 0 { b & 0x0F } else { b >> 4 });
            if v >= n_colors {
                bail!("4-bpp palette index {v} >= {n_colors}");
            }
            *dst = lut[v];
            pos += 1;
        }
    } else {
        let prow = &packed[pos..pos + dst_row.len()];
        for (dst, &v) in dst_row.iter_mut().zip(prow) {
            *dst = lut[usize::from(v)];
        }
        pos += dst_row.len();
    }
    *idx_pos = pos;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_decode_palette_case(
        colors: &[Vec4u8],
        has_alpha: bool,
        tw: usize,
        th: usize,
        stride_px: usize,
    ) {
        let mut frame = vec![Vec4u8(0xAA, 0xBB, 0xCC, 0xDD); stride_px * (th + 2)];
        let x0 = 2usize;
        let y0 = 1usize;
        for ry in 0..th {
            for rx in 0..tw {
                let c = colors[(ry * tw + rx) % colors.len()];
                frame[(y0 + ry) * stride_px + x0 + rx] = c;
            }
        }
        let rect = TileRect {
            x: x0 as u32,
            y: y0 as u32,
            w: tw as u32,
            h: th as u32,
        };
        let ptr = frame.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, frame.len()) };
        let mut scratch = PaletteScratch::new();
        let res = if has_alpha {
            scratch.try_encode_2d::<true>(bp, stride_px, rect)
        } else {
            scratch.try_encode_2d::<false>(bp, stride_px, rect)
        };
        let (n_colors, packed_len) = res.unwrap();
        assert_eq!(n_colors, colors.len());

        if n_colors == 1 {
            assert_eq!(packed_len, 0);
            return;
        }

        let pal_has_alpha = has_alpha
            && scratch.palette[..n_colors]
                .iter()
                .any(|&c| (c >> 24) != 0xFF);
        let mut pal_bytes = Vec::new();
        for &c in &scratch.palette[..n_colors] {
            let b = c.to_le_bytes();
            pal_bytes.extend_from_slice(if pal_has_alpha { &b[..] } else { &b[..3] });
        }
        let packed = &scratch.packed[..packed_len];
        let (lut, dec_n) = prepare_palette_lut(pal_has_alpha, &pal_bytes, packed, tw * th).unwrap();
        assert_eq!(dec_n, n_colors);

        let mut idx_pos = 0;
        let mut out_row = vec![Vec4u8(0, 0, 0, 0); tw];
        for ry in 0..th {
            decode_palette_row(&lut, dec_n, packed, &mut idx_pos, &mut out_row).unwrap();
            for rx in 0..tw {
                let mut exp = frame[(y0 + ry) * stride_px + x0 + rx];
                if !has_alpha {
                    exp.3 = 0xFF;
                }
                assert_eq!(out_row[rx], exp);
            }
        }
    }

    #[test]
    fn test_palette_1_2_4_8_bpp_and_overflow_bailout() {
        // Test exact boundary color counts: 1, 2 (1bpp), 3, 4 (2bpp), 5, 16 (4bpp), 17, 256 (8bpp),
        // with non-power-of-two odd pixel counts to exercise partial trailing bytes.
        for n in [1usize, 2, 3, 4, 5, 16, 17, 256] {
            let colors: Vec<Vec4u8> = (0..n)
                .map(|i| Vec4u8(i as u8, (i * 3) as u8, (i * 7) as u8, 0xFF))
                .collect();
            encode_decode_palette_case(&colors, false, 19, 17, 24);
            encode_decode_palette_case(&colors, true, 19, 17, 24);
        }

        // Translucent palette entries (same RGB, different alpha).
        let trans_colors = vec![
            Vec4u8(10, 20, 30, 0x10),
            Vec4u8(10, 20, 30, 0x80),
            Vec4u8(10, 20, 30, 0xFF),
        ];
        encode_decode_palette_case(&trans_colors, true, 7, 5, 12);

        // 257 unique colors must return None (early bailout).
        let pixels: Vec<Vec4u8> = (0..257u32)
            .map(|i| Vec4u8((i & 0xFF) as u8, (i >> 8) as u8, 0, 0xFF))
            .collect();
        let ptr = pixels.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
        let mut scratch = PaletteScratch::new();
        assert!(
            scratch
                .try_encode_2d::<false>(
                    bp,
                    257,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 257,
                        h: 1,
                    },
                )
                .is_none()
        );
    }

    #[test]
    fn test_palette_decode_rejects_malformed_inputs() {
        // Out-of-range 2-bpp index (3 for a 3-color palette).
        let pal3 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (lut3, n3) = prepare_palette_lut(false, &pal3, &[0b11], 1).unwrap();
        let mut row = [Vec4u8(0, 0, 0, 0); 1];
        assert!(decode_palette_row(&lut3, n3, &[0b11], &mut 0, &mut row).is_err());

        // Out-of-range 4-bpp index (10 for a 5-color palette).
        let pal5 = vec![0u8; 5 * 3];
        let (lut5, n5) = prepare_palette_lut(false, &pal5, &[0x0A], 1).unwrap();
        assert!(decode_palette_row(&lut5, n5, &[0x0A], &mut 0, &mut row).is_err());

        // Out-of-range 8-bpp index (17 for a 17-color palette) caught in prepare_palette_lut.
        let pal17 = vec![0u8; 17 * 3];
        assert!(prepare_palette_lut(false, &pal17, &[17], 1).is_err());

        // Non-zero trailing padding bits on 1/2/4-bpp.
        let pal2 = vec![0u8; 2 * 3];
        assert!(prepare_palette_lut(false, &pal2, &[0b0000_0010], 1).is_err());
        assert!(prepare_palette_lut(false, &pal3, &[0b0000_0100], 1).is_err());
        assert!(prepare_palette_lut(false, &pal5, &[0x10], 1).is_err());

        // Invalid palette byte length / color count / packed length.
        assert!(prepare_palette_lut(false, &[0, 1], &[0], 1).is_err());
        assert!(prepare_palette_lut(false, &[0, 1, 2], &[], 1).is_err());
        assert!(prepare_palette_lut(false, &pal2, &[0, 0], 1).is_err());
    }
}
