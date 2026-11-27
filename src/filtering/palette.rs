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

use std::mem::MaybeUninit;

use crate::buffer_pointer::BufferPointer;
use crate::prelude::*;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::packed_palette_indices_len;
use crate::vec4u8::Vec4u8;

const MAX_PALETTE_COLORS: usize = 256;
const HASH_SLOTS: usize = 512;
const HASH_MASK: usize = HASH_SLOTS - 1;

/// Dominant-color pixel share threshold (%) for [`PaletteScratch::is_text_safe`].
/// Catches ClearType / LCD subpixel-antialiased text tiles that exceed 256
/// unique RGB colors on a solid background.
const TEXT_SAFE_DOMINANT_BG_PCT: usize = 35;

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

/// Result of classifying a tile/quadrant for hybrid lossy/lossless encoding.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum TextSafeClass {
    /// Confirmed `<= 256` unique RGB colors, or early-exited on a dominant
    /// background before observing 257 unique colors (still eligible for
    /// palette encoding).
    PaletteCandidate,
    /// Confirmed `> 256` unique RGB colors (`> 256` BGRA colors too, so
    /// palette encoding cannot succeed) with a single RGB color covering
    /// `>= 35%` of pixels.
    DominantBg,
    /// Neither `<= 256` unique RGB colors nor `>= 35%` dominant background;
    /// eligible for JPEG encoding.
    Photo,
}

impl TextSafeClass {
    #[inline]
    pub(crate) fn is_text_safe(self) -> bool {
        !matches!(self, Self::Photo)
    }
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

    /// Classifies `rect` for hybrid lossy/lossless encoding: `TextSafe` iff
    /// `<= 256` unique RGB colors OR a single RGB color covering `>= 35%` of
    /// its pixels (alpha byte 3 is ignored because alpha is always coded
    /// losslessly).
    ///
    /// Runs unique-color tracking in the existing 512-slot hash table alongside
    /// 2-slot Misra-Gries heavy-hitter tracking (`k = 3`, mathematically
    /// guaranteed to retain any color with `> 1/3 = 33.3%`, hence any `>= 35%`
    /// color). Once the 257th unique RGB color appears, hashing stops and
    /// remaining pixels run a tight Misra-Gries-only loop. Because `n0` and
    /// `n1` are exact lower bounds on true color counts at every prefix, the
    /// scan exits early as soon as either reaches the `35%` threshold;
    /// otherwise exact bounds (`n <= count <= n + decrements`) resolve nearly
    /// all tiles in a single pass, running an exact verification pass only in
    /// the narrow uncertain band.
    #[inline]
    pub(crate) fn classify_text_safe(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> TextSafeClass {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;
        debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);
        if n_px <= MAX_PALETTE_COLORS {
            return TextSafeClass::PaletteCandidate;
        }

        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            self.gens.fill(0);
            self.cur_gen = 1;
        }
        let cur_epoch = self.cur_gen;

        let threshold = (n_px * TEXT_SAFE_DOMINANT_BG_PCT).div_ceil(100);
        let mut unique = 0usize;
        let mut tracking_unique = true;
        let mut last_color = 0u32;

        // 2-slot Misra-Gries heavy-hitter state (`n0 > 0 && n1 > 0 => c0 != c1`).
        let mut c0 = 0u32;
        let mut n0 = 0usize;
        let mut c1 = 0u32;
        let mut n1 = 0usize;
        let mut decrements = 0usize;

        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let mut x = 0usize;
            if tracking_unique {
                while x < tw {
                    // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                    let px = unsafe { row_ptr.add(x).read() };
                    x += 1;
                    let c = u32::from_le_bytes([px.0, px.1, px.2, 0xFF]);

                    if n0 > 0 && c == c0 {
                        n0 += 1;
                    } else if n1 > 0 && c == c1 {
                        n1 += 1;
                    } else if n0 == 0 {
                        c0 = c;
                        n0 = 1;
                    } else if n1 == 0 {
                        c1 = c;
                        n1 = 1;
                    } else {
                        n0 -= 1;
                        n1 -= 1;
                        decrements += 1;
                    }

                    if unique > 0 && c == last_color {
                        continue;
                    }
                    let mut h = (c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
                    loop {
                        if self.gens[h] != cur_epoch {
                            if unique >= MAX_PALETTE_COLORS {
                                tracking_unique = false;
                                break;
                            }
                            self.gens[h] = cur_epoch;
                            self.keys[h] = c;
                            unique += 1;
                            last_color = c;
                            break;
                        }
                        if self.keys[h] == c {
                            last_color = c;
                            break;
                        }
                        h = (h + 1) & HASH_MASK;
                    }
                    if !tracking_unique {
                        break;
                    }
                }
            }
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let px = unsafe { row_ptr.add(x).read() };
                x += 1;
                let c = u32::from_le_bytes([px.0, px.1, px.2, 0xFF]);

                if n0 > 0 && c == c0 {
                    n0 += 1;
                } else if n1 > 0 && c == c1 {
                    n1 += 1;
                } else if n0 == 0 {
                    c0 = c;
                    n0 = 1;
                } else if n1 == 0 {
                    c1 = c;
                    n1 = 1;
                } else {
                    n0 -= 1;
                    n1 -= 1;
                    decrements += 1;
                }
            }
            if n0 >= threshold || n1 >= threshold {
                return if tracking_unique {
                    TextSafeClass::PaletteCandidate
                } else {
                    TextSafeClass::DominantBg
                };
            }
        }

        if tracking_unique {
            return TextSafeClass::PaletteCandidate;
        }

        let check0 = n0 > 0 && n0 + decrements >= threshold;
        let check1 = n1 > 0 && n1 + decrements >= threshold;
        if !check0 && !check1 {
            return TextSafeClass::Photo;
        }

        let mut exact0 = 0usize;
        let mut exact1 = 0usize;
        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            for x in 0..tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let px = unsafe { row_ptr.add(x).read() };
                let c = u32::from_le_bytes([px.0, px.1, px.2, 0xFF]);
                if check0 && c == c0 {
                    exact0 += 1;
                } else if check1 && c == c1 {
                    exact1 += 1;
                }
            }
        }

        if exact0 >= threshold || exact1 >= threshold {
            TextSafeClass::DominantBg
        } else {
            TextSafeClass::Photo
        }
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
    dst_row: &mut [MaybeUninit<Vec4u8>],
) -> Result<()> {
    let mut pos = *idx_pos;
    if n_colors <= 2 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 3];
            dst.write(lut[usize::from((b >> (pos & 7)) & 1)]);
            pos += 1;
        }
    } else if n_colors <= 4 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 2];
            let v = usize::from((b >> ((pos & 3) * 2)) & 3);
            if v >= n_colors {
                bail!("2-bpp palette index {v} >= {n_colors}");
            }
            dst.write(lut[v]);
            pos += 1;
        }
    } else if n_colors <= 16 {
        for dst in dst_row.iter_mut() {
            let b = packed[pos >> 1];
            let v = usize::from(if (pos & 1) == 0 { b & 0x0F } else { b >> 4 });
            if v >= n_colors {
                bail!("4-bpp palette index {v} >= {n_colors}");
            }
            dst.write(lut[v]);
            pos += 1;
        }
    } else {
        let prow = &packed[pos..pos + dst_row.len()];
        for (dst, &v) in dst_row.iter_mut().zip(prow) {
            dst.write(lut[usize::from(v)]);
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
        let mut out_row = vec![MaybeUninit::uninit(); tw];
        for ry in 0..th {
            decode_palette_row(&lut, dec_n, packed, &mut idx_pos, &mut out_row).unwrap();
            for rx in 0..tw {
                let mut exp = frame[(y0 + ry) * stride_px + x0 + rx];
                if !has_alpha {
                    exp.3 = 0xFF;
                }
                // SAFETY: `decode_palette_row` initializes every element on `Ok`.
                assert_eq!(unsafe { out_row[rx].assume_init() }, exp);
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
        let mut row = [MaybeUninit::uninit(); 1];
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

    fn oracle_is_text_safe(frame: &[Vec4u8], stride_px: usize, rect: TileRect) -> bool {
        let mut counts = std::collections::HashMap::<u32, usize>::new();
        let n_px = rect.w as usize * rect.h as usize;
        for ry in 0..rect.h as usize {
            for rx in 0..rect.w as usize {
                let px = frame[(rect.y as usize + ry) * stride_px + rect.x as usize + rx];
                let rgb = u32::from_le_bytes([px.0, px.1, px.2, 0xFF]);
                *counts.entry(rgb).or_insert(0) += 1;
            }
        }
        if counts.len() <= MAX_PALETTE_COLORS {
            return true;
        }
        let max_cnt = counts.values().copied().max().unwrap_or(0);
        max_cnt * 100 >= n_px * TEXT_SAFE_DOMINANT_BG_PCT
    }

    #[test]
    fn test_is_text_safe_thresholds_and_misra_gries_adversarial() {
        let mut scratch = PaletteScratch::new();
        let run = |scratch: &mut PaletteScratch, pixels: &[Vec4u8], w: u32, h: u32| -> bool {
            let ptr = pixels.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
            scratch
                .classify_text_safe(bp, w as usize, TileRect { x: 0, y: 0, w, h })
                .is_text_safe()
        };
        let unique_color = |i: usize| -> Vec4u8 {
            Vec4u8(
                (i & 0xFF) as u8,
                ((i >> 8) & 0xFF) as u8,
                ((i >> 16) & 0xFF) as u8,
                0xFF,
            )
        };

        // 1. Exactly 256 unique RGB colors (even with > 256 unique BGRA via varying alpha) -> true.
        let mut px256: Vec<Vec4u8> = (0..1000)
            .map(|i| {
                let mut c = unique_color(i % 256);
                c.3 = (i % 251) as u8;
                c
            })
            .collect();
        assert!(run(&mut scratch, &px256, 40, 25));

        // 2. 257 unique RGB colors uniformly distributed (~0.4% each) -> false.
        px256[999] = unique_color(256);
        assert!(!run(&mut scratch, &px256, 40, 25));

        // 3. Exact 35% boundary (`n_px = 1000`: 349 -> false, 350 -> true;
        //    `n_px = 1001`: 350 -> false, 351 -> true).
        let dom = Vec4u8(250, 251, 252, 0xFF);
        for (w, h, below, at_thresh) in [(40u32, 25u32, 349usize, 350usize), (77, 13, 350, 351)] {
            let n = (w * h) as usize;
            // Dominant color placed at the end (worst case for naive front-sampling).
            let mut tail_below: Vec<Vec4u8> = (0..n).map(|i| unique_color(i + 1)).collect();
            for px in &mut tail_below[n - below..] {
                *px = dom;
            }
            assert!(!run(&mut scratch, &tail_below, w, h));

            let mut tail_at = tail_below;
            tail_at[n - at_thresh] = dom;
            assert!(run(&mut scratch, &tail_at, w, h));

            // Dominant color at the beginning.
            let mut head_at: Vec<Vec4u8> = (0..n).map(|i| unique_color(i + 1)).collect();
            for px in &mut head_at[..at_thresh] {
                *px = dom;
            }
            assert!(run(&mut scratch, &head_at, w, h));
        }

        // 4. Misra-Gries adversarial cancellation orderings: interleave `[u_a, u_b, dom]`
        //    repeatedly so both slots constantly cancel out. `dom` still survives!
        for (dom_cnt, expected) in [(349usize, false), (350, true)] {
            let n = 1000usize;
            let mut adv = Vec::with_capacity(n);
            let mut next_u = 1usize;
            let mut dom_left = dom_cnt;
            while adv.len() < n {
                if (dom_left > 0 && adv.len() % 3 == 2) || n - adv.len() <= dom_left {
                    adv.push(dom);
                    dom_left -= 1;
                } else {
                    adv.push(unique_color(next_u));
                    next_u += 1;
                }
            }
            assert_eq!(run(&mut scratch, &adv, 40, 25), expected);
            assert_eq!(
                expected,
                oracle_is_text_safe(
                    &adv,
                    40,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 40,
                        h: 25
                    }
                )
            );
        }

        // 5. Two competing ~35% colors (`350` of A, `350` of B, `300` unique) -> true;
        //    `349` of A, `349` of B, `302` unique -> false.
        for (cnt, expected) in [(349usize, false), (350, true)] {
            let mut dual = Vec::with_capacity(1000);
            let a = Vec4u8(1, 2, 3, 0xFF);
            let b = Vec4u8(4, 5, 6, 0xFF);
            for i in 0..cnt {
                dual.push(a);
                dual.push(b);
                if i < 1000 - 2 * cnt {
                    dual.push(unique_color(1000 + i));
                }
            }
            while dual.len() < 1000 {
                dual.push(unique_color(5000 + dual.len()));
            }
            assert_eq!(run(&mut scratch, &dual, 40, 25), expected);
        }

        // 6. Padded stride and non-zero `(x0, y0)`: outer solid background must not leak into rect.
        let stride_px = 64usize;
        let mut padded = vec![dom; stride_px * 50];
        let rect = TileRect {
            x: 7,
            y: 5,
            w: 30,
            h: 20,
        };
        for ry in 0..rect.h as usize {
            for rx in 0..rect.w as usize {
                padded[(rect.y as usize + ry) * stride_px + rect.x as usize + rx] =
                    unique_color(ry * 30 + rx + 1);
            }
        }
        let ptr = padded.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, padded.len()) };
        assert_eq!(
            scratch.classify_text_safe(bp, stride_px, rect),
            TextSafeClass::Photo
        );

        // 7. Oracle cross-check across diverse pseudo-random and bimodal tiles.
        let mut rng = 0x5EED_CAFE_1234_5678u64;
        for case in 0..40usize {
            let w = 24 + (case % 17);
            let h = 16 + ((case * 3) % 13);
            let n = w * h;
            let mut buf = Vec::with_capacity(n);
            let bg_pct = (case * 7) % 60;
            for i in 0..n {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (rng >> 32) as usize;
                let px = if (v % 100) < bg_pct {
                    dom
                } else if case % 5 == 0 {
                    unique_color(v % 250)
                } else {
                    unique_color(1000 + (v ^ i) % 10_000)
                };
                buf.push(px);
            }
            let r = TileRect {
                x: 0,
                y: 0,
                w: w as u32,
                h: h as u32,
            };
            assert_eq!(
                run(&mut scratch, &buf, w as u32, h as u32),
                oracle_is_text_safe(&buf, w, r),
                "oracle mismatch on case {case}"
            );
        }
    }
}
