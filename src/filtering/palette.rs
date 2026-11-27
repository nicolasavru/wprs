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
use crate::serialization::framing::SUBTILE_H;
use crate::serialization::framing::SUBTILE_W;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::packed_palette_indices_len;
use crate::vec4u8::Vec4u8;

const MAX_PALETTE_COLORS: usize = 256;
const HASH_SLOTS: usize = 512;
const HASH_MASK: usize = HASH_SLOTS - 1;

/// Maximum pixel count of one `128x128` split quadrant (`TILE_PIXELS / 4`).
pub(crate) const SUBTILE_PIXELS: usize = SUBTILE_W * SUBTILE_H;
const _: () = assert!(SUBTILE_PIXELS * 4 == TILE_PIXELS);

/// Dominant-color pixel share threshold (%) for hybrid text-safe classification.
/// Catches ClearType / LCD subpixel-antialiased text tiles that exceed 256
/// unique RGB colors on a solid background.
const TEXT_SAFE_DOMINANT_BG_PCT: usize = 35;

/// Loads one 4-byte `Vec4u8` pixel at `row_ptr.add(x)` as a little-endian
/// `u32` (`[b, g, r, a]`) in a single unaligned 32-bit load, preventing LLVM
/// from splitting masked `Xrgb8888` reads into byte + halfword loads.
///
/// # Safety
/// `row_ptr.add(x)` must be valid for a 4-byte read.
#[inline(always)]
unsafe fn load_pixel_le(row_ptr: *const Vec4u8, x: usize) -> u32 {
    // SAFETY: `Vec4u8` is `#[repr(C, packed)]` with size 4 and no padding;
    // caller guarantees `row_ptr.add(x)` is valid for a 4-byte read.
    u32::from_le(unsafe { row_ptr.add(x).cast::<u32>().read_unaligned() })
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum MgChunk24 {
    /// All 8 disjoint triples have 3 pairwise-distinct RGB colors.
    AllDistinctTriples,
    /// `c0 != 0` and all 24 RGB pixels equal `c0`.
    AllMatchC0,
    /// Non-zero 8-bit mask: bit `i` is set iff triple `(x+i, x+8+i, x+16+i)`
    /// contains a duplicate RGB color and must be pushed into Misra-Gries.
    Mixed(u32),
}

/// Classifies 24 contiguous pixels at `row_ptr[x..x + 24]` as 8 disjoint
/// triples `(p[i], p[8+i], p[16+i])` for 2-slot Misra-Gries cancellation.
///
/// # Safety
/// `row_ptr.add(x)` must be valid for 24 `Vec4u8` reads.
#[inline(always)]
unsafe fn classify_mg_chunk_24(row_ptr: *const Vec4u8, x: usize, c0: u32) -> MgChunk24 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(miri)))]
    {
        use std::arch::x86_64::__m256i;
        use std::arch::x86_64::_mm256_castsi256_ps;
        use std::arch::x86_64::_mm256_cmpeq_epi32;
        use std::arch::x86_64::_mm256_loadu_si256;
        use std::arch::x86_64::_mm256_movemask_ps;
        use std::arch::x86_64::_mm256_or_si256;
        use std::arch::x86_64::_mm256_set1_epi32;
        use std::arch::x86_64::_mm256_testz_si256;
        use std::arch::x86_64::_mm256_xor_si256;
        // SAFETY: Caller guarantees 24 `Vec4u8` (96 bytes) readable at `row_ptr.add(x)`,
        // and `target_feature = "avx2"` is enabled at compile time.
        unsafe {
            let p = row_ptr.add(x).cast::<__m256i>();
            let mask = _mm256_set1_epi32(0xFF00_0000u32.cast_signed());
            let va = _mm256_or_si256(_mm256_loadu_si256(p), mask);
            let vb = _mm256_or_si256(_mm256_loadu_si256(p.add(1)), mask);
            let vc = _mm256_or_si256(_mm256_loadu_si256(p.add(2)), mask);
            let eq = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi32(va, vb), _mm256_cmpeq_epi32(vb, vc)),
                _mm256_cmpeq_epi32(va, vc),
            );
            let eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq)).cast_unsigned();
            if eq_mask == 0 {
                return MgChunk24::AllDistinctTriples;
            }
            if c0 != 0 {
                let vc0 = _mm256_set1_epi32(c0.cast_signed());
                let diff = _mm256_or_si256(
                    _mm256_or_si256(_mm256_xor_si256(va, vc0), _mm256_xor_si256(vb, vc0)),
                    _mm256_xor_si256(vc, vc0),
                );
                if _mm256_testz_si256(diff, diff) != 0 {
                    return MgChunk24::AllMatchC0;
                }
            }
            MgChunk24::Mixed(eq_mask)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", not(miri))))]
    {
        let mut va = [0u32; 8];
        let mut vb = [0u32; 8];
        let mut vc = [0u32; 8];
        for i in 0..8 {
            // SAFETY: Caller guarantees 24 `Vec4u8` readable at `row_ptr.add(x)`.
            va[i] = unsafe { load_pixel_le(row_ptr, x + i) } | 0xFF00_0000;
            // SAFETY: Caller guarantees 24 `Vec4u8` readable at `row_ptr.add(x)`.
            vb[i] = unsafe { load_pixel_le(row_ptr, x + 8 + i) } | 0xFF00_0000;
            // SAFETY: Caller guarantees 24 `Vec4u8` readable at `row_ptr.add(x)`.
            vc[i] = unsafe { load_pixel_le(row_ptr, x + 16 + i) } | 0xFF00_0000;
        }
        let mut eq_mask = 0u32;
        for i in 0..8 {
            if va[i] == vb[i] || vb[i] == vc[i] || va[i] == vc[i] {
                eq_mask |= 1u32 << i;
            }
        }
        if eq_mask == 0 {
            return MgChunk24::AllDistinctTriples;
        }
        if c0 != 0 {
            let mut diff = 0u32;
            for ((&a, &b), &c) in va.iter().zip(&vb).zip(&vc) {
                diff |= (a ^ c0) | (b ^ c0) | (c ^ c0);
            }
            if diff == 0 {
                return MgChunk24::AllMatchC0;
            }
        }
        MgChunk24::Mixed(eq_mask)
    }
}

/// Feeds one RGB color `c` (`c & 0xFF00_0000 == 0xFF00_0000`, hence `c != 0`)
/// into 2-slot Misra-Gries state maintaining `n0 == 0 <=> c0 == 0`,
/// `n1 == 0 <=> c1 == 0`, and `n0 > 0 && n1 > 0 => c0 != c1`.
#[inline(always)]
fn mg_push(c: u32, c0: &mut u32, n0: &mut usize, c1: &mut u32, n1: &mut usize) {
    if c == *c0 {
        *n0 += 1;
    } else if c == *c1 {
        *n1 += 1;
    } else if *n0 == 0 {
        *c0 = c;
        *n0 = 1;
    } else if *n1 == 0 {
        *c1 = c;
        *n1 = 1;
    } else {
        *n0 -= 1;
        if *n0 == 0 {
            *c0 = 0;
        }
        *n1 -= 1;
        if *n1 == 0 {
            *c1 = 0;
        }
    }
}

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

struct ColorTable {
    keys: [u32; HASH_SLOTS],
    vals: [u8; HASH_SLOTS],
    gens: [u32; HASH_SLOTS],
    counts: [u32; MAX_PALETTE_COLORS],
    cur_gen: u32,
}

impl ColorTable {
    fn new() -> Self {
        Self {
            keys: [0; HASH_SLOTS],
            vals: [0; HASH_SLOTS],
            gens: [0; HASH_SLOTS],
            counts: [0; MAX_PALETTE_COLORS],
            cur_gen: 0,
        }
    }

    #[inline(always)]
    fn next_epoch(&mut self) -> u32 {
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            self.gens.fill(0);
            self.cur_gen = 1;
        }
        self.cur_gen
    }

    #[inline]
    fn prefix_count(&self, cur_epoch: u32, color: u32) -> usize {
        let mut h = (color.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
        loop {
            if self.gens[h] != cur_epoch {
                return 0;
            }
            if self.keys[h] == color {
                return self.counts[usize::from(self.vals[h])] as usize;
            }
            h = (h + 1) & HASH_MASK;
        }
    }

    /// Completes hybrid `DominantBg` vs `Photo` classification once 256
    /// distinct RGB colors and their exact prefix counts (`self.counts[..256]`)
    /// have been accumulated up to `(ovf_ry, ovf_x)` (exclusive), and the
    /// 257th unique RGB color sits at `(ovf_ry, ovf_x)`.
    ///
    /// Bulk-seeds 2-slot Misra-Gries from the exact prefix counts in O(256),
    /// scans only the remaining suffix `(ovf_ry, ovf_x)..end` using algebraic
    /// triple cancellation (`n0 + n1 + 3 * decrements == n_px`), and runs an
    /// exact count verification over the suffix only in the narrow uncertain
    /// band.
    #[inline(never)]
    fn finish_misra_gries_from_counts(
        &self,
        cur_epoch: u32,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
        ovf_ry: usize,
        ovf_x: usize,
        rgb_palette: &[u32; MAX_PALETTE_COLORS],
        threshold: usize,
    ) -> HybridScan {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;

        let mut c0 = 0u32;
        let mut n0 = 0usize;
        let mut c1 = 0u32;
        let mut n1 = 0usize;

        for (&c, &cnt) in rgb_palette.iter().zip(self.counts.iter()) {
            let cnt = cnt as usize;
            if cnt >= threshold {
                return HybridScan::TextSafeNoPalette;
            }
            if n0 == 0 {
                c0 = c;
                n0 = cnt;
            } else if n1 == 0 {
                c1 = c;
                n1 = cnt;
            } else {
                let d = cnt.min(n0).min(n1);
                n0 -= d;
                n1 -= d;
                if n0 == 0 {
                    c0 = 0;
                }
                if n1 == 0 {
                    c1 = 0;
                }
                let rem = cnt - d;
                if rem > 0 {
                    if n0 == 0 {
                        c0 = c;
                        n0 = rem;
                    } else {
                        c1 = c;
                        n1 = rem;
                    }
                }
            }
        }
        if n1 > n0 {
            std::mem::swap(&mut c0, &mut c1);
            std::mem::swap(&mut n0, &mut n1);
        }

        for ry in ovf_ry..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let mut x = if ry == ovf_ry { ovf_x } else { 0 };

            // Process 24 pixels at a time as 8 disjoint triples
            // `(p[x+i], p[x+8+i], p[x+16+i])`: any triple of 3 pairwise-distinct
            // colors cancels out in 2-slot Misra-Gries without altering
            // `(c0, n0, c1, n1)`, and all-`c0` background spans add 24 in O(1).
            while x + 24 <= tw {
                // SAFETY: `x + 24 <= tw` and `row_ptr` is valid for `tw` reads.
                match unsafe { classify_mg_chunk_24(row_ptr, x, c0) } {
                    MgChunk24::AllDistinctTriples => {},
                    MgChunk24::AllMatchC0 => {
                        n0 += 24;
                    },
                    MgChunk24::Mixed(mut mask) => {
                        while mask != 0 {
                            let i = mask.trailing_zeros() as usize;
                            mask &= mask - 1;
                            // SAFETY: `i < 8`, `x + 24 <= tw`, and `row_ptr` is valid for `tw` reads.
                            let a = unsafe { load_pixel_le(row_ptr, x + i) } | 0xFF00_0000;
                            // SAFETY: `i < 8`, `x + 24 <= tw`, and `row_ptr` is valid for `tw` reads.
                            let b = unsafe { load_pixel_le(row_ptr, x + 8 + i) } | 0xFF00_0000;
                            // SAFETY: `i < 8`, `x + 24 <= tw`, and `row_ptr` is valid for `tw` reads.
                            let c = unsafe { load_pixel_le(row_ptr, x + 16 + i) } | 0xFF00_0000;
                            mg_push(a, &mut c0, &mut n0, &mut c1, &mut n1);
                            mg_push(b, &mut c0, &mut n0, &mut c1, &mut n1);
                            mg_push(c, &mut c0, &mut n0, &mut c1, &mut n1);
                        }
                    },
                }
                x += 24;
            }
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let c = unsafe { load_pixel_le(row_ptr, x) } | 0xFF00_0000;
                mg_push(c, &mut c0, &mut n0, &mut c1, &mut n1);
                x += 1;
            }
            if n1 > n0 {
                std::mem::swap(&mut c0, &mut c1);
                std::mem::swap(&mut n0, &mut n1);
            }
            if n0 >= threshold {
                return HybridScan::TextSafeNoPalette;
            }
        }

        debug_assert!(n0 + n1 <= n_px && (n_px - n0 - n1).is_multiple_of(3));
        let decrements = (n_px - n0 - n1) / 3;
        let check0 = n0 > 0 && n0 + decrements >= threshold;
        let check1 = n1 > 0 && n1 + decrements >= threshold;
        if !check0 && !check1 {
            return HybridScan::Photo;
        }

        let mut exact0 = if check0 {
            self.prefix_count(cur_epoch, c0)
        } else {
            0
        };
        let mut exact1 = if check1 {
            self.prefix_count(cur_epoch, c1)
        } else {
            0
        };
        for ry in ovf_ry..th {
            let row_start = (y0 + ry) * stride_px + x0;
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let x_start = if ry == ovf_ry { ovf_x } else { 0 };
            for x in x_start..tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let c = unsafe { load_pixel_le(row_ptr, x) } | 0xFF00_0000;
                if check0 && c == c0 {
                    exact0 += 1;
                } else if check1 && c == c1 {
                    exact1 += 1;
                }
            }
        }

        if exact0 >= threshold || exact1 >= threshold {
            HybridScan::TextSafeNoPalette
        } else {
            HybridScan::Photo
        }
    }

    /// Translucent `Argb8888` overflow continuation: collapses the 256 distinct
    /// BGRA prefix entries in `(palette_out, self.counts)` into RGB in-place,
    /// continues RGB unique-color tracking from `(ovf_ry, ovf_x)` without
    /// re-scanning the prefix, and hands off to [`Self::finish_misra_gries_from_counts`]
    /// if a 257th unique RGB color appears.
    #[inline(never)]
    fn finish_translucent_rgb(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
        ovf_ry: usize,
        ovf_x: usize,
        palette_out: &mut [u32; MAX_PALETTE_COLORS],
        threshold: usize,
    ) -> HybridScan {
        let cur_epoch = self.next_epoch();
        let mut rgb_count = 0usize;
        for i in 0..MAX_PALETTE_COLORS {
            let rgb = palette_out[i] | 0xFF00_0000;
            let cnt = self.counts[i];
            let mut h = (rgb.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
            loop {
                if self.gens[h] != cur_epoch {
                    self.gens[h] = cur_epoch;
                    self.keys[h] = rgb;
                    let slot = rgb_count & (MAX_PALETTE_COLORS - 1);
                    debug_assert_eq!(slot, rgb_count);
                    self.vals[h] = slot as u8;
                    palette_out[slot] = rgb;
                    self.counts[slot] = cnt;
                    if cnt as usize >= threshold {
                        return HybridScan::TextSafeNoPalette;
                    }
                    rgb_count += 1;
                    break;
                }
                if self.keys[h] == rgb {
                    let idx = usize::from(self.vals[h]);
                    let next = self.counts[idx] + cnt;
                    self.counts[idx] = next;
                    if next as usize >= threshold {
                        return HybridScan::TextSafeNoPalette;
                    }
                    break;
                }
                h = (h + 1) & HASH_MASK;
            }
        }

        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let mut last_color = palette_out[0];
        let mut last_idx = 0u8;
        let mut last_run = 0u32;

        for ry in ovf_ry..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let mut x = if ry == ovf_ry { ovf_x } else { 0 };
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let c = unsafe { load_pixel_le(row_ptr, x) } | 0xFF00_0000;
                if c == last_color {
                    last_run += 1;
                    x += 1;
                    while x + 8 <= tw {
                        let mut diff = 0u32;
                        for i in 0..8 {
                            // SAFETY: `x + 8 <= tw` and `row_ptr` is valid for `tw` reads.
                            diff |= (unsafe { load_pixel_le(row_ptr, x + i) } | 0xFF00_0000)
                                ^ last_color;
                        }
                        if diff != 0 {
                            break;
                        }
                        last_run += 8;
                        x += 8;
                    }
                    continue;
                }
                let next_cnt = self.counts[usize::from(last_idx)] + last_run;
                self.counts[usize::from(last_idx)] = next_cnt;
                if next_cnt as usize >= threshold {
                    return HybridScan::TextSafeNoPalette;
                }
                last_run = 1;

                let mut h = (c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
                loop {
                    if self.gens[h] != cur_epoch {
                        if rgb_count >= MAX_PALETTE_COLORS {
                            return self.finish_misra_gries_from_counts(
                                cur_epoch,
                                aos,
                                stride_px,
                                rect,
                                ry,
                                x,
                                palette_out,
                                threshold,
                            );
                        }
                        self.gens[h] = cur_epoch;
                        self.keys[h] = c;
                        let idx = rgb_count as u8;
                        self.vals[h] = idx;
                        palette_out[rgb_count] = c;
                        self.counts[rgb_count] = 0;
                        rgb_count += 1;
                        last_color = c;
                        last_idx = idx;
                        break;
                    }
                    if self.keys[h] == c {
                        last_color = c;
                        last_idx = self.vals[h];
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
                x += 1;
            }
            if (self.counts[usize::from(last_idx)] + last_run) as usize >= threshold {
                return HybridScan::TextSafeNoPalette;
            }
        }

        // Finished entire rect with `<= 256` unique RGB colors (though `> 256` BGRA).
        HybridScan::TextSafeNoPalette
    }

    #[inline]
    fn scan_hybrid<const TRACK_ALPHA: bool>(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
        palette_out: &mut [u32; MAX_PALETTE_COLORS],
        indices_out: &mut [u8],
    ) -> HybridScan {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;
        debug_assert!(n_px > 0 && n_px <= indices_out.len());

        let cur_epoch = self.next_epoch();
        let alpha_or = if TRACK_ALPHA { 0u32 } else { 0xFF00_0000u32 };

        // Seed with pixel `(0, 0)` so the hot inner loop skips `count > 0`.
        debug_assert!(y0 * stride_px + x0 + tw <= aos.len());
        // SAFETY: Caller guarantees `rect` is non-empty and inside `aos`.
        let first_c = unsafe { load_pixel_le(aos.ptr().add(y0 * stride_px + x0), 0) } | alpha_or;
        let h0 = (first_c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
        self.gens[h0] = cur_epoch;
        self.keys[h0] = first_c;
        self.vals[h0] = 0;
        palette_out[0] = first_c;
        self.counts[0] = 0;

        let mut count = 1usize;
        let mut last_color = first_c;
        let mut last_idx = 0u8;
        let mut last_run = 0u32;

        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let row_out = &mut indices_out[ry * tw..(ry + 1) * tw];
            let mut x = 0usize;
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
                if c == last_color {
                    row_out[x] = last_idx;
                    last_run += 1;
                    x += 1;
                    while let Some(dst8) = row_out.get_mut(x..x + 8) {
                        let mut diff = 0u32;
                        for i in 0..8 {
                            // SAFETY: `x + 8 <= tw` and `row_ptr` is valid for `tw` reads.
                            diff |=
                                (unsafe { load_pixel_le(row_ptr, x + i) } | alpha_or) ^ last_color;
                        }
                        if diff != 0 {
                            break;
                        }
                        dst8.fill(last_idx);
                        last_run += 8;
                        x += 8;
                    }
                    continue;
                }
                self.counts[usize::from(last_idx)] += last_run;
                last_run = 1;

                let mut h = (c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
                loop {
                    if self.gens[h] != cur_epoch {
                        if count >= MAX_PALETTE_COLORS {
                            let threshold = (n_px * TEXT_SAFE_DOMINANT_BG_PCT).div_ceil(100);
                            return if TRACK_ALPHA {
                                self.finish_translucent_rgb(
                                    aos,
                                    stride_px,
                                    rect,
                                    ry,
                                    x,
                                    palette_out,
                                    threshold,
                                )
                            } else {
                                self.finish_misra_gries_from_counts(
                                    cur_epoch,
                                    aos,
                                    stride_px,
                                    rect,
                                    ry,
                                    x,
                                    palette_out,
                                    threshold,
                                )
                            };
                        }
                        self.gens[h] = cur_epoch;
                        self.keys[h] = c;
                        let idx = count as u8;
                        self.vals[h] = idx;
                        palette_out[count] = c;
                        self.counts[count] = 0;
                        count += 1;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    if self.keys[h] == c {
                        let idx = self.vals[h];
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
                x += 1;
            }
        }

        if count == 1 {
            HybridScan::Solid(palette_out[0].to_le_bytes())
        } else {
            HybridScan::Palette { n_colors: count }
        }
    }
}

pub(crate) struct PaletteScratch {
    table: ColorTable,
    pub(crate) palette: [u32; MAX_PALETTE_COLORS],
    quad_palettes: Option<Box<[[u32; MAX_PALETTE_COLORS]; 4]>>,
    indices: Vec<u8>,
    pub(crate) packed: Vec<u8>,
}

/// Result of single-pass hybrid classification + palette construction for a
/// leaf tile or split quadrant.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum HybridScan {
    /// Every pixel in the rect equals this canonical `[b, g, r, a]` pixel.
    Solid([u8; 4]),
    /// Rect fits in `2..=256` unique colors; palette and raster-order indices
    /// are already populated in [`PaletteScratch`] (ready for packing + zstd).
    Palette { n_colors: usize },
    /// Rect exceeds 256 palette colors (cannot use `Palette`), but is `TextSafe`
    /// (`<= 256` unique RGB colors on translucent `Argb8888`, or a single RGB
    /// color covering `>= 35%` of pixels); encodes via `AosZstd`.
    TextSafeNoPalette,
    /// Neither `<= 256` unique RGB colors nor `>= 35%` dominant background;
    /// eligible for JPEG encoding.
    Photo,
}

impl HybridScan {
    #[inline]
    pub(crate) fn is_text_safe(self) -> bool {
        !matches!(self, Self::Photo)
    }
}

impl PaletteScratch {
    pub(crate) fn new() -> Self {
        Self {
            table: ColorTable::new(),
            palette: [0; MAX_PALETTE_COLORS],
            quad_palettes: None,
            indices: vec![0; TILE_PIXELS],
            packed: vec![0; TILE_PIXELS],
        }
    }

    /// Scans `rect` in `aos` (stride `stride_px` pixels). Returns
    /// `Ok((n_colors, packed_len))` if the tile contains at most 256 unique
    /// colors, populating `self.palette[..n_colors]` and
    /// `self.packed[..packed_len]`. Returns `Err(overflow_ry)` immediately upon
    /// seeing a 257th unique color on row `overflow_ry`.
    ///
    /// Reads directly from `BufferPointer` without creating a shared `&[u8]`
    /// slice over client shm memory.
    #[inline]
    pub(crate) fn try_encode_2d<const HAS_ALPHA: bool>(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> std::result::Result<(usize, usize), usize> {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;
        debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

        let cur_epoch = self.table.next_epoch();
        let alpha_or = if HAS_ALPHA { 0u32 } else { 0xFF00_0000u32 };

        debug_assert!(y0 * stride_px + x0 + tw <= aos.len());
        // SAFETY: Caller guarantees `rect` is non-empty and inside `aos`.
        let first_c = unsafe { load_pixel_le(aos.ptr().add(y0 * stride_px + x0), 0) } | alpha_or;
        let h0 = (first_c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
        self.table.gens[h0] = cur_epoch;
        self.table.keys[h0] = first_c;
        self.table.vals[h0] = 0;
        self.palette[0] = first_c;

        let mut count = 1usize;
        let mut last_color = first_c;
        let mut last_idx = 0u8;

        for ry in 0..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos` (`(y0 + ry) * stride_px + x0 + tw <= aos.len()`).
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            let row_out = &mut self.indices[ry * tw..(ry + 1) * tw];
            let mut x = 0usize;
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` `Vec4u8` reads.
                let c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
                if c == last_color {
                    row_out[x] = last_idx;
                    x += 1;
                    while let Some(dst8) = row_out.get_mut(x..x + 8) {
                        let mut diff = 0u32;
                        for i in 0..8 {
                            // SAFETY: `x + 8 <= tw` and `row_ptr` is valid for `tw` reads.
                            diff |=
                                (unsafe { load_pixel_le(row_ptr, x + i) } | alpha_or) ^ last_color;
                        }
                        if diff != 0 {
                            break;
                        }
                        dst8.fill(last_idx);
                        x += 8;
                    }
                    continue;
                }
                let mut h = (c.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK;
                loop {
                    if self.table.gens[h] != cur_epoch {
                        if count >= MAX_PALETTE_COLORS {
                            return Err(ry);
                        }
                        self.table.gens[h] = cur_epoch;
                        self.table.keys[h] = c;
                        let idx = count as u8;
                        self.table.vals[h] = idx;
                        self.palette[count] = c;
                        count += 1;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    if self.table.keys[h] == c {
                        let idx = self.table.vals[h];
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
                x += 1;
            }
        }

        if count == 1 {
            return Ok((1, 0));
        }
        let packed_len = pack_indices(&self.indices[..n_px], count, &mut self.packed);
        Ok((count, packed_len))
    }

    /// Single-pass hybrid classification + palette construction for a
    /// non-splittable leaf (`rect.w * rect.h <= TILE_PIXELS`). On
    /// [`HybridScan::Palette`], populates `self.palette[..n_colors]` and
    /// `self.indices[..n_px]`.
    #[inline]
    pub(crate) fn scan_hybrid_leaf<const TRACK_ALPHA: bool>(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> HybridScan {
        let n_px = rect.w as usize * rect.h as usize;
        debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);
        self.table.scan_hybrid::<TRACK_ALPHA>(
            aos,
            stride_px,
            rect,
            &mut self.palette,
            &mut self.indices[..n_px],
        )
    }

    /// Single-pass hybrid classification + palette construction for split
    /// quadrant `q` (`0..4`, `rect.w * rect.h <= SUBTILE_PIXELS`). On
    /// [`HybridScan::Palette`], stores quadrant `q`'s palette in
    /// `quad_palette(q)[..n_colors]` and indices in
    /// `self.indices[q * SUBTILE_PIXELS .. q * SUBTILE_PIXELS + n_px]` so mixed
    /// `Split` tiles encode palette quadrants with zero second pixel scan.
    #[inline]
    pub(crate) fn scan_hybrid_quadrant<const TRACK_ALPHA: bool>(
        &mut self,
        q: usize,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> HybridScan {
        let n_px = rect.w as usize * rect.h as usize;
        debug_assert!(q < 4 && n_px > 0 && n_px <= SUBTILE_PIXELS);
        let quad_pals = self
            .quad_palettes
            .get_or_insert_with(|| Box::new([[0u32; MAX_PALETTE_COLORS]; 4]));
        let start = q * SUBTILE_PIXELS;
        self.table.scan_hybrid::<TRACK_ALPHA>(
            aos,
            stride_px,
            rect,
            &mut quad_pals[q],
            &mut self.indices[start..start + n_px],
        )
    }

    #[inline]
    pub(crate) fn quad_palette(&self, q: usize) -> &[u32; MAX_PALETTE_COLORS] {
        &self
            .quad_palettes
            .as_ref()
            .expect("quad_palettes initialized by scan_hybrid_quadrant")[q]
    }

    /// Packs `self.indices[start..start + n_px]` into `self.packed` for a
    /// `n_colors`-entry palette (`2..=256`), returning the packed byte length.
    #[inline]
    pub(crate) fn pack_indices_range(
        &mut self,
        start: usize,
        n_px: usize,
        n_colors: usize,
    ) -> usize {
        pack_indices(
            &self.indices[start..start + n_px],
            n_colors,
            &mut self.packed,
        )
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

        // 257 unique colors must return Err(overflow_ry) (early bailout).
        let pixels: Vec<Vec4u8> = (0..257u32)
            .map(|i| Vec4u8((i & 0xFF) as u8, (i >> 8) as u8, 0, 0xFF))
            .collect();
        let ptr = pixels.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
        let mut scratch = PaletteScratch::new();
        assert_eq!(
            scratch.try_encode_2d::<false>(
                bp,
                257,
                TileRect {
                    x: 0,
                    y: 0,
                    w: 257,
                    h: 1,
                },
            ),
            Err(0)
        );
        // Multi-row overflow reports the exact row index where color #257 appeared.
        let mut multi = vec![Vec4u8(0, 10, 20, 0xFF); 16 * 20];
        for (i, px) in multi[..256].iter_mut().enumerate() {
            *px = Vec4u8((i & 0xFF) as u8, 10, 20, 0xFF);
        }
        multi[18 * 16 + 5] = Vec4u8(99, 88, 77, 0xFF);
        let mptr = multi.as_ptr();
        let mbp = unsafe { BufferPointer::new(&mptr, multi.len()) };
        assert_eq!(
            scratch.try_encode_2d::<false>(
                mbp,
                16,
                TileRect {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 20,
                },
            ),
            Err(18)
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

    fn assert_scan_matches_oracle(
        scratch: &mut PaletteScratch,
        frame: &[Vec4u8],
        stride_px: usize,
        rect: TileRect,
        track_alpha: bool,
    ) {
        let ptr = frame.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, frame.len()) };
        let exp_text_safe = oracle_is_text_safe(frame, stride_px, rect);
        let mut ref_scratch = PaletteScratch::new();
        let ref_pal = if track_alpha {
            ref_scratch.try_encode_2d::<true>(bp, stride_px, rect)
        } else {
            ref_scratch.try_encode_2d::<false>(bp, stride_px, rect)
        };

        let n_px = rect.w as usize * rect.h as usize;
        let scan = if track_alpha {
            scratch.scan_hybrid_leaf::<true>(bp, stride_px, rect)
        } else {
            scratch.scan_hybrid_leaf::<false>(bp, stride_px, rect)
        };
        assert_eq!(scan.is_text_safe(), exp_text_safe);
        match (scan, ref_pal) {
            (HybridScan::Solid(px), Ok((1, 0))) => {
                assert_eq!(px, ref_scratch.palette[0].to_le_bytes());
            },
            (HybridScan::Palette { n_colors }, Ok((ref_n, ref_plen))) => {
                assert_eq!(n_colors, ref_n);
                assert_eq!(
                    scratch.palette[..n_colors],
                    ref_scratch.palette[..ref_n],
                    "palette table mismatch"
                );
                let plen = scratch.pack_indices_range(0, n_px, n_colors);
                assert_eq!(plen, ref_plen);
                assert_eq!(
                    scratch.packed[..plen],
                    ref_scratch.packed[..ref_plen],
                    "packed indices mismatch"
                );
            },
            (HybridScan::TextSafeNoPalette, Err(_)) => assert!(exp_text_safe),
            (HybridScan::Photo, Err(_)) => assert!(!exp_text_safe),
            (got, expected) => panic!(
                "scan_hybrid_leaf mismatch: got={got:?}, ref_pal={expected:?}, oracle_text_safe={exp_text_safe}"
            ),
        }

        if n_px <= SUBTILE_PIXELS {
            for q in 0..4 {
                let qscan = if track_alpha {
                    scratch.scan_hybrid_quadrant::<true>(q, bp, stride_px, rect)
                } else {
                    scratch.scan_hybrid_quadrant::<false>(q, bp, stride_px, rect)
                };
                assert_eq!(qscan, scan, "quadrant {q} scan differs from leaf scan");
                if let HybridScan::Palette { n_colors } = qscan {
                    assert_eq!(
                        scratch.quad_palette(q)[..n_colors],
                        ref_scratch.palette[..n_colors]
                    );
                    let plen = scratch.pack_indices_range(q * SUBTILE_PIXELS, n_px, n_colors);
                    assert_eq!(scratch.packed[..plen], ref_scratch.packed[..plen]);
                }
            }
        }
    }

    #[test]
    fn test_is_text_safe_thresholds_and_misra_gries_adversarial() {
        let mut scratch = PaletteScratch::new();
        let run = |scratch: &mut PaletteScratch, pixels: &[Vec4u8], w: u32, h: u32| -> bool {
            let r = TileRect { x: 0, y: 0, w, h };
            assert_scan_matches_oracle(scratch, pixels, w as usize, r, false);
            assert_scan_matches_oracle(scratch, pixels, w as usize, r, true);
            let ptr = pixels.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, pixels.len()) };
            scratch
                .scan_hybrid_leaf::<false>(bp, w as usize, r)
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
        //    `n_px = 1001`: 350 -> false, 351 -> true; also width < 24 and >= 24).
        let dom = Vec4u8(250, 251, 252, 0xFF);
        for (w, h, below, at_thresh) in [
            (40u32, 25u32, 349usize, 350usize),
            (77, 13, 350, 351),
            (20, 50, 349, 350),
            (128, 8, 358, 359),
        ] {
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

            // Dominant RGB color split across 8 distinct alpha values (translucent
            // Argb8888: no single BGRA color reaches 35%, but RGB reaches 35%!).
            let mut trans_dom_at = tail_at.clone();
            for (k, px) in trans_dom_at.iter_mut().enumerate() {
                px.3 = 0x80 + (k & 7) as u8;
            }
            assert!(run(&mut scratch, &trans_dom_at, w, h));
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
        assert_scan_matches_oracle(&mut scratch, &padded, stride_px, rect, false);
        assert_scan_matches_oracle(&mut scratch, &padded, stride_px, rect, true);
        let ptr = padded.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, padded.len()) };
        assert_eq!(
            scratch.scan_hybrid_leaf::<false>(bp, stride_px, rect),
            HybridScan::Photo
        );

        // 7. Solid, small palette (2/4/16/256 colors), and garbage X bytes on Xrgb8888.
        for n_col in [1usize, 2, 4, 16, 256] {
            let mut buf = Vec::with_capacity(32 * 24);
            for i in 0..(32 * 24) {
                let mut c = unique_color(i % n_col);
                c.3 = ((i * 19 + 7) & 0xFF) as u8; // garbage X / varying alpha
                buf.push(c);
            }
            let r = TileRect {
                x: 0,
                y: 0,
                w: 32,
                h: 24,
            };
            assert_scan_matches_oracle(&mut scratch, &buf, 32, r, false);
            assert_scan_matches_oracle(&mut scratch, &buf, 32, r, true);
        }

        // 8. Oracle cross-check across diverse pseudo-random and bimodal tiles
        //    (covering widths < 24 and >= 24, opaque and translucent).
        let mut rng = 0x5EED_CAFE_1234_5678u64;
        for case in 0..60usize {
            let w = 16 + (case * 5) % 41;
            let h = 16 + (case * 3) % 21;
            let n = w * h;
            let mut buf = Vec::with_capacity(n);
            let bg_pct = (case * 7) % 60;
            let vary_alpha = case % 3 == 0;
            for i in 0..n {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (rng >> 32) as usize;
                let mut px = if (v % 100) < bg_pct {
                    dom
                } else if case % 5 == 0 {
                    unique_color(v % 250)
                } else {
                    unique_color(1000 + (v ^ i) % 10_000)
                };
                if vary_alpha {
                    px.3 = (0x80 + (v & 0x7F)) as u8;
                }
                buf.push(px);
            }
            let r = TileRect {
                x: 0,
                y: 0,
                w: w as u32,
                h: h as u32,
            };
            assert_scan_matches_oracle(&mut scratch, &buf, w, r, false);
            assert_scan_matches_oracle(&mut scratch, &buf, w, r, true);
        }
    }
}
