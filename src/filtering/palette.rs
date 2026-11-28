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
//! Uses a 512-slot open-addressed AoS table (4 KiB, 64 B aligned, 1 L1d cache
//! line per probe) with a 16-bit generation counter to avoid per-tile clearing,
//! bailing out immediately on the 257th unique color so photo tiles exit within
//! a few hundred pixels.

use std::mem::MaybeUninit;
use std::slice;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::FrameRows;
use crate::filtering::RowKernel;
use crate::prelude::*;
use crate::serialization::framing::SUBTILE_H;
use crate::serialization::framing::SUBTILE_W;
use crate::serialization::framing::TILE_PIXELS;
use crate::serialization::framing::TileRect;
use crate::serialization::framing::packed_palette_indices_len;
use crate::vec4u8::Vec4u8;

pub(crate) const MAX_PALETTE_COLORS: usize = 256;
const HASH_SLOTS: usize = 512;
const HASH_MASK: usize = HASH_SLOTS - 1;

/// Precomputed 4-bpp (`5..=16`-color) lookup tables in both AoS (`[Vec4u8; 16]`,
/// 64-byte aligned for AVX-512 `vpermd` and scalar tails) and SoA per-channel
/// 16-byte tables (`b, g, r, a` for AVX2/SSSE3 `pshufb` and NEON `vqtbl1q_u8`).
#[repr(align(64))]
pub(crate) struct PaletteLut4 {
    pub(crate) colors: [Vec4u8; 16],
    pub(crate) b: [u8; 16],
    pub(crate) g: [u8; 16],
    pub(crate) r: [u8; 16],
    pub(crate) a: [u8; 16],
}

impl PaletteLut4 {
    #[inline]
    pub(crate) fn from_bytes(pal_has_alpha: bool, palette_bytes: &[u8], n_colors: usize) -> Self {
        debug_assert!((5..=16).contains(&n_colors));
        let mut lut = Self {
            colors: [Vec4u8(0, 0, 0, 0xFF); 16],
            b: [0; 16],
            g: [0; 16],
            r: [0; 16],
            a: [0xFF; 16],
        };
        if pal_has_alpha {
            for (i, chunk) in palette_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .take(n_colors)
                .enumerate()
            {
                lut.colors[i] = Vec4u8(chunk[0], chunk[1], chunk[2], chunk[3]);
                lut.b[i] = chunk[0];
                lut.g[i] = chunk[1];
                lut.r[i] = chunk[2];
                lut.a[i] = chunk[3];
            }
        } else {
            for (i, chunk) in palette_bytes
                .as_chunks::<3>()
                .0
                .iter()
                .take(n_colors)
                .enumerate()
            {
                lut.colors[i] = Vec4u8(chunk[0], chunk[1], chunk[2], 0xFF);
                lut.b[i] = chunk[0];
                lut.g[i] = chunk[1];
                lut.r[i] = chunk[2];
            }
        }
        lut
    }
}

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

#[inline(always)]
pub(crate) fn pack_palette_indices_scalar(
    indices: &[u8],
    n_colors: usize,
    packed: &mut [u8],
) -> usize {
    let n = indices.len();
    let out_len = packed_palette_indices_len(n, n_colors);
    debug_assert!(packed.len() >= out_len);
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

#[inline(always)]
pub(crate) fn validate_palette_indices_scalar(n_colors: usize, packed: &[u8]) -> bool {
    if n_colors == 3 {
        let (chunks, tail) = packed.as_chunks::<8>();
        let mut acc = 0u64;
        for chunk in chunks {
            let w = u64::from_ne_bytes(*chunk);
            acc |= w & (w >> 1);
        }
        if (acc & 0x5555_5555_5555_5555) != 0 {
            return false;
        }
        let mut tail_acc = 0u8;
        for &b in tail {
            tail_acc |= b & (b >> 1);
        }
        (tail_acc & 0x55) == 0
    } else if (5..16).contains(&n_colors) {
        let max_n = n_colors as u8;
        packed
            .iter()
            .all(|&b| (b & 0x0F) < max_n && (b >> 4) < max_n)
    } else if (17..MAX_PALETTE_COLORS).contains(&n_colors) {
        let max_n = n_colors as u8;
        packed.iter().all(|&v| v < max_n)
    } else {
        true
    }
}

#[inline(always)]
pub(crate) fn decode_palette_1bpp_scalar(
    c0: Vec4u8,
    c1: Vec4u8,
    packed: &[u8],
    idx_pos: usize,
    dst: &mut [MaybeUninit<Vec4u8>],
) {
    let lut = [c0, c1];
    for (pos, out) in (idx_pos..).zip(dst.iter_mut()) {
        let b = packed[pos >> 3];
        out.write(lut[usize::from((b >> (pos & 7)) & 1)]);
    }
}

#[inline(always)]
pub(crate) fn decode_palette_2bpp_scalar(
    lut: &[Vec4u8; 4],
    packed: &[u8],
    idx_pos: usize,
    dst: &mut [MaybeUninit<Vec4u8>],
) {
    for (pos, out) in (idx_pos..).zip(dst.iter_mut()) {
        let b = packed[pos >> 2];
        out.write(lut[usize::from((b >> ((pos & 3) * 2)) & 3)]);
    }
}

#[inline(always)]
pub(crate) fn decode_palette_4bpp_scalar(
    lut: &[Vec4u8; 16],
    packed: &[u8],
    idx_pos: usize,
    dst: &mut [MaybeUninit<Vec4u8>],
) {
    for (pos, out) in (idx_pos..).zip(dst.iter_mut()) {
        let b = packed[pos >> 1];
        let v = usize::from(if (pos & 1) == 0 { b & 0x0F } else { b >> 4 });
        out.write(lut[v]);
    }
}

#[inline(always)]
pub(crate) fn decode_palette_8bpp_scalar(
    lut: &[u32; MAX_PALETTE_COLORS],
    packed_row: &[u8],
    dst: &mut [MaybeUninit<Vec4u8>],
) {
    debug_assert_eq!(packed_row.len(), dst.len());
    for (out, &v) in dst.iter_mut().zip(packed_row) {
        let [b, g, r, a] = lut[usize::from(v)].to_ne_bytes();
        out.write(Vec4u8(b, g, r, a));
    }
}

/// Single 8-byte open-addressed color-table slot (`[ColorEntry; 512]` = 4 KiB).
///
/// Packing `key`, `epoch`, and `val` into one 8-byte aligned entry keeps every
/// probe (and linear-probe collisions within the same 8-slot block) inside a
/// single 64-byte L1d cache line.
#[repr(C, align(8))]
#[derive(Copy, Clone, Default)]
struct ColorEntry {
    key: u32,
    epoch: u16,
    val: u8,
    _pad: u8,
}

static_assertions::const_assert_eq!(std::mem::size_of::<ColorEntry>(), 8);
static_assertions::const_assert_eq!(std::mem::align_of::<ColorEntry>(), 8);

impl ColorEntry {
    #[inline(always)]
    const fn new(key: u32, epoch: u16, val: u8) -> Self {
        Self {
            key,
            epoch,
            val,
            _pad: 0,
        }
    }
}

#[inline(always)]
const fn color_hash(color: u32) -> usize {
    (color.wrapping_mul(0x9E37_79B1) >> 23) as usize & HASH_MASK
}

#[repr(C, align(64))]
struct ColorTable {
    entries: [ColorEntry; HASH_SLOTS],
    counts: [u32; MAX_PALETTE_COLORS],
    cur_epoch: u16,
}

impl ColorTable {
    fn new() -> Self {
        Self {
            entries: [ColorEntry::default(); HASH_SLOTS],
            counts: [0; MAX_PALETTE_COLORS],
            cur_epoch: 0,
        }
    }

    #[inline(always)]
    fn next_epoch(&mut self) -> u16 {
        self.cur_epoch = self.cur_epoch.wrapping_add(1);
        if self.cur_epoch == 0 {
            self.entries.fill(ColorEntry::default());
            self.cur_epoch = 1;
        }
        self.cur_epoch
    }

    #[inline]
    fn prefix_count(&self, cur_epoch: u16, color: u32) -> usize {
        let mut h = color_hash(color);
        loop {
            let entry = self.entries[h];
            if entry.epoch != cur_epoch {
                return 0;
            }
            if entry.key == color {
                return self.counts[usize::from(entry.val)] as usize;
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
        cur_epoch: u16,
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

        let pf_rows = stride_px > tw && stride_px * 4 >= super::prefetch::MIN_PAGE_STRIDE_BYTES;
        for ry in ovf_ry..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            if pf_rows && ry + super::prefetch::PHASE_B_ROW_LOOKAHEAD < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                unsafe {
                    super::prefetch::row_t0(
                        row_ptr
                            .add(super::prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                            .cast::<u8>(),
                        tw * 4,
                    );
                }
            }
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
            if pf_rows && ry + super::prefetch::PHASE_B_ROW_LOOKAHEAD < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                unsafe {
                    super::prefetch::row_t0(
                        row_ptr
                            .add(super::prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                            .cast::<u8>(),
                        tw * 4,
                    );
                }
            }
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
            let mut h = color_hash(rgb);
            loop {
                let entry = self.entries[h];
                if entry.epoch != cur_epoch {
                    let slot = rgb_count & (MAX_PALETTE_COLORS - 1);
                    debug_assert_eq!(slot, rgb_count);
                    self.entries[h] = ColorEntry::new(rgb, cur_epoch, slot as u8);
                    palette_out[slot] = rgb;
                    self.counts[slot] = cnt;
                    if cnt as usize >= threshold {
                        return HybridScan::TextSafeNoPalette;
                    }
                    rgb_count += 1;
                    break;
                }
                if entry.key == rgb {
                    let idx = usize::from(entry.val);
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

        let pf_rows = stride_px > tw && stride_px * 4 >= super::prefetch::MIN_PAGE_STRIDE_BYTES;
        for ry in ovf_ry..th {
            let row_start = (y0 + ry) * stride_px + x0;
            debug_assert!(row_start + tw <= aos.len());
            // SAFETY: Caller guarantees `rect` is inside `aos`.
            let row_ptr = unsafe { aos.ptr().add(row_start) };
            if pf_rows && ry + super::prefetch::PHASE_B_ROW_LOOKAHEAD < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                unsafe {
                    super::prefetch::row_t0(
                        row_ptr
                            .add(super::prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                            .cast::<u8>(),
                        tw * 4,
                    );
                }
            }
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

                let mut h = color_hash(c);
                loop {
                    let entry = self.entries[h];
                    if entry.epoch != cur_epoch {
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
                        let idx = rgb_count as u8;
                        self.entries[h] = ColorEntry::new(c, cur_epoch, idx);
                        palette_out[rgb_count] = c;
                        self.counts[rgb_count] = 0;
                        rgb_count += 1;
                        last_color = c;
                        last_idx = idx;
                        break;
                    }
                    if entry.key == c {
                        last_color = c;
                        last_idx = entry.val;
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
        debug_assert!((y0 + th - 1) * stride_px + x0 + tw <= aos.len());
        // SAFETY: Caller guarantees `rect` is non-empty and inside `aos`.
        let mut row_ptr = unsafe { aos.ptr().add(y0 * stride_px + x0) };
        let first_c = unsafe { load_pixel_le(row_ptr, 0) } | alpha_or;
        let h0 = color_hash(first_c);
        self.entries[h0] = ColorEntry::new(first_c, cur_epoch, 0);
        palette_out[0] = first_c;
        self.counts[0] = 0;

        let mut count = 1usize;
        let mut last_color = first_c;
        let mut last_idx = 0u8;
        let mut last_run = 0u32;
        let mut out_ptr = indices_out[..n_px].as_mut_ptr();
        let pf_rows = stride_px > tw && stride_px * 4 >= super::prefetch::MIN_PAGE_STRIDE_BYTES;

        for ry in 0..th {
            if pf_rows && ry + super::prefetch::PHASE_B_ROW_LOOKAHEAD < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                unsafe {
                    super::prefetch::row_t0(
                        row_ptr
                            .add(super::prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                            .cast::<u8>(),
                        tw * 4,
                    );
                }
            }
            // SAFETY: `indices_out[..n_px]` bounds-checked `th * tw == n_px` above,
            // so `out_ptr` is valid for `tw` mutable bytes on every row `0..th`.
            let row_out = unsafe { slice::from_raw_parts_mut(out_ptr, tw) };
            let mut x = 0usize;
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                let mut c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
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
                    // Drain the 0..7 remaining pixels of this run scalarly so
                    // short runs and run tails never re-enter the 8-px SIMD check.
                    while x < tw {
                        // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                        let next_c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
                        if next_c != last_color {
                            c = next_c;
                            break;
                        }
                        row_out[x] = last_idx;
                        last_run += 1;
                        x += 1;
                    }
                    if x >= tw {
                        break;
                    }
                }
                self.counts[usize::from(last_idx)] += last_run;
                last_run = 1;

                let mut h = color_hash(c);
                loop {
                    let entry = self.entries[h];
                    if entry.epoch != cur_epoch {
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
                        let idx = count as u8;
                        self.entries[h] = ColorEntry::new(c, cur_epoch, idx);
                        palette_out[count] = c;
                        self.counts[count] = 0;
                        count += 1;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    if entry.key == c {
                        let idx = entry.val;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
                x += 1;
            }
            // SAFETY: Advances by `tw` up to one-past-the-end of `indices_out[..n_px]`.
            out_ptr = unsafe { out_ptr.add(tw) };
            if ry + 1 < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                row_ptr = unsafe { row_ptr.add(stride_px) };
            }
        }

        if count == 1 {
            HybridScan::Solid(palette_out[0].to_le_bytes())
        } else {
            HybridScan::Palette { n_colors: count }
        }
    }
}

#[repr(C, align(64))]
pub(crate) struct PaletteScratch {
    table: ColorTable,
    pub(crate) palette: [u32; MAX_PALETTE_COLORS],
    quad_palettes: Option<Box<[[u32; MAX_PALETTE_COLORS]; 4]>>,
    indices: Vec<u8>,
    packed: Vec<u8>,
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
            // 8-bpp palettes (`n_colors > 16`) are zero-copy directly from
            // `self.indices`, so `self.packed` only ever holds `<= 4`-bpp data.
            packed: vec![0; TILE_PIXELS / 2],
        }
    }

    /// Scans `rect` in `aos` (stride `stride_px` pixels). Returns `Ok(n_colors)`
    /// if the tile contains at most 256 unique colors, populating
    /// `self.palette[..n_colors]` and `self.indices[..tw * th]`. Callers then
    /// invoke [`Self::pack_indices_range`] (a no-op for 8-bpp `17..=256`) and
    /// read wire-ready bytes via [`Self::packed_indices`]. Returns
    /// `Err(overflow_ry)` immediately upon seeing a 257th unique color on row
    /// `overflow_ry`.
    ///
    /// Reads directly from `BufferPointer` without creating a shared `&[u8]`
    /// slice over client shm memory.
    #[inline]
    pub(crate) fn try_encode_2d<const HAS_ALPHA: bool>(
        &mut self,
        aos: BufferPointer<Vec4u8>,
        stride_px: usize,
        rect: TileRect,
    ) -> std::result::Result<usize, usize> {
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let n_px = tw * th;
        debug_assert!(n_px > 0 && n_px <= TILE_PIXELS);

        let cur_epoch = self.table.next_epoch();
        let alpha_or = if HAS_ALPHA { 0u32 } else { 0xFF00_0000u32 };

        debug_assert!((y0 + th - 1) * stride_px + x0 + tw <= aos.len());
        // SAFETY: Caller guarantees `rect` is non-empty and inside `aos`.
        let mut row_ptr = unsafe { aos.ptr().add(y0 * stride_px + x0) };
        let first_c = unsafe { load_pixel_le(row_ptr, 0) } | alpha_or;
        let h0 = color_hash(first_c);
        self.table.entries[h0] = ColorEntry::new(first_c, cur_epoch, 0);
        self.palette[0] = first_c;

        let mut count = 1usize;
        let mut last_color = first_c;
        let mut last_idx = 0u8;
        let mut out_ptr = self.indices[..n_px].as_mut_ptr();
        let pf_rows = stride_px > tw && stride_px * 4 >= super::prefetch::MIN_PAGE_STRIDE_BYTES;

        for ry in 0..th {
            if pf_rows && ry + super::prefetch::PHASE_B_ROW_LOOKAHEAD < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                unsafe {
                    super::prefetch::row_t0(
                        row_ptr
                            .add(super::prefetch::PHASE_B_ROW_LOOKAHEAD * stride_px)
                            .cast::<u8>(),
                        tw * 4,
                    );
                }
            }
            // SAFETY: `self.indices[..n_px]` bounds-checked `th * tw == n_px` above,
            // so `out_ptr` is valid for `tw` mutable bytes on every row `0..th`.
            let row_out = unsafe { slice::from_raw_parts_mut(out_ptr, tw) };
            let mut x = 0usize;
            while x < tw {
                // SAFETY: `x < tw` and `row_ptr` is valid for `tw` `Vec4u8` reads.
                let mut c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
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
                    // Drain the 0..7 remaining pixels of this run scalarly so
                    // short runs and run tails never re-enter the 8-px SIMD check.
                    while x < tw {
                        // SAFETY: `x < tw` and `row_ptr` is valid for `tw` reads.
                        let next_c = unsafe { load_pixel_le(row_ptr, x) } | alpha_or;
                        if next_c != last_color {
                            c = next_c;
                            break;
                        }
                        row_out[x] = last_idx;
                        x += 1;
                    }
                    if x >= tw {
                        break;
                    }
                }
                let mut h = color_hash(c);
                loop {
                    let entry = self.table.entries[h];
                    if entry.epoch != cur_epoch {
                        if count >= MAX_PALETTE_COLORS {
                            return Err(ry);
                        }
                        let idx = count as u8;
                        self.table.entries[h] = ColorEntry::new(c, cur_epoch, idx);
                        self.palette[count] = c;
                        count += 1;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    if entry.key == c {
                        let idx = entry.val;
                        last_color = c;
                        last_idx = idx;
                        row_out[x] = idx;
                        break;
                    }
                    h = (h + 1) & HASH_MASK;
                }
                x += 1;
            }
            // SAFETY: Advances by `tw` up to one-past-the-end of `self.indices[..n_px]`.
            out_ptr = unsafe { out_ptr.add(tw) };
            if ry + 1 < th {
                // SAFETY: Caller guarantees all `th` rows of `rect` lie inside `aos`.
                row_ptr = unsafe { row_ptr.add(stride_px) };
            }
        }

        Ok(count)
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

    /// Packs `self.indices[start..start + n_px]` into `self.packed` for
    /// 1/2/4-bpp palettes (`2..=16` colors); 8-bpp palettes (`17..=256`) are
    /// zero-copy and read directly from `self.indices` via
    /// [`Self::packed_indices`].
    #[inline]
    pub(super) fn pack_indices_range(
        &mut self,
        tier: super::arch::Tier,
        start: usize,
        n_px: usize,
        n_colors: usize,
    ) {
        if (2..=16).contains(&n_colors) {
            tier.pack_palette_indices(
                &self.indices[start..start + n_px],
                n_colors,
                &mut self.packed,
            );
        }
    }

    /// Returns the bitpacked index bytes for `n_px` pixels starting at
    /// `self.indices[start..]` with `n_colors` palette entries: zero-copy
    /// directly into `self.indices` for 8-bpp (`n_colors > 16`), or into
    /// `self.packed` for 1/2/4-bpp (`2..=16`).
    #[inline]
    pub(crate) fn packed_indices(&self, start: usize, n_px: usize, n_colors: usize) -> &[u8] {
        if n_colors <= 1 {
            &[]
        } else if n_colors > 16 {
            &self.indices[start..start + n_px]
        } else {
            &self.packed[..packed_palette_indices_len(n_px, n_colors)]
        }
    }
}

/// Validates the palette table header (`palette_bytes` length and color count
/// `2..=256`), returning `(n_colors, expected_packed_len)` for `n_pixels`.
#[inline]
pub(crate) fn parse_palette_header(
    pal_has_alpha: bool,
    palette_bytes: &[u8],
    n_pixels: usize,
) -> Result<(usize, usize)> {
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
    Ok((n_colors, packed_palette_indices_len(n_pixels, n_colors)))
}

/// Validates `packed.len()`, canonical zero trailing padding bits, and that
/// every active palette index is `< n_colors` (using `kernel` SIMD for
/// non-power-of-two palette counts `3`, `5..=15`, and `17..=255`).
#[inline(always)]
pub(crate) fn validate_palette_packed<K: RowKernel>(
    kernel: K,
    n_colors: usize,
    packed: &[u8],
    n_pixels: usize,
) -> Result<()> {
    let expected_len = packed_palette_indices_len(n_pixels, n_colors);
    if packed.len() != expected_len {
        bail!(
            "packed palette indices len {} != expected {expected_len}",
            packed.len()
        );
    }
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
        if n_colors == 3 && !kernel.validate_palette_indices(n_colors, packed) {
            bail!("palette index out of range for {n_colors}-color palette");
        }
    } else if n_colors <= 16 {
        if (n_pixels & 1) != 0 && (packed[n_pixels >> 1] >> 4) != 0 {
            bail!("non-zero trailing bits in 4-bpp palette stream");
        }
        if n_colors < 16 && !kernel.validate_palette_indices(n_colors, packed) {
            bail!("palette index out of range for {n_colors}-color palette");
        }
    } else if n_colors < MAX_PALETTE_COLORS && !kernel.validate_palette_indices(n_colors, packed) {
        bail!("palette index out of range for {n_colors}-color palette");
    }
    Ok(())
}

#[inline(always)]
fn read_palette_entry(pal_has_alpha: bool, palette_bytes: &[u8], idx: usize) -> Vec4u8 {
    if pal_has_alpha {
        let o = idx * 4;
        Vec4u8(
            palette_bytes[o],
            palette_bytes[o + 1],
            palette_bytes[o + 2],
            palette_bytes[o + 3],
        )
    } else {
        let o = idx * 3;
        Vec4u8(
            palette_bytes[o],
            palette_bytes[o + 1],
            palette_bytes[o + 2],
            0xFF,
        )
    }
}

#[inline(always)]
fn build_palette_lut8(
    pal_has_alpha: bool,
    palette_bytes: &[u8],
    n_colors: usize,
) -> [u32; MAX_PALETTE_COLORS] {
    let mut lut = [0u32; MAX_PALETTE_COLORS];
    if pal_has_alpha {
        for (dst, chunk) in lut[..n_colors]
            .iter_mut()
            .zip(palette_bytes.as_chunks::<4>().0)
        {
            *dst = u32::from_ne_bytes(*chunk);
        }
    } else {
        for (dst, chunk) in lut[..n_colors]
            .iter_mut()
            .zip(palette_bytes.as_chunks::<3>().0)
        {
            *dst = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], 0xFF]);
        }
    }
    lut
}

/// Validates all palette indices up front (leaving `frame` untouched on error)
/// and decodes the `tw x th` palette tile into `frame` at `(x0, y0)` via
/// `kernel`, using non-temporal streaming stores when `allow_stream` is `true`
/// and every row is aligned to `K::STREAM_ALIGN` with `tw % 16 == 0`.
///
/// Returns `Ok(streamed)` where `streamed` is `true` iff non-temporal stores
/// were issued (caller must call `kernel.fence_stream()`).
///
/// # Safety
/// `(x0, y0, tw, th)` must be non-empty (`tw > 0 && th > 0`), validated
/// in-bounds within `frame`, and disjoint from all other tiles being decoded
/// concurrently. `palette_bytes` and `n_colors` must have been validated by
/// [`parse_palette_header`].
#[inline(always)]
pub(crate) unsafe fn decode_palette_tile<K: RowKernel>(
    kernel: K,
    pal_has_alpha: bool,
    palette_bytes: &[u8],
    n_colors: usize,
    packed: &[u8],
    frame: &FrameRows,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
    allow_stream: bool,
) -> Result<bool> {
    let n_px = tw * th;
    validate_palette_packed(kernel, n_colors, packed, n_px).location(loc!())?;

    // SAFETY: Caller guarantees `th > 0` and `(x0, y0, tw, th)` is in-bounds.
    let stream_rows = allow_stream
        && K::STREAM_ALIGN > 1
        && tw.is_multiple_of(16)
        && (unsafe { frame.row_ptr(y0, x0) } as usize).is_multiple_of(K::STREAM_ALIGN)
        && (frame.stride_px * 4).is_multiple_of(K::STREAM_ALIGN);

    let mut idx_pos = 0usize;
    if n_colors <= 2 {
        let c0 = read_palette_entry(pal_has_alpha, palette_bytes, 0);
        let c1 = read_palette_entry(pal_has_alpha, palette_bytes, 1);
        if stream_rows {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint;
                // `stream_rows` verified `STREAM_ALIGN`, `tw % 16 == 0`, and `idx_pos % 8 == 0`.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe { kernel.decode_palette_1bpp_row::<true>(c0, c1, packed, idx_pos, dst_row) };
                idx_pos += tw;
            }
        } else {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe {
                    kernel.decode_palette_1bpp_row::<false>(c0, c1, packed, idx_pos, dst_row);
                }
                idx_pos += tw;
            }
        }
    } else if n_colors <= 4 {
        let mut lut = [Vec4u8(0, 0, 0, 0xFF); 4];
        for (i, slot) in lut[..n_colors].iter_mut().enumerate() {
            *slot = read_palette_entry(pal_has_alpha, palette_bytes, i);
        }
        if stream_rows {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint;
                // `stream_rows` verified `STREAM_ALIGN`, `tw % 16 == 0`, and `idx_pos % 4 == 0`.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe { kernel.decode_palette_2bpp_row::<true>(&lut, packed, idx_pos, dst_row) };
                idx_pos += tw;
            }
        } else {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe { kernel.decode_palette_2bpp_row::<false>(&lut, packed, idx_pos, dst_row) };
                idx_pos += tw;
            }
        }
    } else if n_colors <= 16 {
        let lut = PaletteLut4::from_bytes(pal_has_alpha, palette_bytes, n_colors);
        if stream_rows {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint;
                // `stream_rows` verified `STREAM_ALIGN`, `tw % 16 == 0`, and `idx_pos % 2 == 0`.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe { kernel.decode_palette_4bpp_row::<true>(&lut, packed, idx_pos, dst_row) };
                idx_pos += tw;
            }
        } else {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe { kernel.decode_palette_4bpp_row::<false>(&lut, packed, idx_pos, dst_row) };
                idx_pos += tw;
            }
        }
    } else {
        let lut = build_palette_lut8(pal_has_alpha, palette_bytes, n_colors);
        if stream_rows {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint;
                // `stream_rows` verified `STREAM_ALIGN` and `tw % 16 == 0`.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe {
                    kernel.decode_palette_8bpp_row::<true>(
                        &lut,
                        &packed[idx_pos..idx_pos + tw],
                        dst_row,
                    );
                }
                idx_pos += tw;
            }
        } else {
            for ry in 0..th {
                // SAFETY: Caller guarantees `(x0, y0, tw, th)` is in-bounds and disjoint.
                let dst_row = unsafe { slice::from_raw_parts_mut(frame.row_ptr(y0 + ry, x0), tw) };
                unsafe {
                    kernel.decode_palette_8bpp_row::<false>(
                        &lut,
                        &packed[idx_pos..idx_pos + tw],
                        dst_row,
                    );
                }
                idx_pos += tw;
            }
        }
    }
    Ok(stream_rows)
}

/// Validates `palette_bytes` and `packed` up front (including all active 2/4/8-bpp
/// indices via `kernel`), returning the `[Vec4u8; 256]` lookup table and active
/// color count `n_colors` (`2..=256`).
#[cfg(test)]
pub(crate) fn prepare_palette_lut<K: RowKernel>(
    kernel: K,
    pal_has_alpha: bool,
    palette_bytes: &[u8],
    packed: &[u8],
    n_pixels: usize,
) -> Result<([Vec4u8; MAX_PALETTE_COLORS], usize)> {
    let (n_colors, _) = parse_palette_header(pal_has_alpha, palette_bytes, n_pixels)?;
    validate_palette_packed(kernel, n_colors, packed, n_pixels)?;
    let mut lut = [Vec4u8(0, 0, 0, 0xFF); MAX_PALETTE_COLORS];
    for (i, slot) in lut[..n_colors].iter_mut().enumerate() {
        *slot = read_palette_entry(pal_has_alpha, palette_bytes, i);
    }
    Ok((lut, n_colors))
}

/// Infallibly decodes one validated row (`dst_row.len()` pixels) from `packed`
/// starting at pixel offset `*idx_pos`, advancing `*idx_pos` by `dst_row.len()`.
#[cfg(test)]
#[inline]
pub(crate) fn decode_palette_row(
    lut: &[Vec4u8; MAX_PALETTE_COLORS],
    n_colors: usize,
    packed: &[u8],
    idx_pos: &mut usize,
    dst_row: &mut [MaybeUninit<Vec4u8>],
) {
    let pos = *idx_pos;
    if n_colors <= 2 {
        decode_palette_1bpp_scalar(lut[0], lut[1], packed, pos, dst_row);
    } else if n_colors <= 4 {
        let lut2: [Vec4u8; 4] = lut[..4].try_into().unwrap();
        decode_palette_2bpp_scalar(&lut2, packed, pos, dst_row);
    } else if n_colors <= 16 {
        let lut4: [Vec4u8; 16] = lut[..16].try_into().unwrap();
        decode_palette_4bpp_scalar(&lut4, packed, pos, dst_row);
    } else {
        let mut lut8 = [0u32; MAX_PALETTE_COLORS];
        for (dst, &px) in lut8.iter_mut().zip(lut.iter()) {
            *dst = u32::from_ne_bytes([px.0, px.1, px.2, px.3]);
        }
        decode_palette_8bpp_scalar(&lut8, &packed[pos..pos + dst_row.len()], dst_row);
    }
    *idx_pos = pos + dst_row.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filtering::ScalarKernel;
    use crate::filtering::arch::Tier;

    fn encode_decode_palette_case(
        tier: Tier,
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
        let n_colors = res.unwrap();
        assert_eq!(n_colors, colors.len());
        scratch.pack_indices_range(tier, 0, tw * th, n_colors);

        let packed = scratch.packed_indices(0, tw * th, n_colors);
        if n_colors == 1 {
            assert!(packed.is_empty());
            return;
        }
        assert_eq!(packed.len(), packed_palette_indices_len(tw * th, n_colors));

        let pal_has_alpha = has_alpha
            && scratch.palette[..n_colors]
                .iter()
                .any(|&c| (c >> 24) != 0xFF);
        let mut pal_bytes = Vec::new();
        for &c in &scratch.palette[..n_colors] {
            let b = c.to_le_bytes();
            pal_bytes.extend_from_slice(if pal_has_alpha { &b[..] } else { &b[..3] });
        }
        let (lut, dec_n) =
            prepare_palette_lut(ScalarKernel, pal_has_alpha, &pal_bytes, packed, tw * th).unwrap();
        assert_eq!(dec_n, n_colors);

        let mut idx_pos = 0;
        let mut out_row = vec![MaybeUninit::uninit(); tw];
        for ry in 0..th {
            decode_palette_row(&lut, dec_n, packed, &mut idx_pos, &mut out_row);
            for rx in 0..tw {
                let mut exp = frame[(y0 + ry) * stride_px + x0 + rx];
                if !has_alpha {
                    exp.3 = 0xFF;
                }
                // SAFETY: `decode_palette_row` initializes every element.
                assert_eq!(unsafe { out_row[rx].assume_init() }, exp);
            }
        }
    }

    #[test]
    fn test_palette_1_2_4_8_bpp_and_overflow_bailout() {
        // Test exact boundary color counts: 1, 2 (1bpp), 3, 4 (2bpp), 5, 15, 16 (4bpp),
        // 17, 128, 255, 256 (8bpp), with odd and SIMD-crossing dimensions across all
        // supported tiers.
        for tier in Tier::supported_tiers() {
            for n in [1usize, 2, 3, 4, 5, 15, 16, 17, 128, 255, 256] {
                let colors: Vec<Vec4u8> = (0..n)
                    .map(|i| Vec4u8(i as u8, (i * 3) as u8, (i * 7) as u8, 0xFF))
                    .collect();
                encode_decode_palette_case(tier, &colors, false, 19, 17, 24);
                encode_decode_palette_case(tier, &colors, true, 19, 17, 24);
                encode_decode_palette_case(tier, &colors, false, 65, 5, 72);
            }

            // Translucent palette entries (same RGB, different alpha).
            let trans_colors = vec![
                Vec4u8(10, 20, 30, 0x10),
                Vec4u8(10, 20, 30, 0x80),
                Vec4u8(10, 20, 30, 0xFF),
            ];
            encode_decode_palette_case(tier, &trans_colors, true, 7, 5, 12);
        }

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
        // Out-of-range 2-bpp index (3 for a 3-color palette) caught up front in
        // `prepare_palette_lut` across start, mid-SIMD-chunk, and tail offsets.
        let pal3 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(prepare_palette_lut(ScalarKernel, false, &pal3, &[0b11], 1).is_err());
        for bad_byte in [0, 31, 63, 64, 129] {
            for shift in [0u8, 2, 4, 6] {
                let mut packed = vec![0b10_01_00_10u8; 130]; // valid 0/1/2 lanes only
                packed[bad_byte] |= 0b11 << shift;
                assert!(
                    prepare_palette_lut(ScalarKernel, false, &pal3, &packed, 130 * 4).is_err(),
                    "expected OOB 2-bpp rejection at byte={bad_byte} shift={shift}"
                );
            }
        }

        // Out-of-range 4-bpp index (`>= n_colors` in lo or hi nibble) caught up front.
        let pal5 = vec![0u8; 5 * 3];
        assert!(prepare_palette_lut(ScalarKernel, false, &pal5, &[0x0A], 1).is_err());
        for bad_byte in [0, 31, 63, 64, 129] {
            for bad_val in [0x05u8, 0x50u8, 0x0Fu8, 0xF0u8] {
                let mut packed = vec![0x43u8; 130]; // valid 3/4 nibbles
                packed[bad_byte] = bad_val;
                assert!(
                    prepare_palette_lut(ScalarKernel, false, &pal5, &packed, 130 * 2).is_err(),
                    "expected OOB 4-bpp rejection at byte={bad_byte} val={bad_val:#x}"
                );
            }
        }

        // Out-of-range 8-bpp index (`>= n_colors`) caught up front across start/mid/tail.
        let pal17 = vec![0u8; 17 * 3];
        assert!(prepare_palette_lut(ScalarKernel, false, &pal17, &[17], 1).is_err());
        for bad_byte in [0, 31, 63, 64, 127, 128, 131] {
            let mut packed = vec![16u8; 132];
            packed[bad_byte] = 17;
            assert!(
                prepare_palette_lut(ScalarKernel, false, &pal17, &packed, 132).is_err(),
                "expected OOB 8-bpp rejection at byte={bad_byte}"
            );
        }

        // Non-zero trailing padding bits on 1/2/4-bpp.
        let pal2 = vec![0u8; 2 * 3];
        assert!(prepare_palette_lut(ScalarKernel, false, &pal2, &[0b0000_0010], 1).is_err());
        assert!(prepare_palette_lut(ScalarKernel, false, &pal3, &[0b0000_0100], 1).is_err());
        assert!(prepare_palette_lut(ScalarKernel, false, &pal5, &[0x10], 1).is_err());

        // Invalid palette byte length / color count (0, 1, 257) / packed length.
        assert!(prepare_palette_lut(ScalarKernel, false, &[], &[], 1).is_err());
        assert!(prepare_palette_lut(ScalarKernel, false, &[0, 1], &[0], 1).is_err());
        assert!(prepare_palette_lut(ScalarKernel, false, &[0, 1, 2], &[], 1).is_err());
        let pal257 = vec![0u8; 257 * 3];
        assert!(prepare_palette_lut(ScalarKernel, false, &pal257, &[0], 1).is_err());
        assert!(prepare_palette_lut(ScalarKernel, false, &pal2, &[0, 0], 1).is_err());

        // Verify OOB index leaves destination frame untouched (zero partial corruption).
        let sentinel = Vec4u8(0xDE, 0xAD, 0xBE, 0xEF);
        let mut canvas = vec![MaybeUninit::new(sentinel); 16 * 4];
        let frame = FrameRows {
            base: canvas.as_mut_ptr(),
            stride_px: 16,
        };
        let mut bad_2bpp = vec![0u8; 16]; // 64 pixels
        bad_2bpp[15] = 0b1100_0000; // last pixel OOB for 3-color palette
        assert!(
            unsafe {
                decode_palette_tile(
                    ScalarKernel,
                    false,
                    &pal3,
                    3,
                    &bad_2bpp,
                    &frame,
                    0,
                    0,
                    16,
                    4,
                    false,
                )
            }
            .is_err()
        );
        for slot in &canvas {
            // SAFETY: Every element was initialized to `sentinel`.
            assert_eq!(unsafe { slot.assume_init() }, sentinel);
        }
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
        let tier = Tier::detect();
        let mut ref_scratch = PaletteScratch::new();
        let ref_pal = if track_alpha {
            ref_scratch.try_encode_2d::<true>(bp, stride_px, rect)
        } else {
            ref_scratch.try_encode_2d::<false>(bp, stride_px, rect)
        };
        if let Ok(ref_n) = ref_pal {
            ref_scratch.pack_indices_range(tier, 0, rect.w as usize * rect.h as usize, ref_n);
        }

        let n_px = rect.w as usize * rect.h as usize;
        let scan = if track_alpha {
            scratch.scan_hybrid_leaf::<true>(bp, stride_px, rect)
        } else {
            scratch.scan_hybrid_leaf::<false>(bp, stride_px, rect)
        };
        assert_eq!(scan.is_text_safe(), exp_text_safe);
        match (scan, ref_pal) {
            (HybridScan::Solid(px), Ok(1)) => {
                assert_eq!(px, ref_scratch.palette[0].to_le_bytes());
            },
            (HybridScan::Palette { n_colors }, Ok(ref_n)) => {
                assert_eq!(n_colors, ref_n);
                assert_eq!(
                    scratch.palette[..n_colors],
                    ref_scratch.palette[..ref_n],
                    "palette table mismatch"
                );
                scratch.pack_indices_range(tier, 0, n_px, n_colors);
                assert_eq!(
                    scratch.packed_indices(0, n_px, n_colors),
                    ref_scratch.packed_indices(0, n_px, ref_n),
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
                    let start = q * SUBTILE_PIXELS;
                    scratch.pack_indices_range(tier, start, n_px, n_colors);
                    assert_eq!(
                        scratch.packed_indices(start, n_px, n_colors),
                        ref_scratch.packed_indices(0, n_px, n_colors)
                    );
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

    fn colors_hashing_to(slot: usize, count: usize, alpha: u8) -> Vec<Vec4u8> {
        let mut out = Vec::with_capacity(count);
        let mut i = 0u32;
        while out.len() < count {
            let px = Vec4u8(
                (i & 0xFF) as u8,
                ((i >> 8) & 0xFF) as u8,
                ((i >> 16) & 0xFF) as u8,
                alpha,
            );
            let key = u32::from_le_bytes([px.0, px.1, px.2, alpha]);
            if color_hash(key) == slot {
                out.push(px);
            }
            i += 1;
        }
        out
    }

    #[test]
    fn test_color_table_epoch_wraparound_clears_stale_entries() {
        let c0 = Vec4u8(10, 20, 30, 0xFF);
        let c1 = Vec4u8(40, 50, 60, 0xFF);
        let c2 = Vec4u8(70, 80, 90, 0xFF);
        let c3 = Vec4u8(100, 110, 120, 0xFF);
        let c0_u32 = u32::from_le_bytes([c0.0, c0.1, c0.2, c0.3]);
        let c1_u32 = u32::from_le_bytes([c1.0, c1.1, c1.2, c1.3]);
        let rect = TileRect {
            x: 0,
            y: 0,
            w: 16,
            h: 2,
        };

        // 1. `try_encode_2d` across `u16::MAX -> 1` wraparound for both `HAS_ALPHA` modes:
        //    stale epoch-1 entries `[c0 -> 0, c1 -> 1]` must be cleared so a post-wrap
        //    tile starting with `[c1, c0]` assigns `c1 -> 0, c0 -> 1` cleanly.
        for has_alpha in [false, true] {
            let mut scratch = PaletteScratch::new();
            let tile_epoch1: Vec<Vec4u8> =
                (0..32).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
            let p1 = tile_epoch1.as_ptr();
            let bp1 = unsafe { BufferPointer::new(&p1, tile_epoch1.len()) };
            let n1 = if has_alpha {
                scratch.try_encode_2d::<true>(bp1, 16, rect).unwrap()
            } else {
                scratch.try_encode_2d::<false>(bp1, 16, rect).unwrap()
            };
            assert_eq!(n1, 2);
            assert_eq!(scratch.table.cur_epoch, 1);
            assert_eq!(scratch.palette[..2], [c0_u32, c1_u32]);

            // Jump right before wraparound (`u16::MAX - 1` -> `u16::MAX` -> `1`).
            scratch.table.cur_epoch = u16::MAX - 1;
            let tile_max: Vec<Vec4u8> = (0..32).map(|i| if i % 2 == 0 { c2 } else { c3 }).collect();
            let p_max = tile_max.as_ptr();
            let bp_max = unsafe { BufferPointer::new(&p_max, tile_max.len()) };
            let n_max = if has_alpha {
                scratch.try_encode_2d::<true>(bp_max, 16, rect).unwrap()
            } else {
                scratch.try_encode_2d::<false>(bp_max, 16, rect).unwrap()
            };
            assert_eq!(n_max, 2);
            assert_eq!(scratch.table.cur_epoch, u16::MAX);

            // Reversed order `[c1, c0]` on wrap to epoch `1` would corrupt indices if
            // stale epoch-1 entries survived wraparound.
            let tile_wrapped: Vec<Vec4u8> =
                (0..32).map(|i| if i % 2 == 0 { c1 } else { c0 }).collect();
            let p_wrap = tile_wrapped.as_ptr();
            let bp_wrap = unsafe { BufferPointer::new(&p_wrap, tile_wrapped.len()) };
            let n_wrap = if has_alpha {
                scratch.try_encode_2d::<true>(bp_wrap, 16, rect).unwrap()
            } else {
                scratch.try_encode_2d::<false>(bp_wrap, 16, rect).unwrap()
            };
            assert_eq!(n_wrap, 2);
            assert_eq!(scratch.table.cur_epoch, 1);
            assert_eq!(scratch.palette[..2], [c1_u32, c0_u32]);
            for (i, &idx) in scratch.indices[..32].iter().enumerate() {
                assert_eq!(idx, (i % 2) as u8);
            }
        }

        // 2. `scan_hybrid_leaf` across `u16::MAX -> 1` wraparound (both `TRACK_ALPHA` modes),
        //    plus `finish_translucent_rgb` triggering the wraparound internally.
        for track_alpha in [false, true] {
            let mut scratch = PaletteScratch::new();
            let tile_epoch1: Vec<Vec4u8> =
                (0..32).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
            let p1 = tile_epoch1.as_ptr();
            let bp1 = unsafe { BufferPointer::new(&p1, tile_epoch1.len()) };
            let s1 = if track_alpha {
                scratch.scan_hybrid_leaf::<true>(bp1, 16, rect)
            } else {
                scratch.scan_hybrid_leaf::<false>(bp1, 16, rect)
            };
            assert_eq!(s1, HybridScan::Palette { n_colors: 2 });
            assert_eq!(scratch.table.cur_epoch, 1);
            assert_eq!(scratch.palette[..2], [c0_u32, c1_u32]);

            scratch.table.cur_epoch = u16::MAX;
            let tile_wrapped: Vec<Vec4u8> =
                (0..32).map(|i| if i % 2 == 0 { c1 } else { c0 }).collect();
            let pw = tile_wrapped.as_ptr();
            let bpw = unsafe { BufferPointer::new(&pw, tile_wrapped.len()) };
            let sw = if track_alpha {
                scratch.scan_hybrid_leaf::<true>(bpw, 16, rect)
            } else {
                scratch.scan_hybrid_leaf::<false>(bpw, 16, rect)
            };
            assert_eq!(sw, HybridScan::Palette { n_colors: 2 });
            assert_eq!(scratch.table.cur_epoch, 1);
            assert_eq!(scratch.palette[..2], [c1_u32, c0_u32]);
            for (i, &idx) in scratch.indices[..32].iter().enumerate() {
                assert_eq!(idx, (i % 2) as u8);
            }
        }

        // `scan_hybrid_leaf::<true>` runs at `u16::MAX`, overflows 256 BGRA colors,
        // and `finish_translucent_rgb` wraps `next_epoch()` to `1`.
        let mut scratch = PaletteScratch::new();
        let seed_tile: Vec<Vec4u8> = (0..32).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
        let ps = seed_tile.as_ptr();
        let bps = unsafe { BufferPointer::new(&ps, seed_tile.len()) };
        assert_eq!(
            scratch.scan_hybrid_leaf::<true>(bps, 16, rect),
            HybridScan::Palette { n_colors: 2 }
        );
        assert_eq!(scratch.table.cur_epoch, 1);

        scratch.table.cur_epoch = u16::MAX - 1;
        let trans_257: Vec<Vec4u8> = (0..300)
            .map(|i| {
                let mut px = if i < 256 { c0 } else { c1 };
                px.3 = (i & 0xFF) as u8;
                px
            })
            .collect();
        let rect_t = TileRect {
            x: 0,
            y: 0,
            w: 20,
            h: 15,
        };
        let pt = trans_257.as_ptr();
        let bpt = unsafe { BufferPointer::new(&pt, trans_257.len()) };
        assert_eq!(
            scratch.scan_hybrid_leaf::<true>(bpt, 20, rect_t),
            HybridScan::TextSafeNoPalette
        );
        assert_eq!(scratch.table.cur_epoch, 1);
    }

    #[test]
    fn test_color_table_hash_collisions_and_slot_511_wraparound() {
        // Synthesize distinct opaque and translucent colors colliding at slot 511
        // (`HASH_SLOTS - 1`), forcing linear probing to wrap `511 -> 0 -> 1 -> ...`,
        // interleaved with colors hashing directly to slot `0`.
        for has_alpha in [false, true] {
            let alpha = if has_alpha { 0x80 } else { 0xFF };
            let mut colliding = colors_hashing_to(HASH_SLOTS - 1, 8, alpha);
            let slot0 = colors_hashing_to(0, 4, alpha);
            colliding.extend_from_slice(&slot0);

            for &c in &colliding[..8] {
                assert_eq!(
                    color_hash(u32::from_le_bytes([c.0, c.1, c.2, alpha])),
                    HASH_SLOTS - 1
                );
            }
            for &c in &colliding[8..] {
                assert_eq!(color_hash(u32::from_le_bytes([c.0, c.1, c.2, alpha])), 0);
            }

            // Roundtrip through `try_encode_2d` across all supported tiers.
            for tier in Tier::supported_tiers() {
                encode_decode_palette_case(tier, &colliding, has_alpha, 19, 13, 24);
            }

            // Verify exact palette order and `scan_hybrid_leaf` / `scan_hybrid_quadrant`
            // agreement on the same colliding sequence.
            let mut pixels = Vec::with_capacity(24 * 16);
            for i in 0..(24 * 16) {
                pixels.push(colliding[i % colliding.len()]);
            }
            let rect = TileRect {
                x: 0,
                y: 0,
                w: 24,
                h: 16,
            };
            let mut scratch = PaletteScratch::new();
            assert_scan_matches_oracle(&mut scratch, &pixels, 24, rect, has_alpha);

            // Also verify `prefix_count` probes across `511 -> 0` wraparound when
            // overflowing 256 colors in `scan_hybrid`: place `dom_wrapped` (hashes to
            // 511, linearly probes to slot 0) once in the 256-color prefix, overflow at
            // pixel 256, and place exactly 349 more copies interleaved in the suffix so
            // total is `1 + 349 = 350` (exact 35% threshold). Missing the `+1` prefix
            // count across `511 -> 0` would drop to `349` and misclassify as `Photo`.
            let dom_wrapped = colliding[1];
            let mut ovf = Vec::with_capacity(40 * 25);
            for &c in &colliding {
                ovf.push(c);
            }
            let mut next = 0u32;
            while ovf.len() < 256 {
                let px = Vec4u8((next & 0xFF) as u8, ((next >> 8) & 0xFF) as u8, 0x77, alpha);
                if !ovf.contains(&px) {
                    ovf.push(px);
                }
                next += 1;
            }
            // Interleave `[u_a, u_b, dom_wrapped]` after pixel 256 so Misra-Gries
            // enters the uncertain band and must call `prefix_count` on `dom_wrapped`.
            let mut dom_left = 349usize;
            let mut tail_id = 0u32;
            while ovf.len() < 1000 {
                let rem = 1000 - ovf.len();
                if (dom_left > 0 && ovf.len() % 3 == 2) || rem <= dom_left {
                    ovf.push(dom_wrapped);
                    dom_left -= 1;
                } else {
                    ovf.push(Vec4u8(
                        (tail_id & 0xFF) as u8,
                        ((tail_id >> 8) & 0xFF) as u8,
                        0xEE,
                        alpha,
                    ));
                    tail_id += 1;
                }
            }
            let rect_ovf = TileRect {
                x: 0,
                y: 0,
                w: 40,
                h: 25,
            };
            assert_scan_matches_oracle(&mut scratch, &ovf, 40, rect_ovf, has_alpha);
            let p = ovf.as_ptr();
            let bp = unsafe { BufferPointer::new(&p, ovf.len()) };
            let scan = if has_alpha {
                scratch.scan_hybrid_leaf::<true>(bp, 40, rect_ovf)
            } else {
                scratch.scan_hybrid_leaf::<false>(bp, 40, rect_ovf)
            };
            assert_eq!(scan, HybridScan::TextSafeNoPalette);
        }
    }

    #[test]
    fn test_two_color_alternating_and_run_drain_edge_cases() {
        // Exercises 1-px `bg <-> fg` alternation, short runs (1..=9 pixels covering the
        // 8-px SIMD threshold and scalar run-drain tail), long runs (>= 16 px), odd
        // widths, padded strides, and `0x0000_0000` with `HAS_ALPHA = true`.
        let pairs = [
            (
                Vec4u8(0x1E, 0x1E, 0x1E, 0xFF),
                Vec4u8(0xD4, 0xD4, 0xD4, 0xFF),
                false,
            ),
            (
                Vec4u8(0x00, 0x00, 0x00, 0x00),
                Vec4u8(0xFF, 0xFF, 0xFF, 0x80),
                true,
            ),
        ];
        let mut scratch = PaletteScratch::new();
        for (bg, fg, has_alpha) in pairs {
            for &tw in &[1usize, 2, 7, 8, 9, 15, 16, 17, 33] {
                let th = 6usize;
                let stride_px = tw + 5;
                let x0 = 3usize;
                let y0 = 1usize;
                let rect = TileRect {
                    x: x0 as u32,
                    y: y0 as u32,
                    w: tw as u32,
                    h: th as u32,
                };
                for run_len in [1usize, 2, 3, 7, 8, 9, 17] {
                    let mut frame = vec![Vec4u8(0x55, 0x66, 0x77, 0xFF); stride_px * (th + 2)];
                    for ry in 0..th {
                        for rx in 0..tw {
                            // Mix row-local run lengths and alternating rows so runs
                            // end mid-row, at exact `tw`, and wrap across rows.
                            let pos = ry * tw + rx;
                            let px = if ((pos / run_len) + (ry & 1)) % 2 == 0 {
                                bg
                            } else {
                                fg
                            };
                            frame[(y0 + ry) * stride_px + x0 + rx] = px;
                        }
                    }
                    assert_scan_matches_oracle(&mut scratch, &frame, stride_px, rect, has_alpha);
                }
            }
        }
    }

    #[test]
    fn test_palette_256_saturation_and_257_bailout_both_alpha_modes() {
        let mut scratch = PaletteScratch::new();
        let rect = TileRect {
            x: 0,
            y: 0,
            w: 16,
            h: 20,
        };

        // 1. 256 unique opaque colors -> `Ok(256)` and `HybridScan::Palette { n_colors: 256 }`.
        let mut buf_opaque: Vec<Vec4u8> = (0..320)
            .map(|i| Vec4u8((i % 256) as u8, 42, 99, 0xFF))
            .collect();
        assert_scan_matches_oracle(&mut scratch, &buf_opaque, 16, rect, false);
        assert_scan_matches_oracle(&mut scratch, &buf_opaque, 16, rect, true);
        let p = buf_opaque.as_ptr();
        let bp = unsafe { BufferPointer::new(&p, buf_opaque.len()) };
        assert_eq!(scratch.try_encode_2d::<false>(bp, 16, rect), Ok(256));
        assert_eq!(scratch.try_encode_2d::<true>(bp, 16, rect), Ok(256));
        assert_eq!(
            scratch.scan_hybrid_leaf::<false>(bp, 16, rect),
            HybridScan::Palette { n_colors: 256 }
        );
        assert_eq!(
            scratch.scan_hybrid_leaf::<true>(bp, 16, rect),
            HybridScan::Palette { n_colors: 256 }
        );

        // 2. 257th unique RGB color on row 19 (uniform, <35% dominant) -> `Err(19)` and `Photo`.
        buf_opaque[19 * 16 + 7] = Vec4u8(1, 2, 3, 0xFF);
        assert_scan_matches_oracle(&mut scratch, &buf_opaque, 16, rect, false);
        assert_scan_matches_oracle(&mut scratch, &buf_opaque, 16, rect, true);
        let p = buf_opaque.as_ptr();
        let bp = unsafe { BufferPointer::new(&p, buf_opaque.len()) };
        assert_eq!(scratch.try_encode_2d::<false>(bp, 16, rect), Err(19));
        assert_eq!(scratch.try_encode_2d::<true>(bp, 16, rect), Err(19));
        assert_eq!(
            scratch.scan_hybrid_leaf::<false>(bp, 16, rect),
            HybridScan::Photo
        );
        assert_eq!(
            scratch.scan_hybrid_leaf::<true>(bp, 16, rect),
            HybridScan::Photo
        );

        // 3. 256 unique translucent BGRA colors sharing 1 RGB value -> `Ok(256)` for
        //    `HAS_ALPHA = true`, `Ok(1)` for `HAS_ALPHA = false`.
        let mut buf_trans: Vec<Vec4u8> = (0..320)
            .map(|i| Vec4u8(10, 20, 30, if i < 256 { i as u8 } else { 0 }))
            .collect();
        assert_scan_matches_oracle(&mut scratch, &buf_trans, 16, rect, false);
        assert_scan_matches_oracle(&mut scratch, &buf_trans, 16, rect, true);
        let p = buf_trans.as_ptr();
        let bp = unsafe { BufferPointer::new(&p, buf_trans.len()) };
        assert_eq!(scratch.try_encode_2d::<false>(bp, 16, rect), Ok(1));
        assert_eq!(scratch.try_encode_2d::<true>(bp, 16, rect), Ok(256));

        // Adding a 257th unique BGRA (only 2nd unique RGB) on row 17: `try_encode_2d::<true>`
        // bails at row 17, `try_encode_2d::<false>` sees 2 colors, `scan_hybrid_leaf::<true>`
        // returns `TextSafeNoPalette` via `finish_translucent_rgb`.
        buf_trans[17 * 16 + 3] = Vec4u8(40, 50, 60, 0x80);
        assert_scan_matches_oracle(&mut scratch, &buf_trans, 16, rect, false);
        assert_scan_matches_oracle(&mut scratch, &buf_trans, 16, rect, true);
        let p = buf_trans.as_ptr();
        let bp = unsafe { BufferPointer::new(&p, buf_trans.len()) };
        assert_eq!(scratch.try_encode_2d::<false>(bp, 16, rect), Ok(2));
        assert_eq!(scratch.try_encode_2d::<true>(bp, 16, rect), Err(17));
        assert_eq!(
            scratch.scan_hybrid_leaf::<false>(bp, 16, rect),
            HybridScan::Palette { n_colors: 2 }
        );
        assert_eq!(
            scratch.scan_hybrid_leaf::<true>(bp, 16, rect),
            HybridScan::TextSafeNoPalette
        );
    }
}
