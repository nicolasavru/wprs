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

//! Keyed 128-bit fused SIMD tile hasher for [`crate::filtering::ServerTileCache`].
//!
//! Replaces intermediate-buffer hashing with an in-register, row-oriented 4-lane
//! absorption primitive so `copy_and_hash_rect_to_shadow` reads each `wl_shm`
//! pixel once, stores raw bytes directly into (potentially uninitialized) shadow
//! memory, optionally masks `Xrgb8888` byte 3 to `0xFF` in-register, and folds
//! pixels directly into four 128-bit state lanes without touching any scratch
//! buffer.
//!
//! Backends (digests are strictly server-local to one [`TileHashKey`] and never
//! cross the wire):
//! - `x86_64` AES-NI (`is_x86_feature_detected!("aes")`, `#[target_feature(enable = "aes")]`):
//!   4-lane `_mm_aesenc_si128(_mm_xor_si128(s_i, m_i), round_key)` with a
//!   multi-round key-separated AES finalizer.
//! - `aarch64` ARMv8 AES (`is_aarch64_feature_detected!("aes")`,
//!   `#[target_feature(enable = "aes")]`): mathematically identical 4-lane
//!   `veorq_u8(vaesmcq_u8(vaeseq_u8(s_i, m_i)), round_key)`.
//! - Portable 128-bit keyed fallback (non-AES CPUs and Miri): 4-lane 64-bit
//!   state (256-bit internal state) with widened `u128` multiply-xor-fold
//!   absorption and 4-round cross-coupled 128-bit finalization.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::mem::MaybeUninit;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::SUBTILE_W;
use crate::filtering::tile_cache::SlotScans;
use crate::serialization::framing::TileRect;
use crate::vec4u8::Vec4u8;

/// 128-bit keyed digest of canonical tile pixels or combined quadrant digests.
pub(crate) type TileHash = [u8; 16];

/// Pixel uniformity and opacity metadata computed in-register during Phase-A
/// scanning with zero extra memory passes.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct RectScanMeta {
    /// `Some([b, g, r, a])` iff the rectangle is non-empty and every pixel
    /// equals this exact canonical pixel (with byte 3 normalized to `0xFF` for
    /// `Xrgb8888`).
    pub(crate) solid_px: Option<[u8; 4]>,
    /// `true` iff every pixel in the rectangle has alpha byte `0xFF` (always
    /// `true` for `Xrgb8888` and for empty rectangles).
    pub(crate) all_opaque: bool,
}

impl RectScanMeta {
    /// Conservative metadata when exact pixel properties were not scanned for
    /// a rectangle (e.g. unhashed tiny cropped patches or direct no-shadow
    /// full frames).
    pub(crate) const UNKNOWN: Self = Self {
        solid_px: None,
        all_opaque: false,
    };

    /// Exact metadata for an empty rectangle (`w == 0 || h == 0`).
    pub(crate) const EMPTY: Self = Self {
        solid_px: None,
        all_opaque: true,
    };

    /// Combines `1..=4` child quadrant metadata entries into the exact
    /// metadata for their parent rectangle in O(1).
    #[inline]
    pub(crate) fn combine(quads: &[Self]) -> Self {
        debug_assert!(!quads.is_empty());
        let first_solid = quads[0].solid_px;
        let solid_px = first_solid.filter(|&px| quads[1..].iter().all(|q| q.solid_px == Some(px)));
        let all_opaque = quads.iter().all(|q| q.all_opaque);
        Self {
            solid_px,
            all_opaque,
        }
    }
}

/// Combined output of a single-pass Phase-A tile hash and pixel-property scan.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct TileHashResult {
    pub(crate) hash: TileHash,
    pub(crate) meta: RectScanMeta,
}

impl TileHashResult {
    pub(crate) const EMPTY: Self = Self {
        hash: [0u8; 16],
        meta: RectScanMeta::UNKNOWN,
    };
}

/// Incremental row-oriented 128-bit keyed tile hasher state (`80 B`, 16-byte
/// aligned so `30` quadrants across a 4K band fit in `< 2.4 KiB` of L1d).
///
/// Produces bit-identical [`TileHashResult`] to [`hash_rect_from_shadow`] and
/// [`copy_and_hash_rect_to_shadow`] across all backends (`x86_aes`,
/// `aarch64_aes`, `portable`) while keeping `RowBand2D` scanning strictly
/// linear row-major with zero vertical-stride re-reads.
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub(crate) struct RowTileHasher {
    lanes: [[u64; 2]; 4],
    pub(crate) first_word: u32,
    pub(crate) w: u16,
    h: u16,
    pub(crate) still_solid: bool,
    pub(crate) still_opaque: bool,
}

/// Per-quadrant row action tag for [`process_band_row`].
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum BandRowAction {
    /// Clean or below hash threshold — skip.
    Skip = 0,
    /// Active hasher, `!bbox.full`: absorb row `y` from L1d-hot `shadow`.
    AbsorbShadow = 1,
    /// Active hasher that just escalated to `bbox.full` on row `y`: absorb
    /// row `y` from L1d-hot `shadow`, then transition to `CopyAndAbsorb`.
    AbsorbShadowThenFull = 2,
    /// Active hasher, `bbox.full`: copy row `y` from `src` to `shadow` and
    /// absorb in-register in one linear pass.
    CopyAndAbsorb = 3,
    /// Newly crossed hash threshold on row `y` (`!bbox.full`): initialize
    /// hasher, absorb L2-hot rows `0..=ry` from `shadow`, then transition to
    /// `AbsorbShadow`.
    StartAbsorb = 4,
    /// Newly escalated to `bbox.full` on row `y` without a prior hasher:
    /// initialize hasher, absorb L2-hot rows `0..=ry` from `shadow`, then
    /// transition to `CopyAndAbsorb`.
    StartFull = 5,
}

const DOMAIN_LEAF: u64 = b'L' as u64;
const DOMAIN_PARENT: u64 = b'P' as u64;

pub(crate) const XRGB_ALPHA_MASK_U32: u32 = u32::from_ne_bytes([0, 0, 0, 0xFF]);

const XRGB_ALPHA_MASK_U64: u64 = u64::from_ne_bytes([0, 0, 0, 0xFF, 0, 0, 0, 0xFF]);

/// Conservative O(1) 3-point pre-probe (`(0, 0)`, `(w / 2, h / 2)`,
/// `(w - 1, h - 1)`) returning `(canonical_first_word, maybe_solid, maybe_opaque)`.
///
/// Short-circuits `read_last` whenever the first two pixels already rule out
/// both uniformity and (when `HAS_ALPHA`) full opacity, avoiding the third
/// far-stride cache line read on non-solid `Xrgb8888` and translucent `Argb8888`
/// tiles.
///
/// Always TOCTOU-sound under concurrent `wl_shm` mutation: a false negative
/// conservatively disables the fast skip (falling back to normal Phase-B
/// encoding), while a positive probe still verifies every single pixel actually
/// copied into `shadow` in-register.
#[inline(always)]
pub(crate) fn probe_three_pixels<const HAS_ALPHA: bool>(
    first_raw: u32,
    mid_raw: u32,
    read_last: impl FnOnce() -> u32,
) -> (u32, bool, bool) {
    let first_word = if HAS_ALPHA {
        first_raw
    } else {
        first_raw | XRGB_ALPHA_MASK_U32
    };
    let mid_word = if HAS_ALPHA {
        mid_raw
    } else {
        mid_raw | XRGB_ALPHA_MASK_U32
    };
    let eq01 = first_word == mid_word;
    let op01 = !HAS_ALPHA || ((first_word & mid_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32);
    if !eq01 && (!HAS_ALPHA || !op01) {
        return (first_word, false, !HAS_ALPHA);
    }
    let last_raw = read_last();
    let last_word = if HAS_ALPHA {
        last_raw
    } else {
        last_raw | XRGB_ALPHA_MASK_U32
    };
    let maybe_solid = eq01 && first_word == last_word;
    let maybe_opaque =
        op01 && (!HAS_ALPHA || (last_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32);
    (first_word, maybe_solid, maybe_opaque)
}

/// Per-connection random secret key for the 128-bit fused tile hasher.
///
/// Keyed randomly per [`crate::filtering::ServerTileCache`] via
/// [`RandomState`] so untrusted pixels cannot construct offline collisions.
#[derive(Debug, Copy, Clone)]
pub(crate) struct TileHashKey {
    /// Four independent 128-bit lane IVs.
    iv: [[u64; 2]; 4],
    /// 128-bit secret round key mixed into every absorption step.
    pub(crate) round_key: [u64; 2],
    /// Six independent 128-bit secret keys for lane folding and finalization.
    final_keys: [[u64; 2]; 6],
}

#[inline]
fn random_block() -> [u64; 2] {
    let state = RandomState::new();
    [
        state.hash_one(0x243F_6A88_85A3_08D3u64),
        state.hash_one(0x1319_8A2E_0370_7344u64),
    ]
}

impl TileHashKey {
    /// Generates a fresh per-connection random key from [`RandomState`].
    pub(crate) fn new_random() -> Self {
        Self {
            iv: [
                random_block(),
                random_block(),
                random_block(),
                random_block(),
            ],
            round_key: random_block(),
            final_keys: [
                random_block(),
                random_block(),
                random_block(),
                random_block(),
                random_block(),
                random_block(),
            ],
        }
    }

    #[cfg(test)]
    pub(crate) fn from_seed(mut state: u64) -> Self {
        let mut next_u64 = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut next_block = || [next_u64(), next_u64()];
        Self {
            iv: [next_block(), next_block(), next_block(), next_block()],
            round_key: next_block(),
            final_keys: [
                next_block(),
                next_block(),
                next_block(),
                next_block(),
                next_block(),
                next_block(),
            ],
        }
    }
}

#[inline(always)]
fn leaf_meta(w: u32, h: u32) -> (u64, u64) {
    ((u64::from(w) << 32) | u64::from(h), DOMAIN_LEAF)
}

#[inline(always)]
fn parent_meta(w: u32, h: u32, n_quads: usize) -> (u64, u64) {
    (
        (u64::from(w) << 32) | u64::from(h),
        DOMAIN_PARENT | ((n_quads as u64) << 8),
    )
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
pub(crate) fn has_hw_aes() -> bool {
    #[cfg(target_feature = "aes")]
    {
        true
    }
    #[cfg(not(target_feature = "aes"))]
    {
        std::arch::is_x86_feature_detected!("aes")
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline(always)]
fn has_hw_aes() -> bool {
    #[cfg(target_feature = "aes")]
    {
        true
    }
    #[cfg(not(target_feature = "aes"))]
    {
        std::arch::is_aarch64_feature_detected!("aes")
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub(crate) mod x86_aes {
    use std::arch::x86_64::__m128i;
    use std::arch::x86_64::_mm_aesenc_si128;
    use std::arch::x86_64::_mm_and_si128;
    use std::arch::x86_64::_mm_cmpeq_epi8;
    use std::arch::x86_64::_mm_cvtsi32_si128;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm_movemask_epi8;
    use std::arch::x86_64::_mm_or_si128;
    use std::arch::x86_64::_mm_set_epi32;
    use std::arch::x86_64::_mm_set_epi64x;
    use std::arch::x86_64::_mm_set1_epi8;
    use std::arch::x86_64::_mm_set1_epi32;
    use std::arch::x86_64::_mm_setzero_si128;
    use std::arch::x86_64::_mm_storeu_si128;
    use std::arch::x86_64::_mm_xor_si128;
    use std::mem::MaybeUninit;
    use std::ptr;

    use super::BandRowAction;
    use super::RectScanMeta;
    use super::RowTileHasher;
    use super::SUBTILE_W;
    use super::SlotScans;
    use super::TileHash;
    use super::TileHashKey;
    use super::TileHashResult;
    use super::XRGB_ALPHA_MASK_U32;
    use super::leaf_meta;
    use super::parent_meta;
    use super::probe_three_pixels;
    use crate::serialization::framing::TileRect;

    #[inline]
    #[target_feature(enable = "aes")]
    pub(crate) fn load_key_block(block: &[u64; 2]) -> __m128i {
        // SAFETY: `[u64; 2]` is 16 readable initialized bytes.
        unsafe { _mm_loadu_si128(block.as_ptr().cast::<__m128i>()) }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn init_lanes(key: &TileHashKey, meta_lo: u64, meta_hi: u64) -> ([__m128i; 4], __m128i) {
        let meta = _mm_set_epi64x(meta_hi as i64, meta_lo as i64);
        let lanes = [
            _mm_xor_si128(load_key_block(&key.iv[0]), meta),
            _mm_xor_si128(load_key_block(&key.iv[1]), meta),
            _mm_xor_si128(load_key_block(&key.iv[2]), meta),
            _mm_xor_si128(load_key_block(&key.iv[3]), meta),
        ];
        (lanes, meta)
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn finalize(key: &TileHashKey, lanes: [__m128i; 4], meta: __m128i) -> TileHash {
        let fk0 = load_key_block(&key.final_keys[0]);
        let fk1 = load_key_block(&key.final_keys[1]);
        let fk2 = load_key_block(&key.final_keys[2]);
        let fk3 = load_key_block(&key.final_keys[3]);
        let fk4 = load_key_block(&key.final_keys[4]);
        let fk5 = load_key_block(&key.final_keys[5]);

        // Separate lanes with distinct keys before folding so equal lanes never
        // cancel symmetrically.
        let s0 = _mm_aesenc_si128(lanes[0], fk0);
        let s1 = _mm_aesenc_si128(lanes[1], fk1);
        let s2 = _mm_aesenc_si128(lanes[2], fk2);
        let s3 = _mm_aesenc_si128(lanes[3], fk3);

        let t0 = _mm_aesenc_si128(_mm_xor_si128(s0, s1), fk4);
        let t1 = _mm_aesenc_si128(_mm_xor_si128(s2, s3), fk5);

        let mut h = _mm_xor_si128(_mm_xor_si128(t0, t1), meta);
        h = _mm_aesenc_si128(h, fk0);
        h = _mm_aesenc_si128(h, fk1);
        h = _mm_aesenc_si128(h, fk2);
        h = _mm_aesenc_si128(h, fk3);
        h = _mm_aesenc_si128(h, fk4);
        h = _mm_aesenc_si128(h, fk5);

        let mut out = [0u8; 16];
        // SAFETY: `out` is 16 writable bytes.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), h) };
        out
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn alpha_lanes_all_opaque(aand0: __m128i, aand1: __m128i) -> bool {
        let combined = _mm_and_si128(aand0, aand1);
        (_mm_movemask_epi8(_mm_cmpeq_epi8(combined, _mm_set1_epi8(-1))) & 0x8888) == 0x8888
    }

    /// Absorbs one row (`full64 + rem` bytes, `rem < 64` and a multiple of 4)
    /// into `lanes`, optionally storing raw bytes into `dst_row` (`COPY`),
    /// tracking solid-color differences (`TRACK_SOLID`), and/or tracking alpha
    /// opacity (`TRACK_OPAQUE`). Keeps a single compact 64B inner loop so LLVM
    /// chooses optimal unrolling per specialization and fits cleanly in L1i /
    /// µop cache.
    ///
    /// # Safety
    /// `src_row` must be readable for `full64 + rem` bytes. When `COPY` is
    /// `true`, `dst_row` must be writable for `full64 + rem` bytes and disjoint
    /// from `src_row`.
    #[inline]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn absorb_row<
        const HAS_ALPHA: bool,
        const COPY: bool,
        const TRACK_SOLID: bool,
        const TRACK_OPAQUE: bool,
    >(
        src_row: *const u8,
        dst_row: *mut u8,
        full64: usize,
        rem: usize,
        rk: __m128i,
        alpha_mask: __m128i,
        first_word: u32,
        first_px: __m128i,
        lanes: &mut [__m128i; 4],
        dacc: &mut __m128i,
        aand0: &mut __m128i,
        aand1: &mut __m128i,
    ) {
        let mut s0 = lanes[0];
        let mut s1 = lanes[1];
        let mut s2 = lanes[2];
        let mut s3 = lanes[3];
        let mut cur_dacc = *dacc;
        let mut a0 = *aand0;
        let mut a1 = *aand1;

        let mut off = 0usize;
        while off < full64 {
            // SAFETY: `off + 64 <= full64`, within caller-validated row bounds.
            let (r0, r1, r2, r3) = unsafe {
                let r0 = _mm_loadu_si128(src_row.add(off).cast::<__m128i>());
                let r1 = _mm_loadu_si128(src_row.add(off + 16).cast::<__m128i>());
                let r2 = _mm_loadu_si128(src_row.add(off + 32).cast::<__m128i>());
                let r3 = _mm_loadu_si128(src_row.add(off + 48).cast::<__m128i>());
                if COPY {
                    _mm_storeu_si128(dst_row.add(off).cast::<__m128i>(), r0);
                    _mm_storeu_si128(dst_row.add(off + 16).cast::<__m128i>(), r1);
                    _mm_storeu_si128(dst_row.add(off + 32).cast::<__m128i>(), r2);
                    _mm_storeu_si128(dst_row.add(off + 48).cast::<__m128i>(), r3);
                }
                (r0, r1, r2, r3)
            };
            let m0 = if HAS_ALPHA {
                r0
            } else {
                _mm_or_si128(r0, alpha_mask)
            };
            let m1 = if HAS_ALPHA {
                r1
            } else {
                _mm_or_si128(r1, alpha_mask)
            };
            let m2 = if HAS_ALPHA {
                r2
            } else {
                _mm_or_si128(r2, alpha_mask)
            };
            let m3 = if HAS_ALPHA {
                r3
            } else {
                _mm_or_si128(r3, alpha_mask)
            };
            if TRACK_SOLID {
                let d01 = _mm_or_si128(_mm_xor_si128(m0, first_px), _mm_xor_si128(m1, first_px));
                let d23 = _mm_or_si128(_mm_xor_si128(m2, first_px), _mm_xor_si128(m3, first_px));
                cur_dacc = _mm_or_si128(cur_dacc, _mm_or_si128(d01, d23));
            }
            if TRACK_OPAQUE {
                a0 = _mm_and_si128(a0, m0);
                a1 = _mm_and_si128(a1, m1);
                a0 = _mm_and_si128(a0, m2);
                a1 = _mm_and_si128(a1, m3);
            }
            s0 = _mm_aesenc_si128(_mm_xor_si128(s0, m0), rk);
            s1 = _mm_aesenc_si128(_mm_xor_si128(s1, m1), rk);
            s2 = _mm_aesenc_si128(_mm_xor_si128(s2, m2), rk);
            s3 = _mm_aesenc_si128(_mm_xor_si128(s3, m3), rk);
            off += 64;
        }

        if rem != 0 {
            let mut left = rem;
            if left >= 16 {
                // SAFETY: `off + 16 <= full64 + rem`.
                let r0 = unsafe {
                    let v = _mm_loadu_si128(src_row.add(off).cast::<__m128i>());
                    if COPY {
                        _mm_storeu_si128(dst_row.add(off).cast::<__m128i>(), v);
                    }
                    v
                };
                let m0 = if HAS_ALPHA {
                    r0
                } else {
                    _mm_or_si128(r0, alpha_mask)
                };
                if TRACK_SOLID {
                    cur_dacc = _mm_or_si128(cur_dacc, _mm_xor_si128(m0, first_px));
                }
                if TRACK_OPAQUE {
                    a0 = _mm_and_si128(a0, m0);
                }
                s0 = _mm_aesenc_si128(_mm_xor_si128(s0, m0), rk);
                off += 16;
                left -= 16;

                if left >= 16 {
                    // SAFETY: `off + 16 <= full64 + rem`.
                    let r1 = unsafe {
                        let v = _mm_loadu_si128(src_row.add(off).cast::<__m128i>());
                        if COPY {
                            _mm_storeu_si128(dst_row.add(off).cast::<__m128i>(), v);
                        }
                        v
                    };
                    let m1 = if HAS_ALPHA {
                        r1
                    } else {
                        _mm_or_si128(r1, alpha_mask)
                    };
                    if TRACK_SOLID {
                        cur_dacc = _mm_or_si128(cur_dacc, _mm_xor_si128(m1, first_px));
                    }
                    if TRACK_OPAQUE {
                        a1 = _mm_and_si128(a1, m1);
                    }
                    s1 = _mm_aesenc_si128(_mm_xor_si128(s1, m1), rk);
                    off += 16;
                    left -= 16;

                    if left >= 16 {
                        // SAFETY: `off + 16 <= full64 + rem`.
                        let r2 = unsafe {
                            let v = _mm_loadu_si128(src_row.add(off).cast::<__m128i>());
                            if COPY {
                                _mm_storeu_si128(dst_row.add(off).cast::<__m128i>(), v);
                            }
                            v
                        };
                        let m2 = if HAS_ALPHA {
                            r2
                        } else {
                            _mm_or_si128(r2, alpha_mask)
                        };
                        if TRACK_SOLID {
                            cur_dacc = _mm_or_si128(cur_dacc, _mm_xor_si128(m2, first_px));
                        }
                        if TRACK_OPAQUE {
                            a0 = _mm_and_si128(a0, m2);
                        }
                        s2 = _mm_aesenc_si128(_mm_xor_si128(s2, m2), rk);
                        off += 16;
                        left -= 16;
                    }
                }
            }

            if left > 0 {
                debug_assert!(left < 16 && left.is_multiple_of(4));
                // SAFETY: `off + left == full64 + rem`, `left` is 4, 8, or 12.
                let (p0, p1, p2) = unsafe {
                    let p0 = src_row.add(off).cast::<u32>().read_unaligned();
                    if COPY {
                        dst_row.add(off).cast::<u32>().write_unaligned(p0);
                    }
                    let mut p1 = 0u32;
                    let mut p2 = 0u32;
                    if left >= 8 {
                        p1 = src_row.add(off + 4).cast::<u32>().read_unaligned();
                        if COPY {
                            dst_row.add(off + 4).cast::<u32>().write_unaligned(p1);
                        }
                        if left == 12 {
                            p2 = src_row.add(off + 8).cast::<u32>().read_unaligned();
                            if COPY {
                                dst_row.add(off + 8).cast::<u32>().write_unaligned(p2);
                            }
                        }
                    }
                    (p0, p1, p2)
                };
                if TRACK_SOLID || TRACK_OPAQUE {
                    let c0 = if HAS_ALPHA {
                        p0
                    } else {
                        p0 | XRGB_ALPHA_MASK_U32
                    };
                    let mut td = c0 ^ first_word;
                    let mut ta = c0;
                    if left >= 8 {
                        let c1 = if HAS_ALPHA {
                            p1
                        } else {
                            p1 | XRGB_ALPHA_MASK_U32
                        };
                        td |= c1 ^ first_word;
                        if TRACK_OPAQUE {
                            ta &= c1;
                        }
                        if left == 12 {
                            let c2 = if HAS_ALPHA {
                                p2
                            } else {
                                p2 | XRGB_ALPHA_MASK_U32
                            };
                            td |= c2 ^ first_word;
                            if TRACK_OPAQUE {
                                ta &= c2;
                            }
                        }
                    }
                    if TRACK_SOLID {
                        cur_dacc = _mm_or_si128(cur_dacc, _mm_cvtsi32_si128(td as i32));
                    }
                    if TRACK_OPAQUE {
                        a0 = _mm_and_si128(a0, _mm_set1_epi32(ta as i32));
                    }
                }

                let rt = _mm_set_epi32(0, p2 as i32, p1 as i32, p0 as i32);
                let mt = if HAS_ALPHA {
                    rt
                } else {
                    _mm_or_si128(rt, alpha_mask)
                };
                s3 = _mm_aesenc_si128(_mm_xor_si128(s3, mt), rk);
            }
        }

        lanes[0] = s0;
        lanes[1] = s1;
        lanes[2] = s2;
        lanes[3] = s3;
        if TRACK_SOLID {
            *dacc = cur_dacc;
        }
        if TRACK_OPAQUE {
            *aand0 = a0;
            *aand1 = a1;
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "aes")]
    fn empty_rect_result(key: &TileHashKey, rect: TileRect) -> TileHashResult {
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta::EMPTY,
        }
    }

    /// Expands the non-solid row loop inline inside a
    /// `#[target_feature(enable = "aes")]` function so LLVM never outlines it
    /// across a call boundary or spills the 4 AES lane registers to stack.
    macro_rules! x86_absorb_rows_fast {
        (
            $src_start:expr, $dst_start:expr, $n_rows:expr, $stride:expr,
            $full64:expr, $rem:expr, $has_alpha:expr, $copy:expr,
            $rk:expr, $alpha_mask:expr, $lanes:expr, $still_opaque:expr
        ) => {{
            let mut src_ptr: *const u8 = $src_start;
            let mut dst_ptr: *mut u8 = $dst_start;
            let n_rows: usize = $n_rows;
            let mut dummy_dacc = _mm_setzero_si128();
            let zero_px = _mm_setzero_si128();
            let row_bytes = $full64 + $rem;
            let pf_rows = $copy
                && $stride > row_bytes
                && $stride >= crate::filtering::prefetch::MIN_PAGE_STRIDE_BYTES;
            if $has_alpha && $still_opaque {
                let mut aand0 = _mm_set1_epi8(-1);
                let mut aand1 = _mm_set1_epi8(-1);
                for ry in 0..n_rows {
                    if pf_rows && ry + crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD < n_rows
                    {
                        crate::filtering::prefetch::row_nta(
                            src_ptr
                                .add(crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD * $stride),
                            row_bytes,
                        );
                    }
                    absorb_row::<true, $copy, false, true>(
                        src_ptr,
                        dst_ptr,
                        $full64,
                        $rem,
                        $rk,
                        $alpha_mask,
                        0,
                        zero_px,
                        $lanes,
                        &mut dummy_dacc,
                        &mut aand0,
                        &mut aand1,
                    );
                    src_ptr = src_ptr.add($stride);
                    if $copy {
                        dst_ptr = dst_ptr.add($stride);
                    }
                }
                alpha_lanes_all_opaque(aand0, aand1)
            } else {
                let mut dummy_a0 = _mm_setzero_si128();
                let mut dummy_a1 = _mm_setzero_si128();
                for ry in 0..n_rows {
                    if pf_rows && ry + crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD < n_rows
                    {
                        crate::filtering::prefetch::row_nta(
                            src_ptr
                                .add(crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD * $stride),
                            row_bytes,
                        );
                    }
                    absorb_row::<$has_alpha, $copy, false, false>(
                        src_ptr,
                        dst_ptr,
                        $full64,
                        $rem,
                        $rk,
                        $alpha_mask,
                        0,
                        zero_px,
                        $lanes,
                        &mut dummy_dacc,
                        &mut dummy_a0,
                        &mut dummy_a1,
                    );
                    src_ptr = src_ptr.add($stride);
                    if $copy {
                        dst_ptr = dst_ptr.add($stride);
                    }
                }
                !$has_alpha
            }
        }};
    }

    /// Expands the solid-candidate row scan inline inside a
    /// `#[target_feature(enable = "aes")]` function so solid-candidate
    /// quadrants never pay a second call boundary or spill the 4 AES lane
    /// registers to stack. Tracks pixel uniformity row-by-row and drops
    /// immediately into `x86_absorb_rows_fast!` if any row contains a differing
    /// pixel.
    macro_rules! x86_absorb_rows_solid_then_fast {
        (
            $src_start:expr, $dst_start:expr, $n_rows:expr, $stride:expr,
            $full64:expr, $rem:expr, $has_alpha:expr, $copy:expr,
            $rk:expr, $alpha_mask:expr, $first_word:expr, $lanes:expr,
            $still_opaque_in:expr
        ) => {{
            let mut src_ptr: *const u8 = $src_start;
            let mut dst_ptr: *mut u8 = $dst_start;
            let n_rows: usize = $n_rows;
            let first_word: u32 = $first_word;
            let first_px = _mm_set1_epi32(first_word as i32);
            let mut dacc = _mm_setzero_si128();
            let mut aand0 = _mm_set1_epi8(-1);
            let mut aand1 = _mm_set1_epi8(-1);
            let mut ry = 0usize;
            let mut still_solid = true;
            let mut still_opaque: bool = $still_opaque_in;
            while ry < n_rows {
                absorb_row::<$has_alpha, $copy, true, $has_alpha>(
                    src_ptr,
                    dst_ptr,
                    $full64,
                    $rem,
                    $rk,
                    $alpha_mask,
                    first_word,
                    first_px,
                    $lanes,
                    &mut dacc,
                    &mut aand0,
                    &mut aand1,
                );
                src_ptr = src_ptr.add($stride);
                if $copy {
                    dst_ptr = dst_ptr.add($stride);
                }
                ry += 1;
                if _mm_movemask_epi8(_mm_cmpeq_epi8(dacc, _mm_setzero_si128())) != 0xFFFF {
                    still_solid = false;
                    if $has_alpha && still_opaque {
                        still_opaque = alpha_lanes_all_opaque(aand0, aand1);
                    }
                    break;
                }
            }
            if ry < n_rows {
                still_opaque = x86_absorb_rows_fast!(
                    src_ptr,
                    dst_ptr,
                    n_rows - ry,
                    $stride,
                    $full64,
                    $rem,
                    $has_alpha,
                    $copy,
                    $rk,
                    $alpha_mask,
                    $lanes,
                    still_opaque
                );
            }
            (still_solid, !$has_alpha || still_opaque)
        }};
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn hash_rect_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        row_ptr: *const u8,
        stride: usize,
        w_h: u64,
        first_word: u32,
    ) -> TileHashResult {
        let w = (w_h >> 32) as u32;
        let h = w_h as u32;
        let (meta_lo, meta_hi) = leaf_meta(w, h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let row_bytes = w as usize * 4;
        let th = h as usize;
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;
        // SAFETY: Caller guarantees `row_ptr` is readable for `th` rows of `row_bytes`.
        let (still_solid, all_opaque) = unsafe {
            x86_absorb_rows_solid_then_fast!(
                row_ptr,
                ptr::null_mut(),
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_ptr: *const u8,
        sh_ptr: *mut u8,
        stride: usize,
        w_h: u64,
    ) -> TileHashResult {
        let w = (w_h >> 32) as u32;
        let h = w_h as u32;
        let (meta_lo, meta_hi) = leaf_meta(w, h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let row_bytes = w as usize * 4;
        let th = h as usize;
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        // Re-reading `(0, 0)` (already hot in L1 from the caller's pre-probe)
        // keeps this function at 5 explicit args (+ sret = 6 register args) so
        // `copy_and_hash_rect_to_shadow` can emit a zero-overhead tail-call
        // `jmp`. Still TOCTOU-sound: every copied pixel is verified against
        // `first_word` in-register.
        // SAFETY: Caller guarantees `src_ptr` is readable for `th > 0` rows of `w > 0` pixels.
        let first_raw = unsafe { src_ptr.cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;
        // SAFETY: Caller guarantees `src_ptr` readable and `sh_ptr` writable for `th` rows.
        let (still_solid, all_opaque) = unsafe {
            x86_absorb_rows_solid_then_fast!(
                src_ptr,
                sh_ptr,
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                true,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_with_overlap_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;
        // SAFETY: Caller guarantees `copy_from_ry > 0`, `th > 0 && tw > 0`,
        // `shadow_base` initialized for rows `0..copy_from_ry`, and `src_base`
        // readable / `shadow_base` writable for rows `copy_from_ry..th`.
        let src_ptr_base = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let first_raw = unsafe { sh_ptr.cast_const().cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;

        // SAFETY: Caller guarantees `sh_ptr` is initialized and readable for `0..copy_from_ry`.
        let (mut still_solid, mut all_opaque) = unsafe {
            x86_absorb_rows_solid_then_fast!(
                sh_ptr.cast_const(),
                ptr::null_mut(),
                copy_from_ry,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };

        if copy_from_ry < th {
            // SAFETY: Caller guarantees `src_ptr_base` readable and `sh_ptr` writable
            // for rows `copy_from_ry..th`.
            let src_ptr = unsafe { src_ptr_base.add(copy_from_ry * stride) };
            let dst_ptr = unsafe { sh_ptr.add(copy_from_ry * stride) };
            let rem_rows = th - copy_from_ry;
            if still_solid {
                (still_solid, all_opaque) = unsafe {
                    x86_absorb_rows_solid_then_fast!(
                        src_ptr, dst_ptr, rem_rows, stride, full64, rem, HAS_ALPHA, true, rk,
                        alpha_mask, first_word, &mut lanes, all_opaque
                    )
                };
            } else {
                all_opaque = unsafe {
                    x86_absorb_rows_fast!(
                        src_ptr, dst_ptr, rem_rows, stride, full64, rem, HAS_ALPHA, true, rk,
                        alpha_mask, &mut lanes, all_opaque
                    )
                };
            }
        }

        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_with_overlap<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;
        // SAFETY: Caller guarantees `copy_from_ry > 0`, `shadow_base` rows
        // `0..copy_from_ry` are initialized and readable, and `src_base` rows
        // `copy_from_ry..th` are readable, with `th > 0 && tw > 0`.
        let src_ptr_base = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let read_probe = |ry: usize, rx: usize| -> u32 {
            let off = ry * stride + rx * 4;
            unsafe {
                let p = if ry < copy_from_ry {
                    sh_ptr.cast_const().add(off)
                } else {
                    src_ptr_base.add(off)
                };
                p.cast::<u32>().read_unaligned()
            }
        };
        let first_raw = unsafe { sh_ptr.cast_const().cast::<u32>().read_unaligned() };
        let mid_raw = read_probe(th >> 1, tw >> 1);
        let (_, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || read_probe(th - 1, tw - 1));
        if still_solid {
            // SAFETY: Bounds and exclusivity invariants forwarded from caller.
            return unsafe {
                copy_and_hash_with_overlap_solid_candidate::<HAS_ALPHA>(
                    key,
                    src_base,
                    shadow_base,
                    stride,
                    rect,
                    copy_from_ry,
                )
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;

        // SAFETY: `sh_ptr` is initialized and readable for `copy_from_ry` rows.
        let mut all_opaque = unsafe {
            x86_absorb_rows_fast!(
                sh_ptr.cast_const(),
                ptr::null_mut(),
                copy_from_ry,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        if copy_from_ry < th {
            // SAFETY: Caller validated `src_base` bounds and exclusive
            // `shadow_base` write access for rows `copy_from_ry..th`.
            all_opaque = unsafe {
                x86_absorb_rows_fast!(
                    src_ptr_base.add(copy_from_ry * stride),
                    sh_ptr.add(copy_from_ry * stride),
                    th - copy_from_ry,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    true,
                    rk,
                    alpha_mask,
                    &mut lanes,
                    all_opaque
                )
            };
        }
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    /// # Safety
    /// Caller must verify CPU AES-NI support and uphold [`super::hash_rect_from_shadow`]
    /// bounds invariants.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn hash_rect_raw<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        base: *const u8,
        stride: usize,
        rect: TileRect,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return empty_rect_result(key, rect);
        }
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        // SAFETY: Caller guarantees `(y0 + ry, x_bytes..x_bytes + row_bytes)` is readable
        // and `th > 0 && tw > 0`, so `(0, 0)`, `(tw/2, th/2)`, and `(tw-1, th-1)` are in-bounds.
        let row_ptr = unsafe { base.add(y0 * stride + x_bytes) };
        let (first_raw, mid_raw) = unsafe {
            (
                row_ptr.cast::<u32>().read_unaligned(),
                row_ptr
                    .add((th >> 1) * stride + (tw >> 1) * 4)
                    .cast::<u32>()
                    .read_unaligned(),
            )
        };
        let (first_word, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
                row_ptr
                    .add((th - 1) * stride + (tw - 1) * 4)
                    .cast::<u32>()
                    .read_unaligned()
            });
        let w_h = (u64::from(rect.w) << 32) | u64::from(rect.h);
        if still_solid {
            // SAFETY: `row_ptr` is readable for `th` rows of `row_bytes`.
            return unsafe {
                hash_rect_solid_candidate::<HAS_ALPHA>(key, row_ptr, stride, w_h, first_word)
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        // SAFETY: `row_ptr` is readable for `th` rows of `full64 + rem` bytes.
        let all_opaque = unsafe {
            x86_absorb_rows_fast!(
                row_ptr,
                ptr::null_mut(),
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    /// # Safety
    /// Caller must verify CPU AES-NI support and uphold
    /// [`super::copy_and_hash_rect_to_shadow`] bounds and exclusivity invariants.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn copy_and_hash_rect_to_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return empty_rect_result(key, rect);
        }
        if copy_from_ry > 0 {
            // SAFETY: Bounds and exclusivity invariants forwarded from caller.
            return unsafe {
                copy_and_hash_with_overlap::<HAS_ALPHA>(
                    key,
                    src_base,
                    shadow_base,
                    stride,
                    rect,
                    copy_from_ry,
                )
            };
        }
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;

        // SAFETY: `copy_from_ry == 0` and `th > 0 && tw > 0`, so `src_base` is
        // readable and `shadow_base` is exclusively writable for all `th` rows.
        let src_ptr = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let (first_raw, mid_raw) = unsafe {
            (
                src_ptr.cast::<u32>().read_unaligned(),
                src_ptr
                    .add((th >> 1) * stride + (tw >> 1) * 4)
                    .cast::<u32>()
                    .read_unaligned(),
            )
        };
        let (_, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
                src_ptr
                    .add((th - 1) * stride + (tw - 1) * 4)
                    .cast::<u32>()
                    .read_unaligned()
            });
        let w_h = (u64::from(rect.w) << 32) | u64::from(rect.h);
        if still_solid {
            // SAFETY: `src_ptr` readable and `sh_ptr` writable for `th` rows.
            return unsafe {
                copy_and_hash_solid_candidate::<HAS_ALPHA>(key, src_ptr, sh_ptr, stride, w_h)
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;

        // SAFETY: Caller validated `src_base` bounds and exclusive `shadow_base`
        // write access for all `th` rows.
        let all_opaque = unsafe {
            x86_absorb_rows_fast!(
                src_ptr,
                sh_ptr,
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                true,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn load_hasher_lanes(h: &RowTileHasher) -> [__m128i; 4] {
        // SAFETY: `h.lanes` is `[ [u64; 2]; 4 ]` (64 readable bytes, 16-byte aligned).
        unsafe {
            [
                _mm_loadu_si128(h.lanes[0].as_ptr().cast::<__m128i>()),
                _mm_loadu_si128(h.lanes[1].as_ptr().cast::<__m128i>()),
                _mm_loadu_si128(h.lanes[2].as_ptr().cast::<__m128i>()),
                _mm_loadu_si128(h.lanes[3].as_ptr().cast::<__m128i>()),
            ]
        }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn store_hasher_lanes(h: &mut RowTileHasher, lanes: [__m128i; 4]) {
        // SAFETY: `h.lanes` is `[ [u64; 2]; 4 ]` (64 writable bytes, 16-byte aligned).
        unsafe {
            _mm_storeu_si128(h.lanes[0].as_mut_ptr().cast::<__m128i>(), lanes[0]);
            _mm_storeu_si128(h.lanes[1].as_mut_ptr().cast::<__m128i>(), lanes[1]);
            _mm_storeu_si128(h.lanes[2].as_mut_ptr().cast::<__m128i>(), lanes[2]);
            _mm_storeu_si128(h.lanes[3].as_mut_ptr().cast::<__m128i>(), lanes[3]);
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn start_row_hasher_from_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        shadow_base: *const u8,
        stride: usize,
        x_bytes: usize,
        y0: usize,
        cw: usize,
        ch: usize,
        absorbed_rows: usize,
    ) -> RowTileHasher {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let row_bytes = cw * 4;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let (meta_lo, meta_hi) = leaf_meta(cw as u32, ch as u32);
        let (mut lanes, _) = init_lanes(key, meta_lo, meta_hi);

        // SAFETY: Caller guarantees `absorbed_rows > 0`, `cw > 0`, and
        // `[y0..y0 + absorbed_rows)` is readable for `row_bytes` at `x_bytes`.
        let sh_ptr = unsafe { shadow_base.add(y0 * stride + x_bytes) };
        let first_raw = unsafe { sh_ptr.cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let probe_ry = absorbed_rows - 1;
        let probe_rx = if absorbed_rows > 1 { cw >> 1 } else { cw - 1 };
        let probe_raw = unsafe {
            sh_ptr
                .add(probe_ry * stride + probe_rx * 4)
                .cast::<u32>()
                .read_unaligned()
        };
        let probe_word = if HAS_ALPHA {
            probe_raw
        } else {
            probe_raw | XRGB_ALPHA_MASK_U32
        };
        let maybe_solid = first_word == probe_word;
        let maybe_opaque =
            !HAS_ALPHA || ((first_word & probe_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32);

        let (still_solid, still_opaque) = if maybe_solid {
            // SAFETY: `sh_ptr` is readable for `absorbed_rows` rows of `row_bytes`.
            unsafe {
                x86_absorb_rows_solid_then_fast!(
                    sh_ptr,
                    ptr::null_mut(),
                    absorbed_rows,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    false,
                    rk,
                    alpha_mask,
                    first_word,
                    &mut lanes,
                    maybe_opaque
                )
            }
        } else {
            // SAFETY: `sh_ptr` is readable for `absorbed_rows` rows of `row_bytes`.
            let opq = unsafe {
                x86_absorb_rows_fast!(
                    sh_ptr,
                    ptr::null_mut(),
                    absorbed_rows,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    false,
                    rk,
                    alpha_mask,
                    &mut lanes,
                    maybe_opaque
                )
            };
            (false, opq)
        };

        let mut out = RowTileHasher {
            lanes: [[0; 2]; 4],
            first_word,
            w: cw as u16,
            h: ch as u16,
            still_solid,
            still_opaque,
        };
        // SAFETY: `out.lanes` is valid for 64 bytes.
        unsafe { store_hasher_lanes(&mut out, lanes) };
        out
    }

    /// Expands one quadrant row absorption inline inside a
    /// `#[target_feature(enable = "aes")]` loop so LLVM never outlines a
    /// per-quadrant call or spills `rk`/`alpha_mask` across quadrants.
    macro_rules! x86_absorb_one_hasher_row {
        ($has_alpha:expr, $copy:expr, $h:expr, $src_row:expr, $dst_row:expr, $rk:expr, $alpha_mask:expr) => {{
            let h: &mut crate::filtering::tile_hash::RowTileHasher = $h;
            let src_row: *const u8 = $src_row;
            let dst_row: *mut u8 = $dst_row;
            let rk = $rk;
            let alpha_mask = $alpha_mask;
            let row_bytes = usize::from(h.w) * 4;
            let full64 = row_bytes & !63;
            let rem = row_bytes & 63;
            let mut lanes = crate::filtering::tile_hash::x86_aes::load_hasher_lanes(h);
            if !h.still_solid {
                let mut dummy_dacc = std::arch::x86_64::_mm_setzero_si128();
                let zero_px = std::arch::x86_64::_mm_setzero_si128();
                if $has_alpha && h.still_opaque {
                    let mut aand0 = std::arch::x86_64::_mm_set1_epi8(-1);
                    let mut aand1 = std::arch::x86_64::_mm_set1_epi8(-1);
                    crate::filtering::tile_hash::x86_aes::absorb_row::<true, $copy, false, true>(
                        src_row,
                        dst_row,
                        full64,
                        rem,
                        rk,
                        alpha_mask,
                        0,
                        zero_px,
                        &mut lanes,
                        &mut dummy_dacc,
                        &mut aand0,
                        &mut aand1,
                    );
                    h.still_opaque =
                        crate::filtering::tile_hash::x86_aes::alpha_lanes_all_opaque(aand0, aand1);
                } else {
                    let mut dummy_a0 = std::arch::x86_64::_mm_setzero_si128();
                    let mut dummy_a1 = std::arch::x86_64::_mm_setzero_si128();
                    crate::filtering::tile_hash::x86_aes::absorb_row::<
                        $has_alpha,
                        $copy,
                        false,
                        false,
                    >(
                        src_row,
                        dst_row,
                        full64,
                        rem,
                        rk,
                        alpha_mask,
                        0,
                        zero_px,
                        &mut lanes,
                        &mut dummy_dacc,
                        &mut dummy_a0,
                        &mut dummy_a1,
                    );
                }
            } else {
                let first_word = h.first_word;
                let first_px = std::arch::x86_64::_mm_set1_epi32(first_word as i32);
                let mut dacc = std::arch::x86_64::_mm_setzero_si128();
                let mut aand0 = std::arch::x86_64::_mm_set1_epi8(-1);
                let mut aand1 = std::arch::x86_64::_mm_set1_epi8(-1);
                crate::filtering::tile_hash::x86_aes::absorb_row::<
                    $has_alpha,
                    $copy,
                    true,
                    $has_alpha,
                >(
                    src_row, dst_row, full64, rem, rk, alpha_mask, first_word, first_px,
                    &mut lanes, &mut dacc, &mut aand0, &mut aand1,
                );
                if std::arch::x86_64::_mm_movemask_epi8(std::arch::x86_64::_mm_cmpeq_epi8(
                    dacc,
                    std::arch::x86_64::_mm_setzero_si128(),
                )) != 0xFFFF
                {
                    h.still_solid = false;
                    if $has_alpha && h.still_opaque {
                        h.still_opaque =
                            crate::filtering::tile_hash::x86_aes::alpha_lanes_all_opaque(
                                aand0, aand1,
                            );
                    }
                }
            }
            crate::filtering::tile_hash::x86_aes::store_hasher_lanes(h, lanes);
        }};
    }
    pub(crate) use x86_absorb_one_hasher_row;

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn process_band_row<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y0: usize,
        ch: usize,
        y: usize,
        actions: &mut [BandRowAction],
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        let row_off = y * stride;
        let absorbed_rows = (y - y0) + 1;
        // SAFETY: Caller validated row `y` bounds in `src_base` and `shadow_base`.
        let src_row_base = unsafe { src_base.add(row_off) };
        let sh_row_base = unsafe { shadow_base.add(row_off) };

        let mut x = 0usize;
        for (act, slot_h) in actions.iter_mut().zip(hashers.iter_mut()) {
            let cw = (width - x).min(SUBTILE_W);
            let x_bytes = x * 4;
            match *act {
                BandRowAction::Skip => {},
                BandRowAction::AbsorbShadow | BandRowAction::AbsorbShadowThenFull => {
                    if *act == BandRowAction::AbsorbShadowThenFull {
                        *act = BandRowAction::CopyAndAbsorb;
                    }
                    // SAFETY: Non-`Skip` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        x86_absorb_one_hasher_row!(
                            HAS_ALPHA,
                            false,
                            h,
                            sh_row_base.cast_const().add(x_bytes),
                            ptr::null_mut(),
                            rk,
                            alpha_mask
                        );
                    }
                },
                BandRowAction::CopyAndAbsorb => {
                    // SAFETY: `CopyAndAbsorb` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        x86_absorb_one_hasher_row!(
                            HAS_ALPHA,
                            true,
                            h,
                            src_row_base.add(x_bytes),
                            sh_row_base.add(x_bytes),
                            rk,
                            alpha_mask
                        );
                    }
                },
                BandRowAction::StartAbsorb | BandRowAction::StartFull => {
                    let next = if *act == BandRowAction::StartFull {
                        BandRowAction::CopyAndAbsorb
                    } else {
                        BandRowAction::AbsorbShadow
                    };
                    // SAFETY: Rows `0..absorbed_rows` are finalized in `shadow_base`.
                    let h = unsafe {
                        start_row_hasher_from_shadow::<HAS_ALPHA>(
                            key,
                            shadow_base.cast_const(),
                            stride,
                            x_bytes,
                            y0,
                            cw,
                            ch,
                            absorbed_rows,
                        )
                    };
                    slot_h.write(h);
                    *act = next;
                },
            }
            x += cw;
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn copy_and_absorb_band_rows<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y_start: usize,
        y_end: usize,
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = _mm_set1_epi32(XRGB_ALPHA_MASK_U32 as i32);
        for y in y_start..y_end {
            let row_off = y * stride;
            // SAFETY: Caller validated `[y_start, y_end)` bounds and exclusive shadow access.
            let src_row_base = unsafe { src_base.add(row_off) };
            let sh_row_base = unsafe { shadow_base.add(row_off) };
            let mut x = 0usize;
            for slot_h in hashers.iter_mut() {
                let cw = (width - x).min(SUBTILE_W);
                let x_bytes = x * 4;
                // SAFETY: Caller guarantees every slot in `hashers` is initialized.
                let h = unsafe { slot_h.assume_init_mut() };
                unsafe {
                    x86_absorb_one_hasher_row!(
                        HAS_ALPHA,
                        true,
                        h,
                        src_row_base.add(x_bytes),
                        sh_row_base.add(x_bytes),
                        rk,
                        alpha_mask
                    );
                }
                x += cw;
            }
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(crate) unsafe fn finish_band_hashers(
        key: &TileHashKey,
        actions: &[BandRowAction],
        hashers: &[MaybeUninit<RowTileHasher>],
        slot_base: usize,
        scans: &SlotScans,
    ) {
        for (qx, (&act, slot_h)) in actions.iter().zip(hashers.iter()).enumerate() {
            if act == BandRowAction::Skip {
                continue;
            }
            // SAFETY: Every non-`Skip` slot was initialized by `process_band_row`.
            let h = unsafe { slot_h.assume_init_ref() };
            let lanes = unsafe { load_hasher_lanes(h) };
            let (meta_lo, meta_hi) = leaf_meta(u32::from(h.w), u32::from(h.h));
            let meta = _mm_set_epi64x(meta_hi as i64, meta_lo as i64);
            let res = TileHashResult {
                hash: finalize(key, lanes, meta),
                meta: RectScanMeta {
                    solid_px: h.still_solid.then(|| h.first_word.to_ne_bytes()),
                    all_opaque: h.still_opaque,
                },
            };
            // SAFETY: `slot_base + qx` is exclusively owned by this worker.
            unsafe { scans.write(slot_base + qx, res) };
        }
    }

    /// # Safety
    /// Caller must verify CPU AES-NI support.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn combine_quad_hashes(
        key: &TileHashKey,
        w: u32,
        h: u32,
        quad_hashes: &[TileHash],
    ) -> TileHash {
        let (meta_lo, meta_hi) = parent_meta(w, h, quad_hashes.len());
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        for (lane, qh) in lanes.iter_mut().zip(quad_hashes.iter()) {
            // SAFETY: `TileHash` is `[u8; 16]`.
            let q = unsafe { _mm_loadu_si128(qh.as_ptr().cast::<__m128i>()) };
            *lane = _mm_aesenc_si128(_mm_xor_si128(*lane, q), rk);
        }
        finalize(key, lanes, meta)
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod aarch64_aes {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::vaeseq_u8;
    use std::arch::aarch64::vaesmcq_u8;
    use std::arch::aarch64::vandq_u8;
    use std::arch::aarch64::vdupq_n_u8;
    use std::arch::aarch64::vdupq_n_u32;
    use std::arch::aarch64::veorq_u8;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vmaxvq_u8;
    use std::arch::aarch64::vorrq_u8;
    use std::arch::aarch64::vreinterpretq_u8_u32;
    use std::arch::aarch64::vst1q_u8;
    use std::mem::MaybeUninit;
    use std::ptr;

    use super::BandRowAction;
    use super::RectScanMeta;
    use super::RowTileHasher;
    use super::SUBTILE_W;
    use super::SlotScans;
    use super::TileHash;
    use super::TileHashKey;
    use super::TileHashResult;
    use super::XRGB_ALPHA_MASK_U32;
    use super::leaf_meta;
    use super::parent_meta;
    use super::probe_three_pixels;
    use crate::serialization::framing::TileRect;

    #[inline]
    #[target_feature(enable = "aes")]
    fn load_key_block(block: &[u64; 2]) -> uint8x16_t {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&block[0].to_le_bytes());
        bytes[8..].copy_from_slice(&block[1].to_le_bytes());
        // SAFETY: `bytes` is 16 readable initialized bytes.
        unsafe { vld1q_u8(bytes.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn init_lanes(key: &TileHashKey, meta_lo: u64, meta_hi: u64) -> ([uint8x16_t; 4], uint8x16_t) {
        let meta = load_key_block(&[meta_lo, meta_hi]);
        let lanes = [
            veorq_u8(load_key_block(&key.iv[0]), meta),
            veorq_u8(load_key_block(&key.iv[1]), meta),
            veorq_u8(load_key_block(&key.iv[2]), meta),
            veorq_u8(load_key_block(&key.iv[3]), meta),
        ];
        (lanes, meta)
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn aes_step(state: uint8x16_t, msg: uint8x16_t, rk: uint8x16_t) -> uint8x16_t {
        // ARMv8 `vaeseq_u8(state, msg)` computes `ShiftRows(SubBytes(state ^ msg))`;
        // `vaesmcq_u8` adds `MixColumns`, and `veorq_u8(..., rk)` mixes `round_key`.
        veorq_u8(vaesmcq_u8(vaeseq_u8(state, msg)), rk)
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn finalize(key: &TileHashKey, lanes: [uint8x16_t; 4], meta: uint8x16_t) -> TileHash {
        let zero = vdupq_n_u8(0);
        let fk0 = load_key_block(&key.final_keys[0]);
        let fk1 = load_key_block(&key.final_keys[1]);
        let fk2 = load_key_block(&key.final_keys[2]);
        let fk3 = load_key_block(&key.final_keys[3]);
        let fk4 = load_key_block(&key.final_keys[4]);
        let fk5 = load_key_block(&key.final_keys[5]);

        let s0 = aes_step(lanes[0], zero, fk0);
        let s1 = aes_step(lanes[1], zero, fk1);
        let s2 = aes_step(lanes[2], zero, fk2);
        let s3 = aes_step(lanes[3], zero, fk3);

        let t0 = aes_step(veorq_u8(s0, s1), zero, fk4);
        let t1 = aes_step(veorq_u8(s2, s3), zero, fk5);

        let mut h = veorq_u8(veorq_u8(t0, t1), meta);
        h = aes_step(h, zero, fk0);
        h = aes_step(h, zero, fk1);
        h = aes_step(h, zero, fk2);
        h = aes_step(h, zero, fk3);
        h = aes_step(h, zero, fk4);
        h = aes_step(h, zero, fk5);

        let mut out = [0u8; 16];
        // SAFETY: `out` is 16 writable bytes.
        unsafe { vst1q_u8(out.as_mut_ptr(), h) };
        out
    }

    #[inline]
    #[target_feature(enable = "aes")]
    fn alpha_lanes_all_opaque(
        aand0: uint8x16_t,
        aand1: uint8x16_t,
        alpha_mask: uint8x16_t,
    ) -> bool {
        let combined = vandq_u8(aand0, aand1);
        vmaxvq_u8(veorq_u8(vandq_u8(combined, alpha_mask), alpha_mask)) == 0
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn absorb_row<
        const HAS_ALPHA: bool,
        const COPY: bool,
        const TRACK_SOLID: bool,
        const TRACK_OPAQUE: bool,
    >(
        src_row: *const u8,
        dst_row: *mut u8,
        full64: usize,
        rem: usize,
        rk: uint8x16_t,
        alpha_mask: uint8x16_t,
        first_word: u32,
        first_px: uint8x16_t,
        lanes: &mut [uint8x16_t; 4],
        dacc: &mut uint8x16_t,
        aand0: &mut uint8x16_t,
        aand1: &mut uint8x16_t,
    ) {
        let mut s0 = lanes[0];
        let mut s1 = lanes[1];
        let mut s2 = lanes[2];
        let mut s3 = lanes[3];
        let mut cur_dacc = *dacc;
        let mut a0 = *aand0;
        let mut a1 = *aand1;

        let mut off = 0usize;
        while off < full64 {
            // SAFETY: `off + 64 <= full64`, within caller-validated row bounds.
            let (r0, r1, r2, r3) = unsafe {
                let r0 = vld1q_u8(src_row.add(off));
                let r1 = vld1q_u8(src_row.add(off + 16));
                let r2 = vld1q_u8(src_row.add(off + 32));
                let r3 = vld1q_u8(src_row.add(off + 48));
                if COPY {
                    vst1q_u8(dst_row.add(off), r0);
                    vst1q_u8(dst_row.add(off + 16), r1);
                    vst1q_u8(dst_row.add(off + 32), r2);
                    vst1q_u8(dst_row.add(off + 48), r3);
                }
                (r0, r1, r2, r3)
            };
            let m0 = if HAS_ALPHA {
                r0
            } else {
                vorrq_u8(r0, alpha_mask)
            };
            let m1 = if HAS_ALPHA {
                r1
            } else {
                vorrq_u8(r1, alpha_mask)
            };
            let m2 = if HAS_ALPHA {
                r2
            } else {
                vorrq_u8(r2, alpha_mask)
            };
            let m3 = if HAS_ALPHA {
                r3
            } else {
                vorrq_u8(r3, alpha_mask)
            };
            if TRACK_SOLID {
                let d01 = vorrq_u8(veorq_u8(m0, first_px), veorq_u8(m1, first_px));
                let d23 = vorrq_u8(veorq_u8(m2, first_px), veorq_u8(m3, first_px));
                cur_dacc = vorrq_u8(cur_dacc, vorrq_u8(d01, d23));
            }
            if TRACK_OPAQUE {
                a0 = vandq_u8(a0, m0);
                a1 = vandq_u8(a1, m1);
                a0 = vandq_u8(a0, m2);
                a1 = vandq_u8(a1, m3);
            }
            s0 = aes_step(s0, m0, rk);
            s1 = aes_step(s1, m1, rk);
            s2 = aes_step(s2, m2, rk);
            s3 = aes_step(s3, m3, rk);
            off += 64;
        }

        if rem != 0 {
            let mut left = rem;
            if left >= 16 {
                // SAFETY: `off + 16 <= full64 + rem`.
                let r0 = unsafe {
                    let v = vld1q_u8(src_row.add(off));
                    if COPY {
                        vst1q_u8(dst_row.add(off), v);
                    }
                    v
                };
                let m0 = if HAS_ALPHA {
                    r0
                } else {
                    vorrq_u8(r0, alpha_mask)
                };
                if TRACK_SOLID {
                    cur_dacc = vorrq_u8(cur_dacc, veorq_u8(m0, first_px));
                }
                if TRACK_OPAQUE {
                    a0 = vandq_u8(a0, m0);
                }
                s0 = aes_step(s0, m0, rk);
                off += 16;
                left -= 16;

                if left >= 16 {
                    // SAFETY: `off + 16 <= full64 + rem`.
                    let r1 = unsafe {
                        let v = vld1q_u8(src_row.add(off));
                        if COPY {
                            vst1q_u8(dst_row.add(off), v);
                        }
                        v
                    };
                    let m1 = if HAS_ALPHA {
                        r1
                    } else {
                        vorrq_u8(r1, alpha_mask)
                    };
                    if TRACK_SOLID {
                        cur_dacc = vorrq_u8(cur_dacc, veorq_u8(m1, first_px));
                    }
                    if TRACK_OPAQUE {
                        a1 = vandq_u8(a1, m1);
                    }
                    s1 = aes_step(s1, m1, rk);
                    off += 16;
                    left -= 16;

                    if left >= 16 {
                        // SAFETY: `off + 16 <= full64 + rem`.
                        let r2 = unsafe {
                            let v = vld1q_u8(src_row.add(off));
                            if COPY {
                                vst1q_u8(dst_row.add(off), v);
                            }
                            v
                        };
                        let m2 = if HAS_ALPHA {
                            r2
                        } else {
                            vorrq_u8(r2, alpha_mask)
                        };
                        if TRACK_SOLID {
                            cur_dacc = vorrq_u8(cur_dacc, veorq_u8(m2, first_px));
                        }
                        if TRACK_OPAQUE {
                            a0 = vandq_u8(a0, m2);
                        }
                        s2 = aes_step(s2, m2, rk);
                        off += 16;
                        left -= 16;
                    }
                }
            }

            if left > 0 {
                debug_assert!(left < 16 && left.is_multiple_of(4));
                let mut tail_words = [0u32; 4];
                // SAFETY: `off + left == full64 + rem`, `left` is 4, 8, or 12.
                let (p0, p1, p2) = unsafe {
                    let p0 = src_row.add(off).cast::<u32>().read_unaligned();
                    tail_words[0] = p0;
                    if COPY {
                        dst_row.add(off).cast::<u32>().write_unaligned(p0);
                    }
                    let mut p1 = 0u32;
                    let mut p2 = 0u32;
                    if left >= 8 {
                        p1 = src_row.add(off + 4).cast::<u32>().read_unaligned();
                        tail_words[1] = p1;
                        if COPY {
                            dst_row.add(off + 4).cast::<u32>().write_unaligned(p1);
                        }
                        if left == 12 {
                            p2 = src_row.add(off + 8).cast::<u32>().read_unaligned();
                            tail_words[2] = p2;
                            if COPY {
                                dst_row.add(off + 8).cast::<u32>().write_unaligned(p2);
                            }
                        }
                    }
                    (p0, p1, p2)
                };
                if TRACK_SOLID || TRACK_OPAQUE {
                    let c0 = if HAS_ALPHA {
                        p0
                    } else {
                        p0 | XRGB_ALPHA_MASK_U32
                    };
                    let mut td = c0 ^ first_word;
                    let mut ta = c0;
                    if left >= 8 {
                        let c1 = if HAS_ALPHA {
                            p1
                        } else {
                            p1 | XRGB_ALPHA_MASK_U32
                        };
                        td |= c1 ^ first_word;
                        if TRACK_OPAQUE {
                            ta &= c1;
                        }
                        if left == 12 {
                            let c2 = if HAS_ALPHA {
                                p2
                            } else {
                                p2 | XRGB_ALPHA_MASK_U32
                            };
                            td |= c2 ^ first_word;
                            if TRACK_OPAQUE {
                                ta &= c2;
                            }
                        }
                    }
                    if TRACK_SOLID {
                        cur_dacc = vorrq_u8(cur_dacc, vreinterpretq_u8_u32(vdupq_n_u32(td)));
                    }
                    if TRACK_OPAQUE {
                        a0 = vandq_u8(a0, vreinterpretq_u8_u32(vdupq_n_u32(ta)));
                    }
                }

                // SAFETY: `tail_words` is 16 initialized bytes.
                let rt = unsafe { vld1q_u8(tail_words.as_ptr().cast::<u8>()) };
                let mt = if HAS_ALPHA {
                    rt
                } else {
                    vorrq_u8(rt, alpha_mask)
                };
                s3 = aes_step(s3, mt, rk);
            }
        }

        lanes[0] = s0;
        lanes[1] = s1;
        lanes[2] = s2;
        lanes[3] = s3;
        if TRACK_SOLID {
            *dacc = cur_dacc;
        }
        if TRACK_OPAQUE {
            *aand0 = a0;
            *aand1 = a1;
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "aes")]
    fn empty_rect_result(key: &TileHashKey, rect: TileRect) -> TileHashResult {
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta::EMPTY,
        }
    }

    macro_rules! a64_absorb_rows_fast {
        (
            $src_start:expr, $dst_start:expr, $n_rows:expr, $stride:expr,
            $full64:expr, $rem:expr, $has_alpha:expr, $copy:expr,
            $rk:expr, $alpha_mask:expr, $lanes:expr, $still_opaque:expr
        ) => {{
            let mut src_ptr: *const u8 = $src_start;
            let mut dst_ptr: *mut u8 = $dst_start;
            let n_rows: usize = $n_rows;
            let mut dummy_dacc = vdupq_n_u8(0);
            let zero_px = vdupq_n_u8(0);
            let row_bytes = $full64 + $rem;
            let pf_rows = $copy
                && $stride > row_bytes
                && $stride >= crate::filtering::prefetch::MIN_PAGE_STRIDE_BYTES;
            if $has_alpha && $still_opaque {
                let mut aand0 = vdupq_n_u8(0xFF);
                let mut aand1 = vdupq_n_u8(0xFF);
                for ry in 0..n_rows {
                    if pf_rows && ry + crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD < n_rows
                    {
                        crate::filtering::prefetch::row_nta(
                            src_ptr
                                .add(crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD * $stride),
                            row_bytes,
                        );
                    }
                    absorb_row::<true, $copy, false, true>(
                        src_ptr,
                        dst_ptr,
                        $full64,
                        $rem,
                        $rk,
                        $alpha_mask,
                        0,
                        zero_px,
                        $lanes,
                        &mut dummy_dacc,
                        &mut aand0,
                        &mut aand1,
                    );
                    src_ptr = src_ptr.add($stride);
                    if $copy {
                        dst_ptr = dst_ptr.add($stride);
                    }
                }
                alpha_lanes_all_opaque(aand0, aand1, $alpha_mask)
            } else {
                let mut dummy_a0 = vdupq_n_u8(0);
                let mut dummy_a1 = vdupq_n_u8(0);
                for ry in 0..n_rows {
                    if pf_rows && ry + crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD < n_rows
                    {
                        crate::filtering::prefetch::row_nta(
                            src_ptr
                                .add(crate::filtering::prefetch::COLD_FILL_ROW_LOOKAHEAD * $stride),
                            row_bytes,
                        );
                    }
                    absorb_row::<$has_alpha, $copy, false, false>(
                        src_ptr,
                        dst_ptr,
                        $full64,
                        $rem,
                        $rk,
                        $alpha_mask,
                        0,
                        zero_px,
                        $lanes,
                        &mut dummy_dacc,
                        &mut dummy_a0,
                        &mut dummy_a1,
                    );
                    src_ptr = src_ptr.add($stride);
                    if $copy {
                        dst_ptr = dst_ptr.add($stride);
                    }
                }
                !$has_alpha
            }
        }};
    }

    macro_rules! a64_absorb_rows_solid_then_fast {
        (
            $src_start:expr, $dst_start:expr, $n_rows:expr, $stride:expr,
            $full64:expr, $rem:expr, $has_alpha:expr, $copy:expr,
            $rk:expr, $alpha_mask:expr, $first_word:expr, $lanes:expr,
            $still_opaque_in:expr
        ) => {{
            let mut src_ptr: *const u8 = $src_start;
            let mut dst_ptr: *mut u8 = $dst_start;
            let n_rows: usize = $n_rows;
            let first_word: u32 = $first_word;
            let first_px = vreinterpretq_u8_u32(vdupq_n_u32(first_word));
            let mut dacc = vdupq_n_u8(0);
            let mut aand0 = vdupq_n_u8(0xFF);
            let mut aand1 = vdupq_n_u8(0xFF);
            let mut ry = 0usize;
            let mut still_solid = true;
            let mut still_opaque: bool = $still_opaque_in;
            while ry < n_rows {
                absorb_row::<$has_alpha, $copy, true, $has_alpha>(
                    src_ptr,
                    dst_ptr,
                    $full64,
                    $rem,
                    $rk,
                    $alpha_mask,
                    first_word,
                    first_px,
                    $lanes,
                    &mut dacc,
                    &mut aand0,
                    &mut aand1,
                );
                src_ptr = src_ptr.add($stride);
                if $copy {
                    dst_ptr = dst_ptr.add($stride);
                }
                ry += 1;
                if vmaxvq_u8(dacc) != 0 {
                    still_solid = false;
                    if $has_alpha && still_opaque {
                        still_opaque = alpha_lanes_all_opaque(aand0, aand1, $alpha_mask);
                    }
                    break;
                }
            }
            if ry < n_rows {
                still_opaque = a64_absorb_rows_fast!(
                    src_ptr,
                    dst_ptr,
                    n_rows - ry,
                    $stride,
                    $full64,
                    $rem,
                    $has_alpha,
                    $copy,
                    $rk,
                    $alpha_mask,
                    $lanes,
                    still_opaque
                );
            }
            (still_solid, !$has_alpha || still_opaque)
        }};
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn hash_rect_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        row_ptr: *const u8,
        stride: usize,
        w_h: u64,
        first_word: u32,
    ) -> TileHashResult {
        let w = (w_h >> 32) as u32;
        let h = w_h as u32;
        let (meta_lo, meta_hi) = leaf_meta(w, h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let row_bytes = w as usize * 4;
        let th = h as usize;
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;
        // SAFETY: Caller guarantees `row_ptr` is readable for `th` rows of `row_bytes`.
        let (still_solid, all_opaque) = unsafe {
            a64_absorb_rows_solid_then_fast!(
                row_ptr,
                ptr::null_mut(),
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_ptr: *const u8,
        sh_ptr: *mut u8,
        stride: usize,
        w_h: u64,
    ) -> TileHashResult {
        let w = (w_h >> 32) as u32;
        let h = w_h as u32;
        let (meta_lo, meta_hi) = leaf_meta(w, h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let row_bytes = w as usize * 4;
        let th = h as usize;
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        // SAFETY: Caller guarantees `src_ptr` is readable for `th > 0` rows of `w > 0` pixels.
        let first_raw = unsafe { src_ptr.cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;
        // SAFETY: Caller guarantees `src_ptr` readable and `sh_ptr` writable for `th` rows.
        let (still_solid, all_opaque) = unsafe {
            a64_absorb_rows_solid_then_fast!(
                src_ptr,
                sh_ptr,
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                true,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_with_overlap_solid_candidate<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;
        // SAFETY: Caller guarantees `copy_from_ry > 0`, `th > 0 && tw > 0`,
        // `shadow_base` initialized for rows `0..copy_from_ry`, and `src_base`
        // readable / `shadow_base` writable for rows `copy_from_ry..th`.
        let src_ptr_base = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let first_raw = unsafe { sh_ptr.cast_const().cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let still_opaque = !HAS_ALPHA || (first_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32;

        // SAFETY: Caller guarantees `sh_ptr` is initialized and readable for `0..copy_from_ry`.
        let (mut still_solid, mut all_opaque) = unsafe {
            a64_absorb_rows_solid_then_fast!(
                sh_ptr.cast_const(),
                ptr::null_mut(),
                copy_from_ry,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                first_word,
                &mut lanes,
                still_opaque
            )
        };

        if copy_from_ry < th {
            // SAFETY: Caller guarantees `src_ptr_base` readable and `sh_ptr` writable
            // for rows `copy_from_ry..th`.
            let src_ptr = unsafe { src_ptr_base.add(copy_from_ry * stride) };
            let dst_ptr = unsafe { sh_ptr.add(copy_from_ry * stride) };
            let rem_rows = th - copy_from_ry;
            if still_solid {
                (still_solid, all_opaque) = unsafe {
                    a64_absorb_rows_solid_then_fast!(
                        src_ptr, dst_ptr, rem_rows, stride, full64, rem, HAS_ALPHA, true, rk,
                        alpha_mask, first_word, &mut lanes, all_opaque
                    )
                };
            } else {
                all_opaque = unsafe {
                    a64_absorb_rows_fast!(
                        src_ptr, dst_ptr, rem_rows, stride, full64, rem, HAS_ALPHA, true, rk,
                        alpha_mask, &mut lanes, all_opaque
                    )
                };
            }
        }

        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn copy_and_hash_with_overlap<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;
        // SAFETY: Caller guarantees `copy_from_ry > 0`, `shadow_base` rows
        // `0..copy_from_ry` are initialized and readable, and `src_base` rows
        // `copy_from_ry..th` are readable, with `th > 0 && tw > 0`.
        let src_ptr_base = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let read_probe = |ry: usize, rx: usize| -> u32 {
            let off = ry * stride + rx * 4;
            unsafe {
                let p = if ry < copy_from_ry {
                    sh_ptr.cast_const().add(off)
                } else {
                    src_ptr_base.add(off)
                };
                p.cast::<u32>().read_unaligned()
            }
        };
        let first_raw = unsafe { sh_ptr.cast_const().cast::<u32>().read_unaligned() };
        let mid_raw = read_probe(th >> 1, tw >> 1);
        let (_, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || read_probe(th - 1, tw - 1));
        if still_solid {
            // SAFETY: Bounds and exclusivity invariants forwarded from caller.
            return unsafe {
                copy_and_hash_with_overlap_solid_candidate::<HAS_ALPHA>(
                    key,
                    src_base,
                    shadow_base,
                    stride,
                    rect,
                    copy_from_ry,
                )
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;

        // SAFETY: `sh_ptr` is initialized and readable for `copy_from_ry` rows.
        let mut all_opaque = unsafe {
            a64_absorb_rows_fast!(
                sh_ptr.cast_const(),
                ptr::null_mut(),
                copy_from_ry,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        if copy_from_ry < th {
            // SAFETY: Caller validated `src_base` bounds and exclusive
            // `shadow_base` write access for rows `copy_from_ry..th`.
            all_opaque = unsafe {
                a64_absorb_rows_fast!(
                    src_ptr_base.add(copy_from_ry * stride),
                    sh_ptr.add(copy_from_ry * stride),
                    th - copy_from_ry,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    true,
                    rk,
                    alpha_mask,
                    &mut lanes,
                    all_opaque
                )
            };
        }
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    /// # Safety
    /// Caller must verify CPU ARMv8 AES support and uphold
    /// [`super::hash_rect_from_shadow`] bounds invariants.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn hash_rect_raw<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        base: *const u8,
        stride: usize,
        rect: TileRect,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return empty_rect_result(key, rect);
        }
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        // SAFETY: Caller guarantees `(y0 + ry, x_bytes..x_bytes + row_bytes)` is readable
        // and `th > 0 && tw > 0`.
        let row_ptr = unsafe { base.add(y0 * stride + x_bytes) };
        let (first_raw, mid_raw) = unsafe {
            (
                row_ptr.cast::<u32>().read_unaligned(),
                row_ptr
                    .add((th >> 1) * stride + (tw >> 1) * 4)
                    .cast::<u32>()
                    .read_unaligned(),
            )
        };
        let (first_word, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
                row_ptr
                    .add((th - 1) * stride + (tw - 1) * 4)
                    .cast::<u32>()
                    .read_unaligned()
            });
        let w_h = (u64::from(rect.w) << 32) | u64::from(rect.h);
        if still_solid {
            // SAFETY: `row_ptr` is readable for `th` rows of `row_bytes`.
            return unsafe {
                hash_rect_solid_candidate::<HAS_ALPHA>(key, row_ptr, stride, w_h, first_word)
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        // SAFETY: `row_ptr` is readable for `th` rows of `row_bytes == full64 + rem`.
        let all_opaque = unsafe {
            a64_absorb_rows_fast!(
                row_ptr,
                ptr::null_mut(),
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                false,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    /// # Safety
    /// Caller must verify CPU ARMv8 AES support and uphold
    /// [`super::copy_and_hash_rect_to_shadow`] bounds and exclusivity invariants.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn copy_and_hash_rect_to_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return empty_rect_result(key, rect);
        }
        if copy_from_ry > 0 {
            // SAFETY: Bounds and exclusivity invariants forwarded from caller.
            return unsafe {
                copy_and_hash_with_overlap::<HAS_ALPHA>(
                    key,
                    src_base,
                    shadow_base,
                    stride,
                    rect,
                    copy_from_ry,
                )
            };
        }
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;

        // SAFETY: `copy_from_ry == 0` and `th > 0 && tw > 0`, so `src_base` is
        // readable and `shadow_base` is exclusively writable for all `th` rows.
        let src_ptr = unsafe { src_base.add(base_off) };
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let (first_raw, mid_raw) = unsafe {
            (
                src_ptr.cast::<u32>().read_unaligned(),
                src_ptr
                    .add((th >> 1) * stride + (tw >> 1) * 4)
                    .cast::<u32>()
                    .read_unaligned(),
            )
        };
        let (_, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
                src_ptr
                    .add((th - 1) * stride + (tw - 1) * 4)
                    .cast::<u32>()
                    .read_unaligned()
            });
        let w_h = (u64::from(rect.w) << 32) | u64::from(rect.h);
        if still_solid {
            // SAFETY: `src_ptr` readable and `sh_ptr` writable for `th` rows.
            return unsafe {
                copy_and_hash_solid_candidate::<HAS_ALPHA>(key, src_ptr, sh_ptr, stride, w_h)
            };
        }

        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;

        // SAFETY: Caller validated `src_base` bounds and exclusive `shadow_base`
        // write access for all `th` rows.
        let all_opaque = unsafe {
            a64_absorb_rows_fast!(
                src_ptr,
                sh_ptr,
                th,
                stride,
                full64,
                rem,
                HAS_ALPHA,
                true,
                rk,
                alpha_mask,
                &mut lanes,
                still_opaque
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta),
            meta: RectScanMeta {
                solid_px: None,
                all_opaque,
            },
        }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn load_hasher_lanes(h: &RowTileHasher) -> [uint8x16_t; 4] {
        // SAFETY: `h.lanes` is `[ [u64; 2]; 4 ]` (64 readable bytes).
        unsafe {
            [
                vld1q_u8(h.lanes[0].as_ptr().cast::<u8>()),
                vld1q_u8(h.lanes[1].as_ptr().cast::<u8>()),
                vld1q_u8(h.lanes[2].as_ptr().cast::<u8>()),
                vld1q_u8(h.lanes[3].as_ptr().cast::<u8>()),
            ]
        }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn store_hasher_lanes(h: &mut RowTileHasher, lanes: [uint8x16_t; 4]) {
        // SAFETY: `h.lanes` is `[ [u64; 2]; 4 ]` (64 writable bytes).
        unsafe {
            vst1q_u8(h.lanes[0].as_mut_ptr().cast::<u8>(), lanes[0]);
            vst1q_u8(h.lanes[1].as_mut_ptr().cast::<u8>(), lanes[1]);
            vst1q_u8(h.lanes[2].as_mut_ptr().cast::<u8>(), lanes[2]);
            vst1q_u8(h.lanes[3].as_mut_ptr().cast::<u8>(), lanes[3]);
        }
    }

    #[cold]
    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn start_row_hasher_from_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        shadow_base: *const u8,
        stride: usize,
        x_bytes: usize,
        y0: usize,
        cw: usize,
        ch: usize,
        absorbed_rows: usize,
    ) -> RowTileHasher {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let row_bytes = cw * 4;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let (meta_lo, meta_hi) = leaf_meta(cw as u32, ch as u32);
        let (mut lanes, _) = init_lanes(key, meta_lo, meta_hi);

        // SAFETY: Caller guarantees `absorbed_rows > 0`, `cw > 0`, and
        // `[y0..y0 + absorbed_rows)` is readable for `row_bytes` at `x_bytes`.
        let sh_ptr = unsafe { shadow_base.add(y0 * stride + x_bytes) };
        let first_raw = unsafe { sh_ptr.cast::<u32>().read_unaligned() };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let probe_ry = absorbed_rows - 1;
        let probe_rx = if absorbed_rows > 1 { cw >> 1 } else { cw - 1 };
        let probe_raw = unsafe {
            sh_ptr
                .add(probe_ry * stride + probe_rx * 4)
                .cast::<u32>()
                .read_unaligned()
        };
        let probe_word = if HAS_ALPHA {
            probe_raw
        } else {
            probe_raw | XRGB_ALPHA_MASK_U32
        };
        let maybe_solid = first_word == probe_word;
        let maybe_opaque =
            !HAS_ALPHA || ((first_word & probe_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32);

        let (still_solid, still_opaque) = if maybe_solid {
            // SAFETY: `sh_ptr` is readable for `absorbed_rows` rows of `row_bytes`.
            unsafe {
                a64_absorb_rows_solid_then_fast!(
                    sh_ptr,
                    ptr::null_mut(),
                    absorbed_rows,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    false,
                    rk,
                    alpha_mask,
                    first_word,
                    &mut lanes,
                    maybe_opaque
                )
            }
        } else {
            // SAFETY: `sh_ptr` is readable for `absorbed_rows` rows of `row_bytes`.
            let opq = unsafe {
                a64_absorb_rows_fast!(
                    sh_ptr,
                    ptr::null_mut(),
                    absorbed_rows,
                    stride,
                    full64,
                    rem,
                    HAS_ALPHA,
                    false,
                    rk,
                    alpha_mask,
                    &mut lanes,
                    maybe_opaque
                )
            };
            (false, opq)
        };

        let mut out = RowTileHasher {
            lanes: [[0; 2]; 4],
            first_word,
            w: cw as u16,
            h: ch as u16,
            still_solid,
            still_opaque,
        };
        // SAFETY: `out.lanes` is valid for 64 bytes.
        unsafe { store_hasher_lanes(&mut out, lanes) };
        out
    }

    macro_rules! a64_absorb_one_hasher_row {
        ($has_alpha:expr, $copy:expr, $h:expr, $src_row:expr, $dst_row:expr, $rk:expr, $alpha_mask:expr) => {{
            let h: &mut RowTileHasher = $h;
            let src_row: *const u8 = $src_row;
            let dst_row: *mut u8 = $dst_row;
            let rk = $rk;
            let alpha_mask = $alpha_mask;
            let row_bytes = usize::from(h.w) * 4;
            let full64 = row_bytes & !63;
            let rem = row_bytes & 63;
            let mut lanes = load_hasher_lanes(h);
            if !h.still_solid {
                let mut dummy_dacc = vdupq_n_u8(0);
                let zero_px = vdupq_n_u8(0);
                if $has_alpha && h.still_opaque {
                    let mut aand0 = vdupq_n_u8(0xFF);
                    let mut aand1 = vdupq_n_u8(0xFF);
                    absorb_row::<true, $copy, false, true>(
                        src_row,
                        dst_row,
                        full64,
                        rem,
                        rk,
                        alpha_mask,
                        0,
                        zero_px,
                        &mut lanes,
                        &mut dummy_dacc,
                        &mut aand0,
                        &mut aand1,
                    );
                    h.still_opaque = alpha_lanes_all_opaque(aand0, aand1, alpha_mask);
                } else {
                    let mut dummy_a0 = vdupq_n_u8(0);
                    let mut dummy_a1 = vdupq_n_u8(0);
                    absorb_row::<$has_alpha, $copy, false, false>(
                        src_row,
                        dst_row,
                        full64,
                        rem,
                        rk,
                        alpha_mask,
                        0,
                        zero_px,
                        &mut lanes,
                        &mut dummy_dacc,
                        &mut dummy_a0,
                        &mut dummy_a1,
                    );
                }
            } else {
                let first_word = h.first_word;
                let first_px = vreinterpretq_u8_u32(vdupq_n_u32(first_word));
                let mut dacc = vdupq_n_u8(0);
                let mut aand0 = vdupq_n_u8(0xFF);
                let mut aand1 = vdupq_n_u8(0xFF);
                absorb_row::<$has_alpha, $copy, true, $has_alpha>(
                    src_row, dst_row, full64, rem, rk, alpha_mask, first_word, first_px,
                    &mut lanes, &mut dacc, &mut aand0, &mut aand1,
                );
                if vmaxvq_u8(dacc) != 0 {
                    h.still_solid = false;
                    if $has_alpha && h.still_opaque {
                        h.still_opaque = alpha_lanes_all_opaque(aand0, aand1, alpha_mask);
                    }
                }
            }
            store_hasher_lanes(h, lanes);
        }};
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn process_band_row<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y0: usize,
        ch: usize,
        y: usize,
        actions: &mut [BandRowAction],
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        let row_off = y * stride;
        let absorbed_rows = (y - y0) + 1;
        // SAFETY: Caller validated row `y` bounds in `src_base` and `shadow_base`.
        let src_row_base = unsafe { src_base.add(row_off) };
        let sh_row_base = unsafe { shadow_base.add(row_off) };

        let mut x = 0usize;
        for (act, slot_h) in actions.iter_mut().zip(hashers.iter_mut()) {
            let cw = (width - x).min(SUBTILE_W);
            let x_bytes = x * 4;
            match *act {
                BandRowAction::Skip => {},
                BandRowAction::AbsorbShadow | BandRowAction::AbsorbShadowThenFull => {
                    if *act == BandRowAction::AbsorbShadowThenFull {
                        *act = BandRowAction::CopyAndAbsorb;
                    }
                    // SAFETY: Non-`Skip` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        a64_absorb_one_hasher_row!(
                            HAS_ALPHA,
                            false,
                            h,
                            sh_row_base.cast_const().add(x_bytes),
                            ptr::null_mut(),
                            rk,
                            alpha_mask
                        );
                    }
                },
                BandRowAction::CopyAndAbsorb => {
                    // SAFETY: `CopyAndAbsorb` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        a64_absorb_one_hasher_row!(
                            HAS_ALPHA,
                            true,
                            h,
                            src_row_base.add(x_bytes),
                            sh_row_base.add(x_bytes),
                            rk,
                            alpha_mask
                        );
                    }
                },
                BandRowAction::StartAbsorb | BandRowAction::StartFull => {
                    let next = if *act == BandRowAction::StartFull {
                        BandRowAction::CopyAndAbsorb
                    } else {
                        BandRowAction::AbsorbShadow
                    };
                    // SAFETY: Rows `0..absorbed_rows` are finalized in `shadow_base`.
                    let h = unsafe {
                        start_row_hasher_from_shadow::<HAS_ALPHA>(
                            key,
                            shadow_base.cast_const(),
                            stride,
                            x_bytes,
                            y0,
                            cw,
                            ch,
                            absorbed_rows,
                        )
                    };
                    slot_h.write(h);
                    *act = next;
                },
            }
            x += cw;
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn copy_and_absorb_band_rows<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y_start: usize,
        y_end: usize,
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let rk = load_key_block(&key.round_key);
        let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(XRGB_ALPHA_MASK_U32));
        for y in y_start..y_end {
            let row_off = y * stride;
            // SAFETY: Caller validated `[y_start, y_end)` bounds and exclusive shadow access.
            let src_row_base = unsafe { src_base.add(row_off) };
            let sh_row_base = unsafe { shadow_base.add(row_off) };
            let mut x = 0usize;
            for slot_h in hashers.iter_mut() {
                let cw = (width - x).min(SUBTILE_W);
                let x_bytes = x * 4;
                // SAFETY: Caller guarantees every slot in `hashers` is initialized.
                let h = unsafe { slot_h.assume_init_mut() };
                unsafe {
                    a64_absorb_one_hasher_row!(
                        HAS_ALPHA,
                        true,
                        h,
                        src_row_base.add(x_bytes),
                        sh_row_base.add(x_bytes),
                        rk,
                        alpha_mask
                    );
                }
                x += cw;
            }
        }
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn finish_band_hashers(
        key: &TileHashKey,
        actions: &[BandRowAction],
        hashers: &[MaybeUninit<RowTileHasher>],
        slot_base: usize,
        scans: &SlotScans,
    ) {
        for (qx, (&act, slot_h)) in actions.iter().zip(hashers.iter()).enumerate() {
            if act == BandRowAction::Skip {
                continue;
            }
            // SAFETY: Every non-`Skip` slot was initialized by `process_band_row`.
            let h = unsafe { slot_h.assume_init_ref() };
            let lanes = unsafe { load_hasher_lanes(h) };
            let (meta_lo, meta_hi) = leaf_meta(u32::from(h.w), u32::from(h.h));
            let meta = load_key_block(&[meta_lo, meta_hi]);
            let res = TileHashResult {
                hash: finalize(key, lanes, meta),
                meta: RectScanMeta {
                    solid_px: h.still_solid.then(|| h.first_word.to_ne_bytes()),
                    all_opaque: h.still_opaque,
                },
            };
            // SAFETY: `slot_base + qx` is exclusively owned by this worker.
            unsafe { scans.write(slot_base + qx, res) };
        }
    }

    /// # Safety
    /// Caller must verify CPU ARMv8 AES support.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn combine_quad_hashes(
        key: &TileHashKey,
        w: u32,
        h: u32,
        quad_hashes: &[TileHash],
    ) -> TileHash {
        let (meta_lo, meta_hi) = parent_meta(w, h, quad_hashes.len());
        let (mut lanes, meta) = init_lanes(key, meta_lo, meta_hi);
        let rk = load_key_block(&key.round_key);
        for (lane, qh) in lanes.iter_mut().zip(quad_hashes.iter()) {
            // SAFETY: `TileHash` is `[u8; 16]`.
            let q = unsafe { vld1q_u8(qh.as_ptr()) };
            *lane = aes_step(*lane, q, rk);
        }
        finalize(key, lanes, meta)
    }
}

/// Always-compiled portable 128-bit keyed fallback (used on CPUs without
/// hardware AES, under Miri, and exercised directly in unit tests).
mod portable {
    use std::mem::MaybeUninit;
    use std::ptr;

    use super::BandRowAction;
    use super::RectScanMeta;
    use super::RowTileHasher;
    use super::SUBTILE_W;
    use super::SlotScans;
    use super::TileHash;
    use super::TileHashKey;
    use super::TileHashResult;
    use super::XRGB_ALPHA_MASK_U32;
    use super::XRGB_ALPHA_MASK_U64;
    use super::leaf_meta;
    use super::parent_meta;
    use super::probe_three_pixels;
    use crate::serialization::framing::TileRect;

    const P_K0: u64 = 0xA076_1D64_78BD_642F;
    const P_K1: u64 = 0xE703_7ED1_A0B4_28DB;
    const P_K2: u64 = 0x8EBC_6AF0_9C88_C6E3;
    const P_K3: u64 = 0x5899_65CC_7537_4CC3;

    #[inline(always)]
    fn mum(a: u64, b: u64) -> u64 {
        let p = u128::from(a).wrapping_mul(u128::from(b));
        (p as u64) ^ ((p >> 64) as u64)
    }

    #[inline(always)]
    fn step(s: u64, lo: u64, hi: u64, rk0: u64, rk1: u64) -> u64 {
        let m0 = mum(s ^ lo ^ rk0, (hi ^ rk1 ^ P_K0) | 1);
        let m1 = mum(m0 ^ hi, P_K1);
        m1.wrapping_add(s.rotate_left(29) ^ lo)
    }

    #[inline(always)]
    fn init_lanes(key: &TileHashKey, meta_lo: u64, meta_hi: u64) -> [u64; 4] {
        [
            key.iv[0][0] ^ key.iv[0][1].rotate_left(17) ^ meta_lo,
            key.iv[1][0] ^ key.iv[1][1].rotate_left(23) ^ meta_hi,
            key.iv[2][0] ^ key.iv[2][1].rotate_left(31) ^ meta_lo.wrapping_add(P_K2),
            key.iv[3][0] ^ key.iv[3][1].rotate_left(41) ^ meta_hi.wrapping_add(P_K3),
        ]
    }

    #[inline(always)]
    fn finalize(key: &TileHashKey, lanes: [u64; 4], meta_lo: u64, meta_hi: u64) -> TileHash {
        let s0 = mum(
            lanes[0] ^ key.final_keys[0][0],
            (key.final_keys[0][1] ^ P_K0) | 1,
        );
        let s1 = mum(
            lanes[1] ^ key.final_keys[1][0],
            (key.final_keys[1][1] ^ P_K1) | 1,
        );
        let s2 = mum(
            lanes[2] ^ key.final_keys[2][0],
            (key.final_keys[2][1] ^ P_K2) | 1,
        );
        let s3 = mum(
            lanes[3] ^ key.final_keys[3][0],
            (key.final_keys[3][1] ^ P_K3) | 1,
        );

        let mut h0 = (s0 ^ s2 ^ meta_lo).wrapping_add(key.final_keys[4][0]);
        let mut h1 = (s1 ^ s3 ^ meta_hi).wrapping_add(key.final_keys[4][1]);

        for &(fk0, fk1) in &[
            (key.final_keys[0][0], key.final_keys[1][1]),
            (key.final_keys[2][0], key.final_keys[3][1]),
            (key.final_keys[4][0], key.final_keys[4][1]),
            (key.final_keys[5][0], key.final_keys[5][1]),
        ] {
            let p0 = u128::from(h0 ^ fk0).wrapping_mul(u128::from(P_K0));
            let p1 = u128::from(h1 ^ fk1).wrapping_mul(u128::from(P_K1));
            let nh0 = ((p0 as u64) ^ ((p1 >> 64) as u64)).wrapping_add(h1.rotate_left(27));
            let nh1 = ((p1 as u64) ^ ((p0 >> 64) as u64)).wrapping_add(h0.rotate_left(31));
            h0 = nh0 ^ (nh0 >> 33);
            h1 = nh1 ^ (nh1 >> 29);
        }

        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&h0.to_le_bytes());
        out[8..].copy_from_slice(&h1.to_le_bytes());
        out
    }

    #[inline(always)]
    fn dup_u32_to_u64(word: u32) -> u64 {
        (u64::from(word) << 32) | u64::from(word)
    }

    #[inline(always)]
    unsafe fn load_and_maybe_copy_16<const HAS_ALPHA: bool, const COPY: bool>(
        src: *const u8,
        dst: *mut u8,
    ) -> (u64, u64) {
        // SAFETY: Caller guarantees `src` is readable for 16 bytes, and when
        // `COPY` is `true`, `dst` is writable for 16 bytes.
        let (r0, r1) = unsafe {
            let r0 = src.cast::<u64>().read_unaligned();
            let r1 = src.add(8).cast::<u64>().read_unaligned();
            if COPY {
                dst.cast::<u64>().write_unaligned(r0);
                dst.add(8).cast::<u64>().write_unaligned(r1);
            }
            (r0, r1)
        };
        if HAS_ALPHA {
            (r0, r1)
        } else {
            (r0 | XRGB_ALPHA_MASK_U64, r1 | XRGB_ALPHA_MASK_U64)
        }
    }

    #[inline(always)]
    unsafe fn absorb_row<
        const HAS_ALPHA: bool,
        const COPY: bool,
        const TRACK_SOLID: bool,
        const TRACK_OPAQUE: bool,
    >(
        src_row: *const u8,
        dst_row: *mut u8,
        full64: usize,
        rem: usize,
        rk0: u64,
        rk1: u64,
        first_word: u32,
        first_pair: u64,
        lanes: &mut [u64; 4],
        diff_acc: &mut u64,
        alpha_and: &mut u64,
    ) {
        let mut s0 = lanes[0];
        let mut s1 = lanes[1];
        let mut s2 = lanes[2];
        let mut s3 = lanes[3];
        let mut dacc = *diff_acc;
        let mut aand = *alpha_and;

        let mut off = 0usize;
        while off < full64 {
            // SAFETY: `off + 64 <= full64`, within caller-validated row bounds.
            unsafe {
                let (m0l, m0h) =
                    load_and_maybe_copy_16::<HAS_ALPHA, COPY>(src_row.add(off), dst_row.add(off));
                let (m1l, m1h) = load_and_maybe_copy_16::<HAS_ALPHA, COPY>(
                    src_row.add(off + 16),
                    dst_row.add(off + 16),
                );
                let (m2l, m2h) = load_and_maybe_copy_16::<HAS_ALPHA, COPY>(
                    src_row.add(off + 32),
                    dst_row.add(off + 32),
                );
                let (m3l, m3h) = load_and_maybe_copy_16::<HAS_ALPHA, COPY>(
                    src_row.add(off + 48),
                    dst_row.add(off + 48),
                );
                if TRACK_SOLID {
                    dacc |= (m0l ^ first_pair)
                        | (m0h ^ first_pair)
                        | (m1l ^ first_pair)
                        | (m1h ^ first_pair)
                        | (m2l ^ first_pair)
                        | (m2h ^ first_pair)
                        | (m3l ^ first_pair)
                        | (m3h ^ first_pair);
                }
                if TRACK_OPAQUE {
                    aand &= m0l & m0h & m1l & m1h & m2l & m2h & m3l & m3h;
                }
                s0 = step(s0, m0l, m0h, rk0, rk1);
                s1 = step(s1, m1l, m1h, rk0, rk1);
                s2 = step(s2, m2l, m2h, rk0, rk1);
                s3 = step(s3, m3l, m3h, rk0, rk1);
            }
            off += 64;
        }

        if rem != 0 {
            let mut left = rem;
            if left >= 16 {
                // SAFETY: `off + 16 <= full64 + rem`.
                let (ml, mh) = unsafe {
                    load_and_maybe_copy_16::<HAS_ALPHA, COPY>(src_row.add(off), dst_row.add(off))
                };
                if TRACK_SOLID {
                    dacc |= (ml ^ first_pair) | (mh ^ first_pair);
                }
                if TRACK_OPAQUE {
                    aand &= ml & mh;
                }
                s0 = step(s0, ml, mh, rk0, rk1);
                off += 16;
                left -= 16;

                if left >= 16 {
                    // SAFETY: `off + 16 <= full64 + rem`.
                    let (ml, mh) = unsafe {
                        load_and_maybe_copy_16::<HAS_ALPHA, COPY>(
                            src_row.add(off),
                            dst_row.add(off),
                        )
                    };
                    if TRACK_SOLID {
                        dacc |= (ml ^ first_pair) | (mh ^ first_pair);
                    }
                    if TRACK_OPAQUE {
                        aand &= ml & mh;
                    }
                    s1 = step(s1, ml, mh, rk0, rk1);
                    off += 16;
                    left -= 16;

                    if left >= 16 {
                        // SAFETY: `off + 16 <= full64 + rem`.
                        let (ml, mh) = unsafe {
                            load_and_maybe_copy_16::<HAS_ALPHA, COPY>(
                                src_row.add(off),
                                dst_row.add(off),
                            )
                        };
                        if TRACK_SOLID {
                            dacc |= (ml ^ first_pair) | (mh ^ first_pair);
                        }
                        if TRACK_OPAQUE {
                            aand &= ml & mh;
                        }
                        s2 = step(s2, ml, mh, rk0, rk1);
                        off += 16;
                        left -= 16;
                    }
                }
            }

            if left > 0 {
                debug_assert!(left < 16 && left.is_multiple_of(4));
                let mut tail = [0u8; 16];
                let mut td = 0u32;
                let mut ta = u32::MAX;
                // SAFETY: `off + left == full64 + rem`, `left` is 4, 8, or 12.
                unsafe {
                    let p0 = src_row.add(off).cast::<[u8; 4]>().read();
                    tail[0..4].copy_from_slice(&p0);
                    if COPY {
                        dst_row.add(off).cast::<[u8; 4]>().write(p0);
                    }
                    if TRACK_SOLID || TRACK_OPAQUE {
                        let c0 = u32::from_ne_bytes(p0)
                            | if HAS_ALPHA { 0 } else { XRGB_ALPHA_MASK_U32 };
                        td |= c0 ^ first_word;
                        if TRACK_OPAQUE {
                            ta &= c0;
                        }
                    }
                    if left >= 8 {
                        let p1 = src_row.add(off + 4).cast::<[u8; 4]>().read();
                        tail[4..8].copy_from_slice(&p1);
                        if COPY {
                            dst_row.add(off + 4).cast::<[u8; 4]>().write(p1);
                        }
                        if TRACK_SOLID || TRACK_OPAQUE {
                            let c1 = u32::from_ne_bytes(p1)
                                | if HAS_ALPHA { 0 } else { XRGB_ALPHA_MASK_U32 };
                            td |= c1 ^ first_word;
                            if TRACK_OPAQUE {
                                ta &= c1;
                            }
                        }
                        if left == 12 {
                            let p2 = src_row.add(off + 8).cast::<[u8; 4]>().read();
                            tail[8..12].copy_from_slice(&p2);
                            if COPY {
                                dst_row.add(off + 8).cast::<[u8; 4]>().write(p2);
                            }
                            if TRACK_SOLID || TRACK_OPAQUE {
                                let c2 = u32::from_ne_bytes(p2)
                                    | if HAS_ALPHA { 0 } else { XRGB_ALPHA_MASK_U32 };
                                td |= c2 ^ first_word;
                                if TRACK_OPAQUE {
                                    ta &= c2;
                                }
                            }
                        }
                    }
                }
                if TRACK_SOLID {
                    dacc |= u64::from(td);
                }
                if TRACK_OPAQUE {
                    aand &= dup_u32_to_u64(ta);
                }

                let mut lo = u64::from_ne_bytes(tail[..8].try_into().unwrap());
                let mut hi = u64::from_ne_bytes(tail[8..].try_into().unwrap());
                if !HAS_ALPHA {
                    lo |= XRGB_ALPHA_MASK_U64;
                    hi |= XRGB_ALPHA_MASK_U64;
                }
                s3 = step(s3, lo, hi, rk0, rk1);
            }
        }

        lanes[0] = s0;
        lanes[1] = s1;
        lanes[2] = s2;
        lanes[3] = s3;
        if TRACK_SOLID {
            *diff_acc = dacc;
        }
        if TRACK_OPAQUE {
            *alpha_and = aand;
        }
    }

    #[inline(always)]
    unsafe fn absorb_rows_phased<const HAS_ALPHA: bool, const COPY: bool>(
        mut src_ptr: *const u8,
        mut dst_ptr: *mut u8,
        n_rows: usize,
        stride: usize,
        full64: usize,
        rem: usize,
        rk0: u64,
        rk1: u64,
        first_word: u32,
        first_pair: u64,
        lanes: &mut [u64; 4],
        mut still_solid: bool,
        mut still_opaque: bool,
    ) -> (bool, bool) {
        let mut ry = 0usize;
        if still_solid {
            let mut diff_acc = 0u64;
            let mut alpha_and = u64::MAX;
            while ry < n_rows {
                // SAFETY: Caller guarantees `src_ptr` readable and (when `COPY`)
                // `dst_ptr` writable for `full64 + rem` bytes across `n_rows`.
                unsafe {
                    absorb_row::<HAS_ALPHA, COPY, true, HAS_ALPHA>(
                        src_ptr,
                        dst_ptr,
                        full64,
                        rem,
                        rk0,
                        rk1,
                        first_word,
                        first_pair,
                        lanes,
                        &mut diff_acc,
                        &mut alpha_and,
                    );
                    src_ptr = src_ptr.add(stride);
                    if COPY {
                        dst_ptr = dst_ptr.add(stride);
                    }
                }
                ry += 1;
                if diff_acc != 0 {
                    still_solid = false;
                    if HAS_ALPHA && still_opaque {
                        still_opaque = (alpha_and & XRGB_ALPHA_MASK_U64) == XRGB_ALPHA_MASK_U64;
                    }
                    break;
                }
            }
        }
        if ry < n_rows {
            let mut dummy_diff = 0u64;
            if HAS_ALPHA && still_opaque {
                let mut alpha_and = u64::MAX;
                while ry < n_rows {
                    // SAFETY: Remaining rows lie within caller-validated bounds.
                    unsafe {
                        absorb_row::<true, COPY, false, true>(
                            src_ptr,
                            dst_ptr,
                            full64,
                            rem,
                            rk0,
                            rk1,
                            0,
                            0,
                            lanes,
                            &mut dummy_diff,
                            &mut alpha_and,
                        );
                        src_ptr = src_ptr.add(stride);
                        if COPY {
                            dst_ptr = dst_ptr.add(stride);
                        }
                    }
                    ry += 1;
                }
                still_opaque = (alpha_and & XRGB_ALPHA_MASK_U64) == XRGB_ALPHA_MASK_U64;
            } else {
                let mut dummy_alpha = u64::MAX;
                while ry < n_rows {
                    // SAFETY: Remaining rows lie within caller-validated bounds.
                    unsafe {
                        absorb_row::<HAS_ALPHA, COPY, false, false>(
                            src_ptr,
                            dst_ptr,
                            full64,
                            rem,
                            rk0,
                            rk1,
                            0,
                            0,
                            lanes,
                            &mut dummy_diff,
                            &mut dummy_alpha,
                        );
                        src_ptr = src_ptr.add(stride);
                        if COPY {
                            dst_ptr = dst_ptr.add(stride);
                        }
                    }
                    ry += 1;
                }
            }
        }
        (still_solid, !HAS_ALPHA || still_opaque)
    }

    #[inline(never)]
    pub(super) unsafe fn hash_rect_raw<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        base: *const u8,
        stride: usize,
        rect: TileRect,
    ) -> TileHashResult {
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let mut lanes = init_lanes(key, meta_lo, meta_hi);
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return TileHashResult {
                hash: finalize(key, lanes, meta_lo, meta_hi),
                meta: RectScanMeta::EMPTY,
            };
        }
        let [rk0, rk1] = key.round_key;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        // SAFETY: Caller guarantees `(y0 + ry, x_bytes..x_bytes + row_bytes)` is readable
        // and `th > 0 && tw > 0`.
        let row_ptr = unsafe { base.add(y0 * stride + x_bytes) };
        let (first_raw, mid_raw) = unsafe {
            (
                u32::from_ne_bytes(row_ptr.cast::<[u8; 4]>().read()),
                u32::from_ne_bytes(
                    row_ptr
                        .add((th >> 1) * stride + (tw >> 1) * 4)
                        .cast::<[u8; 4]>()
                        .read(),
                ),
            )
        };
        let (first_word, still_solid, still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(first_raw, mid_raw, || unsafe {
                u32::from_ne_bytes(
                    row_ptr
                        .add((th - 1) * stride + (tw - 1) * 4)
                        .cast::<[u8; 4]>()
                        .read(),
                )
            });
        let first_pair = dup_u32_to_u64(first_word);
        // SAFETY: `row_ptr` is readable for `th` rows of `row_bytes`.
        let (solid, all_opaque) = unsafe {
            absorb_rows_phased::<HAS_ALPHA, false>(
                row_ptr,
                ptr::null_mut(),
                th,
                stride,
                full64,
                rem,
                rk0,
                rk1,
                first_word,
                first_pair,
                &mut lanes,
                still_solid,
                still_opaque,
            )
        };
        TileHashResult {
            hash: finalize(key, lanes, meta_lo, meta_hi),
            meta: RectScanMeta {
                solid_px: solid.then(|| first_word.to_ne_bytes()),
                all_opaque,
            },
        }
    }

    #[inline(never)]
    pub(super) unsafe fn copy_and_hash_rect_to_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        rect: TileRect,
        copy_from_ry: usize,
    ) -> TileHashResult {
        let (meta_lo, meta_hi) = leaf_meta(rect.w, rect.h);
        let mut lanes = init_lanes(key, meta_lo, meta_hi);
        let tw = rect.w as usize;
        let th = rect.h as usize;
        let row_bytes = tw * 4;
        if th == 0 || row_bytes == 0 {
            return TileHashResult {
                hash: finalize(key, lanes, meta_lo, meta_hi),
                meta: RectScanMeta::EMPTY,
            };
        }
        let [rk0, rk1] = key.round_key;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let x_bytes = rect.x as usize * 4;
        let y0 = rect.y as usize;
        let base_off = y0 * stride + x_bytes;

        // SAFETY: Caller guarantees `shadow_base` rows `0..copy_from_ry`
        // are initialized and readable for `row_bytes`, and `src_base` rows
        // `copy_from_ry..th` are readable for `row_bytes`.
        let sh_ptr = unsafe { shadow_base.add(base_off) };
        let read_probe = |ry: usize, rx: usize| -> u32 {
            let off = ry * stride + rx * 4;
            unsafe {
                let p = if ry < copy_from_ry {
                    sh_ptr.cast_const().add(off)
                } else {
                    src_base.add(base_off + off)
                };
                u32::from_ne_bytes(p.cast::<[u8; 4]>().read())
            }
        };
        let (first_word, mut still_solid, mut still_opaque) =
            probe_three_pixels::<HAS_ALPHA>(read_probe(0, 0), read_probe(th >> 1, tw >> 1), || {
                read_probe(th - 1, tw - 1)
            });
        let first_pair = dup_u32_to_u64(first_word);

        if copy_from_ry > 0 {
            // SAFETY: `sh_ptr` is initialized and readable for `copy_from_ry` rows.
            (still_solid, still_opaque) = unsafe {
                absorb_rows_phased::<HAS_ALPHA, false>(
                    sh_ptr.cast_const(),
                    ptr::null_mut(),
                    copy_from_ry,
                    stride,
                    full64,
                    rem,
                    rk0,
                    rk1,
                    first_word,
                    first_pair,
                    &mut lanes,
                    still_solid,
                    still_opaque,
                )
            };
        }

        if copy_from_ry < th {
            // SAFETY: Caller validated `src_base` bounds and exclusive
            // `shadow_base` write access for rows `copy_from_ry..th`.
            (still_solid, still_opaque) = unsafe {
                absorb_rows_phased::<HAS_ALPHA, true>(
                    src_base.add(base_off + copy_from_ry * stride),
                    sh_ptr.add(copy_from_ry * stride),
                    th - copy_from_ry,
                    stride,
                    full64,
                    rem,
                    rk0,
                    rk1,
                    first_word,
                    first_pair,
                    &mut lanes,
                    still_solid,
                    still_opaque,
                )
            };
        }
        TileHashResult {
            hash: finalize(key, lanes, meta_lo, meta_hi),
            meta: RectScanMeta {
                solid_px: still_solid.then(|| first_word.to_ne_bytes()),
                all_opaque: !HAS_ALPHA || still_opaque,
            },
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn start_row_hasher_from_shadow<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        shadow_base: *const u8,
        stride: usize,
        x_bytes: usize,
        y0: usize,
        cw: usize,
        ch: usize,
        absorbed_rows: usize,
    ) -> RowTileHasher {
        let row_bytes = cw * 4;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let (meta_lo, meta_hi) = leaf_meta(cw as u32, ch as u32);
        let mut lanes = init_lanes(key, meta_lo, meta_hi);
        let [rk0, rk1] = key.round_key;

        // SAFETY: Caller guarantees `absorbed_rows > 0`, `cw > 0`, and
        // `[y0..y0 + absorbed_rows)` is readable for `row_bytes` at `x_bytes`.
        let sh_ptr = unsafe { shadow_base.add(y0 * stride + x_bytes) };
        let first_raw = unsafe { u32::from_ne_bytes(sh_ptr.cast::<[u8; 4]>().read()) };
        let first_word = if HAS_ALPHA {
            first_raw
        } else {
            first_raw | XRGB_ALPHA_MASK_U32
        };
        let probe_ry = absorbed_rows - 1;
        let probe_rx = if absorbed_rows > 1 { cw >> 1 } else { cw - 1 };
        let probe_raw = unsafe {
            u32::from_ne_bytes(
                sh_ptr
                    .add(probe_ry * stride + probe_rx * 4)
                    .cast::<[u8; 4]>()
                    .read(),
            )
        };
        let probe_word = if HAS_ALPHA {
            probe_raw
        } else {
            probe_raw | XRGB_ALPHA_MASK_U32
        };
        let maybe_solid = first_word == probe_word;
        let maybe_opaque =
            !HAS_ALPHA || ((first_word & probe_word & XRGB_ALPHA_MASK_U32) == XRGB_ALPHA_MASK_U32);
        let first_pair = dup_u32_to_u64(first_word);

        // SAFETY: `sh_ptr` is readable for `absorbed_rows` rows of `row_bytes`.
        let (still_solid, still_opaque) = unsafe {
            absorb_rows_phased::<HAS_ALPHA, false>(
                sh_ptr,
                ptr::null_mut(),
                absorbed_rows,
                stride,
                full64,
                rem,
                rk0,
                rk1,
                first_word,
                first_pair,
                &mut lanes,
                maybe_solid,
                maybe_opaque,
            )
        };

        RowTileHasher {
            lanes: [[lanes[0], 0], [lanes[1], 0], [lanes[2], 0], [lanes[3], 0]],
            first_word,
            w: cw as u16,
            h: ch as u16,
            still_solid,
            still_opaque,
        }
    }

    #[inline(always)]
    unsafe fn absorb_one_hasher_row<const HAS_ALPHA: bool, const COPY: bool>(
        h: &mut RowTileHasher,
        src_row: *const u8,
        dst_row: *mut u8,
        rk0: u64,
        rk1: u64,
    ) {
        let row_bytes = usize::from(h.w) * 4;
        let full64 = row_bytes & !63;
        let rem = row_bytes & 63;
        let mut lanes = [h.lanes[0][0], h.lanes[1][0], h.lanes[2][0], h.lanes[3][0]];
        let first_pair = dup_u32_to_u64(h.first_word);
        // SAFETY: Caller guarantees `src_row` readable and (when `COPY`) `dst_row` writable.
        let (still_solid, still_opaque) = unsafe {
            absorb_rows_phased::<HAS_ALPHA, COPY>(
                src_row,
                dst_row,
                1,
                0,
                full64,
                rem,
                rk0,
                rk1,
                h.first_word,
                first_pair,
                &mut lanes,
                h.still_solid,
                h.still_opaque,
            )
        };
        h.lanes = [[lanes[0], 0], [lanes[1], 0], [lanes[2], 0], [lanes[3], 0]];
        h.still_solid = still_solid;
        h.still_opaque = still_opaque;
    }

    #[inline(never)]
    pub(super) unsafe fn process_band_row<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y0: usize,
        ch: usize,
        y: usize,
        actions: &mut [BandRowAction],
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let [rk0, rk1] = key.round_key;
        let row_off = y * stride;
        let absorbed_rows = (y - y0) + 1;
        // SAFETY: Caller validated row `y` bounds in `src_base` and `shadow_base`.
        let src_row_base = unsafe { src_base.add(row_off) };
        let sh_row_base = unsafe { shadow_base.add(row_off) };

        let mut x = 0usize;
        for (act, slot_h) in actions.iter_mut().zip(hashers.iter_mut()) {
            let cw = (width - x).min(SUBTILE_W);
            let x_bytes = x * 4;
            match *act {
                BandRowAction::Skip => {},
                BandRowAction::AbsorbShadow | BandRowAction::AbsorbShadowThenFull => {
                    if *act == BandRowAction::AbsorbShadowThenFull {
                        *act = BandRowAction::CopyAndAbsorb;
                    }
                    // SAFETY: Non-`Skip` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        absorb_one_hasher_row::<HAS_ALPHA, false>(
                            h,
                            sh_row_base.cast_const().add(x_bytes),
                            ptr::null_mut(),
                            rk0,
                            rk1,
                        );
                    }
                },
                BandRowAction::CopyAndAbsorb => {
                    // SAFETY: `CopyAndAbsorb` guarantees `slot_h` is initialized.
                    let h = unsafe { slot_h.assume_init_mut() };
                    unsafe {
                        absorb_one_hasher_row::<HAS_ALPHA, true>(
                            h,
                            src_row_base.add(x_bytes),
                            sh_row_base.add(x_bytes),
                            rk0,
                            rk1,
                        );
                    }
                },
                BandRowAction::StartAbsorb | BandRowAction::StartFull => {
                    let next = if *act == BandRowAction::StartFull {
                        BandRowAction::CopyAndAbsorb
                    } else {
                        BandRowAction::AbsorbShadow
                    };
                    // SAFETY: Rows `0..absorbed_rows` are finalized in `shadow_base`.
                    let h = unsafe {
                        start_row_hasher_from_shadow::<HAS_ALPHA>(
                            key,
                            shadow_base.cast_const(),
                            stride,
                            x_bytes,
                            y0,
                            cw,
                            ch,
                            absorbed_rows,
                        )
                    };
                    slot_h.write(h);
                    *act = next;
                },
            }
            x += cw;
        }
    }

    #[inline(never)]
    pub(super) unsafe fn copy_and_absorb_band_rows<const HAS_ALPHA: bool>(
        key: &TileHashKey,
        src_base: *const u8,
        shadow_base: *mut u8,
        stride: usize,
        width: usize,
        y_start: usize,
        y_end: usize,
        hashers: &mut [MaybeUninit<RowTileHasher>],
    ) {
        let [rk0, rk1] = key.round_key;
        for y in y_start..y_end {
            let row_off = y * stride;
            // SAFETY: Caller validated `[y_start, y_end)` bounds and exclusive shadow access.
            let src_row_base = unsafe { src_base.add(row_off) };
            let sh_row_base = unsafe { shadow_base.add(row_off) };
            let mut x = 0usize;
            for slot_h in hashers.iter_mut() {
                let cw = (width - x).min(SUBTILE_W);
                let x_bytes = x * 4;
                // SAFETY: Caller guarantees every slot in `hashers` is initialized.
                let h = unsafe { slot_h.assume_init_mut() };
                unsafe {
                    absorb_one_hasher_row::<HAS_ALPHA, true>(
                        h,
                        src_row_base.add(x_bytes),
                        sh_row_base.add(x_bytes),
                        rk0,
                        rk1,
                    );
                }
                x += cw;
            }
        }
    }

    #[inline(never)]
    pub(super) unsafe fn finish_band_hashers(
        key: &TileHashKey,
        actions: &[BandRowAction],
        hashers: &[MaybeUninit<RowTileHasher>],
        slot_base: usize,
        scans: &SlotScans,
    ) {
        for (qx, (&act, slot_h)) in actions.iter().zip(hashers.iter()).enumerate() {
            if act == BandRowAction::Skip {
                continue;
            }
            // SAFETY: Every non-`Skip` slot was initialized by `process_band_row`.
            let h = unsafe { slot_h.assume_init_ref() };
            let lanes = [h.lanes[0][0], h.lanes[1][0], h.lanes[2][0], h.lanes[3][0]];
            let (meta_lo, meta_hi) = leaf_meta(u32::from(h.w), u32::from(h.h));
            let res = TileHashResult {
                hash: finalize(key, lanes, meta_lo, meta_hi),
                meta: RectScanMeta {
                    solid_px: h.still_solid.then(|| h.first_word.to_ne_bytes()),
                    all_opaque: h.still_opaque,
                },
            };
            // SAFETY: `slot_base + qx` is exclusively owned by this worker.
            unsafe { scans.write(slot_base + qx, res) };
        }
    }

    pub(super) fn combine_quad_hashes(
        key: &TileHashKey,
        w: u32,
        h: u32,
        quad_hashes: &[TileHash],
    ) -> TileHash {
        let (meta_lo, meta_hi) = parent_meta(w, h, quad_hashes.len());
        let mut lanes = init_lanes(key, meta_lo, meta_hi);
        let [rk0, rk1] = key.round_key;
        for (lane, qh) in lanes.iter_mut().zip(quad_hashes.iter()) {
            let lo = u64::from_ne_bytes(qh[..8].try_into().unwrap());
            let hi = u64::from_ne_bytes(qh[8..].try_into().unwrap());
            *lane = step(*lane, lo, hi, rk0, rk1);
        }
        finalize(key, lanes, meta_lo, meta_hi)
    }
}

/// Computes the keyed 128-bit digest and in-register pixel scan metadata of
/// `rect` inside `shadow_base` (row stride `stride` bytes), normalizing byte 3
/// (`X`) to `0xFF` in-register when `!HAS_ALPHA` so arbitrary `Xrgb8888`
/// padding never causes false misses.
///
/// # Safety
/// `shadow_base` must be readable for `(rect.y + rect.h) * stride` bytes and
/// `(rect.x + rect.w) * 4 <= stride`, with no concurrent mutable writes to
/// `rect`'s pixels.
#[inline]
pub(crate) unsafe fn hash_rect_from_shadow<const HAS_ALPHA: bool>(
    key: &TileHashKey,
    shadow_base: *const u8,
    stride: usize,
    rect: TileRect,
) -> TileHashResult {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; caller upholds bounds invariants.
        return unsafe { x86_aes::hash_rect_raw::<HAS_ALPHA>(key, shadow_base, stride, rect) };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; caller upholds bounds invariants.
        return unsafe { aarch64_aes::hash_rect_raw::<HAS_ALPHA>(key, shadow_base, stride, rect) };
    }
    // SAFETY: Caller upholds bounds invariants.
    unsafe { portable::hash_rect_raw::<HAS_ALPHA>(key, shadow_base, stride, rect) }
}

/// Copies `rect` from `src` (`wl_shm` via [`BufferPointer`]) into `shadow_base`
/// for rows `[copy_from_ry..rect.h)` in a single load pass (reading initialized
/// `shadow_base` rows for `[0..copy_from_ry)`) and returns its keyed 128-bit
/// digest plus in-register [`RectScanMeta`] directly from registers without any
/// intermediate buffer.
///
/// Stores exact raw bytes from `src` into `shadow_base`, while masking byte 3
/// (`X`) to `0xFF` in-register for the digest and uniformity check when
/// `!HAS_ALPHA`.
///
/// # Safety
/// `shadow_base` must be exclusively writable for `rect` across `(rect.y +
/// rect.h) * stride` bytes with `(rect.x + rect.w) * 4 <= stride`, disjoint
/// from `src`, `copy_from_ry <= rect.h as usize`, and rows `[0..copy_from_ry)`
/// must already be initialized (rows `[copy_from_ry..rect.h)` may be
/// uninitialized spare capacity).
#[inline]
pub(crate) unsafe fn copy_and_hash_rect_to_shadow<const HAS_ALPHA: bool>(
    key: &TileHashKey,
    src: BufferPointer<u8>,
    shadow_base: *mut u8,
    stride: usize,
    rect: TileRect,
    copy_from_ry: usize,
) -> TileHashResult {
    let th = rect.h as usize;
    let row_bytes = rect.w as usize * 4;
    debug_assert!(copy_from_ry <= th);
    if copy_from_ry >= th {
        // SAFETY: All `th` rows are already finalized in `shadow_base`.
        return unsafe {
            hash_rect_from_shadow::<HAS_ALPHA>(key, shadow_base.cast_const(), stride, rect)
        };
    }
    if row_bytes > 0 {
        let last_end = (rect.y as usize + th - 1) * stride + rect.x as usize * 4 + row_bytes;
        assert!(last_end <= src.len());
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; `src` bounds checked above;
        // caller upholds `shadow_base` bounds and exclusivity invariants.
        return unsafe {
            x86_aes::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                rect,
                copy_from_ry,
            )
        };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; `src` bounds checked above;
        // caller upholds `shadow_base` bounds and exclusivity invariants.
        return unsafe {
            aarch64_aes::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                rect,
                copy_from_ry,
            )
        };
    }
    // SAFETY: `src` bounds checked above; caller upholds `shadow_base` bounds and exclusivity.
    unsafe {
        portable::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
            key,
            src.ptr(),
            shadow_base,
            stride,
            rect,
            copy_from_ry,
        )
    }
}

/// Computes the keyed 128-bit digest and in-register pixel scan metadata of
/// `rect` read through [`BufferPointer`] via raw loads (never creating `&[u8]`
/// over shared memory).
#[inline]
pub(crate) fn hash_rect_from_ptr<const HAS_ALPHA: bool>(
    key: &TileHashKey,
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    rect: TileRect,
) -> TileHashResult {
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let tw = rect.w as usize;
    let th = rect.h as usize;
    if th > 0 && tw > 0 {
        let last_end = (y0 + th - 1) * stride_px + x0 + tw;
        assert!(last_end <= aos.len());
    }
    let stride = stride_px * 4;
    let base = aos.ptr().cast::<u8>();
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; `aos` bounds validated above.
        return unsafe { x86_aes::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect) };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; `aos` bounds validated above.
        return unsafe { aarch64_aes::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect) };
    }
    // SAFETY: `aos` bounds validated above.
    unsafe { portable::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect) }
}

/// Processes one horizontal row `y` of active/starting quadrant hashers inside
/// a `RowBand2D`, absorbing L1d-hot `shadow` rows for still-diffing quadrants,
/// copying+absorbing `src -> shadow` linearly in-register for `bbox.full`
/// quadrants, and initializing newly triggered quadrants from L2-hot `shadow`
/// rows `0..=(y - y0)`.
///
/// # Safety
/// `shadow_base` must be exclusively writable for `[y0, y0 + ch)` across
/// `width * 4 <= stride` bytes, disjoint from `src`, `actions.len() ==
/// hashers.len() == width.div_ceil(SUBTILE_W)`, and every non-`Skip`/non-`Start*`
/// entry in `hashers` must already be initialized.
#[inline]
pub(crate) unsafe fn process_band_row<const HAS_ALPHA: bool>(
    key: &TileHashKey,
    src: BufferPointer<u8>,
    shadow_base: *mut u8,
    stride: usize,
    width: usize,
    y0: usize,
    ch: usize,
    y: usize,
    actions: &mut [BandRowAction],
    hashers: &mut [MaybeUninit<RowTileHasher>],
) {
    debug_assert_eq!(actions.len(), hashers.len());
    debug_assert!(y >= y0 && y < y0 + ch);
    if width > 0 {
        let row_end = y * stride + width * 4;
        assert!(row_end <= src.len());
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; `src` bounds checked above.
        unsafe {
            x86_aes::process_band_row::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                width,
                y0,
                ch,
                y,
                actions,
                hashers,
            );
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; `src` bounds checked above.
        unsafe {
            aarch64_aes::process_band_row::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                width,
                y0,
                ch,
                y,
                actions,
                hashers,
            );
        }
        return;
    }
    // SAFETY: `src` bounds checked above; caller upholds `shadow_base` invariants.
    unsafe {
        portable::process_band_row::<HAS_ALPHA>(
            key,
            src.ptr(),
            shadow_base,
            stride,
            width,
            y0,
            ch,
            y,
            actions,
            hashers,
        );
    }
}

/// Streams remaining rows `[y_start, y_end)` of a fully-escalated `RowBand2D`
/// linearly row-major, copying `src -> shadow` and absorbing every quadrant
/// hasher in-register in one contiguous pass.
///
/// # Safety
/// Every entry in `hashers` must already be initialized, and `shadow_base`
/// must be exclusively writable for `[y_start, y_end)` across `width * 4 <=
/// stride` bytes, disjoint from `src`.
#[inline]
pub(crate) unsafe fn copy_and_absorb_band_rows<const HAS_ALPHA: bool>(
    key: &TileHashKey,
    src: BufferPointer<u8>,
    shadow_base: *mut u8,
    stride: usize,
    width: usize,
    y_start: usize,
    y_end: usize,
    hashers: &mut [MaybeUninit<RowTileHasher>],
) {
    if y_start >= y_end || width == 0 {
        return;
    }
    let last_end = (y_end - 1) * stride + width * 4;
    assert!(last_end <= src.len());
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; `src` bounds checked above.
        unsafe {
            x86_aes::copy_and_absorb_band_rows::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                width,
                y_start,
                y_end,
                hashers,
            );
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; `src` bounds checked above.
        unsafe {
            aarch64_aes::copy_and_absorb_band_rows::<HAS_ALPHA>(
                key,
                src.ptr(),
                shadow_base,
                stride,
                width,
                y_start,
                y_end,
                hashers,
            );
        }
        return;
    }
    // SAFETY: `src` bounds checked above; caller upholds `shadow_base` invariants.
    unsafe {
        portable::copy_and_absorb_band_rows::<HAS_ALPHA>(
            key,
            src.ptr(),
            shadow_base,
            stride,
            width,
            y_start,
            y_end,
            hashers,
        );
    }
}

/// Finalizes every active quadrant hasher in `hashers` directly from registers
/// and writes its [`TileHashResult`] into `scans[slot_base + qx]`.
///
/// # Safety
/// Every non-[`BandRowAction::Skip`] slot in `hashers` must be initialized,
/// and `slot_base..slot_base + actions.len()` in `scans` must be exclusively
/// owned by this worker.
#[inline]
pub(crate) unsafe fn finish_band_hashers(
    key: &TileHashKey,
    actions: &[BandRowAction],
    hashers: &[MaybeUninit<RowTileHasher>],
    slot_base: usize,
    scans: &SlotScans,
) {
    debug_assert_eq!(actions.len(), hashers.len());
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI; caller upholds slot invariants.
        unsafe { x86_aes::finish_band_hashers(key, actions, hashers, slot_base, scans) };
        return;
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES; caller upholds slot invariants.
        unsafe { aarch64_aes::finish_band_hashers(key, actions, hashers, slot_base, scans) };
        return;
    }
    // SAFETY: Caller upholds slot invariants.
    unsafe { portable::finish_band_hashers(key, actions, hashers, slot_base, scans) };
}

/// Combines `2..=4` quadrant hashes into a single domain-separated keyed
/// 128-bit parent tile digest directly in registers without re-reading pixels.
#[inline]
pub(crate) fn combine_quad_hashes(
    key: &TileHashKey,
    w: u32,
    h: u32,
    quad_hashes: &[TileHash],
) -> TileHash {
    assert!((2..=4).contains(&quad_hashes.len()));
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified AES-NI.
        return unsafe { x86_aes::combine_quad_hashes(key, w, h, quad_hashes) };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if has_hw_aes() {
        // SAFETY: `has_hw_aes()` verified ARMv8 AES.
        return unsafe { aarch64_aes::combine_quad_hashes(key, w, h, quad_hashes) };
    }
    portable::combine_quad_hashes(key, w, h, quad_hashes)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ptr;

    use super::*;

    #[derive(Copy, Clone)]
    enum Backend {
        Portable,
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        X86Aes,
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        Aarch64Aes,
    }

    impl Backend {
        fn all_supported() -> Vec<Self> {
            #[cfg(any(
                all(target_arch = "x86_64", not(miri)),
                all(target_arch = "aarch64", not(miri)),
            ))]
            {
                let mut v = vec![Self::Portable];
                #[cfg(target_arch = "x86_64")]
                if has_hw_aes() {
                    v.push(Self::X86Aes);
                }
                #[cfg(target_arch = "aarch64")]
                if has_hw_aes() {
                    v.push(Self::Aarch64Aes);
                }
                v
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", not(miri)),
                all(target_arch = "aarch64", not(miri)),
            )))]
            {
                vec![Self::Portable]
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Portable => "portable",
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Self::X86Aes => "x86_aes",
                #[cfg(all(target_arch = "aarch64", not(miri)))]
                Self::Aarch64Aes => "aarch64_aes",
            }
        }

        unsafe fn hash_shadow<const HAS_ALPHA: bool>(
            self,
            key: &TileHashKey,
            base: *const u8,
            stride: usize,
            rect: TileRect,
        ) -> TileHashResult {
            match self {
                Self::Portable => unsafe {
                    portable::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect)
                },
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Self::X86Aes => unsafe {
                    x86_aes::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect)
                },
                #[cfg(all(target_arch = "aarch64", not(miri)))]
                Self::Aarch64Aes => unsafe {
                    aarch64_aes::hash_rect_raw::<HAS_ALPHA>(key, base, stride, rect)
                },
            }
        }

        unsafe fn copy_and_hash<const HAS_ALPHA: bool>(
            self,
            key: &TileHashKey,
            src: BufferPointer<u8>,
            shadow_base: *mut u8,
            stride: usize,
            rect: TileRect,
            copy_from_ry: usize,
        ) -> TileHashResult {
            match self {
                Self::Portable => unsafe {
                    portable::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                        key,
                        src.ptr(),
                        shadow_base,
                        stride,
                        rect,
                        copy_from_ry,
                    )
                },
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Self::X86Aes => unsafe {
                    x86_aes::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                        key,
                        src.ptr(),
                        shadow_base,
                        stride,
                        rect,
                        copy_from_ry,
                    )
                },
                #[cfg(all(target_arch = "aarch64", not(miri)))]
                Self::Aarch64Aes => unsafe {
                    aarch64_aes::copy_and_hash_rect_to_shadow::<HAS_ALPHA>(
                        key,
                        src.ptr(),
                        shadow_base,
                        stride,
                        rect,
                        copy_from_ry,
                    )
                },
            }
        }

        fn combine(self, key: &TileHashKey, w: u32, h: u32, quads: &[TileHash]) -> TileHash {
            match self {
                Self::Portable => portable::combine_quad_hashes(key, w, h, quads),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Self::X86Aes => unsafe { x86_aes::combine_quad_hashes(key, w, h, quads) },
                #[cfg(all(target_arch = "aarch64", not(miri)))]
                Self::Aarch64Aes => unsafe { aarch64_aes::combine_quad_hashes(key, w, h, quads) },
            }
        }
    }

    fn fill_pseudo_random(buf: &mut [u8], mut seed: u64) {
        for chunk in buf.chunks_mut(8) {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let b = z.to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
    }

    fn hamming_distance(a: TileHash, b: TileHash) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    fn check_rect_parity_case(
        backend: Backend,
        key: &TileHashKey,
        tw: usize,
        th: usize,
        x0: usize,
        y0: usize,
        pad_px: usize,
    ) {
        let full_w = x0 + tw + pad_px;
        let full_h = y0 + th + 2;
        let stride_px = full_w;
        let stride = stride_px * 4;
        let byte_len = full_h * stride;

        let mut src = vec![0u8; byte_len];
        fill_pseudo_random(
            &mut src,
            (tw as u64) << 16 | (th as u64) << 4 | pad_px as u64,
        );

        let mut src_garbage_x = src.clone();
        for px in src_garbage_x.as_chunks_mut::<4>().0 {
            px[3] = px[3].wrapping_mul(37).wrapping_add(0xA5);
        }

        let rect = TileRect {
            x: x0 as u32,
            y: y0 as u32,
            w: tw as u32,
            h: th as u32,
        };
        let src_ptr = src.as_ptr();
        let src_bp = unsafe { BufferPointer::new(&src_ptr, byte_len) };
        let aos_bp = unsafe { src_bp.cast::<Vec4u8>() };

        let x_ptr = src_garbage_x.as_ptr();
        let x_bp = unsafe { BufferPointer::new(&x_ptr, byte_len) };
        let x_aos_bp = unsafe { x_bp.cast::<Vec4u8>() };

        // Xrgb8888 (`HAS_ALPHA = false`)
        let h_shadow_xrgb =
            unsafe { backend.hash_shadow::<false>(key, src.as_ptr(), stride, rect) };
        let h_ptr_xrgb =
            unsafe { backend.hash_shadow::<false>(key, aos_bp.ptr().cast::<u8>(), stride, rect) };
        let h_garbage_xrgb =
            unsafe { backend.hash_shadow::<false>(key, x_aos_bp.ptr().cast::<u8>(), stride, rect) };
        assert_eq!(
            h_shadow_xrgb,
            h_ptr_xrgb,
            "{}: Xrgb shadow vs ptr mismatch on {tw}x{th}",
            backend.name()
        );
        assert_eq!(
            h_shadow_xrgb,
            h_garbage_xrgb,
            "{}: Xrgb garbage byte 3 changed hash on {tw}x{th}",
            backend.name()
        );

        // Argb8888 (`HAS_ALPHA = true`)
        let h_shadow_argb = unsafe { backend.hash_shadow::<true>(key, src.as_ptr(), stride, rect) };
        let h_ptr_argb =
            unsafe { backend.hash_shadow::<true>(key, aos_bp.ptr().cast::<u8>(), stride, rect) };
        assert_eq!(
            h_shadow_argb,
            h_ptr_argb,
            "{}: Argb shadow vs ptr mismatch on {tw}x{th}",
            backend.name()
        );

        // Flip 1 alpha bit inside `rect`: Argb must change, Xrgb must not.
        let mut src_alpha_flip = src.clone();
        let mid_off = (y0 + th / 2) * stride + (x0 + tw / 2) * 4 + 3;
        src_alpha_flip[mid_off] ^= 0x01;
        let h_flip_xrgb =
            unsafe { backend.hash_shadow::<false>(key, src_alpha_flip.as_ptr(), stride, rect) };
        let h_flip_argb =
            unsafe { backend.hash_shadow::<true>(key, src_alpha_flip.as_ptr(), stride, rect) };
        assert_eq!(h_flip_xrgb, h_shadow_xrgb);
        assert_ne!(
            h_flip_argb.hash,
            h_shadow_argb.hash,
            "{}: Argb missed 1-bit alpha flip on {tw}x{th}",
            backend.name()
        );

        // `copy_and_hash_rect_to_shadow` across `copy_from_ry = 0, th / 2, th`,
        // writing into genuinely uninitialized spare capacity for rows `copy_from_ry..th`
        // and verifying raw pixels are preserved verbatim in `shadow`.
        for copy_from_ry in [0usize, th / 2, th] {
            for (has_alpha, expected) in [(false, h_shadow_xrgb), (true, h_shadow_argb)] {
                let mut shadow: Vec<u8> = Vec::with_capacity(byte_len);
                let sh_ptr = shadow.as_mut_ptr();
                // Initialize rows `0..copy_from_ry` of `rect` inside `shadow`.
                for ry in 0..copy_from_ry {
                    let row_off = (y0 + ry) * stride + x0 * 4;
                    unsafe {
                        ptr::copy_nonoverlapping(
                            src.as_ptr().add(row_off),
                            sh_ptr.add(row_off),
                            tw * 4,
                        );
                    }
                }
                let got = if has_alpha {
                    unsafe {
                        backend.copy_and_hash::<true>(
                            key,
                            src_bp,
                            sh_ptr,
                            stride,
                            rect,
                            copy_from_ry,
                        )
                    }
                } else {
                    unsafe {
                        backend.copy_and_hash::<false>(
                            key,
                            x_bp,
                            sh_ptr,
                            stride,
                            rect,
                            copy_from_ry,
                        )
                    }
                };
                assert_eq!(
                    got,
                    expected,
                    "{}: copy_and_hash mismatch (has_alpha={has_alpha}, copy_from_ry={copy_from_ry}) on {tw}x{th}",
                    backend.name()
                );

                // Verify every copied row in `shadow` received exact raw bytes
                // (including raw X bytes on `Xrgb8888`, never overwritten with `0xFF`).
                let expected_raw_src = if has_alpha { &src } else { &src_garbage_x };
                for ry in copy_from_ry..th {
                    let row_off = (y0 + ry) * stride + x0 * 4;
                    let mut actual_row = vec![0u8; tw * 4];
                    unsafe {
                        ptr::copy_nonoverlapping(
                            sh_ptr.add(row_off),
                            actual_row.as_mut_ptr(),
                            tw * 4,
                        );
                    }
                    assert_eq!(
                        actual_row.as_slice(),
                        &expected_raw_src[row_off..row_off + tw * 4],
                        "{}: raw shadow bytes corrupted at ry={ry} on {tw}x{th}",
                        backend.name()
                    );
                }
            }
        }
    }

    fn check_solid_and_opaque_case(
        backend: Backend,
        key: &TileHashKey,
        tw: usize,
        th: usize,
        x0: usize,
        y0: usize,
        pad_px: usize,
    ) {
        let full_w = x0 + tw + pad_px;
        let full_h = y0 + th + 2;
        let stride = full_w * 4;
        let byte_len = full_h * stride;
        let rect = TileRect {
            x: x0 as u32,
            y: y0 as u32,
            w: tw as u32,
            h: th as u32,
        };
        let verify_both = |buf: &[u8], has_alpha: bool, expected: RectScanMeta, label: &str| {
            let bp_ptr = buf.as_ptr();
            let bp = unsafe { BufferPointer::new(&bp_ptr, byte_len) };
            let r_sh = if has_alpha {
                unsafe { backend.hash_shadow::<true>(key, buf.as_ptr(), stride, rect) }
            } else {
                unsafe { backend.hash_shadow::<false>(key, buf.as_ptr(), stride, rect) }
            };
            assert_eq!(
                r_sh.meta,
                expected,
                "{}: shadow meta mismatch ({label}) on {tw}x{th}",
                backend.name()
            );
            for copy_from_ry in [0usize, th / 2, th] {
                let mut shadow: Vec<u8> = Vec::with_capacity(byte_len);
                let sh_ptr = shadow.as_mut_ptr();
                for ry in 0..copy_from_ry {
                    let row_off = (y0 + ry) * stride + x0 * 4;
                    unsafe {
                        ptr::copy_nonoverlapping(
                            buf.as_ptr().add(row_off),
                            sh_ptr.add(row_off),
                            tw * 4,
                        );
                    }
                }
                let r_cp = if has_alpha {
                    unsafe {
                        backend.copy_and_hash::<true>(key, bp, sh_ptr, stride, rect, copy_from_ry)
                    }
                } else {
                    unsafe {
                        backend.copy_and_hash::<false>(key, bp, sh_ptr, stride, rect, copy_from_ry)
                    }
                };
                assert_eq!(
                    r_cp,
                    r_sh,
                    "{}: copy_and_hash mismatch ({label}, copy_from_ry={copy_from_ry}) on {tw}x{th}",
                    backend.name()
                );
            }
        };

        // 1. Solid Xrgb8888 with varying garbage byte 3 on every pixel ->
        //    canonical Solid([b, g, r, 0xFF]), all_opaque = true.
        let base_rgb = [0x24u8, 0x7A, 0xD1, 0xFF];
        let mut buf = vec![0x99u8; byte_len];
        for ry in 0..th {
            for rx in 0..tw {
                let off = (y0 + ry) * stride + (x0 + rx) * 4;
                buf[off] = base_rgb[0];
                buf[off + 1] = base_rgb[1];
                buf[off + 2] = base_rgb[2];
                buf[off + 3] = ((ry * 31 + rx * 17 + 5) & 0xFF) as u8;
            }
        }
        verify_both(
            &buf,
            false,
            RectScanMeta {
                solid_px: Some(base_rgb),
                all_opaque: true,
            },
            "xrgb_solid_garbage_x",
        );

        // 2. Solid opaque Argb8888 (`alpha = 0xFF`).
        for ry in 0..th {
            for rx in 0..tw {
                let off = (y0 + ry) * stride + (x0 + rx) * 4;
                buf[off..off + 4].copy_from_slice(&base_rgb);
            }
        }
        verify_both(
            &buf,
            true,
            RectScanMeta {
                solid_px: Some(base_rgb),
                all_opaque: true,
            },
            "argb_solid_opaque",
        );

        // 3. Solid translucent Argb8888 (`alpha = 0x80`): solid_px = Some, all_opaque = false.
        let translucent = [0x24u8, 0x7A, 0xD1, 0x80];
        for ry in 0..th {
            for rx in 0..tw {
                let off = (y0 + ry) * stride + (x0 + rx) * 4;
                buf[off..off + 4].copy_from_slice(&translucent);
            }
        }
        verify_both(
            &buf,
            true,
            RectScanMeta {
                solid_px: Some(translucent),
                all_opaque: false,
            },
            "argb_solid_translucent",
        );

        // 4. Single-pixel perturbations at the 3 pre-probe coordinates PLUS
        //    non-probe 4th pixels (verifying that matching `(0,0)`, `(tw/2,th/2)`,
        //    and `(tw-1,th-1)` still detects a differing 4th pixel as non-solid /
        //    non-opaque across both `copy_from_ry == 0` and overlap paths).
        let mut positions = vec![(0, 0), (tw / 2, th / 2), (tw - 1, th - 1)];
        for cand in [
            (tw - 1, 0),
            (0, th - 1),
            (1.min(tw - 1), 0),
            (0, 1.min(th - 1)),
        ] {
            if !positions.contains(&cand) {
                positions.push(cand);
            }
        }
        let is_single_px = tw == 1 && th == 1;
        for (px_x, px_y) in positions {
            // Reset to solid opaque.
            for ry in 0..th {
                for rx in 0..tw {
                    let off = (y0 + ry) * stride + (x0 + rx) * 4;
                    buf[off..off + 4].copy_from_slice(&base_rgb);
                }
            }
            let diff_off = (y0 + px_y) * stride + (x0 + px_x) * 4;

            // (a) Single-pixel RGB diff: non-solid (when >1 px), still all_opaque.
            buf[diff_off + 1] ^= 0x01;
            let expected_rgb_diff = RectScanMeta {
                solid_px: is_single_px
                    .then(|| [base_rgb[0], base_rgb[1] ^ 0x01, base_rgb[2], 0xFF]),
                all_opaque: true,
            };
            verify_both(&buf, false, expected_rgb_diff, "xrgb_single_rgb_diff");
            verify_both(&buf, true, expected_rgb_diff, "argb_single_rgb_diff");
            buf[diff_off + 1] ^= 0x01;

            // (b) Single-pixel alpha = 0xFE with identical RGB:
            //     Xrgb still solid + opaque; Argb non-opaque (and non-solid when >1 px).
            buf[diff_off + 3] = 0xFE;
            verify_both(
                &buf,
                false,
                RectScanMeta {
                    solid_px: Some(base_rgb),
                    all_opaque: true,
                },
                "xrgb_single_alpha_fe",
            );
            verify_both(
                &buf,
                true,
                RectScanMeta {
                    solid_px: is_single_px.then(|| [base_rgb[0], base_rgb[1], base_rgb[2], 0xFE]),
                    all_opaque: false,
                },
                "argb_single_alpha_fe",
            );
            buf[diff_off + 3] = 0xFF;

            // (c) Non-solid opaque probe points (`still_solid = false, still_opaque = true`)
            //     with a single non-probe translucent pixel (`alpha = 0xFE`):
            //     verifies the `!still_solid && still_opaque` fast loop catches `0xFE`.
            if !is_single_px && (px_x, px_y) != (0, 0) {
                let first_off = y0 * stride + x0 * 4;
                buf[first_off] ^= 0x10;
                buf[diff_off + 3] = 0xFE;
                verify_both(
                    &buf,
                    true,
                    RectScanMeta {
                        solid_px: None,
                        all_opaque: false,
                    },
                    "argb_nonsolid_single_alpha_fe",
                );
                buf[first_off] ^= 0x10;
                buf[diff_off + 3] = 0xFF;
            }

            // (d) Solid translucent probe points (`still_solid = true, still_opaque = false`)
            //     with a single differing RGB pixel -> non-solid + non-opaque.
            if !is_single_px {
                for ry in 0..th {
                    for rx in 0..tw {
                        let off = (y0 + ry) * stride + (x0 + rx) * 4;
                        buf[off..off + 4].copy_from_slice(&translucent);
                    }
                }
                buf[diff_off + 2] ^= 0x01;
                verify_both(
                    &buf,
                    true,
                    RectScanMeta {
                        solid_px: None,
                        all_opaque: false,
                    },
                    "argb_translucent_single_rgb_diff",
                );
            }
        }
    }

    /// Miri-safe parity and uninitialized-shadow soundness test (runs under both
    /// standard test execution and Miri).
    #[test]
    fn test_tile_hash_miri_uninit_and_edge_widths() {
        let key = TileHashKey::from_seed(0xC0FF_EE42);
        for backend in Backend::all_supported() {
            // Empty rect metadata check.
            let empty_buf = [0u8; 16];
            for r in [
                TileRect {
                    x: 0,
                    y: 0,
                    w: 0,
                    h: 4,
                },
                TileRect {
                    x: 0,
                    y: 0,
                    w: 4,
                    h: 0,
                },
            ] {
                let res_x =
                    unsafe { backend.hash_shadow::<false>(&key, empty_buf.as_ptr(), 16, r) };
                let res_a = unsafe { backend.hash_shadow::<true>(&key, empty_buf.as_ptr(), 16, r) };
                assert_eq!(res_x.meta, RectScanMeta::EMPTY);
                assert_eq!(res_a.meta, RectScanMeta::EMPTY);
            }

            for &tw in &[1usize, 2, 3, 4, 5, 15, 16, 17, 33] {
                for &th in &[1usize, 3] {
                    check_rect_parity_case(backend, &key, tw, th, 0, 0, 0);
                    check_rect_parity_case(backend, &key, tw, th, 3, 1, 7);
                    check_solid_and_opaque_case(backend, &key, tw, th, 0, 0, 0);
                    check_solid_and_opaque_case(backend, &key, tw, th, 2, 1, 5);
                }
            }
        }

        // `RectScanMeta::combine` algebraic invariants.
        let s1 = RectScanMeta {
            solid_px: Some([1, 2, 3, 0xFF]),
            all_opaque: true,
        };
        let s2 = RectScanMeta {
            solid_px: Some([4, 5, 6, 0xFF]),
            all_opaque: true,
        };
        let st = RectScanMeta {
            solid_px: Some([1, 2, 3, 0x80]),
            all_opaque: false,
        };
        let opq = RectScanMeta {
            solid_px: None,
            all_opaque: true,
        };
        assert_eq!(RectScanMeta::combine(&[s1, s1, s1, s1]), s1);
        assert_eq!(RectScanMeta::combine(&[s1, s2]), opq);
        assert_eq!(RectScanMeta::combine(&[s1, opq]), opq);
        assert_eq!(RectScanMeta::combine(&[st, st]), st);
        assert_eq!(RectScanMeta::combine(&[s1, st]), RectScanMeta::UNKNOWN);
        assert_eq!(
            RectScanMeta::combine(&[opq, RectScanMeta::UNKNOWN]),
            RectScanMeta::UNKNOWN
        );

        // Verify public dispatch wrappers end-to-end against uninitialized shadow.
        let tw = 19usize;
        let th = 5usize;
        let stride = (tw + 3) * 4;
        let byte_len = th * stride;
        let mut src = vec![0u8; byte_len];
        fill_pseudo_random(&mut src, 0x1234_5678);
        let rect = TileRect {
            x: 0,
            y: 0,
            w: tw as u32,
            h: th as u32,
        };
        let ptr = src.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, byte_len) };
        let aos = unsafe { bp.cast::<Vec4u8>() };

        let h_ptr = hash_rect_from_ptr::<false>(&key, aos, tw + 3, rect);
        let h_sh = unsafe { hash_rect_from_shadow::<false>(&key, src.as_ptr(), stride, rect) };
        let mut shadow: Vec<u8> = Vec::with_capacity(byte_len);
        let h_copy = unsafe {
            copy_and_hash_rect_to_shadow::<false>(&key, bp, shadow.as_mut_ptr(), stride, rect, 0)
        };
        assert_eq!(h_ptr, h_sh);
        assert_eq!(h_ptr, h_copy);
    }

    /// Full solid-color and `all_opaque` detection sweep across all widths,
    /// heights, strides, formats, and backends.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_hash_solid_and_all_opaque_detection() {
        let key = TileHashKey::new_random();
        let widths = [
            1usize, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256,
        ];
        let heights = [1usize, 3, 16, 128];

        for backend in Backend::all_supported() {
            for &tw in &widths {
                for &th in &heights {
                    check_solid_and_opaque_case(backend, &key, tw, th, 0, 0, 0);
                    check_solid_and_opaque_case(backend, &key, tw, th, 5, 2, 11);
                }
            }
        }
    }

    /// Full parity sweep across all required widths/heights, packed & padded
    /// strides, both `Xrgb8888` and `Argb8888`, on both hardware AES and portable.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_hash_full_width_height_parity() {
        let key = TileHashKey::new_random();
        let widths = [
            1usize, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256,
        ];
        let heights = [1usize, 3, 16, 128];

        for backend in Backend::all_supported() {
            for &tw in &widths {
                for &th in &heights {
                    check_rect_parity_case(backend, &key, tw, th, 0, 0, 0);
                    check_rect_parity_case(backend, &key, tw, th, 5, 2, 11);
                }
            }
        }

        // Exact end-of-allocation boundary check (no over-read past the last byte).
        for &tw in &widths {
            let th = 3usize;
            let stride = tw * 4;
            let mut src = vec![0u8; th * stride];
            fill_pseudo_random(&mut src, tw as u64);
            let rect = TileRect {
                x: 0,
                y: 0,
                w: tw as u32,
                h: th as u32,
            };
            let ptr = src.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, src.len()) };
            let aos = unsafe { bp.cast::<Vec4u8>() };
            let mut shadow = Vec::with_capacity(src.len());
            let h1 = hash_rect_from_ptr::<true>(&key, aos, tw, rect);
            let h2 = unsafe { hash_rect_from_shadow::<true>(&key, src.as_ptr(), stride, rect) };
            let h3 = unsafe {
                copy_and_hash_rect_to_shadow::<true>(&key, bp, shadow.as_mut_ptr(), stride, rect, 0)
            };
            assert_eq!(h1, h2);
            assert_eq!(h1, h3);
        }
    }

    /// Strict Avalanche Criterion (SAC), zero stuck bits, and zero collisions
    /// across pixels, dimensions, and `combine_quad_hashes` on every backend.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tile_hash_avalanche_and_collision_resistance() {
        let key = TileHashKey::from_seed(0xDEAD_BEEF_CAFE_BABE);

        for backend in Backend::all_supported() {
            // 1. Pixel single-bit flips across a 37x19 tile (exercises full
            //    64B lanes, 16B remainder lanes, and 4B tail lane).
            let tw = 37usize;
            let th = 19usize;
            let stride = tw * 4;
            let mut base = vec![0u8; th * stride];
            fill_pseudo_random(&mut base, 0x55AA_1122);
            let rect = TileRect {
                x: 0,
                y: 0,
                w: tw as u32,
                h: th as u32,
            };
            let base_hash =
                unsafe { backend.hash_shadow::<true>(&key, base.as_ptr(), stride, rect) }.hash;

            let mut total_flips = 0u64;
            let mut n_trials = 0u64;
            let mut or_diff = [0u8; 16];
            let byte_positions = [
                0,
                1,
                2,
                3,
                15,
                16,
                31,
                47,
                63,
                64,
                127,
                128,
                144,
                147,
                stride * (th / 2) + 5,
                stride * (th - 1),
                stride * (th - 1) + 64,
                stride * (th - 1) + 128,
                stride * (th - 1) + 144,
                stride * th - 1,
            ];
            for &pos in &byte_positions {
                for bit in 0..8u8 {
                    base[pos] ^= 1u8 << bit;
                    let h =
                        unsafe { backend.hash_shadow::<true>(&key, base.as_ptr(), stride, rect) }
                            .hash;
                    base[pos] ^= 1u8 << bit;
                    let dist = hamming_distance(base_hash, h);
                    assert!(
                        dist >= 34,
                        "{}: weak avalanche ({dist}/128 bits) flipping byte {pos} bit {bit}",
                        backend.name()
                    );
                    for i in 0..16 {
                        or_diff[i] |= base_hash[i] ^ h[i];
                    }
                    total_flips += u64::from(dist);
                    n_trials += 1;
                }
            }
            let mean_flips = total_flips as f64 / n_trials as f64;
            assert!(
                (58.0..=70.0).contains(&mean_flips),
                "{}: pixel SAC mean {mean_flips:.2} outside [58, 70]",
                backend.name()
            );
            assert_eq!(
                or_diff,
                [0xFF; 16],
                "{}: stuck output bit(s) in pixel SAC",
                backend.name()
            );

            // 2. `combine_quad_hashes` single-bit flips across all 4 quadrants.
            let mut quads = [[0u8; 16]; 4];
            for (q, slot) in quads.iter_mut().enumerate() {
                fill_pseudo_random(slot, 0x9000 + q as u64);
            }
            let base_parent = backend.combine(&key, 256, 256, &quads);
            let mut parent_flips = 0u64;
            let mut parent_trials = 0u64;
            let mut parent_or = [0u8; 16];
            for q in 0..4 {
                for byte in 0..16 {
                    for bit in [0u8, 3, 7] {
                        quads[q][byte] ^= 1u8 << bit;
                        let h = backend.combine(&key, 256, 256, &quads);
                        quads[q][byte] ^= 1u8 << bit;
                        let dist = hamming_distance(base_parent, h);
                        assert!(
                            dist >= 32,
                            "{}: weak parent avalanche ({dist}/128) at quad {q} byte {byte} bit {bit}",
                            backend.name()
                        );
                        for i in 0..16 {
                            parent_or[i] |= base_parent[i] ^ h[i];
                        }
                        parent_flips += u64::from(dist);
                        parent_trials += 1;
                    }
                }
            }
            let parent_mean = parent_flips as f64 / parent_trials as f64;
            assert!(
                (58.0..=70.0).contains(&parent_mean),
                "{}: parent SAC mean {parent_mean:.2} outside [58, 70]",
                backend.name()
            );
            assert_eq!(
                parent_or,
                [0xFF; 16],
                "{}: stuck output bit(s) in parent SAC",
                backend.name()
            );

            // 2b. Dimension `(w, h)` single-bit flips (across both leaf hashing
            //     and `combine_quad_hashes` metadata words).
            let mut dim_flips = 0u64;
            let mut dim_trials = 0u64;
            let mut dim_or = [0u8; 16];
            for bit in 0..32u32 {
                let hw = backend.combine(&key, 256 ^ (1u32 << bit), 256, &quads);
                let hh = backend.combine(&key, 256, 256 ^ (1u32 << bit), &quads);
                for h in [hw, hh] {
                    let dist = hamming_distance(base_parent, h);
                    assert!(
                        dist >= 32,
                        "{}: weak dim avalanche ({dist}/128) at bit {bit}",
                        backend.name()
                    );
                    for i in 0..16 {
                        dim_or[i] |= base_parent[i] ^ h[i];
                    }
                    dim_flips += u64::from(dist);
                    dim_trials += 1;
                }
            }
            let dim_mean = dim_flips as f64 / dim_trials as f64;
            assert!(
                (56.0..=72.0).contains(&dim_mean),
                "{}: dimension SAC mean {dim_mean:.2} outside [56, 72]",
                backend.name()
            );
            assert_eq!(
                dim_or,
                [0xFF; 16],
                "{}: stuck output bit(s) in dimension SAC",
                backend.name()
            );

            // 3. Zero collisions across 4,096 distinct tiles (near-identical
            //    1-bit-diff neighbors, solid tiles, and all-zero varying dims).
            let mut seen: HashSet<TileHash> = HashSet::with_capacity(5000);
            let mut tile = vec![0u8; 16 * 16 * 4];
            let r16 = TileRect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            };
            // 2,048 single-bit neighbors of an all-zero 16x16 tile.
            assert!(
                seen.insert(
                    unsafe { backend.hash_shadow::<true>(&key, tile.as_ptr(), 64, r16) }.hash
                )
            );
            for bit_idx in 0..2048usize {
                tile[bit_idx / 8] ^= 1u8 << (bit_idx % 8);
                let h = unsafe { backend.hash_shadow::<true>(&key, tile.as_ptr(), 64, r16) }.hash;
                assert!(
                    seen.insert(h),
                    "{}: collision on 1-bit neighbor {bit_idx}",
                    backend.name()
                );
                tile[bit_idx / 8] ^= 1u8 << (bit_idx % 8);
            }
            // 1,024 distinct solid-color 16x16 tiles.
            for c in 1..=1024u32 {
                let px = [
                    (c & 0xFF) as u8,
                    ((c >> 8) & 0xFF) as u8,
                    (c.wrapping_mul(37) & 0xFF) as u8,
                    0xFF,
                ];
                for dst in tile.as_chunks_mut::<4>().0 {
                    dst.copy_from_slice(&px);
                }
                let h = unsafe { backend.hash_shadow::<true>(&key, tile.as_ptr(), 64, r16) }.hash;
                assert!(
                    seen.insert(h),
                    "{}: collision on solid tile {c}",
                    backend.name()
                );
            }
            // 1,024 distinct all-zero dimensions `w in 1..=32, h in 1..=32`.
            let zeros = vec![0u8; 32 * 32 * 4];
            for w in 1..=32u32 {
                for h in 1..=32u32 {
                    if w == 16 && h == 16 {
                        continue;
                    }
                    let r = TileRect { x: 0, y: 0, w, h };
                    let digest = unsafe {
                        backend.hash_shadow::<true>(&key, zeros.as_ptr(), w as usize * 4, r)
                    }
                    .hash;
                    assert!(
                        seen.insert(digest),
                        "{}: collision on zero tile {w}x{h}",
                        backend.name()
                    );
                }
            }
        }
    }

    /// Domain separation (dimensions, leaf vs parent, quadrant order/count) and
    /// independent random keying across both backends.
    #[test]
    fn test_tile_hash_domain_separation_and_keying() {
        let key_a = TileHashKey::new_random();
        let key_b = TileHashKey::new_random();

        let mut buf = vec![0u8; 128 * 64 * 4];
        fill_pseudo_random(&mut buf, 0xFACE_B00C);

        for backend in Backend::all_supported() {
            // Same exact byte stream at 128x64 vs 64x128 must hash differently.
            let h_128x64 = unsafe {
                backend.hash_shadow::<true>(
                    &key_a,
                    buf.as_ptr(),
                    128 * 4,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 128,
                        h: 64,
                    },
                )
            }
            .hash;
            let h_64x128 = unsafe {
                backend.hash_shadow::<true>(
                    &key_a,
                    buf.as_ptr(),
                    64 * 4,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 64,
                        h: 128,
                    },
                )
            }
            .hash;
            assert_ne!(
                h_128x64,
                h_64x128,
                "{}: 128x64 collided with 64x128",
                backend.name()
            );

            // Different random keys must produce different digests for identical input.
            let h_key_b = unsafe {
                backend.hash_shadow::<true>(
                    &key_b,
                    buf.as_ptr(),
                    128 * 4,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 128,
                        h: 64,
                    },
                )
            }
            .hash;
            assert_ne!(
                h_128x64,
                h_key_b,
                "{}: distinct random keys produced identical digest",
                backend.name()
            );

            // Leaf vs `combine_quad_hashes` domain separation even when raw bytes match.
            let q0 = [0x11u8; 16];
            let q1 = [0x22u8; 16];
            let q2 = [0x33u8; 16];
            let q3 = [0x44u8; 16];
            let mut quad_as_pixels = [0u8; 64];
            quad_as_pixels[0..16].copy_from_slice(&q0);
            quad_as_pixels[16..32].copy_from_slice(&q1);
            quad_as_pixels[32..48].copy_from_slice(&q2);
            quad_as_pixels[48..64].copy_from_slice(&q3);

            let leaf_16x1 = unsafe {
                backend.hash_shadow::<true>(
                    &key_a,
                    quad_as_pixels.as_ptr(),
                    64,
                    TileRect {
                        x: 0,
                        y: 0,
                        w: 16,
                        h: 1,
                    },
                )
            }
            .hash;
            let parent_16x1 = backend.combine(&key_a, 16, 1, &[q0, q1, q2, q3]);
            assert_ne!(
                leaf_16x1,
                parent_16x1,
                "{}: leaf collided with combine_quad_hashes",
                backend.name()
            );

            // Quadrant order, count, dimensions, and key all separate parent hashes.
            let p_ab = backend.combine(&key_a, 256, 128, &[q0, q1]);
            let p_ba = backend.combine(&key_a, 256, 128, &[q1, q0]);
            let p_abc = backend.combine(&key_a, 256, 128, &[q0, q1, q2]);
            let p_dim = backend.combine(&key_a, 128, 256, &[q0, q1]);
            let p_key_b = backend.combine(&key_b, 256, 128, &[q0, q1]);
            assert_ne!(p_ab, p_ba);
            assert_ne!(p_ab, p_abc);
            assert_ne!(p_ab, p_dim);
            assert_ne!(p_ab, p_key_b);
        }
    }

    /// Verifies that incremental row-major [`RowTileHasher`] streaming
    /// (`process_band_row`, `copy_and_absorb_band_rows`, and
    /// `finish_band_hashers`) produces bit-identical 128-bit digests,
    /// [`RectScanMeta`], and raw shadow bytes to [`hash_rect_from_shadow`]
    /// across every supported backend (`Portable`, `X86Aes`, `Aarch64Aes`) and
    /// Miri.
    #[test]
    fn test_row_tile_hasher_streaming_parity() {
        let key = TileHashKey::from_seed(0x9876_5432_10FE_DCBA);

        #[cfg(not(miri))]
        let shapes = [
            (1usize, 1usize),
            (7, 8),
            (63, 19),
            (128, 128),
            (129, 17),
            (300, 32),
        ];
        #[cfg(miri)]
        let shapes = [(7usize, 4usize), (129, 6)];

        for backend in Backend::all_supported() {
            for &(width, ch) in &shapes {
                let stride = (width + 3) * 4;
                let y0 = 2usize;
                let total_h = y0 + ch + 1;
                let n_qx = width.div_ceil(SUBTILE_W);

                for has_alpha in [false, true] {
                    // Build a mixed source frame: Q0 non-solid, Q1 solid opaque,
                    // Q2 solid translucent (or garbage-X when `!has_alpha`).
                    let mut src = vec![0xAAu8; total_h * stride];
                    fill_pseudo_random(&mut src, (width as u64) * 97 + ch as u64);
                    if n_qx >= 2 {
                        let cw1 = (width - SUBTILE_W).min(SUBTILE_W);
                        for ry in 0..ch {
                            for rx in 0..cw1 {
                                let off = (y0 + ry) * stride + (SUBTILE_W + rx) * 4;
                                src[off..off + 4].copy_from_slice(&[0x12, 0x34, 0x56, 0xFF]);
                            }
                        }
                    }
                    if n_qx >= 3 {
                        let cw2 = (width - 2 * SUBTILE_W).min(SUBTILE_W);
                        for ry in 0..ch {
                            for rx in 0..cw2 {
                                let off = (y0 + ry) * stride + (2 * SUBTILE_W + rx) * 4;
                                src[off] = 0x77;
                                src[off + 1] = 0x88;
                                src[off + 2] = 0x99;
                                src[off + 3] = if has_alpha {
                                    0x80
                                } else {
                                    ((ry * 13 + rx * 7) & 0xFF) as u8
                                };
                            }
                        }
                    }

                    // Reference hashes over `src` directly.
                    let mut expected = Vec::with_capacity(n_qx);
                    for qx in 0..n_qx {
                        let x0 = qx * SUBTILE_W;
                        let cw = (width - x0).min(SUBTILE_W);
                        let qrect = TileRect {
                            x: x0 as u32,
                            y: y0 as u32,
                            w: cw as u32,
                            h: ch as u32,
                        };
                        let res = if has_alpha {
                            unsafe {
                                backend.hash_shadow::<true>(&key, src.as_ptr(), stride, qrect)
                            }
                        } else {
                            unsafe {
                                backend.hash_shadow::<false>(&key, src.as_ptr(), stride, qrect)
                            }
                        };
                        expected.push(res);
                    }

                    // Test transitions: start at row 0, mid-band, or last row,
                    // escalating to `CopyAndAbsorb` at `start_ry`, mid, or never.
                    let start_rows = [0usize, ch / 2, ch - 1];
                    for &start_ry in &start_rows {
                        for &escalate_full in &[false, true] {
                            let full_ry = if escalate_full {
                                (start_ry + 1).min(ch - 1)
                            } else {
                                ch
                            };
                            let mut shadow = vec![0x55u8; total_h * stride];
                            // Pre-populate rows `0..=start_ry` in shadow.
                            for ry in 0..=start_ry {
                                let off = (y0 + ry) * stride;
                                shadow[off..off + width * 4]
                                    .copy_from_slice(&src[off..off + width * 4]);
                            }
                            let mut actions = vec![
                                if escalate_full && start_ry == full_ry {
                                    BandRowAction::StartFull
                                } else {
                                    BandRowAction::StartAbsorb
                                };
                                n_qx
                            ];
                            let mut hashers = vec![MaybeUninit::<RowTileHasher>::uninit(); n_qx];

                            let run_row =
                                |y: usize,
                                 acts: &mut [BandRowAction],
                                 hs: &mut [MaybeUninit<RowTileHasher>],
                                 sh: &mut [u8]| {
                                    match backend {
                                        Backend::Portable => unsafe {
                                            if has_alpha {
                                                portable::process_band_row::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            } else {
                                                portable::process_band_row::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            }
                                        },
                                        #[cfg(all(target_arch = "x86_64", not(miri)))]
                                        Backend::X86Aes => unsafe {
                                            if has_alpha {
                                                x86_aes::process_band_row::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            } else {
                                                x86_aes::process_band_row::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            }
                                        },
                                        #[cfg(all(target_arch = "aarch64", not(miri)))]
                                        Backend::Aarch64Aes => unsafe {
                                            if has_alpha {
                                                aarch64_aes::process_band_row::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            } else {
                                                aarch64_aes::process_band_row::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    sh.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0,
                                                    ch,
                                                    y,
                                                    acts,
                                                    hs,
                                                );
                                            }
                                        },
                                    }
                                };

                            run_row(y0 + start_ry, &mut actions, &mut hashers, &mut shadow);

                            let mut ry = start_ry + 1;
                            while ry < ch {
                                if escalate_full && ry > full_ry {
                                    // Exercise `copy_and_absorb_band_rows` for the remaining tail.
                                    match backend {
                                        Backend::Portable => unsafe {
                                            if has_alpha {
                                                portable::copy_and_absorb_band_rows::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            } else {
                                                portable::copy_and_absorb_band_rows::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            }
                                        },
                                        #[cfg(all(target_arch = "x86_64", not(miri)))]
                                        Backend::X86Aes => unsafe {
                                            if has_alpha {
                                                x86_aes::copy_and_absorb_band_rows::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            } else {
                                                x86_aes::copy_and_absorb_band_rows::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            }
                                        },
                                        #[cfg(all(target_arch = "aarch64", not(miri)))]
                                        Backend::Aarch64Aes => unsafe {
                                            if has_alpha {
                                                aarch64_aes::copy_and_absorb_band_rows::<true>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            } else {
                                                aarch64_aes::copy_and_absorb_band_rows::<false>(
                                                    &key,
                                                    src.as_ptr(),
                                                    shadow.as_mut_ptr(),
                                                    stride,
                                                    width,
                                                    y0 + ry,
                                                    y0 + ch,
                                                    &mut hashers,
                                                );
                                            }
                                        },
                                    }
                                    break;
                                }
                                if ry <= full_ry {
                                    let off = (y0 + ry) * stride;
                                    shadow[off..off + width * 4]
                                        .copy_from_slice(&src[off..off + width * 4]);
                                    if escalate_full && ry == full_ry {
                                        actions.fill(BandRowAction::AbsorbShadowThenFull);
                                    }
                                }
                                run_row(y0 + ry, &mut actions, &mut hashers, &mut shadow);
                                ry += 1;
                            }

                            let mut scans = SlotScans::new(n_qx, key);
                            match backend {
                                Backend::Portable => unsafe {
                                    portable::finish_band_hashers(
                                        &key, &actions, &hashers, 0, &scans,
                                    );
                                },
                                #[cfg(all(target_arch = "x86_64", not(miri)))]
                                Backend::X86Aes => unsafe {
                                    x86_aes::finish_band_hashers(
                                        &key, &actions, &hashers, 0, &scans,
                                    );
                                },
                                #[cfg(all(target_arch = "aarch64", not(miri)))]
                                Backend::Aarch64Aes => unsafe {
                                    aarch64_aes::finish_band_hashers(
                                        &key, &actions, &hashers, 0, &scans,
                                    );
                                },
                            }

                            let scans_slice = scans.as_slice();
                            for (qx, &exp) in expected.iter().enumerate() {
                                let got = scans_slice[qx];
                                assert_eq!(
                                    got,
                                    exp,
                                    "{}: RowTileHasher mismatch {width}x{ch} qx={qx} has_alpha={has_alpha} start_ry={start_ry} full={escalate_full}",
                                    backend.name()
                                );
                            }
                            for r in 0..ch {
                                let off = (y0 + r) * stride;
                                assert_eq!(
                                    &shadow[off..off + width * 4],
                                    &src[off..off + width * 4],
                                    "{}: shadow copy mismatch {width}x{ch} ry={r}",
                                    backend.name()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
