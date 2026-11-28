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

//! Fast vertical scroll-blit detection and in-place scroll replay.
//!
//! Detects full-window and split-pane vertical scrolling (both smooth small-`dy`
//! scrolls and coarse wheel/page scrolls) on incremental same-size commits
//! before 2D sub-tile diffing:
//! - **Tier 0 activity screen**: zero-hash 16-pixel (`64 B`) SIMD probe grid
//!   (`step_x = 240`, `step_y <= 97`) that early-exits static and small-edit
//!   frames with minimal cache traffic and bounds active vertical ranges.
//! - **Phase 1b ruler sampling**: inserts up to 18 bursts of 7 consecutive
//!   `curr` rows into an L1d-resident 256-slot hash table and scans every 7th
//!   `prev` row so any integer shift `dy` (even `|dy| = 1`) hits every active
//!   burst while scanning `7x` fewer `prev` rows and avoiding aliasing with
//!   even UI line heights.
//! - **Phase 2 extent expansion**: row-major 16-pixel horizontal block scan,
//!   1D Kadane maximum-subarray selection, single-pixel X/Y edge refinement,
//!   and a 16x16 verification & profitability gate.

use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::buffer_pointer::BufferPointer;
use crate::filtering::RowKernel;
use crate::filtering::arch;
use crate::serialization::framing::RowBand;
use crate::serialization::framing::ScrollRect;
use crate::serialization::wayland::BufferFormat;
use crate::serialization::wayland::BufferMetadata;
use crate::sharding_compression::ShardingDecompressor;
use crate::vec4u8::Vec4u8;

/// Minimum scrolling region width and height (pixels) required to attempt or
/// emit a [`ScrollRect`].
const MIN_SCROLL_DIM: u32 = 256;

/// Minimum copied height (`height - |dy|`) for an accepted [`ScrollRect`].
const MIN_SCROLL_COPY_H: u32 = 64;

/// Minimum copied pixel area (`width * (height - |dy|)`) for an accepted
/// [`ScrollRect`].
const MIN_SCROLL_COPY_AREA: u64 = 256 * 256;

/// Width in pixels of each cache-line-aligned horizontal probe strip
/// (`16 * 4 = 64 B`, one cache line).
pub(crate) const STRIP_W: usize = 16;

/// Horizontal pixel offset of the first probe strip (clears left borders and
/// guarantees `MIN_SCROLL_DIM >= STRIP_START_X + STRIP_W`).
const STRIP_START_X: usize = 112;

/// Spacing in pixels between horizontal probe strips (`240 + 16 = 256`,
/// guaranteeing every `>= 256`px pane fully contains at least one strip).
const STRIP_STEP_X: usize = 240;

/// Maximum number of horizontal probe strips placed across a frame (covers up
/// to `32 * 240 = 7680`px 8K width).
pub(crate) const MAX_STRIPS: usize = 32;

/// Maximum vertical stride in Tier 0 (prime `97` avoids phase-locking with
/// common 16/18/20/24px UI line heights, and `256 > 2 * 97` guarantees every
/// `>= 256`px tall pane intersects at least two Tier-0 rows).
const TIER0_MAX_STEP_Y: usize = 97;

/// Prime row stride for the Phase 1b `prev` scan (coprime to common even UI
/// line heights); each `curr` anchor inserts `PREV_STEP` consecutive rows so
/// every integer `dy` matches once per anchor.
const PREV_STEP: usize = 7;

/// Maximum number of `PREV_STEP`-row bursts inserted per strip (`18 * 7 = 126`
/// rows, keeping the 256-slot L1d hash table load factor `<= 0.50`).
const MAX_ANCHORS: usize = 18;

/// Open-addressed L1d-resident hash table capacity (`256 * 16 B = 4 KiB`).
const HASH_SLOTS: usize = 256;
const HASH_MASK: usize = HASH_SLOTS - 1;

/// Minimum unique non-uniform row hash votes required to promote a `dy`
/// candidate into Phase 2 extent expansion.
const MIN_VOTES: u16 = 3;

/// Maximum tracked candidate `dy` buckets during Phase 1b `prev` voting.
const VOTE_SLOTS: usize = 64;

/// Coarse column-block width (pixels) for horizontal extent expansion.
const X_BLOCK_W: usize = 16;

/// Stack buffer capacity for coarse X/Y Kadane scores (`512 * 16 = 8192`px
/// without heap allocation).
const STACK_SCORE_CAP: usize = 512;

/// Minimum byte payload before [`apply_scroll_rect_in_place_parallel`] wakes
/// background worker threads.
const PARALLEL_SCROLL_MIN_BYTES: usize = 512 * 1024;

/// Minimum byte payload before [`apply_scroll_rect_in_place_parallel`] and
/// swapchain scroll blits emit non-temporal streaming stores into the
/// destination buffer (keeps smaller L2/L3-resident buffers warm in cache while
/// bypassing destination RFO traffic on large 4K frames).
pub(crate) const STREAM_SCROLL_MIN_BYTES: usize = 12 * 1024 * 1024;

/// Minimum per-worker byte reuse distance (`|dy| * stripe_bytes`) before
/// non-temporal streaming stores are enabled on overlapping (`|dy| < copy_h`)
/// in-place scrolls. Below private L2 capacity (`~512 KiB`), every destination
/// cache line was just fetched into L1D/L2 by the `src` read `|dy|` rows
/// earlier, so regular stores pay zero RFO traffic whereas `vmovntdq` triggers
/// expensive cache-hit evictions and write-combining flushes.
const STREAM_OVERLAP_MIN_REUSE_BYTES: usize = 512 * 1024;

/// Minimum vertical column stripe width (bytes) when partitioning overlapping
/// (`|dy| < copy_h`) parallel scroll blits across workers.
const MIN_VERTICAL_STRIPE_BYTES: usize = 512;

/// 16 distinct odd 32-bit lane multipliers (MSB set) for 16-pixel (`64 B`)
/// strip hashing across scalar and SIMD tiers.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        not(miri),
        any(
            target_feature = "ssse3",
            target_feature = "avx2",
            all(
                target_feature = "avx512bw",
                target_feature = "avx512vl",
                target_feature = "avx512vbmi",
            ),
        ),
    ),
    allow(dead_code)
)]
pub(crate) const STRIP_HASH_MULS: [u32; STRIP_W] = [
    0x9E37_79B1,
    0x85EB_CA77,
    0xC2B2_AE3D,
    0xA24B_AED5,
    0xB7E1_5163,
    0xD1B5_4A35,
    0x94D0_49BB,
    0xE703_7ED1,
    0x9E37_79B9,
    0xC6BC_2797,
    0xA5A3_564F,
    0xD6E8_FEB9,
    0xB529_7A4D,
    0xE19B_01AB,
    0x9FB2_1C65,
    0xF135_7AEB,
];

/// Multiplicative 64-bit mixing constants for strip row hash finalization.
const HASH_MUL1: u64 = 0x517c_c1b7_2722_0a95;
const HASH_MUL2: u64 = 0x9e37_79b9_7f4a_7c15;

/// Finalizes the two folded 64-bit strip hash lanes `(x0, x1)` into a 64-bit
/// row hash shared identically across scalar and SIMD tiers.
#[inline(always)]
pub(crate) fn finalize_strip_hash(x0: u64, x1: u64) -> u64 {
    let h = (x0.wrapping_mul(HASH_MUL1) ^ x1.rotate_left(29)).wrapping_mul(HASH_MUL2);
    h ^ (h >> 32)
}

/// Applies a validated [`ScrollRect`] in place on a pixel buffer of `stride`
/// bytes per row (`buf.len() >= (scroll.y + scroll.height) * stride`).
pub fn apply_scroll_rect_in_place(buf: &mut [u8], stride: usize, scroll: ScrollRect) {
    let x_bytes = scroll.x as usize * 4;
    let row_bytes = scroll.width as usize * 4;
    let shift = scroll.dy.unsigned_abs() as usize;
    let copy_h = scroll.height as usize - shift;
    let y0 = scroll.y as usize;
    if shift == 0 || copy_h == 0 || row_bytes == 0 {
        return;
    }

    if scroll.dy < 0 {
        // Content moves up: copy top-to-bottom (`dst_y < src_y`).
        if row_bytes == stride && x_bytes == 0 {
            let src_start = (y0 + shift) * stride;
            let src_end = (y0 + scroll.height as usize) * stride;
            let dst_start = y0 * stride;
            buf.copy_within(src_start..src_end, dst_start);
        } else {
            for ry in 0..copy_h {
                let src_off = (y0 + shift + ry) * stride + x_bytes;
                let dst_off = (y0 + ry) * stride + x_bytes;
                buf.copy_within(src_off..src_off + row_bytes, dst_off);
            }
        }
    } else {
        // Content moves down: copy bottom-to-top (`dst_y > src_y`).
        if row_bytes == stride && x_bytes == 0 {
            let src_start = y0 * stride;
            let src_end = (y0 + copy_h) * stride;
            let dst_start = (y0 + shift) * stride;
            buf.copy_within(src_start..src_end, dst_start);
        } else {
            for ry in (0..copy_h).rev() {
                let src_off = (y0 + ry) * stride + x_bytes;
                let dst_off = (y0 + shift + ry) * stride + x_bytes;
                buf.copy_within(src_off..src_off + row_bytes, dst_off);
            }
        }
    }
}

struct SyncMutBuf {
    base: *mut u8,
}

impl SyncMutBuf {
    #[inline(always)]
    fn base(&self) -> *mut u8 {
        self.base
    }
}

// SAFETY: `apply_scroll_rect_in_place_parallel_with` partitions work into
// pairwise disjoint horizontal row bands (when `shift >= copy_h`) or pairwise
// disjoint vertical column stripes (when `shift < copy_h`), and each task is
// claimed at most once via `AtomicUsize`.
unsafe impl Send for SyncMutBuf {}
unsafe impl Sync for SyncMutBuf {}

/// Computes the unified `(rows, row_len, first_src_row, first_dst_row, step)`
/// traversal parameters for a scroll blit stripe or band.
#[inline(always)]
fn scroll_row_plan(
    stride: usize,
    sx: usize,
    len: usize,
    src_y0: usize,
    dst_y0: usize,
    copy_h: usize,
    rev: bool,
    disjoint: bool,
) -> (usize, usize, usize, usize, isize) {
    if disjoint && sx == 0 && len == stride {
        (1usize, copy_h * stride, src_y0, dst_y0, 0isize)
    } else if !rev {
        (copy_h, len, src_y0, dst_y0, stride as isize)
    } else {
        (
            copy_h,
            len,
            src_y0 + copy_h - 1,
            dst_y0 + copy_h - 1,
            -(stride as isize),
        )
    }
}

/// Returns `true` when each scroll row stripe jumps across at least one 4 KiB
/// page boundary (breaking hardware stride prefetching) while remaining small
/// enough to prefetch 2 rows ahead into L1d without evicting the active row.
#[inline(always)]
pub(super) fn should_prefetch_scroll_stripe(stride: usize, row_len: usize) -> bool {
    stride.saturating_sub(row_len) >= super::prefetch::MIN_PAGE_STRIDE_BYTES
        && row_len <= super::prefetch::MIN_PAGE_STRIDE_BYTES
}

/// Copies `copy_h` rows of `len` bytes at horizontal byte offset `sx` within
/// `base` (`stride` bytes/row) from `src_y0` to `dst_y0` using regular temporal
/// `memcpy` (`ptr::copy_nonoverlapping`).
///
/// # Safety
/// Every accessed row `[y * stride + sx .. y * stride + sx + len)` must lie
/// within `base`'s allocation, writes must be exclusively owned by this worker,
/// `src_y0 != dst_y0` must hold, and when `disjoint` is `true`, the full `src`
/// and `dst` row ranges must not overlap.
#[inline]
pub(super) unsafe fn apply_scroll_rows_temporal(
    base: *mut u8,
    stride: usize,
    sx: usize,
    len: usize,
    src_y0: usize,
    dst_y0: usize,
    copy_h: usize,
    rev: bool,
    disjoint: bool,
) {
    if copy_h == 0 || len == 0 {
        return;
    }
    let (rows, row_len, first_src_row, first_dst_row, step) =
        scroll_row_plan(stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint);
    // SAFETY: Caller guarantees every `[y * stride + sx .. y * stride + sx + len)`
    // is in-bounds and non-overlapping per row (and across all rows when `disjoint`).
    // `wrapping_offset` keeps the final post-loop pointer step UB-free.
    unsafe {
        let mut s = base.add(first_src_row * stride + sx).cast_const();
        let mut d = base.add(first_dst_row * stride + sx);
        if should_prefetch_scroll_stripe(stride, row_len) {
            let pf_step = step * super::prefetch::SCROLL_DECODE_ROW_LOOKAHEAD as isize;
            for i in 0..rows {
                if i + super::prefetch::SCROLL_DECODE_ROW_LOOKAHEAD < rows {
                    super::prefetch::row_t0(s.offset(pf_step), row_len);
                }
                ptr::copy_nonoverlapping(s, d, row_len);
                s = s.wrapping_offset(step);
                d = d.wrapping_offset(step);
            }
        } else {
            for _ in 0..rows {
                ptr::copy_nonoverlapping(s, d, row_len);
                s = s.wrapping_offset(step);
                d = d.wrapping_offset(step);
            }
        }
    }
}

/// Copies `copy_h` rows of `len` bytes at horizontal byte offset `sx` within
/// `base` (`stride` bytes/row) from `src_y0` to `dst_y0` using cache-line-aligned
/// non-temporal streaming stores and a single trailing [`RowKernel::fence_stream`].
///
/// Specialized on `PREFETCH_SRC` so the non-prefetching full-width path stays
/// single-loop and keeps `K::copy_bytes_stream` inlined in both variants.
///
/// # Safety
/// Caller must uphold [`apply_scroll_rows_temporal`] invariants.
#[cfg_attr(not(all(target_arch = "x86_64", not(miri))), allow(dead_code))]
#[inline(always)]
pub(super) unsafe fn apply_scroll_rows_stream_with<K: RowKernel, const PREFETCH_SRC: bool>(
    kernel: K,
    base: *mut u8,
    stride: usize,
    sx: usize,
    len: usize,
    src_y0: usize,
    dst_y0: usize,
    copy_h: usize,
    rev: bool,
    disjoint: bool,
) {
    if copy_h == 0 || len == 0 {
        return;
    }
    let (rows, row_len, first_src_row, first_dst_row, step) =
        scroll_row_plan(stride, sx, len, src_y0, dst_y0, copy_h, rev, disjoint);
    // SAFETY: Caller guarantees every `[y * stride + sx .. y * stride + sx + len)`
    // is in-bounds and non-overlapping per row (and across all rows when `disjoint`).
    // `wrapping_offset` keeps the final post-loop pointer step UB-free.
    unsafe {
        let mut s = base.add(first_src_row * stride + sx).cast_const();
        let mut d = base.add(first_dst_row * stride + sx);
        if PREFETCH_SRC {
            let pf_step = step * super::prefetch::SCROLL_DECODE_ROW_LOOKAHEAD as isize;
            for i in 0..rows {
                if i + super::prefetch::SCROLL_DECODE_ROW_LOOKAHEAD < rows {
                    // Never prefetch `d` on NT streaming stores; only prefetch strided `s`.
                    super::prefetch::row_t0(s.offset(pf_step), row_len);
                }
                kernel.copy_bytes_stream(s, d, row_len);
                s = s.wrapping_offset(step);
                d = d.wrapping_offset(step);
            }
        } else {
            for _ in 0..rows {
                kernel.copy_bytes_stream(s, d, row_len);
                s = s.wrapping_offset(step);
                d = d.wrapping_offset(step);
            }
        }
    }
    kernel.fence_stream();
}

/// Copies `rows` tightly-packed rows of `row_bytes` from `src_packed` into
/// `dst` at `(x_bytes, dst_y0)` (`dst_stride` bytes/row) using cache-line-aligned
/// non-temporal streaming stores and a single trailing [`RowKernel::fence_stream`].
#[inline(always)]
pub(super) fn copy_packed_rows_stream_with<K: RowKernel>(
    kernel: K,
    src_packed: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    x_bytes: usize,
    dst_y0: usize,
    row_bytes: usize,
    rows: usize,
) {
    if rows == 0 || row_bytes == 0 {
        return;
    }
    assert!(src_packed.len() >= rows * row_bytes);
    assert!(x_bytes + row_bytes <= dst_stride);
    assert!(dst.len() >= (dst_y0 + rows) * dst_stride);
    if x_bytes == 0 && row_bytes == dst_stride {
        // SAFETY: Bounds asserted above; `src_packed` and `dst` are distinct slices.
        unsafe {
            kernel.copy_bytes_stream(
                src_packed.as_ptr(),
                dst.as_mut_ptr().add(dst_y0 * dst_stride),
                rows * row_bytes,
            );
        }
    } else {
        for i in 0..rows {
            let dst_off = (dst_y0 + i) * dst_stride + x_bytes;
            // SAFETY: Bounds asserted above; `src_packed` and `dst` are distinct slices.
            unsafe {
                kernel.copy_bytes_stream(
                    src_packed.as_ptr().add(i * row_bytes),
                    dst.as_mut_ptr().add(dst_off),
                    row_bytes,
                );
            }
        }
    }
    kernel.fence_stream();
}

/// Copies `rows` tightly-packed rows of `row_bytes` from `src_packed` into
/// `dst` (typically a `wl_shm` swapchain slot) using non-temporal streaming
/// stores on the highest host SIMD tier, issuing one `sfence` per batch.
pub(crate) fn copy_packed_rows_stream(
    src_packed: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    x_bytes: usize,
    dst_y0: usize,
    row_bytes: usize,
    rows: usize,
) {
    arch::Tier::detect().copy_packed_rows_stream(
        src_packed, dst, dst_stride, x_bytes, dst_y0, row_bytes, rows,
    );
}

/// Applies a validated [`ScrollRect`] in place using the supplied SIMD `tier`
/// and streaming-store mode, partitioning disjoint (`|dy| >= copy_h`) blits
/// into horizontal row bands and overlapping (`|dy| < copy_h`) blits into
/// cache-line-aligned vertical column stripes.
pub(super) fn apply_scroll_rect_in_place_parallel_with(
    buf: &mut [u8],
    stride: usize,
    scroll: ScrollRect,
    decompressor: &mut ShardingDecompressor,
    tier: arch::Tier,
    stream: bool,
) {
    let x_bytes = scroll.x as usize * 4;
    let row_bytes = scroll.width as usize * 4;
    let shift = scroll.dy.unsigned_abs() as usize;
    let copy_h = scroll.height as usize - shift;
    let y0 = scroll.y as usize;
    assert!(x_bytes + row_bytes <= stride);
    assert!(buf.len() >= (y0 + scroll.height as usize) * stride);
    if shift == 0 || copy_h == 0 || row_bytes == 0 {
        return;
    }

    let move_up = scroll.dy < 0;
    let src_y0 = if move_up { y0 + shift } else { y0 };
    let dst_y0 = if move_up { y0 } else { y0 + shift };
    let disjoint = shift >= copy_h;
    let copy_bytes = copy_h.saturating_mul(row_bytes);
    let n_threads = decompressor.pool.parallelism();
    let use_stream = stream && tier.stream_align() > 1;

    if n_threads <= 1 || copy_bytes < PARALLEL_SCROLL_MIN_BYTES {
        // SAFETY: Bounds asserted above, single-threaded exclusive `buf`, `shift >= 1`.
        unsafe {
            if use_stream {
                tier.apply_scroll_rows_stream(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    disjoint,
                );
            } else {
                apply_scroll_rows_temporal(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    disjoint,
                );
            }
        }
        return;
    }

    if disjoint {
        let n_bands = n_threads.min(copy_h);
        if n_bands <= 1 {
            // SAFETY: Bounds asserted above, single band, `shift >= copy_h`.
            unsafe {
                if use_stream {
                    tier.apply_scroll_rows_stream(
                        buf.as_mut_ptr(),
                        stride,
                        x_bytes,
                        row_bytes,
                        src_y0,
                        dst_y0,
                        copy_h,
                        false,
                        true,
                    );
                } else {
                    apply_scroll_rows_temporal(
                        buf.as_mut_ptr(),
                        stride,
                        x_bytes,
                        row_bytes,
                        src_y0,
                        dst_y0,
                        copy_h,
                        false,
                        true,
                    );
                }
            }
            return;
        }
        let sync_buf = SyncMutBuf {
            base: buf.as_mut_ptr(),
        };
        let next_band = AtomicUsize::new(0);
        decompressor.pool.execute(n_bands, |_| {
            loop {
                let idx = next_band.fetch_add(1, Ordering::Relaxed);
                if idx >= n_bands {
                    break;
                }
                let r0 = (idx * copy_h) / n_bands;
                let r1 = ((idx + 1) * copy_h) / n_bands;
                let band_h = r1 - r0;
                // SAFETY: `disjoint` guarantees `[src_y0, src_y0 + copy_h)` and
                // `[dst_y0, dst_y0 + copy_h)` are completely disjoint, and
                // horizontal bands `[r0, r1)` partition `0..copy_h` without overlap.
                unsafe {
                    if use_stream {
                        tier.apply_scroll_rows_stream(
                            sync_buf.base(),
                            stride,
                            x_bytes,
                            row_bytes,
                            src_y0 + r0,
                            dst_y0 + r0,
                            band_h,
                            false,
                            true,
                        );
                    } else {
                        apply_scroll_rows_temporal(
                            sync_buf.base(),
                            stride,
                            x_bytes,
                            row_bytes,
                            src_y0 + r0,
                            dst_y0 + r0,
                            band_h,
                            false,
                            true,
                        );
                    }
                }
            }
        });
        return;
    }

    if row_bytes < MIN_VERTICAL_STRIPE_BYTES * 2 {
        // SAFETY: Bounds asserted above, single-threaded exclusive `buf`, `shift >= 1`.
        unsafe {
            if use_stream {
                tier.apply_scroll_rows_stream(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    false,
                );
            } else {
                apply_scroll_rows_temporal(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    false,
                );
            }
        }
        return;
    }

    let n_stripes = n_threads.min(row_bytes / MIN_VERTICAL_STRIPE_BYTES).max(1);
    if n_stripes <= 1 {
        // SAFETY: Bounds asserted above, single-threaded exclusive `buf`, `shift >= 1`.
        unsafe {
            if use_stream {
                tier.apply_scroll_rows_stream(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    false,
                );
            } else {
                apply_scroll_rows_temporal(
                    buf.as_mut_ptr(),
                    stride,
                    x_bytes,
                    row_bytes,
                    src_y0,
                    dst_y0,
                    copy_h,
                    !move_up,
                    false,
                );
            }
        }
        return;
    }

    // Snap interior vertical stripe boundaries to actual 64 B cache-line
    // addresses of the first row so middle stripes have zero false sharing and
    // zero unaligned head/tail bytes whenever `stride % 64 == 0`.
    let first_row_addr = buf.as_ptr() as usize + y0 * stride + x_bytes;
    let end_addr = first_row_addr + row_bytes;
    let align_interior = |off: usize| -> usize {
        let aligned = (first_row_addr + off + 32) & !63;
        aligned.clamp(first_row_addr, end_addr) - first_row_addr
    };

    let sync_buf = SyncMutBuf {
        base: buf.as_mut_ptr(),
    };
    let next_stripe = AtomicUsize::new(0);
    decompressor.pool.execute(n_stripes, |_| {
        loop {
            let idx = next_stripe.fetch_add(1, Ordering::Relaxed);
            if idx >= n_stripes {
                break;
            }
            let start = if idx == 0 {
                0
            } else {
                align_interior((idx * row_bytes) / n_stripes)
            };
            let end = if idx + 1 == n_stripes {
                row_bytes
            } else {
                align_interior(((idx + 1) * row_bytes) / n_stripes)
            };
            let len = end - start;
            if len == 0 {
                continue;
            }
            let sx = x_bytes + start;
            // SAFETY: Stripes `[start, end)` form an exact monotone partition
            // of `[0, row_bytes)` (adjacent stripes share the identical
            // `align_interior` split point), `shift >= 1` ensures `src` and
            // `dst` lie on distinct rows, and `!move_up` sets overlap-safe order.
            unsafe {
                if use_stream {
                    tier.apply_scroll_rows_stream(
                        sync_buf.base(),
                        stride,
                        sx,
                        len,
                        src_y0,
                        dst_y0,
                        copy_h,
                        !move_up,
                        false,
                    );
                } else {
                    apply_scroll_rows_temporal(
                        sync_buf.base(),
                        stride,
                        sx,
                        len,
                        src_y0,
                        dst_y0,
                        copy_h,
                        !move_up,
                        false,
                    );
                }
            }
        }
    });
}

/// Applies a validated [`ScrollRect`] in place, parallelizing large blits
/// across `decompressor`'s worker pool via horizontal row bands (when
/// `|dy| >= copy_h`) or cache-line-aligned vertical column stripes (when
/// `|dy| < copy_h`), using non-temporal streaming stores on large `wl_shm`
/// payloads to avoid destination RFO traffic.
pub fn apply_scroll_rect_in_place_parallel(
    buf: &mut [u8],
    stride: usize,
    scroll: ScrollRect,
    decompressor: &mut ShardingDecompressor,
) {
    let row_bytes = scroll.width as usize * 4;
    let shift = scroll.dy.unsigned_abs() as usize;
    let copy_h = (scroll.height as usize).saturating_sub(shift);
    let copy_bytes = copy_h.saturating_mul(row_bytes);
    let disjoint = shift >= copy_h;
    let n_threads = decompressor.pool.parallelism();
    let n_stripes = if n_threads <= 1
        || copy_bytes < PARALLEL_SCROLL_MIN_BYTES
        || row_bytes < MIN_VERTICAL_STRIPE_BYTES * 2
    {
        1
    } else {
        n_threads.min(row_bytes / MIN_VERTICAL_STRIPE_BYTES).max(1)
    };
    let stripe_bytes = row_bytes / n_stripes;
    let stream = copy_bytes >= STREAM_SCROLL_MIN_BYTES
        && (disjoint || shift.saturating_mul(stripe_bytes) >= STREAM_OVERLAP_MIN_REUSE_BYTES);
    apply_scroll_rect_in_place_parallel_with(
        buf,
        stride,
        scroll,
        decompressor,
        arch::Tier::detect(),
        stream,
    );
}

#[inline(always)]
fn pixel_key<const HAS_ALPHA: bool>(px: Vec4u8) -> u32 {
    let raw = u32::from_le_bytes([px.0, px.1, px.2, px.3]);
    if HAS_ALPHA { raw } else { raw & 0x00FF_FFFF }
}

/// Reads a single pixel from `aos` at `(y, x)` without forming a Rust slice
/// over potentially shared memory.
///
/// # Safety
/// `y * stride_px + x < aos.len()`.
#[inline(always)]
unsafe fn read_curr_px<const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    stride_px: usize,
    y: usize,
    x: usize,
) -> u32 {
    // SAFETY: Caller guarantees `y * stride_px + x < aos.len()`.
    let px = unsafe { aos.ptr().add(y * stride_px + x).read() };
    pixel_key::<HAS_ALPHA>(px)
}

#[inline(always)]
fn read_prev_px<const HAS_ALPHA: bool>(
    prev: &[Vec4u8],
    stride_px: usize,
    y: usize,
    x: usize,
) -> u32 {
    pixel_key::<HAS_ALPHA>(prev[y * stride_px + x])
}

#[derive(Copy, Clone)]
struct RowHashSlot {
    hash: u64,
    y: u32,
    /// `0` = empty, `1` = unique row in sampled `curr`, `2` = duplicate (ignored).
    count: u8,
}

impl RowHashSlot {
    const EMPTY: Self = Self {
        hash: 0,
        y: 0,
        count: 0,
    };
}

struct RowHashTable {
    slots: [RowHashSlot; HASH_SLOTS],
    unique_count: usize,
}

impl RowHashTable {
    fn new() -> Self {
        Self {
            slots: [RowHashSlot::EMPTY; HASH_SLOTS],
            unique_count: 0,
        }
    }

    #[inline]
    fn insert(&mut self, hash: u64, y: u32) {
        let mut idx = (hash as usize) & HASH_MASK;
        for _ in 0..HASH_SLOTS {
            let slot = &mut self.slots[idx];
            if slot.count == 0 {
                slot.hash = hash;
                slot.y = y;
                slot.count = 1;
                self.unique_count += 1;
                return;
            }
            if slot.hash == hash {
                if slot.count == 1 {
                    slot.count = 2;
                    self.unique_count -= 1;
                }
                return;
            }
            idx = (idx + 1) & HASH_MASK;
        }
    }

    #[inline]
    fn lookup_unique(&self, hash: u64) -> Option<u32> {
        let mut idx = (hash as usize) & HASH_MASK;
        for _ in 0..HASH_SLOTS {
            let slot = self.slots[idx];
            if slot.count == 0 {
                return None;
            }
            if slot.hash == hash {
                return (slot.count == 1).then_some(slot.y);
            }
            idx = (idx + 1) & HASH_MASK;
        }
        None
    }
}

#[derive(Copy, Clone)]
struct VoteEntry {
    dy: i32,
    votes: u16,
}

struct VoteTable {
    entries: [VoteEntry; VOTE_SLOTS],
    len: usize,
}

impl VoteTable {
    fn new() -> Self {
        Self {
            entries: [VoteEntry { dy: 0, votes: 0 }; VOTE_SLOTS],
            len: 0,
        }
    }

    #[inline]
    fn record(&mut self, dy: i32) {
        for entry in &mut self.entries[..self.len] {
            if entry.dy == dy {
                entry.votes = entry.votes.saturating_add(1);
                return;
            }
        }
        if self.len < VOTE_SLOTS {
            self.entries[self.len] = VoteEntry { dy, votes: 1 };
            self.len += 1;
        } else {
            let mut min_idx = 0;
            for i in 1..VOTE_SLOTS {
                if self.entries[i].votes < self.entries[min_idx].votes {
                    min_idx = i;
                }
            }
            if self.entries[min_idx].votes <= 1 {
                self.entries[min_idx] = VoteEntry { dy, votes: 1 };
            } else {
                for entry in &mut self.entries {
                    entry.votes = entry.votes.saturating_sub(1);
                }
            }
        }
    }

    fn top_two(&self) -> ([VoteEntry; 2], usize) {
        let mut best = [VoteEntry { dy: 0, votes: 0 }; 2];
        let mut n = 0;
        for &e in &self.entries[..self.len] {
            if e.votes < MIN_VOTES {
                continue;
            }
            if e.votes > best[0].votes {
                best[1] = best[0];
                best[0] = e;
                n = (n + 1).min(2);
            } else if e.votes > best[1].votes {
                best[1] = e;
                n = 2;
            }
        }
        (best, n)
    }
}

/// Runs 1D Kadane's algorithm over `scores`, returning the half-open index
/// range `[start, end)` achieving the maximum strictly positive sum (or `None`
/// if no positive subarray exists).
fn kadane_best_range(scores: &[i32]) -> Option<(usize, usize)> {
    let mut best_sum = 0i32;
    let mut best_start = 0usize;
    let mut best_end = 0usize;
    let mut cur_sum = 0i32;
    let mut cur_start = 0usize;

    for (i, &score) in scores.iter().enumerate() {
        if cur_sum <= 0 {
            cur_sum = score;
            cur_start = i;
        } else {
            cur_sum += score;
        }
        if cur_sum > best_sum {
            best_sum = cur_sum;
            best_start = cur_start;
            best_end = i + 1;
        }
    }

    (best_sum > 0).then_some((best_start, best_end))
}

/// Returns the number of cache-line-aligned 16px horizontal probe strips placed
/// across `0..width` spaced by `STRIP_STEP_X` (`240` px) starting at
/// `STRIP_START_X` (`112` px).
#[inline(always)]
fn strip_count(width: usize) -> usize {
    if width < MIN_SCROLL_DIM as usize {
        return 0;
    }
    (((width - STRIP_START_X - STRIP_W) / STRIP_STEP_X) + 1).min(MAX_STRIPS)
}

/// Returns the horizontal pixel offset of probe strip `si`.
#[inline(always)]
const fn strip_x(si: usize) -> usize {
    STRIP_START_X + si * STRIP_STEP_X
}

/// Evaluates whether a single pixel column `x` across `sample_rows` matches
/// the shift `dy`, returning `(moved, bad)`.
///
/// # Safety
/// All `sample_rows` and their shifted `src_y` counterparts must be in-bounds,
/// and `x < width`.
#[inline]
unsafe fn eval_column<const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    stride_px: usize,
    sample_rows: &[usize],
    dy: i32,
    x: usize,
) -> (usize, usize) {
    let mut moved = 0usize;
    let mut bad = 0usize;
    for &dst_y in sample_rows {
        let src_y = (dst_y as isize - dy as isize) as usize;
        // SAFETY: Caller guarantees `(dst_y, x)` is in-bounds.
        let c = unsafe { read_curr_px::<HAS_ALPHA>(aos, stride_px, dst_y, x) };
        let ps = read_prev_px::<HAS_ALPHA>(prev, stride_px, src_y, x);
        if c == ps {
            let p0 = read_prev_px::<HAS_ALPHA>(prev, stride_px, dst_y, x);
            if c != p0 {
                moved += 1;
            }
        } else {
            bad += 1;
        }
    }
    (moved, bad)
}

/// Evaluates a single row `dst_y` across `sample_cols`, returning `(moved, ok)`.
///
/// # Safety
/// `dst_y`, `src_y`, and all `sample_cols` must be in-bounds.
#[inline(always)]
unsafe fn eval_row_cols<const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    stride_px: usize,
    dst_y: usize,
    src_y: usize,
    sample_cols: &[usize],
) -> (usize, usize) {
    let mut moved = 0usize;
    let mut ok = 0usize;
    for &x in sample_cols {
        // SAFETY: Caller guarantees `(dst_y, x)` is in-bounds.
        let c = unsafe { read_curr_px::<HAS_ALPHA>(aos, stride_px, dst_y, x) };
        let ps = read_prev_px::<HAS_ALPHA>(prev, stride_px, src_y, x);
        if c == ps {
            ok += 1;
            let p0 = read_prev_px::<HAS_ALPHA>(prev, stride_px, dst_y, x);
            if c != p0 {
                moved += 1;
            }
        }
    }
    (moved, ok)
}

/// Scores a single candidate row `dst_y` across `sample_cols`.
///
/// # Safety
/// `dst_y`, `(dst_y - dy)`, and all `sample_cols` must be in-bounds.
#[inline(always)]
unsafe fn score_candidate_row<const HAS_ALPHA: bool>(
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    stride_px: usize,
    sample_cols: &[usize],
    dy: i32,
    dst_y: usize,
) -> (i32, bool) {
    let src_y = (dst_y as isize - dy as isize) as usize;
    // SAFETY: Caller guarantees `dst_y`, `src_y`, and `sample_cols` are in-bounds.
    let (moved, ok) =
        unsafe { eval_row_cols::<HAS_ALPHA>(aos, prev, stride_px, dst_y, src_y, sample_cols) };
    if ok * 4 >= sample_cols.len() * 3 {
        if moved > 0 { (3, true) } else { (0, false) }
    } else {
        (-5, false)
    }
}

/// Phase 2: given candidate vertical shift `dy` detected on strip `seed_sx`,
/// expands and refines exact `(x, y, width, height)` bounds inside
/// `[band_y0, band_y1)` and applies the profitability & match-ratio gate.
///
/// # Safety
/// `aos` and `prev` must be valid for `height * stride_px` pixels,
/// `band_y1 <= height`, `width <= stride_px`, and `seed_sx + STRIP_W <= width`.
#[inline(always)]
pub(super) unsafe fn expand_and_verify_candidate_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    width: usize,
    stride_px: usize,
    band_y0: usize,
    band_y1: usize,
    seed_sx: usize,
    dy: i32,
) -> Option<(ScrollRect, u64)> {
    let shift = dy.unsigned_abs() as usize;
    let band_h = band_y1 - band_y0;
    if shift == 0 || shift >= band_h || band_h - shift < MIN_SCROLL_COPY_H as usize {
        return None;
    }

    let (valid_y_start, valid_y_end) = if dy < 0 {
        (band_y0, band_y1 - shift)
    } else {
        (band_y0 + shift, band_y1)
    };
    let valid_h = valid_y_end - valid_y_start;
    if valid_h < MIN_SCROLL_COPY_H as usize {
        return None;
    }

    // Step 2a: coarse seed vertical range on `seed_sx` using a prime step
    // (`13` on large frames, `5` on short spans) to avoid aliasing with common
    // 16/18/20/24px UI line heights.
    let y_coarse_step = if valid_h < 256 { 5 } else { 13 };
    let n_seed_steps = valid_h.div_ceil(y_coarse_step);
    let mut stack_scores = [0i32; STACK_SCORE_CAP];
    let mut heap_scores = Vec::new();
    let seed_scores: &mut [i32] = if n_seed_steps <= STACK_SCORE_CAP {
        &mut stack_scores[..n_seed_steps]
    } else {
        heap_scores.resize(n_seed_steps, 0);
        &mut heap_scores
    };
    let curr_base = aos.ptr();
    let prev_base = prev.as_ptr();
    for (i, slot) in seed_scores.iter_mut().enumerate() {
        let dst_y = valid_y_start + i * y_coarse_step;
        let src_y = (dst_y as isize - dy as isize) as usize;
        // SAFETY: `dst_y` and `src_y` lie in `[band_y0, band_y1) <= height`,
        // and `seed_sx + STRIP_W <= width <= stride_px`.
        *slot = unsafe {
            kernel.score_strip_row::<HAS_ALPHA>(
                curr_base.add(dst_y * stride_px + seed_sx),
                prev_base.add(src_y * stride_px + seed_sx),
                prev_base.add(dst_y * stride_px + seed_sx),
            )
        };
    }
    let (seed_i0, seed_i1) = kadane_best_range(seed_scores)?;
    let seed_y0 = valid_y_start + seed_i0 * y_coarse_step;
    let seed_y1 = (valid_y_start + seed_i1 * y_coarse_step).min(valid_y_end);
    if seed_y1 - seed_y0 < MIN_SCROLL_COPY_H as usize / 2 {
        return None;
    }

    // Step 2b: row-major horizontal extent discovery over up to 17 evenly
    // spaced rows inside `[seed_y0, seed_y1)`, followed by single-pixel X edge
    // refinement.
    const MAX_X_SAMPLE_ROWS: usize = 17;
    let seed_h = seed_y1 - seed_y0;
    let n_sample_rows = seed_h.min(MAX_X_SAMPLE_ROWS);
    let mut sample_rows = [0usize; MAX_X_SAMPLE_ROWS];
    for (i, slot) in sample_rows[..n_sample_rows].iter_mut().enumerate() {
        *slot = seed_y0 + ((i * 2 + 1) * seed_h) / (n_sample_rows * 2);
    }
    let sample_rows = &sample_rows[..n_sample_rows];

    let n_blocks = width.div_ceil(X_BLOCK_W);
    let mut stack_moved = [0u16; STACK_SCORE_CAP];
    let mut stack_bad = [0u16; STACK_SCORE_CAP];
    let mut heap_moved = Vec::new();
    let mut heap_bad = Vec::new();
    let (block_moved, block_bad): (&mut [u16], &mut [u16]) = if n_blocks <= STACK_SCORE_CAP {
        (&mut stack_moved[..n_blocks], &mut stack_bad[..n_blocks])
    } else {
        heap_moved.resize(n_blocks, 0);
        heap_bad.resize(n_blocks, 0);
        (&mut heap_moved, &mut heap_bad)
    };

    // Sweep sequentially along each sampled row so hardware prefetching keeps
    // `curr` and `prev` linear instead of striding vertically per column.
    for &dst_y in sample_rows {
        let src_y = (dst_y as isize - dy as isize) as usize;
        for bx in 0..n_blocks {
            let bx_start = bx * X_BLOCK_W;
            let bw = (width - bx_start).min(X_BLOCK_W);
            let xs = [bx_start + bw / 4, bx_start + (3 * bw) / 4];
            let mut m = 0u16;
            let mut b = 0u16;
            for x in xs {
                // SAFETY: `(dst_y, x)` and `(src_y, x)` are in-bounds.
                let c = unsafe { read_curr_px::<HAS_ALPHA>(aos, stride_px, dst_y, x) };
                let ps = read_prev_px::<HAS_ALPHA>(prev, stride_px, src_y, x);
                if c == ps {
                    let p0 = read_prev_px::<HAS_ALPHA>(prev, stride_px, dst_y, x);
                    if c != p0 {
                        m += 1;
                    }
                } else {
                    b += 1;
                }
            }
            block_moved[bx] += m;
            block_bad[bx] += b;
        }
    }

    let total_per_block = (n_sample_rows * 2) as u16;
    let mut block_scores_stack = [0i32; STACK_SCORE_CAP];
    let mut block_scores_heap = Vec::new();
    let block_scores: &mut [i32] = if n_blocks <= STACK_SCORE_CAP {
        &mut block_scores_stack[..n_blocks]
    } else {
        block_scores_heap.resize(n_blocks, 0);
        &mut block_scores_heap
    };
    for bx in 0..n_blocks {
        block_scores[bx] = if block_bad[bx] * 5 <= total_per_block {
            if block_moved[bx] > 0 { 3 } else { 0 }
        } else {
            -5
        };
    }

    let (bx0, bx1) = kadane_best_range(block_scores)?;
    let mut x0 = bx0 * X_BLOCK_W;
    let mut x1 = (bx1 * X_BLOCK_W).min(width);

    // Single-pixel horizontal edge refinement around `x0` and `x1`, stepping
    // across matching neutral margins (up to `MAX_EDGE_GAP` columns) so
    // scrolling background margins are included without crossing into fixed
    // sidebars or scrollbars.
    const MAX_EDGE_GAP: usize = 48;
    let is_clean_scroll_col = |moved: usize, bad: usize| moved > 0 && bad * 8 <= n_sample_rows;
    let is_bad_col = |bad: usize| bad * 4 > n_sample_rows;

    // Refine left edge `x0`.
    {
        // SAFETY: `x0 < width` and `sample_rows` are in-bounds.
        let (_, b0) =
            unsafe { eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, x0) };
        if is_bad_col(b0) {
            let limit = (x0 + X_BLOCK_W).min(x1);
            while x0 < limit {
                // SAFETY: `x0 < limit <= x1 <= width`.
                let (_, b) =
                    unsafe { eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, x0) };
                if !is_bad_col(b) {
                    break;
                }
                x0 += 1;
            }
        } else {
            let mut best_x0 = x0;
            let mut gap = 0usize;
            for cx in (0..x0).rev() {
                // SAFETY: `cx < width`.
                let (m, b) =
                    unsafe { eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, cx) };
                if is_bad_col(b) {
                    break;
                }
                if is_clean_scroll_col(m, b) {
                    best_x0 = cx;
                    gap = 0;
                } else {
                    gap += 1;
                    if gap >= MAX_EDGE_GAP {
                        break;
                    }
                }
            }
            x0 = best_x0;
        }
    }

    // Refine right edge `x1`.
    if x1 > x0 {
        // SAFETY: `x1 - 1 < width`.
        let (_, b_last) =
            unsafe { eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, x1 - 1) };
        if is_bad_col(b_last) {
            let limit = x1.saturating_sub(X_BLOCK_W).max(x0);
            while x1 > limit {
                // SAFETY: `x1 - 1 < width`.
                let (_, b) = unsafe {
                    eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, x1 - 1)
                };
                if !is_bad_col(b) {
                    break;
                }
                x1 -= 1;
            }
        } else {
            let mut best_x1 = x1;
            let mut gap = 0usize;
            for cx in x1..width {
                // SAFETY: `cx < width`.
                let (m, b) =
                    unsafe { eval_column::<HAS_ALPHA>(aos, prev, stride_px, sample_rows, dy, cx) };
                if is_bad_col(b) {
                    break;
                }
                if is_clean_scroll_col(m, b) {
                    best_x1 = cx + 1;
                    gap = 0;
                } else {
                    gap += 1;
                    if gap >= MAX_EDGE_GAP {
                        break;
                    }
                }
            }
            x1 = best_x1;
        }
    }

    if x1 <= x0 || (x1 - x0) < MIN_SCROLL_DIM as usize {
        return None;
    }

    // Step 2c: vertical extent `[dst_y0, dst_y1)` across 8 evenly spaced
    // columns inside `[x0, x1)`, with single-row refinement at both edges.
    const Y_SAMPLE_COLS: usize = 8;
    let span_w = x1 - x0;
    let mut sample_cols = [0usize; Y_SAMPLE_COLS];
    for (i, col) in sample_cols.iter_mut().enumerate() {
        *col = x0 + ((i * 2 + 1) * span_w) / (Y_SAMPLE_COLS * 2);
    }

    for (i, slot) in seed_scores.iter_mut().enumerate() {
        let dst_y = valid_y_start + i * y_coarse_step;
        // SAFETY: `dst_y`, `dst_y - dy`, and `sample_cols` are in-bounds.
        *slot = unsafe {
            score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, dst_y).0
        };
    }
    let (yi0, yi1) = kadane_best_range(seed_scores)?;
    let mut dst_y0 = valid_y_start + yi0 * y_coarse_step;
    let mut dst_y1 = (valid_y_start + yi1 * y_coarse_step).min(valid_y_end);

    // Refine top edge `dst_y0` row-by-row, bridging matching blank line-spacing
    // gaps (up to `MAX_EDGE_GAP` rows) until a mismatch or `valid_y_start`.
    {
        // SAFETY: `dst_y0`, `dst_y0 - dy`, and `sample_cols` are in-bounds.
        let (s0, _) = unsafe {
            score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, dst_y0)
        };
        if s0 < 0 {
            let limit = (dst_y0 + y_coarse_step).min(dst_y1);
            while dst_y0 < limit
                && unsafe {
                    score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, dst_y0)
                        .0
                } < 0
            {
                dst_y0 += 1;
            }
        } else {
            let mut best_y0 = dst_y0;
            let mut gap = 0usize;
            for cy in (valid_y_start..dst_y0).rev() {
                // SAFETY: `cy`, `cy - dy`, and `sample_cols` are in-bounds.
                let (s, strong) = unsafe {
                    score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, cy)
                };
                if s < 0 {
                    break;
                }
                if strong {
                    best_y0 = cy;
                    gap = 0;
                } else {
                    gap += 1;
                    if gap >= MAX_EDGE_GAP {
                        break;
                    }
                }
            }
            dst_y0 = best_y0;
        }
    }

    // Refine bottom edge `dst_y1` row-by-row.
    if dst_y1 > dst_y0 {
        // SAFETY: `dst_y1 - 1`, `dst_y1 - 1 - dy`, and `sample_cols` are in-bounds.
        let (s_last, _) = unsafe {
            score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, dst_y1 - 1)
        };
        if s_last < 0 {
            let limit = dst_y1.saturating_sub(y_coarse_step).max(dst_y0);
            while dst_y1 > limit
                && unsafe {
                    score_candidate_row::<HAS_ALPHA>(
                        aos,
                        prev,
                        stride_px,
                        &sample_cols,
                        dy,
                        dst_y1 - 1,
                    )
                    .0
                } < 0
            {
                dst_y1 -= 1;
            }
        } else {
            let mut best_y1 = dst_y1;
            let mut gap = 0usize;
            for cy in dst_y1..valid_y_end {
                // SAFETY: `cy`, `cy - dy`, and `sample_cols` are in-bounds.
                let (s, strong) = unsafe {
                    score_candidate_row::<HAS_ALPHA>(aos, prev, stride_px, &sample_cols, dy, cy)
                };
                if s < 0 {
                    break;
                }
                if strong {
                    best_y1 = cy + 1;
                    gap = 0;
                } else {
                    gap += 1;
                    if gap >= MAX_EDGE_GAP {
                        break;
                    }
                }
            }
            dst_y1 = best_y1;
        }
    }

    if dst_y1 <= dst_y0 {
        return None;
    }
    let copy_h = dst_y1 - dst_y0;
    let scroll_h = copy_h + shift;
    let scroll_w = x1 - x0;
    let copy_area = scroll_w as u64 * copy_h as u64;
    if copy_h < MIN_SCROLL_COPY_H as usize
        || scroll_h < MIN_SCROLL_DIM as usize
        || copy_area < MIN_SCROLL_COPY_AREA
    {
        return None;
    }

    // Step 2d: 16x16 2D grid verification and profitability gate.
    const VERIFY_N: usize = 16;
    let mut n_match_scroll = 0usize;
    let mut n_strict_win = 0usize;
    let mut n_regression = 0usize;
    let total_verify = VERIFY_N * VERIFY_N;
    for gy in 0..VERIFY_N {
        let dst_y = dst_y0 + ((gy * 2 + 1) * copy_h) / (VERIFY_N * 2);
        let src_y = (dst_y as isize - dy as isize) as usize;
        for gx in 0..VERIFY_N {
            let x = x0 + ((gx * 2 + 1) * scroll_w) / (VERIFY_N * 2);
            // SAFETY: `(dst_y, x)` and `(src_y, x)` are in-bounds.
            let c = unsafe { read_curr_px::<HAS_ALPHA>(aos, stride_px, dst_y, x) };
            let ps = read_prev_px::<HAS_ALPHA>(prev, stride_px, src_y, x);
            let p0 = read_prev_px::<HAS_ALPHA>(prev, stride_px, dst_y, x);
            let eq_scroll = c == ps;
            let eq_static = c == p0;
            if eq_scroll {
                n_match_scroll += 1;
                if !eq_static {
                    n_strict_win += 1;
                }
            } else if eq_static {
                n_regression += 1;
            }
        }
    }

    // Require >= 75% scroll match across the destination rect and strictly more
    // than 2x as many fixed changed pixels as regressions.
    if n_match_scroll * 100 < total_verify * 75 {
        return None;
    }
    if n_strict_win < 2 || n_strict_win <= n_regression * 2 {
        return None;
    }

    let sr_y = if dy < 0 { dst_y0 } else { dst_y0 - shift };
    let sr = ScrollRect {
        x: x0 as u32,
        y: sr_y as u32,
        width: scroll_w as u32,
        height: scroll_h as u32,
        dy,
    };
    Some((sr, copy_area))
}

/// Phase 1b: 7-row burst ruler sampling on `curr` against every 7th `prev` row,
/// split out of Tier 0 so unchanged frames never allocate the 4 KiB
/// [`RowHashTable`] / [`VoteTable`] stack frame or execute stack probes.
///
/// # Safety
/// `aos` and `prev` must be valid for `height * stride_px` pixels,
/// `band_y1 <= height`, `width <= stride_px`, and `n_strips == strip_count(width)`.
#[inline(always)]
pub(super) unsafe fn detect_scroll_phase1b_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    width: usize,
    stride_px: usize,
    band_y0: usize,
    band_y1: usize,
    n_strips: usize,
    tier0_step_y: usize,
    changed_rows: &[u16; MAX_STRIPS],
    changed_non_uniform: &[u16; MAX_STRIPS],
    min_y: &[u32; MAX_STRIPS],
    max_y: &[u32; MAX_STRIPS],
) -> Option<(ScrollRect, u64)> {
    // Pick up to 2 best active strips, preferring horizontal separation >= 2
    // strips for the second probe so a local overlay/animation doesn't hide a
    // scrolling pane.
    let strip_rank = |si: usize, rows: u16| -> (u16, u16) { (changed_non_uniform[si], rows) };
    let mut best_si = None;
    let mut best_rank = (0u16, 0u16);
    for (si, &rows) in changed_rows[..n_strips].iter().enumerate() {
        if rows == 0 {
            continue;
        }
        let rank = strip_rank(si, rows);
        if rank > best_rank {
            best_rank = rank;
            best_si = Some(si);
        }
    }
    let first_si = best_si?;

    let mut second_si = None;
    let mut second_rank = (0u16, 0u16);
    for (si, &rows) in changed_rows[..n_strips].iter().enumerate() {
        if rows == 0 || si.abs_diff(first_si) < 2 {
            continue;
        }
        let rank = strip_rank(si, rows);
        if rank > second_rank {
            second_rank = rank;
            second_si = Some(si);
        }
    }
    if second_si.is_none() {
        for (si, &rows) in changed_rows[..n_strips].iter().enumerate() {
            if rows == 0 || si == first_si {
                continue;
            }
            let rank = strip_rank(si, rows);
            if rank > second_rank {
                second_rank = rank;
                second_si = Some(si);
            }
        }
    }

    let curr_base = aos.ptr();
    let prev_base = prev.as_ptr();
    let mut best_result: Option<(ScrollRect, u64)> = None;
    let probe_order = [Some(first_si), second_si];
    for (attempt, si_opt) in probe_order.into_iter().enumerate() {
        let Some(si) = si_opt else {
            break;
        };
        let sx = strip_x(si);

        // Bound anchor placement to the strip's active vertical window (at
        // least `MIN_SCROLL_DIM` tall) so small scrolling panes on large
        // monitors still get dense anchor coverage.
        let mut win_y0 = (min_y[si] as usize)
            .saturating_sub(tier0_step_y)
            .max(band_y0);
        let mut win_y1 = ((max_y[si] as usize) + tier0_step_y + 1).min(band_y1);
        if win_y1 - win_y0 < MIN_SCROLL_DIM as usize {
            let deficit = MIN_SCROLL_DIM as usize - (win_y1 - win_y0);
            let grow_up = deficit / 2;
            win_y0 = win_y0.saturating_sub(grow_up).max(band_y0);
            win_y1 = (win_y0 + MIN_SCROLL_DIM as usize).min(band_y1);
            win_y0 = win_y1.saturating_sub(MIN_SCROLL_DIM as usize).max(band_y0);
        }

        let win_h = win_y1 - win_y0;
        let anchor_step = (win_h / MAX_ANCHORS).max(PREV_STEP);
        let mut table = RowHashTable::new();
        let mut curr_min_y = usize::MAX;
        let mut curr_max_y = 0usize;
        let mut anchor_y = win_y0;
        let mut n_anchors = 0usize;
        while anchor_y + PREV_STEP <= win_y1 && n_anchors < MAX_ANCHORS {
            for ry in anchor_y..anchor_y + PREV_STEP {
                let off = ry * stride_px + sx;
                // SAFETY: `ry < win_y1 <= band_y1 <= height` and `sx + STRIP_W <= width <= stride_px`.
                let (changed, hash_opt) = unsafe {
                    kernel
                        .probe_curr_strip_hash::<HAS_ALPHA>(curr_base.add(off), prev_base.add(off))
                };
                if changed {
                    curr_min_y = curr_min_y.min(ry);
                    curr_max_y = curr_max_y.max(ry);
                    if let Some(h) = hash_opt {
                        table.insert(h, ry as u32);
                    }
                }
            }
            anchor_y += anchor_step;
            n_anchors += 1;
        }
        if table.unique_count < MIN_VOTES as usize
            || curr_max_y.saturating_sub(curr_min_y) + 1 < MIN_SCROLL_COPY_H as usize
        {
            continue;
        }

        let mut votes = VoteTable::new();
        let mut yp = band_y0;
        while yp < band_y1 {
            // SAFETY: `yp < band_y1 <= height` and `sx + STRIP_W <= width <= stride_px`.
            if let Some(h) = unsafe {
                kernel.hash_prev_strip_non_uniform::<HAS_ALPHA>(prev_base.add(yp * stride_px + sx))
            } && let Some(yc) = table.lookup_unique(h)
            {
                let dy = yc as i32 - yp as i32;
                if dy != 0 {
                    votes.record(dy);
                }
            }
            yp += PREV_STEP;
        }

        let (top, n_top) = votes.top_two();
        for cand in &top[..n_top] {
            // SAFETY: Band bounds and `sx + STRIP_W <= width` were verified above.
            if let Some((sr, area)) = unsafe {
                kernel.expand_and_verify_scroll_candidate::<HAS_ALPHA>(
                    aos, prev, width, stride_px, band_y0, band_y1, sx, cand.dy,
                )
            } && best_result.is_none_or(|(_, best_area)| area > best_area)
            {
                best_result = Some((sr, area));
            }
        }

        if best_result.is_some() || (attempt == 0 && n_top > 0 && top[0].votes >= MIN_VOTES * 2) {
            break;
        }
    }

    best_result
}

/// Detects the dominant vertical [`ScrollRect`] within a single contiguous
/// band `[band_y0, band_y1)`, keeping the Tier-0 clean-frame fast path tiny and
/// stack-probe-free before handing off to [`RowKernel::detect_scroll_phase1b`].
///
/// # Safety
/// `aos` and `prev` must be valid for `height * stride_px` pixels,
/// `band_y1 <= height`, and `width <= stride_px`.
#[inline(always)]
pub(super) unsafe fn detect_scroll_in_band_with<K: RowKernel, const HAS_ALPHA: bool>(
    kernel: K,
    aos: BufferPointer<Vec4u8>,
    prev: &[Vec4u8],
    width: usize,
    stride_px: usize,
    band_y0: usize,
    band_y1: usize,
) -> Option<(ScrollRect, u64)> {
    let band_h = band_y1.saturating_sub(band_y0);
    if width < MIN_SCROLL_DIM as usize || band_h < MIN_SCROLL_DIM as usize {
        return None;
    }

    let n_strips = strip_count(width);
    if n_strips == 0 {
        return None;
    }

    // Tier 0: zero-hash coarse SIMD activity grid (72 probes on 1080p, 352 on 4K).
    let tier0_step_y = (band_h / 4).clamp(32, TIER0_MAX_STEP_Y);
    let mut changed_rows = [0u16; MAX_STRIPS];
    let mut changed_non_uniform = [0u16; MAX_STRIPS];
    let mut min_y = [0u32; MAX_STRIPS];
    let mut max_y = [0u32; MAX_STRIPS];
    let mut any_changed = false;

    let curr_base = aos.ptr();
    let prev_base = prev.as_ptr();
    let mut ry = band_y0 + tier0_step_y / 2;
    while ry < band_y1 {
        let row_start = ry * stride_px + STRIP_START_X;
        let ry_u32 = ry as u32;
        // SAFETY: `ry < band_y1 <= height` and `strip_x(n_strips - 1) + STRIP_W <= width <= stride_px`.
        let row_curr = unsafe { curr_base.add(row_start) };
        let row_prev = unsafe { prev_base.add(row_start) };
        for si in 0..n_strips {
            let off_x = si * STRIP_STEP_X;
            // SAFETY: `STRIP_START_X + off_x + STRIP_W <= width <= stride_px`, so both
            // offset pointers and their 16-pixel windows are strictly in-bounds on row `ry`.
            if let Some(non_uniform) = unsafe {
                kernel.probe_strip_activity::<HAS_ALPHA>(row_curr.add(off_x), row_prev.add(off_x))
            } {
                any_changed = true;
                let prev_rows = changed_rows[si];
                changed_rows[si] = prev_rows + 1;
                if non_uniform {
                    changed_non_uniform[si] += 1;
                }
                if prev_rows == 0 {
                    min_y[si] = ry_u32;
                }
                max_y[si] = ry_u32;
            }
        }
        ry += tier0_step_y;
    }
    if !any_changed {
        return None;
    }

    // SAFETY: Band bounds and `n_strips == strip_count(width)` were verified above.
    unsafe {
        kernel.detect_scroll_phase1b::<HAS_ALPHA>(
            aos,
            prev,
            width,
            stride_px,
            band_y0,
            band_y1,
            n_strips,
            tier0_step_y,
            &changed_rows,
            &changed_non_uniform,
            &min_y,
            &max_y,
        )
    }
}

/// Detects a profitable vertical scroll blit using an explicit SIMD `tier`.
pub(super) fn detect_scroll_rect_with_tier(
    curr: BufferPointer<u8>,
    prev_shadow: &[u8],
    metadata: BufferMetadata,
    bands: Option<&[RowBand]>,
    tier: arch::Tier,
) -> Option<ScrollRect> {
    let total_pixels = metadata.validate().ok()?;
    let byte_len = total_pixels.checked_mul(4)?;
    if curr.len() < byte_len || prev_shadow.len() < byte_len {
        return None;
    }
    let width = metadata.width as usize;
    let height = metadata.height as usize;
    if width < MIN_SCROLL_DIM as usize || height < MIN_SCROLL_DIM as usize {
        return None;
    }
    let stride_px = (metadata.stride as usize) / 4;
    let has_alpha = metadata.format == BufferFormat::Argb8888;
    // SAFETY: `Vec4u8` is `repr(C, packed)` with size 4 and align 1, and
    // `curr.len() >= byte_len` is a multiple of 4 from `BufferMetadata::validate`
    // (`curr` is sliced to `byte_len` before casting).
    let aos = unsafe { curr.split_at(byte_len).0.cast::<Vec4u8>() };
    let prev: &[Vec4u8] = bytemuck::cast_slice(&prev_shadow[..byte_len]);

    let full_band = [RowBand {
        y0: 0,
        y1: height as u32,
    }];
    let active_bands = bands.unwrap_or(&full_band);

    let mut best: Option<(ScrollRect, u64)> = None;
    for band in active_bands {
        let y0 = band.y0 as usize;
        let y1 = band.y1 as usize;
        if y0 >= y1 || y1 > height || y1 - y0 < MIN_SCROLL_DIM as usize {
            continue;
        }
        // SAFETY: `aos` and `prev` both cover `total_pixels = height * stride_px`,
        // `y1 <= height`, and `width <= stride_px`.
        let cand = unsafe {
            if has_alpha {
                tier.detect_scroll_in_band::<true>(aos, prev, width, stride_px, y0, y1)
            } else {
                tier.detect_scroll_in_band::<false>(aos, prev, width, stride_px, y0, y1)
            }
        };
        if let Some((sr, area)) = cand
            && best.is_none_or(|(_, best_area)| area > best_area)
        {
            best = Some((sr, area));
        }
    }

    best.map(|(sr, _)| {
        debug_assert!(
            sr.validate_in_bounds(metadata.width as u32, metadata.height as u32)
                .is_ok()
        );
        sr
    })
}

/// Detects a profitable vertical scroll blit between `prev_shadow` and `curr`
/// within `bands` (or the full frame when `bands` is `None`).
///
/// Reads `curr` exclusively through [`BufferPointer`] within the caller's
/// fault-probed `bands` so shared-memory aliasing and `SIGBUS` boundaries are
/// preserved. Returns `None` when no profitable scroll is found.
#[inline(never)]
pub fn detect_scroll_rect(
    curr: BufferPointer<u8>,
    prev_shadow: &[u8],
    metadata: BufferMetadata,
    bands: Option<&[RowBand]>,
) -> Option<ScrollRect> {
    detect_scroll_rect_with_tier(curr, prev_shadow, metadata, bands, arch::Tier::detect())
}
