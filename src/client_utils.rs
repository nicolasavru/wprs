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

use anyhow::Result;
use smithay_client_toolkit::data_device_manager::data_device::DataDevice;
use smithay_client_toolkit::primary_selection::device::PrimarySelectionDevice;
use smithay_client_toolkit::reexports::client::protocol::wl_keyboard::WlKeyboard;
use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;

use crate::constants;
use crate::filtering;
use crate::serialization::framing::RowBand;
use crate::serialization::geometry::Rectangle;

#[derive(Debug)]
pub(crate) struct SeatObject<P> {
    pub(crate) seat: WlSeat,
    pub(crate) keyboard: Option<WlKeyboard>,
    pub(crate) pointer: Option<P>,
    pub(crate) data_device: DataDevice,
    pub(crate) primary_selection_device: Option<PrimarySelectionDevice>,
}

/// Accumulated buffer-space surface damage. `Full` stays sticky until
/// [`Self::apply_and_clear`], and `Rects` promotes to `Full` once its length
/// reaches [`constants::SENT_DAMAGE_LIMIT`].
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub enum FrameDamage {
    #[default]
    Empty,
    Rects(Vec<Rectangle<i32>>),
    Full,
}

impl FrameDamage {
    fn add_rects(&mut self, mut rects: Vec<Rectangle<i32>>) {
        let merged = match std::mem::replace(self, Self::Full) {
            Self::Full => return,
            Self::Empty => rects,
            Self::Rects(mut existing) => {
                existing.append(&mut rects);
                existing
            },
        };
        if merged.len() < constants::SENT_DAMAGE_LIMIT {
            *self = Self::Rects(merged);
        }
    }

    /// Merges a commit's buffer-space damage. Missing (`None`) or empty damage
    /// with `had_new_buffer == true` marks [`Self::Full`] (used for full-buffer
    /// syncs and undamaged new buffers).
    pub(crate) fn merge_commit(
        &mut self,
        damage: Option<Vec<Rectangle<i32>>>,
        had_new_buffer: bool,
    ) {
        match damage {
            Some(rects) if !rects.is_empty() => self.add_rects(rects),
            _ if had_new_buffer => *self = Self::Full,
            _ => {},
        }
    }

    pub(crate) fn apply_and_clear(&mut self, wl_surface: &WlSurface) {
        match std::mem::replace(self, Self::Empty) {
            Self::Empty => {},
            Self::Rects(damage_rects) => {
                for damage_rect in damage_rects {
                    wl_surface.damage_buffer(
                        damage_rect.loc.x,
                        damage_rect.loc.y,
                        damage_rect.size.w,
                        damage_rect.size.h,
                    );
                }
            },
            Self::Full => {
                wl_surface.damage_buffer(0, 0, i32::MAX, i32::MAX);
            },
        }
    }
}

/// Merges `cur` into a back slot's stale row-band history `prev`, returning
/// `None` if either is a full-frame update or the union covers `>= 75%` of
/// `height`.
pub(crate) fn combine_band_history(
    prev: Option<Vec<RowBand>>,
    cur: Option<&[RowBand]>,
    height: i32,
) -> Option<Vec<RowBand>> {
    let (mut prev, cur) = (prev?, cur?);
    prev.extend_from_slice(cur);
    filtering::merge_row_bands(prev, height)
}

/// Subtracts sorted, non-overlapping `[y0, y1)` row-bands `cur` from sorted,
/// non-overlapping `prev`, returning the sorted, non-overlapping remainder
/// `prev \ cur`.
pub(crate) fn subtract_row_bands(prev: &[RowBand], cur: &[RowBand]) -> Vec<RowBand> {
    let mut out = Vec::new();
    let mut j = 0;

    for &band in prev {
        let mut start = band.y0;
        while j < cur.len() && cur[j].y1 <= start {
            j += 1;
        }
        while j < cur.len() && cur[j].y0 < band.y1 {
            if cur[j].y0 > start {
                out.push(RowBand {
                    y0: start,
                    y1: cur[j].y0,
                });
            }
            start = cur[j].y1;
            if start >= band.y1 {
                break;
            }
            j += 1;
        }
        if start < band.y1 {
            out.push(RowBand {
                y0: start,
                y1: band.y1,
            });
        }
    }

    out
}

/// Outcome of advancing a lazy 2-slot swapchain for a frame update.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SwapchainStep {
    /// `active_buffer` was released and reused in place.
    ReusedActive,
    /// `active_buffer` was held; swapped to the released back slot whose stale
    /// row-band history relative to the previous active slot was `prev_bands`
    /// (`None` means the entire back slot was stale). The previous active slot
    /// is now stored in `back_buffer`.
    Swapped { prev_bands: Option<Vec<RowBand>> },
    /// Allocated a fresh slot because no released back slot was available. The
    /// previous active slot is now stored in `back_buffer`.
    Allocated,
}

/// Advances a lazy 2-slot swapchain for an update covering `cur_bands` (`None`
/// for a full-frame update), maintaining `back_buffer`'s stale row-band history
/// relative to `active_buffer`.
///
/// `alloc_buffer` receives `back_buffer_held`: `false` on first-time lazy
/// double-buffer allocation and `true` when both slots are held by the
/// compositor (triple-buffer fallback). On allocation error, `active_buffer`
/// and `back_buffer` are left unchanged.
pub(crate) fn advance_two_slot_swapchain<P, B>(
    pool: &mut P,
    active_buffer: &mut B,
    back_buffer: &mut Option<(B, Option<Vec<RowBand>>)>,
    cur_bands: Option<&[RowBand]>,
    height: i32,
    mut is_released: impl FnMut(&mut P, &B) -> bool,
    alloc_buffer: impl FnOnce(&mut P, bool) -> Result<B>,
) -> Result<SwapchainStep> {
    if is_released(pool, active_buffer) {
        if let Some((_, back_bands)) = back_buffer {
            *back_bands = combine_band_history(back_bands.take(), cur_bands, height);
        }
        return Ok(SwapchainStep::ReusedActive);
    }

    if let Some((back_buf, prev_bands)) = back_buffer.take() {
        if is_released(pool, &back_buf) {
            let old_active = std::mem::replace(active_buffer, back_buf);
            *back_buffer = Some((old_active, cur_bands.map(ToOwned::to_owned)));
            return Ok(SwapchainStep::Swapped { prev_bands });
        }
        *back_buffer = Some((back_buf, prev_bands));
    }

    let back_buffer_held = back_buffer.is_some();
    let new_buffer = alloc_buffer(pool, back_buffer_held)?;
    let old_active = std::mem::replace(active_buffer, new_buffer);
    *back_buffer = Some((old_active, cur_bands.map(ToOwned::to_owned)));
    Ok(SwapchainStep::Allocated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(y0: u32, y1: u32) -> RowBand {
        RowBand { y0, y1 }
    }

    #[test]
    fn test_subtract_row_bands() {
        let cases: &[(&[RowBand], &[RowBand], &[RowBand])] = &[
            // Empty `prev` or empty `cur`.
            (&[], &[b(2, 5)], &[]),
            (&[b(2, 5), b(8, 10)], &[], &[b(2, 5), b(8, 10)]),
            // Disjoint (separated and edge-touching on both sides).
            (&[b(2, 5)], &[b(7, 9)], &[b(2, 5)]),
            (&[b(2, 5)], &[b(5, 8)], &[b(2, 5)]),
            (&[b(5, 8)], &[b(1, 3)], &[b(5, 8)]),
            (&[b(5, 8)], &[b(2, 5)], &[b(5, 8)]),
            (
                &[b(1, 3), b(6, 8)],
                &[b(3, 6), b(8, 10)],
                &[b(1, 3), b(6, 8)],
            ),
            // Exact match (single and multi-band) -> empty.
            (&[b(2, 5)], &[b(2, 5)], &[]),
            (&[b(2, 5), b(10, 14)], &[b(2, 5), b(10, 14)], &[]),
            // Wide `cur` spanning multiple `prev` bands.
            (&[b(2, 5), b(7, 10), b(12, 15)], &[b(0, 13)], &[b(13, 15)]),
            (&[b(2, 5), b(7, 10)], &[b(2, 10)], &[]),
            // Partial left and right overlap (including exact edge alignment).
            (&[b(4, 10)], &[b(2, 7)], &[b(7, 10)]),
            (&[b(4, 10)], &[b(4, 7)], &[b(7, 10)]),
            (&[b(4, 10)], &[b(7, 12)], &[b(4, 7)]),
            (&[b(4, 10)], &[b(7, 10)], &[b(4, 7)]),
            // Splitting a `prev` band around an interior `cur` band.
            (&[b(2, 10)], &[b(4, 7)], &[b(2, 4), b(7, 10)]),
            // Multiple overlapping and touching `cur` bands across one or many `prev` bands.
            (
                &[b(0, 20)],
                &[b(2, 5), b(5, 8), b(12, 15)],
                &[b(0, 2), b(8, 12), b(15, 20)],
            ),
            (
                &[b(2, 6), b(8, 16), b(18, 22)],
                &[b(4, 10), b(12, 14), b(15, 20)],
                &[b(2, 4), b(10, 12), b(14, 15), b(20, 22)],
            ),
            // `cur` ending at one `prev` band's upper bound followed by another `cur` band.
            (
                &[b(2, 6), b(8, 12)],
                &[b(4, 6), b(6, 10)],
                &[b(2, 4), b(10, 12)],
            ),
        ];

        for &(prev, cur, expected) in cases {
            let out = subtract_row_bands(prev, cur);
            assert_eq!(out, expected, "prev={prev:?}, cur={cur:?}");

            // Verify canonical invariants and row-by-row set difference.
            for w in out.windows(2) {
                assert!(w[0].y1 <= w[1].y0, "overlapping/unsorted: {out:?}");
            }
            for band in &out {
                assert!(band.y0 < band.y1, "empty band: {out:?}");
            }
            for y in 0..24 {
                let in_out = out.iter().any(|r| y >= r.y0 && y < r.y1);
                let in_prev = prev.iter().any(|r| y >= r.y0 && y < r.y1);
                let in_cur = cur.iter().any(|r| y >= r.y0 && y < r.y1);
                assert_eq!(
                    in_out,
                    in_prev && !in_cur,
                    "row {y}: prev={prev:?}, cur={cur:?}"
                );
            }
        }
    }

    #[test]
    fn test_advance_two_slot_swapchain_alloc_error_preserves_state() {
        let mut active = 0usize;
        let mut back: Option<(usize, Option<Vec<RowBand>>)> = None;
        let mut seen_held = Vec::new();

        // First allocation failure when `back` is `None`: `back_buffer_held == false`.
        let err = advance_two_slot_swapchain(
            &mut (),
            &mut active,
            &mut back,
            Some(&[b(1, 2)]),
            20,
            |_, _| false,
            |_, held| {
                seen_held.push(held);
                Err(anyhow::anyhow!("oom"))
            },
        );
        assert!(err.is_err());
        assert_eq!(active, 0);
        assert_eq!(back, None);
        assert_eq!(seen_held, vec![false]);

        // Allocation failure when both `active` and `back` are held:
        // `back_buffer_held == true` and existing `back` is preserved intact.
        back = Some((1, Some(vec![b(3, 5)])));
        let err = advance_two_slot_swapchain(
            &mut (),
            &mut active,
            &mut back,
            Some(&[b(6, 8)]),
            20,
            |_, _| false,
            |_, held| {
                seen_held.push(held);
                Err(anyhow::anyhow!("oom"))
            },
        );
        assert!(err.is_err());
        assert_eq!(active, 0);
        assert_eq!(back, Some((1, Some(vec![b(3, 5)]))));
        assert_eq!(seen_held, vec![false, true]);
    }

    #[test]
    fn test_frame_damage_sticky_full_and_limit_promotion() {
        let rect = Rectangle::new(1, 2, 3, 4);
        let limit = constants::SENT_DAMAGE_LIMIT;

        // `Empty` + `LIMIT` promotes directly to `Full`.
        let mut from_empty = FrameDamage::Empty;
        from_empty.add_rects(vec![rect; limit]);
        assert_eq!(from_empty, FrameDamage::Full);

        // `Empty` + `LIMIT - 1` stays `Rects`.
        let mut below_limit = FrameDamage::Empty;
        below_limit.add_rects(vec![rect; limit - 1]);
        assert_eq!(below_limit, FrameDamage::Rects(vec![rect; limit - 1]));

        // `Rects` + 1 reaching `LIMIT` promotes to `Full`, and `Full` stays sticky.
        below_limit.add_rects(vec![rect]);
        assert_eq!(below_limit, FrameDamage::Full);
        below_limit.add_rects(vec![rect]);
        assert_eq!(below_limit, FrameDamage::Full);

        // `merge_commit`: `(Empty, None | Some([]), false) -> Empty`,
        // `(Empty, None | Some([]), true) -> Full`,
        // `(Empty, Some(rects), true) -> Rects`,
        // `(Rects, None | Some([]), false) -> unchanged`,
        // `(Rects, Some(rects), false)` appends via `add_rects`, and
        // `(Rects, None | Some([]), true) -> Full`.
        let mut commit_dmg = FrameDamage::Empty;
        commit_dmg.merge_commit(None, false);
        assert_eq!(commit_dmg, FrameDamage::Empty);
        commit_dmg.merge_commit(Some(Vec::new()), false);
        assert_eq!(commit_dmg, FrameDamage::Empty);
        commit_dmg.merge_commit(None, true);
        assert_eq!(commit_dmg, FrameDamage::Full);

        let mut empty_new_buf = FrameDamage::Empty;
        empty_new_buf.merge_commit(Some(Vec::new()), true);
        assert_eq!(empty_new_buf, FrameDamage::Full);

        let mut new_buf = FrameDamage::Empty;
        new_buf.merge_commit(Some(vec![rect]), true);
        assert_eq!(new_buf, FrameDamage::Rects(vec![rect]));

        let mut partial = FrameDamage::Rects(vec![rect]);
        partial.merge_commit(None, false);
        assert_eq!(partial, FrameDamage::Rects(vec![rect]));
        partial.merge_commit(Some(Vec::new()), false);
        assert_eq!(partial, FrameDamage::Rects(vec![rect]));
        partial.merge_commit(Some(vec![rect]), false);
        assert_eq!(partial, FrameDamage::Rects(vec![rect, rect]));
        partial.merge_commit(None, true);
        assert_eq!(partial, FrameDamage::Full);

        let mut partial_empty_new_buf = FrameDamage::Rects(vec![rect]);
        partial_empty_new_buf.merge_commit(Some(Vec::new()), true);
        assert_eq!(partial_empty_new_buf, FrameDamage::Full);
    }
}
