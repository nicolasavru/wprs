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
use crate::serialization::framing::TileRect;
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

/// Maximum number of 2D dirty tile rects retained in a back slot's history
/// before promoting to full-slot stale (`None`).
pub(crate) const MAX_TRACKED_DIRTY_RECTS: usize = 128;

/// Returns `true` if `outer` completely covers `inner`.
#[inline]
pub(crate) fn rect_contains(outer: TileRect, inner: TileRect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && u64::from(inner.x) + u64::from(inner.w) <= u64::from(outer.x) + u64::from(outer.w)
        && u64::from(inner.y) + u64::from(inner.h) <= u64::from(outer.y) + u64::from(outer.h)
}

/// Returns `true` if `rect` is completely covered by some single rect in `cover`.
#[inline]
pub(crate) fn rect_covered_by(rect: TileRect, cover: &[TileRect]) -> bool {
    cover.iter().any(|&c| rect_contains(c, rect))
}

/// Captures a frame's 2D dirty tile rects as initial back-slot history,
/// promoting to `None` (full-slot stale) on full frames or when exceeding
/// [`MAX_TRACKED_DIRTY_RECTS`].
pub(crate) fn snapshot_rect_history(cur: Option<&[TileRect]>) -> Option<Vec<TileRect>> {
    cur.filter(|rects| rects.len() <= MAX_TRACKED_DIRTY_RECTS)
        .map(ToOwned::to_owned)
}

/// Merges `cur` into a back slot's stale 2D tile-rect history `prev`, returning
/// `None` if either is a full-frame update or the deduplicated list exceeds
/// [`MAX_TRACKED_DIRTY_RECTS`].
pub(crate) fn combine_rect_history(
    prev: Option<Vec<TileRect>>,
    cur: Option<&[TileRect]>,
) -> Option<Vec<TileRect>> {
    let (mut prev, cur) = (prev?, cur?);
    for &rect in cur {
        if rect_covered_by(rect, &prev) {
            continue;
        }
        prev.retain(|&existing| !rect_contains(rect, existing));
        if prev.len() >= MAX_TRACKED_DIRTY_RECTS {
            return None;
        }
        prev.push(rect);
    }
    Some(prev)
}

/// Outcome of advancing a lazy 2-slot swapchain for a frame update.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SwapchainStep<H> {
    /// `active_buffer` was released and reused in place.
    ReusedActive,
    /// `active_buffer` was held; swapped to the released back slot whose stale
    /// region history relative to the previous active slot was `prev_history`.
    /// The previous active slot is now stored in `back_buffer`.
    Swapped { prev_history: H },
    /// Allocated a fresh slot because no released back slot was available. The
    /// previous active slot is now stored in `back_buffer`.
    Allocated,
}

/// Advances a lazy 2-slot swapchain, maintaining `back_buffer`'s stale region
/// history `H` relative to `active_buffer`.
///
/// `alloc_buffer` receives `back_buffer_held`: `false` on first-time lazy
/// double-buffer allocation and `true` when both slots are held by the
/// compositor (triple-buffer fallback). On allocation error, `active_buffer`
/// and `back_buffer` are left unchanged.
pub(crate) fn advance_two_slot_swapchain<P, B, H: Default, C: Copy>(
    pool: &mut P,
    active_buffer: &mut B,
    back_buffer: &mut Option<(B, H)>,
    cur: C,
    mut is_released: impl FnMut(&mut P, &B) -> bool,
    combine: impl FnOnce(H, C) -> H,
    snapshot: impl FnOnce(C) -> H,
    alloc_buffer: impl FnOnce(&mut P, bool) -> Result<B>,
) -> Result<SwapchainStep<H>> {
    if is_released(pool, active_buffer) {
        if let Some((_, back_history)) = back_buffer {
            *back_history = combine(std::mem::take(back_history), cur);
        }
        return Ok(SwapchainStep::ReusedActive);
    }

    if let Some((back_buf, prev_history)) = back_buffer.take() {
        if is_released(pool, &back_buf) {
            let old_active = std::mem::replace(active_buffer, back_buf);
            *back_buffer = Some((old_active, snapshot(cur)));
            return Ok(SwapchainStep::Swapped { prev_history });
        }
        *back_buffer = Some((back_buf, prev_history));
    }

    let back_buffer_held = back_buffer.is_some();
    let new_buffer = alloc_buffer(pool, back_buffer_held)?;
    let old_active = std::mem::replace(active_buffer, new_buffer);
    *back_buffer = Some((old_active, snapshot(cur)));
    Ok(SwapchainStep::Allocated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(y0: u32, y1: u32) -> RowBand {
        RowBand { y0, y1 }
    }

    fn tr(x: u32, y: u32, w: u32, h: u32) -> TileRect {
        TileRect { x, y, w, h }
    }

    #[test]
    fn test_rect_containment_and_history_tracking() {
        let parent = tr(0, 0, 256, 256);
        let q0 = tr(0, 0, 128, 128);
        let q1 = tr(128, 0, 128, 128);
        let shifted = tr(256, 0, 128, 128);

        assert!(rect_contains(parent, parent));
        assert!(rect_contains(parent, q0));
        assert!(rect_contains(parent, q1));
        assert!(!rect_contains(q0, parent));
        assert!(!rect_contains(parent, shifted));
        assert!(!rect_contains(tr(0, 0, 128, 128), tr(1, 0, 128, 128)));
        assert!(!rect_contains(tr(10, 10, 10, 10), tr(9, 10, 10, 10)));
        assert!(rect_covered_by(q0, &[shifted, parent]));
        assert!(!rect_covered_by(shifted, &[parent]));

        // Deduplicates repeated cursor-blink rects without growing history.
        let mut hist = snapshot_rect_history(Some(&[q0]));
        for _ in 0..(MAX_TRACKED_DIRTY_RECTS * 2) {
            hist = combine_rect_history(hist, Some(&[q0]));
        }
        assert_eq!(hist, Some(vec![q0]));

        // Larger incoming parent supersedes covered quadrants in place.
        hist = combine_rect_history(hist, Some(&[q1]));
        assert_eq!(hist, Some(vec![q0, q1]));
        hist = combine_rect_history(hist, Some(&[parent]));
        assert_eq!(hist, Some(vec![parent]));
        // Subsequent quadrant inside parent is ignored.
        hist = combine_rect_history(hist, Some(&[q0]));
        assert_eq!(hist, Some(vec![parent]));

        // Exceeding `MAX_TRACKED_DIRTY_RECTS` promotes to `None` (full-slot stale).
        let distinct: Vec<TileRect> = (0..=MAX_TRACKED_DIRTY_RECTS as u32)
            .map(|i| tr(i * 128, 0, 128, 128))
            .collect();
        assert_eq!(
            snapshot_rect_history(Some(&distinct[..MAX_TRACKED_DIRTY_RECTS])),
            Some(distinct[..MAX_TRACKED_DIRTY_RECTS].to_vec())
        );
        assert_eq!(snapshot_rect_history(Some(&distinct)), None);
        assert_eq!(
            combine_rect_history(
                Some(distinct[..MAX_TRACKED_DIRTY_RECTS].to_vec()),
                Some(&distinct[MAX_TRACKED_DIRTY_RECTS..])
            ),
            None
        );

        // Non-incremental (`None`) always yields `None`.
        assert_eq!(snapshot_rect_history(None), None);
        assert_eq!(combine_rect_history(Some(vec![q0]), None), None);
        assert_eq!(combine_rect_history(None, Some(&[q0])), None);
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
            Some([b(1, 2)].as_slice()),
            |_, _| false,
            |prev, cur| combine_band_history(prev, cur, 20),
            |cur| cur.map(ToOwned::to_owned),
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
            Some([b(6, 8)].as_slice()),
            |_, _| false,
            |prev, cur| combine_band_history(prev, cur, 20),
            |cur| cur.map(ToOwned::to_owned),
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
