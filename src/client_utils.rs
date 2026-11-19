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

use smithay_client_toolkit::data_device_manager::data_device::DataDevice;
use smithay_client_toolkit::primary_selection::device::PrimarySelectionDevice;
use smithay_client_toolkit::reexports::client::protocol::wl_keyboard::WlKeyboard;
use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;

use crate::constants;
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

#[cfg(test)]
mod tests {
    use super::*;

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
