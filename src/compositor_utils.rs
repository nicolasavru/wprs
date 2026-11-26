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

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Duration;

use smithay::output::Mode;
use smithay::output::Output;
use smithay::output::Scale;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Buffer as BufferCoords;
use smithay::utils::Logical;
use smithay::utils::Rectangle;
use smithay::utils::Size;
use smithay::utils::Transform;
use smithay::utils::user_data::UserDataMap;
use smithay::wayland::compositor::Damage;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::shm;
use smithay::wayland::shm::BufferAccessError;
use smithay::wayland::shm::BufferData;

use crate::buffer_pointer::BufferPointer;
use crate::constants;
use crate::prelude::*;
use crate::serialization::framing::RowBand;
use crate::serialization::geometry::Rectangle as SerRectangle;
use crate::serialization::wayland::OutputInfo;
use crate::serialization::wayland::ViewportState;

pub fn with_buffer_contents<F, T>(buffer: &WlBuffer, f: F) -> Result<T, BufferAccessError>
where
    F: FnOnce(BufferPointer<u8>, BufferData) -> T,
{
    shm::with_buffer_contents(buffer, |ptr, len, spec| {
        // A failed `wl_shm_pool.resize` (`mremap`) leaves Smithay's pool map
        // null after posting `InvalidFd` to the client.
        if ptr.is_null() {
            return Err(BufferAccessError::BadMap);
        }
        // Smithay's `wl_shm_pool.create_buffer` handler already validates
        // `offset >= 0`, `width > 0`, `height > 0`, `stride >= width * bpp`,
        // and `offset + height * stride <= pool.size` (`<= len`), and refuses
        // to shrink pools.
        let start = usize::try_from(spec.offset).expect("validated by Smithay create_buffer");
        let height = usize::try_from(spec.height).expect("validated by Smithay create_buffer");
        let stride = usize::try_from(spec.stride).expect("validated by Smithay create_buffer");
        let buffer_len = height
            .checked_mul(stride)
            .expect("validated by Smithay create_buffer");
        let end = start
            .checked_add(buffer_len)
            .expect("validated by Smithay create_buffer");
        assert!(end <= len, "shm buffer range {start}..{end} exceeds {len}");
        // SAFETY: `ptr` is non-null and `start..end` lies within the `len`-byte
        // mapping provided by Smithay's `shm::with_buffer_contents`.
        unsafe {
            let ptr = ptr.add(start);
            let buf = BufferPointer::new(&ptr, buffer_len);
            Ok(f(buf, spec))
        }
    })
    .flatten()
}

fn fault_probe_range(data: BufferPointer<u8>, start: usize, end: usize) {
    if start >= end {
        return;
    }
    assert!(end <= data.len());
    let base = data.ptr();
    // Probe one byte per 4 KiB page (plus the final byte) on the main thread
    // while Smithay's thread-local `SIGBUS_GUARD` is armed. This catches shm
    // truncation or holes already present at probe time (Smithay's handler
    // replaces the VMA with anonymous zero pages); truncation racing the
    // `WorkerPool` reads afterwards still aborts `wprsd` with SIGBUS because
    // Smithay's guard is thread-local and Smithay does not expose the pool fd
    // for `F_SEAL_SHRINK`.
    let mut offset = start;
    while offset < end {
        // SAFETY: `offset < end <= data.len()`, and `read_volatile` prevents
        // the compiler from eliding the page-touching load.
        unsafe {
            std::ptr::read_volatile(base.add(offset));
        }
        offset = offset.saturating_add(4096);
    }
    // SAFETY: `start < end <= data.len()`, so `end - 1 < data.len()`.
    unsafe {
        std::ptr::read_volatile(base.add(end - 1));
    }
}

pub fn fault_probe_buffer(data: BufferPointer<u8>, stride: usize, bands: Option<&[RowBand]>) {
    match bands {
        None => fault_probe_range(data, 0, data.len()),
        Some(bands) => {
            for band in bands {
                fault_probe_range(data, band.y0 as usize * stride, band.y1 as usize * stride);
            }
        },
    }
}

// Based on https://github.com/Smithay/smithay/blob/b1c682742ac7b9fa08736476df3e651489709ac2/src/desktop/wayland/utils.rs.
#[derive(Debug, Default)]
pub(crate) struct SurfaceFrameThrottlingState(Mutex<Option<Duration>>);

impl SurfaceFrameThrottlingState {
    pub fn update(&self, time: Duration, throttle: Duration) -> bool {
        let mut guard = self.0.lock().unwrap();
        let send_throttled_frame = guard
            .map(|last| time.saturating_sub(last) > throttle)
            .unwrap_or(true);
        if send_throttled_frame {
            *guard = Some(time);
        }
        send_throttled_frame
    }
}

pub fn send_frames(
    surface: &WlSurface,
    data_map: &UserDataMap,
    surface_attributes: &mut SurfaceAttributes,
    time: Duration,
    throttle: Duration,
) -> Result<()> {
    data_map.insert_if_missing_threadsafe(SurfaceFrameThrottlingState::default);
    let surface_frame_throttling_state = data_map
        .get::<SurfaceFrameThrottlingState>()
        .location(loc!())?;
    let frame_overdue = surface_frame_throttling_state.update(time, throttle);

    if frame_overdue {
        for callback in surface_attributes.frame_callbacks.drain(..) {
            debug!(
                "Sending callback for surface {:?}: {:?}",
                surface.id(),
                callback.id()
            );
            callback.done(time.as_millis() as u32);
        }
    }
    Ok(())
}

pub fn update_output(local_output: &mut Output, output: OutputInfo) {
    let current_mode = local_output.current_mode().unwrap_or(Mode {
        size: (0, 0).into(),
        refresh: 0,
    });
    let received_mode = Mode {
        size: output.mode.dimensions.into(),
        refresh: output.mode.refresh_rate,
    };
    if current_mode != received_mode {
        local_output.delete_mode(current_mode);
    }

    local_output.change_current_state(
        Some(received_mode),
        Some(output.transform.into()),
        Some(Scale::Integer(output.scale_factor)),
        Some(output.location.into()),
    );

    if output.mode.preferred {
        local_output.set_preferred(received_mode);
    }
}

pub fn update_surface_outputs<'a, F>(
    surface: &WlSurface,
    new_ids: &HashSet<u32>,
    old_ids: &HashSet<u32>,
    output_accessor: F,
) where
    F: Fn(&u32) -> Option<&'a Output>,
{
    let entered_ids = new_ids.difference(old_ids);
    let left_ids = old_ids.difference(new_ids);

    // careful, a surface can be on multiple outputs, and the surface scale is the largest scale among them
    for id in entered_ids {
        let output = output_accessor(id);
        if let Some(output) = output {
            output.enter(surface);
        }
    }

    for id in left_ids {
        let output = output_accessor(id);
        if let Some(output) = output {
            output.leave(surface);
        }
    }
}

/// Convert a damage rectangle from surface coordinates to buffer coordinates.
///
/// The viewport is part of this mapping, not just buffer_scale: chromium leaves
/// buffer_scale at 1 and scales entirely through the viewport, so using
/// buffer_scale alone makes this the identity and the damage then covers only
/// the top-left corner of the buffer.
fn surface_damage_to_buffer(
    rect: Rectangle<i32, Logical>,
    buffer_scale: i32,
    transform: Transform,
    viewport_state: Option<&ViewportState>,
    buffer_size: (i32, i32),
) -> Rectangle<i32, BufferCoords> {
    let src = viewport_state.and_then(|viewport_state| viewport_state.src);
    let dst = viewport_state.and_then(|viewport_state| viewport_state.dst);
    let scale = f64::from(buffer_scale);

    // Whole buffer in surface-oriented buffer pixels (axes transposed for
    // rotating transforms, without dividing by `buffer_scale` and multiplying
    // back).
    let oriented_size: Size<i32, BufferCoords> = transform.transform_size(Size::from(buffer_size));
    let area_w = f64::from(oriented_size.w);
    let area_h = f64::from(oriented_size.h);

    let (src_px_w, src_px_h, off_px_x, off_px_y) = match src {
        Some(src) => (
            src.size.w * scale,
            src.size.h * scale,
            src.loc.x * scale,
            src.loc.y * scale,
        ),
        None => (area_w, area_h, 0.0, 0.0),
    };
    let (scale_w, scale_h) = match dst {
        Some(dst) if dst.w > 0 && dst.h > 0 => {
            (src_px_w / f64::from(dst.w), src_px_h / f64::from(dst.h))
        },
        _ => (scale, scale),
    };

    let rect = rect.to_f64();
    let px0 = (rect.loc.x * scale_w + off_px_x).clamp(0.0, area_w);
    let py0 = (rect.loc.y * scale_h + off_px_y).clamp(0.0, area_h);
    let px1 = ((rect.loc.x + rect.size.w) * scale_w + off_px_x).clamp(0.0, area_w);
    let py1 = ((rect.loc.y + rect.size.h) * scale_h + off_px_y).clamp(0.0, area_h);
    if px1 <= px0 || py1 <= py0 {
        return Rectangle::default();
    }

    // Round outwards at buffer-pixel resolution: too much damage costs a
    // repaint, too little leaves stale pixels. `buffer_transform` is what the
    // client already applied to the buffer, so inverting it maps back to raw
    // buffer coordinates.
    transform.invert().transform_rect_in(
        Rectangle::<i32, BufferCoords>::from_extremities(
            (px0.floor() as i32, py0.floor() as i32),
            (px1.ceil() as i32, py1.ceil() as i32),
        ),
        &oriented_size,
    )
}

/// Bounds retained `SurfaceAttributes::damage` at `SENT_DAMAGE_LIMIT` by
/// collapsing it to a single full-buffer rect so damage queued across failed
/// or bufferless commits cannot grow without bound or be clipped by an offset
/// `wp_viewport` source crop.
pub fn cap_surface_damage(damage: &mut Vec<Damage>) {
    if damage.len() >= constants::SENT_DAMAGE_LIMIT {
        damage.clear();
        damage.push(Damage::Buffer(Rectangle::new(
            (0, 0).into(),
            (i32::MAX, i32::MAX).into(),
        )));
    }
}

/// Converts committed damage to buffer-pixel rects. Callers must pass
/// Smithay-validated `buffer_scale >= 1` and positive `buffer_size`.
pub fn damage_to_buffer_rects(
    raw_damage: &[Damage],
    buffer_scale: i32,
    transform: Transform,
    viewport_state: Option<&ViewportState>,
    buffer_size: (i32, i32),
) -> Vec<SerRectangle<i32>> {
    raw_damage
        .iter()
        .map(|damage| match damage {
            Damage::Buffer(rect) => *rect,
            Damage::Surface(rect) => surface_damage_to_buffer(
                *rect,
                buffer_scale,
                transform,
                viewport_state,
                buffer_size,
            ),
        })
        .map(Into::into)
        .collect()
}

#[cfg(test)]
mod tests {
    use smithay::utils::Point;

    use super::*;
    use crate::serialization::geometry;

    fn viewport(src: Option<geometry::Rectangle<f64>>, dst: Option<(i32, i32)>) -> ViewportState {
        ViewportState {
            src,
            dst: dst.map(|(w, h)| geometry::Size { w, h }),
        }
    }

    fn damage(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    fn buffer_rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, BufferCoords> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    /// A surface with no viewport and a buffer matching its size needs no
    /// conversion at all.
    #[test]
    fn identity() {
        assert_eq!(
            surface_damage_to_buffer(
                damage(10, 20, 30, 40),
                1,
                Transform::Normal,
                None,
                (100, 100),
            ),
            buffer_rect(10, 20, 30, 40),
        );
    }

    #[test]
    fn buffer_scale_without_viewport() {
        assert_eq!(
            surface_damage_to_buffer(
                damage(10, 20, 30, 40),
                2,
                Transform::Normal,
                None,
                (200, 200),
            ),
            buffer_rect(20, 40, 60, 80),
        );
    }

    /// Regression test for damage covering only the top-left corner of the
    /// buffer. Chromium leaves buffer_scale at 1 and scales through the
    /// viewport instead, so the destination ratio is the only thing that
    /// relates surface coordinates to the 2x buffer.
    #[test]
    fn viewport_destination_scales_damage() {
        let vp = viewport(None, Some((1733, 729)));
        let buffer = (3466, 1458);

        // Full-surface damage has to reach the whole buffer.
        assert_eq!(
            surface_damage_to_buffer(
                damage(0, 0, 1733, 729),
                1,
                Transform::Normal,
                Some(&vp),
                buffer,
            ),
            buffer_rect(0, 0, 3466, 1458),
        );

        // And a partial rect keeps its position relative to the buffer.
        assert_eq!(
            surface_damage_to_buffer(
                damage(16, 153, 1701, 403),
                1,
                Transform::Normal,
                Some(&vp),
                buffer,
            ),
            buffer_rect(32, 306, 3402, 806),
        );

        // Rotating transform transposes the buffer size before dividing by `dst`.
        let rotated_vp = viewport(None, Some((729, 1733)));
        assert_eq!(
            surface_damage_to_buffer(
                damage(0, 0, 729, 1733),
                1,
                Transform::_90,
                Some(&rotated_vp),
                buffer,
            ),
            buffer_rect(0, 0, 3466, 1458),
        );
    }

    /// The viewport source rectangle is defined in surface-local coordinates,
    /// so it both sets the pre-destination size and offsets the result.
    #[test]
    fn viewport_source_crops_and_offsets() {
        let viewport = viewport(
            Some(geometry::Rectangle::new(10.0, 10.0, 100.0, 50.0)),
            Some((200, 100)),
        );
        assert_eq!(
            surface_damage_to_buffer(
                damage(0, 0, 200, 100),
                1,
                Transform::Normal,
                Some(&viewport),
                (400, 200),
            ),
            buffer_rect(10, 10, 100, 50),
        );

        // `wp_viewport` `src` is in surface-local coordinates, so `buffer_scale = 2`
        // scales both `src.loc` and `src.size` into buffer pixels.
        assert_eq!(
            surface_damage_to_buffer(
                damage(0, 0, 200, 100),
                2,
                Transform::Normal,
                Some(&viewport),
                (400, 200),
            ),
            buffer_rect(20, 20, 200, 100),
        );
    }

    /// A rotating transform mirrors the rect within the whole buffer, not
    /// within the viewport source rectangle.
    #[test]
    fn viewport_source_with_transform() {
        let viewport = viewport(
            Some(geometry::Rectangle::new(10.0, 10.0, 100.0, 50.0)),
            Some((200, 100)),
        );
        assert_eq!(
            surface_damage_to_buffer(
                damage(0, 0, 200, 100),
                1,
                Transform::_180,
                Some(&viewport),
                (400, 200),
            ),
            buffer_rect(290, 140, 100, 50),
        );
    }

    /// Damage must never come out too small: a rect that lands on fractional
    /// coordinates has to grow to cover them at buffer-pixel resolution.
    #[test]
    fn rounds_outward() {
        let viewport_1x = viewport(None, Some((100, 100)));
        // 3/2 scaling: x 1 -> 1.5 floors to 1, right edge 2 -> 3.0 stays 3.
        assert_eq!(
            surface_damage_to_buffer(
                damage(1, 1, 1, 1),
                1,
                Transform::Normal,
                Some(&viewport_1x),
                (150, 150),
            ),
            buffer_rect(1, 1, 2, 2),
        );

        // Fractional `wp_viewport` + `buffer_scale = 2` rounds at buffer-pixel
        // resolution (`[1.5, 3.0)` -> `[1, 3)`), not coarse logical resolution (`[0, 4)`).
        let viewport_2x = viewport(None, Some((200, 200)));
        assert_eq!(
            surface_damage_to_buffer(
                damage(1, 1, 1, 1),
                2,
                Transform::Normal,
                Some(&viewport_2x),
                (300, 300),
            ),
            buffer_rect(1, 1, 2, 2),
        );

        // Odd-dimension rotated/mirrored HiDPI buffers must map exact edge rows/cols
        // without logical `area.ceil()` shifting by `buffer_scale` or overshooting.
        for (transform, buffer, expected) in [
            (Transform::_180, (100, 101), buffer_rect(80, 81, 20, 20)),
            (
                Transform::Flipped180,
                (100, 101),
                buffer_rect(0, 81, 20, 20),
            ),
            (Transform::_90, (100, 101), buffer_rect(0, 81, 20, 20)),
            (Transform::Flipped, (101, 100), buffer_rect(81, 0, 20, 20)),
            (Transform::_270, (101, 100), buffer_rect(81, 0, 20, 20)),
        ] {
            assert_eq!(
                surface_damage_to_buffer(damage(0, 0, 10, 10), 2, transform, None, buffer),
                expected,
                "{transform:?} {buffer:?}"
            );
        }

        // Non-power-of-two scale (`7`) stays within exact buffer bounds without
        // 1-ULP `(w / scale) * scale` overshoot (`29`).
        assert_eq!(
            surface_damage_to_buffer(damage(0, 0, 5, 5), 7, Transform::Normal, None, (29, 29)),
            buffer_rect(0, 0, 29, 29),
        );
    }

    /// Every transform, against the region the compositor actually samples.
    ///
    /// Confirmed against mutter: with any of these wrong the window stops
    /// repainting entirely.
    #[test]
    fn every_transform_maps_to_the_displayed_region() {
        // A 1200x800 buffer shown through a viewport that crops to 560x360 and
        // scales it down to a 600x400 surface.
        let viewport = viewport(
            Some(geometry::Rectangle::new(20.0, 20.0, 560.0, 360.0)),
            Some((600, 400)),
        );
        let buffer = (1200, 800);

        let cases = [
            (Transform::Normal, buffer_rect(20, 20, 560, 360)),
            (Transform::_90, buffer_rect(20, 220, 360, 560)),
            (Transform::_180, buffer_rect(620, 420, 560, 360)),
            (Transform::_270, buffer_rect(820, 20, 360, 560)),
            (Transform::Flipped, buffer_rect(620, 20, 560, 360)),
            (Transform::Flipped90, buffer_rect(20, 20, 360, 560)),
            (Transform::Flipped180, buffer_rect(20, 420, 560, 360)),
            (Transform::Flipped270, buffer_rect(820, 220, 360, 560)),
        ];

        for (transform, expected) in cases {
            let damage = surface_damage_to_buffer(
                damage(0, 0, 600, 400),
                1,
                transform,
                Some(&viewport),
                buffer,
            );
            assert_eq!(damage, expected, "{transform:?}");

            // Whatever the transform, the damage has to land inside the buffer.
            assert!(
                damage.loc.x >= 0 && damage.loc.y >= 0,
                "{transform:?} damage starts outside the buffer: {damage:?}"
            );
            assert!(
                damage.loc.x + damage.size.w <= 1200 && damage.loc.y + damage.size.h <= 800,
                "{transform:?} damage runs past the buffer: {damage:?}"
            );
        }
    }

    #[test]
    fn full_surface_max_and_negative_damage_clamps_under_transforms() {
        let transforms = [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ];
        for transform in transforms {
            let full = surface_damage_to_buffer(
                damage(0, 0, i32::MAX, i32::MAX),
                2,
                transform,
                None,
                (3840, 2160),
            );
            assert_eq!(full, buffer_rect(0, 0, 3840, 2160), "{transform:?}");
            assert_eq!(
                crate::filtering::compute_partial_row_bands(&[full.into()], 3840, 2160),
                None,
                "{transform:?}"
            );

            let neg = surface_damage_to_buffer(
                damage(i32::MIN / 2, -50, i32::MAX, i32::MAX),
                2,
                transform,
                None,
                (3840, 2160),
            );
            assert_eq!(neg, buffer_rect(0, 0, 3840, 2160), "{transform:?}");

            // Completely off-screen, edge-touching, and non-positive rects
            // clamp to empty `Rectangle::default()` (`Size::new`
            // `debug_assert`s non-negative dimensions, while release builds
            // accept negative `wl_surface.damage` sizes into `pub w`/`pub h`).
            let (logical_w, logical_h) = match transform {
                Transform::Normal
                | Transform::_180
                | Transform::Flipped
                | Transform::Flipped180 => (1920, 1080),
                Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270 => {
                    (1080, 1920)
                },
            };
            let mut neg_w = damage(10, 10, 0, 20);
            neg_w.size.w = -5;
            let mut neg_h = damage(10, 10, 20, 0);
            neg_h.size.h = -5;
            for empty_rect in [
                damage(-20, 10, 20, 20),
                damage(logical_w, 10, 20, 20),
                damage(10, -20, 20, 20),
                damage(10, logical_h, 20, 20),
                damage(10, 10, 0, 20),
                damage(10, 10, 20, 0),
                neg_w,
                neg_h,
            ] {
                assert_eq!(
                    surface_damage_to_buffer(empty_rect, 2, transform, None, (3840, 2160)),
                    Rectangle::default(),
                    "{transform:?} {empty_rect:?}"
                );
            }
        }
    }

    #[test]
    fn cap_surface_damage_and_damage_to_buffer_rects() {
        let mut raw_damage: Vec<Damage> = (0..constants::SENT_DAMAGE_LIMIT - 1)
            .map(|i| {
                if i == 0 {
                    Damage::Buffer(buffer_rect(5, 6, 7, 8))
                } else {
                    Damage::Surface(damage(10, 20, 30, 40))
                }
            })
            .collect();
        cap_surface_damage(&mut raw_damage);
        assert_eq!(raw_damage.len(), constants::SENT_DAMAGE_LIMIT - 1);

        let converted =
            damage_to_buffer_rects(&raw_damage, 2, Transform::Normal, None, (3840, 2160));
        assert_eq!(converted.len(), constants::SENT_DAMAGE_LIMIT - 1);
        assert_eq!(converted[0], SerRectangle::new(5, 6, 7, 8));
        assert_eq!(converted[1], SerRectangle::new(20, 40, 60, 80));

        // At `SENT_DAMAGE_LIMIT`, `cap_surface_damage` collapses to a single
        // `Damage::Buffer` rect that forces a full-buffer sync (`None`) even
        // under an offset `wp_viewport` source crop covering `< 75%` of height.
        raw_damage.push(Damage::Surface(damage(1, 1, 2, 2)));
        cap_surface_damage(&mut raw_damage);
        assert_eq!(
            raw_damage,
            vec![Damage::Buffer(buffer_rect(0, 0, i32::MAX, i32::MAX))]
        );

        let cropped_vp = viewport(
            Some(geometry::Rectangle::new(0.0, 400.0, 1920.0, 680.0)),
            Some((1920, 680)),
        );
        for transform in [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ] {
            let capped_rects =
                damage_to_buffer_rects(&raw_damage, 2, transform, Some(&cropped_vp), (1920, 1080));
            assert_eq!(
                capped_rects,
                vec![SerRectangle::new(0, 0, i32::MAX, i32::MAX)]
            );
            assert_eq!(
                crate::filtering::compute_partial_row_bands(&capped_rects, 1920, 1080),
                None,
                "{transform:?}"
            );
        }
    }

    #[test]
    fn test_fault_probe_buffer_bounds_and_bands() {
        for len in [0usize, 1, 4096, 8195] {
            let buf = vec![0x5Au8; len];
            let ptr = buf.as_ptr();
            let bp = unsafe { BufferPointer::new(&ptr, len) };
            fault_probe_buffer(bp, len.max(1), None);
        }

        let buf = vec![0xA5u8; 192 * 64];
        let ptr = buf.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, buf.len()) };
        fault_probe_buffer(bp, 64, Some(&[]));
        fault_probe_buffer(
            bp,
            64,
            Some(&[
                RowBand { y0: 0, y1: 0 },
                RowBand { y0: 5, y1: 0 },
                RowBand { y0: 192, y1: 192 },
            ]),
        );
        fault_probe_buffer(bp, 64, Some(&[RowBand { y0: 2, y1: 140 }]));
    }

    #[test]
    #[should_panic]
    fn test_fault_probe_buffer_out_of_bounds() {
        let buf = [0u8; 191];
        let ptr = buf.as_ptr();
        let bp = unsafe { BufferPointer::new(&ptr, buf.len()) };
        fault_probe_buffer(
            bp,
            64,
            Some(&[RowBand { y0: 0, y1: 1 }, RowBand { y0: 2, y1: 3 }]),
        );
    }
}
