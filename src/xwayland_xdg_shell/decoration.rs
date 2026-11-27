use std::num::NonZeroU32;
use std::time::Duration;

use smithay::utils::Logical;
use smithay::utils::Rectangle;
use smithay::utils::Serial;
use smithay::xwayland::X11Surface;
use smithay_client_toolkit::compositor::SurfaceData;
use smithay_client_toolkit::reexports::client::Connection;
use smithay_client_toolkit::reexports::client::Proxy;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::client::protocol::wl_pointer::WlPointer;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;
use smithay_client_toolkit::reexports::csd_frame::CursorIcon;
use smithay_client_toolkit::reexports::csd_frame::DecorationsFrame;
use smithay_client_toolkit::reexports::csd_frame::FrameAction;
use smithay_client_toolkit::reexports::csd_frame::FrameClick;
use smithay_client_toolkit::reexports::csd_frame::ResizeEdge;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_toplevel::ResizeEdge as SctkResizeEdge;
use smithay_client_toolkit::seat::pointer::BTN_LEFT;
use smithay_client_toolkit::seat::pointer::BTN_RIGHT;
use smithay_client_toolkit::seat::pointer::PointerData;
use smithay_client_toolkit::seat::pointer::PointerEvent;
use smithay_client_toolkit::seat::pointer::PointerEventKind;
use smithay_client_toolkit::shell::xdg::fallback_frame::FallbackFrame;
use tracing::warn;

use crate::prelude::*;
use crate::xwayland_xdg_shell::WprsState;
use crate::xwayland_xdg_shell::client::Role;
use crate::xwayland_xdg_shell::client::WprsClientState;
use crate::xwayland_xdg_shell::client::XWaylandSubSurface;
use crate::xwayland_xdg_shell::client::XWaylandXdgToplevel;
use crate::xwayland_xdg_shell::xsurface_from_client_surface;

/// Active interactive resize state for a decorated [`XWaylandSubSurface`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubSurfaceResizeState {
    pub edge: ResizeEdge,
    pub start_geometry: Rectangle<i32, Logical>,
    pub start_pointer: (f64, f64),
}

pub(crate) fn resize_edge_cursor(edge: ResizeEdge) -> CursorIcon {
    match edge {
        ResizeEdge::Top => CursorIcon::NResize,
        ResizeEdge::TopRight => CursorIcon::NeResize,
        ResizeEdge::Right => CursorIcon::EResize,
        ResizeEdge::BottomRight => CursorIcon::SeResize,
        ResizeEdge::Bottom => CursorIcon::SResize,
        ResizeEdge::BottomLeft => CursorIcon::SwResize,
        ResizeEdge::Left => CursorIcon::WResize,
        ResizeEdge::TopLeft => CursorIcon::NwResize,
        _ => CursorIcon::Default,
    }
}

/// Computes the target X11 geometry for an interactive [`XWaylandSubSurface`]
/// border drag, compensating for how each [`FallbackFrame`] border subsurface's
/// root origin moves as `cur_geo` changes from `start_geo`.
pub(crate) fn compute_subsurface_resize_geometry(
    start_geo: Rectangle<i32, Logical>,
    cur_geo: Rectangle<i32, Logical>,
    edge: ResizeEdge,
    start_pointer: (f64, f64),
    cur_pointer: (f64, f64),
    min_size: Option<(i32, i32)>,
    max_size: Option<(i32, i32)>,
) -> Rectangle<i32, Logical> {
    // `FallbackFrame` places its border subsurfaces relative to the main surface:
    // - `TOP_BORDER` (`Top`, `TopLeft`, `TopRight`): `(-B, -(H + B))`
    // - `BOTTOM_BORDER` (`Bottom`, `BottomLeft`, `BottomRight`): `(-B, height)`
    // - `LEFT_BORDER` (`Left`): `(-B, -H)`
    // - `RIGHT_BORDER` (`Right`): `(width, -H)`
    // Compute how far the grabbed border subsurface's root origin shifted from
    // `start_geo` to `cur_geo` so `cur_pointer` maps back to total displacement.
    let origin_shift_x = match edge {
        ResizeEdge::Right => {
            (cur_geo.loc.x + cur_geo.size.w) - (start_geo.loc.x + start_geo.size.w)
        },
        _ => cur_geo.loc.x - start_geo.loc.x,
    };
    let origin_shift_y = match edge {
        ResizeEdge::Bottom | ResizeEdge::BottomLeft | ResizeEdge::BottomRight => {
            (cur_geo.loc.y + cur_geo.size.h) - (start_geo.loc.y + start_geo.size.h)
        },
        _ => cur_geo.loc.y - start_geo.loc.y,
    };

    let total_dx = (cur_pointer.0 - start_pointer.0).round() as i32 + origin_shift_x;
    let total_dy = (cur_pointer.1 - start_pointer.1).round() as i32 + origin_shift_y;

    let min_w = min_size.map_or(1, |(w, _)| w).max(1);
    let min_h = min_size.map_or(1, |(_, h)| h).max(1);
    let max_w = max_size
        .and_then(|(w, _)| (w >= min_w).then_some(w))
        .unwrap_or(i32::MAX);
    let max_h = max_size
        .and_then(|(_, h)| (h >= min_h).then_some(h))
        .unwrap_or(i32::MAX);

    let (new_x, new_w) = match edge {
        ResizeEdge::Left | ResizeEdge::TopLeft | ResizeEdge::BottomLeft => {
            let w = start_geo
                .size
                .w
                .saturating_sub(total_dx)
                .clamp(min_w, max_w);
            (start_geo.loc.x + (start_geo.size.w - w), w)
        },
        ResizeEdge::Right | ResizeEdge::TopRight | ResizeEdge::BottomRight => {
            let w = start_geo
                .size
                .w
                .saturating_add(total_dx)
                .clamp(min_w, max_w);
            (start_geo.loc.x, w)
        },
        _ => (start_geo.loc.x, start_geo.size.w),
    };

    let (new_y, new_h) = match edge {
        ResizeEdge::Top | ResizeEdge::TopLeft | ResizeEdge::TopRight => {
            let h = start_geo
                .size
                .h
                .saturating_sub(total_dy)
                .clamp(min_h, max_h);
            (start_geo.loc.y + (start_geo.size.h - h), h)
        },
        ResizeEdge::Bottom | ResizeEdge::BottomLeft | ResizeEdge::BottomRight => {
            let h = start_geo
                .size
                .h
                .saturating_add(total_dy)
                .clamp(min_h, max_h);
            (start_geo.loc.y, h)
        },
        _ => (start_geo.loc.y, start_geo.size.h),
    };

    Rectangle::new((new_x, new_y).into(), (new_w, new_h).into())
}

fn parent(surface: &WlSurface) -> Option<&WlSurface> {
    surface.data::<SurfaceData<()>>()?.parent_surface()
}

fn surface_tree_root(surface: &WlSurface) -> &WlSurface {
    let mut surface = surface;
    while let Some(parent) = parent(surface) {
        surface = parent;
    }
    surface
}

#[instrument(skip(state, conn, pointer), level = "debug")]
pub fn handle_window_frame_pointer_event(
    state: &mut WprsState,
    conn: &Connection,
    qh: &QueueHandle<WprsState>,
    pointer: &WlPointer,
    event: &PointerEvent,
) -> Result<()> {
    let parent_surface = parent(&event.surface).unwrap_or(surface_tree_root(&event.surface));
    let Some(xwayland_surface) =
        xsurface_from_client_surface(&state.surface_bimap, &mut state.surfaces, parent_surface)
    else {
        info!("frame owner not found");
        return Ok(());
    };

    let Some(x11_surface) = xwayland_surface.x11_surface.as_ref().cloned() else {
        return Ok(());
    };
    match &mut xwayland_surface.role {
        Some(Role::SubSurface(subsurface)) => {
            subsurface
                .handle_pointer_event(
                    &mut state.client_state,
                    &x11_surface,
                    conn,
                    qh,
                    pointer,
                    event,
                )
                .location(loc!())?;
        },
        Some(Role::XdgToplevel(toplevel)) => {
            toplevel
                .handle_pointer_event(
                    &mut state.client_state,
                    &x11_surface,
                    conn,
                    qh,
                    pointer,
                    event,
                )
                .location(loc!())?;
        },
        _ => unreachable!(
            "expected role xdg_toplevel or subsurface, found role {:?}",
            &xwayland_surface.role
        ),
    }
    Ok(())
}

pub trait FramedSurface {
    fn frame_action(
        &mut self,
        x11_surface: &X11Surface,
        client_pointer: &WlPointer,
        serial: Serial,
        action: FrameAction,
        position: (f64, f64),
    ) -> Result<()>;

    fn frame(&mut self) -> Option<&mut FallbackFrame<WprsState>>;

    fn handle_pointer_event_inner(
        &mut self,
        client_state: &mut WprsClientState,
        x11_surface: &X11Surface,
        qh: &QueueHandle<WprsState>,
        pointer: &WlPointer,
        event: &PointerEvent,
    ) -> Result<Option<CursorIcon>>;

    fn handle_pointer_event(
        &mut self,
        client_state: &mut WprsClientState,
        x11_surface: &X11Surface,
        conn: &Connection,
        qh: &QueueHandle<WprsState>,
        pointer: &WlPointer,
        event: &PointerEvent,
    ) -> Result<()> {
        let new_cursor = self
            .handle_pointer_event_inner(client_state, x11_surface, qh, pointer, event)
            .location(loc!())?;

        if let (Some(new_cursor), cur_cursor) =
            (new_cursor, client_state.cursor_icon.unwrap_or_default())
        {
            // when entering a surface, the current cursor is always undefined
            if new_cursor != cur_cursor || matches!(event.kind, PointerEventKind::Enter { .. }) {
                client_state.cursor_icon = Some(new_cursor);
                if let Some(themed_pointer) = client_state
                    .seat_objects
                    .last()
                    .and_then(|seat| seat.pointer.as_ref())
                {
                    let _ = themed_pointer.set_cursor(conn, new_cursor);
                }
            }
        }

        if let Some(frame) = self.frame()
            && !frame.is_hidden()
            && frame.is_dirty()
        {
            frame.draw();
        }

        Ok(())
    }
}

impl FramedSurface for XWaylandXdgToplevel {
    fn frame_action(
        &mut self,
        x11_surface: &X11Surface,
        client_pointer: &WlPointer,
        serial: Serial,
        action: FrameAction,
        _position: (f64, f64),
    ) -> Result<()> {
        let window = &self.local_window;
        let pointer_data = client_pointer.data::<PointerData<()>>().unwrap();
        let client_seat = pointer_data.seat();
        match action {
            FrameAction::Close => {
                x11_surface.close().location(loc!())?;
            },
            FrameAction::Minimize => {
                window.set_minimized();
            },
            FrameAction::Maximize => {
                window.set_maximized();
            },
            FrameAction::UnMaximize => {
                window.unset_maximized();
            },
            FrameAction::ShowMenu(x, y) => {
                window.show_window_menu(client_seat, serial.into(), (x, y));
            },
            FrameAction::Resize(edge) => {
                let edge = match edge {
                    ResizeEdge::None => SctkResizeEdge::None,
                    ResizeEdge::Top => SctkResizeEdge::Top,
                    ResizeEdge::Bottom => SctkResizeEdge::Bottom,
                    ResizeEdge::Left => SctkResizeEdge::Left,
                    ResizeEdge::TopLeft => SctkResizeEdge::TopLeft,
                    ResizeEdge::BottomLeft => SctkResizeEdge::BottomLeft,
                    ResizeEdge::Right => SctkResizeEdge::Right,
                    ResizeEdge::TopRight => SctkResizeEdge::TopRight,
                    ResizeEdge::BottomRight => SctkResizeEdge::BottomRight,
                    _ => SctkResizeEdge::None,
                };
                window.resize(client_seat, serial.into(), edge);
            },
            FrameAction::Move => {
                window.move_(client_seat, serial.into());
            },
            _ => {
                warn!("Unknown frame action: {:?}", action);
            },
        }
        Ok(())
    }

    fn frame(&mut self) -> Option<&mut FallbackFrame<WprsState>> {
        Some(&mut self.window_frame)
    }

    fn handle_pointer_event_inner(
        &mut self,
        client_state: &mut WprsClientState,
        x11_surface: &X11Surface,
        _qh: &QueueHandle<WprsState>,
        pointer: &WlPointer,
        event: &PointerEvent,
    ) -> Result<Option<CursorIcon>> {
        let (x, y) = event.position;
        let frame = &mut self.window_frame;
        let mut new_cursor = None;
        match event.kind {
            PointerEventKind::Enter { serial } => {
                new_cursor = Some(
                    frame
                        .click_point_moved(Duration::ZERO, &event.surface.id(), x, y)
                        .unwrap_or(CursorIcon::Default),
                );
                client_state.last_enter_serial = serial;
            },
            PointerEventKind::Leave { serial: _ } => {
                frame.click_point_left();
            },
            PointerEventKind::Motion { time: _ } => {
                new_cursor = frame.click_point_moved(Duration::ZERO, &event.surface.id(), x, y);
            },
            PointerEventKind::Press { button, serial, .. }
            | PointerEventKind::Release { button, serial, .. } => {
                let kind = &event.kind;
                let pressed = matches!(event.kind, PointerEventKind::Press { .. });
                let click = match button {
                    BTN_LEFT => FrameClick::Normal,
                    BTN_RIGHT => FrameClick::Alternate,
                    _ => return Ok(None),
                };

                if let Some(action) = frame.on_click(Duration::ZERO, click, pressed) {
                    debug!("button: {click:?}, kind: {kind:?}, action {action:?}");

                    self.frame_action(x11_surface, pointer, serial.into(), action, (x, y))
                        .location(loc!())?;
                }
            },
            PointerEventKind::Axis { .. } => {},
        }

        Ok(new_cursor)
    }
}

impl FramedSurface for XWaylandSubSurface {
    fn frame_action(
        &mut self,
        x11_surface: &X11Surface,
        _client_pointer: &WlPointer,
        _serial: Serial,
        action: FrameAction,
        position: (f64, f64),
    ) -> Result<()> {
        match action {
            FrameAction::Close => {
                x11_surface.close().location(loc!())?;
            },
            FrameAction::Resize(edge) => {
                if !matches!(edge, ResizeEdge::None) {
                    self.move_active = false;
                    self.resize_state = Some(SubSurfaceResizeState {
                        edge,
                        start_geometry: x11_surface.geometry(),
                        start_pointer: position,
                    });
                }
            },
            FrameAction::Move => {
                self.resize_state = None;
                self.move_pointer_location = position;
                self.move_active = true;
            },
            _ => {},
        };

        Ok(())
    }

    fn frame(&mut self) -> Option<&mut FallbackFrame<WprsState>> {
        self.frame.as_mut()
    }

    fn handle_pointer_event_inner(
        &mut self,
        client_state: &mut WprsClientState,
        x11_surface: &X11Surface,
        qh: &QueueHandle<WprsState>,
        pointer: &WlPointer,
        event: &PointerEvent,
    ) -> Result<Option<CursorIcon>> {
        let Some(frame) = self.frame.as_mut() else {
            return Ok(None);
        };
        let mut new_cursor: Option<CursorIcon> = None;

        let (x, y) = event.position;
        match event.kind {
            PointerEventKind::Enter { serial } => {
                new_cursor = Some(
                    frame
                        .click_point_moved(Duration::ZERO, &event.surface.id(), x, y)
                        .unwrap_or(CursorIcon::Default),
                );
                client_state.last_enter_serial = serial;
            },
            PointerEventKind::Leave { serial: _ } => {
                self.move_active = false;
                self.resize_state = None;
                frame.click_point_left();
            },
            PointerEventKind::Motion { time: _ } => {
                new_cursor = frame.click_point_moved(Duration::ZERO, &event.surface.id(), x, y);

                if let Some(resize_state) = self.resize_state {
                    new_cursor = Some(resize_edge_cursor(resize_state.edge));

                    if !self.pending_frame_callback {
                        let cur_geo = x11_surface.geometry();
                        let new_geo = compute_subsurface_resize_geometry(
                            resize_state.start_geometry,
                            cur_geo,
                            resize_state.edge,
                            resize_state.start_pointer,
                            (x, y),
                            x11_surface.min_size().map(|s| (s.w, s.h)),
                            x11_surface.max_size().map(|s| (s.w, s.h)),
                        );

                        if new_geo != cur_geo {
                            if let Some(frame) = self.frame.as_mut()
                                && !frame.is_hidden()
                            {
                                let width = NonZeroU32::new(new_geo.size.w.max(1) as u32).unwrap();
                                let height = NonZeroU32::new(new_geo.size.h.max(1) as u32).unwrap();
                                frame.resize(width, height);
                                frame.draw();
                            }

                            self.move_(new_geo.loc.x, new_geo.loc.y, qh);
                            x11_surface.configure(new_geo).location(loc!())?;
                        }
                    }
                } else if self.move_active {
                    new_cursor = Some(CursorIcon::Move);

                    if !self.pending_frame_callback {
                        let (init_x, init_y) = self.move_pointer_location;
                        let offset_x = (x - init_x).round() as i32;
                        let offset_y = (y - init_y).round() as i32;

                        if (offset_x, offset_y) != (0, 0) {
                            let mut geo = x11_surface.geometry();
                            geo.loc.x += offset_x;
                            geo.loc.y += offset_y;

                            // frame() will cause the compositor to send us a frame callback on the next commit.
                            // So if we commit() the parent and call frame(), then we know that the compositor
                            // has processed our last set_position call, and therefore the next set of motion
                            // coordinates will be relative to it.  Without this, we won't know if the coordinates
                            // receive are relative to our current position or a previous position.
                            self.move_(geo.loc.x, geo.loc.y, qh);

                            x11_surface.configure(geo).location(loc!())?;
                        }
                    }
                }
            },
            PointerEventKind::Press { button, serial, .. }
            | PointerEventKind::Release { button, serial, .. } => {
                let kind = &event.kind;
                let pressed = matches!(event.kind, PointerEventKind::Press { .. });
                let click = match button {
                    BTN_LEFT => FrameClick::Normal,
                    BTN_RIGHT => FrameClick::Alternate,
                    _ => return Ok(None),
                };

                if !pressed && (self.move_active || self.resize_state.is_some()) {
                    self.move_active = false;
                    self.resize_state = None;
                    return Ok(None);
                }

                if let Some(action) = frame.on_click(Duration::ZERO, click, pressed) {
                    debug!("button: {click:?}, kind: {kind:?}, action {action:?}");

                    self.frame_action(x11_surface, pointer, serial.into(), action, event.position)
                        .location(loc!())?;
                } else {
                    self.move_active = false;
                    self.resize_state = None;
                }
            },
            PointerEventKind::Axis { .. } => {},
        }

        Ok(new_cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new((x, y).into(), (w, h).into())
    }

    /// Simulates what `FallbackFrame` border-local pointer coordinates the
    /// Wayland compositor delivers for a cursor displaced by `(dx, dy)` in
    /// screen space once the window is at `cur_geo`.
    fn simulated_border_pointer(
        start_geo: Rectangle<i32, Logical>,
        cur_geo: Rectangle<i32, Logical>,
        edge: ResizeEdge,
        start_pointer: (f64, f64),
        screen_delta: (f64, f64),
    ) -> (f64, f64) {
        let origin_shift_x = match edge {
            ResizeEdge::Right => {
                (cur_geo.loc.x + cur_geo.size.w) - (start_geo.loc.x + start_geo.size.w)
            },
            _ => cur_geo.loc.x - start_geo.loc.x,
        };
        let origin_shift_y = match edge {
            ResizeEdge::Bottom | ResizeEdge::BottomLeft | ResizeEdge::BottomRight => {
                (cur_geo.loc.y + cur_geo.size.h) - (start_geo.loc.y + start_geo.size.h)
            },
            _ => cur_geo.loc.y - start_geo.loc.y,
        };
        (
            start_pointer.0 + screen_delta.0 - f64::from(origin_shift_x),
            start_pointer.1 + screen_delta.1 - f64::from(origin_shift_y),
        )
    }

    #[test]
    fn test_resize_edge_cursor_all_edges() {
        assert_eq!(resize_edge_cursor(ResizeEdge::Top), CursorIcon::NResize);
        assert_eq!(
            resize_edge_cursor(ResizeEdge::TopRight),
            CursorIcon::NeResize
        );
        assert_eq!(resize_edge_cursor(ResizeEdge::Right), CursorIcon::EResize);
        assert_eq!(
            resize_edge_cursor(ResizeEdge::BottomRight),
            CursorIcon::SeResize
        );
        assert_eq!(resize_edge_cursor(ResizeEdge::Bottom), CursorIcon::SResize);
        assert_eq!(
            resize_edge_cursor(ResizeEdge::BottomLeft),
            CursorIcon::SwResize
        );
        assert_eq!(resize_edge_cursor(ResizeEdge::Left), CursorIcon::WResize);
        assert_eq!(
            resize_edge_cursor(ResizeEdge::TopLeft),
            CursorIcon::NwResize
        );
        assert_eq!(resize_edge_cursor(ResizeEdge::None), CursorIcon::Default);
    }

    #[test]
    fn test_subsurface_resize_geometry_all_edges_and_multi_step_drag() {
        let start = rect(100, 200, 300, 240);
        let min = Some((120, 100));
        let max = Some((450, 360));

        // Simulate a 4-step drag for every edge, including stationary hold and
        // reversal past clamping limits. `TopRight` and `BottomRight` are
        // critical because `TOP_BORDER`/`BOTTOM_BORDER` keep `pos.x = -BORDER`
        // while `width` changes.
        let cases: &[(
            ResizeEdge,
            (f64, f64),
            &[(f64, f64)],
            Rectangle<i32, Logical>,
        )] = &[
            (
                ResizeEdge::Right,
                (2.0, 50.0),
                &[(20.0, 10.0), (20.0, 10.0), (200.0, 0.0), (50.0, 0.0)],
                rect(100, 200, 350, 240),
            ),
            (
                ResizeEdge::Left,
                (2.0, 50.0),
                &[(-25.0, 15.0), (-25.0, 15.0), (250.0, 0.0), (40.0, 0.0)],
                rect(140, 200, 260, 240),
            ),
            (
                ResizeEdge::Bottom,
                (150.0, 2.0),
                &[(10.0, 30.0), (10.0, 30.0), (0.0, 200.0), (0.0, -40.0)],
                rect(100, 200, 300, 200),
            ),
            (
                ResizeEdge::Top,
                (150.0, 2.0),
                &[(10.0, -30.0), (10.0, -30.0), (0.0, 200.0), (0.0, 50.0)],
                rect(100, 250, 300, 190),
            ),
            (
                ResizeEdge::TopLeft,
                (2.0, 2.0),
                &[(-20.0, -15.0), (-20.0, -15.0), (250.0, 250.0), (30.0, 20.0)],
                rect(130, 220, 270, 220),
            ),
            (
                ResizeEdge::TopRight,
                (306.0, 2.0),
                &[(10.0, -5.0), (10.0, -5.0), (30.0, -5.0), (30.0, -5.0)],
                rect(100, 195, 330, 245),
            ),
            (
                ResizeEdge::BottomLeft,
                (2.0, 2.0),
                &[(-15.0, 25.0), (-15.0, 25.0), (50.0, -30.0), (50.0, -30.0)],
                rect(150, 200, 250, 210),
            ),
            (
                ResizeEdge::BottomRight,
                (306.0, 2.0),
                &[(15.0, 20.0), (15.0, 20.0), (40.0, 35.0), (40.0, 35.0)],
                rect(100, 200, 340, 275),
            ),
            (ResizeEdge::None, (10.0, 10.0), &[(50.0, 50.0)], start),
        ];

        for &(edge, start_ptr, steps, expected_final) in cases {
            let mut cur = start;
            for &screen_delta in steps {
                let ptr = simulated_border_pointer(start, cur, edge, start_ptr, screen_delta);
                let next =
                    compute_subsurface_resize_geometry(start, cur, edge, start_ptr, ptr, min, max);
                // Re-running with unchanged cursor after applying `next` must be idempotent.
                let ptr_after =
                    simulated_border_pointer(start, next, edge, start_ptr, screen_delta);
                let steady = compute_subsurface_resize_geometry(
                    start, next, edge, start_ptr, ptr_after, min, max,
                );
                assert_eq!(
                    steady, next,
                    "drift detected for edge {edge:?} at {screen_delta:?}"
                );
                cur = next;
            }
            assert_eq!(cur, expected_final, "final rect mismatch for edge {edge:?}");
        }

        // Clamping at `min_size` anchors the opposite edge (`start_x + start_w`
        // for Left/TopLeft, `start_y + start_h` for Top/TopLeft).
        let clamped_tl = compute_subsurface_resize_geometry(
            start,
            start,
            ResizeEdge::TopLeft,
            (2.0, 2.0),
            (500.0, 500.0),
            min,
            max,
        );
        assert_eq!(clamped_tl, rect(280, 340, 120, 100));
        assert_eq!(
            clamped_tl.loc.x + clamped_tl.size.w,
            start.loc.x + start.size.w
        );
        assert_eq!(
            clamped_tl.loc.y + clamped_tl.size.h,
            start.loc.y + start.size.h
        );

        // Invalid `max < min` and negative/zero bounds must not panic.
        let bad_hints = compute_subsurface_resize_geometry(
            start,
            start,
            ResizeEdge::BottomRight,
            (306.0, 2.0),
            (506.0, 102.0),
            Some((200, -5)),
            Some((100, 0)),
        );
        assert_eq!(bad_hints, rect(100, 200, 500, 340));
    }
}
