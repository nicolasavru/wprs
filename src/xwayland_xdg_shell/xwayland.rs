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

use std::fs::File;
use std::io;
use std::num::NonZeroU32;
use std::os::fd::OwnedFd;
use std::thread;

use smithay::reexports::wayland_server::Resource;
use smithay::utils::Logical;
use smithay::utils::Rectangle;
use smithay::utils::SERIAL_COUNTER;
use smithay::wayland::selection::SelectionTarget;
use smithay::xwayland::X11Surface;
use smithay::xwayland::X11Wm;
use smithay::xwayland::XwmHandler;
use smithay::xwayland::xwm::Reorder;
use smithay::xwayland::xwm::ResizeEdge as X11ResizeEdge;
use smithay::xwayland::xwm::WmWindowProperty;
use smithay::xwayland::xwm::XwmId;
use smithay_client_toolkit::globals::ProvidesBoundGlobal;
use smithay_client_toolkit::reexports::csd_frame::DecorationsFrame;
use smithay_client_toolkit::reexports::protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_toplevel::ResizeEdge as SctkResizeEdge;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::xdg::window::DecorationMode;

use crate::prelude::*;
use crate::xwayland_xdg_shell::WprsState;
use crate::xwayland_xdg_shell::client::Role;
use crate::xwayland_xdg_shell::compositor::popup_parent_origin;
use crate::xwayland_xdg_shell::xsurface_from_x11_surface;

pub(crate) fn x11_resize_edge_to_xdg(edge: X11ResizeEdge) -> SctkResizeEdge {
    match edge {
        X11ResizeEdge::Top => SctkResizeEdge::Top,
        X11ResizeEdge::Bottom => SctkResizeEdge::Bottom,
        X11ResizeEdge::Left => SctkResizeEdge::Left,
        X11ResizeEdge::TopLeft => SctkResizeEdge::TopLeft,
        X11ResizeEdge::BottomLeft => SctkResizeEdge::BottomLeft,
        X11ResizeEdge::Right => SctkResizeEdge::Right,
        X11ResizeEdge::TopRight => SctkResizeEdge::TopRight,
        X11ResizeEdge::BottomRight => SctkResizeEdge::BottomRight,
    }
}

impl XwmHandler for WprsState {
    fn xwm_state(&mut self, _xwm: XwmId) -> &mut X11Wm {
        self.compositor_state.xwm.as_mut().unwrap()
    }

    fn new_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn new_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn map_window_request(&mut self, _xwm: XwmId, window: X11Surface) {
        window.set_mapped(true).log_and_ignore(loc!());
    }

    fn mapped_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    #[instrument(skip(self, _xwm), level = "debug")]
    fn unmapped_window(&mut self, _xwm: XwmId, window: X11Surface) {
        let surface_id = window.wl_surface().map(|s| s.id()).or_else(|| {
            let win_id = window.window_id();
            self.surfaces.iter().find_map(|(id, xws)| {
                xws.x11_surface
                    .as_ref()
                    .is_some_and(|s| s.window_id() == win_id)
                    .then(|| id.clone())
            })
        });
        if let Some(surface_id) = surface_id {
            self.remove_surface(&surface_id);
        }

        // Without this, xwayland still thinks the key that triggered the
        // window close is still held down and sends key repeat events.
        if let Some(keyboard) = self.compositor_state.seat.get_keyboard()
            && let Some(focus) = keyboard.current_focus()
            && focus.window_id() == window.window_id()
        {
            let serial = SERIAL_COUNTER.next_serial();
            keyboard.set_focus(self, None, serial);
            self.release_all_pressed_keys().log_and_ignore(loc!());
        }

        if !window.is_override_redirect() && window.alive() {
            window.set_mapped(false).log_and_ignore(loc!());
        }
    }

    fn destroyed_window(&mut self, xwm: XwmId, window: X11Surface) {
        self.unmapped_window(xwm, window);
    }

    fn configure_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        x: Option<i32>,
        y: Option<i32>,
        w: Option<u32>,
        h: Option<u32>,
        _reorder: Option<Reorder>,
    ) {
        let mut geo = window.geometry();

        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(role) = &xwayland_surface.role {
                // toplevel windows are not allowed to move themselves
                if !matches!(role, Role::XdgToplevel { .. }) {
                    if let Some(x) = x
                        && x != 0
                    {
                        geo.loc.x = x;
                    }
                    if let Some(y) = y
                        && y != 0
                    {
                        geo.loc.y = y;
                    }
                }

                // toplevels are also not allowed to resize themselves after they are configured,
                // or else their frame will desync with their surface
                if !matches!(role, Role::XdgToplevel(toplevel) if toplevel.configured) {
                    if let Some(w) = w
                        && w > 1
                    {
                        geo.size.w = w as i32;
                    }
                    if let Some(h) = h
                        && h > 1
                    {
                        geo.size.h = h as i32;
                    }
                }
            }
        } else {
            // we don't know when we will be assigned a role, so accept any requested
            // dimensions.
            if let Some(x) = x
                && x != 0
            {
                geo.loc.x = x;
            }
            if let Some(y) = y
                && y != 0
            {
                geo.loc.y = y;
            }
            if let Some(w) = w
                && w > 1
            {
                geo.size.w = w as i32;
            }
            if let Some(h) = h
                && h > 1
            {
                geo.size.h = h as i32;
            }
        }

        if window.is_mapped() {
            // Under Wayland, windows don't get to resize themselves. Many X apps
            // need a synthetic configure reply though. Additionally, some broken
            // toolkits (read: Java) will still render the window at the size they
            // asked for, even if the request wasn't granted, and also ignore
            // ConfigureNotify events where the size equals the current size, so
            // trigger a redraw by resizing the window by a small amount and then
            // resizing it back to the original size.
            let mut hack_geo = geo;
            hack_geo.size.w -= 1;

            window.configure(hack_geo).unwrap();
            window.configure(geo).unwrap();
        } else {
            window.configure(geo).unwrap();
        }
    }

    fn configure_notify(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        geometry: Rectangle<i32, Logical>,
        _above: Option<u32>,
    ) {
        let Some(surface_id) = self.surfaces.iter().find_map(|(id, xws)| {
            xws.x11_surface
                .as_ref()
                .is_some_and(|s| s == &window)
                .then(|| id.clone())
        }) else {
            return;
        };

        let parent_origin = self
            .surfaces
            .get(&surface_id)
            .and_then(|xws| match &xws.role {
                Some(Role::XdgPopup(popup)) => popup_parent_origin(&self.surfaces, &popup.parent),
                Some(Role::SubSurface(_)) => xws
                    .parent_id
                    .as_ref()
                    .and_then(|pid| self.surfaces.get(pid))
                    .and_then(|p| p.get_x11_surface().ok())
                    .map(|s| (s.geometry().loc.into(), (0, 0).into())),
                _ => None,
            });

        let Some(xwayland_surface) = self.surfaces.get_mut(&surface_id) else {
            return;
        };
        match &mut xwayland_surface.role {
            Some(Role::SubSurface(subsurface))
                if !subsurface.move_active && subsurface.resize_state.is_none() =>
            {
                if let Some((parent_x11_loc, _)) = parent_origin {
                    subsurface.offset = (-parent_x11_loc.x, -parent_x11_loc.y).into();
                }
                if let Some(frame) = subsurface.frame.as_mut()
                    && !frame.is_hidden()
                {
                    let width = NonZeroU32::new(geometry.size.w.max(1) as u32).unwrap();
                    let height = NonZeroU32::new(geometry.size.h.max(1) as u32).unwrap();
                    frame.resize(width, height);
                    frame.draw();
                }
                subsurface.move_(geometry.loc.x, geometry.loc.y, &self.client_state.qh);
            },
            Some(Role::XdgPopup(popup)) => {
                if let Some((parent_x11_loc, parent_frame_offset)) = parent_origin {
                    popup
                        .reposition(
                            &self.client_state.xdg_shell_state,
                            geometry,
                            parent_x11_loc,
                            parent_frame_offset,
                        )
                        .log_and_ignore(loc!());
                }
            },
            _ => {},
        }
    }

    // For maximize and fullscreeen: send the appropriate request to the wayland
    // compositor we're running in and let the subsequent configure trigger the
    // appropriate X11 configures.
    //
    // For unmaximize and unfullscreen: the wayland compositor we're running in
    // will follow up with a configure with the geometry to use, so we don't
    // need to worry about that saving the old geometry and restoring it here.

    fn maximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                toplevel.local_window.set_maximized();
            } else {
                warn!("Received maximize request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received maximize request for unknown surface.");
        }
    }

    fn unmaximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                toplevel.local_window.unset_maximized();
            } else {
                warn!("Received unmaximize request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received unmaximize request for unknown surface.");
        }
    }

    fn fullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &mut xwayland_surface.role {
                toplevel.local_window.set_fullscreen(None);
            } else {
                warn!("Received fullscreen request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received fullscreen request for unknown surface.");
        }
    }

    fn unfullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &mut xwayland_surface.role {
                toplevel.local_window.unset_fullscreen();
            } else {
                warn!("Received unfullscreen request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received unfullscreen request for unknown surface.");
        }
    }

    fn minimize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                toplevel.local_window.set_minimized();
            } else {
                warn!("Received minimize request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received minimize request for unknown surface.");
        }
    }

    fn resize_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        _button: u32,
        edges: X11ResizeEdge,
    ) {
        let Some((seat, serial)) = self.client_state.interactive_grab_seat_and_serial() else {
            warn!("Received resize request without an active seat/serial.");
            return;
        };
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                toplevel
                    .local_window
                    .resize(&seat, serial, x11_resize_edge_to_xdg(edges));
            } else {
                warn!("Received resize request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received resize request for unknown surface.");
        }
    }

    fn move_request(&mut self, _xwm: XwmId, window: X11Surface, _button: u32) {
        let Some((seat, serial)) = self.client_state.interactive_grab_seat_and_serial() else {
            warn!("Received move request without an active seat/serial.");
            return;
        };
        if let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) {
            if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                toplevel.local_window.move_(&seat, serial);
            } else {
                warn!("Received move request for non-XdgToplevel surface.");
            }
        } else {
            warn!("Received move request for unknown surface.");
        }
    }

    #[instrument(skip(self, _xwm), level = "debug")]
    fn allow_selection_access(&mut self, _xwm: XwmId, selection: SelectionTarget) -> bool {
        match selection {
            SelectionTarget::Clipboard => self.client_state.selection_offer.is_some(),
            SelectionTarget::Primary => self.client_state.primary_selection_offer.is_some(),
        }
    }

    #[instrument(skip(self, _xwm), level = "debug")]
    fn send_selection(
        &mut self,
        _xwm: XwmId,
        selection: SelectionTarget,
        mime_type: String,
        fd: OwnedFd,
    ) {
        let read_pipe = match selection {
            SelectionTarget::Primary => {
                let Some(cur_offer) = self.client_state.primary_selection_offer.clone() else {
                    warn!("primary_selection_offer was empty");
                    return;
                };

                cur_offer.receive(mime_type.clone()).ok()
            },
            SelectionTarget::Clipboard => {
                let Some(cur_offer) = self.client_state.selection_offer.clone() else {
                    warn!("selection_offer was empty");
                    return;
                };
                cur_offer.receive(mime_type.clone()).ok()
            },
        };

        if let Some(mut read_pipe) = read_pipe {
            debug!("spawning send_selection thread for mime {mime_type}");
            thread::spawn(move || {
                debug!("in send_selection thread for mime {mime_type}");
                let mut f = File::from(fd);

                // NOTE: this block is useful debugging.
                // let mut buf = Vec::new();
                // let bytes_copied = read_pipe.read_to_end(&mut buf).unwrap();
                // debug!("read selection: {buf:?}");
                // f.write_all(&buf);

                let bytes_copied = io::copy(&mut read_pipe, &mut f);
                debug!("wrote selection: {bytes_copied:?} bytes");
            });
        }
    }

    #[instrument(skip(self, _xwm), level = "debug")]
    fn new_selection(
        &mut self,
        _xwm: XwmId,
        selection: SelectionTarget,
        mut mime_types: Vec<String>,
    ) {
        if let Some(seat_obj) = self.client_state.seat_objects.last() {
            mime_types.push("_xwayland_xdg_shell_marker".to_owned());

            match selection {
                SelectionTarget::Clipboard => {
                    self.client_state.selection_offer = None;
                    let source = self
                        .client_state
                        .data_device_manager_state
                        .create_copy_paste_source(
                            &self.client_state.qh,
                            mime_types.iter().map(String::as_str),
                        );

                    source.set_selection(
                        &seat_obj.data_device,
                        self.client_state.last_implicit_grab_serial,
                    );

                    self.client_state.selection_source = Some(source);
                },
                SelectionTarget::Primary => {
                    self.client_state.primary_selection_offer = None;
                    if let (Some(primary_selection_manager_state), Some(primary_selection_device)) = (
                        &self.client_state.primary_selection_manager_state,
                        &seat_obj.primary_selection_device,
                    ) {
                        let source = primary_selection_manager_state.create_selection_source(
                            &self.client_state.qh,
                            mime_types.iter().map(String::as_str),
                        );

                        source.set_selection(
                            primary_selection_device,
                            self.client_state.last_implicit_grab_serial,
                        );

                        self.client_state.primary_selection_source = Some(source);
                    }
                },
            };
        }
    }

    #[instrument(skip(self, _xwm), level = "debug")]
    fn cleared_selection(&mut self, _xwm: XwmId, selection: SelectionTarget) {
        let serial = self.client_state.last_implicit_grab_serial;
        match selection {
            SelectionTarget::Clipboard => {
                if let Some(source) = self.client_state.selection_source.take() {
                    if let Some(seat_obj) = self.client_state.seat_objects.last() {
                        seat_obj.data_device.unset_selection(serial);
                    }
                    drop(source);
                }
            },
            SelectionTarget::Primary => {
                if let Some(source) = self.client_state.primary_selection_source.take() {
                    if let Some(seat_obj) = self.client_state.seat_objects.last()
                        && let Some(primary_selection_device) = &seat_obj.primary_selection_device
                    {
                        primary_selection_device.unset_selection(serial);
                    }
                    drop(source);
                }
            },
        }
    }

    fn property_notify(&mut self, _xwm: XwmId, window: X11Surface, property: WmWindowProperty) {
        let has_xdg_decoration =
            <XdgShell as ProvidesBoundGlobal<ZxdgDecorationManagerV1, 1>>::bound_global(
                &self.client_state.xdg_shell_state,
            )
            .is_ok();
        let Some(xwayland_surface) = xsurface_from_x11_surface(&mut self.surfaces, &window) else {
            return;
        };

        match property {
            WmWindowProperty::Title => {
                if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                    toplevel.local_window.set_title(window.title());
                }
            },
            WmWindowProperty::Class => {
                if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                    toplevel.local_window.set_app_id(window.class());
                }
            },
            WmWindowProperty::NormalHints => {
                if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role {
                    toplevel.sync_size_hints(&window);
                }
            },
            WmWindowProperty::MotifHints => match &mut xwayland_surface.role {
                Some(Role::XdgToplevel(toplevel)) => {
                    if !toplevel.sync_decoration_mode(window.is_decorated()) {
                        return;
                    }
                    if !toplevel.configured {
                        return;
                    }
                    let host_in_client_mode = toplevel
                        .last_configure
                        .as_ref()
                        .is_some_and(|c| c.decoration_mode == DecorationMode::Client);
                    if host_in_client_mode
                        && (!has_xdg_decoration
                            || toplevel.requested_decoration_mode == DecorationMode::Client)
                    {
                        let last_configure = toplevel.last_configure.clone();
                        let cur_loc = window.geometry().loc;
                        toplevel
                            .apply_decoration(
                                &window,
                                last_configure.as_ref(),
                                xwayland_surface.buffer.as_ref().map(|b| &b.metadata),
                                |_, _| cur_loc.into(),
                            )
                            .log_and_ignore(loc!());
                        if !toplevel.window_frame.is_hidden() && toplevel.window_frame.is_dirty() {
                            toplevel.window_frame.draw();
                        }
                    }
                    toplevel.local_window.wl_surface().commit();
                },
                Some(Role::SubSurface(subsurface)) => {
                    subsurface
                        .update_decorations(
                            &window,
                            &self.client_state.shm_state,
                            self.client_state.subcompositor_state.clone(),
                            &self.client_state.qh,
                        )
                        .log_and_ignore(loc!());
                },
                _ => {},
            },
            _ => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x11_resize_edge_to_xdg_all_edges() {
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::Top),
            SctkResizeEdge::Top
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::Bottom),
            SctkResizeEdge::Bottom
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::Left),
            SctkResizeEdge::Left
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::Right),
            SctkResizeEdge::Right
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::TopLeft),
            SctkResizeEdge::TopLeft
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::TopRight),
            SctkResizeEdge::TopRight
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::BottomLeft),
            SctkResizeEdge::BottomLeft
        );
        assert_eq!(
            x11_resize_edge_to_xdg(X11ResizeEdge::BottomRight),
            SctkResizeEdge::BottomRight
        );
    }
}
