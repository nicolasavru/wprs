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

use std::collections::HashMap;

use smithay_client_toolkit::output::OutputState;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_positioner;
use smithay_client_toolkit::shell::xdg;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::xdg::XdgSurface;
use smithay_client_toolkit::shell::xdg::popup;
use smithay_client_toolkit::shell::xdg::window::Window;
use smithay_client_toolkit::shell::xdg::window::WindowDecorations;

use crate::client::RemoteSurface;
use crate::client::Role;
use crate::client::WprsClientState;
use crate::client::find_output;
use crate::prelude::*;
use crate::serialization::geometry::Size;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::xdg_shell::DecorationMode;
use crate::serialization::xdg_shell::FullscreenState;
use crate::serialization::xdg_shell::XdgPositioner;

#[derive(Debug)]
pub struct RemoteXdgToplevel {
    pub local_window: Window,
    // TODO: add configured field to Window, have it be set before dispatching
    // first configure;
    pub configured: bool,
    pub parent: Option<WlSurfaceId>,
    pub title: Option<String>,
    pub title_prefix: String,
    pub app_id: Option<String>,
    pub decoration_mode: Option<DecorationMode>,
    pub max_size: Size<i32>,
    pub min_size: Size<i32>,
}

impl RemoteXdgToplevel {
    pub fn set_role(
        surface_state: &SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        xdg_shell_state: &XdgShell,
        output_state: &OutputState,
        qh: &QueueHandle<WprsClientState>,
        title_prefix: &str,
    ) -> Result<()> {
        {
            let surface = surfaces.get(&surface_id).location(loc!())?;
            if matches!(surface.role, Some(Role::XdgToplevel(_))) {
                return Ok(());
            }
            if surface.role.is_some() {
                bail!("surface {surface_id:?} already has a non-toplevel role");
            }
        }
        let toplevel_state = surface_state.xdg_toplevel().location(loc!())?;
        let desired_parent = toplevel_state.parent.filter(|&id| id != surface_id);
        let parent_window = desired_parent.and_then(|id| {
            surfaces
                .get(&id)
                .and_then(|s| s.xdg_toplevel().ok())
                .map(|t| t.local_window.clone())
        });
        let applied_parent = parent_window.as_ref().and(desired_parent);

        let local_surface = surfaces
            .get_mut(&surface_id)
            .location(loc!())?
            .local_surface
            .take()
            .location(loc!())?;

        let local_window =
            xdg_shell_state.create_window(local_surface, WindowDecorations::ServerDefault, qh);

        if let Some(parent_window) = &parent_window {
            local_window.set_parent(Some(parent_window));
        }

        if let Some(maximized) = toplevel_state.maximized {
            if maximized {
                local_window.set_maximized();
            } else {
                local_window.unset_maximized();
            }
        }

        match toplevel_state.fullscreen {
            Some(FullscreenState::Set(output_id)) => {
                let output = output_id.and_then(|id| find_output(output_state, id));
                local_window.set_fullscreen(output.as_ref());
            },
            Some(FullscreenState::Unset) => {
                local_window.unset_fullscreen();
            },
            None => {},
        }

        let new_toplevel = Self {
            local_window,
            configured: false,
            parent: applied_parent,
            title: None,
            title_prefix: title_prefix.to_owned(),
            app_id: None,
            decoration_mode: None,
            max_size: (0, 0).into(),
            min_size: (0, 0).into(),
        };

        let surface = surfaces.get_mut(&surface_id).location(loc!())?;
        surface.role = Some(Role::XdgToplevel(new_toplevel));
        Ok(())
    }

    fn set_title(&mut self, title: Option<String>) {
        if self.title != title {
            self.title = title;
            if let Some(title) = &self.title {
                self.local_window
                    .set_title(format!("{}{}", self.title_prefix, title));
            }
        }
    }

    fn set_app_id(&mut self, app_id: Option<String>) {
        if self.app_id != app_id {
            self.app_id = app_id;
            if let Some(app_id) = &self.app_id {
                self.local_window.set_app_id(app_id);
            }
        }
    }

    fn set_decoration_mode(&mut self, mode: Option<DecorationMode>) {
        if self.decoration_mode != mode {
            self.decoration_mode = mode;
            self.local_window
                .request_decoration_mode(mode.map(Into::into));
        }
    }

    fn set_max_size(&mut self, max_size: Size<i32>) {
        if self.max_size != max_size {
            self.max_size = max_size;
            self.local_window
                .set_max_size(Some((self.max_size.w as u32, self.max_size.h as u32)));
        }
    }

    fn set_min_size(&mut self, min_size: Size<i32>) {
        if self.min_size != min_size {
            self.min_size = min_size;
            self.local_window
                .set_min_size(Some((self.min_size.w as u32, self.min_size.h as u32)));
        }
    }

    pub fn update(
        surface_state: SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
    ) -> Result<()> {
        let toplevel_state = surface_state
            .role
            .location(loc!())?
            .into_xdg_toplevel()
            // The error type is the enum. :(
            .map_err(|_| anyhow!("role wasn't xdg toplevel"))
            .location(loc!())?;

        let desired_parent = toplevel_state.parent.filter(|&id| id != surface_id);
        let current_parent = surfaces
            .get(&surface_id)
            .location(loc!())?
            .xdg_toplevel()
            .location(loc!())?
            .parent;
        let parent_update = if current_parent != desired_parent {
            let parent_window = desired_parent.and_then(|id| {
                surfaces
                    .get(&id)
                    .and_then(|s| s.xdg_toplevel().ok())
                    .map(|t| t.local_window.clone())
            });
            (desired_parent.is_none() || parent_window.is_some())
                .then_some((parent_window.as_ref().and(desired_parent), parent_window))
        } else {
            None
        };

        let remote_toplevel = surfaces
            .get_mut(&surface_id)
            .location(loc!())?
            .role
            .as_mut()
            .location(loc!())?
            .as_xdg_toplevel_mut()
            .location(loc!())?;

        if let Some((applied_parent, parent_window)) = parent_update {
            remote_toplevel
                .local_window
                .set_parent(parent_window.as_ref());
            remote_toplevel.parent = applied_parent;
        }

        if let Some(xdg_surface_state) = &surface_state.xdg_surface_state {
            if let Some(window_geometry) = xdg_surface_state.window_geometry {
                remote_toplevel.set_window_geometry(
                    window_geometry.loc.x as u32,
                    window_geometry.loc.y as u32,
                    window_geometry.size.w as u32,
                    window_geometry.size.h as u32,
                );
            }
            remote_toplevel.set_max_size(xdg_surface_state.max_size);
            remote_toplevel.set_min_size(xdg_surface_state.min_size);
        }

        remote_toplevel.set_title(toplevel_state.title);
        remote_toplevel.set_app_id(toplevel_state.app_id);
        remote_toplevel.set_decoration_mode(toplevel_state.decoration_mode);

        Ok(())
    }

    pub fn apply(
        surface_state: SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        xdg_shell_state: &XdgShell,
        output_state: &OutputState,
        qh: &QueueHandle<WprsClientState>,
        title_prefix: &str,
    ) -> Result<()> {
        Self::set_role(
            &surface_state,
            surface_id,
            surfaces,
            xdg_shell_state,
            output_state,
            qh,
            title_prefix,
        )
        .location(loc!())?;
        Self::update(surface_state, surface_id, surfaces)
    }
}

#[derive(Debug)]
pub struct RemoteXdgPopup {
    pub local_popup: popup::Popup,
    // TODO: add configured field to Popup, have it be set before dispatching
    // first configure;
    pub configured: bool,
    pub positioner: XdgPositioner,
    pub last_reposition_token: Option<u32>,
}

impl RemoteXdgPopup {
    pub fn new_positioner(
        xdg_shell_state: &XdgShell,
        positioner: &XdgPositioner,
    ) -> Result<xdg::XdgPositioner> {
        let new_positioner = xdg::XdgPositioner::new(xdg_shell_state).location(loc!())?;
        new_positioner.set_size(positioner.width, positioner.height);
        new_positioner.set_anchor_rect(
            positioner.anchor_rect.loc.x,
            positioner.anchor_rect.loc.y,
            positioner.anchor_rect.size.w,
            positioner.anchor_rect.size.h,
        );
        new_positioner.set_anchor(
            xdg_positioner::Anchor::try_from(positioner.anchor_edges)
                // The error type is (). :(
                .map_err(|_| anyhow!("invalid anchor"))
                .location(loc!())?,
        );
        new_positioner.set_gravity(
            xdg_positioner::Gravity::try_from(positioner.gravity)
                // The error type is (). :(
                .map_err(|_| anyhow!("invalid anchor"))
                .location(loc!())?,
        );
        new_positioner.set_constraint_adjustment(
            xdg_positioner::ConstraintAdjustment::from_bits_retain(
                positioner.constraint_adjustment,
            ),
        );
        new_positioner.set_offset(positioner.offset.x, positioner.offset.y);
        if positioner.reactive {
            new_positioner.set_reactive();
        }
        if let Some(parent_size) = &positioner.parent_size {
            new_positioner.set_parent_size(parent_size.w, parent_size.h);
        };
        if let Some(parent_configure) = positioner.parent_configure {
            new_positioner.set_parent_configure(parent_configure);
        };
        Ok(new_positioner)
    }

    pub fn set_role(
        surface_state: &SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        xdg_shell_state: &XdgShell,
        seat: Option<&WlSeat>,
        qh: &QueueHandle<WprsClientState>,
    ) -> Result<()> {
        {
            let surface = surfaces.get(&surface_id).location(loc!())?;
            if matches!(surface.role, Some(Role::XdgPopup(_))) {
                return Ok(());
            }
            if surface.role.is_some() {
                bail!("surface {surface_id:?} already has a non-popup role");
            }
        }
        let popup_state = surface_state.xdg_popup().location(loc!())?;

        let parent = surfaces
            .get(&popup_state.parent_surface_id)
            .location(loc!())?
            .xdg_surface();

        let positioner =
            Self::new_positioner(xdg_shell_state, &popup_state.positioner).location(loc!())?;

        let local_surface = surfaces
            .get_mut(&surface_id)
            .location(loc!())?
            .local_surface
            .take()
            .location(loc!())?;

        let local_popup = popup::Popup::from_surface(
            parent.as_ref(),
            &positioner,
            qh,
            local_surface,
            xdg_shell_state,
        )
        .location(loc!())?;

        if let (Some(grab_serial), Some(seat)) = (popup_state.grab_serial, seat) {
            local_popup.xdg_popup().grab(seat, grab_serial);
        }

        let new_popup = Self {
            local_popup,
            configured: false,
            positioner: popup_state.positioner,
            last_reposition_token: popup_state.reposition_token,
        };
        let surface = surfaces.get_mut(&surface_id).location(loc!())?;
        surface.role = Some(Role::XdgPopup(new_popup));
        Ok(())
    }

    pub fn update(
        surface_state: SurfaceState,
        surface: &mut RemoteSurface,
        xdg_shell_state: &XdgShell,
    ) -> Result<()> {
        let remote_popup = surface.xdg_popup_mut().location(loc!())?;
        // TODO: why isn't this always set?
        // let xdg_surface_state = surface_state.xdg_surface_state.as_ref().location(loc!())?;
        if let Some(xdg_surface_state) = surface_state.xdg_surface_state
            && let Some(window_geometry) = xdg_surface_state.window_geometry
        {
            remote_popup.set_window_geometry(
                window_geometry.loc.x as u32,
                window_geometry.loc.y as u32,
                window_geometry.size.w as u32,
                window_geometry.size.h as u32,
            );
        }

        let popup_state = surface_state.xdg_popup().location(loc!())?;
        let reposition_changed = remote_popup.positioner != popup_state.positioner
            || (popup_state.reposition_token.is_some()
                && remote_popup.last_reposition_token != popup_state.reposition_token);
        if reposition_changed {
            let positioner =
                Self::new_positioner(xdg_shell_state, &popup_state.positioner).location(loc!())?;
            remote_popup
                .local_popup
                .reposition(&positioner, popup_state.reposition_token.unwrap_or(0));
            remote_popup.positioner = popup_state.positioner;
            remote_popup.last_reposition_token = popup_state.reposition_token;
        }

        Ok(())
    }

    pub fn apply(
        surface_state: SurfaceState,
        surface_id: WlSurfaceId,
        surfaces: &mut HashMap<WlSurfaceId, RemoteSurface>,
        xdg_shell_state: &XdgShell,
        seat: Option<&WlSeat>,
        qh: &QueueHandle<WprsClientState>,
    ) -> Result<()> {
        Self::set_role(
            &surface_state,
            surface_id,
            surfaces,
            xdg_shell_state,
            seat,
            qh,
        )
        .location(loc!())?;
        let surface = surfaces.get_mut(&surface_id).location(loc!())?;
        Self::update(surface_state, surface, xdg_shell_state)
    }
}
