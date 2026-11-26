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
use std::collections::HashSet;
use std::ffi::OsStr;
use std::sync::Arc;

use bimap::BiMap;
use calloop::RegistrationToken;
use smithay::backend::input::KeyState as SmithayKeyState;
use smithay::input::keyboard::FilterResult;
use smithay::input::keyboard::KeysymHandle;
use smithay::input::keyboard::ModifiersState;
use smithay::output::Output;
use smithay::reexports::calloop::LoopHandle;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::backend::ObjectId as CompositorObjectId;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface as CompositorWlSurface;
use smithay::utils::Rectangle;
use smithay::utils::Serial;
use smithay::xwayland::X11Surface;
use smithay::xwayland::xwm::WmWindowType;
use smithay_client_toolkit::compositor::CompositorState;
use smithay_client_toolkit::compositor::Surface;
use smithay_client_toolkit::compositor::SurfaceData;
use smithay_client_toolkit::output::OutputData;
use smithay_client_toolkit::reexports::client::Connection;
use smithay_client_toolkit::reexports::client::Proxy;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::client::backend::ObjectId as ClientObjectId;
use smithay_client_toolkit::reexports::client::globals::GlobalList;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Transform as SctkTransform;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shm::Shm;
use smithay_client_toolkit::subcompositor::SubcompositorState;
use tracing::Span;

use crate::args;
use crate::client_utils::FrameDamage;
use crate::compositor_utils;
use crate::prelude::*;
use crate::serialization::wayland::KeyState;
use crate::xwayland_xdg_shell::client::XWaylandSubSurface;

pub mod client;
pub mod compositor;
pub mod decoration;
pub mod wmname;
pub mod xwayland;

use client::Role;
use client::WprsClientState;
use client::XWaylandBuffer;
use client::XWaylandXdgPopup;
use client::XWaylandXdgToplevel;
use compositor::DecorationBehavior;
use compositor::WprsCompositorState;
use compositor::X11Parent;
use compositor::XwaylandOptions;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum WaylandWindowType {
    Toplevel,
    Popup,
    SubSurface,
}

/// Classifies an X11 window into its target Wayland shell role.
///
/// Override-redirect windows and EWMH popup window types always map to
/// `Popup` even when `WM_TRANSIENT_FOR` is set, so context menus and dropdowns
/// receive `xdg_positioner` constraint adjustments instead of being downgraded
/// to subsurfaces.
pub(crate) fn classify_wayland_window_type(
    window_type: Option<WmWindowType>,
    is_override_redirect: bool,
    has_explicit_transient_parent: bool,
) -> WaylandWindowType {
    if is_override_redirect {
        return WaylandWindowType::Popup;
    }

    // https://specifications.freedesktop.org/wm-spec/wm-spec-latest.html#idm45317634120064
    let resolved_type = window_type.unwrap_or(if has_explicit_transient_parent {
        WmWindowType::Dialog
    } else {
        WmWindowType::Normal
    });

    match resolved_type {
        WmWindowType::DropdownMenu
        | WmWindowType::Menu
        | WmWindowType::Notification
        | WmWindowType::PopupMenu
        | WmWindowType::Toolbar
        | WmWindowType::Tooltip => WaylandWindowType::Popup,
        WmWindowType::Dialog
        | WmWindowType::Normal
        | WmWindowType::Splash
        | WmWindowType::Utility => {
            if has_explicit_transient_parent {
                WaylandWindowType::SubSurface
            } else {
                WaylandWindowType::Toplevel
            }
        },
    }
}

#[derive(Debug)]
pub struct XWaylandSurface {
    pub(crate) x11_surface: Option<X11Surface>,
    pub(crate) buffer: Option<XWaylandBuffer>,
    pub(crate) buffer_attached: bool,
    pub(crate) buffer_scale: i32,
    pub(crate) buffer_transform: SctkTransform,
    // None when the surface is owned by a role object (e.g., a Window).
    pub(crate) local_surface: Option<Surface>,
    pub(crate) role: Option<Role>,
    pub(crate) parent_id: Option<CompositorObjectId>,
    pub(crate) children: HashSet<CompositorObjectId>,
    pub(crate) output_ids: HashSet<u32>,
    pub(crate) damage: FrameDamage,
}

impl Default for XWaylandSurface {
    fn default() -> Self {
        Self {
            x11_surface: None,
            buffer: None,
            buffer_attached: false,
            buffer_scale: 1,
            buffer_transform: SctkTransform::Normal,
            local_surface: None,
            role: None,
            parent_id: None,
            children: HashSet::new(),
            output_ids: HashSet::new(),
            damage: FrameDamage::Empty,
        }
    }
}

impl XWaylandSurface {
    pub fn get_x11_surface(&self) -> Result<&X11Surface> {
        self.x11_surface
            .as_ref()
            .ok_or(anyhow!("x11_surface was None"))
    }

    pub fn new(
        compositor_wl_surface: &CompositorWlSurface,
        compositor_state: &CompositorState,
        qh: &QueueHandle<WprsState>,
        surface_bimap: &mut BiMap<CompositorObjectId, ClientObjectId>,
    ) -> Result<Self> {
        let local_surface = Surface::new(compositor_state, qh).location(loc!())?;
        surface_bimap.insert(compositor_wl_surface.id(), local_surface.wl_surface().id());

        Ok(Self {
            local_surface: Some(local_surface),
            ..Self::default()
        })
    }

    pub(crate) fn has_client_surface(&self) -> bool {
        self.local_surface.is_some()
            || matches!(
                self.role,
                Some(Role::XdgToplevel(_) | Role::XdgPopup(_) | Role::SubSurface(_))
            )
    }

    pub(crate) fn sync_buffer_scale_and_transform(
        &mut self,
        buffer_scale: i32,
        buffer_transform: SctkTransform,
    ) {
        let buffer_scale = buffer_scale.max(1);
        let scale_changed = self.buffer_scale != buffer_scale;
        let transform_changed = self.buffer_transform != buffer_transform;
        self.buffer_scale = buffer_scale;
        self.buffer_transform = buffer_transform;

        if !self.has_client_surface() {
            return;
        }
        let wl_surface = self.wl_surface();
        if scale_changed {
            wl_surface.set_buffer_scale(buffer_scale);
        }
        if transform_changed {
            wl_surface.set_buffer_transform(buffer_transform);
        }
    }

    fn update_local_surface(
        &mut self,
        compositor_wl_surface: &CompositorWlSurface,
        parent: Option<&ClientWlSurface>,
        compositor_state: &CompositorState,
        qh: &QueueHandle<WprsState>,
        surface_bimap: &mut BiMap<CompositorObjectId, ClientObjectId>,
    ) -> Result<()> {
        let local_surface =
            Surface::with_data(compositor_state, qh, parent.cloned(), 1, ()).location(loc!())?;
        if self.buffer_scale != 1 {
            local_surface
                .wl_surface()
                .set_buffer_scale(self.buffer_scale);
        }
        if self.buffer_transform != SctkTransform::Normal {
            local_surface
                .wl_surface()
                .set_buffer_transform(self.buffer_transform);
        }

        surface_bimap.insert(compositor_wl_surface.id(), local_surface.wl_surface().id());
        self.local_surface = Some(local_surface);

        Ok(())
    }

    fn ready(&self) -> bool {
        match &self.role {
            Some(Role::XdgToplevel(toplevel)) if !toplevel.configured => false,
            Some(Role::XdgPopup(popup)) if !popup.configured => false,
            _ => self.x11_surface.is_some() || matches!(self.role, Some(Role::Cursor)),
        }
    }

    fn needs_configure(&self) -> bool {
        match &self.role {
            Some(Role::XdgToplevel(toplevel)) if !toplevel.configured => true,
            Some(Role::XdgPopup(popup)) if !popup.configured => true,
            _ => false,
        }
    }

    fn try_draw_buffer(&mut self) {
        if !self.buffer_attached
            && let Some(buffer) = &self.buffer
        {
            let surface = self.wl_surface().clone();
            // The only possible error here is AlreadyActive, which we can
            // ignore.
            _ = buffer.active_buffer.attach_to(&surface);
            self.damage.apply_and_clear(&surface);

            self.buffer_attached = true;
        }
    }

    fn commit_buffer(&mut self, qh: &QueueHandle<WprsState>) {
        if !self.buffer_attached && self.buffer.is_some() {
            self.try_draw_buffer();

            self.frame(qh);
            self.commit();
        }
    }

    #[instrument(skip(xdg_shell_state, qh), level = "debug")]
    fn update_x11_surface(
        &mut self,
        x11_surface: X11Surface,
        wayland_window_type: WaylandWindowType,
        parent: Option<X11Parent>,
        xdg_shell_state: &XdgShell,
        shm_state: &Shm,
        subcompositor_state: Arc<SubcompositorState>,
        qh: &QueueHandle<WprsState>,
        decoration_behavior: DecorationBehavior,
    ) -> Result<()> {
        self.x11_surface = Some(x11_surface);
        if self.role.is_some() {
            return Ok(());
        }

        match wayland_window_type {
            WaylandWindowType::Toplevel => {
                debug!("creating xdg_toplevel for {self:?}");
                self.parent_id = None;
                XWaylandXdgToplevel::set_role(
                    self,
                    xdg_shell_state,
                    shm_state,
                    subcompositor_state,
                    qh,
                    decoration_behavior,
                )
                .location(loc!())?;
            },
            WaylandWindowType::Popup => match parent {
                None => {
                    debug!(
                        "creating xdg_toplevel for {self:?} instead of popup because parent was None"
                    );
                    self.parent_id = None;
                    XWaylandXdgToplevel::set_role(
                        self,
                        xdg_shell_state,
                        shm_state,
                        subcompositor_state,
                        qh,
                        decoration_behavior,
                    )
                    .location(loc!())?;
                },
                Some(X11Parent {
                    for_popup: Some(popup_parent),
                    ..
                }) => {
                    debug!("creating xdg_popup for {self:?}");
                    self.parent_id = Some(popup_parent.surface_id.clone());
                    XWaylandXdgPopup::set_role(self, &popup_parent, xdg_shell_state, qh)
                        .location(loc!())?;
                },
                Some(X11Parent {
                    surface_id,
                    for_popup: None,
                    for_subsurface,
                }) => {
                    debug!(
                        "creating subsurface for {self:?} instead of popup because parent has no xdg_surface ancestor"
                    );
                    self.parent_id = Some(surface_id);
                    XWaylandSubSurface::set_role(
                        self,
                        for_subsurface,
                        shm_state,
                        subcompositor_state,
                        qh,
                    )
                    .location(loc!())?;
                },
            },
            WaylandWindowType::SubSurface => match parent {
                Some(parent) => {
                    debug!("creating subsurface for {self:?}");
                    self.parent_id = Some(parent.surface_id);
                    XWaylandSubSurface::set_role(
                        self,
                        parent.for_subsurface,
                        shm_state,
                        subcompositor_state,
                        qh,
                    )
                    .location(loc!())?;
                },
                None => {
                    debug!(
                        "creating xdg_toplevel for {self:?} instead of subsurface because parent was None"
                    );
                    self.parent_id = None;
                    XWaylandXdgToplevel::set_role(
                        self,
                        xdg_shell_state,
                        shm_state,
                        subcompositor_state,
                        qh,
                        decoration_behavior,
                    )
                    .location(loc!())?;
                },
            },
            // TODO: do we need a None for hidden helper windows?
        }

        Ok(())
    }
}

impl WaylandSurface for XWaylandSurface {
    fn wl_surface(&self) -> &ClientWlSurface {
        match &self.role {
            None | Some(Role::Cursor) => self.local_surface.as_ref().unwrap().wl_surface(),
            Some(Role::XdgToplevel(remote_xdg_toplevel)) => {
                remote_xdg_toplevel.local_window.wl_surface()
            },
            Some(Role::XdgPopup(remote_xdg_popup)) => remote_xdg_popup.local_popup.wl_surface(),
            Some(Role::SubSurface(remote_subsurface)) => remote_subsurface.wl_surface(),
        }
    }
}

#[derive(Debug)]
pub struct WprsState {
    pub dh: DisplayHandle,
    pub event_loop_handle: LoopHandle<'static, Self>,
    pub registration_tokens: Vec<RegistrationToken>,
    pub client_state: WprsClientState,
    pub compositor_state: WprsCompositorState,
    pub surface_bimap: BiMap<CompositorObjectId, ClientObjectId>,
    pub surfaces: HashMap<CompositorObjectId, XWaylandSurface>,
}

impl WprsState {
    pub fn new<K, V, I>(
        dh: DisplayHandle,
        globals: &GlobalList,
        qh: QueueHandle<Self>,
        conn: Connection,
        event_loop_handle: LoopHandle<'static, Self>,
        decoration_behavior: DecorationBehavior,
        xwayland_options: XwaylandOptions<K, V, I>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let mut registration_tokens = vec![];
        Ok(Self {
            dh: dh.clone(),
            event_loop_handle: event_loop_handle.clone(),
            client_state: WprsClientState::new(globals, qh, conn).location(loc!())?,
            compositor_state: WprsCompositorState::new(
                dh,
                &event_loop_handle,
                decoration_behavior,
                xwayland_options,
                &mut registration_tokens,
            ),
            surface_bimap: BiMap::new(),
            surfaces: HashMap::new(),
            registration_tokens,
        })
    }

    #[instrument(skip(self), level = "debug")]
    pub fn remove_surface(&mut self, surface_id: &CompositorObjectId) {
        let children = match self.surfaces.get(surface_id) {
            Some(surface) => surface.children.clone(),
            None => HashSet::new(),
        };

        for child in children {
            self.remove_surface(&child);
        }

        let removed_parent_id = self
            .surfaces
            .remove(surface_id)
            .and_then(|xwayland_surface| xwayland_surface.parent_id);

        if let Some(parent_id) = &removed_parent_id
            && let Some(parent_xwayland_surface) = self.surfaces.get_mut(parent_id)
        {
            parent_xwayland_surface
                .children
                .retain(|child_surface_id| child_surface_id != surface_id);
        }

        let fallback_focus = removed_parent_id.filter(|id| self.surfaces.contains_key(id));
        if self.client_state.last_pointer_surface.as_ref() == Some(surface_id) {
            self.client_state.last_pointer_surface = fallback_focus.clone();
        }
        if self.client_state.last_keyboard_surface.as_ref() == Some(surface_id) {
            self.client_state.last_keyboard_surface = fallback_focus;
        }

        // this MUST come after removing xwayland_surface, because xwayland_surface's role needs
        // to be destroyed before it's client wl_surface.
        // ultimately, the wayland object should be destroyed in order from:
        // xdg_popup/xdg_toplevel -> xdg_surface -> wl_surface
        self.surface_bimap.remove_by_left(surface_id);
    }

    #[instrument(
        skip(self, keycode, state),
        fields(keycode = "<redacted>", state = "<redacted>"),
        level = "debug"
    )]
    pub(crate) fn set_key_state(
        &mut self,
        keycode: u32,
        state: KeyState,
        serial: Serial,
    ) -> Result<()> {
        let keyboard = self.compositor_state.seat.get_keyboard().location(loc!())?;

        if args::get_log_priv_data() {
            Span::current().record("keycode", field::debug(&keycode));
            Span::current().record("state", field::debug(&state));
        }

        fn filter(
            _: &mut WprsState,
            modifiers_state: &ModifiersState,
            keysym: KeysymHandle,
        ) -> FilterResult<()> {
            if args::get_log_priv_data() {
                Span::current().record("modifiers_state", field::debug(&modifiers_state));
                Span::current().record("keysym", field::debug(&keysym));
            }
            FilterResult::Forward
        }

        // our keycode is getting offset by 8 for reasons
        // see https://github.com/Smithay/smithay/pull/1536
        let x11_keycode = (keycode + 8).into();
        let time = self.compositor_state.start_time.elapsed().as_millis() as u32;
        match state {
            KeyState::Pressed => {
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Pressed,
                    serial,
                    time,
                    filter,
                );
                self.compositor_state.pressed_keys.insert(keycode);
            },
            KeyState::Released => {
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Released,
                    serial,
                    time,
                    filter,
                );
                self.compositor_state.pressed_keys.remove(&keycode);
            },
            KeyState::Repeated => {
                // Map repeated to released + pressed
                // Smithay 0.7 keystates don't support repetition
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Released,
                    serial,
                    time,
                    filter,
                );
                keyboard.input::<(), _>(
                    self,
                    x11_keycode,
                    SmithayKeyState::Pressed,
                    serial,
                    time,
                    filter,
                );
            },
        }

        Ok(())
    }

    pub(crate) fn compositor_surface_from_id(
        &self,
        compositor_surface_id: &CompositorObjectId,
    ) -> Option<CompositorWlSurface> {
        let client = self.dh.get_client(compositor_surface_id.clone()).ok()?;
        client
            .object_from_protocol_id(&self.dh, compositor_surface_id.protocol_id())
            .ok()
    }

    pub fn compositor_surface_from_client_surface(
        &self,
        client_surface: &ClientWlSurface,
    ) -> Option<CompositorWlSurface> {
        let compositor_surface_id = self.surface_bimap.get_by_right(&client_surface.id())?;
        self.compositor_surface_from_id(compositor_surface_id)
    }

    pub(crate) fn enter_surfaces_on_output(&self, output_id: u32, output: &Output) {
        for (surface_id, xwayland_surface) in &self.surfaces {
            if xwayland_surface.output_ids.contains(&output_id)
                && let Some(surface) = self.compositor_surface_from_id(surface_id)
            {
                output.enter(&surface);
            }
        }
    }

    pub(crate) fn remove_output_from_surfaces(&mut self, output_id: u32, output: Option<&Output>) {
        let surface_ids: Vec<_> = self.surfaces.keys().cloned().collect();
        for surface_id in surface_ids {
            if let Some(output) = output
                && let Some(surface) = self.compositor_surface_from_id(&surface_id)
            {
                output.leave(&surface);
            }
            if let Some(xwayland_surface) = self.surfaces.get_mut(&surface_id) {
                xwayland_surface.output_ids.remove(&output_id);
            }
        }
    }

    pub fn sync_surface_outputs(&mut self, surface: &ClientWlSurface) {
        let (Some(compositor_surface), Some(outputs)) = (
            self.compositor_surface_from_client_surface(surface),
            surface.data::<SurfaceData<()>>().map(SurfaceData::outputs),
        ) else {
            return;
        };

        let new_ids: HashSet<u32> = HashSet::from_iter(outputs.filter_map(|output| {
            output
                .data::<OutputData>()
                .map(|data| data.with_output_info(|info| info.id))
        }));

        let output_rects = self.compositor_state.output_logical_rects();
        let Some(xwayland_surface) =
            xsurface_from_client_surface(&self.surface_bimap, &mut self.surfaces, surface)
        else {
            return;
        };

        compositor_utils::update_surface_outputs(
            &compositor_surface,
            &new_ids,
            &xwayland_surface.output_ids,
            |id| {
                self.compositor_state
                    .outputs
                    .get(id)
                    .map(|(output, _)| output)
            },
        );

        xwayland_surface.output_ids = new_ids;

        // When a configured toplevel with no active child popups/subsurfaces
        // moves onto a different output, move its X11 geometry into that output
        // so RandR-aware apps query the right monitor bounds.
        if let Some(Role::XdgToplevel(toplevel)) = &xwayland_surface.role
            && toplevel.configured
            && xwayland_surface.children.is_empty()
            && let Some(x11_surface) = &xwayland_surface.x11_surface
        {
            let geo = x11_surface.geometry();
            if compositor::toplevel_needs_output_reposition(
                &output_rects,
                &xwayland_surface.output_ids,
                geo,
            ) {
                let new_loc = compositor::compute_toplevel_x11_location(
                    &output_rects,
                    &xwayland_surface.output_ids,
                    (geo.size.w, geo.size.h),
                    false,
                );
                x11_surface
                    .configure(Rectangle::new(new_loc.into(), geo.size))
                    .log_and_ignore(loc!());
            }
        }
    }
}

impl Drop for WprsState {
    fn drop(&mut self) {
        for token in self.registration_tokens.drain(..) {
            self.event_loop_handle.remove(token);
        }
    }
}

pub fn xsurface_from_client_surface<'a>(
    surface_bimap: &BiMap<CompositorObjectId, ClientObjectId>,
    surfaces: &'a mut HashMap<CompositorObjectId, XWaylandSurface>,
    surface: &ClientWlSurface,
) -> Option<&'a mut XWaylandSurface> {
    debug!(
        "xsurface_from_client_surface, {:?}, {:?}",
        surface.id(),
        surfaces
    );
    let compositor_surface_id = surface_bimap.get_by_right(&surface.id())?;
    surfaces.get_mut(compositor_surface_id)
}

pub fn xsurface_from_x11_surface<'a>(
    surfaces: &'a mut HashMap<CompositorObjectId, XWaylandSurface>,
    surface: &X11Surface,
) -> Option<&'a mut XWaylandSurface> {
    surfaces.values_mut().find(|xws| {
        xws.x11_surface
            .as_ref()
            .map(|s| s == surface)
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_wayland_window_type() {
        // Override-redirect windows ALWAYS map to Popup even when
        // WM_TRANSIENT_FOR is set (GTK/Qt/Java/Chromium context menus).
        for window_type in [
            None,
            Some(WmWindowType::Normal),
            Some(WmWindowType::Dialog),
            Some(WmWindowType::Utility),
            Some(WmWindowType::PopupMenu),
            Some(WmWindowType::DropdownMenu),
            Some(WmWindowType::Tooltip),
        ] {
            for has_transient in [false, true] {
                assert_eq!(
                    classify_wayland_window_type(window_type, true, has_transient),
                    WaylandWindowType::Popup,
                    "override-redirect {window_type:?} transient={has_transient}"
                );
            }
        }

        // EWMH popup types also always map to Popup even when non-OR + transient.
        for window_type in [
            WmWindowType::DropdownMenu,
            WmWindowType::Menu,
            WmWindowType::Notification,
            WmWindowType::PopupMenu,
            WmWindowType::Toolbar,
            WmWindowType::Tooltip,
        ] {
            for has_transient in [false, true] {
                assert_eq!(
                    classify_wayland_window_type(Some(window_type), false, has_transient),
                    WaylandWindowType::Popup,
                    "non-OR {window_type:?} transient={has_transient}"
                );
            }
        }

        // Non-override-redirect normal/dialog/utility/splash windows become
        // SubSurface only when explicitly transient, otherwise Toplevel.
        for window_type in [
            None,
            Some(WmWindowType::Normal),
            Some(WmWindowType::Dialog),
            Some(WmWindowType::Utility),
            Some(WmWindowType::Splash),
        ] {
            assert_eq!(
                classify_wayland_window_type(window_type, false, true),
                WaylandWindowType::SubSurface,
                "transient non-OR {window_type:?}"
            );
            assert_eq!(
                classify_wayland_window_type(window_type, false, false),
                WaylandWindowType::Toplevel,
                "standalone non-OR {window_type:?}"
            );
        }
    }
}
