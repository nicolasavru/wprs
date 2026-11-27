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
use std::num::NonZeroU32;
use std::sync::Arc;

use enum_as_inner::EnumAsInner;
use smithay::backend::input::ButtonState;
use smithay::input::keyboard::Layout;
use smithay::input::keyboard::XkbContext;
use smithay::input::pointer::ButtonEvent;
use smithay::input::pointer::MotionEvent;
use smithay::input::pointer::PointerTarget;
use smithay::reexports::wayland_protocols::wp::primary_selection::zv1::client::zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1;
use smithay::reexports::wayland_protocols::wp::primary_selection::zv1::client::zwp_primary_selection_source_v1::ZwpPrimarySelectionSourceV1;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::Resource;
use smithay::utils::Logical;
use smithay::utils::Point as SmithayPoint;
use smithay::utils::Rectangle;
use smithay::utils::SERIAL_COUNTER;
use smithay::wayland::compositor;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::selection::data_device;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::primary_selection;
use smithay::wayland::shm::BufferData;
use smithay::xwayland::X11Surface;
use smithay_client_toolkit::compositor::CompositorHandler;
use smithay_client_toolkit::compositor::CompositorState;
use smithay_client_toolkit::compositor::FrameCallbackData;
use smithay_client_toolkit::compositor::Surface;
use smithay_client_toolkit::data_device_manager::data_device::DataDeviceHandler;
use smithay_client_toolkit::data_device_manager::data_offer::DataOfferHandler;
use smithay_client_toolkit::data_device_manager::data_offer::DragOffer;
use smithay_client_toolkit::data_device_manager::data_offer::SelectionOffer;
use smithay_client_toolkit::data_device_manager::data_source::CopyPasteSource;
use smithay_client_toolkit::data_device_manager::data_source::DataSourceHandler;
use smithay_client_toolkit::data_device_manager::DataDeviceManagerState;
use smithay_client_toolkit::data_device_manager::WritePipe;
use smithay_client_toolkit::output::OutputHandler;
use smithay_client_toolkit::output::OutputState;
use smithay_client_toolkit::primary_selection::selection::PrimarySelectionSource;
use smithay_client_toolkit::primary_selection::PrimarySelectionManagerState;
use smithay_client_toolkit::primary_selection::device::PrimarySelectionDeviceHandler;
use smithay_client_toolkit::primary_selection::offer::PrimarySelectionOffer;
use smithay_client_toolkit::primary_selection::selection::PrimarySelectionSourceHandler;
use smithay_client_toolkit::reexports::client::globals::GlobalList;
use smithay_client_toolkit::reexports::client::protocol::wl_data_device::WlDataDevice;
use smithay_client_toolkit::reexports::client::protocol::wl_data_device_manager::DndAction;
use smithay_client_toolkit::reexports::client::protocol::wl_data_source::WlDataSource;
use smithay_client_toolkit::reexports::client::protocol::wl_keyboard::WlKeyboard;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Transform;
use smithay_client_toolkit::reexports::client::protocol::wl_output::WlOutput;
use smithay_client_toolkit::reexports::client::protocol::wl_pointer::WlPointer;
use smithay_client_toolkit::reexports::client::protocol::wl_seat::WlSeat;
use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::Event as WlSubcompositorEvent;
use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::WlSubcompositor;
use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::Event as WlSubsurfaceEvent;
use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::WlSubsurface;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface;
use smithay_client_toolkit::reexports::client::Connection;
use smithay_client_toolkit::reexports::client::Dispatch;
use smithay_client_toolkit::reexports::client::Proxy;
use smithay_client_toolkit::reexports::client::QueueHandle;
use smithay_client_toolkit::reexports::csd_frame::CursorIcon;
use smithay_client_toolkit::reexports::csd_frame::DecorationsFrame;
use smithay_client_toolkit::reexports::csd_frame::WindowManagerCapabilities;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_positioner::Anchor;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_positioner::ConstraintAdjustment;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_positioner::Gravity;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_surface::XdgSurface as SctkXdgSurface;
use smithay_client_toolkit::registry::ProvidesRegistryState;
use smithay_client_toolkit::registry::RegistryState;
use smithay_client_toolkit::registry_handlers;
use smithay_client_toolkit::seat::keyboard::KeyEvent;
use smithay_client_toolkit::seat::keyboard::KeyboardHandler;
use smithay_client_toolkit::seat::keyboard::Keymap;
use smithay_client_toolkit::seat::keyboard::Keysym;
use smithay_client_toolkit::seat::keyboard::Modifiers;
use smithay_client_toolkit::seat::keyboard::RawModifiers;
use smithay_client_toolkit::seat::keyboard::RepeatInfo;
use smithay_client_toolkit::seat::pointer::PointerData;
use smithay_client_toolkit::seat::pointer::PointerEvent;
use smithay_client_toolkit::seat::pointer::PointerEventKind;
use smithay_client_toolkit::seat::pointer::PointerHandler;
use smithay_client_toolkit::seat::pointer::ThemeSpec;
use smithay_client_toolkit::seat::pointer::ThemedPointer;
use smithay_client_toolkit::seat::Capability;
use smithay_client_toolkit::seat::SeatHandler;
use smithay_client_toolkit::seat::SeatState;
use smithay_client_toolkit::shell::xdg::fallback_frame::FallbackFrame;
use smithay_client_toolkit::shell::xdg::popup::Popup;
use smithay_client_toolkit::shell::xdg::popup::PopupConfigure;
use smithay_client_toolkit::shell::xdg::popup::PopupHandler;
use smithay_client_toolkit::shell::xdg::window::DecorationMode;
use smithay_client_toolkit::shell::xdg::window::Window;
use smithay_client_toolkit::shell::xdg::window::WindowConfigure;
use smithay_client_toolkit::shell::xdg::window::WindowDecorations;
use smithay_client_toolkit::shell::xdg::window::WindowHandler;
use smithay_client_toolkit::shell::xdg::XdgPositioner;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::xdg::XdgSurface;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shm::slot::Buffer;
use smithay_client_toolkit::shm::slot::SlotPool;
use smithay_client_toolkit::shm::Shm;
use smithay_client_toolkit::shm::ShmHandler;
use smithay_client_toolkit::subcompositor::SubcompositorState;
use tracing::Span;

use crate::args;
use crate::buffer_pointer::BufferPointer;
use crate::client_utils::FrameDamage;
use crate::client_utils::SeatObject;
use crate::client_utils::SwapchainStep;
use crate::client_utils::advance_two_slot_swapchain;
use crate::client_utils::combine_band_history;
use crate::compositor_utils;
use crate::filtering;
use crate::prelude::*;
use crate::serialization;
use crate::serialization::framing::RowBand;
use crate::serialization::geometry::Point;
use crate::serialization::geometry::Rectangle as SerRectangle;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::KeyState;
use crate::xwayland_xdg_shell::compositor::DecorationBehavior;
use crate::xwayland_xdg_shell::compositor::X11ParentForPopup;
use crate::xwayland_xdg_shell::compositor::X11ParentForSubsurface;
use crate::xwayland_xdg_shell::compositor::popup_parent_origin;
use crate::xwayland_xdg_shell::decoration::SubSurfaceResizeState;
use crate::xwayland_xdg_shell::decoration::handle_window_frame_pointer_event;
use crate::xwayland_xdg_shell::xsurface_from_client_surface;
use crate::xwayland_xdg_shell::WprsState;
use crate::xwayland_xdg_shell::XWaylandSurface;

const DEFAULT_WINDOW_SIZE: (i32, i32) = (512, 256);

pub(crate) const POPUP_CONSTRAINT_ADJUSTMENT: ConstraintAdjustment = ConstraintAdjustment::SlideX
    .union(ConstraintAdjustment::SlideY)
    .union(ConstraintAdjustment::FlipX)
    .union(ConstraintAdjustment::FlipY);

/// Translates an X11 popup top-left coordinate into an `xdg_positioner` anchor
/// coordinate relative to the parent's `xdg_surface` window geometry.
pub(crate) fn popup_anchor_point(
    popup_x11_loc: Point<i32>,
    parent_x11_loc: Point<i32>,
    parent_frame_offset: Point<i32>,
) -> Point<i32> {
    (
        popup_x11_loc.x - parent_x11_loc.x + parent_frame_offset.x,
        popup_x11_loc.y - parent_x11_loc.y + parent_frame_offset.y,
    )
        .into()
}

/// Inverse of [`popup_anchor_point`]: translates a parent-window-geometry-relative
/// `xdg_popup.configure` position back to X11 root coordinates.
pub(crate) fn popup_configure_to_x11_loc(
    configure_pos: (i32, i32),
    parent_x11_loc: Point<i32>,
    parent_frame_offset: Point<i32>,
) -> Point<i32> {
    (
        parent_x11_loc.x - parent_frame_offset.x + configure_pos.0,
        parent_x11_loc.y - parent_frame_offset.y + configure_pos.1,
    )
        .into()
}

/// Converts a surface-local Wayland pointer position into global X11 root
/// coordinates so Smithay's `ClickGrab` keeps pointer coordinates aligned when
/// dragging from a parent window onto an override-redirect popup.
pub(crate) fn pointer_global_location(
    surface_loc: SmithayPoint<i32, Logical>,
    position: (f64, f64),
) -> SmithayPoint<f64, Logical> {
    (
        f64::from(surface_loc.x) + position.0,
        f64::from(surface_loc.y) + position.1,
    )
        .into()
}

/// Returns whether entering an X11 surface should call `X11Wm::raise_window`.
///
/// Override-redirect popups must never be pushed into `_NET_CLIENT_LIST_STACKING`,
/// and a parent with mapped child popups/subsurfaces must not be raised above
/// its own open children.
pub(crate) fn should_raise_window_on_enter(
    is_override_redirect: bool,
    has_mapped_children: bool,
) -> bool {
    !is_override_redirect && !has_mapped_children
}

fn configure_popup_positioner(
    xdg_shell_state: &XdgShell,
    geometry: Rectangle<i32, Logical>,
    parent_x11_loc: Point<i32>,
    parent_frame_offset: Point<i32>,
) -> Result<XdgPositioner> {
    let positioner = XdgPositioner::new(xdg_shell_state).location(loc!())?;
    positioner.set_size(geometry.size.w.max(1), geometry.size.h.max(1));
    let anchor = popup_anchor_point(geometry.loc.into(), parent_x11_loc, parent_frame_offset);
    positioner.set_anchor_rect(anchor.x, anchor.y, 1, 1);
    positioner.set_anchor(Anchor::TopLeft);
    positioner.set_gravity(Gravity::BottomRight);
    // Allow sliding/flipping at screen edges without resizing override-redirect
    // X11 popups, which cannot be resized by the window manager.
    positioner.set_constraint_adjustment(POPUP_CONSTRAINT_ADJUSTMENT);
    positioner.set_reactive();
    Ok(positioner)
}

/// Determines which `xdg-decoration` mode to request from the host Wayland
/// compositor for a toplevel window.
///
/// Under `Auto`, client-side-decorated X11 windows (`x11_is_decorated == true`)
/// request `DecorationMode::Client` so SSD compositors don't add a second
/// titlebar, while undecorated X11 windows request `DecorationMode::Server` and
/// fall back to `FallbackFrame` if the host configures `Client`.
pub(crate) fn desired_decoration_mode(
    behavior: DecorationBehavior,
    x11_is_decorated: bool,
) -> DecorationMode {
    match behavior {
        DecorationBehavior::Auto if !x11_is_decorated => DecorationMode::Server,
        DecorationBehavior::Auto
        | DecorationBehavior::AlwaysEnabled
        | DecorationBehavior::AlwaysDisabled => DecorationMode::Client,
    }
}

pub(crate) fn window_decorations_for_mode(mode: DecorationMode) -> WindowDecorations {
    match mode {
        DecorationMode::Client => WindowDecorations::RequestClient,
        DecorationMode::Server => WindowDecorations::RequestServer,
    }
}

/// Sanitizes X11 `WM_NORMAL_HINTS` min/max sizes for `xdg_toplevel.set_min_size`
/// / `set_max_size`: clamps negatives to `0` and ensures a non-zero max bound is
/// never smaller than the corresponding min bound (which is an xdg-shell
/// protocol error).
pub(crate) fn sanitize_xdg_size_hints(
    min_size: Option<(i32, i32)>,
    max_size: Option<(i32, i32)>,
) -> (Option<(u32, u32)>, Option<(u32, u32)>) {
    let min = min_size.map(|(w, h)| (w.max(0) as u32, h.max(0) as u32));
    let (min_w, min_h) = min.unwrap_or((0, 0));
    let max = max_size.map(|(w, h)| {
        let mut mw = w.max(0) as u32;
        let mut mh = h.max(0) as u32;
        if mw > 0 && mw < min_w {
            mw = min_w;
        }
        if mh > 0 && mh < min_h {
            mh = min_h;
        }
        (mw, mh)
    });
    (min, max)
}

pub(crate) fn is_active_source<S, W>(
    active: Option<&S>,
    source: &W,
    inner: impl FnOnce(&S) -> &W,
) -> bool
where
    W: PartialEq,
{
    active.is_some_and(|active| inner(active) == source)
}

pub(crate) fn clear_if_active_source<S, W>(
    active: &mut Option<S>,
    source: &W,
    inner: impl FnOnce(&S) -> &W,
) -> bool
where
    W: PartialEq,
{
    if is_active_source(active.as_ref(), source, inner) {
        *active = None;
        true
    } else {
        false
    }
}

#[derive(Debug)]
pub struct WprsClientState {
    pub qh: QueueHandle<WprsState>,
    pub(crate) conn: Connection,

    pub registry_state: RegistryState,
    pub seat_state: SeatState,
    pub output_state: OutputState,
    pub compositor_state: CompositorState,
    pub subcompositor_state: Arc<SubcompositorState>,
    pub shm_state: Shm,
    pub xdg_shell_state: XdgShell,

    pub(crate) data_device_manager_state: DataDeviceManagerState,
    pub(crate) primary_selection_manager_state: Option<PrimarySelectionManagerState>,

    pub exit: bool,
    pub pool: Option<SlotPool>,

    pub last_enter_serial: u32,
    pub(crate) last_implicit_grab_serial: u32,
    pub(crate) last_implicit_grab_seat: Option<WlSeat>,
    pub(crate) last_pointer_press: Option<(WlSeat, u32)>,
    pub(crate) last_pointer_surface: Option<ObjectId>,
    pub(crate) last_keyboard_surface: Option<ObjectId>,

    pub(crate) seat_objects: Vec<SeatObject<ThemedPointer>>,
    pub(crate) cursor_icon: Option<CursorIcon>,
    pub(crate) selection_offer: Option<SelectionOffer>,
    pub(crate) selection_source: Option<CopyPasteSource>,
    pub(crate) primary_selection_offer: Option<PrimarySelectionOffer>,
    pub(crate) primary_selection_source: Option<PrimarySelectionSource>,
}

impl WprsClientState {
    pub fn new(globals: &GlobalList, qh: QueueHandle<WprsState>, conn: Connection) -> Result<Self> {
        let shm_state = Shm::bind(globals, &qh).context(loc!(), "wl_shm is not available")?;
        let pool =
            Some(SlotPool::new(3840 * 2160, &shm_state).context(loc!(), "failed to create pool")?);
        let compositor_state = CompositorState::bind(globals, &qh)
            .context(loc!(), "wl_compositor is not available")?;
        let subcompositor_state = Arc::new(
            SubcompositorState::bind(compositor_state.wl_compositor().clone(), globals, &qh)
                .context(loc!(), "wl_subcompositor is not available")?,
        );

        Ok(Self {
            qh: qh.clone(),
            conn,
            registry_state: RegistryState::new(globals),
            seat_state: SeatState::new(globals, &qh),
            output_state: OutputState::new(globals, &qh),
            compositor_state,
            subcompositor_state,
            shm_state,
            xdg_shell_state: XdgShell::bind(globals, &qh)
                .context(loc!(), "xdg shell is not available")?,
            data_device_manager_state: DataDeviceManagerState::bind(globals, &qh)
                .context(loc!(), "data device manager is not available")?,
            primary_selection_manager_state: PrimarySelectionManagerState::bind(globals, &qh)
                .context(loc!(), "primary selection manager is not available")
                .warn(loc!())
                .ok(),

            exit: false,
            pool,

            last_enter_serial: 0,
            last_implicit_grab_serial: 0,
            last_implicit_grab_seat: None,
            last_pointer_press: None,
            last_pointer_surface: None,
            last_keyboard_surface: None,

            seat_objects: Vec::new(),
            cursor_icon: None,
            selection_offer: None,
            selection_source: None,
            primary_selection_offer: None,
            primary_selection_source: None,
        })
    }

    pub(crate) fn interactive_grab_seat_and_serial(&self) -> Option<(WlSeat, u32)> {
        if let Some((seat, serial)) = &self.last_pointer_press {
            return Some((seat.clone(), *serial));
        }
        let seat = self
            .last_implicit_grab_seat
            .clone()
            .or_else(|| self.seat_objects.last().map(|s| s.seat.clone()))?;
        Some((seat, self.last_implicit_grab_serial))
    }
}

impl CompositorHandler for WprsState {
    #[instrument(skip(self, _conn, _qh, _new_factor), level = "debug")]
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _new_factor: i32,
    ) {
        self.sync_surface_outputs(surface);
    }

    #[instrument(skip(self, _conn, _qh, _new_transform), level = "debug")]
    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _new_transform: Transform,
    ) {
        self.sync_surface_outputs(surface);
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        time: u32,
    ) {
        if let Some(compositor_surface_id) = self.surface_bimap.get_by_right(&surface.id()) {
            let xwayland_surface = self.surfaces.get_mut(compositor_surface_id).unwrap();
            if let Some(Role::SubSurface(subsurface)) = &mut xwayland_surface.role {
                subsurface.pending_frame_callback = false;
            }
            if let Some(x11_surface) = &xwayland_surface.x11_surface
                && let Some(wl_surface) = x11_surface.wl_surface()
            {
                compositor::with_states(&wl_surface, |surface_data| {
                    for callback in surface_data
                        .cached_state
                        .get::<SurfaceAttributes>()
                        .current()
                        .frame_callbacks
                        .drain(..)
                    {
                        debug!(
                            "Sending callback for client surface {:?}, compositor surface {:?}: {:?}.",
                            surface.id(),
                            xwayland_surface.wl_surface().id(),
                            callback.id()
                        );
                        callback
                            .done(self.compositor_state.start_time.elapsed().as_millis() as u32);
                    }
                });
            }
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _output: &WlOutput,
    ) {
        self.sync_surface_outputs(surface);
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &WlSurface,
        _output: &WlOutput,
    ) {
        self.sync_surface_outputs(surface);
    }
}

impl OutputHandler for WprsState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.client_state.output_state
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn new_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        let output_id = output_info.id;
        let local_output = self.compositor_state.new_output(output_info.into());
        self.enter_surfaces_on_output(output_id, &local_output);
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn update_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        self.compositor_state.update_output(output_info.into());
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn output_destroyed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: WlOutput) {
        let output_info = self.output_state().info(&output).unwrap();
        let output_id = output_info.id;
        let removed = self.compositor_state.destroy_output(output_info.into());
        self.remove_output_from_surfaces(output_id, removed.as_ref());
    }
}

impl WindowHandler for WprsState {
    fn request_close(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, window: &Window) {
        let compositor_surface_id = self
            .surface_bimap
            .get_by_right(&window.wl_surface().id())
            .unwrap();
        let xwayland_surface = self.surfaces.get_mut(compositor_surface_id).unwrap();
        let x11_surface = &xwayland_surface.x11_surface.as_ref().unwrap();
        x11_surface.close().log_and_ignore(loc!());
    }

    #[instrument(skip(self, _conn, _qh, _serial), level = "debug")]
    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        window: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        let Some(compositor_surface_id) =
            self.surface_bimap.get_by_right(&window.wl_surface().id())
        else {
            warn!("Received configure for already-destroyed window {window:?}.");
            return;
        };

        let output_rects = self.compositor_state.output_logical_rects();
        let maximized_or_fullscreen = configure.is_maximized() || configure.is_fullscreen();
        let xwayland_surface = self.surfaces.get_mut(compositor_surface_id).unwrap();
        let surface_output_ids = &xwayland_surface.output_ids;

        let xdg_toplevel = match &mut xwayland_surface.role {
            Some(Role::XdgToplevel(xdg_toplevel)) => xdg_toplevel,
            _ => unreachable!(
                "expected role xdg_toplevel, found role {:?}",
                &xwayland_surface.role
            ),
        };
        let x11_surface = &xwayland_surface.x11_surface.as_ref().unwrap();

        x11_surface
            .set_maximized(configure.is_maximized())
            .log_and_ignore(loc!());
        x11_surface
            .set_fullscreen(configure.is_fullscreen())
            .log_and_ignore(loc!());

        xdg_toplevel.last_configure = Some(configure.clone());
        xdg_toplevel.sync_decoration_mode(x11_surface.is_decorated());

        xdg_toplevel
            .apply_decoration(
                x11_surface,
                Some(&configure),
                xwayland_surface
                    .buffer
                    .as_ref()
                    .map(|buffer| &buffer.metadata),
                |w, h| {
                    crate::xwayland_xdg_shell::compositor::compute_toplevel_x11_location(
                        &output_rects,
                        surface_output_ids,
                        (w, h),
                        maximized_or_fullscreen,
                    )
                },
            )
            .log_and_ignore(loc!());

        if !xdg_toplevel.window_frame.is_hidden() && xdg_toplevel.window_frame.is_dirty() {
            xdg_toplevel.window_frame.draw();
        }

        // The code below commits the buffer we received but couldn't attach
        // because we hadn't received our initial configure. In the normal
        // configure case, the above code will have sent some X11 configure
        // events and xwayland can commit in response if it desires.
        if xdg_toplevel.configured {
            return;
        }

        xdg_toplevel.configured = true;

        xwayland_surface.commit_buffer(&self.client_state.qh);
    }
}

impl PopupHandler for WprsState {
    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        popup: &Popup,
        configure: PopupConfigure,
    ) {
        let Some(compositor_surface_id) = self
            .surface_bimap
            .get_by_right(&popup.wl_surface().id())
            .cloned()
        else {
            warn!("Received configure for already-destroyed popup {popup:?}.");
            return;
        };

        let parent_origin = self
            .surfaces
            .get(&compositor_surface_id)
            .and_then(|s| match &s.role {
                Some(Role::XdgPopup(xdg_popup)) => {
                    popup_parent_origin(&self.surfaces, &xdg_popup.parent)
                },
                _ => None,
            });

        let Some(xwayland_surface) = self.surfaces.get_mut(&compositor_surface_id) else {
            return;
        };

        let x11_surface = log_and_return!(xwayland_surface.get_x11_surface());
        let geo = if x11_surface.is_override_redirect() {
            None
        } else {
            let x11_loc = match parent_origin {
                Some((parent_x11_loc, parent_frame_offset)) => popup_configure_to_x11_loc(
                    configure.position,
                    parent_x11_loc,
                    parent_frame_offset,
                ),
                None => configure.position.into(),
            };
            Some(Rectangle::new(
                x11_loc.into(),
                (configure.width, configure.height).into(),
            ))
        };

        x11_surface.configure(geo).log_and_ignore(loc!());

        let xdg_popup = match &mut xwayland_surface.role {
            Some(Role::XdgPopup(xdg_popup)) => xdg_popup,
            _ => unreachable!(
                "expected role xdg_popup, found role {:?}",
                &xwayland_surface.role
            ),
        };
        if let Some(geo) = geo {
            xdg_popup.last_geometry = geo;
        }

        // The code below commits the buffer we received but couldn't attach
        // because we hadn't received our initial commit. In the normal
        // configure case, the above code will have sent some X11 configure
        // events and the xwayland can commit in response if so desires.
        if xdg_popup.configured {
            return;
        }

        xdg_popup.configured = true;

        xwayland_surface.commit_buffer(&self.client_state.qh);
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn done(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, popup: &Popup) {
        let Some(compositor_surface_id) = self
            .surface_bimap
            .get_by_right(&popup.wl_surface().id())
            .cloned()
        else {
            return;
        };
        if let Some(x11_surface) = self
            .surfaces
            .get(&compositor_surface_id)
            .and_then(|s| s.x11_surface.as_ref())
            && !x11_surface.is_override_redirect()
        {
            x11_surface.close().log_and_ignore(loc!());
        }
        self.remove_surface(&compositor_surface_id);
    }
}

impl SeatHandler for WprsState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.client_state.seat_state
    }

    fn new_seat(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _seat: WlSeat) {}

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: WlSeat,
        capability: Capability,
    ) {
        let seat_obj = if let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter_mut()
            .find(|s| s.seat == seat)
        {
            seat_obj
        } else {
            // create the data device here for this seat
            let data_device_manager = &self.client_state.data_device_manager_state;
            let data_device = data_device_manager.get_data_device(qh, &seat);

            let primary_selection_device = self
                .client_state
                .primary_selection_manager_state
                .as_ref()
                .map(|x| x.get_selection_device(qh, &seat));

            self.client_state.seat_objects.push(SeatObject {
                seat: seat.clone(),
                keyboard: None,
                pointer: None,
                data_device,
                primary_selection_device,
            });
            self.client_state.seat_objects.last_mut().unwrap()
        };

        if capability == Capability::Keyboard && seat_obj.keyboard.is_none() {
            println!("Set keyboard capability");
            let keyboard = self
                .client_state
                .seat_state
                .get_keyboard(qh, &seat, None)
                .expect("Failed to create keyboard");
            seat_obj.keyboard.replace(keyboard);
        }

        if capability == Capability::Pointer && seat_obj.pointer.is_none() {
            println!("Set pointer capability");
            let themed_pointer = self
                .client_state
                .seat_state
                .get_pointer_with_theme::<_, ()>(
                    qh,
                    &seat,
                    self.client_state.shm_state.wl_shm(),
                    self.client_state.compositor_state.create_surface(qh),
                    ThemeSpec::default(),
                )
                .expect("Failed to create pointer");
            seat_obj.pointer.replace(themed_pointer);
        }
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        seat: WlSeat,
        capability: Capability,
    ) {
        if let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter_mut()
            .find(|s| s.seat == seat)
        {
            match capability {
                Capability::Keyboard if let Some(k) = seat_obj.keyboard.take() => {
                    k.release();
                },
                Capability::Pointer if let Some(p) = seat_obj.pointer.take() => {
                    p.pointer().release();
                },
                _ => {},
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, seat: WlSeat) {
        self.client_state.seat_objects.retain(|s| s.seat != seat);
        if self
            .client_state
            .last_pointer_press
            .as_ref()
            .is_some_and(|(s, _)| *s == seat)
        {
            self.client_state.last_pointer_press = None;
        }
        if self.client_state.last_implicit_grab_seat.as_ref() == Some(&seat) {
            self.client_state.last_implicit_grab_seat = None;
        }
    }
}

impl KeyboardHandler for WprsState {
    fn enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        surface: &WlSurface,
        serial: u32,
        raw: &[u32],
        _keysyms: &[Keysym],
    ) {
        // see linux/input-event-codes.h for keycodes
        let modifier_keycodes = HashSet::from([
            /* KEY_LEFTCTRL */ 29, /* KEY_RIGHTCTRL */ 97, /* KEY_LEFTALT */ 56,
            /* KEY_RIGHTALT */ 100, /* KEY_LEFTMETA	*/ 125, /* KEY_RIGHTMETA */ 126,
            /* KEY_LEFTSHIFT */ 42, /* KEY_RIGHTSHIFT */ 54,
        ]);

        let keyboard = log_and_return!(
            self.compositor_state
                .seat
                .get_keyboard()
                .ok_or("seat has no keyboard")
        );

        // We simulate keycodes before focusing since that is what a normal wayland application would see.
        // Process modifier keys first so that they apply to other held keys.
        let mut delayed_keycodes = Vec::new();
        for keycode in raw {
            if modifier_keycodes.contains(keycode) {
                log_and_return!(self.set_key_state(
                    *keycode,
                    KeyState::Pressed,
                    SERIAL_COUNTER.next_serial(),
                ));
            } else {
                delayed_keycodes.push(keycode);
            }
        }
        for keycode in delayed_keycodes {
            log_and_return!(self.set_key_state(
                *keycode,
                KeyState::Pressed,
                SERIAL_COUNTER.next_serial()
            ));
        }
        let Some(compositor_surface_id) = self.surface_bimap.get_by_right(&surface.id()).cloned()
        else {
            // surface was already destroyed
            return;
        };
        self.client_state.last_keyboard_surface = Some(compositor_surface_id.clone());
        let Some(xwayland_surface) = self.surfaces.get(&compositor_surface_id) else {
            return;
        };
        let x11_surface = log_and_return!(xwayland_surface.get_x11_surface()).clone();
        let client = x11_surface.wl_surface().unwrap().client();
        x11_surface.set_activated(true).unwrap();
        let serial = self.compositor_state.serial_map.insert(serial);
        keyboard.set_focus(self, Some(x11_surface), serial);
        data_device::set_data_device_focus(
            &self.compositor_state.dh,
            &self.compositor_state.seat,
            client.clone(),
        );
        primary_selection::set_primary_focus(
            &self.compositor_state.dh,
            &self.compositor_state.seat,
            client,
        );
    }

    fn leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        surface: &WlSurface,
        serial: u32,
    ) {
        let Some(xwayland_surface) =
            xsurface_from_client_surface(&self.surface_bimap, &mut self.surfaces, surface)
        else {
            // surface was already destroyed
            return;
        };
        let x11_surface = log_and_return!(xwayland_surface.get_x11_surface()).clone();
        x11_surface.set_activated(false).unwrap();
        let keyboard = log_and_return!(
            self.compositor_state
                .seat
                .get_keyboard()
                .ok_or("seat has no keyboard")
        );

        let serial = self.compositor_state.serial_map.insert(serial);
        keyboard.set_focus(self, None, serial);
        data_device::set_data_device_focus(
            &self.compositor_state.dh,
            &self.compositor_state.seat,
            None,
        );
        primary_selection::set_primary_focus(
            &self.compositor_state.dh,
            &self.compositor_state.seat,
            None,
        );

        for keycode in self.compositor_state.pressed_keys.clone() {
            log_and_return!(self.set_key_state(keycode, KeyState::Released, serial));
        }
    }

    // INTENTIONALLY NOT LOGGING KEY EVENTS
    #[instrument(
        skip(self, _conn, _qh, keyboard, event),
        fields(event),
        level = "debug"
    )]
    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        keyboard: &WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        if args::get_log_priv_data() {
            Span::current().record("event", field::debug(&event));
        }
        self.client_state.last_implicit_grab_serial = serial;
        if let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter()
            .find(|s| s.keyboard.as_ref() == Some(keyboard))
        {
            self.client_state.last_implicit_grab_seat = Some(seat_obj.seat.clone());
        }
        let serial = self.compositor_state.serial_map.insert(serial);
        log_and_return!(self.set_key_state(event.raw_code, KeyState::Pressed, serial));
    }

    // INTENTIONALLY NOT LOGGING KEY EVENTS
    #[instrument(
        skip(self, _conn, _qh, keyboard, event),
        fields(event),
        level = "debug"
    )]
    fn repeat_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        keyboard: &WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        if args::get_log_priv_data() {
            Span::current().record("event", field::debug(&event));
        }
        self.client_state.last_implicit_grab_serial = serial;
        if let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter()
            .find(|s| s.keyboard.as_ref() == Some(keyboard))
        {
            self.client_state.last_implicit_grab_seat = Some(seat_obj.seat.clone());
        }
        let serial = self.compositor_state.serial_map.insert(serial);
        log_and_return!(self.set_key_state(event.raw_code, KeyState::Repeated, serial));
    }

    // INTENTIONALLY NOT LOGGING KEY EVENTS
    #[instrument(
        skip(self, _conn, _qh, _keyboard, event),
        fields(event = "<redacted>"),
        level = "debug"
    )]
    fn release_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        if args::get_log_priv_data() {
            Span::current().record("event", field::debug(&event));
        }
        let serial = self.compositor_state.serial_map.insert(serial);

        log_and_return!(self.set_key_state(event.raw_code, KeyState::Released, serial));
    }

    fn update_repeat_info(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        info: RepeatInfo,
    ) {
        let keyboard = log_and_return!(
            self.compositor_state
                .seat
                .get_keyboard()
                .ok_or("seat has no keyboard")
        );
        let (rate, delay) = match info {
            RepeatInfo::Repeat { rate, delay } => (rate.get(), delay),
            RepeatInfo::Disable => (0, 0),
        };
        keyboard.change_repeat_info(rate as i32, delay as i32);
    }

    fn update_keymap(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        keymap: Keymap<'_>,
    ) {
        let keyboard = log_and_return!(
            self.compositor_state
                .seat
                .get_keyboard()
                .ok_or("seat has no keyboard")
        );
        log_and_return!(keyboard.set_keymap_from_string(self, keymap.as_string()));
    }

    fn update_modifiers(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &WlKeyboard,
        _serial: u32,
        modifiers: Modifiers,
        _raw_modifiers: RawModifiers,
        variant: u32,
    ) {
        let keyboard = log_and_return!(
            self.compositor_state
                .seat
                .get_keyboard()
                .ok_or("seat has no keyboard")
        );
        keyboard.with_xkb_state(self, |mut context: XkbContext| {
            context.set_layout(Layout(variant));
        });

        // see linux/input-event-codes.h for keycodes
        let mod_state = keyboard.modifier_state();
        for (new_modifier, current_modifier, keycode) in [
            (
                modifiers.caps_lock,
                mod_state.caps_lock,
                /* KEY_CAPSLOCK */ 58,
            ),
            (
                modifiers.num_lock,
                mod_state.num_lock,
                /* KEY_NUMLOCK */ 69,
            ),
        ] {
            if new_modifier != current_modifier {
                log_and_return!(self.set_key_state(
                    keycode,
                    KeyState::Pressed,
                    SERIAL_COUNTER.next_serial(),
                ));
                log_and_return!(self.set_key_state(
                    keycode,
                    KeyState::Released,
                    SERIAL_COUNTER.next_serial(),
                ));
            }
        }
    }
}

impl PointerHandler for WprsState {
    #[instrument(skip(self, conn, qh, pointer), level = "debug")]
    fn pointer_frame(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        pointer: &WlPointer,
        events: &[PointerEvent],
    ) {
        let compositor_seat = self.compositor_state.seat.clone();
        let compositor_pointer = compositor_seat.get_pointer().unwrap();
        let has_xwayland_enter = events.iter().any(|event| {
            matches!(event.kind, PointerEventKind::Enter { .. })
                && self
                    .surface_bimap
                    .get_by_right(&event.surface.id())
                    .is_some()
        });
        let mut handled_xwayland_event = false;

        for event in events {
            let Some(compositor_surface_id) = self
                .surface_bimap
                .get_by_right(&event.surface.id())
                .cloned()
            else {
                handle_window_frame_pointer_event(self, conn, qh, pointer, event)
                    .log_and_ignore(loc!());
                continue;
            };
            let Some(xwayland_surface) = self.surfaces.get(&compositor_surface_id) else {
                continue;
            };
            let Ok(x11_surface) = xwayland_surface.get_x11_surface().cloned() else {
                continue;
            };
            let has_mapped_children = !xwayland_surface.children.is_empty();
            let surface_loc = x11_surface.geometry().loc;
            let global_loc = pointer_global_location(surface_loc, event.position);
            handled_xwayland_event = true;

            match event.kind {
                PointerEventKind::Enter { serial } => {
                    self.client_state.last_enter_serial = serial;
                    self.client_state.last_pointer_surface = Some(compositor_surface_id);
                    if should_raise_window_on_enter(
                        x11_surface.is_override_redirect(),
                        has_mapped_children,
                    ) && let Some(xwm) = self.compositor_state.xwm.as_mut()
                    {
                        xwm.raise_window(&x11_surface).log_and_ignore(loc!());
                    }
                    let serial = self.compositor_state.serial_map.insert(serial);
                    compositor_pointer.motion(
                        self,
                        Some((x11_surface, surface_loc.to_f64())),
                        &MotionEvent {
                            location: global_loc,
                            serial,
                            time: 0, // unused
                        },
                    );
                },
                PointerEventKind::Leave { serial } => {
                    let serial = self.compositor_state.serial_map.insert(serial);
                    if !has_xwayland_enter && !self.compositor_state.pressed_buttons.is_empty() {
                        let time = self.compositor_state.start_time.elapsed().as_millis() as u32;
                        let buttons: Vec<u32> =
                            self.compositor_state.pressed_buttons.drain().collect();
                        for button in buttons {
                            compositor_pointer.button(
                                self,
                                &ButtonEvent {
                                    time,
                                    button,
                                    serial,
                                    state: ButtonState::Released,
                                },
                            );
                        }
                    }
                    compositor_pointer.motion(
                        self,
                        None,
                        &MotionEvent {
                            location: global_loc,
                            serial,
                            time: 0, // unused
                        },
                    );
                },
                PointerEventKind::Motion { time } => {
                    compositor_pointer.motion(
                        self,
                        Some((x11_surface, surface_loc.to_f64())),
                        &MotionEvent {
                            location: global_loc,
                            serial: 0.into(), // unused
                            time,
                        },
                    );
                },
                PointerEventKind::Press {
                    time,
                    button,
                    serial,
                } => {
                    self.client_state.last_implicit_grab_serial = serial;
                    if let Some(pointer_data) = pointer.data::<PointerData<()>>() {
                        let seat = pointer_data.seat().clone();
                        self.client_state.last_implicit_grab_seat = Some(seat.clone());
                        self.client_state.last_pointer_press = Some((seat, serial));
                    }
                    self.compositor_state.pressed_buttons.insert(button);
                    let serial = self.compositor_state.serial_map.insert(serial);
                    compositor_pointer.button(
                        self,
                        &ButtonEvent {
                            time,
                            button,
                            serial,
                            state: ButtonState::Pressed,
                        },
                    );
                },
                PointerEventKind::Release {
                    time,
                    button,
                    serial,
                } => {
                    self.compositor_state.pressed_buttons.remove(&button);
                    let serial = self.compositor_state.serial_map.insert(serial);
                    compositor_pointer.button(
                        self,
                        &ButtonEvent {
                            time,
                            button,
                            serial,
                            state: ButtonState::Released,
                        },
                    );
                    if !compositor_pointer.is_grabbed()
                        && compositor_pointer.current_focus().as_ref() != Some(&x11_surface)
                    {
                        compositor_pointer.motion(
                            self,
                            Some((x11_surface, surface_loc.to_f64())),
                            &MotionEvent {
                                location: global_loc,
                                serial,
                                time,
                            },
                        );
                    }
                },
                PointerEventKind::Axis {
                    time,
                    horizontal,
                    vertical,
                    source,
                } => x11_surface.axis(
                    &compositor_seat,
                    self,
                    compositor_utils::build_axis_frame(
                        time,
                        horizontal.into(),
                        vertical.into(),
                        source.map(Into::into),
                    ),
                ),
            }
        }
        if handled_xwayland_event {
            compositor_pointer.frame(self);
        }
    }
}

impl ShmHandler for WprsState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.client_state.shm_state
    }
}

fn copy_row_bands_or_full(
    data: BufferPointer<u8>,
    bands: Option<&[RowBand]>,
    stride: usize,
    canvas: &mut [u8],
) {
    if let Some(bands) = bands {
        for band in bands {
            let start = band.y0 as usize * stride;
            let len = (band.y1 - band.y0) as usize * stride;
            let (_, after_start) = data.split_at(start);
            let (band_data, _) = after_start.split_at(len);
            band_data.copy_to_nonoverlapping(&mut canvas[start..start + len]);
        }
    } else {
        data.copy_to_nonoverlapping(canvas);
    }
}

fn write_xwayland_swapchain<P, B>(
    pool: &mut P,
    active_buffer: &mut B,
    back_buffer: &mut Option<(B, Option<Vec<RowBand>>)>,
    cur_bands: Option<&[RowBand]>,
    height: i32,
    is_released: impl FnMut(&mut P, &B) -> bool,
    write_slot: impl FnOnce(&mut P, &B, Option<&[RowBand]>),
    alloc_and_write_full: impl FnOnce(&mut P, bool) -> Result<B>,
) -> Result<()> {
    match advance_two_slot_swapchain(
        pool,
        active_buffer,
        back_buffer,
        cur_bands,
        is_released,
        |prev, cur| combine_band_history(prev, cur, height),
        |cur| cur.map(ToOwned::to_owned),
        alloc_and_write_full,
    )? {
        SwapchainStep::ReusedActive => write_slot(pool, active_buffer, cur_bands),
        SwapchainStep::Swapped { prev_history } => {
            let merged = combine_band_history(prev_history, cur_bands, height);
            write_slot(pool, active_buffer, merged.as_deref());
        },
        SwapchainStep::Allocated => {},
    }
    Ok(())
}

fn create_buffer_with_data(
    metadata: BufferMetadata,
    data: BufferPointer<u8>,
    pool: &mut SlotPool,
) -> Result<Buffer> {
    let (buffer, canvas) = pool
        .create_buffer(
            metadata.width,
            metadata.height,
            metadata.stride,
            metadata.format.into(),
        )
        .location(loc!())?;
    data.copy_to_nonoverlapping(canvas);
    Ok(buffer)
}

#[derive(Debug)]
pub struct XWaylandBuffer {
    pub metadata: BufferMetadata,
    pub active_buffer: Buffer,
    /// Previous swapchain slot and the merged row-bands written to
    /// `active_buffer` since this slot was last current (`None` requires a full
    /// copy on next reuse).
    back_buffer: Option<(Buffer, Option<Vec<RowBand>>)>,
}

impl XWaylandBuffer {
    #[instrument(skip_all, level = "debug")]
    pub fn new(
        metadata: BufferMetadata,
        data: BufferPointer<u8>,
        pool: &mut SlotPool,
    ) -> Result<Self> {
        let active_buffer = create_buffer_with_data(metadata, data, pool).location(loc!())?;

        Ok(Self {
            metadata,
            active_buffer,
            back_buffer: None,
        })
    }

    #[instrument(skip_all, level = "debug")]
    pub fn write_data(
        &mut self,
        data: BufferPointer<u8>,
        damage: &[SerRectangle<i32>],
        pool: &mut SlotPool,
    ) -> Result<()> {
        let metadata = self.metadata;
        let cur_bands =
            filtering::compute_partial_row_bands(damage, metadata.width, metadata.height);

        write_xwayland_swapchain(
            pool,
            &mut self.active_buffer,
            &mut self.back_buffer,
            cur_bands.as_deref(),
            metadata.height,
            |pool, buf| pool.canvas(buf).is_some(),
            |pool, buf, bands| {
                let canvas = pool.canvas(buf).expect("released slot must provide canvas");
                copy_row_bands_or_full(data, bands, metadata.stride as usize, canvas);
            },
            |pool, back_buffer_held| {
                debug!(back_buffer_held, "creating new buffer");
                create_buffer_with_data(metadata, data, pool)
            },
        )
    }
}

impl XWaylandSurface {
    #[instrument(skip(data, damage, pool), level = "debug")]
    pub fn update_buffer(
        &mut self,
        metadata: &BufferData,
        data: BufferPointer<u8>,
        damage: &[SerRectangle<i32>],
        pool: &mut SlotPool,
    ) -> Result<()> {
        let metadata =
            serialization::wayland::BufferMetadata::from_buffer_data(metadata).location(loc!())?;
        if let Some(buffer) = &mut self.buffer
            && buffer.metadata == metadata
        {
            debug!("metadata matched, reusing buffer, {metadata:?}");
            buffer.write_data(data, damage, pool).location(loc!())?;
        } else {
            debug!(
                "creating new buffer, old {:?}, new {:?}",
                self.buffer.as_ref().map(|b| b.metadata),
                metadata
            );
            self.buffer = Some(XWaylandBuffer::new(metadata, data, pool).location(loc!())?);
            self.damage = FrameDamage::Full;
        }

        Ok(())
    }

    pub fn commit(&self) {
        self.wl_surface().commit();
    }

    pub fn frame(&self, qh: &QueueHandle<WprsState>) {
        self.wl_surface()
            .frame(qh, FrameCallbackData(self.wl_surface().clone()));
    }

    pub fn get_role(&self) -> Result<&Role> {
        self.role.as_ref().ok_or(anyhow!("Role was None."))
    }

    pub fn get_mut_role(&mut self) -> Result<&mut Role> {
        self.role.as_mut().ok_or(anyhow!("Role was None."))
    }

    pub fn xdg_toplevel(&mut self) -> Result<&mut XWaylandXdgToplevel> {
        self.get_mut_role()?
            .as_xdg_toplevel_mut()
            .ok_or(anyhow!("Role was not XdgToplevel."))
    }

    pub fn xdg_popup(&mut self) -> Result<&mut XWaylandXdgPopup> {
        self.get_mut_role()?
            .as_xdg_popup_mut()
            .ok_or(anyhow!("Role was not XdgPopup."))
    }
}

#[derive(Debug, EnumAsInner)]
pub enum Role {
    Cursor,
    XdgToplevel(XWaylandXdgToplevel),
    XdgPopup(XWaylandXdgPopup),
    SubSurface(XWaylandSubSurface),
}

#[derive(Debug)]
pub struct XWaylandXdgToplevel {
    pub local_window: Window,
    pub window_frame: FallbackFrame<WprsState>,
    pub frame_offset: Point<i32>,
    pub configured: bool,
    pub decoration_behavior: DecorationBehavior,
    pub requested_decoration_mode: DecorationMode,
    pub last_configure: Option<WindowConfigure>,
}

impl XWaylandXdgToplevel {
    #[instrument(skip(x11_location), ret, level = "debug")]
    pub fn enable_decorations(
        &mut self,
        x11_surface: &X11Surface,
        configure: Option<&WindowConfigure>,
        buffer_metadata: Option<&BufferMetadata>,
        x11_location: impl FnOnce(i32, i32) -> Point<i32>,
    ) -> Result<(i32, i32)> {
        let default_window_size = (
            NonZeroU32::new(DEFAULT_WINDOW_SIZE.0 as u32),
            NonZeroU32::new(DEFAULT_WINDOW_SIZE.1 as u32),
        );
        let window_frame = &mut self.window_frame;
        window_frame.set_hidden(false);
        if let Some(configure) = configure {
            window_frame.update_state(configure.state);
            window_frame.update_wm_capabilities(configure.capabilities);
        }

        // configure.new_size has outer_dimensions, we want width and height to
        // be inner dimensions.
        let (width, height) = match (configure, buffer_metadata) {
            (
                Some(WindowConfigure {
                    new_size: (Some(width), Some(height)),
                    ..
                }),
                _,
            ) => window_frame.subtract_borders(*width, *height),
            (_, Some(buffer_metadata)) => (
                NonZeroU32::new(buffer_metadata.width as u32),
                NonZeroU32::new(buffer_metadata.height as u32),
            ),
            _ => {
                warn!(
                    "Unable to get size from either configure or buffer_metadata, using default size: {:?}",
                    default_window_size
                );
                default_window_size
            },
        };

        // Clamp the size to at least one pixel.
        let width = width.unwrap_or(NonZeroU32::new(1).unwrap());
        let height = height.unwrap_or(NonZeroU32::new(1).unwrap());

        window_frame.resize(width, height);

        // Everything after this wants u32s or i32s.
        let width = width.get();
        let height = height.get();
        let loc = x11_location(width as i32, height as i32);

        // X11's ConfigureNotify wants the outer coordinates but the inner
        // dimensions. And don't worry about border_width. /sigh
        x11_surface
            .configure(Rectangle::new(
                loc.into(),
                (width as i32, height as i32).into(),
            ))
            .location(loc!())?;

        // Top-left corner of frame relative to the inner surface, so x and y
        // will always be negative. -x and -y are thus the coordinates of the
        // inner top-left corner (assuming the outer coordinates are 0).
        let (x, y) = window_frame.location();
        self.frame_offset = (-x, -y).into();

        let (outer_w, outer_h) = window_frame.add_borders(width, height);

        // set_window_geometry wants the "\"visisble bounds\" from the user's
        // perspective", but excluding things like drop shadows, so outer
        // coordinates and dimensions. *However*, the frame subsurfaces are
        // placed at negative offsets from the main buffer and x and y are in
        // surface-local coordinates, so passing them in directly is what we
        // must do.
        self.local_window
            .xdg_surface()
            .set_window_geometry(x, y, outer_w as i32, outer_h as i32);

        Ok((width as i32, height as i32))
    }

    #[instrument(skip_all, level = "debug")]
    pub fn disable_decoration(
        &mut self,
        x11_surface: &X11Surface,
        configure: Option<&WindowConfigure>,
        buffer_metadata: Option<&BufferMetadata>,
        x11_location: impl FnOnce(i32, i32) -> Point<i32>,
    ) -> Result<(i32, i32)> {
        let default_window_size = DEFAULT_WINDOW_SIZE;
        let window_frame = &mut self.window_frame;
        window_frame.set_hidden(true);
        self.frame_offset = (0, 0).into();

        let (width, height) = match (configure, buffer_metadata) {
            (
                Some(WindowConfigure {
                    new_size: (Some(width), Some(height)),
                    ..
                }),
                _,
            ) => (width.get() as i32, height.get() as i32),
            (_, Some(buffer_metadata)) => (buffer_metadata.width, buffer_metadata.height),
            _ => {
                warn!(
                    "Unable to get size from either configure or buffer_metadata, using default size: {:?}",
                    default_window_size
                );
                default_window_size
            },
        };

        let loc = x11_location(width, height);
        x11_surface
            .configure(Rectangle::new(loc.into(), (width, height).into()))
            .location(loc!())?;

        self.local_window
            .xdg_surface()
            .set_window_geometry(0, 0, width, height);

        Ok((width, height))
    }

    pub fn apply_decoration(
        &mut self,
        x11_surface: &X11Surface,
        configure: Option<&WindowConfigure>,
        buffer_metadata: Option<&BufferMetadata>,
        x11_location: impl FnOnce(i32, i32) -> Point<i32>,
    ) -> Result<(i32, i32)> {
        match self.decoration_behavior {
            DecorationBehavior::Auto if let Some(configure) = configure => {
                match configure.decoration_mode {
                    DecorationMode::Server => {
                        // wayland compositor has drawn decorations so it doesn't need ours
                        self.disable_decoration(
                            x11_surface,
                            Some(configure),
                            buffer_metadata,
                            x11_location,
                        )
                    },
                    DecorationMode::Client => {
                        // X11 app drew its own CSD; skip our fallback frame.
                        if x11_surface.is_decorated() {
                            self.disable_decoration(
                                x11_surface,
                                Some(configure),
                                buffer_metadata,
                                x11_location,
                            )
                        } else {
                            self.enable_decorations(
                                x11_surface,
                                Some(configure),
                                buffer_metadata,
                                x11_location,
                            )
                        }
                    },
                }
            },
            DecorationBehavior::AlwaysEnabled => {
                self.enable_decorations(x11_surface, configure, buffer_metadata, x11_location)
            },
            DecorationBehavior::Auto | DecorationBehavior::AlwaysDisabled => {
                self.disable_decoration(x11_surface, configure, buffer_metadata, x11_location)
            },
        }
    }

    /// Requests the appropriate host `xdg-decoration` mode when the desired
    /// mode changed (avoiding redundant requests and configure loops).
    pub fn sync_decoration_mode(&mut self, x11_is_decorated: bool) -> bool {
        let desired = desired_decoration_mode(self.decoration_behavior, x11_is_decorated);
        if self.requested_decoration_mode != desired {
            self.requested_decoration_mode = desired;
            self.local_window.request_decoration_mode(Some(desired));
            true
        } else {
            false
        }
    }

    pub fn sync_size_hints(&self, x11_surface: &X11Surface) {
        let (min_size, max_size) = sanitize_xdg_size_hints(
            x11_surface.min_size().map(|s| (s.w, s.h)),
            x11_surface.max_size().map(|s| (s.w, s.h)),
        );
        self.local_window.set_min_size(min_size);
        self.local_window.set_max_size(max_size);
    }

    pub fn set_role(
        surface: &mut XWaylandSurface,
        xdg_shell_state: &XdgShell,
        shm_state: &Shm,
        subcompositor_state: Arc<SubcompositorState>,
        qh: &QueueHandle<WprsState>,
        decoration_behavior: DecorationBehavior,
    ) -> Result<()> {
        let local_surface = surface.local_surface.take().location(loc!())?;
        let x11_surface = surface.get_x11_surface().location(loc!())?;
        let requested_decoration_mode =
            desired_decoration_mode(decoration_behavior, x11_surface.is_decorated());
        let local_window = xdg_shell_state.create_window(
            local_surface,
            window_decorations_for_mode(requested_decoration_mode),
            qh,
        );

        local_window.set_title(x11_surface.title());
        let class = x11_surface.class();
        if !class.is_empty() {
            local_window.set_app_id(class);
        }

        let (min_size, max_size) = sanitize_xdg_size_hints(
            x11_surface.min_size().map(|s| (s.w, s.h)),
            x11_surface.max_size().map(|s| (s.w, s.h)),
        );
        if min_size.is_some() {
            local_window.set_min_size(min_size);
        }
        if max_size.is_some() {
            local_window.set_max_size(max_size);
        }

        local_window.commit();

        let window_frame =
            FallbackFrame::new(&local_window, shm_state, subcompositor_state, qh.clone())
                .map_err(|e| anyhow!("failed to create client side decorations frame: {e:?}."))
                .location(loc!())?;

        let new_toplevel = Self {
            local_window,
            window_frame,
            frame_offset: (0, 0).into(),
            configured: false,
            decoration_behavior,
            requested_decoration_mode,
            last_configure: None,
        };
        surface.role = Some(Role::XdgToplevel(new_toplevel));
        Ok(())
    }
}

impl WaylandSurface for XWaylandXdgToplevel {
    fn wl_surface(&self) -> &WlSurface {
        self.local_window.wl_surface()
    }
}

impl XdgSurface for XWaylandXdgToplevel {
    fn xdg_surface(&self) -> &SctkXdgSurface {
        self.local_window.xdg_surface()
    }
}

#[derive(Debug)]
pub struct SubSurface {
    pub subsurface: WlSubsurface,
    pub surface: Surface,
}

impl WaylandSurface for SubSurface {
    fn wl_surface(&self) -> &WlSurface {
        self.surface.wl_surface()
    }
}

impl Drop for SubSurface {
    fn drop(&mut self) {
        self.subsurface.destroy();
    }
}

#[derive(Debug)]
pub struct XWaylandSubSurface {
    pub local_subsurface: SubSurface,
    pub parent_surface: WlSurface,
    pub offset: Point<i32>,
    pub frame: Option<FallbackFrame<WprsState>>,
    pub move_active: bool,
    pub move_pointer_location: (f64, f64),
    pub resize_state: Option<SubSurfaceResizeState>,
    pub pending_frame_callback: bool,
    pub buffer_attached: bool,
}

impl XWaylandSubSurface {
    pub(crate) fn set_role(
        surface: &mut XWaylandSurface,
        parent: X11ParentForSubsurface,
        shm_state: &Shm,
        subcompositor_state: Arc<SubcompositorState>,
        qh: &QueueHandle<WprsState>,
    ) -> Result<()> {
        let local_surface = surface.local_surface.take().unwrap();
        let subsurface = subcompositor_state
            .subsurface_from_surface(local_surface.wl_surface(), qh)
            .unwrap();

        let local_subsurface = SubSurface {
            subsurface,
            surface: local_surface,
        };
        local_subsurface.subsurface.set_desync();

        let x11_surface = surface.get_x11_surface().location(loc!())?;
        let geometry = x11_surface.geometry();

        // is_decorated means that the surface is already decorated and does NOT want our decorations.
        let frame = if !x11_surface.is_decorated() && !x11_surface.is_override_redirect() {
            let mut frame = FallbackFrame::new(
                &local_subsurface,
                shm_state,
                subcompositor_state,
                qh.clone(),
            )
            .map_err(|e| anyhow!("failed to create client side decorations frame: {e:?}."))
            .location(loc!())?;

            // not an xdg-shell window, so we can't fullscreen/maximize/etc.
            frame.update_wm_capabilities(WindowManagerCapabilities::empty());

            //we want width and height to be inner dimensions.
            let width = NonZeroU32::new(geometry.size.w.max(1) as u32).unwrap();
            let height = NonZeroU32::new(geometry.size.h.max(1) as u32).unwrap();
            frame.resize(width, height);

            Some(frame)
        } else {
            None
        };

        x11_surface.configure(None).location(loc!())?;

        let new_subsurface = Self {
            local_subsurface,
            parent_surface: parent.surface,
            offset: (-parent.x11_loc.x, -parent.x11_loc.y).into(),
            frame,
            move_active: false,
            move_pointer_location: (0 as f64, 0 as f64),
            resize_state: None,
            pending_frame_callback: false,
            buffer_attached: false,
        };
        surface.role = Some(Role::SubSurface(new_subsurface));

        // the initial commit must include both the subsurface position and the buffer
        // to prevent desyncs
        if let Some(Role::SubSurface(subsurface)) = &mut surface.role {
            subsurface.move_without_commit(geometry.loc.x, geometry.loc.y, qh);
        }

        Ok(())
    }

    pub(crate) fn update_decorations(
        &mut self,
        x11_surface: &X11Surface,
        shm_state: &Shm,
        subcompositor_state: Arc<SubcompositorState>,
        qh: &QueueHandle<WprsState>,
    ) -> Result<()> {
        if x11_surface.is_decorated() || x11_surface.is_override_redirect() {
            self.move_active = false;
            self.resize_state = None;
            if let Some(frame) = self.frame.as_mut() {
                frame.set_hidden(true);
            }
            self.wl_surface().commit();
            return Ok(());
        }

        let geometry = x11_surface.geometry();
        let width = NonZeroU32::new(geometry.size.w.max(1) as u32).unwrap();
        let height = NonZeroU32::new(geometry.size.h.max(1) as u32).unwrap();
        if let Some(frame) = self.frame.as_mut() {
            frame.set_hidden(false);
            frame.resize(width, height);
            frame.draw();
        } else {
            let mut frame = FallbackFrame::new(
                &self.local_subsurface,
                shm_state,
                subcompositor_state,
                qh.clone(),
            )
            .map_err(|e| anyhow!("failed to create client side decorations frame: {e:?}."))
            .location(loc!())?;
            frame.update_wm_capabilities(WindowManagerCapabilities::empty());
            frame.resize(width, height);
            frame.draw();
            self.frame = Some(frame);
        }
        self.wl_surface().commit();
        Ok(())
    }

    pub(crate) fn move_without_commit(&mut self, x: i32, y: i32, qh: &QueueHandle<WprsState>) {
        if !self.pending_frame_callback {
            let local_wl_surface = self.wl_surface();

            self.local_subsurface
                .subsurface
                .set_position(x + self.offset.x, y + self.offset.y);
            local_wl_surface.frame(qh, FrameCallbackData(local_wl_surface.clone()));
            self.parent_surface.commit();

            self.pending_frame_callback = true;
        }
    }

    pub(crate) fn move_(&mut self, x: i32, y: i32, qh: &QueueHandle<WprsState>) {
        if !self.pending_frame_callback {
            self.move_without_commit(x, y, qh);
            self.wl_surface().commit();
        }
    }
}

impl WaylandSurface for XWaylandSubSurface {
    fn wl_surface(&self) -> &WlSurface {
        self.local_subsurface.wl_surface()
    }
}

#[derive(Debug)]
pub struct XWaylandXdgPopup {
    pub local_popup: Popup,
    pub parent: ObjectId,
    pub configured: bool,
    pub last_geometry: Rectangle<i32, Logical>,
    pub reposition_token: u32,
}

impl XWaylandXdgPopup {
    pub(crate) fn set_role(
        surface: &mut XWaylandSurface,
        parent: &X11ParentForPopup,
        xdg_shell_state: &XdgShell,
        qh: &QueueHandle<WprsState>,
    ) -> Result<()> {
        let x11_surface = &surface.get_x11_surface().location(loc!())?;
        let geometry = x11_surface.geometry();
        let positioner = configure_popup_positioner(
            xdg_shell_state,
            geometry,
            parent.x11_loc,
            parent.frame_offset,
        )
        .location(loc!())?;

        x11_surface.configure(None).location(loc!())?;

        let local_popup = Popup::from_surface(
            Some(&parent.xdg_surface),
            &positioner,
            qh,
            surface.local_surface.take().unwrap(),
            xdg_shell_state,
        )
        .location(loc!())?;
        local_popup.wl_surface().commit();

        let new_popup = Self {
            local_popup,
            parent: parent.surface_id.clone(),
            configured: false,
            last_geometry: geometry,
            reposition_token: 0,
        };
        surface.role = Some(Role::XdgPopup(new_popup));
        Ok(())
    }

    pub(crate) fn reposition(
        &mut self,
        xdg_shell_state: &XdgShell,
        geometry: Rectangle<i32, Logical>,
        parent_x11_loc: Point<i32>,
        parent_frame_offset: Point<i32>,
    ) -> Result<()> {
        if geometry == self.last_geometry {
            return Ok(());
        }
        let positioner = configure_popup_positioner(
            xdg_shell_state,
            geometry,
            parent_x11_loc,
            parent_frame_offset,
        )
        .location(loc!())?;
        self.reposition_token = self.reposition_token.wrapping_add(1);
        self.last_geometry = geometry;
        self.local_popup
            .reposition(&positioner, self.reposition_token);
        Ok(())
    }
}

impl WaylandSurface for XWaylandXdgPopup {
    fn wl_surface(&self) -> &WlSurface {
        self.local_popup.wl_surface()
    }
}

impl XdgSurface for XWaylandXdgPopup {
    fn xdg_surface(&self) -> &SctkXdgSurface {
        self.local_popup.xdg_surface()
    }
}

impl DataDeviceHandler for WprsState {
    // Smithay 0.7's `X11Wm` does not implement the Xdnd protocol (`XdndAware`,
    // `XdndEnter`/`Position`/`Status`/`Drop`/`Finished`, `XdndSelection`), so
    // Wayland <-> X11 drag-and-drop bridging is not yet wired here.
    fn enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        data_device: &WlDataDevice,
        _x: f64,
        _y: f64,
        _wl_surface: &WlSurface,
    ) {
        let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter()
            .find(|seat| seat.data_device.inner() == data_device)
        else {
            return;
        };
        if let Some(drag_offer) = seat_obj.data_device.data().drag_offer() {
            debug!(
                "data offer entered x: {:.2} y: {:.2}",
                drag_offer.x, drag_offer.y
            );
        }
    }

    fn leave(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _data_device: &WlDataDevice) {
        debug!("data offer left");
    }

    fn motion(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _data_device: &WlDataDevice,
        _x: f64,
        _y: f64,
    ) {
    }

    #[instrument(skip_all, level = "debug")]
    fn selection(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        data_device: &WlDataDevice,
    ) {
        let Some(seat_obj) = self
            .client_state
            .seat_objects
            .iter()
            .find(|seat| seat.data_device.inner() == data_device)
        else {
            return;
        };
        let Some(offer) = seat_obj.data_device.data().selection_offer() else {
            self.client_state.selection_offer = None;
            if let Some(xwm) = &mut self.compositor_state.xwm {
                xwm.new_selection(SelectionTarget::Clipboard, None)
                    .log_and_ignore(loc!());
            }
            return;
        };
        let mime_types = offer.with_mime_types(<[String]>::to_vec);
        if mime_types.contains(&"_xwayland_xdg_shell_marker".to_string()) {
            return;
        }
        self.client_state.selection_offer = Some(offer);
        if let Some(xwm) = &mut self.compositor_state.xwm {
            xwm.new_selection(SelectionTarget::Clipboard, Some(mime_types))
                .log_and_ignore(loc!());
        }
    }

    fn drop_performed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _data_device: &WlDataDevice,
    ) {
    }
}

impl DataOfferHandler for WprsState {
    fn source_actions(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _offer: &mut DragOffer,
        actions: DndAction,
    ) {
        debug!("Source actions: {actions:?}");
    }

    fn selected_action(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _offer: &mut DragOffer,
        actions: DndAction,
    ) {
        debug!("Selected action: {actions:?}");
    }
}

impl DataSourceHandler for WprsState {
    fn accept_mime(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _source: &WlDataSource,
        mime: Option<String>,
    ) {
        debug!("Source mime type: {mime:?} was accepted");
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn send_request(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &WlDataSource,
        mime: String,
        write_pipe: WritePipe,
    ) {
        if !is_active_source(
            self.client_state.selection_source.as_ref(),
            source,
            CopyPasteSource::inner,
        ) {
            debug!("ignoring send_request for inactive clipboard data source");
            return;
        }
        if let Some(xwm) = &mut self.compositor_state.xwm {
            xwm.send_selection(
                SelectionTarget::Clipboard,
                mime,
                write_pipe.into(),
                self.event_loop_handle.clone(),
            )
            .log_and_ignore(loc!());
        }
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn cancelled(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, source: &WlDataSource) {
        clear_if_active_source(
            &mut self.client_state.selection_source,
            source,
            CopyPasteSource::inner,
        );
    }

    #[instrument(skip_all, level = "debug")]
    fn dnd_dropped(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _source: &WlDataSource) {
    }

    #[instrument(skip_all, level = "debug")]
    fn dnd_finished(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _source: &WlDataSource,
    ) {
    }

    #[instrument(skip_all, level = "debug")]
    fn action(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _source: &WlDataSource,
        _action: DndAction,
    ) {
    }
}

impl PrimarySelectionDeviceHandler for WprsState {
    #[instrument(skip_all, level = "debug")]
    fn selection(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        primary_selection_device: &ZwpPrimarySelectionDeviceV1,
    ) {
        if self.client_state.primary_selection_manager_state.is_none() {
            return;
        }
        let Some(primary_selection_device) = self
            .client_state
            .seat_objects
            .iter()
            .filter_map(|seat| seat.primary_selection_device.as_ref())
            .find(|device| device.inner() == primary_selection_device)
        else {
            return;
        };
        let Some(offer) = primary_selection_device.data().selection_offer() else {
            self.client_state.primary_selection_offer = None;
            if let Some(xwm) = &mut self.compositor_state.xwm {
                xwm.new_selection(SelectionTarget::Primary, None)
                    .log_and_ignore(loc!());
            }
            return;
        };
        let mime_types = offer.with_mime_types(<[String]>::to_vec);
        if mime_types.contains(&"_xwayland_xdg_shell_marker".to_string()) {
            return;
        }
        self.client_state.primary_selection_offer = Some(offer);
        if let Some(xwm) = &mut self.compositor_state.xwm {
            xwm.new_selection(SelectionTarget::Primary, Some(mime_types))
                .log_and_ignore(loc!());
        }
    }
}

impl PrimarySelectionSourceHandler for WprsState {
    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn send_request(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
        mime: String,
        write_pipe: WritePipe,
    ) {
        if !is_active_source(
            self.client_state.primary_selection_source.as_ref(),
            source,
            PrimarySelectionSource::inner,
        ) {
            debug!("ignoring send_request for inactive primary selection source");
            return;
        }
        if let Some(xwm) = &mut self.compositor_state.xwm {
            xwm.send_selection(
                SelectionTarget::Primary,
                mime,
                write_pipe.into(),
                self.event_loop_handle.clone(),
            )
            .log_and_ignore(loc!());
        }
    }

    #[instrument(skip(self, _conn, _qh), level = "debug")]
    fn cancelled(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        source: &ZwpPrimarySelectionSourceV1,
    ) {
        clear_if_active_source(
            &mut self.client_state.primary_selection_source,
            source,
            PrimarySelectionSource::inner,
        );
    }
}

smithay_client_toolkit::delegate_dispatch2!(WprsState);
smithay_client_toolkit::delegate_registry!(WprsState);

impl ProvidesRegistryState for WprsState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.client_state.registry_state
    }
    registry_handlers![OutputState, SeatState,];
}

struct SubCompositorData;

impl Dispatch<WlSubcompositor, SubCompositorData> for WprsState {
    fn event(
        _state: &mut Self,
        _subcompositor: &WlSubcompositor,
        _event: WlSubcompositorEvent,
        _data: &SubCompositorData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        dbg!("SUBCOMPOSITOR DISPATCH");
    }
}

struct SubSurfaceData;

impl Dispatch<WlSubsurface, SubSurfaceData> for WprsState {
    fn event(
        _state: &mut Self,
        _subsurface: &WlSubsurface,
        _event: WlSubsurfaceEvent,
        _data: &SubSurfaceData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        dbg!("SUBSURFACE DISPATCH");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_frame(
        active: &mut usize,
        back: &mut Option<(usize, Option<Vec<RowBand>>)>,
        slots: &mut Vec<Vec<u8>>,
        released: &[bool],
        frame: &[u8],
        cur_bands: Option<Vec<RowBand>>,
        height: i32,
        stride: usize,
    ) -> Option<Option<Vec<RowBand>>> {
        let ptr = frame.as_ptr();
        // SAFETY: `frame` outlives `data` within this call.
        let data = unsafe { BufferPointer::new(&ptr, frame.len()) };
        let mut written = None;
        write_xwayland_swapchain(
            slots,
            active,
            back,
            cur_bands.as_deref(),
            height,
            |_, &slot_idx| released[slot_idx],
            |slots, &slot_idx, bands| {
                assert!(released[slot_idx], "wrote held slot {slot_idx}");
                written = Some(bands.map(ToOwned::to_owned));
                copy_row_bands_or_full(data, bands, stride, &mut slots[slot_idx]);
            },
            |slots, _| {
                let idx = slots.len();
                let mut buf = vec![0u8; frame.len()];
                data.copy_to_nonoverlapping(&mut buf);
                slots.push(buf);
                Ok(idx)
            },
        )
        .unwrap();
        written
    }

    #[test]
    fn test_two_slot_swapchain_ping_pong_accumulation_and_fallback() {
        let width = 8;
        let height = 20;
        let stride = width * 4 + 8; // padded stride (40 bytes/row)
        let byte_len = height as usize * stride;
        let b = |y0, y1| RowBand { y0, y1 };

        let make_frame = |seed: u8| -> Vec<u8> {
            (0..byte_len)
                .map(|i| seed.wrapping_mul(31).wrapping_add(i as u8))
                .collect()
        };

        // Initial full frame in slot 0.
        let mut frame = make_frame(1);
        let mut slots = vec![frame.clone()];
        let mut active = 0usize;
        let mut back: Option<(usize, Option<Vec<RowBand>>)> = None;

        // 1. Compositor holds slot 0 (`released[0] == false`): partial update on
        // rows 2..5 allocates slot 1 via fallback (`back` was `None`) and copies full.
        let f2 = make_frame(2);
        frame[2 * stride..5 * stride].copy_from_slice(&f2[2 * stride..5 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[false],
            &frame,
            Some(vec![b(2, 5)]),
            height,
            stride,
        );
        assert_eq!(written, None);
        assert_eq!(active, 1);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((0, Some(vec![b(2, 5)]))));

        // 2. Ping-pong: compositor now holds slot 1 and releases slot 0.
        // Updating rows 10..13 must copy `2..5 ∪ 10..13` into slot 0 so slot 0
        // matches `frame` without missing frame 2's rows!
        let f3 = make_frame(3);
        frame[10 * stride..13 * stride].copy_from_slice(&f3[10 * stride..13 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[true, false],
            &frame,
            Some(vec![b(10, 13)]),
            height,
            stride,
        );
        assert_eq!(written, Some(Some(vec![b(2, 5), b(10, 13)])));
        assert_eq!(active, 0);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((1, Some(vec![b(10, 13)]))));

        // 3. In-place reuse of `active` (slot 0 released across two frames) must
        // accumulate both frames' bands into `back_bands` for slot 1.
        let f4 = make_frame(4);
        frame[0..2 * stride].copy_from_slice(&f4[0..2 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[true, false],
            &frame,
            Some(vec![b(0, 2)]),
            height,
            stride,
        );
        assert_eq!(written, Some(Some(vec![b(0, 2)])));
        assert_eq!(active, 0);
        assert_eq!(slots[active], frame);

        let f5 = make_frame(5);
        frame[stride..4 * stride].copy_from_slice(&f5[stride..4 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[true, false],
            &frame,
            Some(vec![b(1, 4)]),
            height,
            stride,
        );
        assert_eq!(written, Some(Some(vec![b(1, 4)])));
        assert_eq!(active, 0);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((1, Some(vec![b(0, 4), b(10, 13)]))));

        // Now hold slot 0 and release slot 1: slot 1 catches up `0..4 ∪ 10..13 ∪ 18..20`.
        let f6 = make_frame(6);
        frame[18 * stride..20 * stride].copy_from_slice(&f6[18 * stride..20 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[false, true],
            &frame,
            Some(vec![b(18, 20)]),
            height,
            stride,
        );
        assert_eq!(written, Some(Some(vec![b(0, 4), b(10, 13), b(18, 20)])));
        assert_eq!(active, 1);
        assert_eq!(slots[active], frame);

        // 4. `>= 75%` height promotion (`0..13 ∪ 18..20` covers `15 / 20 == 75%`)
        // promotes `back_bands` to `None`.
        let f7 = make_frame(7);
        frame[0..13 * stride].copy_from_slice(&f7[0..13 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[false, true],
            &frame,
            Some(vec![b(0, 13)]),
            height,
            stride,
        );
        assert_eq!(written, Some(Some(vec![b(0, 13)])));
        assert_eq!(active, 1);
        assert_eq!(back, Some((0, None)));
        assert_eq!(slots[active], frame);

        // 4b. Releasing slot 0 whose history is `None` must full-copy it.
        let f9 = make_frame(9);
        frame[8 * stride..10 * stride].copy_from_slice(&f9[8 * stride..10 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[true, false],
            &frame,
            Some(vec![b(8, 10)]),
            height,
            stride,
        );
        assert_eq!(written, Some(None));
        assert_eq!(active, 0);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((1, Some(vec![b(8, 10)]))));

        // 4c. Full-frame (`cur_bands == None`) in-place write resets back history to `None`.
        frame = make_frame(10);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[true, false],
            &frame,
            None,
            height,
            stride,
        );
        assert_eq!(written, Some(None));
        assert_eq!(active, 0);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((1, None)));

        // 4d. Swapping to slot 1 after the full-frame write must full-copy it.
        let f11 = make_frame(11);
        frame[3 * stride..4 * stride].copy_from_slice(&f11[3 * stride..4 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[false, true],
            &frame,
            Some(vec![b(3, 4)]),
            height,
            stride,
        );
        assert_eq!(written, Some(None));
        assert_eq!(active, 1);
        assert_eq!(slots[active], frame);

        // 5. Both `active` (1) and `back` (0) held -> allocates 3rd slot (2).
        let f8 = make_frame(8);
        frame[5 * stride..7 * stride].copy_from_slice(&f8[5 * stride..7 * stride]);
        let written = apply_frame(
            &mut active,
            &mut back,
            &mut slots,
            &[false, false],
            &frame,
            Some(vec![b(5, 7)]),
            height,
            stride,
        );
        assert_eq!(written, None);
        assert_eq!(active, 2);
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[active], frame);
        assert_eq!(back, Some((1, Some(vec![b(5, 7)]))));
    }

    #[test]
    fn test_popup_coordinate_roundtrip_and_constraints() {
        // Must allow sliding and flipping at screen edges without resizing
        // X11 override-redirect windows.
        assert!(POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::SlideX));
        assert!(POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::SlideY));
        assert!(POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::FlipX));
        assert!(POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::FlipY));
        assert!(!POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::ResizeX));
        assert!(!POPUP_CONSTRAINT_ADJUSTMENT.contains(ConstraintAdjustment::ResizeY));

        // Asymmetric `frame_offset` catches `.x` vs `.y` axis mix-ups.
        for (popup_loc, parent_loc, frame_offset) in [
            ((660, 340), (560, 240), (0, 0)),
            ((660, 340), (560, 240), (4, 30)),
            ((2100, 120), (1920, 0), (8, 36)),
            ((10, 5), (0, 0), (6, 28)),
        ] {
            let anchor =
                popup_anchor_point(popup_loc.into(), parent_loc.into(), frame_offset.into());
            assert_eq!(
                anchor,
                (
                    popup_loc.0 - parent_loc.0 + frame_offset.0,
                    popup_loc.1 - parent_loc.1 + frame_offset.1,
                )
                    .into()
            );
            assert_eq!(
                popup_configure_to_x11_loc(
                    (anchor.x, anchor.y),
                    parent_loc.into(),
                    frame_offset.into(),
                ),
                popup_loc.into()
            );

            // A host slide/flip offset must shift X11 root coordinates by the
            // exact same delta.
            let slid = (anchor.x - 45, anchor.y - 120);
            assert_eq!(
                popup_configure_to_x11_loc(slid, parent_loc.into(), frame_offset.into()),
                (popup_loc.0 - 45, popup_loc.1 - 120).into()
            );
        }
    }

    #[test]
    fn test_pointer_global_location_preserves_click_grab_coordinates() {
        let parent_loc = SmithayPoint::<i32, Logical>::from((560, 240));
        let menu_loc = SmithayPoint::<i32, Logical>::from((660, 340));

        // Right-click on parent at local (100, 100) starts ClickGrab at global (660, 340).
        let press_global = pointer_global_location(parent_loc, (100.0, 100.0));
        assert_eq!(press_global, (660.0, 340.0).into());

        // Drag onto override-redirect menu at menu-local (20, 35) while ClickGrab
        // keeps focus on parent: `global - parent_loc` lands at parent-local (120, 135),
        // which Xwayland maps to X11 root (680, 375) == menu-local (20, 35).
        let menu_global = pointer_global_location(menu_loc, (20.0, 35.0));
        assert_eq!(menu_global, (680.0, 375.0).into());
        let delivered_to_parent = menu_global - parent_loc.to_f64();
        assert_eq!(delivered_to_parent, (120.0, 135.0).into());
        assert_eq!(
            delivered_to_parent + parent_loc.to_f64() - menu_loc.to_f64(),
            (20.0, 35.0).into()
        );
    }

    #[test]
    fn test_should_raise_window_on_enter() {
        assert!(!should_raise_window_on_enter(true, false));
        assert!(!should_raise_window_on_enter(true, true));
        assert!(!should_raise_window_on_enter(false, true));
        assert!(should_raise_window_on_enter(false, false));
    }

    #[test]
    fn test_desired_decoration_mode_and_size_hints() {
        // Auto: CSD X11 window requests Client (avoid double titlebar on SSD
        // compositors); undecorated X11 window requests Server (fallback to
        // FallbackFrame if host configures Client).
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::Auto, true),
            DecorationMode::Client
        );
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::Auto, false),
            DecorationMode::Server
        );
        // Explicit AlwaysEnabled / AlwaysDisabled both request Client so host
        // SSD never stacks on top of FallbackFrame or overrides AlwaysDisabled.
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::AlwaysEnabled, false),
            DecorationMode::Client
        );
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::AlwaysEnabled, true),
            DecorationMode::Client
        );
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::AlwaysDisabled, false),
            DecorationMode::Client
        );
        assert_eq!(
            desired_decoration_mode(DecorationBehavior::AlwaysDisabled, true),
            DecorationMode::Client
        );
        assert_eq!(
            window_decorations_for_mode(DecorationMode::Client),
            WindowDecorations::RequestClient
        );
        assert_eq!(
            window_decorations_for_mode(DecorationMode::Server),
            WindowDecorations::RequestServer
        );

        // Size-hint sanitization: negative -> 0, non-zero max < min -> min,
        // zero max stays unbounded (0).
        assert_eq!(sanitize_xdg_size_hints(None, None), (None, None));
        assert_eq!(
            sanitize_xdg_size_hints(Some((320, 240)), Some((1920, 1080))),
            (Some((320, 240)), Some((1920, 1080)))
        );
        assert_eq!(
            sanitize_xdg_size_hints(Some((-10, 200)), Some((150, 100))),
            (Some((0, 200)), Some((150, 200)))
        );
        assert_eq!(
            sanitize_xdg_size_hints(Some((400, 300)), Some((0, 200))),
            (Some((400, 300)), Some((0, 300)))
        );
    }

    #[test]
    fn test_active_selection_source_guard() {
        let mut active: Option<u32> = None;
        assert!(!is_active_source(active.as_ref(), &1, |v| v));
        assert!(!clear_if_active_source(&mut active, &1, |v| v));

        active = Some(1);
        assert!(is_active_source(active.as_ref(), &1, |v| v));
        assert!(!is_active_source(active.as_ref(), &2, |v| v));

        // Superseded source 1 arriving late must not clobber active source 2.
        active = Some(2);
        assert!(!is_active_source(active.as_ref(), &1, |v| v));
        assert!(!clear_if_active_source(&mut active, &1, |v| v));
        assert_eq!(active, Some(2));

        // Cancelling the active source clears it.
        assert!(clear_if_active_source(&mut active, &2, |v| v));
        assert_eq!(active, None);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_xdg_decoration_move_resize_and_selection_wire() {
        use std::os::fd::OwnedFd;
        use std::os::unix::net::UnixStream;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::Ordering;

        use smithay::input::Seat as ServerSeat;
        use smithay::input::SeatHandler as ServerSeatHandler;
        use smithay::input::SeatState as ServerSeatState;
        use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as ServerDecoMode;
        use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::ResizeEdge as ServerResizeEdge;
        use smithay::reexports::wayland_server::Client as ServerClient;
        use smithay::reexports::wayland_server::Display;
        use smithay::reexports::wayland_server::backend::ClientData;
        use smithay::reexports::wayland_server::backend::ClientId;
        use smithay::reexports::wayland_server::backend::DisconnectReason;
        use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat as ServerWlSeat;
        use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface as ServerWlSurface;
        use smithay::utils::Serial as ServerSerial;
        use smithay::wayland::compositor::CompositorClientState;
        use smithay::wayland::compositor::CompositorHandler as ServerCompositorHandler;
        use smithay::wayland::compositor::CompositorState as ServerCompositorState;
        use smithay::wayland::selection::SelectionHandler as ServerSelectionHandler;
        use smithay::wayland::selection::SelectionSource;
        use smithay::wayland::selection::data_device::ClientDndGrabHandler;
        use smithay::wayland::selection::data_device::DataDeviceHandler as ServerDataDeviceHandler;
        use smithay::wayland::selection::data_device::DataDeviceState as ServerDataDeviceState;
        use smithay::wayland::selection::data_device::ServerDndGrabHandler;
        use smithay::wayland::selection::primary_selection::PrimarySelectionHandler as ServerPrimarySelectionHandler;
        use smithay::wayland::selection::primary_selection::PrimarySelectionState as ServerPrimarySelectionState;
        use smithay::wayland::shell::xdg::PopupSurface;
        use smithay::wayland::shell::xdg::PositionerState;
        use smithay::wayland::shell::xdg::ToplevelSurface;
        use smithay::wayland::shell::xdg::XdgShellHandler;
        use smithay::wayland::shell::xdg::XdgShellState;
        use smithay::wayland::shell::xdg::decoration::XdgDecorationHandler;
        use smithay::wayland::shell::xdg::decoration::XdgDecorationState;
        use smithay::xwayland::xwm::ResizeEdge as X11ResizeEdge;
        use smithay_client_toolkit::data_device_manager::data_device::DataDevice;
        use smithay_client_toolkit::primary_selection::device::PrimarySelectionDevice;
        use smithay_client_toolkit::reexports::client::EventQueue;
        use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
        use smithay_client_toolkit::reexports::client::protocol::wl_callback;

        use crate::xwayland_xdg_shell::xwayland::x11_resize_edge_to_xdg;

        #[derive(Default)]
        struct TestClientData {
            compositor_state: CompositorClientState,
        }

        impl ClientData for TestClientData {
            fn initialized(&self, _client_id: ClientId) {}
            fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
        }

        struct TestHostServer {
            compositor_state: ServerCompositorState,
            xdg_shell_state: XdgShellState,
            _xdg_decoration_state: XdgDecorationState,
            seat_state: ServerSeatState<Self>,
            data_device_state: ServerDataDeviceState,
            primary_selection_state: ServerPrimarySelectionState,
            deco_requests: Vec<Option<ServerDecoMode>>,
            move_requests: Vec<u32>,
            resize_requests: Vec<(u32, ServerResizeEdge)>,
            minimize_count: usize,
            selection_events: Vec<(SelectionTarget, bool)>,
            first_toplevel_surface: Option<ServerWlSurface>,
        }

        impl ServerCompositorHandler for TestHostServer {
            fn compositor_state(&mut self) -> &mut ServerCompositorState {
                &mut self.compositor_state
            }
            fn client_compositor_state<'a>(
                &self,
                client: &'a ServerClient,
            ) -> &'a CompositorClientState {
                &client
                    .get_data::<TestClientData>()
                    .unwrap()
                    .compositor_state
            }
            fn commit(&mut self, _surface: &ServerWlSurface) {}
        }

        impl XdgShellHandler for TestHostServer {
            fn xdg_shell_state(&mut self) -> &mut XdgShellState {
                &mut self.xdg_shell_state
            }
            fn new_toplevel(&mut self, surface: ToplevelSurface) {
                if self.first_toplevel_surface.is_none() {
                    self.first_toplevel_surface = Some(surface.wl_surface().clone());
                }
                surface.send_configure();
            }
            fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {}
            fn grab(&mut self, _surface: PopupSurface, _seat: ServerWlSeat, _serial: ServerSerial) {
            }
            fn reposition_request(
                &mut self,
                _surface: PopupSurface,
                _positioner: PositionerState,
                _token: u32,
            ) {
            }
            fn move_request(
                &mut self,
                _surface: ToplevelSurface,
                _seat: ServerWlSeat,
                serial: ServerSerial,
            ) {
                self.move_requests.push(u32::from(serial));
            }
            fn resize_request(
                &mut self,
                _surface: ToplevelSurface,
                _seat: ServerWlSeat,
                serial: ServerSerial,
                edges: ServerResizeEdge,
            ) {
                self.resize_requests.push((u32::from(serial), edges));
            }
            fn minimize_request(&mut self, _surface: ToplevelSurface) {
                self.minimize_count += 1;
            }
        }

        impl XdgDecorationHandler for TestHostServer {
            fn new_decoration(&mut self, _toplevel: ToplevelSurface) {}
            fn request_mode(&mut self, toplevel: ToplevelSurface, mode: ServerDecoMode) {
                self.deco_requests.push(Some(mode));
                toplevel.with_pending_state(|state| {
                    state.decoration_mode = Some(mode);
                });
                toplevel.send_configure();
            }
            fn unset_mode(&mut self, _toplevel: ToplevelSurface) {
                self.deco_requests.push(None);
            }
        }

        impl ServerSeatHandler for TestHostServer {
            type KeyboardFocus = ServerWlSurface;
            type PointerFocus = ServerWlSurface;
            type TouchFocus = ServerWlSurface;
            fn seat_state(&mut self) -> &mut ServerSeatState<Self> {
                &mut self.seat_state
            }
        }

        impl ServerSelectionHandler for TestHostServer {
            type SelectionUserData = ();
            fn new_selection(
                &mut self,
                ty: SelectionTarget,
                source: Option<SelectionSource>,
                _seat: ServerSeat<Self>,
            ) {
                self.selection_events.push((ty, source.is_some()));
            }
            fn send_selection(
                &mut self,
                _ty: SelectionTarget,
                _mime_type: String,
                _fd: OwnedFd,
                _seat: ServerSeat<Self>,
                _user_data: &Self::SelectionUserData,
            ) {
            }
        }

        impl ServerDataDeviceHandler for TestHostServer {
            fn data_device_state(&self) -> &ServerDataDeviceState {
                &self.data_device_state
            }
        }
        impl ClientDndGrabHandler for TestHostServer {}
        impl ServerDndGrabHandler for TestHostServer {}

        impl ServerPrimarySelectionHandler for TestHostServer {
            fn primary_selection_state(&self) -> &ServerPrimarySelectionState {
                &self.primary_selection_state
            }
        }

        smithay::delegate_compositor!(TestHostServer);
        smithay::delegate_xdg_shell!(TestHostServer);
        smithay::delegate_xdg_decoration!(TestHostServer);
        smithay::delegate_seat!(TestHostServer);
        smithay::delegate_data_device!(TestHostServer);
        smithay::delegate_primary_selection!(TestHostServer);

        struct TestClient {
            registry_state: RegistryState,
            output_state: OutputState,
            seat_state: SeatState,
            data_device_manager: DataDeviceManagerState,
            primary_selection_manager: PrimarySelectionManagerState,
            seat: Option<WlSeat>,
            data_device: Option<DataDevice>,
            primary_device: Option<PrimarySelectionDevice>,
            active_clipboard: Option<CopyPasteSource>,
            active_primary: Option<PrimarySelectionSource>,
            cancelled_clipboard: usize,
            cancelled_primary: usize,
            configured_modes: Vec<DecorationMode>,
        }

        impl OutputHandler for TestClient {
            fn output_state(&mut self) -> &mut OutputState {
                &mut self.output_state
            }
            fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}
            fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}
            fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlOutput) {}
        }

        impl CompositorHandler for TestClient {
            fn scale_factor_changed(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlSurface,
                _: i32,
            ) {
            }
            fn transform_changed(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlSurface,
                _: Transform,
            ) {
            }
            fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: u32) {}
            fn surface_enter(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlSurface,
                _: &WlOutput,
            ) {
            }
            fn surface_leave(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlSurface,
                _: &WlOutput,
            ) {
            }
        }

        impl WindowHandler for TestClient {
            fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {}
            fn configure(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &Window,
                configure: WindowConfigure,
                _: u32,
            ) {
                self.configured_modes.push(configure.decoration_mode);
            }
        }

        impl SeatHandler for TestClient {
            fn seat_state(&mut self) -> &mut SeatState {
                &mut self.seat_state
            }
            fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlSeat) {}
            fn new_capability(
                &mut self,
                _: &Connection,
                qh: &QueueHandle<Self>,
                seat: WlSeat,
                _: Capability,
            ) {
                if self.seat.is_none() {
                    self.data_device = Some(self.data_device_manager.get_data_device(qh, &seat));
                    self.primary_device = Some(
                        self.primary_selection_manager
                            .get_selection_device(qh, &seat),
                    );
                    self.seat = Some(seat);
                }
            }
            fn remove_capability(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: WlSeat,
                _: Capability,
            ) {
            }
            fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlSeat) {}
        }

        impl DataDeviceHandler for TestClient {
            fn enter(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlDataDevice,
                _: f64,
                _: f64,
                _: &WlSurface,
            ) {
            }
            fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
            fn motion(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlDataDevice,
                _: f64,
                _: f64,
            ) {
            }
            fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
            fn drop_performed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
        }

        impl DataOfferHandler for TestClient {
            fn source_actions(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &mut DragOffer,
                _: DndAction,
            ) {
            }
            fn selected_action(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &mut DragOffer,
                _: DndAction,
            ) {
            }
        }

        impl DataSourceHandler for TestClient {
            fn accept_mime(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlDataSource,
                _: Option<String>,
            ) {
            }
            fn send_request(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlDataSource,
                _: String,
                _: WritePipe,
            ) {
            }
            fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
                self.cancelled_clipboard += 1;
                clear_if_active_source(&mut self.active_clipboard, source, CopyPasteSource::inner);
            }
            fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
            fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
            fn action(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &WlDataSource,
                _: DndAction,
            ) {
            }
        }

        impl PrimarySelectionDeviceHandler for TestClient {
            fn selection(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &ZwpPrimarySelectionDeviceV1,
            ) {
            }
        }

        impl PrimarySelectionSourceHandler for TestClient {
            fn send_request(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                _: &ZwpPrimarySelectionSourceV1,
                _: String,
                _: WritePipe,
            ) {
            }
            fn cancelled(
                &mut self,
                _: &Connection,
                _: &QueueHandle<Self>,
                source: &ZwpPrimarySelectionSourceV1,
            ) {
                self.cancelled_primary += 1;
                clear_if_active_source(
                    &mut self.active_primary,
                    source,
                    PrimarySelectionSource::inner,
                );
            }
        }

        impl ProvidesRegistryState for TestClient {
            fn registry(&mut self) -> &mut RegistryState {
                &mut self.registry_state
            }
            registry_handlers![OutputState, SeatState];
        }

        impl Dispatch<wl_callback::WlCallback, Arc<AtomicBool>> for TestClient {
            fn event(
                _state: &mut Self,
                _proxy: &wl_callback::WlCallback,
                event: wl_callback::Event,
                done: &Arc<AtomicBool>,
                _conn: &Connection,
                _qh: &QueueHandle<Self>,
            ) {
                if matches!(event, wl_callback::Event::Done { .. }) {
                    done.store(true, Ordering::Release);
                }
            }
        }

        smithay_client_toolkit::delegate_dispatch2!(TestClient);
        smithay_client_toolkit::delegate_registry!(TestClient);

        let mut display: Display<TestHostServer> = Display::new().unwrap();
        let mut dh = display.handle();
        let mut seat_state = ServerSeatState::new();
        let mut server_seat = seat_state.new_wl_seat(&dh, "seat0");
        let server_kb = server_seat
            .add_keyboard(Default::default(), 200, 25)
            .unwrap();
        let _ptr = server_seat.add_pointer();

        let mut server = TestHostServer {
            compositor_state: ServerCompositorState::new::<TestHostServer>(&dh),
            xdg_shell_state: XdgShellState::new::<TestHostServer>(&dh),
            _xdg_decoration_state: XdgDecorationState::new::<TestHostServer>(&dh),
            seat_state,
            data_device_state: ServerDataDeviceState::new::<TestHostServer>(&dh),
            primary_selection_state: ServerPrimarySelectionState::new::<TestHostServer>(&dh),
            deco_requests: Vec::new(),
            move_requests: Vec::new(),
            resize_requests: Vec::new(),
            minimize_count: 0,
            selection_events: Vec::new(),
            first_toplevel_surface: None,
        };

        let (server_sock, client_sock) = UnixStream::pair().unwrap();
        let server_client = dh
            .insert_client(server_sock, Arc::new(TestClientData::default()))
            .unwrap();
        // Give the client data-device & primary-selection focus so set_selection succeeds.
        data_device::set_data_device_focus(&dh, &server_seat, Some(server_client.clone()));
        primary_selection::set_primary_focus(&dh, &server_seat, Some(server_client));

        let conn = Connection::from_socket(client_sock).unwrap();
        // Dispatch server in a scoped thread just for `registry_queue_init`,
        // then drive both sides synchronously on the test thread.
        let (globals, mut queue): (_, EventQueue<TestClient>) = std::thread::scope(|s| {
            let h = s.spawn(|| registry_queue_init::<TestClient>(&conn).unwrap());
            loop {
                display.dispatch_clients(&mut server).unwrap();
                display.flush_clients().unwrap();
                if h.is_finished() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            h.join().unwrap()
        });
        let qh = queue.handle();

        let compositor = CompositorState::bind(&globals, &qh).unwrap();
        let xdg_shell = XdgShell::bind(&globals, &qh).unwrap();
        let mut client = TestClient {
            registry_state: RegistryState::new(&globals),
            output_state: OutputState::new(&globals, &qh),
            seat_state: SeatState::new(&globals, &qh),
            data_device_manager: DataDeviceManagerState::bind(&globals, &qh).unwrap(),
            primary_selection_manager: PrimarySelectionManagerState::bind(&globals, &qh).unwrap(),
            seat: None,
            data_device: None,
            primary_device: None,
            active_clipboard: None,
            active_primary: None,
            cancelled_clipboard: 0,
            cancelled_primary: 0,
            configured_modes: Vec::new(),
        };

        let roundtrip = |display: &mut Display<TestHostServer>,
                         server: &mut TestHostServer,
                         conn: &Connection,
                         queue: &mut EventQueue<TestClient>,
                         client: &mut TestClient| {
            let done = Arc::new(AtomicBool::new(false));
            let _cb = conn.display().sync(&queue.handle(), Arc::clone(&done));
            conn.flush().unwrap();
            display.dispatch_clients(server).unwrap();
            display.flush_clients().unwrap();
            while !done.load(Ordering::Acquire) {
                queue.blocking_dispatch(client).unwrap();
            }
        };

        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        let seat = client.seat.clone().expect("seat advertised");

        // 1. Decoration mode requests: Auto+CSD -> ClientSide, Auto+SSD -> ServerSide,
        //    dynamic toggle ServerSide -> ClientSide.
        let csd_surf = compositor.create_surface(&qh);
        let csd_win = xdg_shell.create_window(
            csd_surf,
            window_decorations_for_mode(desired_decoration_mode(DecorationBehavior::Auto, true)),
            &qh,
        );
        csd_win.commit();

        let ssd_surf = compositor.create_surface(&qh);
        let ssd_win = xdg_shell.create_window(
            ssd_surf,
            window_decorations_for_mode(desired_decoration_mode(DecorationBehavior::Auto, false)),
            &qh,
        );
        ssd_win.commit();
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        assert_eq!(
            server.deco_requests,
            vec![
                Some(ServerDecoMode::ClientSide),
                Some(ServerDecoMode::ServerSide),
            ]
        );

        ssd_win.request_decoration_mode(Some(DecorationMode::Client));
        ssd_win.commit();
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        assert_eq!(
            server.deco_requests,
            vec![
                Some(ServerDecoMode::ClientSide),
                Some(ServerDecoMode::ServerSide),
                Some(ServerDecoMode::ClientSide),
            ]
        );

        // 2. Interactive move, minimize, and resize for all 8 X11ResizeEdge variants.
        csd_win.move_(&seat, 101);
        csd_win.set_minimized();
        let all_edges = [
            (X11ResizeEdge::Top, ServerResizeEdge::Top),
            (X11ResizeEdge::Bottom, ServerResizeEdge::Bottom),
            (X11ResizeEdge::Left, ServerResizeEdge::Left),
            (X11ResizeEdge::Right, ServerResizeEdge::Right),
            (X11ResizeEdge::TopLeft, ServerResizeEdge::TopLeft),
            (X11ResizeEdge::TopRight, ServerResizeEdge::TopRight),
            (X11ResizeEdge::BottomLeft, ServerResizeEdge::BottomLeft),
            (X11ResizeEdge::BottomRight, ServerResizeEdge::BottomRight),
        ];
        for (idx, &(x11_edge, _)) in all_edges.iter().enumerate() {
            csd_win.resize(&seat, 200 + idx as u32, x11_resize_edge_to_xdg(x11_edge));
        }
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        assert_eq!(server.move_requests, vec![101]);
        assert_eq!(server.minimize_count, 1);
        let expected_resizes: Vec<(u32, ServerResizeEdge)> = all_edges
            .iter()
            .enumerate()
            .map(|(idx, &(_, srv_edge))| (200 + idx as u32, srv_edge))
            .collect();
        assert_eq!(server.resize_requests, expected_resizes);

        // 3. Clipboard & Primary selection: superseding source does NOT clear
        //    the newer active source on stale `cancelled`, and `unset_selection`
        //    clears the compositor selection.
        let focus_surf = server
            .first_toplevel_surface
            .clone()
            .expect("toplevel surface created");
        server_kb.set_focus(&mut server, Some(focus_surf), SERIAL_COUNTER.next_serial());
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        let clip_s1 = client
            .data_device_manager
            .create_copy_paste_source(&qh, ["text/plain"]);
        clip_s1.set_selection(client.data_device.as_ref().unwrap(), 301);
        client.active_clipboard = Some(clip_s1);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        let clip_s2 = client
            .data_device_manager
            .create_copy_paste_source(&qh, ["text/plain;charset=utf-8"]);
        clip_s2.set_selection(client.data_device.as_ref().unwrap(), 302);
        // Keep `old_clip_s1` alive long enough to receive its `cancelled` event on wire.
        let old_clip_s1 = client.active_clipboard.replace(clip_s2);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        assert_eq!(client.cancelled_clipboard, 1);
        assert!(
            client.active_clipboard.is_some(),
            "stale cancelled event must not clear active clipboard source"
        );
        drop(old_clip_s1);

        // Now clear selection (matching `XwmHandler::cleared_selection`).
        if let Some(source) = client.active_clipboard.take() {
            client.data_device.as_ref().unwrap().unset_selection(303);
            drop(source);
        }
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        // Same wire check for Primary selection.
        let prim_s1 = client
            .primary_selection_manager
            .create_selection_source(&qh, ["text/plain"]);
        prim_s1.set_selection(client.primary_device.as_ref().unwrap(), 401);
        client.active_primary = Some(prim_s1);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        let prim_s2 = client
            .primary_selection_manager
            .create_selection_source(&qh, ["text/plain;charset=utf-8"]);
        prim_s2.set_selection(client.primary_device.as_ref().unwrap(), 402);
        let old_prim_s1 = client.active_primary.replace(prim_s2);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        assert_eq!(client.cancelled_primary, 1);
        assert!(
            client.active_primary.is_some(),
            "stale cancelled event must not clear active primary source"
        );
        drop(old_prim_s1);

        if let Some(source) = client.active_primary.take() {
            client.primary_device.as_ref().unwrap().unset_selection(403);
            drop(source);
        }
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        assert_eq!(
            server.selection_events,
            vec![
                (SelectionTarget::Clipboard, true),
                (SelectionTarget::Clipboard, true),
                (SelectionTarget::Clipboard, false),
                (SelectionTarget::Primary, true),
                (SelectionTarget::Primary, true),
                (SelectionTarget::Primary, false),
            ]
        );
    }
}
