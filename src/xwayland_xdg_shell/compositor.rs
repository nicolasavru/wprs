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
use std::collections::hash_map::Entry;
use std::ffi::OsStr;
use std::os::fd::OwnedFd;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

use calloop::RegistrationToken;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::input::Seat;
use smithay::input::SeatHandler;
use smithay::input::SeatState;
use smithay::input::pointer::CursorImageStatus;
use smithay::input::pointer::CursorImageSurfaceData;
use smithay::output::Output;
use smithay::output::PhysicalProperties;
use smithay::reexports::calloop::LoopHandle;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::IsAlive;
use smithay::utils::Logical;
use smithay::utils::Rectangle;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor;
use smithay::wayland::compositor::BufferAssignment;
use smithay::wayland::compositor::CompositorClientState;
use smithay::wayland::compositor::CompositorHandler;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::output::OutputHandler;
use smithay::wayland::output::OutputManagerState;
use smithay::wayland::output::WlOutputData;
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::SelectionSource;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::data_device::ClientDndGrabHandler;
use smithay::wayland::selection::data_device::DataDeviceHandler;
use smithay::wayland::selection::data_device::DataDeviceState;
use smithay::wayland::selection::data_device::ServerDndGrabHandler;
use smithay::wayland::selection::primary_selection::PrimarySelectionHandler;
use smithay::wayland::selection::primary_selection::PrimarySelectionState;
use smithay::wayland::shm::ShmHandler;
use smithay::wayland::shm::ShmState;
use smithay::wayland::xwayland_shell::XWaylandShellHandler;
use smithay::wayland::xwayland_shell::XWaylandShellState;
use smithay::xwayland::X11Surface;
use smithay::xwayland::X11Wm;
use smithay::xwayland::XWayland;
use smithay::xwayland::XWaylandClientData;
use smithay::xwayland::XWaylandEvent;
use smithay::xwayland::xwm::XwmId;
use smithay_client_toolkit::reexports::client::protocol::wl_output::Transform as SctkTransform;
use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as SctkWlSurface;
use smithay_client_toolkit::reexports::csd_frame::DecorationsFrame;
use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_surface;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::xdg::XdgSurface;

use crate::client_utils::FrameDamage;
use crate::compositor_utils;
use crate::prelude::*;
use crate::serialization::geometry::Point;
use crate::serialization::wayland::OutputInfo;
use crate::serialization::wayland::Transform as SerTransform;
use crate::utils::SerialMap;
use crate::xwayland_xdg_shell::WprsState;
use crate::xwayland_xdg_shell::XWaylandSurface;
use crate::xwayland_xdg_shell::client::Role;
use crate::xwayland_xdg_shell::wmname;

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Deserialize, Serialize)]
pub enum DecorationBehavior {
    #[default]
    Auto,
    AlwaysEnabled,
    AlwaysDisabled,
}

pub struct XwaylandOptions<K, V, I>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    pub display: Option<u32>,
    pub env: I,
}

/// Returns an output's logical bounding rectangle matching what Smithay
/// advertises over `zxdg_output_v1` (position plus oriented scaled size).
pub(crate) fn output_logical_rect(output: &Output) -> Option<Rectangle<i32, Logical>> {
    let mode = output.current_mode()?;
    let logical_size = output.current_transform().transform_size(
        mode.size
            .to_f64()
            .to_logical(output.current_scale().fractional_scale())
            .to_i32_round(),
    );
    Some(Rectangle::new(output.current_location(), logical_size))
}

/// Computes the X11 virtual screen origin shift needed to keep all outputs at
/// non-negative X11 root coordinates.
fn x11_screen_origin(outputs: &[(u32, Rectangle<i32, Logical>)]) -> (i32, i32) {
    let min_x = outputs
        .iter()
        .map(|(_, r)| r.loc.x)
        .min()
        .unwrap_or(0)
        .min(0);
    let min_y = outputs
        .iter()
        .map(|(_, r)| r.loc.y)
        .min()
        .unwrap_or(0)
        .min(0);
    (min_x, min_y)
}

/// Computes an X11 root-coordinate top-left placement for a toplevel window of
/// `window_size` on the real output layout without inflating screen dimensions.
///
/// Maximized/fullscreen or screen-sized windows land at the target output's
/// origin; smaller windows are centered inside the target output so popups and
/// pointer drags have real positive screen margin on every side.
pub(crate) fn compute_toplevel_x11_location(
    outputs: &[(u32, Rectangle<i32, Logical>)],
    surface_output_ids: &HashSet<u32>,
    window_size: (i32, i32),
    maximized_or_fullscreen: bool,
) -> Point<i32> {
    if outputs.is_empty() {
        return (0, 0).into();
    }
    let (min_x, min_y) = x11_screen_origin(outputs);
    let target = outputs
        .iter()
        .filter(|(id, _)| surface_output_ids.contains(id))
        .min_by_key(|(id, r)| (r.loc.x, r.loc.y, *id))
        .or_else(|| outputs.iter().min_by_key(|(id, r)| (r.loc.x, r.loc.y, *id)))
        .unwrap()
        .1;

    let origin_x = target.loc.x - min_x;
    let origin_y = target.loc.y - min_y;
    if maximized_or_fullscreen {
        return (origin_x, origin_y).into();
    }

    let x = origin_x + (target.size.w - window_size.0).max(0) / 2;
    let y = origin_y + (target.size.h - window_size.1).max(0) / 2;
    (x, y).into()
}

/// Returns true when a configured toplevel's X11 geometry center no longer
/// lies inside any of the outputs its Wayland surface currently occupies.
pub(crate) fn toplevel_needs_output_reposition(
    outputs: &[(u32, Rectangle<i32, Logical>)],
    surface_output_ids: &HashSet<u32>,
    current_geometry: Rectangle<i32, Logical>,
) -> bool {
    if outputs.is_empty() || surface_output_ids.is_empty() {
        return false;
    }
    let (min_x, min_y) = x11_screen_origin(outputs);
    let center_x = current_geometry.loc.x + current_geometry.size.w / 2;
    let center_y = current_geometry.loc.y + current_geometry.size.h / 2;
    let mut matched_any = false;
    for (id, rect) in outputs {
        if !surface_output_ids.contains(id) {
            continue;
        }
        matched_any = true;
        let x0 = rect.loc.x - min_x;
        let y0 = rect.loc.y - min_y;
        if center_x >= x0
            && center_x < x0 + rect.size.w
            && center_y >= y0
            && center_y < y0 + rect.size.h
        {
            return false;
        }
    }
    matched_any
}

fn create_or_update_output<D>(
    outputs: &mut HashMap<u32, (Output, GlobalId)>,
    dh: &DisplayHandle,
    output: OutputInfo,
) -> Output
where
    D: smithay::reexports::wayland_server::GlobalDispatch<
            smithay::reexports::wayland_server::protocol::wl_output::WlOutput,
            WlOutputData,
        > + 'static,
{
    match outputs.entry(output.id) {
        Entry::Occupied(mut entry) => {
            let local_output = &mut entry.get_mut().0;
            compositor_utils::update_output(local_output, output);
            local_output.clone()
        },
        Entry::Vacant(entry) => {
            let mut new_output = Output::new(
                format!("{}_{}", output.id, output.name.as_deref().unwrap_or("None")),
                PhysicalProperties {
                    size: output.physical_size.into(),
                    subpixel: output.subpixel.into(),
                    make: output.make.clone(),
                    model: output.model.clone(),
                },
            );
            compositor_utils::update_output(&mut new_output, output);
            let global_id = new_output.create_global::<D>(dh);
            entry.insert((new_output.clone(), global_id));
            new_output
        },
    }
}

#[derive(Debug)]
pub struct WprsCompositorState {
    pub dh: DisplayHandle,
    pub compositor_state: CompositorState,
    pub start_time: Instant,
    pub shm_state: ShmState,
    pub seat_state: SeatState<WprsState>,
    pub output_manager_state: OutputManagerState,
    pub data_device_state: DataDeviceState,
    pub xwayland_shell_state: XWaylandShellState,
    pub primary_selection_state: PrimarySelectionState,
    pub decoration_behavior: DecorationBehavior,

    pub seat: Seat<WprsState>,

    pub outputs: HashMap<u32, (Output, GlobalId)>,
    pub(crate) serial_map: SerialMap,
    pub(crate) pressed_keys: HashSet<u32>,
    pub(crate) pressed_buttons: HashSet<u32>,

    pub xwm: Option<X11Wm>,

    /// unpaired x11 surfaces
    pub x11_surfaces: Vec<X11Surface>,
}

impl WprsCompositorState {
    /// # Panics
    /// On failure launching xwayland.
    pub fn new<K, V, I>(
        dh: DisplayHandle,
        event_loop_handle: &LoopHandle<'static, WprsState>,
        decoration_behavior: DecorationBehavior,
        xwayland_options: XwaylandOptions<K, V, I>,
        registration_tokens: &mut Vec<RegistrationToken>,
    ) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let mut seat_state = SeatState::new();
        let seat = seat_state.new_wl_seat(&dh, "wprs");

        let (xwayland, client) = XWayland::spawn(
            &dh,
            xwayland_options.display,
            xwayland_options.env,
            false,
            Stdio::inherit(),
            Stdio::inherit(),
            |_| {},
        )
        .expect("failed to start xwayland.");

        let ret = event_loop_handle.insert_source(xwayland, move |event, _, data| match event {
            XWaylandEvent::Ready {
                x11_socket,
                display_number,
            } => {
                let wm =
                    X11Wm::start_wm(data.event_loop_handle.clone(), x11_socket, client.clone())
                        .expect("Failed to attach X11 Window Manager.");

                // Oh Java...
                wmname::set_wmname(Some(&format!(":{display_number}")), "LG3D")
                    .expect("Failed to set WM name.");

                data.compositor_state.xwm = Some(wm);
            },
            XWaylandEvent::Error => {
                let _ = data.compositor_state.xwm.take();
            },
        });

        match ret {
            Ok(token) => {
                registration_tokens.push(token);
            },
            Err(e) => {
                error!(
                    "Failed to insert the XWaylandSource into the event loop: {}",
                    e
                );
            },
        }

        Self {
            dh: dh.clone(),
            compositor_state: CompositorState::new::<WprsState>(&dh),
            start_time: Instant::now(),
            shm_state: ShmState::new::<WprsState>(&dh, Vec::new()),
            seat_state,
            output_manager_state: OutputManagerState::new_with_xdg_output::<WprsState>(&dh),
            xwayland_shell_state: XWaylandShellState::new::<WprsState>(&dh),
            data_device_state: DataDeviceState::new::<WprsState>(&dh),
            primary_selection_state: PrimarySelectionState::new::<WprsState>(&dh),
            decoration_behavior,
            seat,
            outputs: HashMap::new(),
            serial_map: SerialMap::new(),
            pressed_keys: HashSet::new(),
            pressed_buttons: HashSet::new(),
            xwm: None,
            x11_surfaces: Vec::new(),
        }
    }

    pub(crate) fn output_logical_rects(&self) -> Vec<(u32, Rectangle<i32, Logical>)> {
        self.outputs
            .iter()
            .filter_map(|(&id, (output, _))| Some((id, output_logical_rect(output)?)))
            .collect()
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn new_output(&mut self, output: OutputInfo) -> Output {
        create_or_update_output::<WprsState>(&mut self.outputs, &self.dh, output)
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn update_output(&mut self, output: OutputInfo) {
        let (local_output, _) = match self.outputs.entry(output.id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(_) => {
                warn!("update to unknown display {:?}", output.id);
                return;
            },
        };

        compositor_utils::update_output(local_output, output);
    }

    #[instrument(skip(self), level = "debug")]
    pub(crate) fn destroy_output(&mut self, output: OutputInfo) -> Option<Output> {
        if let Some((_, (local_output, global_id))) = self.outputs.remove_entry(&output.id) {
            self.dh.remove_global::<WprsState>(global_id);
            Some(local_output)
        } else {
            None
        }
    }
}

impl BufferHandler for WprsState {
    #[instrument(skip(self), level = "debug")]
    fn buffer_destroyed(&mut self, buffer: &WlBuffer) {}
}

impl SelectionHandler for WprsState {
    type SelectionUserData = ();

    // We need to implement this trait for copying to clients, but all our
    // clients are xwayland clients and so the methods below should never be
    // called.

    #[instrument(skip(self, _seat), level = "debug")]
    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        _seat: Seat<Self>,
    ) {
        error!("new_selection called");
    }

    #[instrument(skip(self, _fd, _seat, _user_data), level = "debug")]
    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        mime_type: String,
        _fd: OwnedFd,
        _seat: Seat<Self>,
        _user_data: &Self::SelectionUserData,
    ) {
        error!("new_selection called");
    }
}

impl DataDeviceHandler for WprsState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.compositor_state.data_device_state
    }
}

impl PrimarySelectionHandler for WprsState {
    fn primary_selection_state(
        &self,
    ) -> &smithay::wayland::selection::primary_selection::PrimarySelectionState {
        &self.compositor_state.primary_selection_state
    }
}

impl ClientDndGrabHandler for WprsState {}
impl ServerDndGrabHandler for WprsState {}

impl XWaylandShellHandler for WprsState {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.compositor_state.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm: XwmId, wl_surface: WlSurface, surface: X11Surface) {
        // TODO: we should implement this and get rid of the deferring commit logic below
        debug!(
            "X11 window {:?} associated with surface {:?}",
            surface, wl_surface
        )
    }
}

fn execute_or_defer_commit(state: &mut WprsState, surface: WlSurface) -> Result<()> {
    if !surface.alive() {
        return Ok(());
    }
    commit(&surface, state).location(loc!())?;

    let xwayland_surface = state.surfaces.get(&surface.id());

    // we may not have matched an X11 surface to the wayland surface yet.
    // defer if that is the case.
    if !xwayland_surface.is_some_and(XWaylandSurface::ready) {
        debug!("deferring commit");
        state.event_loop_handle.insert_idle(|state| {
            execute_or_defer_commit(state, surface).log_and_ignore(loc!());
        });
    }
    Ok(())
}

impl CompositorHandler for WprsState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client
            .get_data::<XWaylandClientData>()
            .unwrap()
            .compositor_state
    }

    #[instrument(skip(self), level = "debug")]
    fn commit(&mut self, surface: &WlSurface) {
        execute_or_defer_commit(self, surface.clone()).log_and_ignore(loc!());
    }
}

#[instrument(skip(state), level = "debug")]
pub fn commit(surface: &WlSurface, state: &mut WprsState) -> Result<()> {
    compositor::with_states(surface, |surface_data| -> Result<()> {
        commit_inner(surface, surface_data, state).location(loc!())?;
        Ok(())
    })
    .location(loc!())?;
    on_commit_buffer_handler::<WprsState>(surface);
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct X11ParentForPopup {
    pub(crate) surface_id: ObjectId,
    pub(crate) xdg_surface: xdg_surface::XdgSurface,
    pub(crate) x11_loc: Point<i32>,
    pub(crate) frame_offset: Point<i32>,
}

#[derive(Debug, Clone)]
pub(crate) struct X11ParentForSubsurface {
    pub(crate) surface: SctkWlSurface,
    pub(crate) x11_loc: Point<i32>,
}

#[derive(Debug, Clone)]
pub(crate) struct X11Parent {
    pub(crate) surface_id: ObjectId,
    pub(crate) for_popup: Option<X11ParentForPopup>,
    pub(crate) for_subsurface: X11ParentForSubsurface,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum CandidateRole {
    Toplevel,
    Popup,
    SubSurface,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParentCandidate<Id> {
    pub(crate) id: Id,
    pub(crate) x11_window_id: u32,
    pub(crate) pid: Option<u32>,
    pub(crate) geometry: Rectangle<i32, Logical>,
    pub(crate) role: CandidateRole,
}

fn role_priority(role: CandidateRole) -> u8 {
    match role {
        CandidateRole::Popup => 0,
        CandidateRole::Toplevel => 1,
        CandidateRole::SubSurface => 2,
    }
}

fn point_distance_sq_to_rect(p: Point<i32>, rect: Rectangle<i32, Logical>) -> i64 {
    let min_x = i64::from(rect.loc.x);
    let max_x = i64::from(rect.loc.x) + i64::from(rect.size.w.max(0));
    let min_y = i64::from(rect.loc.y);
    let max_y = i64::from(rect.loc.y) + i64::from(rect.size.h.max(0));
    let px = i64::from(p.x);
    let py = i64::from(p.y);
    let dx = if px < min_x {
        min_x - px
    } else if px > max_x {
        px - max_x
    } else {
        0
    };
    let dy = if py < min_y {
        min_y - py
    } else if py > max_y {
        py - max_y
    } else {
        0
    };
    dx * dx + dy * dy
}

fn rect_contains_point(rect: Rectangle<i32, Logical>, p: Point<i32>) -> bool {
    p.x >= rect.loc.x
        && p.x < rect.loc.x + rect.size.w
        && p.y >= rect.loc.y
        && p.y < rect.loc.y + rect.size.h
}

fn rects_intersect(a: Rectangle<i32, Logical>, b: Rectangle<i32, Logical>) -> bool {
    let aw = a.size.w.max(1);
    let ah = a.size.h.max(1);
    let bw = b.size.w.max(1);
    let bh = b.size.h.max(1);
    a.loc.x < b.loc.x + bw
        && a.loc.x + aw > b.loc.x
        && a.loc.y < b.loc.y + bh
        && a.loc.y + ah > b.loc.y
}

fn candidate_near_child(
    candidate: Rectangle<i32, Logical>,
    child: Rectangle<i32, Logical>,
) -> bool {
    // Allow a small margin so popups flipped or offset slightly past a parent's
    // border still match the parent that spawned them.
    const PROXIMITY_MARGIN_PX: i64 = 64;
    rect_contains_point(candidate, child.loc.into())
        || rects_intersect(candidate, child)
        || point_distance_sq_to_rect(child.loc.into(), candidate)
            <= PROXIMITY_MARGIN_PX * PROXIMITY_MARGIN_PX
}

fn best_candidate_by_geometry<'a, Id: Copy + 'a>(
    child_geometry: Rectangle<i32, Logical>,
    pool: impl Iterator<Item = &'a ParentCandidate<Id>>,
) -> Option<Id> {
    let child_loc: Point<i32> = child_geometry.loc.into();
    pool.min_by_key(|c| {
        let overlap_rank = if rect_contains_point(c.geometry, child_loc) {
            0u8
        } else if rects_intersect(c.geometry, child_geometry) {
            1u8
        } else {
            2u8
        };
        (
            overlap_rank,
            point_distance_sq_to_rect(child_loc, c.geometry),
            role_priority(c.role),
        )
    })
    .map(|c| c.id)
}

/// Selects the best mapped parent surface for an X11 popup or transient window.
///
/// Priority order:
/// 1. Explicit `WM_TRANSIENT_FOR` (never self).
/// 2. Focused surface (`last_pointer`, then `last_keyboard`) matching `child_pid`
///    whose geometry is near `child_geometry`.
/// 3. Focused surface (`last_pointer`, then `last_keyboard`) matching `child_pid`
///    (handles popups mapped at `(0, 0)` before their initial `ConfigureNotify`).
/// 4. Closest/containing mapped surface matching `child_pid`.
/// 5. Focused surface (`last_pointer`, then `last_keyboard`) regardless of PID
///    (for clients without `_NET_WM_PID`), preferring one near `child_geometry`.
/// 6. Closest/containing mapped surface regardless of PID.
pub(crate) fn select_x11_parent_id<Id: Eq + Copy>(
    child_id: Id,
    child_transient_for: Option<u32>,
    child_pid: Option<u32>,
    child_geometry: Rectangle<i32, Logical>,
    last_pointer: Option<Id>,
    last_keyboard: Option<Id>,
    candidates: &[ParentCandidate<Id>],
) -> Option<Id> {
    let valid = |c: &&ParentCandidate<Id>| c.id != child_id;

    if let Some(transient_window_id) = child_transient_for
        && let Some(parent) = candidates
            .iter()
            .filter(valid)
            .find(|c| c.x11_window_id == transient_window_id)
    {
        return Some(parent.id);
    }

    let focused_candidates = |pid_filter: Option<u32>| -> Vec<&ParentCandidate<Id>> {
        [last_pointer, last_keyboard]
            .into_iter()
            .flatten()
            .filter(|&id| id != child_id)
            .filter_map(|id| {
                candidates
                    .iter()
                    .find(|c| c.id == id && pid_filter.is_none_or(|pid| c.pid == Some(pid)))
            })
            .collect()
    };

    if let Some(pid) = child_pid {
        let focused_same_pid = focused_candidates(Some(pid));
        if let Some(near) = focused_same_pid
            .iter()
            .find(|c| candidate_near_child(c.geometry, child_geometry))
        {
            return Some(near.id);
        }
        if let Some(first) = focused_same_pid.first() {
            return Some(first.id);
        }
        if let Some(by_geo) = best_candidate_by_geometry(
            child_geometry,
            candidates
                .iter()
                .filter(valid)
                .filter(|c| c.pid == Some(pid)),
        ) {
            return Some(by_geo);
        }
    }

    let focused_any = focused_candidates(None);
    if let Some(near) = focused_any
        .iter()
        .find(|c| candidate_near_child(c.geometry, child_geometry))
    {
        return Some(near.id);
    }
    if let Some(first) = focused_any.first() {
        return Some(first.id);
    }

    best_candidate_by_geometry(child_geometry, candidates.iter().filter(valid))
}

fn find_xdg_ancestor_for_popup(
    state: &WprsState,
    start_id: &ObjectId,
) -> Option<X11ParentForPopup> {
    let mut cur_id = Some(start_id.clone());
    for _ in 0..64 {
        let id = cur_id?;
        let surface = state.surfaces.get(&id)?;
        let parent_geo = surface.get_x11_surface().ok()?.geometry();
        match &surface.role {
            Some(Role::XdgToplevel(toplevel)) => {
                return Some(X11ParentForPopup {
                    surface_id: id,
                    xdg_surface: toplevel.xdg_surface().clone(),
                    x11_loc: parent_geo.loc.into(),
                    frame_offset: toplevel.frame_offset,
                });
            },
            Some(Role::XdgPopup(popup)) => {
                return Some(X11ParentForPopup {
                    surface_id: id,
                    xdg_surface: popup.xdg_surface().clone(),
                    x11_loc: parent_geo.loc.into(),
                    frame_offset: (0, 0).into(),
                });
            },
            Some(Role::SubSurface(_)) => {
                cur_id = surface.parent_id.clone();
            },
            _ => return None,
        }
    }
    None
}

fn build_x11_parent(state: &WprsState, parent_id: &ObjectId) -> Option<X11Parent> {
    let parent = state.surfaces.get(parent_id)?;
    let parent_geo = parent.get_x11_surface().ok()?.geometry();
    let parent_loc: Point<i32> = parent_geo.loc.into();

    match &parent.role {
        Some(Role::XdgToplevel(toplevel)) => Some(X11Parent {
            surface_id: parent_id.clone(),
            for_popup: Some(X11ParentForPopup {
                surface_id: parent_id.clone(),
                xdg_surface: toplevel.xdg_surface().clone(),
                x11_loc: parent_loc,
                frame_offset: toplevel.frame_offset,
            }),
            for_subsurface: X11ParentForSubsurface {
                surface: toplevel.wl_surface().clone(),
                x11_loc: parent_loc,
            },
        }),
        Some(Role::XdgPopup(popup)) => Some(X11Parent {
            surface_id: parent_id.clone(),
            for_popup: Some(X11ParentForPopup {
                surface_id: parent_id.clone(),
                xdg_surface: popup.xdg_surface().clone(),
                x11_loc: parent_loc,
                frame_offset: (0, 0).into(),
            }),
            for_subsurface: X11ParentForSubsurface {
                surface: popup.wl_surface().clone(),
                x11_loc: parent_loc,
            },
        }),
        Some(Role::SubSurface(subsurface)) => Some(X11Parent {
            surface_id: parent_id.clone(),
            for_popup: find_xdg_ancestor_for_popup(state, parent_id),
            for_subsurface: X11ParentForSubsurface {
                surface: subsurface.wl_surface().clone(),
                x11_loc: parent_loc,
            },
        }),
        Some(Role::Cursor) | None => None,
    }
}

pub(crate) fn popup_parent_origin(
    surfaces: &HashMap<ObjectId, XWaylandSurface>,
    parent_id: &ObjectId,
) -> Option<(Point<i32>, Point<i32>)> {
    let parent = surfaces.get(parent_id)?;
    let parent_loc: Point<i32> = parent.get_x11_surface().ok()?.geometry().loc.into();
    let frame_offset = match &parent.role {
        Some(Role::XdgToplevel(toplevel)) => toplevel.frame_offset,
        Some(Role::XdgPopup(_)) => (0, 0).into(),
        _ => return None,
    };
    Some((parent_loc, frame_offset))
}

fn parent_candidates(state: &WprsState) -> Vec<ParentCandidate<&ObjectId>> {
    state
        .surfaces
        .iter()
        .filter_map(|(id, xwls)| {
            let role = match &xwls.role {
                Some(Role::XdgToplevel(_)) => CandidateRole::Toplevel,
                Some(Role::XdgPopup(_)) => CandidateRole::Popup,
                Some(Role::SubSurface(_)) => CandidateRole::SubSurface,
                Some(Role::Cursor) | None => return None,
            };
            let x11_surface = xwls.x11_surface.as_ref()?;
            Some(ParentCandidate {
                id,
                x11_window_id: x11_surface.window_id(),
                pid: x11_surface.pid(),
                geometry: x11_surface.geometry(),
                role,
            })
        })
        .collect()
}

fn find_explicit_transient_parent_id(
    state: &WprsState,
    child_id: &ObjectId,
    x11_surface: &X11Surface,
) -> Option<ObjectId> {
    let transient_window_id = x11_surface.is_transient_for()?;
    parent_candidates(state)
        .into_iter()
        .find(|c| c.id != child_id && c.x11_window_id == transient_window_id)
        .map(|c| c.id.clone())
}

fn resolve_x11_parent(
    state: &WprsState,
    child_id: &ObjectId,
    x11_surface: &X11Surface,
) -> Option<X11Parent> {
    let candidates = parent_candidates(state);
    let parent_id = select_x11_parent_id(
        child_id,
        x11_surface.is_transient_for(),
        x11_surface.pid(),
        x11_surface.geometry(),
        state.client_state.last_pointer_surface.as_ref(),
        state.client_state.last_keyboard_surface.as_ref(),
        &candidates,
    )?
    .clone();
    build_x11_parent(state, &parent_id)
}

#[instrument(skip(state), level = "debug")]
pub fn commit_inner(
    surface: &WlSurface,
    surface_data: &SurfaceData,
    state: &mut WprsState,
) -> Result<()> {
    let mut guard = surface_data.cached_state.get::<SurfaceAttributes>();
    let surface_attributes = guard.current();
    compositor_utils::cap_surface_damage(&mut surface_attributes.damage);

    let x11_surface = state
        .compositor_state
        .x11_surfaces
        .iter()
        .position(|x11s| x11s.wl_surface().map(|s| s == *surface).unwrap_or(false))
        .map(|pos| state.compositor_state.x11_surfaces.swap_remove(pos));
    debug!("matched x11 surface: {x11_surface:?}");

    let role_plan = x11_surface.as_ref().map(|x11_surface| {
        let explicit_parent_id =
            find_explicit_transient_parent_id(state, &surface.id(), x11_surface);
        let wayland_window_type = crate::xwayland_xdg_shell::classify_wayland_window_type(
            x11_surface.window_type(),
            x11_surface.is_override_redirect(),
            explicit_parent_id.is_some(),
        );
        let parent = match wayland_window_type {
            crate::xwayland_xdg_shell::WaylandWindowType::Toplevel => None,
            crate::xwayland_xdg_shell::WaylandWindowType::Popup => {
                resolve_x11_parent(state, &surface.id(), x11_surface)
            },
            crate::xwayland_xdg_shell::WaylandWindowType::SubSurface => explicit_parent_id
                .as_ref()
                .and_then(|id| build_x11_parent(state, id)),
        };
        (wayland_window_type, parent)
    });

    let mut assigned_parent_id: Option<ObjectId> = None;
    {
        let xwayland_surface = state.surfaces.entry(surface.id()).or_default();

        if let (Some(x11_surface), Some((wayland_window_type, parent))) = (x11_surface, role_plan) {
            if xwayland_surface.local_surface.is_none() {
                let needs_subsurface_parent =
                    matches!(
                        wayland_window_type,
                        crate::xwayland_xdg_shell::WaylandWindowType::SubSurface
                    ) || (matches!(
                        wayland_window_type,
                        crate::xwayland_xdg_shell::WaylandWindowType::Popup
                    ) && parent.as_ref().is_some_and(|p| p.for_popup.is_none()));
                let subsurface_parent = needs_subsurface_parent
                    .then(|| parent.as_ref().map(|p| &p.for_subsurface.surface))
                    .flatten();
                xwayland_surface
                    .update_local_surface(
                        surface,
                        subsurface_parent,
                        &state.client_state.compositor_state,
                        &state.client_state.qh,
                        &mut state.surface_bimap,
                    )
                    .location(loc!())?;
            }

            xwayland_surface
                .update_x11_surface(
                    x11_surface,
                    wayland_window_type,
                    parent,
                    &state.client_state.xdg_shell_state,
                    &state.client_state.shm_state,
                    state.client_state.subcompositor_state.clone(),
                    &state.client_state.qh,
                    state.compositor_state.decoration_behavior,
                )
                .location(loc!())?;
            assigned_parent_id = xwayland_surface.parent_id.clone();
        }
    }

    if let Some(parent_id) = assigned_parent_id
        && parent_id != surface.id()
        && let Some(parent_xwayland_surface) = state.surfaces.get_mut(&parent_id)
    {
        debug!(
            "registering child {:?} with parent {:?}",
            surface.id(),
            &parent_id
        );
        parent_xwayland_surface.children.insert(surface.id());
    }

    let xwayland_surface = state.surfaces.entry(surface.id()).or_default();

    let client_transform: SctkTransform =
        SerTransform::from(surface_attributes.buffer_transform).into();
    xwayland_surface
        .sync_buffer_scale_and_transform(surface_attributes.buffer_scale, client_transform);

    debug!("buffer assignment: {:?}", &surface_attributes.buffer);

    match &surface_attributes.buffer {
        Some(BufferAssignment::NewBuffer(buffer)) => {
            let damage =
                compositor_utils::with_buffer_contents(buffer, |data, spec| -> Result<_> {
                    // Xwayland surfaces have no viewport (`viewport_state` is `None`),
                    // while `(spec.width, spec.height)` clamps damage to the buffer
                    // bounds and inverts rotating/flipping `buffer_transform`s.
                    let damage = compositor_utils::damage_to_buffer_rects(
                        &surface_attributes.damage,
                        surface_attributes.buffer_scale,
                        surface_attributes.buffer_transform.into(),
                        None,
                        (spec.width, spec.height),
                    );
                    xwayland_surface.update_buffer(
                        &spec,
                        data,
                        &damage,
                        state.client_state.pool.as_mut().location(loc!())?,
                    )?;
                    Ok(damage)
                })
                .location(loc!())?
                .location(loc!())?;

            // Clear `surface_attributes.damage` only after the fallible buffer
            // update succeeds so a failed commit leaves its damage queued.
            surface_attributes.damage.clear();
            xwayland_surface.damage.merge_commit(Some(damage), true);
            xwayland_surface.buffer_attached = false;
        },
        Some(BufferAssignment::Removed) => {
            surface_attributes.damage.clear();
            xwayland_surface.buffer = None;
            xwayland_surface.buffer_attached = false;
            xwayland_surface.damage = FrameDamage::Empty;
            if xwayland_surface.has_client_surface() {
                xwayland_surface.wl_surface().attach(None, 0, 0);
            }
        },
        // Retain `surface_attributes.damage` across bufferless commits (bounded
        // by `cap_surface_damage`) until the next `NewBuffer` or `Removed` commit.
        None => {},
    }

    if let Some(Role::XdgToplevel(toplevel)) = &mut xwayland_surface.role
        && toplevel.configured
        && toplevel.window_frame.is_dirty()
    {
        toplevel.window_frame.draw();
    }

    if let Some(Role::SubSurface(subsurface)) = &mut xwayland_surface.role
        && let Some(decorated_subsurface) = &mut subsurface.frame
        && decorated_subsurface.is_dirty()
    {
        decorated_subsurface.draw();
    }

    if xwayland_surface.ready() {
        if let Some(Role::SubSurface(subsurface)) = &mut xwayland_surface.role {
            if !subsurface.pending_frame_callback {
                xwayland_surface.frame(&state.client_state.qh);
            }
        } else {
            xwayland_surface.frame(&state.client_state.qh);
        }

        xwayland_surface.try_draw_buffer();
    }

    if xwayland_surface.ready() || xwayland_surface.needs_configure() {
        xwayland_surface.commit();
    }

    if xwayland_surface.x11_surface.is_none() || matches!(xwayland_surface.role, Some(Role::Cursor))
    {
        compositor_utils::send_frames(
            surface,
            &surface_data.data_map,
            surface_attributes,
            state.compositor_state.start_time.elapsed(),
            Duration::ZERO,
        )
        .location(loc!())?;
    }
    Ok(())
}

impl ShmHandler for WprsState {
    fn shm_state(&self) -> &ShmState {
        &self.compositor_state.shm_state
    }
}

impl SeatHandler for WprsState {
    type KeyboardFocus = X11Surface;
    type PointerFocus = X11Surface;
    type TouchFocus = X11Surface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.compositor_state.seat_state
    }

    #[instrument(skip(self, _seat), level = "debug")]
    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        // TODO: support multiple seats
        let Some(themed_pointer) = self
            .client_state
            .seat_objects
            .last()
            .and_then(|seat| seat.pointer.as_ref())
        else {
            return;
        };
        let pointer = themed_pointer.pointer();

        // TODO: move to a fn on serialization::CursorImaveStatus
        match image {
            CursorImageStatus::Hidden => {
                themed_pointer.hide_cursor().log_and_ignore(loc!());
            },
            CursorImageStatus::Surface(surface) => {
                let (hotspot, buffer_scale, buffer_transform) =
                    compositor::with_states(&surface, |surface_data| {
                        let hotspot = surface_data
                            .data_map
                            .get::<CursorImageSurfaceData>()
                            .unwrap()
                            .lock()
                            .unwrap()
                            .hotspot;
                        let mut guard = surface_data.cached_state.get::<SurfaceAttributes>();
                        let attrs = guard.current();
                        (
                            hotspot,
                            attrs.buffer_scale,
                            SerTransform::from(attrs.buffer_transform).into(),
                        )
                    });

                let xwayland_surface = self.surfaces.entry(surface.id()).or_default();
                if !xwayland_surface.has_client_surface() {
                    log_and_return!(xwayland_surface.update_local_surface(
                        &surface,
                        None,
                        &self.client_state.compositor_state,
                        &self.client_state.qh,
                        &mut self.surface_bimap,
                    ));
                }

                xwayland_surface.role = Some(Role::Cursor);
                xwayland_surface.sync_buffer_scale_and_transform(buffer_scale, buffer_transform);
                if !xwayland_surface.buffer_attached && xwayland_surface.buffer.is_some() {
                    xwayland_surface.try_draw_buffer();
                    xwayland_surface.commit();
                }

                // TODO: expose serial to this function, then remove
                // last_enter_serial on client.
                pointer.set_cursor(
                    self.client_state.last_enter_serial,
                    Some(xwayland_surface.wl_surface()),
                    hotspot.x,
                    hotspot.y,
                );
            },
            CursorImageStatus::Named(name) => {
                themed_pointer
                    .set_cursor(&self.client_state.conn, name)
                    .log_and_ignore(loc!());
            },
        }
    }
}

impl OutputHandler for WprsState {}

smithay::delegate_compositor!(WprsState);
smithay::delegate_shm!(WprsState);
smithay::delegate_seat!(WprsState);
smithay::delegate_data_device!(WprsState);
smithay::delegate_output!(WprsState);
smithay::delegate_primary_selection!(WprsState);
smithay::delegate_xwayland_shell!(WprsState);

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use smithay::reexports::wayland_server::Display;
    use smithay::reexports::wayland_server::backend::ClientData;
    use smithay::reexports::wayland_server::backend::ClientId;
    use smithay::reexports::wayland_server::backend::DisconnectReason;
    use smithay::reexports::wayland_server::protocol::wl_output::Transform as SmithayWlTransform;
    use smithay_client_toolkit::reexports::client::Connection;
    use smithay_client_toolkit::reexports::client::Dispatch;
    use smithay_client_toolkit::reexports::client::EventQueue;
    use smithay_client_toolkit::reexports::client::Proxy;
    use smithay_client_toolkit::reexports::client::QueueHandle;
    use smithay_client_toolkit::reexports::client::WEnum;
    use smithay_client_toolkit::reexports::client::protocol::wl_callback;
    use smithay_client_toolkit::reexports::client::protocol::wl_compositor::WlCompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_output;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry;
    use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;
    use smithay_client_toolkit::reexports::protocols::xdg::xdg_output::zv1::client::zxdg_output_manager_v1::ZxdgOutputManagerV1;
    use smithay_client_toolkit::reexports::protocols::xdg::xdg_output::zv1::client::zxdg_output_v1;

    use super::*;
    use crate::serialization::wayland::Mode as SerMode;
    use crate::serialization::wayland::Subpixel as SerSubpixel;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new((x, y).into(), (w, h).into())
    }

    #[test]
    fn test_compute_toplevel_x11_location_single_multi_negative_and_maximized() {
        let empty: HashSet<u32> = HashSet::new();

        // Detached / zero outputs still produces a valid `(0, 0)` origin.
        assert_eq!(
            compute_toplevel_x11_location(&[], &empty, (800, 600), false),
            (0, 0).into()
        );

        // Single 1920x1080 monitor: floating window is centered so popups/drags
        // have positive room on every side; maximized/full-size/oversized clamp
        // cleanly to `(0, 0)`.
        let single = [(1, rect(0, 0, 1920, 1080))];
        assert_eq!(
            compute_toplevel_x11_location(&single, &empty, (800, 600), false),
            (560, 240).into()
        );
        assert_eq!(
            compute_toplevel_x11_location(&single, &empty, (800, 600), true),
            (0, 0).into()
        );
        assert_eq!(
            compute_toplevel_x11_location(&single, &empty, (1920, 1080), false),
            (0, 0).into()
        );
        assert_eq!(
            compute_toplevel_x11_location(&single, &empty, (2400, 1400), false),
            (0, 0).into()
        );

        // Multi-monitor: surface on monitor 2 lands on monitor 2; surface with
        // no entered outputs picks the leftmost monitor deterministically.
        let dual = [(2, rect(1920, 0, 2560, 1440)), (1, rect(0, 0, 1920, 1080))];
        assert_eq!(
            compute_toplevel_x11_location(&dual, &empty, (800, 600), false),
            (560, 240).into()
        );
        let on_second = HashSet::from([2]);
        assert_eq!(
            compute_toplevel_x11_location(&dual, &on_second, (800, 600), false),
            (2800, 420).into()
        );
        assert_eq!(
            compute_toplevel_x11_location(&dual, &on_second, (800, 600), true),
            (1920, 0).into()
        );

        // Negative logical position is shifted into non-negative X11 root space.
        let negative_left = [(1, rect(-1920, 0, 1920, 1080)), (2, rect(0, 0, 2560, 1440))];
        let on_first = HashSet::from([1]);
        assert_eq!(
            compute_toplevel_x11_location(&negative_left, &on_first, (800, 600), false),
            (560, 240).into()
        );
        assert_eq!(
            compute_toplevel_x11_location(&negative_left, &on_second, (800, 600), true),
            (1920, 0).into()
        );
    }

    #[test]
    fn test_toplevel_needs_output_reposition() {
        let dual = [(1, rect(0, 0, 1920, 1080)), (2, rect(1920, 0, 2560, 1440))];
        let geo_on_m1 = rect(560, 240, 800, 600);
        assert!(!toplevel_needs_output_reposition(
            &dual,
            &HashSet::new(),
            geo_on_m1
        ));
        assert!(!toplevel_needs_output_reposition(
            &dual,
            &HashSet::from([1]),
            geo_on_m1
        ));
        // Straddling both monitors still matches monitor 1: do not jump.
        assert!(!toplevel_needs_output_reposition(
            &dual,
            &HashSet::from([1, 2]),
            geo_on_m1
        ));
        // Strictly on monitor 2: reposition into monitor 2.
        assert!(toplevel_needs_output_reposition(
            &dual,
            &HashSet::from([2]),
            geo_on_m1
        ));
        // Unknown output ID does not trigger a spurious reposition.
        assert!(!toplevel_needs_output_reposition(
            &dual,
            &HashSet::from([99]),
            geo_on_m1
        ));
    }

    #[derive(Default)]
    struct TestClientState {
        compositor_state: CompositorClientState,
    }

    impl ClientData for TestClientState {
        fn initialized(&self, _client_id: ClientId) {}
        fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
    }

    struct MiniServer {
        compositor_state: CompositorState,
        _output_manager_state: OutputManagerState,
        outputs: HashMap<u32, (Output, GlobalId)>,
        last_commit: Option<(i32, SmithayWlTransform)>,
    }

    impl CompositorHandler for MiniServer {
        fn compositor_state(&mut self) -> &mut CompositorState {
            &mut self.compositor_state
        }

        fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
            &client
                .get_data::<TestClientState>()
                .unwrap()
                .compositor_state
        }

        fn commit(&mut self, surface: &WlSurface) {
            compositor::with_states(surface, |data| {
                let mut guard = data.cached_state.get::<SurfaceAttributes>();
                let attrs = guard.current();
                self.last_commit = Some((attrs.buffer_scale, attrs.buffer_transform));
            });
        }
    }

    impl OutputHandler for MiniServer {}

    smithay::delegate_compositor!(MiniServer);
    smithay::delegate_output!(MiniServer);

    #[derive(Debug, Default)]
    struct ObservedOutput {
        mode: Option<(i32, i32)>,
        physical: Option<(i32, i32)>,
        scale: Option<i32>,
        name: Option<String>,
        xdg_pos: Option<(i32, i32)>,
        xdg_size: Option<(i32, i32)>,
    }

    #[derive(Default)]
    struct MiniClient {
        compositor: Option<WlCompositor>,
        xdg_output_manager: Option<ZxdgOutputManagerV1>,
        wl_outputs: HashMap<u32, wl_output::WlOutput>,
        xdg_outputs: HashMap<u32, zxdg_output_v1::ZxdgOutputV1>,
        observed: HashMap<u32, ObservedOutput>,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for MiniClient {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _data: &(),
            _conn: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_compositor" => {
                        state.compositor =
                            Some(registry.bind::<WlCompositor, _, _>(name, version.min(6), qh, ()));
                    },
                    "zxdg_output_manager_v1" => {
                        let mgr = registry.bind::<ZxdgOutputManagerV1, _, _>(
                            name,
                            version.min(3),
                            qh,
                            (),
                        );
                        for (&id, wl_out) in &state.wl_outputs {
                            state
                                .xdg_outputs
                                .entry(id)
                                .or_insert_with(|| mgr.get_xdg_output(wl_out, qh, id));
                        }
                        state.xdg_output_manager = Some(mgr);
                    },
                    "wl_output" => {
                        let wl_out = registry.bind::<wl_output::WlOutput, _, _>(
                            name,
                            version.min(4),
                            qh,
                            name,
                        );
                        if let Some(mgr) = &state.xdg_output_manager {
                            state
                                .xdg_outputs
                                .insert(name, mgr.get_xdg_output(&wl_out, qh, name));
                        }
                        state.wl_outputs.insert(name, wl_out);
                        state.observed.entry(name).or_default();
                    },
                    _ => {},
                }
            }
        }
    }

    impl Dispatch<wl_callback::WlCallback, Arc<AtomicBool>> for MiniClient {
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

    impl Dispatch<WlCompositor, ()> for MiniClient {
        fn event(
            _state: &mut Self,
            _proxy: &WlCompositor,
            _event: <WlCompositor as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<ClientWlSurface, ()> for MiniClient {
        fn event(
            _state: &mut Self,
            _proxy: &ClientWlSurface,
            _event: <ClientWlSurface as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<ZxdgOutputManagerV1, ()> for MiniClient {
        fn event(
            _state: &mut Self,
            _proxy: &ZxdgOutputManagerV1,
            _event: <ZxdgOutputManagerV1 as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<wl_output::WlOutput, u32> for MiniClient {
        fn event(
            state: &mut Self,
            _proxy: &wl_output::WlOutput,
            event: wl_output::Event,
            id: &u32,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            let entry = state.observed.entry(*id).or_default();
            match event {
                wl_output::Event::Geometry {
                    physical_width,
                    physical_height,
                    ..
                } => {
                    entry.physical = Some((physical_width, physical_height));
                },
                wl_output::Event::Mode {
                    flags: WEnum::Value(flags),
                    width,
                    height,
                    ..
                } if flags.contains(wl_output::Mode::Current) => {
                    entry.mode = Some((width, height));
                },
                wl_output::Event::Scale { factor } => entry.scale = Some(factor),
                wl_output::Event::Name { name } => entry.name = Some(name),
                _ => {},
            }
        }
    }

    impl Dispatch<zxdg_output_v1::ZxdgOutputV1, u32> for MiniClient {
        fn event(
            state: &mut Self,
            _proxy: &zxdg_output_v1::ZxdgOutputV1,
            event: zxdg_output_v1::Event,
            id: &u32,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            let entry = state.observed.entry(*id).or_default();
            match event {
                zxdg_output_v1::Event::LogicalPosition { x, y } => entry.xdg_pos = Some((x, y)),
                zxdg_output_v1::Event::LogicalSize { width, height } => {
                    entry.xdg_size = Some((width, height));
                },
                _ => {},
            }
        }
    }

    fn roundtrip(
        display: &mut Display<MiniServer>,
        server: &mut MiniServer,
        conn: &Connection,
        queue: &mut EventQueue<MiniClient>,
        client: &mut MiniClient,
    ) {
        let done = Arc::new(AtomicBool::new(false));
        let _cb = conn.display().sync(&queue.handle(), Arc::clone(&done));
        conn.flush().unwrap();
        display.dispatch_clients(server).unwrap();
        display.flush_clients().unwrap();
        while !done.load(Ordering::Acquire) {
            queue.blocking_dispatch(client).unwrap();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_xwayland_compositor_outputs_and_surface_scale_wire() {
        let mut display: Display<MiniServer> = Display::new().unwrap();
        let mut dh = display.handle();
        let mut server = MiniServer {
            compositor_state: CompositorState::new::<MiniServer>(&dh),
            _output_manager_state: OutputManagerState::new_with_xdg_output::<MiniServer>(&dh),
            outputs: HashMap::new(),
            last_commit: None,
        };

        let (server_sock, client_sock) = UnixStream::pair().unwrap();
        dh.insert_client(server_sock, Arc::new(TestClientState::default()))
            .unwrap();
        let conn = Connection::from_socket(client_sock).unwrap();
        let mut queue: EventQueue<MiniClient> = conn.new_event_queue();
        let qh = queue.handle();
        let _registry = conn.display().get_registry(&qh, ());
        let mut client = MiniClient::default();

        // Advertise two real outputs (including fractional scale) through
        // `create_or_update_output` — must NOT inflate dimensions by 3x.
        let out_1 = OutputInfo {
            id: 1,
            model: "Laptop".to_string(),
            make: "Vendor".to_string(),
            location: (0, 0).into(),
            physical_size: (300, 190).into(),
            subpixel: SerSubpixel::HorizontalRgb,
            transform: SerTransform::Normal,
            scale_factor: 1,
            mode: SerMode {
                dimensions: (1920, 1200).into(),
                refresh_rate: 60000,
                current: true,
                preferred: true,
            },
            logical_position: Some((0, 0).into()),
            logical_size: Some((1920, 1200).into()),
            name: Some("eDP-1".to_string()),
            description: None,
        };
        let out_2 = OutputInfo {
            id: 2,
            model: "External4K".to_string(),
            make: "Vendor".to_string(),
            location: (0, 0).into(),
            physical_size: (600, 340).into(),
            subpixel: SerSubpixel::HorizontalRgb,
            transform: SerTransform::Normal,
            scale_factor: 2,
            mode: SerMode {
                dimensions: (3840, 2160).into(),
                refresh_rate: 60000,
                current: true,
                preferred: true,
            },
            logical_position: Some((1920, 0).into()),
            logical_size: Some((2560, 1440).into()),
            name: Some("DP-1".to_string()),
            description: None,
        };
        let o1 = create_or_update_output::<MiniServer>(&mut server.outputs, &dh, out_1);
        let o2 = create_or_update_output::<MiniServer>(&mut server.outputs, &dh, out_2);
        assert_eq!(output_logical_rect(&o1), Some(rect(0, 0, 1920, 1200)));
        assert_eq!(output_logical_rect(&o2), Some(rect(1920, 0, 2560, 1440)));

        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);

        assert!(client.xdg_output_manager.is_some());
        let by_name: HashMap<String, &ObservedOutput> = client
            .observed
            .values()
            .map(|o| (o.name.clone().expect("wl_output.name"), o))
            .collect();
        assert_eq!(by_name["1_eDP-1"].mode, Some((1920, 1200)));
        assert_eq!(by_name["1_eDP-1"].physical, Some((300, 190)));
        assert_eq!(by_name["1_eDP-1"].scale, Some(1));
        assert_eq!(by_name["1_eDP-1"].xdg_pos, Some((0, 0)));
        assert_eq!(by_name["1_eDP-1"].xdg_size, Some((1920, 1200)));

        assert_eq!(by_name["2_DP-1"].mode, Some((3840, 2160)));
        assert_eq!(by_name["2_DP-1"].physical, Some((600, 340)));
        assert_eq!(by_name["2_DP-1"].scale, Some(2));
        assert_eq!(by_name["2_DP-1"].xdg_pos, Some((1920, 0)));
        assert_eq!(by_name["2_DP-1"].xdg_size, Some((2560, 1440)));

        // Verify `XWaylandSurface::sync_buffer_scale_and_transform` clamps
        // invalid scales and safely queues state while `local_surface` is None.
        let mut pre_surface = XWaylandSurface::default();
        assert!(!pre_surface.has_client_surface());
        pre_surface.sync_buffer_scale_and_transform(0, SctkTransform::_90);
        assert_eq!(pre_surface.buffer_scale, 1);
        assert_eq!(pre_surface.buffer_transform, SctkTransform::_90);
        pre_surface.sync_buffer_scale_and_transform(2, SctkTransform::_180);
        assert_eq!(pre_surface.buffer_scale, 2);
        assert_eq!(pre_surface.buffer_transform, SctkTransform::_180);

        // Verify real wire transmission of scale + transform when a client
        // surface exists.
        let wl_surface = client.compositor.as_ref().unwrap().create_surface(&qh, ());
        wl_surface.set_buffer_scale(pre_surface.buffer_scale);
        wl_surface.set_buffer_transform(pre_surface.buffer_transform);
        wl_surface.commit();
        roundtrip(&mut display, &mut server, &conn, &mut queue, &mut client);
        assert_eq!(server.last_commit, Some((2, SmithayWlTransform::_180)));
    }

    #[test]
    fn test_select_x11_parent_id_priority_and_edge_cases() {
        let win_a = ParentCandidate {
            id: 1u32,
            x11_window_id: 101,
            pid: Some(1000),
            geometry: rect(560, 240, 800, 600),
            role: CandidateRole::Toplevel,
        };
        let win_b_same_pos_other_app = ParentCandidate {
            id: 2u32,
            x11_window_id: 102,
            pid: Some(2000),
            geometry: rect(560, 240, 800, 600),
            role: CandidateRole::Toplevel,
        };
        let win_a2_monitor2 = ParentCandidate {
            id: 3u32,
            x11_window_id: 103,
            pid: Some(1000),
            geometry: rect(2800, 420, 800, 600),
            role: CandidateRole::Toplevel,
        };
        let popup_a = ParentCandidate {
            id: 4u32,
            x11_window_id: 104,
            pid: Some(1000),
            geometry: rect(700, 350, 200, 180),
            role: CandidateRole::Popup,
        };
        let candidates = [win_a, win_b_same_pos_other_app, win_a2_monitor2, popup_a];

        // 1. Explicit WM_TRANSIENT_FOR always wins (unless self).
        assert_eq!(
            select_x11_parent_id(
                99,
                Some(101),
                Some(1000),
                rect(2900, 500, 150, 100),
                Some(3),
                Some(2),
                &candidates,
            ),
            Some(1)
        );
        // Self-transient does not return self.
        assert_eq!(
            select_x11_parent_id(
                1,
                Some(101),
                Some(1000),
                rect(600, 300, 150, 100),
                Some(2),
                Some(1),
                &candidates,
            ),
            Some(4)
        );
        // Unmapped/roleless WM_TRANSIENT_FOR (unknown window id) falls back cleanly.
        assert_eq!(
            select_x11_parent_id(
                99,
                Some(9999),
                Some(2000),
                rect(620, 310, 150, 100),
                Some(2),
                Some(1),
                &candidates,
            ),
            Some(2)
        );

        // 2. Stale pointer on another app (PID 2000) must NOT steal a menu
        // spawned by PID 1000: falls back to keyboard focus / PID match.
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                Some(1000),
                rect(620, 310, 150, 100),
                Some(2),
                Some(1),
                &candidates,
            ),
            Some(1)
        );
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                Some(1000),
                rect(2850, 480, 150, 100),
                Some(2),
                Some(2),
                &candidates,
            ),
            Some(3)
        );

        // 3. Keyboard-triggered context menu on monitor 1 (win_a) while pointer
        // hovers a second window of the SAME PID on monitor 2 (win_a2_monitor2):
        // proximity check picks win_a instead of monitor 2!
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                Some(1000),
                rect(620, 310, 150, 100),
                Some(3),
                Some(1),
                &candidates,
            ),
            Some(1)
        );

        // 4. Popup mapped at (0, 0) before initial ConfigureNotify still picks
        // the focused same-PID window (last_pointer first).
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                Some(1000),
                rect(0, 0, 150, 100),
                Some(3),
                Some(1),
                &candidates,
            ),
            Some(3)
        );

        // 5. Nested submenu spawned while pointer is inside `popup_a` parents
        // to `popup_a` rather than the toplevel behind it.
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                Some(1000),
                rect(895, 380, 140, 120),
                Some(4),
                Some(1),
                &candidates,
            ),
            Some(4)
        );

        // 6. PID-less client (e.g. forwarded X11 app without _NET_WM_PID) uses
        // live pointer/keyboard focus, then geometry fallback.
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                None,
                rect(600, 300, 100, 80),
                Some(2),
                Some(1),
                &candidates,
            ),
            Some(2)
        );
        assert_eq!(
            select_x11_parent_id(
                99,
                None,
                None,
                rect(2850, 480, 100, 80),
                None,
                None,
                &candidates,
            ),
            Some(3)
        );
    }
}
