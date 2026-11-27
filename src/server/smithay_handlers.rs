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

/// Handlers for events from Smithay.
use std::collections::HashSet;
use std::mem;
use std::os::fd::OwnedFd;
use std::time::Duration;

use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::input::pointer::CursorImageStatus as SmithayCursorImageStatus;
use smithay::input::Seat;
use smithay::input::SeatHandler;
use smithay::input::SeatState;
use smithay::reexports::calloop::timer::TimeoutAction;
use smithay::reexports::calloop::timer::Timer;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as XdgDecorationMode;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::Mode as KdeDecorationMode;
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::OrgKdeKwinServerDecoration;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::backend::ClientData;
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::backend::DisconnectReason;
use smithay::reexports::wayland_server::protocol::wl_buffer;
use smithay::reexports::wayland_server::protocol::wl_output;
use smithay::reexports::wayland_server::protocol::wl_seat;
use smithay::reexports::wayland_server::protocol::wl_surface;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::WEnum;
use smithay::utils::Serial;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor;
use smithay::wayland::compositor::BufferAssignment as SmithayBufferAssignment;
use smithay::wayland::compositor::CompositorClientState;
use smithay::wayland::compositor::CompositorHandler;
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::compositor::SubsurfaceCachedState;
use smithay::wayland::compositor::SurfaceAttributes;
use smithay::wayland::compositor::SurfaceData;
use smithay::wayland::output::OutputHandler;
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::SelectionSource;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::primary_selection::PrimarySelectionHandler;
use smithay::wayland::selection::primary_selection::PrimarySelectionState;
use smithay::wayland::shell::kde::decoration::KdeDecorationHandler;
use smithay::wayland::shell::kde::decoration::KdeDecorationState;
use smithay::wayland::shell::xdg::Configure;
use smithay::wayland::shell::xdg::PopupSurface;
use smithay::wayland::shell::xdg::PositionerState;
use smithay::wayland::shell::xdg::SurfaceCachedState;
use smithay::wayland::shell::xdg::ToplevelSurface;
use smithay::wayland::shell::xdg::XdgShellHandler;
use smithay::wayland::shell::xdg::XdgShellState;
use smithay::wayland::shell::xdg::XdgToplevelSurfaceData;
use smithay::wayland::shell::xdg::decoration::XdgDecorationHandler;
use smithay::wayland::shm::ShmHandler;
use smithay::wayland::shm::ShmState;
use smithay::wayland::viewporter::ViewportCachedState;

use crate::compositor_utils;
use crate::constants;
use crate::filtering;
use crate::prelude::*;
use crate::serialization;
use crate::serialization::Request;
use crate::serialization::SendType;
use crate::serialization::SerializerWriter;
use crate::serialization::geometry::Rectangle;
use crate::serialization::wayland::BufferAssignment;
use crate::serialization::wayland::BufferMetadata;
use crate::serialization::wayland::DataRequest;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceRequest;
use crate::serialization::wayland::DataTransferTarget;
use crate::serialization::wayland::Role;
use crate::serialization::wayland::SourceMetadata;
use crate::serialization::wayland::SubSurfaceState;
use crate::serialization::wayland::SubsurfacePosition;
use crate::serialization::wayland::SurfaceRequest;
use crate::serialization::wayland::SurfaceRequestPayload;
use crate::serialization::wayland::SurfaceState;
use crate::serialization::wayland::Transform;
use crate::serialization::wayland::WlSurfaceId;
use crate::serialization::xdg_shell::DecorationMode;
use crate::serialization::xdg_shell::Move;
use crate::serialization::xdg_shell::PopupRequest;
use crate::serialization::xdg_shell::PopupRequestPayload;
use crate::serialization::xdg_shell::Resize;
use crate::serialization::xdg_shell::ToplevelRequest;
use crate::serialization::xdg_shell::ToplevelRequestPayload;
use crate::serialization::xdg_shell::XdgPopupState;
use crate::serialization::xdg_shell::XdgPositioner;
use crate::serialization::xdg_shell::XdgSurfaceState;
use crate::serialization::xdg_shell::XdgToplevelState;
use crate::server::LockedSurfaceState;
use crate::server::SyncedSurfaceBuffer;
use crate::server::WprsServerState;

impl BufferHandler for WprsServerState {
    #[instrument(skip(self), level = "debug")]
    fn buffer_destroyed(&mut self, buffer: &wl_buffer::WlBuffer) {}
}

impl WprsServerState {
    fn send_toplevel_request(&self, toplevel: &ToplevelSurface, payload: ToplevelRequestPayload) {
        let surface = toplevel.wl_surface();
        let Some(client) = surface.client() else {
            return;
        };
        let surface_id = WlSurfaceId::new(surface);
        if !self.committed_surfaces.contains(&surface_id) {
            return;
        }
        self.serializer
            .writer()
            .send(SendType::Object(Request::Toplevel(ToplevelRequest {
                client: serialization::ClientId::new(&client),
                surface: surface_id,
                payload,
            })))
    }

    pub fn update_state_and_send_toplevel_request<F>(
        &mut self,
        surface: &ToplevelSurface,
        mut toplevel_state_mutator: F,
        payload_to_send: ToplevelRequestPayload,
    ) where
        F: FnMut(&mut XdgToplevelState),
    {
        // Changing maximized/fullscreen is not idempotent, and can be ignored by the compositor,
        // so we must send every request we receive once the surface has been committed to `wprsc`.
        // Before the initial commit (or while disconnected), record the state so the initial
        // `Commit` (or reconnect sync) applies it without sending a `ToplevelRequest` for an
        // unknown surface on `wprsc`.
        let surface_id = WlSurfaceId::new(surface.wl_surface());
        let is_committed =
            self.serializer.other_end_connected() && self.committed_surfaces.contains(&surface_id);
        compositor::with_states(surface.wl_surface(), |surface_data| {
            let locked_state = surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .expect("LockedSurfaceState inserted in new_toplevel");
            let tracked = &mut locked_state.0.lock().unwrap();

            if let Some(Role::XdgToplevel(toplevel_state)) = &mut tracked.state.role {
                toplevel_state_mutator(toplevel_state);
                if !is_committed {
                    tracked.dirty = true;
                }
            }

            if is_committed {
                self.send_toplevel_request(surface, payload_to_send);
            }
        });

        if !self.serializer.other_end_connected() && surface.is_initial_configure_sent() {
            self.send_detached_toplevel_configure(surface);
        }
    }

    fn send_detached_toplevel_configure(&self, toplevel: &ToplevelSurface) {
        let (decoration_mode, maximized, fullscreen) =
            compositor::with_states(toplevel.wl_surface(), |surface_data| {
                surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .and_then(|locked| match &locked.0.lock().unwrap().state.role {
                        Some(Role::XdgToplevel(tl)) => {
                            Some((tl.decoration_mode, tl.maximized, tl.fullscreen))
                        },
                        _ => None,
                    })
                    .unwrap_or((None, None, None))
            });
        toplevel.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Activated);
            match maximized {
                Some(true) => {
                    state.states.set(xdg_toplevel::State::Maximized);
                },
                Some(false) => {
                    state.states.unset(xdg_toplevel::State::Maximized);
                },
                None => {},
            }
            match fullscreen {
                Some(true) => {
                    state.states.set(xdg_toplevel::State::Fullscreen);
                },
                Some(false) => {
                    state.states.unset(xdg_toplevel::State::Fullscreen);
                },
                None => {},
            }
            state.decoration_mode = Some(
                decoration_mode
                    .map(Into::into)
                    .unwrap_or(XdgDecorationMode::ClientSide),
            );
        });
        toplevel.send_configure();
    }
}

impl XdgShellHandler for WprsServerState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    #[instrument(skip(self), level = "debug")]
    fn new_toplevel(&mut self, toplevel: ToplevelSurface) {
        log_and_return!(self.insert_surface(toplevel.wl_surface()));
        let surface_id = WlSurfaceId::new(toplevel.wl_surface());
        self.committed_surfaces.remove(&surface_id);
        compositor::with_states(toplevel.wl_surface(), |surface_data| {
            let tracked = &mut surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .unwrap()
                .0
                .lock()
                .unwrap();
            tracked.state.role = Some(Role::XdgToplevel(Default::default()));
            tracked.has_committed = false;
            tracked.dirty = true;
        });
        // Defer the initial configure until the client performs its initial
        // commit (per `xdg-shell` and `xdg-decoration-unstable-v1`): when
        // `wprsc` is connected, `handle_toplevel_configure` forwards the host
        // compositor's configure; when detached, `commit` emits the fallback
        // configure after `zxdg_toplevel_decoration_v1` has been created.
    }

    #[instrument(skip(self), level = "debug")]
    fn toplevel_destroyed(&mut self, toplevel: ToplevelSurface) {
        // If client() returns None, the surface was already destroyed and an
        // appropriate message would have been sent to the client, so we don't
        // need to worry about destroying the toplevel.
        if toplevel.wl_surface().client().is_some() {
            let surface_id = WlSurfaceId::new(toplevel.wl_surface());
            self.synced_buffers.remove(&surface_id);
            self.dirty_sync_surfaces.remove(&surface_id);
            compositor::with_states(toplevel.wl_surface(), |surface_data| {
                let mut tracked = surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .expect("LockedSurfaceState inserted in new_toplevel")
                    .0
                    .lock()
                    .unwrap();
                tracked.state.role = None;
                tracked.has_committed = false;
                tracked.dirty = true;
            });

            self.send_toplevel_request(&toplevel, ToplevelRequestPayload::Destroyed);
            self.committed_surfaces.remove(&surface_id);
        }
    }

    #[instrument(skip(self))]
    fn new_popup(&mut self, popup: PopupSurface, positioner: PositionerState) {
        log_and_return!(self.insert_surface(popup.wl_surface()));
        let surface_id = WlSurfaceId::new(popup.wl_surface());
        self.committed_surfaces.remove(&surface_id);

        // Uses with_states internally and with_states is not reentrant.
        let popup_state = log_and_return!(XdgPopupState::new(&popup, &positioner));
        compositor::with_states(popup.wl_surface(), |surface_data| {
            let tracked = &mut surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .unwrap()
                .0
                .lock()
                .unwrap();
            tracked.state.role = Some(Role::XdgPopup(popup_state));
            tracked.has_committed = false;
            tracked.dirty = true;
        });
    }

    #[instrument(skip(self), level = "debug")]
    fn popup_destroyed(&mut self, popup: PopupSurface) {
        // If client() returns None, the surface was already destroyed and an
        // appropriate message would have been sent to the client, so we don't
        // need to worry about destroying the popup.
        if let Some(client) = popup.wl_surface().client() {
            let surface_id = WlSurfaceId::new(popup.wl_surface());
            self.synced_buffers.remove(&surface_id);
            self.dirty_sync_surfaces.remove(&surface_id);
            let was_committed = self.committed_surfaces.remove(&surface_id);
            // Uses with_states internally and with_states is not reentrant.
            compositor::with_states(popup.wl_surface(), |surface_data| {
                let mut tracked = surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .expect("LockedSurfaceState inserted in new_popup")
                    .0
                    .lock()
                    .unwrap();
                tracked.state.role = None;
                tracked.has_committed = false;
                tracked.dirty = true;
            });

            if was_committed {
                self.serializer
                    .writer()
                    .send(SendType::Object(Request::Popup(PopupRequest {
                        client: serialization::ClientId::new(&client),
                        surface: surface_id,
                        payload: PopupRequestPayload::Destroyed,
                    })));
            }
        };
    }

    fn grab(&mut self, surface: PopupSurface, _seat: wl_seat::WlSeat, serial: Serial) {
        if !self.serializer.other_end_connected() {
            return;
        }
        let surface_id = WlSurfaceId::new(surface.wl_surface());
        if self.committed_surfaces.contains(&surface_id) {
            warn!("Ignoring popup grab request for already-committed surface {surface_id:?}.");
            return;
        }
        let Some(host_serial) = self.serial_map.get(serial) else {
            warn!("Received popup grab request with unknown serial {serial:?}.");
            return;
        };

        compositor::with_states(surface.wl_surface(), |surface_data| {
            let Some(locked_state) = surface_data.data_map.get::<LockedSurfaceState>() else {
                return;
            };
            let mut tracked = locked_state.0.lock().unwrap();
            if let Some(Role::XdgPopup(popup_state)) = &mut tracked.state.role {
                popup_state.grab_serial = Some(host_serial);
                tracked.dirty = true;
            }
        });
    }

    fn ack_configure(&mut self, _surface: wl_surface::WlSurface, _configure: Configure) {}

    // TODO: implement ClientId from WLSurface constructor
    fn maximize_request(&mut self, surface: ToplevelSurface) {
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.maximized = Some(true),
            ToplevelRequestPayload::SetMaximized,
        );
    }

    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.maximized = Some(false),
            ToplevelRequestPayload::UnsetMaximized,
        );
    }

    fn fullscreen_request(
        &mut self,
        surface: ToplevelSurface,
        _output: Option<wl_output::WlOutput>,
    ) {
        // TODO: do anything with output? Probably not, but also depends on how
        // exactly we handle output enter/exit events and updating outputs
        // between client reconnections.
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.fullscreen = Some(true),
            ToplevelRequestPayload::SetFullscreen,
        );
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        self.update_state_and_send_toplevel_request(
            &surface,
            |toplevel_state| toplevel_state.fullscreen = Some(false),
            ToplevelRequestPayload::UnsetFullscreen,
        );
    }

    fn minimize_request(&mut self, surface: ToplevelSurface) {
        self.send_toplevel_request(&surface, ToplevelRequestPayload::SetMinimized);
    }

    fn move_request(&mut self, surface: ToplevelSurface, _seat: wl_seat::WlSeat, serial: Serial) {
        let Some(client_serial) = self.serial_map.remove(serial) else {
            warn!("Received move request with unknown serial {serial:?}.");
            return;
        };

        self.send_toplevel_request(
            &surface,
            ToplevelRequestPayload::Move(Move {
                serial: client_serial,
            }),
        );
    }

    fn resize_request(
        &mut self,
        surface: ToplevelSurface,
        _seat: wl_seat::WlSeat,
        serial: Serial,
        edges: xdg_toplevel::ResizeEdge,
    ) {
        let Some(client_serial) = self.serial_map.remove(serial) else {
            warn!("Received resize request with unknown serial {serial:?}.");
            return;
        };

        self.send_toplevel_request(
            &surface,
            ToplevelRequestPayload::Resize(Resize {
                serial: client_serial,
                edge: edges.into(),
            }),
        );
    }

    fn reposition_request(&mut self, popup: PopupSurface, positioner: PositionerState, token: u32) {
        popup.with_pending_state(|state| {
            state.positioner = positioner;
            state.geometry = positioner.get_geometry();
        });

        let surface = popup.wl_surface();
        let surface_id = WlSurfaceId::new(surface);
        let is_committed =
            self.serializer.other_end_connected() && self.committed_surfaces.contains(&surface_id);
        let xdg_positioner = XdgPositioner::new(&positioner);
        let surface_order = is_committed.then(|| get_surface_z_order(surface));
        compositor::with_states(surface, |surface_data| {
            let locked_state = surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .expect("LockedSurfaceState inserted in new_popup");
            let tracked = &mut locked_state.0.lock().unwrap();

            if let Some(Role::XdgPopup(popup_state)) = &mut tracked.state.role {
                popup_state.positioner = xdg_positioner;
                popup_state.reposition_token = Some(token);
            } else {
                error!("reposition called on surface that wasn't a popup");
                return;
            }

            if let Some(surface_order) = surface_order {
                tracked.state.z_ordered_children = surface_order;
                let surface_state_to_send = tracked.state.clone_without_buffer();
                let client = surface_state_to_send.client;
                let surface = surface_state_to_send.id;
                self.serializer
                    .writer()
                    .send(SendType::Object(Request::Surface(SurfaceRequest {
                        client,
                        surface,
                        payload: SurfaceRequestPayload::Commit(surface_state_to_send),
                    })));
                if tracked.state.buffer.is_some() {
                    tracked.dirty = false;
                }
            } else {
                tracked.dirty = true;
            }
        });

        if !is_committed {
            popup.send_repositioned(token);
        }
    }

    // TODO: show_window_menu
}

impl SelectionHandler for WprsServerState {
    type SelectionUserData = ();

    #[instrument(skip(self, _seat), level = "debug")]
    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        _seat: Seat<Self>,
    ) {
        if let Some(source) = source {
            self.serializer
                .writer()
                .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                    DataSourceRequest::SetSelection(
                        match ty {
                            SelectionTarget::Clipboard => DataSource::Selection,
                            SelectionTarget::Primary => DataSource::Primary,
                        },
                        SourceMetadata::from_mime_types(source.mime_types()),
                    ),
                ))));
        }
    }

    #[instrument(skip(self, fd, _seat, _user_data), level = "debug")]
    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        _user_data: &Self::SelectionUserData,
    ) {
        let target = match ty {
            SelectionTarget::Clipboard => DataTransferTarget::Selection,
            SelectionTarget::Primary => DataTransferTarget::Primary,
        };
        self.request_host_data_transfer(target, mime_type, fd);
    }
}

impl PrimarySelectionHandler for WprsServerState {
    fn primary_selection_state(&self) -> &PrimarySelectionState {
        &self.primary_selection_state
    }
}

impl CompositorHandler for WprsServerState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    #[instrument(skip(self), level = "debug")]
    fn commit(&mut self, surface: &WlSurface) {
        commit(surface, self).log_and_ignore(loc!());
    }
}

fn take_dirty_sync_descendants<N>(
    dirty: &mut HashSet<WlSurfaceId>,
    surface: &N,
    children: &impl Fn(&N) -> Vec<N>,
    is_sync: &impl Fn(&N) -> bool,
    id: &impl Fn(&N) -> WlSurfaceId,
) -> Vec<WlSurfaceId> {
    let mut taken = Vec::new();
    for child in children(surface) {
        let child_sync = is_sync(&child);
        let child_id = id(&child);
        let child_was_dirty = dirty.remove(&child_id);
        if child_was_dirty {
            taken.push(child_id);
        }
        if child_sync || child_was_dirty {
            taken.extend(take_dirty_sync_descendants(
                dirty, &child, children, is_sync, id,
            ));
        }
    }
    taken
}

#[instrument(ret, level = "debug")]
pub fn get_child_positions(surface: &WlSurface) -> Vec<SubsurfacePosition> {
    compositor::get_children(surface)
        .iter()
        .map(|child| SubsurfacePosition {
            id: WlSurfaceId::new(child),
            position: compositor::with_states(child, |surface_data| {
                surface_data
                    .cached_state
                    .get::<SubsurfaceCachedState>()
                    .pending()
                    .location
                    .into()
            }),
        })
        .collect()
}

pub(crate) fn get_surface_z_order(surface: &WlSurface) -> Vec<SubsurfacePosition> {
    let mut surface_order = get_child_positions(surface);
    // TODO: get actual surface position, for now put it at the end
    surface_order.push(SubsurfacePosition {
        id: WlSurfaceId::new(surface),
        position: (0, 0).into(),
    });
    surface_order
}

pub(crate) fn sync_subsurface_role(
    surface_state: &mut SurfaceState,
    is_subsurface_role: bool,
    parent_id: Option<WlSurfaceId>,
) {
    if !is_subsurface_role {
        return;
    }

    match parent_id {
        Some(parent_id) => match &mut surface_state.role {
            Some(Role::SubSurface(subsurface_state)) => {
                subsurface_state.parent = parent_id;
            },
            _ => {
                surface_state.role = Some(Role::SubSurface(SubSurfaceState {
                    parent: parent_id,
                    location: (0, 0).into(),
                    sync: true,
                }));
            },
        },
        None => {
            debug!("NO PARENT FOR SUBSURFACE {:?}", surface_state);
            surface_state.role = None;
        },
    }
}

#[instrument(skip_all, level = "debug")]
pub fn commit(surface: &WlSurface, state: &mut WprsServerState) -> Result<()> {
    let surface_order = get_surface_z_order(surface);

    // TODO: https://github.com/Smithay/smithay/issues/538 - move into commit.
    let parent = compositor::get_parent(surface);
    // Smithay resets `SubsurfaceState::sync` to `true` when `wl_subsurface` is
    // destroyed while clearing `parent`; a parentless surface has no ancestor
    // to flush it.
    let sync = parent.is_some() && compositor::is_sync_subsurface(surface);
    let surface_id = WlSurfaceId::new(surface);
    // Smithay applies a desync root's effectively-sync descendants before the
    // root (and `on_commit_buffer_handler` leaves sync subsurfaces' `NewBuffer`
    // in place): each sync descendant sends its buffer/`Commit` and marks
    // itself in `dirty_sync_surfaces`, and the desync root drains those marks
    // here to force `Commit(root)` so `wprsc`'s `commit_sync_children` flushes
    // them.
    let taken_dirty = if sync {
        Vec::new()
    } else {
        take_dirty_sync_descendants(
            &mut state.dirty_sync_surfaces,
            surface,
            &compositor::get_children,
            &compositor::is_sync_subsurface,
            &WlSurfaceId::new,
        )
    };
    let children_dirty = !taken_dirty.is_empty();

    let res = state
        .insert_surface(surface)
        .location(loc!())
        .and_then(|()| {
            compositor::with_states(surface, |surface_data| {
                commit_impl(
                    surface,
                    surface_data,
                    state,
                    sync,
                    parent,
                    surface_order,
                    children_dirty,
                )
            })
            .location(loc!())
        });
    let sent = match res {
        Ok(sent) => sent,
        Err(err) => {
            // Restore drained descendant marks so an errored parent commit does
            // not leave already-applied sync children unflushed.
            state.dirty_sync_surfaces.extend(taken_dirty);
            return Err(err);
        },
    };
    if sync {
        if sent {
            state.dirty_sync_surfaces.insert(surface_id);
        }
    } else {
        state.dirty_sync_surfaces.remove(&surface_id);
    }
    on_commit_buffer_handler::<WprsServerState>(surface);
    // When no `wprsc` is connected to supply a host configure, answer the
    // initial (or post-unmap remap) xdg-shell commit directly so applications
    // receive their initial configure instead of blocking indefinitely.
    if !state.serializer.other_end_connected() {
        if let Some(toplevel) = state
            .xdg_shell_state
            .toplevel_surfaces()
            .iter()
            .find(|t| t.wl_surface() == surface)
            .cloned()
            && !toplevel.is_initial_configure_sent()
        {
            state.send_detached_toplevel_configure(&toplevel);
        } else if let Some(popup) = state
            .xdg_shell_state
            .popup_surfaces()
            .iter()
            .find(|p| p.wl_surface() == surface)
            .cloned()
            && !popup.is_initial_configure_sent()
        {
            popup.send_configure().log_and_ignore(loc!());
        }
    }
    Ok(())
}

// TODO: maybe make these methods on the relevant states

#[instrument(skip_all, level = "debug")]
pub fn set_regions(surface_attributes: &SurfaceAttributes, surface_state: &mut SurfaceState) {
    surface_state.input_region = surface_attributes.input_region.as_ref().map(Into::into);
    surface_state.opaque_region = surface_attributes.opaque_region.as_ref().map(Into::into);
}

#[instrument(skip_all, level = "debug")]
pub fn set_transformation(
    surface_attributes: &SurfaceAttributes,
    surface_state: &mut SurfaceState,
) {
    surface_state.buffer_scale = surface_attributes.buffer_scale;
    surface_state.buffer_transform = Some(surface_attributes.buffer_transform.into());
}

pub fn set_viewport_state(viewport_state: &ViewportCachedState, surface_state: &mut SurfaceState) {
    surface_state.viewport_state = Some(viewport_state.into());
}

#[instrument(skip_all, level = "debug")]
pub fn set_xdg_surface_attributes(surface_data: &SurfaceData, surface_state: &mut SurfaceState) {
    if surface_data.cached_state.has::<SurfaceCachedState>() {
        let mut guard = surface_data.cached_state.get::<SurfaceCachedState>();
        let surface_cached_state = guard.current();

        let xdg_surface_state = XdgSurfaceState {
            window_geometry: surface_cached_state
                .geometry
                .as_ref()
                .map(|geometry| (*geometry).into()),
            max_size: surface_cached_state.max_size.into(),
            min_size: surface_cached_state.min_size.into(),
        };
        surface_state.xdg_surface_state = Some(xdg_surface_state);
    }
}

#[instrument(skip_all, level = "debug")]
pub fn set_xdg_toplevel_attributes(
    surface_data: &SurfaceData,
    toplevel_state: &mut XdgToplevelState,
) -> Result<()> {
    let toplevel_attributes = surface_data
        .data_map
        .get::<XdgToplevelSurfaceData>()
        .location(loc!())?
        .lock()
        .unwrap();
    // Be careful about not moving objects out of
    // toplevel_attributes here.
    toplevel_state.parent = toplevel_attributes.parent.as_ref().map(WlSurfaceId::new);
    toplevel_state.title.clone_from(&toplevel_attributes.title);
    toplevel_state
        .app_id
        .clone_from(&toplevel_attributes.app_id);

    // toplevel_state.maximized = toplevel_attributes
    //     .current
    //     .states
    //     .contains(xdg_toplevel::State::Maximized);

    // match toplevel_attributes.current.decoration_mode {
    //     Some(
    // }

    // dbg!("TOPLEVEL ATTRIBUTES", &toplevel_attributes);
    Ok(())
}

#[allow(clippy::iter_with_drain)]
#[instrument(skip(state), level = "debug")]
pub fn commit_impl(
    surface: &WlSurface,
    surface_data: &SurfaceData,
    state: &mut WprsServerState,
    sync: bool,
    parent: Option<WlSurface>,
    surface_order: Vec<SubsurfacePosition>,
    children_dirty: bool,
) -> Result<bool> {
    let tracked = &mut surface_data
        .data_map
        .get::<LockedSurfaceState>()
        .location(loc!())?
        .0
        .lock()
        .unwrap();
    let connected = state.serializer.other_end_connected();
    let surface_id = tracked.state.id;
    let was_committed = state.committed_surfaces.contains(&surface_id);
    let was_dirty = tracked.dirty;
    let prev_without_buffer = connected.then(|| tracked.state.clone_without_buffer());
    let surface_state = &mut tracked.state;

    sync_subsurface_role(
        surface_state,
        matches!(surface_data.role, Some("subsurface")),
        parent.as_ref().map(WlSurfaceId::new),
    );

    surface_state.z_ordered_children = surface_order;

    let mut guard = surface_data.cached_state.get::<ViewportCachedState>();
    let viewport_state = guard.current();

    let mut guard = surface_data.cached_state.get::<SurfaceAttributes>();
    let surface_attributes = guard.current();
    compositor_utils::cap_surface_damage(&mut surface_attributes.damage);
    let mut frame_callbacks = mem::take(&mut surface_attributes.frame_callbacks);

    if !frame_callbacks.is_empty() {
        let surface = surface.clone();
        state
            .lh
            .insert_source(
                Timer::from_duration(
                    state
                        .frame_interval
                        // "The server should give some time for the client to
                        // draw and commit after sending the frame callback
                        // events to let it hit the next output refresh."
                        .saturating_sub(Duration::from_millis(2)),
                ),
                move |_, _, state| {
                    if !surface.is_alive() {
                        return TimeoutAction::Drop;
                    }

                    if state.serializer.other_end_connected() {
                        // We can't use into_iter() because we can't move
                        // frame_callbacks because this is a FnMut. However, this
                        // works because this branch will only ever be taken once.
                        for callback in frame_callbacks.drain(..) {
                            debug!(
                                "Sending callback for surface {:?}: {:?}",
                                surface.id(),
                                callback.id()
                            );
                            callback.done(state.start_time.elapsed().as_millis() as u32);
                        }
                        TimeoutAction::Drop
                    } else {
                        TimeoutAction::ToDuration(state.frame_interval)
                    }
                },
            )
            .expect("timer registration should never fail");
    }

    set_regions(surface_attributes, surface_state);
    set_transformation(surface_attributes, surface_state);
    set_viewport_state(viewport_state, surface_state);
    set_xdg_surface_attributes(surface_data, surface_state);

    match &mut surface_state.role {
        Some(Role::Cursor(_)) => {},
        Some(Role::SubSurface(subsurface_state)) => {
            subsurface_state.sync = sync;
            subsurface_state.location = surface_data
                .cached_state
                .get::<SubsurfaceCachedState>()
                .pending()
                .location
                .into();
        },
        Some(Role::XdgToplevel(toplevel_state)) => {
            set_xdg_toplevel_attributes(surface_data, toplevel_state).location(loc!())?;
        },
        Some(Role::XdgPopup(_)) => {},
        None => {},
    }

    debug!("buffer assignment: {:?}", &surface_attributes.buffer);
    let mut damage_to_send = None;
    let mut buffer_to_send = None;

    match &surface_attributes.buffer {
        Some(SmithayBufferAssignment::NewBuffer(buffer)) => {
            enum CommitBufferOutcome {
                Disconnected,
                Unchanged {
                    synced: SyncedSurfaceBuffer,
                },
                Sent {
                    synced: SyncedSurfaceBuffer,
                    metadata: BufferMetadata,
                    damage: Option<Vec<Rectangle<i32>>>,
                },
            }

            // Remove the existing shadow buffer before reading `wl_shm` so a
            // `BadMap` or validation failure never leaves a half-updated shadow
            // in `synced_buffers`.
            let prev_synced = state.synced_buffers.remove(&surface_state.id);
            let mut raw_buffer_sent = false;
            let writer = state.serializer.writer();
            let commit_buf_res =
                compositor_utils::with_buffer_contents(buffer, |data, spec| -> Result<_> {
                    let metadata = surface_state.set_buffer(&spec).location(loc!())?;
                    if !connected {
                        return Ok(CommitBufferOutcome::Disconnected);
                    }

                    match prev_synced {
                        Some(mut synced) if synced.metadata == metadata => {
                            // Use partial row-bands when `< 75%` of height is
                            // damaged to avoid scanning undamaged rows, while
                            // still running full-frame incremental diffing on
                            // `>= 75%` or undamaged commits to skip unchanged
                            // `128x128` quadrants.
                            let scan_bands = (!surface_attributes.damage.is_empty()).then(|| {
                                let damage = compositor_utils::damage_to_buffer_rects(
                                    &surface_attributes.damage,
                                    surface_state.buffer_scale,
                                    surface_state
                                        .buffer_transform
                                        .unwrap_or(Transform::Normal)
                                        .into(),
                                    surface_state.viewport_state.as_ref(),
                                    (metadata.width, metadata.height),
                                );
                                filtering::compute_partial_row_bands(
                                    &damage,
                                    metadata.width,
                                    metadata.height,
                                )
                            });
                            let scan_bands = scan_bands.flatten();
                            compositor_utils::fault_probe_buffer(
                                data,
                                metadata.stride as usize,
                                scan_bands.as_deref(),
                            );
                            let dirty_tiles = filtering::filter_and_compress_incremental_streaming(
                                data,
                                surface_state.client,
                                surface_state.id,
                                metadata,
                                state.compression_mode,
                                scan_bands.as_deref(),
                                &mut synced.pixels,
                                &mut state.compressor,
                                |raw_buffer_to_send| {
                                    raw_buffer_sent = true;
                                    writer.send(SendType::RawBuffer(raw_buffer_to_send));
                                },
                            );
                            if dirty_tiles.is_empty() {
                                return Ok(CommitBufferOutcome::Unchanged { synced });
                            }
                            let damage =
                                (dirty_tiles.len() < constants::SENT_DAMAGE_LIMIT).then(|| {
                                    dirty_tiles
                                        .into_iter()
                                        .map(|r| {
                                            Rectangle::new(
                                                r.x as i32, r.y as i32, r.w as i32, r.h as i32,
                                            )
                                        })
                                        .collect()
                                });
                            Ok(CommitBufferOutcome::Sent {
                                synced,
                                metadata,
                                damage,
                            })
                        },
                        other => {
                            let mut pixels = other.map_or_else(Vec::new, |old| old.pixels);
                            compositor_utils::fault_probe_buffer(
                                data,
                                metadata.stride as usize,
                                None,
                            );
                            filtering::filter_and_compress_full_streaming(
                                data,
                                surface_state.client,
                                surface_state.id,
                                metadata,
                                state.compression_mode,
                                Some(&mut pixels),
                                &mut state.compressor,
                                |raw_buffer_to_send| {
                                    raw_buffer_sent = true;
                                    writer.send(SendType::RawBuffer(raw_buffer_to_send));
                                },
                            );
                            Ok(CommitBufferOutcome::Sent {
                                synced: SyncedSurfaceBuffer { metadata, pixels },
                                metadata,
                                damage: None,
                            })
                        },
                    }
                });

            match commit_buf_res {
                Ok(Ok(CommitBufferOutcome::Sent {
                    synced,
                    metadata,
                    damage,
                })) => {
                    surface_attributes.damage.clear();
                    state.synced_buffers.insert(surface_state.id, synced);
                    damage_to_send = damage;
                    buffer_to_send = Some(BufferAssignment::New(metadata));
                },
                Ok(Ok(CommitBufferOutcome::Unchanged { synced })) => {
                    surface_attributes.damage.clear();
                    state.synced_buffers.insert(surface_state.id, synced);
                },
                Ok(Ok(CommitBufferOutcome::Disconnected)) => {
                    surface_attributes.damage.clear();
                },
                Ok(Err(err)) => {
                    return Err(err).location(loc!());
                },
                Err(err) => {
                    // Smithay reports SIGBUS as `Err(BadMap)` after the
                    // closure returns, so `raw_buffer_sent` may already be
                    // `true`.
                    if raw_buffer_sent {
                        warn!(
                            "shm buffer faulted during commit ({err:?}); detaching remote surface buffer"
                        );
                        surface_attributes.damage.clear();
                        surface_state.buffer = None;
                        buffer_to_send = Some(BufferAssignment::Removed);
                    } else {
                        return Err(err).location(loc!());
                    }
                },
            }
        },
        Some(SmithayBufferAssignment::Removed) => {
            surface_attributes.damage.clear();
            surface_state.buffer = None;
            state.synced_buffers.remove(&surface_state.id);
            buffer_to_send = Some(BufferAssignment::Removed);
        },
        // Retain `surface_attributes.damage` across bufferless commits (bounded
        // by `cap_surface_damage`) so damage posted before a subsequent
        // same-size `NewBuffer` commit is included in its incremental row-bands.
        None => {},
    }

    let unmapped_xdg_surface = matches!(buffer_to_send, Some(BufferAssignment::Removed))
        && matches!(
            surface_state.role,
            Some(Role::XdgToplevel(_) | Role::XdgPopup(_))
        );
    tracked.has_committed = true;

    let Some(prev_without_buffer) = prev_without_buffer else {
        if let Some(Role::XdgPopup(popup_state)) = &mut tracked.state.role {
            popup_state.grab_serial = None;
        }
        if unmapped_xdg_surface {
            tracked.dirty = true;
        }
        return Ok(false);
    };

    let mut surface_state_to_send = tracked.state.clone_without_buffer();
    surface_state_to_send.buffer = buffer_to_send;
    if was_committed
        && !was_dirty
        && surface_state_to_send.buffer.is_none()
        && surface_state_to_send == prev_without_buffer
        && !children_dirty
    {
        return Ok(false);
    }

    surface_state_to_send.damage = damage_to_send;

    let client = surface_state_to_send.client;
    let surface = surface_state_to_send.id;
    state
        .serializer
        .writer()
        .send(SendType::Object(Request::Surface(SurfaceRequest {
            client,
            surface,
            payload: SurfaceRequestPayload::Commit(surface_state_to_send),
        })));
    state.committed_surfaces.insert(surface);
    if let Some(Role::XdgPopup(popup_state)) = &mut tracked.state.role {
        popup_state.grab_serial = None;
    }
    // Keep `dirty = true` after an `xdg_shell` unmap (`attach(NULL)`) so the
    // subsequent bufferless remap commit is not deduplicated away.
    tracked.dirty = unmapped_xdg_surface;
    Ok(true)
}

impl ShmHandler for WprsServerState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl SeatHandler for WprsServerState {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    #[instrument(skip(self, _seat), level = "debug")]
    fn cursor_image(&mut self, _seat: &Seat<Self>, image: SmithayCursorImageStatus) {
        self.cursor_status = Some(image.clone());
        self.send_cursor_status(&image);
    }
}

impl WprsServerState {
    fn set_decoration_mode(&mut self, surface: &WlSurface, mode: Option<DecorationMode>) -> bool {
        let surface_id = WlSurfaceId::new(surface);
        let is_committed =
            self.serializer.other_end_connected() && self.committed_surfaces.contains(&surface_id);
        let surface_order = is_committed.then(|| get_surface_z_order(surface));
        compositor::with_states(surface, |surface_data| {
            let Some(locked_state) = surface_data.data_map.get::<LockedSurfaceState>() else {
                return false;
            };
            let tracked = &mut locked_state.0.lock().unwrap();

            // Even though the KDE server decoration protocol can send
            // decorations for any surface, only forward them for XdgToplevel
            // since that's what wprs currently expects.
            let Some(Role::XdgToplevel(toplevel_state)) = &mut tracked.state.role else {
                return false;
            };
            let changed = toplevel_state.decoration_mode != mode;
            toplevel_state.decoration_mode = mode;

            if let Some(surface_order) = surface_order {
                // `zxdg_toplevel_decoration_v1::{set_mode, unset_mode}` are not
                // double-buffered: forward mode changes immediately so `wprsc`
                // updates the host compositor without waiting for a commit.
                if changed {
                    tracked.state.z_ordered_children = surface_order;
                    let surface_state_to_send = tracked.state.clone_without_buffer();
                    let client = surface_state_to_send.client;
                    let surface = surface_state_to_send.id;
                    self.serializer
                        .writer()
                        .send(SendType::Object(Request::Surface(SurfaceRequest {
                            client,
                            surface,
                            payload: SurfaceRequestPayload::Commit(surface_state_to_send),
                        })));
                    if tracked.state.buffer.is_some() {
                        tracked.dirty = false;
                    }
                    return true;
                }
            } else {
                tracked.dirty = true;
            }
            false
        })
    }

    fn send_post_commit_decoration_configure(&self, toplevel: &ToplevelSurface) {
        if !self.serializer.other_end_connected() {
            self.send_detached_toplevel_configure(toplevel);
        } else {
            toplevel.send_configure();
        }
    }
}

impl XdgDecorationHandler for WprsServerState {
    #[instrument(skip(self), level = "debug")]
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        // Wait for the initial commit before sending the initial configure.
        // Only emit a configure here if the toplevel was already configured
        // before creating its decoration object so the client still receives
        // the mandatory initial `zxdg_toplevel_decoration_v1.configure`.
        if toplevel.is_initial_configure_sent() {
            self.send_post_commit_decoration_configure(&toplevel);
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: XdgDecorationMode) {
        let mode: Option<DecorationMode> = mode.try_into().log(loc!()).ok();
        if mode.is_some() {
            let forwarded = self.set_decoration_mode(toplevel.wl_surface(), mode);
            if !forwarded && toplevel.is_initial_configure_sent() {
                self.send_post_commit_decoration_configure(&toplevel);
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        let forwarded = self.set_decoration_mode(toplevel.wl_surface(), None);
        if !forwarded && toplevel.is_initial_configure_sent() {
            self.send_post_commit_decoration_configure(&toplevel);
        }
    }
}

impl KdeDecorationHandler for WprsServerState {
    fn kde_decoration_state(&self) -> &KdeDecorationState {
        &self.kde_decoration_state
    }

    #[instrument(skip(self, _surface, _decoration), level = "debug")]
    fn new_decoration(&mut self, _surface: &WlSurface, _decoration: &OrgKdeKwinServerDecoration) {}

    #[instrument(skip(self, decoration), level = "debug")]
    fn request_mode(
        &mut self,
        surface: &WlSurface,
        decoration: &OrgKdeKwinServerDecoration,
        mode: WEnum<KdeDecorationMode>,
    ) {
        let mode = mode.into_result().log(loc!()).ok();
        if let Some(mode) = mode {
            decoration.mode(mode);
        }

        let mode: Option<DecorationMode> = mode.and_then(|m| m.try_into().log(loc!()).ok());
        if mode.is_some() {
            self.set_decoration_mode(surface, mode);
        };
    }

    #[instrument(skip(self, _decoration), level = "debug")]
    fn release(&mut self, _decoration: &OrgKdeKwinServerDecoration, surface: &WlSurface) {
        self.set_decoration_mode(surface, None);
    }
}

pub struct ClientState {
    compositor_state: CompositorClientState,
    pub writer: SerializerWriter<Request>,
}

impl ClientState {
    pub fn new(writer: SerializerWriter<Request>) -> Self {
        Self {
            compositor_state: CompositorClientState::default(),
            writer,
        }
    }
}

impl ClientData for ClientState {
    #[instrument(skip(self), level = "debug")]
    fn initialized(&self, client_id: ClientId) {}

    #[instrument(skip(self), level = "debug")]
    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        self.writer
            .send(SendType::Object(Request::ClientDisconnected(
                client_id.into(),
            )));
    }
}

impl OutputHandler for WprsServerState {}

smithay::delegate_compositor!(WprsServerState);
smithay::delegate_xdg_shell!(WprsServerState);
smithay::delegate_xdg_decoration!(WprsServerState);
smithay::delegate_kde_decoration!(WprsServerState);
smithay::delegate_shm!(WprsServerState);
smithay::delegate_seat!(WprsServerState);
smithay::delegate_output!(WprsServerState);
smithay::delegate_primary_selection!(WprsServerState);
smithay::delegate_viewporter!(WprsServerState);

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use crossbeam_channel::Receiver;
    use smithay::reexports::calloop::EventLoop;
    use smithay::reexports::wayland_server::Display;
    use smithay_client_toolkit::reexports::client::Connection;
    use smithay_client_toolkit::reexports::client::Dispatch;
    use smithay_client_toolkit::reexports::client::EventQueue;
    use smithay_client_toolkit::reexports::client::Proxy;
    use smithay_client_toolkit::reexports::client::QueueHandle;
    use smithay_client_toolkit::reexports::client::protocol::wl_buffer::WlBuffer;
    use smithay_client_toolkit::reexports::client::protocol::wl_callback;
    use smithay_client_toolkit::reexports::client::protocol::wl_compositor::WlCompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_output;
    use smithay_client_toolkit::reexports::client::protocol::wl_pointer;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry;
    use smithay_client_toolkit::reexports::client::protocol::wl_seat as client_seat;
    use smithay_client_toolkit::reexports::client::protocol::wl_shm;
    use smithay_client_toolkit::reexports::client::protocol::wl_shm_pool::WlShmPool;
    use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::WlSubcompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::WlSubsurface;
    use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;
    use smithay_client_toolkit::reexports::protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
    use smithay_client_toolkit::reexports::protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_popup;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_positioner;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_surface;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_toplevel;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_wm_base::XdgWmBase;
    use smithay_client_toolkit::reexports::protocols::xdg::xdg_output::zv1::client::zxdg_output_manager_v1::ZxdgOutputManagerV1;
    use smithay_client_toolkit::reexports::protocols::xdg::xdg_output::zv1::client::zxdg_output_v1;

    use super::*;
    use crate::serialization::ClientId as WireClientId;
    use crate::serialization::Event;
    use crate::serialization::RecvType;
    use crate::serialization::Serializer;
    use crate::serialization::xdg_shell::ToplevelConfigure;
    use crate::serialization::xdg_shell::ToplevelEvent;

    #[derive(Debug, Eq, PartialEq)]
    enum ClientConfigureEvent {
        Decoration(zxdg_toplevel_decoration_v1::Mode),
        Toplevel {
            width: i32,
            height: i32,
            maximized: bool,
            activated: bool,
        },
        Popup {
            x: i32,
            y: i32,
            width: i32,
            height: i32,
        },
        PopupRepositioned(u32),
        PopupDone,
        SurfaceConfigure,
    }

    #[derive(Debug, Default, Eq, PartialEq)]
    struct WireOutputSnapshot {
        wl_geometry: Option<(i32, i32)>,
        wl_mode: Option<(i32, i32, i32, bool)>,
        wl_scale: Option<i32>,
        wl_name: Option<String>,
        xdg_logical_position: Option<(i32, i32)>,
        xdg_logical_size: Option<(i32, i32)>,
        xdg_name: Option<String>,
    }

    #[derive(Default)]
    struct WireClientData {
        compositor: Option<WlCompositor>,
        subcompositor: Option<WlSubcompositor>,
        shm: Option<wl_shm::WlShm>,
        seat: Option<client_seat::WlSeat>,
        pointer: Option<wl_pointer::WlPointer>,
        last_button_serial: Option<u32>,
        xdg_wm_base: Option<XdgWmBase>,
        xdg_decoration_manager: Option<ZxdgDecorationManagerV1>,
        xdg_output_manager: Option<ZxdgOutputManagerV1>,
        wl_outputs: HashMap<u32, wl_output::WlOutput>,
        xdg_outputs: HashMap<u32, zxdg_output_v1::ZxdgOutputV1>,
        output_snapshots: HashMap<u32, WireOutputSnapshot>,
        configures: Vec<ClientConfigureEvent>,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for WireClientData {
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
                    "wl_subcompositor" => {
                        state.subcompositor = Some(registry.bind::<WlSubcompositor, _, _>(
                            name,
                            version.min(1),
                            qh,
                            (),
                        ));
                    },
                    "wl_shm" => {
                        state.shm = Some(registry.bind::<wl_shm::WlShm, _, _>(
                            name,
                            version.min(1),
                            qh,
                            (),
                        ));
                    },
                    "wl_seat" => {
                        state.seat = Some(registry.bind::<client_seat::WlSeat, _, _>(
                            name,
                            version.min(7),
                            qh,
                            (),
                        ));
                    },
                    "xdg_wm_base" => {
                        state.xdg_wm_base =
                            Some(registry.bind::<XdgWmBase, _, _>(name, version.min(6), qh, ()));
                    },
                    "zxdg_decoration_manager_v1" => {
                        state.xdg_decoration_manager =
                            Some(registry.bind::<ZxdgDecorationManagerV1, _, _>(
                                name,
                                version.min(1),
                                qh,
                                (),
                            ));
                    },
                    "zxdg_output_manager_v1" => {
                        let manager = registry.bind::<ZxdgOutputManagerV1, _, _>(
                            name,
                            version.min(3),
                            qh,
                            (),
                        );
                        for (&global_name, wl_out) in &state.wl_outputs {
                            state
                                .xdg_outputs
                                .entry(global_name)
                                .or_insert_with(|| manager.get_xdg_output(wl_out, qh, global_name));
                        }
                        state.xdg_output_manager = Some(manager);
                    },
                    "wl_output" => {
                        let wl_out = registry.bind::<wl_output::WlOutput, _, _>(
                            name,
                            version.min(4),
                            qh,
                            name,
                        );
                        if let Some(manager) = &state.xdg_output_manager {
                            state
                                .xdg_outputs
                                .insert(name, manager.get_xdg_output(&wl_out, qh, name));
                        }
                        state.wl_outputs.insert(name, wl_out);
                        state.output_snapshots.entry(name).or_default();
                    },
                    _ => {},
                }
            } else if let wl_registry::Event::GlobalRemove { name } = event {
                state.xdg_outputs.remove(&name);
                state.wl_outputs.remove(&name);
                state.output_snapshots.remove(&name);
            }
        }
    }

    impl Dispatch<wl_callback::WlCallback, Arc<AtomicBool>> for WireClientData {
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

    impl Dispatch<WlCompositor, ()> for WireClientData {
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

    impl Dispatch<WlSubcompositor, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &WlSubcompositor,
            _event: <WlSubcompositor as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<ClientWlSurface, ()> for WireClientData {
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

    impl Dispatch<WlSubsurface, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &WlSubsurface,
            _event: <WlSubsurface as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<XdgWmBase, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            wm_base: &XdgWmBase,
            event: <XdgWmBase as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_wm_base::Event::Ping { serial } = event {
                wm_base.pong(serial);
            }
        }
    }

    impl Dispatch<xdg_surface::XdgSurface, ()> for WireClientData {
        fn event(
            state: &mut Self,
            xdg_surface: &xdg_surface::XdgSurface,
            event: xdg_surface::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let xdg_surface::Event::Configure { serial } = event {
                state
                    .configures
                    .push(ClientConfigureEvent::SurfaceConfigure);
                xdg_surface.ack_configure(serial);
            }
        }
    }

    impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &xdg_toplevel::XdgToplevel,
            event: xdg_toplevel::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let xdg_toplevel::Event::Configure {
                width,
                height,
                states,
            } = event
            {
                let mut maximized = false;
                let mut activated = false;
                for &chunk in states.as_chunks::<4>().0 {
                    let raw = u32::from_ne_bytes(chunk);
                    if raw == xdg_toplevel::State::Maximized as u32 {
                        maximized = true;
                    } else if raw == xdg_toplevel::State::Activated as u32 {
                        activated = true;
                    }
                }
                state.configures.push(ClientConfigureEvent::Toplevel {
                    width,
                    height,
                    maximized,
                    activated,
                });
            }
        }
    }

    impl Dispatch<xdg_positioner::XdgPositioner, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &xdg_positioner::XdgPositioner,
            _event: xdg_positioner::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<xdg_popup::XdgPopup, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &xdg_popup::XdgPopup,
            event: xdg_popup::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                xdg_popup::Event::Configure {
                    x,
                    y,
                    width,
                    height,
                } => {
                    state.configures.push(ClientConfigureEvent::Popup {
                        x,
                        y,
                        width,
                        height,
                    });
                },
                xdg_popup::Event::Repositioned { token } => {
                    state
                        .configures
                        .push(ClientConfigureEvent::PopupRepositioned(token));
                },
                xdg_popup::Event::PopupDone => {
                    state.configures.push(ClientConfigureEvent::PopupDone);
                },
                _ => {},
            }
        }
    }

    impl Dispatch<wl_shm::WlShm, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &wl_shm::WlShm,
            _event: wl_shm::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<client_seat::WlSeat, ()> for WireClientData {
        fn event(
            state: &mut Self,
            seat: &client_seat::WlSeat,
            event: client_seat::Event,
            _data: &(),
            _conn: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let client_seat::Event::Capabilities {
                capabilities: smithay_client_toolkit::reexports::client::WEnum::Value(caps),
            } = event
                && caps.contains(client_seat::Capability::Pointer)
                && state.pointer.is_none()
            {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
        }
    }

    impl Dispatch<wl_pointer::WlPointer, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &wl_pointer::WlPointer,
            event: wl_pointer::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let wl_pointer::Event::Button { serial, .. } = event {
                state.last_button_serial = Some(serial);
            }
        }
    }

    impl Dispatch<WlShmPool, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &WlShmPool,
            _event: <WlShmPool as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<WlBuffer, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &WlBuffer,
            _event: <WlBuffer as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<ZxdgDecorationManagerV1, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &ZxdgDecorationManagerV1,
            _event: <ZxdgDecorationManagerV1 as Proxy>::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, ()> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
            event: zxdg_toplevel_decoration_v1::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let zxdg_toplevel_decoration_v1::Event::Configure {
                mode: smithay_client_toolkit::reexports::client::WEnum::Value(mode),
            } = event
            {
                state
                    .configures
                    .push(ClientConfigureEvent::Decoration(mode));
            }
        }
    }

    impl Dispatch<ZxdgOutputManagerV1, ()> for WireClientData {
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

    impl Dispatch<wl_output::WlOutput, u32> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &wl_output::WlOutput,
            event: wl_output::Event,
            global_name: &u32,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            let entry = state.output_snapshots.entry(*global_name).or_default();
            match event {
                wl_output::Event::Geometry { x, y, .. } => {
                    entry.wl_geometry = Some((x, y));
                },
                wl_output::Event::Mode {
                    flags: smithay_client_toolkit::reexports::client::WEnum::Value(flags),
                    width,
                    height,
                    refresh,
                } => {
                    if flags.contains(wl_output::Mode::Current) {
                        entry.wl_mode = Some((
                            width,
                            height,
                            refresh,
                            flags.contains(wl_output::Mode::Preferred),
                        ));
                    }
                },
                wl_output::Event::Scale { factor } => {
                    entry.wl_scale = Some(factor);
                },
                wl_output::Event::Name { name } => {
                    entry.wl_name = Some(name);
                },
                _ => {},
            }
        }
    }

    impl Dispatch<zxdg_output_v1::ZxdgOutputV1, u32> for WireClientData {
        fn event(
            state: &mut Self,
            _proxy: &zxdg_output_v1::ZxdgOutputV1,
            event: zxdg_output_v1::Event,
            global_name: &u32,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            let entry = state.output_snapshots.entry(*global_name).or_default();
            match event {
                zxdg_output_v1::Event::LogicalPosition { x, y } => {
                    entry.xdg_logical_position = Some((x, y));
                },
                zxdg_output_v1::Event::LogicalSize { width, height } => {
                    entry.xdg_logical_size = Some((width, height));
                },
                zxdg_output_v1::Event::Name { name } => {
                    entry.xdg_name = Some(name);
                },
                _ => {},
            }
        }
    }

    struct WaylandTestHarness {
        display: Display<WprsServerState>,
        _event_loop: EventLoop<'static, WprsServerState>,
        state: WprsServerState,
        conn: Connection,
        event_queue: EventQueue<WireClientData>,
        qh: QueueHandle<WireClientData>,
        client_data: WireClientData,
        rx: Option<Receiver<SendType<Request>>>,
        conn_id: Option<u64>,
    }

    impl WaylandTestHarness {
        fn new(connected: bool) -> Self {
            let serializer: Serializer<Request, Event> = Serializer::new_test_server();
            let display: Display<WprsServerState> = Display::new().unwrap();
            let event_loop: EventLoop<WprsServerState> = EventLoop::try_new().unwrap();
            let mut state = WprsServerState::new(
                display.handle(),
                event_loop.handle(),
                serializer,
                false,
                Duration::from_millis(16),
                false,
            );
            state.seat.add_pointer();

            let (server_sock, client_sock) = UnixStream::pair().unwrap();
            display
                .handle()
                .insert_client(
                    server_sock,
                    Arc::new(ClientState::new(state.serializer.writer())),
                )
                .unwrap();

            let conn = Connection::from_socket(client_sock).unwrap();
            let event_queue: EventQueue<WireClientData> = conn.new_event_queue();
            let qh = event_queue.handle();
            let _registry = conn.display().get_registry(&qh, ());

            let mut harness = Self {
                display,
                _event_loop: event_loop,
                state,
                conn,
                event_queue,
                qh,
                client_data: WireClientData::default(),
                rx: None,
                conn_id: None,
            };
            harness.roundtrip();
            harness.roundtrip();
            if connected {
                harness.connect_wprsc();
                harness.drain_requests();
            }
            harness
        }

        fn connect_wprsc(&mut self) {
            if let Some(conn_id) = self.conn_id.take() {
                self.state.serializer.disconnect_test_client(conn_id);
            }
            let (conn_id, rx) = self.state.serializer.connect_test_client();
            self.conn_id = Some(conn_id);
            self.rx = Some(rx);
            self.state.handle_event(conn_id, RecvType::Connected);
        }

        fn roundtrip(&mut self) {
            let done = Arc::new(AtomicBool::new(false));
            let _cb = self.conn.display().sync(&self.qh, Arc::clone(&done));
            self.conn.flush().unwrap();
            self.display.dispatch_clients(&mut self.state).unwrap();
            self.display.flush_clients().unwrap();
            while !done.load(Ordering::Acquire) {
                self.event_queue
                    .blocking_dispatch(&mut self.client_data)
                    .unwrap();
            }
        }

        fn drain_requests(&self) -> Vec<Request> {
            let Some(rx) = &self.rx else {
                return Vec::new();
            };
            let mut out = Vec::new();
            while let Ok(msg) = rx.try_recv() {
                if let SendType::Object(req) = msg {
                    out.push(req);
                }
            }
            out
        }

        fn server_surface_id(&self, surface: &ClientWlSurface) -> WlSurfaceId {
            let protocol_id = surface.id().protocol_id();
            *self
                .state
                .object_map
                .iter()
                .find_map(|(sid, oid)| (oid.protocol_id() == protocol_id).then_some(sid))
                .expect("surface not registered in object_map")
        }
    }

    fn make_surface_state(id: u64, role: Option<Role>) -> SurfaceState {
        SurfaceState {
            client: WireClientId(1),
            id: WlSurfaceId(id),
            buffer: None,
            role,
            buffer_scale: 1,
            buffer_transform: None,
            opaque_region: None,
            input_region: None,
            z_ordered_children: Vec::new(),
            damage: None,
            output_ids: Vec::new(),
            viewport_state: None,
            xdg_surface_state: None,
        }
    }

    #[test]
    fn test_sync_subsurface_role() {
        let p1 = WlSurfaceId(10);
        let p2 = WlSurfaceId(20);

        // Initial attach creates `Role::SubSurface`.
        let mut state = make_surface_state(1, None);
        sync_subsurface_role(&mut state, true, Some(p1));
        assert_eq!(
            state.role,
            Some(Role::SubSurface(SubSurfaceState {
                parent: p1,
                location: (0, 0).into(),
                sync: true,
            }))
        );

        // Reparenting updates `parent` in place.
        sync_subsurface_role(&mut state, true, Some(p2));
        assert_eq!(
            state.role,
            Some(Role::SubSurface(SubSurfaceState {
                parent: p2,
                location: (0, 0).into(),
                sync: true,
            }))
        );

        // Detached subsurface clears `role` to `None`.
        sync_subsurface_role(&mut state, true, None);
        assert_eq!(state.role, None);

        // Non-subsurface roles are untouched.
        let mut toplevel = make_surface_state(2, Some(Role::XdgToplevel(Default::default())));
        sync_subsurface_role(&mut toplevel, false, None);
        assert!(matches!(toplevel.role, Some(Role::XdgToplevel(_))));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_subsurface_reparent_and_detach_wire() {
        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let subcompositor = h.client_data.subcompositor.clone().unwrap();
        let xdg_wm_base = h.client_data.xdg_wm_base.clone().unwrap();

        let p1 = compositor.create_surface(&h.qh, ());
        let p2 = compositor.create_surface(&h.qh, ());
        let child = compositor.create_surface(&h.qh, ());

        // Attach `child` to `p1` (desync, offset (10, 20)) and commit both.
        let sub1 = subcompositor.get_subsurface(&child, &p1, &h.qh, ());
        sub1.set_desync();
        sub1.set_position(10, 20);
        child.commit();
        p1.commit();
        h.roundtrip();

        let p1_id = h.server_surface_id(&p1);
        let child_id = h.server_surface_id(&child);
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == child_id
                && st.role == Some(Role::SubSurface(SubSurfaceState {
                    parent: p1_id,
                    location: (10, 20).into(),
                    sync: false,
                }))
        )));

        // Destroy `p1` BEFORE `sub1` (orphans `child`, leaving `child` desync
        // and parentless): `child`'s next commit must clear `role` to `None`
        // and must not leak `child_id` into `dirty_sync_surfaces`.
        p1.destroy();
        child.commit();
        h.roundtrip();
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Destroyed,
                ..
            }) if *surface == p1_id
        )));
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == child_id && st.role.is_none()
        )));
        assert!(!h.state.dirty_sync_surfaces.contains(&child_id));

        // Destroy old inert `sub1` and reparent the reused `child` onto `p2`
        // (default sync=true, offset (30, 40)), like Firefox reusing a tooltip
        // child surface across tooltip windows.
        sub1.destroy();
        let sub2 = subcompositor.get_subsurface(&child, &p2, &h.qh, ());
        sub2.set_position(30, 40);
        child.commit();
        p2.commit();
        h.roundtrip();

        let p2_id = h.server_surface_id(&p2);
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == child_id
                && st.role == Some(Role::SubSurface(SubSurfaceState {
                    parent: p2_id,
                    location: (30, 40).into(),
                    sync: true,
                }))
        )));
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == p2_id
                && st.z_ordered_children.contains(&SubsurfacePosition {
                    id: child_id,
                    position: (30, 40).into(),
                })
        )));

        // Direct reparent from live `p2` to toplevel `p3` without an
        // intermediate detached commit on `child` must update `parent` to `p3`
        // and survive reconnect `sync_surface_tree`.
        let p3 = compositor.create_surface(&h.qh, ());
        let p3_xdg = xdg_wm_base.get_xdg_surface(&p3, &h.qh, ());
        let _p3_toplevel = p3_xdg.get_toplevel(&h.qh, ());
        sub2.destroy();
        let _sub3 = subcompositor.get_subsurface(&child, &p3, &h.qh, ());
        child.commit();
        p3.commit();
        h.roundtrip();

        let p3_id = h.server_surface_id(&p3);
        let reqs = h.drain_requests();
        assert!(reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == child_id
                && matches!(
                    st.role,
                    Some(Role::SubSurface(SubSurfaceState { parent, .. })) if parent == p3_id
                )
        )));

        // Reconnect: `sync_surface_tree` must send live `parent == p3_id` and
        // live `z_ordered_children` without deadlocking `PrivateSurfaceData`.
        h.connect_wprsc();
        let sync_reqs = h.drain_requests();
        assert!(sync_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == p3_id
                && st.z_ordered_children.iter().any(|c| c.id == child_id)
        )));
        assert!(sync_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == child_id
                && matches!(
                    st.role,
                    Some(Role::SubSurface(SubSurfaceState { parent, .. })) if parent == p3_id
                )
        )));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_pre_commit_toplevel_requests_and_reuse_commits_wire() {
        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();
        let xdg_wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let dec_mgr = h.client_data.xdg_decoration_manager.clone().unwrap();

        // 1. Electron startup sequence: `get_toplevel` + `get_toplevel_decoration`
        // + `set_mode(ServerSide)` + `unset_maximized()` BEFORE initial commit.
        // No `Request::Toplevel` may be sent before the initial commit.
        let surface = compositor.create_surface(&h.qh, ());
        let xdg_surf = xdg_wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel = xdg_surf.get_toplevel(&h.qh, ());
        let dec = dec_mgr.get_toplevel_decoration(&toplevel, &h.qh, ());
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        toplevel.unset_maximized();
        h.roundtrip();

        let pre_commit_reqs = h.drain_requests();
        assert!(
            pre_commit_reqs.is_empty(),
            "expected no requests before initial commit, got {pre_commit_reqs:?}"
        );

        // Initial bufferless commit carries the buffered `maximized` and
        // `decoration_mode` state.
        surface.commit();
        h.roundtrip();
        let surface_id = h.server_surface_id(&surface);
        let init_reqs = h.drain_requests();
        assert!(init_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == surface_id
                && matches!(
                    &st.role,
                    Some(Role::XdgToplevel(tl))
                        if tl.maximized == Some(false)
                            && tl.decoration_mode == Some(DecorationMode::Server)
                )
        )));

        // 2. Post-commit `set_maximized` forwards `ToplevelRequest` and
        // post-commit `set_mode(ClientSide)` forwards an immediate bufferless
        // `Commit` without waiting for `wl_surface.commit()`.
        toplevel.set_maximized();
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ClientSide);
        h.roundtrip();
        let post_reqs = h.drain_requests();
        assert!(post_reqs.iter().any(|r| matches!(
            r,
            Request::Toplevel(ToplevelRequest {
                surface,
                payload: ToplevelRequestPayload::SetMaximized,
                ..
            }) if *surface == surface_id
        )));
        assert!(post_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == surface_id
                && matches!(
                    &st.role,
                    Some(Role::XdgToplevel(tl))
                        if tl.decoration_mode == Some(DecorationMode::Client)
                )
        )));

        // 3. `attach(buffer)` -> `attach(NULL)` unmap -> bufferless remap commit
        // must NOT be deduplicated away.
        let shm_path = std::env::temp_dir().join(format!(
            "wprs_shm_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let shm_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&shm_path)
            .unwrap();
        shm_file.set_len(4).unwrap();
        _ = std::fs::remove_file(&shm_path);
        let pool = shm.create_pool(shm_file.as_fd(), 4, &h.qh, ());
        let buffer = pool.create_buffer(0, 1, 1, 4, wl_shm::Format::Argb8888, &h.qh, ());

        surface.attach(Some(&buffer), 0, 0);
        surface.commit();
        h.roundtrip();
        h.drain_requests();

        surface.attach(None, 0, 0);
        surface.commit();
        h.roundtrip();
        let unmap_reqs = h.drain_requests();
        assert!(unmap_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(st),
                ..
            }) if *surface == surface_id && st.buffer == Some(BufferAssignment::Removed)
        )));

        // Remap commit (bufferless, identical metadata) must still be sent.
        surface.commit();
        h.roundtrip();
        let remap_reqs = h.drain_requests();
        assert!(
            remap_reqs.iter().any(|r| matches!(
                r,
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                }) if *surface == surface_id && st.buffer.is_none()
            )),
            "expected bufferless remap Commit after attach(NULL), got {remap_reqs:?}"
        );

        // 4. Electron `Hide()` -> `Show()` reusing the same `wl_surface`:
        // destroying `xdg_toplevel` sends `Destroyed`, recreating it suppresses
        // pre-commit `unset_maximized()`, and the bufferless initial commit is
        // forwarded instead of deduplicated against the mutated `SurfaceState`.
        dec.destroy();
        toplevel.destroy();
        xdg_surf.destroy();
        h.roundtrip();
        let hide_reqs = h.drain_requests();
        assert!(hide_reqs.iter().any(|r| matches!(
            r,
            Request::Toplevel(ToplevelRequest {
                surface,
                payload: ToplevelRequestPayload::Destroyed,
                ..
            }) if *surface == surface_id
        )));

        let xdg_surf2 = xdg_wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel2 = xdg_surf2.get_toplevel(&h.qh, ());
        toplevel2.unset_maximized();
        h.roundtrip();
        assert!(
            h.drain_requests().is_empty(),
            "recreated toplevel must not send pre-commit requests"
        );

        surface.commit();
        h.roundtrip();
        let show_reqs = h.drain_requests();
        assert!(
            show_reqs.iter().any(|r| matches!(
                r,
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(st),
                    ..
                }) if *surface == surface_id
                    && matches!(&st.role, Some(Role::XdgToplevel(tl)) if tl.maximized == Some(false))
            )),
            "reused wl_surface initial bufferless commit was dropped: {show_reqs:?}"
        );

        // 5. Uncommitted toplevel is skipped by reconnect `sync_surface_tree`
        // and does not emit `ToplevelRequestPayload::Destroyed` if destroyed
        // before committing.
        let uncommitted_surf = compositor.create_surface(&h.qh, ());
        let uncommitted_xdg = xdg_wm_base.get_xdg_surface(&uncommitted_surf, &h.qh, ());
        let uncommitted_tl = uncommitted_xdg.get_toplevel(&h.qh, ());
        h.roundtrip();
        let uncommitted_id = h.server_surface_id(&uncommitted_surf);

        h.connect_wprsc();
        let sync_reqs = h.drain_requests();
        assert!(sync_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest {
                surface,
                payload: SurfaceRequestPayload::Commit(_),
                ..
            }) if *surface == surface_id
        )));
        assert!(!sync_reqs.iter().any(|r| matches!(
            r,
            Request::Surface(SurfaceRequest { surface, .. }) if *surface == uncommitted_id
        )));

        uncommitted_tl.destroy();
        uncommitted_xdg.destroy();
        h.roundtrip();
        let destroy_uncommitted_reqs = h.drain_requests();
        assert!(
            !destroy_uncommitted_reqs.iter().any(|r| matches!(
                r,
                Request::Toplevel(ToplevelRequest { surface, .. }) if *surface == uncommitted_id
            )),
            "destroying uncommitted toplevel sent premature Destroyed: {destroy_uncommitted_reqs:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_connected_toplevel_defers_configure_until_host_configure_wire() {
        let mut h = WaylandTestHarness::new(true);
        let compositor = h.client_data.compositor.clone().unwrap();
        let xdg_wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let dec_mgr = h.client_data.xdg_decoration_manager.clone().unwrap();

        let surface = compositor.create_surface(&h.qh, ());
        let xdg_surf = xdg_wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel = xdg_surf.get_toplevel(&h.qh, ());
        let dec = dec_mgr.get_toplevel_decoration(&toplevel, &h.qh, ());
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        toplevel.unset_maximized();
        h.roundtrip();

        // No configure before initial commit.
        assert!(
            h.client_data.configures.is_empty(),
            "received premature configure before initial commit: {:?}",
            h.client_data.configures
        );

        // Initial commit forwards to `wprsc` and waits for the host configure.
        surface.commit();
        h.roundtrip();
        assert!(
            h.client_data.configures.is_empty(),
            "connected server must wait for host configure: {:?}",
            h.client_data.configures
        );

        // Simulated host `WindowHandler::configure` from `wprsc` delivers
        // `zxdg_toplevel_decoration_v1.configure` + `xdg_toplevel.configure` +
        // `xdg_surface.configure` atomically.
        let surface_id = h.server_surface_id(&surface);
        let conn_id = h.conn_id.unwrap();
        let mut host_configure =
            smithay_client_toolkit::shell::xdg::window::WindowConfigure::default();
        host_configure.new_size = (
            std::num::NonZeroU32::new(800),
            std::num::NonZeroU32::new(600),
        );
        host_configure.decoration_mode =
            smithay_client_toolkit::shell::xdg::window::DecorationMode::Server;
        host_configure.state = smithay_client_toolkit::reexports::csd_frame::WindowState::ACTIVATED;
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Toplevel(ToplevelEvent::Configure(
                ToplevelConfigure::from_smithay(&surface_id, host_configure),
            ))),
        );
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Decoration(zxdg_toplevel_decoration_v1::Mode::ServerSide),
                ClientConfigureEvent::Toplevel {
                    width: 800,
                    height: 600,
                    maximized: false,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );

        // Post-commit no-op `set_mode(ServerSide)` still receives a configure.
        h.client_data.configures.clear();
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        h.roundtrip();
        assert_eq!(
            h.client_data.configures.last(),
            Some(&ClientConfigureEvent::SurfaceConfigure)
        );

        // Late `get_toplevel_decoration` created AFTER initial configure still
        // receives its mandatory initial `zxdg_toplevel_decoration_v1.configure`.
        let surf2 = compositor.create_surface(&h.qh, ());
        let xdg2 = xdg_wm_base.get_xdg_surface(&surf2, &h.qh, ());
        let tl2 = xdg2.get_toplevel(&h.qh, ());
        surf2.commit();
        h.roundtrip();
        let surf2_id = h.server_surface_id(&surf2);
        let mut host_configure2 =
            smithay_client_toolkit::shell::xdg::window::WindowConfigure::default();
        host_configure2.decoration_mode =
            smithay_client_toolkit::shell::xdg::window::DecorationMode::Client;
        host_configure2.state =
            smithay_client_toolkit::reexports::csd_frame::WindowState::ACTIVATED;
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Toplevel(ToplevelEvent::Configure(
                ToplevelConfigure::from_smithay(&surf2_id, host_configure2),
            ))),
        );
        h.roundtrip();
        h.client_data.configures.clear();

        let _dec2 = dec_mgr.get_toplevel_decoration(&tl2, &h.qh, ());
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Decoration(zxdg_toplevel_decoration_v1::Mode::ClientSide),
                ClientConfigureEvent::Toplevel {
                    width: 0,
                    height: 0,
                    maximized: false,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_detached_toplevel_initial_and_post_commit_configures_wire() {
        let mut h = WaylandTestHarness::new(false);
        let compositor = h.client_data.compositor.clone().unwrap();
        let shm = h.client_data.shm.clone().unwrap();
        let xdg_wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let dec_mgr = h.client_data.xdg_decoration_manager.clone().unwrap();

        let surface = compositor.create_surface(&h.qh, ());
        let xdg_surf = xdg_wm_base.get_xdg_surface(&surface, &h.qh, ());
        let toplevel = xdg_surf.get_toplevel(&h.qh, ());
        let dec = dec_mgr.get_toplevel_decoration(&toplevel, &h.qh, ());
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        toplevel.set_maximized();
        h.roundtrip();

        // Still no configure before initial commit while detached.
        assert!(h.client_data.configures.is_empty());

        // Initial commit emits decoration + toplevel (Activated + Maximized) +
        // surface configure atomically.
        surface.commit();
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Decoration(zxdg_toplevel_decoration_v1::Mode::ServerSide),
                ClientConfigureEvent::Toplevel {
                    width: 0,
                    height: 0,
                    maximized: true,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );

        // Post-commit `set_mode(ClientSide)` while detached emits a new
        // decoration configure + surface configure without waiting for commit.
        h.client_data.configures.clear();
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ClientSide);
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Decoration(zxdg_toplevel_decoration_v1::Mode::ClientSide),
                ClientConfigureEvent::Toplevel {
                    width: 0,
                    height: 0,
                    maximized: true,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );

        // Post-commit `unset_maximized` while detached emits updated configure.
        h.client_data.configures.clear();
        toplevel.unset_maximized();
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Toplevel {
                    width: 0,
                    height: 0,
                    maximized: false,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );

        // Map -> `attach(NULL)` unmap (no configure on unmap) -> bufferless
        // remap commit emits a fresh initial decoration + surface configure.
        let shm_path = std::env::temp_dir().join(format!(
            "wprs_detached_shm_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let shm_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&shm_path)
            .unwrap();
        shm_file.set_len(4).unwrap();
        _ = std::fs::remove_file(&shm_path);
        let pool = shm.create_pool(shm_file.as_fd(), 4, &h.qh, ());
        let buffer = pool.create_buffer(0, 1, 1, 4, wl_shm::Format::Argb8888, &h.qh, ());

        h.client_data.configures.clear();
        surface.attach(Some(&buffer), 0, 0);
        surface.commit();
        h.roundtrip();
        assert!(h.client_data.configures.is_empty());

        surface.attach(None, 0, 0);
        surface.commit();
        h.roundtrip();
        assert!(
            h.client_data.configures.is_empty(),
            "unmap commit must not emit a premature configure"
        );

        surface.commit();
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Decoration(zxdg_toplevel_decoration_v1::Mode::ClientSide),
                ClientConfigureEvent::Toplevel {
                    width: 0,
                    height: 0,
                    maximized: false,
                    activated: true,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
    }

    #[test]
    fn test_take_dirty_sync_descendants() {
        // Tree:
        //   0 (desync root)
        //   ├─ 1 (sync)
        //   │  └─ 2 (sync)
        //   └─ 3 (desync)
        //      └─ 4 (sync)
        let tree: HashMap<u64, Vec<u64>> =
            HashMap::from([(0, vec![1, 3]), (1, vec![2]), (3, vec![4])]);
        let sync_nodes: HashSet<u64> = HashSet::from([1, 2, 4]);

        let cases: &[(&str, &[u64], &[u64], &[u64])] = &[
            // (label, initial dirty, expected taken, expected remaining)
            ("dirty sync child", &[1], &[1], &[]),
            (
                "clean sync child with dirty sync grandchild",
                &[2],
                &[2],
                &[],
            ),
            ("dirty child flipped desync", &[3, 4], &[3, 4], &[]),
            ("clean desync child with dirty grandchild", &[4], &[], &[4]),
        ];

        for &(label, initial, expected_taken, expected_remaining) in cases {
            let mut dirty: HashSet<WlSurfaceId> =
                initial.iter().copied().map(WlSurfaceId).collect();
            let mut taken = take_dirty_sync_descendants(
                &mut dirty,
                &0,
                &|n| tree.get(n).cloned().unwrap_or_default(),
                &|n| sync_nodes.contains(n),
                &|n| WlSurfaceId(*n),
            );
            taken.sort_unstable_by_key(|id| id.0);
            let expected_taken: Vec<WlSurfaceId> =
                expected_taken.iter().copied().map(WlSurfaceId).collect();
            let expected_remaining: HashSet<WlSurfaceId> = expected_remaining
                .iter()
                .copied()
                .map(WlSurfaceId)
                .collect();
            assert_eq!(taken, expected_taken, "{label}: taken");
            assert_eq!(dirty, expected_remaining, "{label}: remaining");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_xdg_output_forwards_multi_monitor_and_fractional_layout() {
        use crate::serialization::wayland::Mode as SerMode;
        use crate::serialization::wayland::OutputEvent;
        use crate::serialization::wayland::OutputInfo;
        use crate::serialization::wayland::Subpixel as SerSubpixel;
        use crate::serialization::wayland::Transform as SerTransform;

        let mut h = WaylandTestHarness::new(true);
        assert!(
            h.client_data.xdg_output_manager.is_some(),
            "wprsd must advertise zxdg_output_manager_v1"
        );
        let conn_id = h.conn_id.unwrap();

        let make_output = |id: u32,
                           name: &str,
                           dimensions: (i32, i32),
                           scale_factor: i32,
                           transform: SerTransform,
                           logical_position: (i32, i32),
                           logical_size: (i32, i32)| OutputInfo {
            id,
            model: format!("Model-{id}"),
            make: "Vendor".to_string(),
            // Sway reports `(0, 0)` on `wl_output.geometry` for every monitor.
            location: (0, 0).into(),
            physical_size: (600, 340).into(),
            subpixel: SerSubpixel::HorizontalRgb,
            transform,
            scale_factor,
            mode: SerMode {
                dimensions: dimensions.into(),
                refresh_rate: 60000,
                current: true,
                preferred: true,
            },
            logical_position: Some(logical_position.into()),
            logical_size: Some(logical_size.into()),
            name: Some(name.to_string()),
            description: None,
        };

        let left = make_output(
            1,
            "DP-1",
            (1920, 1080),
            1,
            SerTransform::Normal,
            (0, 0),
            (1920, 1080),
        );
        let middle = make_output(
            2,
            "DP-2",
            (3840, 2160),
            2,
            SerTransform::Normal,
            (1920, 0),
            (2560, 1440),
        );
        let right = make_output(
            3,
            "HDMI-A-1",
            (1920, 1080),
            2,
            SerTransform::_90,
            (4480, 0),
            (720, 1280),
        );

        for output in [left, middle.clone(), right] {
            h.state.handle_event(
                conn_id,
                RecvType::Object(Event::Output(OutputEvent::New(output))),
            );
        }
        // Two roundtrips: first discovers `wl_output` globals and sends
        // `get_xdg_output`; second receives initial `zxdg_output_v1` events.
        h.roundtrip();
        h.roundtrip();

        let by_name: HashMap<String, &WireOutputSnapshot> = h
            .client_data
            .output_snapshots
            .values()
            .map(|snap| (snap.wl_name.clone().expect("wl_output.name"), snap))
            .collect();
        assert_eq!(by_name.len(), 3);

        assert_eq!(by_name["1_DP-1"].wl_geometry, Some((0, 0)));
        assert_eq!(by_name["1_DP-1"].wl_mode, Some((1920, 1080, 60000, true)));
        assert_eq!(by_name["1_DP-1"].wl_scale, Some(1));
        assert_eq!(by_name["1_DP-1"].xdg_logical_position, Some((0, 0)));
        assert_eq!(by_name["1_DP-1"].xdg_logical_size, Some((1920, 1080)));
        assert_eq!(by_name["1_DP-1"].xdg_name.as_deref(), Some("1_DP-1"));

        assert_eq!(by_name["2_DP-2"].wl_geometry, Some((1920, 0)));
        assert_eq!(by_name["2_DP-2"].wl_mode, Some((3840, 2160, 60000, true)));
        assert_eq!(by_name["2_DP-2"].wl_scale, Some(2));
        assert_eq!(by_name["2_DP-2"].xdg_logical_position, Some((1920, 0)));
        assert_eq!(by_name["2_DP-2"].xdg_logical_size, Some((2560, 1440)));

        assert_eq!(by_name["3_HDMI-A-1"].wl_geometry, Some((4480, 0)));
        assert_eq!(
            by_name["3_HDMI-A-1"].wl_mode,
            Some((1920, 1080, 60000, true))
        );
        assert_eq!(by_name["3_HDMI-A-1"].wl_scale, Some(2));
        assert_eq!(by_name["3_HDMI-A-1"].xdg_logical_position, Some((4480, 0)));
        assert_eq!(by_name["3_HDMI-A-1"].xdg_logical_size, Some((720, 1280)));

        // Updating position, mode, and fractional scale must update both
        // `wl_output` (preserving `Preferred`) and `zxdg_output_v1`.
        let mut updated_middle = middle;
        updated_middle.mode.dimensions = (2560, 1440).into();
        updated_middle.scale_factor = 1;
        updated_middle.logical_position = Some((1920, 100).into());
        updated_middle.logical_size = Some((2048, 1152).into());
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Output(OutputEvent::Update(updated_middle.clone()))),
        );
        h.roundtrip();

        let middle_snap = h
            .client_data
            .output_snapshots
            .values()
            .find(|snap| snap.wl_name.as_deref() == Some("2_DP-2"))
            .unwrap();
        assert_eq!(middle_snap.wl_geometry, Some((1920, 100)));
        assert_eq!(middle_snap.wl_mode, Some((2560, 1440, 60000, true)));
        assert_eq!(middle_snap.wl_scale, Some(1));
        assert_eq!(middle_snap.xdg_logical_position, Some((1920, 100)));
        assert_eq!(middle_snap.xdg_logical_size, Some((2048, 1152)));

        // Destroying an output withdraws its global cleanly.
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Output(OutputEvent::Destroy(updated_middle))),
        );
        h.roundtrip();
        assert_eq!(h.client_data.output_snapshots.len(), 2);
        assert!(
            h.client_data
                .output_snapshots
                .values()
                .all(|snap| snap.wl_name.as_deref() != Some("2_DP-2"))
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_connected_popup_defers_configure_and_forwards_reposition_and_done_wire() {
        use crate::serialization::xdg_shell::PopupConfigure;
        use crate::serialization::xdg_shell::PopupConfigureKind;
        use crate::serialization::xdg_shell::PopupEvent;

        let mut h = WaylandTestHarness::new(true);
        let conn_id = h.conn_id.unwrap();
        let compositor = h.client_data.compositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();

        let parent_surface = compositor.create_surface(&h.qh, ());
        let parent_xdg = wm_base.get_xdg_surface(&parent_surface, &h.qh, ());
        let _parent_toplevel = parent_xdg.get_toplevel(&h.qh, ());
        parent_surface.commit();
        h.roundtrip();
        h.drain_requests();
        h.client_data.configures.clear();

        // Create a non-reactive popup with slide/flip constraints.
        let popup_surface = compositor.create_surface(&h.qh, ());
        let popup_xdg = wm_base.get_xdg_surface(&popup_surface, &h.qh, ());
        let pos = wm_base.create_positioner(&h.qh, ());
        pos.set_size(200, 150);
        pos.set_anchor_rect(100, 100, 1, 1);
        pos.set_anchor(xdg_positioner::Anchor::TopLeft);
        pos.set_gravity(xdg_positioner::Gravity::BottomRight);
        let constraints = xdg_positioner::ConstraintAdjustment::SlideX
            | xdg_positioner::ConstraintAdjustment::SlideY
            | xdg_positioner::ConstraintAdjustment::FlipX
            | xdg_positioner::ConstraintAdjustment::FlipY;
        pos.set_constraint_adjustment(constraints);
        let popup = popup_xdg.get_popup(Some(&parent_xdg), &pos, &h.qh, ());
        h.roundtrip();

        // No premature configure before initial commit or host response.
        assert!(h.client_data.configures.is_empty());

        popup_surface.commit();
        h.roundtrip();
        assert!(
            h.client_data.configures.is_empty(),
            "connected popup must wait for host compositor configure"
        );
        let popup_sid = h.server_surface_id(&popup_surface);
        let reqs = h.drain_requests();
        let popup_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("popup commit must be forwarded to wprsc");
        assert_eq!(
            popup_state.positioner.constraint_adjustment,
            constraints.bits()
        );
        assert_eq!(popup_state.reposition_token, None);

        // Host compositor replies with slid/flipped coordinates. Because no
        // premature configure was sent, this succeeds even though the popup is
        // not reactive.
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Popup(PopupEvent::Configure(PopupConfigure {
                surface_id: popup_sid,
                position: (40, 20).into(),
                width: 200,
                height: 150,
                kind: PopupConfigureKind::Initial,
            }))),
        );
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Popup {
                    x: 40,
                    y: 20,
                    width: 200,
                    height: 150,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
        h.client_data.configures.clear();

        // Reposition request forwards the token and waits for host reply.
        let pos2 = wm_base.create_positioner(&h.qh, ());
        pos2.set_size(220, 160);
        pos2.set_anchor_rect(120, 80, 1, 1);
        pos2.set_anchor(xdg_positioner::Anchor::TopLeft);
        pos2.set_gravity(xdg_positioner::Gravity::BottomRight);
        popup.reposition(&pos2, 42);
        h.roundtrip();
        assert!(
            h.client_data.configures.is_empty(),
            "connected reposition must wait for host constrained configure"
        );
        let reqs = h.drain_requests();
        let repositioned_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("reposition must forward updated XdgPopupState");
        assert_eq!(repositioned_state.reposition_token, Some(42));
        assert_eq!(repositioned_state.positioner.width, 220);
        assert_eq!(repositioned_state.positioner.height, 160);

        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Popup(PopupEvent::Configure(PopupConfigure {
                surface_id: popup_sid,
                position: (10, 15).into(),
                width: 220,
                height: 160,
                kind: PopupConfigureKind::Reposition { token: 42 },
            }))),
        );
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::PopupRepositioned(42),
                ClientConfigureEvent::Popup {
                    x: 10,
                    y: 15,
                    width: 220,
                    height: 160,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
        h.client_data.configures.clear();

        // Host dismissal forwards `xdg_popup.popup_done`.
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::Popup(PopupEvent::Done(popup_sid))),
        );
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![ClientConfigureEvent::PopupDone]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_detached_popup_initial_and_reposition_configures_wire() {
        let mut h = WaylandTestHarness::new(false);
        let compositor = h.client_data.compositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();

        let parent_surface = compositor.create_surface(&h.qh, ());
        let parent_xdg = wm_base.get_xdg_surface(&parent_surface, &h.qh, ());
        let _parent_toplevel = parent_xdg.get_toplevel(&h.qh, ());
        parent_surface.commit();
        h.roundtrip();
        h.client_data.configures.clear();

        let popup_surface = compositor.create_surface(&h.qh, ());
        let popup_xdg = wm_base.get_xdg_surface(&popup_surface, &h.qh, ());
        let pos = wm_base.create_positioner(&h.qh, ());
        pos.set_size(180, 90);
        pos.set_anchor_rect(50, 60, 1, 1);
        pos.set_anchor(xdg_positioner::Anchor::TopLeft);
        pos.set_gravity(xdg_positioner::Gravity::BottomRight);
        let popup = popup_xdg.get_popup(Some(&parent_xdg), &pos, &h.qh, ());
        h.roundtrip();
        assert!(h.client_data.configures.is_empty());

        popup_surface.commit();
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::Popup {
                    x: 50,
                    y: 60,
                    width: 180,
                    height: 90,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
        h.client_data.configures.clear();

        // Detached reposition immediately answers with the UPDATED geometry.
        let pos2 = wm_base.create_positioner(&h.qh, ());
        pos2.set_size(240, 120);
        pos2.set_anchor_rect(75, 85, 1, 1);
        pos2.set_anchor(xdg_positioner::Anchor::TopLeft);
        pos2.set_gravity(xdg_positioner::Gravity::BottomRight);
        popup.reposition(&pos2, 7);
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![
                ClientConfigureEvent::PopupRepositioned(7),
                ClientConfigureEvent::Popup {
                    x: 75,
                    y: 85,
                    width: 240,
                    height: 120,
                },
                ClientConfigureEvent::SurfaceConfigure,
            ]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_popup_grab_forwards_mapped_serial_and_skips_unmapped_or_stale_wire() {
        use crate::serialization::wayland::PointerEvent;
        use crate::serialization::wayland::PointerEventKind;
        use crate::serialization::xdg_shell::PopupEvent;

        let mut h = WaylandTestHarness::new(true);
        let conn_id = h.conn_id.unwrap();
        let compositor = h.client_data.compositor.clone().unwrap();
        let wm_base = h.client_data.xdg_wm_base.clone().unwrap();
        let seat = h.client_data.seat.clone().unwrap();

        let parent_surface = compositor.create_surface(&h.qh, ());
        let parent_xdg = wm_base.get_xdg_surface(&parent_surface, &h.qh, ());
        let _parent_toplevel = parent_xdg.get_toplevel(&h.qh, ());
        parent_surface.commit();
        h.roundtrip();
        let parent_sid = h.server_surface_id(&parent_surface);
        h.drain_requests();

        // Deliver a pointer button press carrying host serial 4242 so `wprsd`
        // records a `server_serial -> 4242` mapping in `serial_map`.
        h.state.handle_event(
            conn_id,
            RecvType::Object(Event::PointerFrame(vec![
                PointerEvent {
                    surface_id: parent_sid,
                    position: (20.0, 20.0).into(),
                    kind: PointerEventKind::Enter { serial: 4241 },
                },
                PointerEvent {
                    surface_id: parent_sid,
                    position: (20.0, 20.0).into(),
                    kind: PointerEventKind::Press {
                        button: 0x110,
                        serial: 4242,
                    },
                },
            ])),
        );
        h.roundtrip();
        let server_button_serial = h
            .client_data
            .last_button_serial
            .expect("client must receive wl_pointer.button serial");

        let make_positioner = |wm_base: &XdgWmBase, qh: &QueueHandle<WireClientData>| {
            let pos = wm_base.create_positioner(qh, ());
            pos.set_size(120, 80);
            pos.set_anchor_rect(10, 10, 1, 1);
            pos.set_anchor(xdg_positioner::Anchor::TopLeft);
            pos.set_gravity(xdg_positioner::Gravity::BottomRight);
            pos
        };

        // 1. Grabbing popup before initial commit forwards mapped host serial 4242.
        let popup1_surface = compositor.create_surface(&h.qh, ());
        let popup1_xdg = wm_base.get_xdg_surface(&popup1_surface, &h.qh, ());
        let pos1 = make_positioner(&wm_base, &h.qh);
        let popup1 = popup1_xdg.get_popup(Some(&parent_xdg), &pos1, &h.qh, ());
        popup1.grab(&seat, server_button_serial);
        popup1_surface.commit();
        h.roundtrip();

        let popup1_sid = h.server_surface_id(&popup1_surface);
        let reqs = h.drain_requests();
        let popup1_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup1_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("popup1 commit must be forwarded");
        assert_eq!(popup1_state.grab_serial, Some(4242));

        // 2. Nested grabbing popup reusing the SAME serial still resolves to
        //    4242 (non-destructive `SerialMap::get`).
        let popup2_surface = compositor.create_surface(&h.qh, ());
        let popup2_xdg = wm_base.get_xdg_surface(&popup2_surface, &h.qh, ());
        let pos2 = make_positioner(&wm_base, &h.qh);
        let popup2 = popup2_xdg.get_popup(Some(&popup1_xdg), &pos2, &h.qh, ());
        popup2.grab(&seat, server_button_serial);
        popup2_surface.commit();
        h.roundtrip();

        let popup2_sid = h.server_surface_id(&popup2_surface);
        let reqs = h.drain_requests();
        let popup2_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup2_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("popup2 commit must be forwarded");
        assert_eq!(popup2_state.grab_serial, Some(4242));

        // 3. Popup with an unmapped serial skips the grab (`grab_serial: None`).
        let popup3_surface = compositor.create_surface(&h.qh, ());
        let popup3_xdg = wm_base.get_xdg_surface(&popup3_surface, &h.qh, ());
        let pos3 = make_positioner(&wm_base, &h.qh);
        let popup3 = popup3_xdg.get_popup(Some(&parent_xdg), &pos3, &h.qh, ());
        popup3.grab(&seat, 999_999);
        popup3_surface.commit();
        h.roundtrip();

        let popup3_sid = h.server_surface_id(&popup3_surface);
        let reqs = h.drain_requests();
        let popup3_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup3_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("popup3 commit must be forwarded");
        assert_eq!(popup3_state.grab_serial, None);

        // 4. Reconnect clears `serial_map` and never replays stale `grab_serial`
        //    on synced popups OR new popups using a pre-reconnect serial.
        h.connect_wprsc();
        let sync_reqs = h.drain_requests();
        for sid in [popup1_sid, popup2_sid] {
            let synced = sync_reqs
                .iter()
                .find_map(|req| match req {
                    Request::Surface(SurfaceRequest {
                        surface,
                        payload: SurfaceRequestPayload::Commit(state),
                        ..
                    }) if *surface == sid => state.xdg_popup().ok().copied(),
                    _ => None,
                })
                .expect("existing popup must be synced on reconnect");
            assert_eq!(synced.grab_serial, None);
        }

        let popup4_surface = compositor.create_surface(&h.qh, ());
        let popup4_xdg = wm_base.get_xdg_surface(&popup4_surface, &h.qh, ());
        let pos4 = make_positioner(&wm_base, &h.qh);
        let popup4 = popup4_xdg.get_popup(Some(&parent_xdg), &pos4, &h.qh, ());
        popup4.grab(&seat, server_button_serial);
        popup4_surface.commit();
        h.roundtrip();

        let popup4_sid = h.server_surface_id(&popup4_surface);
        let reqs = h.drain_requests();
        let popup4_state = reqs
            .iter()
            .find_map(|req| match req {
                Request::Surface(SurfaceRequest {
                    surface,
                    payload: SurfaceRequestPayload::Commit(state),
                    ..
                }) if *surface == popup4_sid => state.xdg_popup().ok().copied(),
                _ => None,
            })
            .expect("popup4 commit must be forwarded");
        assert_eq!(popup4_state.grab_serial, None);

        // 5. Host grab dismissal delivers `xdg_popup.popup_done`.
        h.client_data.configures.clear();
        let conn_id2 = h.conn_id.unwrap();
        h.state.handle_event(
            conn_id2,
            RecvType::Object(Event::Popup(PopupEvent::Done(popup1_sid))),
        );
        h.roundtrip();
        assert_eq!(
            h.client_data.configures,
            vec![ClientConfigureEvent::PopupDone]
        );
    }
}
