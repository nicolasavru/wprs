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
use smithay::input::pointer::AxisFrame;
use smithay::input::pointer::ButtonEvent;
use smithay::input::pointer::CursorImageStatus as SmithayCursorImageStatus;
use smithay::input::pointer::GestureHoldBeginEvent;
use smithay::input::pointer::GestureHoldEndEvent;
use smithay::input::pointer::GesturePinchBeginEvent;
use smithay::input::pointer::GesturePinchEndEvent;
use smithay::input::pointer::GesturePinchUpdateEvent;
use smithay::input::pointer::GestureSwipeBeginEvent;
use smithay::input::pointer::GestureSwipeEndEvent;
use smithay::input::pointer::GestureSwipeUpdateEvent;
use smithay::input::pointer::GrabStartData;
use smithay::input::pointer::MotionEvent;
use smithay::input::pointer::PointerGrab;
use smithay::input::pointer::PointerInnerHandle;
use smithay::input::pointer::RelativeMotionEvent;
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
use smithay::reexports::wayland_server::protocol::wl_data_device_manager::DndAction;
use smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource;
use smithay::reexports::wayland_server::protocol::wl_output;
use smithay::reexports::wayland_server::protocol::wl_seat;
use smithay::reexports::wayland_server::protocol::wl_surface;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::WEnum;
use smithay::utils::Logical;
use smithay::utils::Point;
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
use smithay::wayland::selection::data_device::with_source_metadata;
use smithay::wayland::selection::data_device::ClientDndGrabHandler;
use smithay::wayland::selection::data_device::DataDeviceHandler;
use smithay::wayland::selection::data_device::DataDeviceState;
use smithay::wayland::selection::data_device::ServerDndGrabHandler;
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
use crate::serialization::wayland::ClientSurface;
use crate::serialization::wayland::DataDestinationRequest;
use crate::serialization::wayland::DataRequest;
use crate::serialization::wayland::DataSource;
use crate::serialization::wayland::DataSourceRequest;
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
        self.serializer
            .writer()
            .send(SendType::Object(Request::Toplevel(ToplevelRequest {
                client: serialization::ClientId::new(&client),
                surface: (&surface.id()).into(),
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
        // Changing maximized/fullscreen is not idempotent, and can be ignored be the compositor,
        // so we must send every request we receive.  However, in case we need to resync the client surface
        // (e.g. if the client reconnects or a toplevel request is sent before the initial commit)
        // we also keep track of the current state.
        compositor::with_states(surface.wl_surface(), |surface_data| {
            let locked_state = surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .expect("LockedSurfaceState inserted in new_toplevel");
            let surface_state = &mut locked_state.0.lock().unwrap();

            if let Some(Role::XdgToplevel(toplevel_state)) = &mut surface_state.role {
                toplevel_state_mutator(toplevel_state);
            }

            self.send_toplevel_request(surface, payload_to_send);
        });
    }
}

impl XdgShellHandler for WprsServerState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    #[instrument(skip(self), level = "debug")]
    fn new_toplevel(&mut self, toplevel: ToplevelSurface) {
        log_and_return!(self.insert_surface(toplevel.wl_surface()));
        compositor::with_states(toplevel.wl_surface(), |surface_data| {
            let surface_state = &mut surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .unwrap()
                .0
                .lock()
                .unwrap();
            surface_state.role = Some(Role::XdgToplevel(Default::default()));
        });

        toplevel.with_pending_state(|state| {
            state.states.set(xdg_toplevel::State::Activated);
        });
        toplevel.send_configure();
    }

    #[instrument(skip(self), level = "debug")]
    fn toplevel_destroyed(&mut self, toplevel: ToplevelSurface) {
        // If client() returns None, the surface was already destroyed and an
        // appropriate message would have been sent to the client, so we don't
        // need to worry about destroying the toplevel,
        if toplevel.wl_surface().client().is_some() {
            self.synced_buffers
                .remove(&WlSurfaceId::new(toplevel.wl_surface()));
            compositor::with_states(toplevel.wl_surface(), |surface_data| {
                surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .expect("LockedSurfaceState inserted in new_toplevel")
                    .0
                    .lock()
                    .unwrap()
                    .role = None;
            });

            self.send_toplevel_request(&toplevel, ToplevelRequestPayload::Destroyed);
        }
    }

    #[instrument(skip(self))]
    fn new_popup(&mut self, popup: PopupSurface, positioner: PositionerState) {
        log_and_return!(self.insert_surface(popup.wl_surface()));

        // Uses with_states internally and with_states is not reentrant.
        let popup_state = log_and_return!(XdgPopupState::new(&popup, &positioner));
        compositor::with_states(popup.wl_surface(), |surface_data| {
            let surface_state = &mut surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .unwrap()
                .0
                .lock()
                .unwrap();
            surface_state.role = Some(Role::XdgPopup(popup_state));
        });

        // TODO: this sometimes sends duplicate configures and causes "The popup
        // positioner is not reactive" errors, but without this popups break
        // completely.
        popup.send_configure().log_and_ignore(loc!());
    }

    #[instrument(skip(self), level = "debug")]
    fn popup_destroyed(&mut self, popup: PopupSurface) {
        // If client() returns None, the surface was already destroyed and an
        // appropriate message would have been sent to the client, so we don't
        // need to worry about destroying the popup,
        if let Some(client) = popup.wl_surface().client() {
            self.synced_buffers
                .remove(&WlSurfaceId::new(popup.wl_surface()));
            // Uses with_states internally and with_states is not reentrant.
            compositor::with_states(popup.wl_surface(), |surface_data| {
                surface_data
                    .data_map
                    .get::<LockedSurfaceState>()
                    .expect("LockedSurfaceState inserted in new_popup")
                    .0
                    .lock()
                    .unwrap()
                    .role = None;
            });

            self.serializer
                .writer()
                .send(SendType::Object(Request::Popup(PopupRequest {
                    client: serialization::ClientId::new(&client),
                    surface: (&popup.wl_surface().id()).into(),
                    payload: PopupRequestPayload::Destroyed,
                })));
        };
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: wl_seat::WlSeat, _serial: Serial) {
        // TODO: this works in sway but breaks popups in mutter
        // "This means it requests to be sent a popup_done event when the pointer leaves the grab area.", do we need to do something here?
        // maybe mutter is denying the grab? maybe because we're passing 0 as the serial?

        // let mut surface_state = self
        //     .surfaces
        //     .get_mut(&serialization::wayland::WlSurfaceId::new(surface.wl_surface()))
        //     .unwrap();
        // surface_state.xdg_popup().unwrap().grab_requested = true;
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
        popup.send_repositioned(token);

        let surface = popup.wl_surface();
        let xdg_positioner = XdgPositioner::new(&positioner);
        let surface_order = get_surface_z_order(surface);
        compositor::with_states(surface, |surface_data| {
            let locked_state = surface_data
                .data_map
                .get::<LockedSurfaceState>()
                .expect("LockedSurfaceState inserted in new_popup");
            let surface_state = &mut locked_state.0.lock().unwrap();

            if let Some(Role::XdgPopup(popup_state)) = &mut surface_state.role {
                popup_state.positioner = xdg_positioner;
            } else {
                error!("reposition called on surface that wasn't a popup");
                return;
            }
            surface_state.z_ordered_children = surface_order;

            let surface_state_to_send = surface_state.clone_without_buffer();
            let client = surface_state_to_send.client;
            let surface = surface_state_to_send.id;
            self.serializer
                .writer()
                .send(SendType::Object(Request::Surface(SurfaceRequest {
                    client,
                    surface,
                    payload: SurfaceRequestPayload::Commit(surface_state_to_send),
                })));
        });
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
        let data_source = match ty {
            SelectionTarget::Clipboard => {
                self.selection_pipe = Some(fd);

                DataSource::Selection
            },
            SelectionTarget::Primary => {
                self.primary_selection_pipe = Some(fd);

                DataSource::Primary
            },
        };

        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer(
                    data_source,
                    mime_type,
                )),
            )));
    }
}

impl DataDeviceHandler for WprsServerState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

impl PrimarySelectionHandler for WprsServerState {
    fn primary_selection_state(&self) -> &PrimarySelectionState {
        &self.primary_selection_state
    }
}

impl ClientDndGrabHandler for WprsServerState {
    #[instrument(skip(self, _seat), level = "debug")]
    fn started(
        &mut self,
        source: Option<WlDataSource>,
        icon: Option<WlSurface>,
        _seat: Seat<Self>,
    ) {
        let icon_surface = icon.as_ref().and_then(|surface| {
            let client_surface = ClientSurface::new(surface).log(loc!()).ok()?;
            self.insert_surface(surface).log(loc!()).ok()?;
            self.sync_surface_if_unsynced(surface);
            Some(client_surface)
        });
        self.dnd_source = source;
        if let Some(source) = &self.dnd_source {
            with_source_metadata(source, |source_metadata| {
                debug!("START DRAG: {source:?}, {source_metadata:?}");
                self.serializer
                    .writer()
                    .send(SendType::Object(Request::Data(DataRequest::SourceRequest(
                        DataSourceRequest::StartDrag(source_metadata.clone().into(), icon_surface),
                    ))));
            })
            .log_and_ignore(loc!());
        }
    }

    #[instrument(skip(self, _target, _validated, _seat), level = "debug")]
    fn dropped(&mut self, _target: Option<WlSurface>, _validated: bool, _seat: Seat<Self>) {}
}

impl ServerDndGrabHandler for WprsServerState {
    #[instrument(skip(self, _seat), level = "debug")]
    fn accept(&mut self, mime_type: Option<String>, _seat: Seat<Self>) {
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::DnDAcceptMimeType(
                    mime_type,
                )),
            )));
    }

    #[instrument(skip(self, _seat), level = "debug")]
    fn action(&mut self, action: DndAction, _seat: Seat<Self>) {
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::DnDSetDestinationActions(
                    action.into(),
                )),
            )));
    }

    #[instrument(skip(self, _seat), level = "debug")]
    fn dropped(&mut self, _seat: Seat<Self>) {}

    #[instrument(skip(self, _seat), level = "debug")]
    fn cancelled(&mut self, _seat: Seat<Self>) {}

    #[instrument(skip(self, _seat), level = "debug")]
    fn send(&mut self, mime_type: String, fd: OwnedFd, _seat: Seat<Self>) {
        self.dnd_pipe = Some(fd);
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::RequestDataTransfer(
                    DataSource::DnD,
                    mime_type,
                )),
            )));
    }

    #[instrument(skip(self, _seat), level = "debug")]
    fn finished(&mut self, _seat: Seat<Self>) {
        self.serializer
            .writer()
            .send(SendType::Object(Request::Data(
                DataRequest::DestinationRequest(DataDestinationRequest::DnDFinish),
            )));
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
    let surface_state = &mut surface_data
        .data_map
        .get::<LockedSurfaceState>()
        .location(loc!())?
        .0
        .lock()
        .unwrap();
    let connected = state.serializer.other_end_connected();
    let prev_without_buffer = connected.then(|| surface_state.clone_without_buffer());

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

    let Some(prev_without_buffer) = prev_without_buffer else {
        return Ok(false);
    };

    let mut surface_state_to_send = surface_state.clone_without_buffer();
    surface_state_to_send.buffer = buffer_to_send;
    if surface_state_to_send.buffer.is_none()
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
    fn set_decoration_mode(&self, surface: &WlSurface, mode: Option<DecorationMode>) {
        compositor::with_states(surface, |surface_data| {
            let Some(locked_state) = surface_data.data_map.get::<LockedSurfaceState>() else {
                return;
            };
            let surface_state = &mut locked_state.0.lock().unwrap();

            // even though the kde server decoration protocol can send decorations for any surface, only send them
            // for XdgTopLevel since that's what wprs currently expects
            if let Some(Role::XdgToplevel(toplevel_state)) = &mut surface_state.role {
                toplevel_state.decoration_mode = mode;
            }
        });
    }
}

impl XdgDecorationHandler for WprsServerState {
    #[instrument(skip(self), level = "debug")]
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {}

    #[instrument(skip(self), level = "debug")]
    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: XdgDecorationMode) {
        let mode: Option<DecorationMode> = mode.try_into().log(loc!()).ok();
        if mode.is_some() {
            self.set_decoration_mode(toplevel.wl_surface(), mode);
        };
    }

    #[instrument(skip(self), level = "debug")]
    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        self.set_decoration_mode(toplevel.wl_surface(), None);
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

pub(crate) struct DndGrab {
    start_data: GrabStartData<WprsServerState>,
}

impl DndGrab {
    pub fn new(
        focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        button: u32,
        location: (f64, f64),
    ) -> Self {
        Self {
            start_data: GrabStartData {
                focus,
                button,
                location: location.into(),
            },
        }
    }
}

impl PointerGrab<WprsServerState> for DndGrab {
    fn start_data(&self) -> &GrabStartData<WprsServerState> {
        &self.start_data
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn motion(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &MotionEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn relative_motion(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        focus: Option<(
            <WprsServerState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &RelativeMotionEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn button(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &ButtonEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn axis(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        details: AxisFrame,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn frame(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_swipe_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeBeginEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_swipe_update(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeUpdateEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_swipe_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureSwipeEndEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_pinch_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchBeginEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_pinch_update(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchUpdateEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_pinch_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GesturePinchEndEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_hold_begin(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureHoldBeginEvent,
    ) {
    }

    #[instrument(skip(self, _data, _handle), level = "debug")]
    fn gesture_hold_end(
        &mut self,
        _data: &mut WprsServerState,
        _handle: &mut PointerInnerHandle<'_, WprsServerState>,
        event: &GestureHoldEndEvent,
    ) {
    }

    #[instrument(skip(self, _data), level = "debug")]
    fn unset(&mut self, _data: &mut WprsServerState) {}
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
smithay::delegate_data_device!(WprsServerState);
smithay::delegate_output!(WprsServerState);
smithay::delegate_primary_selection!(WprsServerState);
smithay::delegate_viewporter!(WprsServerState);

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
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
    use smithay_client_toolkit::reexports::client::protocol::wl_callback;
    use smithay_client_toolkit::reexports::client::protocol::wl_compositor::WlCompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_registry;
    use smithay_client_toolkit::reexports::client::protocol::wl_subcompositor::WlSubcompositor;
    use smithay_client_toolkit::reexports::client::protocol::wl_subsurface::WlSubsurface;
    use smithay_client_toolkit::reexports::client::protocol::wl_surface::WlSurface as ClientWlSurface;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_surface;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_toplevel;
    use smithay_client_toolkit::reexports::protocols::xdg::shell::client::xdg_wm_base::XdgWmBase;

    use super::*;
    use crate::serialization::ClientId as WireClientId;
    use crate::serialization::Event;
    use crate::serialization::RecvType;
    use crate::serialization::Serializer;

    #[derive(Default)]
    struct WireClientData {
        compositor: Option<WlCompositor>,
        subcompositor: Option<WlSubcompositor>,
        xdg_wm_base: Option<XdgWmBase>,
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
                    "xdg_wm_base" => {
                        state.xdg_wm_base =
                            Some(registry.bind::<XdgWmBase, _, _>(name, version.min(6), qh, ()));
                    },
                    _ => {},
                }
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
            _state: &mut Self,
            xdg_surface: &xdg_surface::XdgSurface,
            event: xdg_surface::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let xdg_surface::Event::Configure { serial } = event {
                xdg_surface.ack_configure(serial);
            }
        }
    }

    impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WireClientData {
        fn event(
            _state: &mut Self,
            _proxy: &xdg_toplevel::XdgToplevel,
            _event: xdg_toplevel::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
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
            let state = WprsServerState::new(
                display.handle(),
                event_loop.handle(),
                serializer,
                false,
                Duration::from_millis(16),
                false,
            );

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
}
